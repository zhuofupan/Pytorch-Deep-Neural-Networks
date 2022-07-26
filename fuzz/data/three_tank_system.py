# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 18:39:38 2022

@author: Fuzz4
"""
import numpy as np
import torch

A0 = 154     # Cross section area of tanks (cm2)
sn = 0.5     # Cross section area of pipes (cm2)
H_max = 62   # Max. height of tanks (cm)
Q1_max = 120 # Max. flow rate of pump 1 (cm3 /s)
Q2_max = 120 # Max. flow rate of pump 2 (cm3 /s)
a1 = 0.46    # Coeff. of flow for pipe 1
a2 = 0.60    # Coeff. of flow for pipe 2
a3 = 0.45    # Coeff. of flow for pipe 3
g = 9.81*100 # cm/s2
s13 = s23 = s0 = sn

class Three_tank():
    def __init__(self):
        self.B = np.array([[1/A0, 0], [0, 1/A0], [0, 0]], dtype = np.float32)/2
        self.C = np.array([[1, 0, 0], [0, 1, 0]], dtype = np.float32)
        
    def fault_signal(self, f_id, N):
        T = np.repeat( np.arange(0,N).reshape(-1, 1), 2, axis = 1)
        # Additive actuator faults in pumps: U or Q
        if f_id == 1:
            F = 12 * np.sin(T)
        elif f_id == 2:
            F = 0.6 * T
            F[:, 0] = 0
        elif f_id == 3:
            F = 0.6 * T + 12 + 12 * (-1)**(1/np.pi * T)
            F[:, 1] = 0
        # Additive sensor faults in measurements: Y
        if f_id == 4:
            F = 1.2 * T
            F[:, 1] = 0
        elif f_id == 5:
            F = 1.2 * T + 12 + 12 * (-1)**(1/np.pi * T)
            F[:, 0] = 0
        # Multiplicative faults, may cause changes in the system eigen-dynamics
        elif f_id == 6:
            # θ * sqrt(2g h1) → A11, A13, A31, A32, A33
            F = np.array([[0.5, 0, 0.5], [0, 0, 0], [0.5, 0.5, 0.5]], dtype = np.float32)
        elif f_id == 7:
            # θ * a1 s13 sign(h1 - h3) sqrt( 2g|h1 - h3 |) → -A11, -A13, A31, A32, A33
            F = np.array([[0.5, 0, 0.5], [0, 0, 0], [-0.5, -0.5, -0.5]], dtype = np.float32)
        elif f_id == 8:
            # θ * a3 s23 sign(h3 - h2) sqrt( 2g|h3 - h2 |) → A22, A23, -A31, -A32, -A33
            F = np.array([[0, 0, 0], [0, -0.5, -0.5], [0.5, 0.5, 0.5]], dtype = np.float32)
        return F
            
    def linearising(self, h_op):
        '''
        A0 der(h1) = Q1 − Q13, A0 der(h2) = Q2 + Q32 − Q20, A0 der(h3) = Q13 − Q32
        Q13 = a1 s13 sgn(h1 − h3 ) sqrt( 2g|h1 − h3 |)
        Q32 = a3 s23 sgn(h3 − h2 ) sqrt( 2g|h3 − h2 |), Q20 = a2 s0 sqrt( 2g h2)
        '''
        
        h1 = torch.tensor(h_op[0], requires_grad=True)
        h2 = torch.tensor(h_op[1], requires_grad=True)
        h3 = torch.tensor(h_op[2], requires_grad=True)
        
        dAdh = torch.zeros((3,3))
        A1 = -a1 * s13 * torch.sign(h1-h3) * torch.sqrt(2*g* torch.abs(h1-h3)) / A0
        A1.backward()
        dAdh[0,0], dAdh[0,1], dAdh[0,2] = h1.grad.data, torch.zeros(1), h3.grad.data
        h1.grad, h3.grad = torch.zero_(h1.grad), torch.zero_(h3.grad)
        
        A2 = (a3 * s23 * torch.sign(h3-h2) * torch.sqrt(2*g* torch.abs(h3-h2)) - \
              a2 * s0 * torch.sqrt(2*g*h2)) / A0
        A2.backward()
        dAdh[1,0], dAdh[1,1], dAdh[1,2] = torch.zeros(1), h2.grad.data, h3.grad.data
        h2.grad, h3.grad = torch.zero_(h2.grad), torch.zero_(h3.grad)
        
        A3 = (a1 * s13 * torch.sign(h1-h3) * torch.sqrt(2*g* torch.abs(h1-h3)) - \
              a3 * s23 * torch.sign(h3-h2) * torch.sqrt(2*g* torch.abs(h3-h2))) / A0
        A3.backward()
        dAdh[2,0], dAdh[2,1], dAdh[2,2] = h1.grad.data, h2.grad.data, h3.grad.data
        h1.grad, h2.grad, h3.grad = torch.zero_(h1.grad), torch.zero_(h2.grad), torch.zero_(h3.grad)
        
        return dAdh.data.numpy()
    
    def get_steady_operating_point(self, h_op):
        self.h_op = h_op
        self.A = self.linearising(h_op)
        self.Q_op = np.linalg.pinv(self.B) @(-self.A @ h_op)
    
    # 非线性系统
    def nonlinear_forward(self, Q, f_id = 0, h0 = np.ones((3,))*20):
        N = Q.shape[0]
        if f_id > 0: F = self.fault_signal(f_id)
        
        if f_id <=3: Q += F
        Q[:,0] = np.clip(Q[:,0], 0, Q1_max)
        Q[:,1] = np.clip(Q[:,1], 0, Q2_max)
        
        h = h0
        Y = np.zeros((N, 2))
        for t in range(N):
            Q13 = a1 * s13 * np.sign(h[0]-h[2]) * np.sqrt(2*g* np.abs(h[0]-h[2]))
            Q32 = a3 * s23 * np.sign(h[2]-h[1]) * np.sqrt(2*g* np.abs(h[2]-h[1]))
            Q20 = a2 * s0 * np.sqrt(2*g* h[1])
            F6 = 0
            if f_id == 6:
                F6 = 0.5 * np.sqrt(2*g*(h[0]))
            elif f_id == 7:
                Q13 *= (1-0.5)
            elif f_id == 8:
                Q32 *= (1-0.5)
            
            h[0] = (Q[t,0] - Q13 + F6)/A0
            h[1] = (Q[t,1] + Q32 - Q20)/A0
            h[2] = (Q13 - Q32 + F6)/A0
            h = np.clip(h, 0, H_max)
            Y[t] = self.C @ h
        if f_id in [4,5]:
            Y += F
            Y = np.clip(Y, 0, H_max)
        return (Q, Y)
    
    # 线性化的非线性系统，要把 Y 控制到 h_op 附近（稳态工作点）
    def linear_forward(self, h_op, Q, f_id = 0):
        N = Q.shape[0]
        if f_id > 0: F = self.fault_signal(f_id)
        
        if f_id <=3: Q += F
        Q[:,0] = np.clip(Q[:,0], 0, Q1_max)
        Q[:,1] = np.clip(Q[:,1], 0, Q2_max)
        
        # x = h - h_op < H_max - h_op; u = Q - Q_op < Q_max - Q_op
        self.get_steady_operating_point(h_op)
        # 稳态 h_op 开始
        x = np.zeros_like(h_op)
        # 从0开始控到稳态 h_op
        # x = -h_op
        U = Q - self.Q_op
        Y = np.zeros((N, 2))
        for t in range(N):
            A = self.A
            if f_id == 6:
                A += F* np.sqrt(2*g*(x[0]+ h_op[0]))
            elif f_id == 7:
                dh1_3 = x[0]+h_op[0]-x[2]-h_op[2]
                A += F* a1* s13 * np.sign(dh1_3) * np.sqrt(2*g* np.abs(dh1_3))
            elif f_id == 8:
                dh2_3 = x[1]+h_op[1]-x[2]-h_op[2]
                A += F* a3* s23 * np.sign(dh2_3) * np.sqrt(2*g* np.abs(dh2_3))
            x = A @ x + self.B @ U[t]
            x = np.clip(x, -h_op, H_max - h_op)
            Y[t] = self.C @ x + h_op
        if f_id in [4,5]:
            Y += F
            Y = np.clip(Y, 0, H_max)
        return (Q, Y)

if __name__ == '__main__':
    h_op1 = np.array([15, 10, 12.5], dtype = np.float32).reshape((-1,))
    h_op2 = np.array([20, 10, 15], dtype = np.float32).reshape((-1,))
    h_op3 = np.array([25, 10, 17.5], dtype = np.float32).reshape((-1,))
    h_op4 = np.array([50, 40, 45], dtype = np.float32).reshape((-1,))