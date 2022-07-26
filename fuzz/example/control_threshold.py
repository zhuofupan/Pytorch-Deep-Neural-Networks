# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 13:12:08 2022

@author: Fuzz4
"""
import numpy as np
from matplotlib import pyplot as plt

A = np.array([[1, 3],[0, 1]], dtype=np.float64)
B = np.array([[0.],[1.]], dtype=np.float64)
K = np.array([-2.4735, -5.1037], dtype=np.float64).reshape((1,2))
x0 = np.array([[1.],[2.]], dtype=np.float64)
W = A + B @ K

def x_seq(V, N):
    X = np.zeros((2,N+1), dtype=np.float64)
    x = x0.reshape((2,))
    X[:,0] = x
    for i in range(1, N+1):
        v = V[i-1].reshape((2,))
        # print(A.shape, (B@K).shape, x.shape, v.shape)
        dx = W @ x + v
        x = 0.01 * dx + x
        X[:,i] = x
    return X

def compute_pro(X, k):
    xs = np.array([[0.],[0.]])
    count = 0
    for i in range(1, X.shape[1]):
        x = X[:,i]
        # if k == 0: print(i, np.sum((x-xs)**2))
        L = np.sqrt(np.sum((x-xs)**2))
        R = k*np.sqrt(np.sum(xs**2))
        
        # R = np.exp(k)
        if L - R >= 0:
            xs = x
            count += 1
    pro = count/(X.shape[1]-1)
    return pro

def learn_threshold(X, exp_pro):
    # k_lb, k_ub = np.log(1e-8), np.log(1)
    k_lb, k_ub = 0, 1
    pro_lb = compute_pro(X, k_ub)
    while pro_lb > exp_pro:
        k_ub *= 2
        pro_lb = compute_pro(X, k_ub)
    print('>>> Low bound {} (k = {})\n>>> Upper bound {} (k = {})'.format(pro_lb, k_ub, compute_pro(X, k_lb), k_lb))
    pro = 0
    k_medians, pros = [], []
    while np.abs(pro - exp_pro) > 1e-3:
        k_median = (k_ub + k_lb)/2
        pro = compute_pro(X, k_median)
        k_medians.append(k_median)
        pros.append(pro)
        print('k_median: {}, pro: {:.2f}%'.format(k_median, pro*100))
        if pro > exp_pro:
            k_lb = k_median
        else:
            k_ub = k_median
    
    k_medians = np.array(k_medians)
    pros = np.array(pros)*100
    
    print('\nk_medians = {}'.format(k_medians))
    print('\npros = {}'.format(pros))
    
    fig = plt.figure(figsize=[24,15])  #定义新的三维坐标轴
    ax = fig.add_subplot(111)
    n = np.arange(k_medians.shape[0]) + 1
    labels = ['k', 'pro']
    l1 = ax.plot(n, k_medians, c = 'b', linewidth = 3, label = labels[0])
    # plt.yticks(fontsize=58)
    # plt.yscale('log')
    ax.tick_params('x', labelsize = 58)
    ax.tick_params('y', labelsize = 58)
    ax.set_xlabel('Epoch', fontsize = 68)  # 设置x坐标轴
    ax.set_ylabel('k', fontsize = 68)  # 设置y坐标轴
    ax = ax.twinx()
    ax.tick_params('y', labelsize = 58)
    ax.set_ylabel('pro(%)', fontsize = 68)  # 设置y坐标轴
    l2 = ax.plot(n, pros, c = 'r', linewidth = 3, label = labels[1])
    
    ls = l1+l2
    labs = [l.get_label() for l in ls]
    lgd = ax.legend(ls, labs, loc = 'upper right', fontsize=58)
    lgd.get_frame().set_alpha(0.5)
    plt.show()
    
    return k_median, pro

if __name__ == '__main__':
    N = 2200
    V = np.random.normal(loc=0.0, scale=1.0, size=(N,2)).astype(np.float64)
    # U = np.sin(np.arange(N, dtype=np.float64))
    X = x_seq(V, N)
    # plot x
    fig = plt.figure(figsize=[24,15])  #定义新的三维坐标轴
    ax = fig.add_subplot(111)
    n = np.arange(X.shape[1]) + 1
    for i in range(X.shape[0]):
        ax.plot(n, X[i], linewidth = 3, label = 'x'+str(i+1))
    lgd = ax.legend(loc = 'upper right', fontsize=58)
    lgd.get_frame().set_alpha(0.5)
    ax.tick_params('x', labelsize = 58)
    ax.tick_params('y', labelsize = 58)
    ax.set_xlabel('Time', fontsize = 68)  # 设置x坐标轴
    ax.set_ylabel('Status', fontsize = 68)  # 设置y坐标轴
    plt.show()
    # solve k
    exp_pro = 80/100
    delta = 5/100
    
    k, pro = learn_threshold(X, exp_pro = exp_pro)
    print('\nk = {}, pro = {:.2f}%'.format(k, pro*100))
    
    eig, _ = np.linalg.eig(W)
    eig = np.max(np.real(eig))
    alf = np.abs(exp_pro - 1 + 1/eig **2);
    N_min = ((2*np.sqrt(2*np.log(2/delta)) * alf + np.sqrt(8*np.log(2/delta)*alf**2 - 4 *alf**2*(2*np.log(2))))/(2*alf**2))**2
    error = np.sqrt(np.log(2/delta)/(2*N_min))
    print('\neig = {:.4f}, alf = {:.4f}, N_min = {:.4f}, error = {:.4f}'.format(eig, alf, N_min, error) )
    