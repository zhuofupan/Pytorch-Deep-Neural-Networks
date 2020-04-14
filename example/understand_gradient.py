# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

class Understand_Gradient(nn.Module):
    def __init__(self,):
        '''
            x (1,1,5,5) -W0 (1,1,2,2)-> z0 (1,1,4,4) -ReLU-> 
            h0 (1,1,4,4) -W1 (1,1,3,3)-> z1 (1,1,2,2) -ReLU-> 
            h1 (1,4) -W2 (2,4)T-> z2(1,2) -Sigmoid-> y (1,2)
            
            MSE: ∂L/∂y = 2/n(y-l) = [[-0.3, 0]]
            Sgd: ∂y/∂z2 = y(1-y) = [[0.21, 0]], ∂L/∂z2 = [[-0.063, 0]]
            Lin: ∂z2/∂h1 = W2, ∂L/∂h1 = [[-0.063, 0, 0, 0.198]], ∂z2/∂W2T = h1T, ∂L/∂W2T (4,2)
            
            ReL: ∂h1/∂z1 = z1>0?, ∂L/∂z1 = [[[[-0.063, 0],[0, 0.198]]]] (1, 1, 2, 2)
            Cov: ∂L/∂h0 = [[[[W1 * ∂L/∂z1_11, 0],[0, 0]] + [[0, w1 * ∂L/∂z1_12],[0, 0]]  
                        + [[0, 0],[W1 * ∂L/∂z1_21, 0]] + [[0, 0],[0, W1 * ∂L/∂z1_22]]]] (1,1,4,4)
                 ∂L/∂W1 = [[[h0(1~3,1~3) * ∂L/∂z1_11] + [h0(1~3,2~4) * ∂L/∂z1_12]  
                        + ... + [h0(2~4,2~4) * ∂L/∂z1_22]]] (1,1,3,3)
    
            ReL: ∂h0/∂z0 = z0>0?, ∂L/∂z0 (1,1,4,4)
            Cov: ∂L/∂x = [[[[W0 * ∂L/∂z0_11, 0, 0, 0],...] + [[0, w0 * ∂L/∂z0_12, 0, 0],...]  
                       + ... + [...,[0, 0, 0, W0 * ∂L/∂z0_44]]]] (1,1,5,5)
                 ∂L/∂W0 = [[[x(1~2,1~2) * ∂L/∂z0_11] + [x(1~2,2~3) * ∂L/∂z0_12] 
                       + ... + [x(4~5,4~5) * ∂L/∂z0_44]]] (1,1,2,2)
        '''
        nn.Module.__init__(self)
        self.conv = nn.Sequential( nn.Conv2d(1,1,(2,2), bias = False),
                                   nn.ReLU(),
                                   nn.Conv2d(1,1,(3,3), bias = False),
                                   nn.ReLU())
        self.fc = nn.Sequential( nn.Linear(4,2, bias = False),
                                 nn.Sigmoid())
        
        w0 = torch.Tensor([[1, 0],[0, -1]]).view(1,1,2,2)
        w1 = torch.Tensor([[1, 0, -1],[0, -1, 1],[0, 1, -1]]).view(1,1,3,3)
        w2 = torch.Tensor([[1, 0],[0, 1],[0, 1],[-3.1527,-8]]).t()

        self.conv[0].weight.data = w0
        self.conv[2].weight.data = w1
        self.fc[0].weight.data = w2
        
        self.L = torch.nn.MSELoss()
        self.hook_layer()
    
    def run(self):
        #x = torch.Tensor([[1, 2, 3, 4],[2, 0, 4, 3],[3, 4, 2, 1],[4, 2, 6, 1]]).view(1,1,4,4)
        x = torch.Tensor([[5, 6, 7, 5, 1],[7, 4, 4, 4, 1],[4, 5, 4, 0, 1],
                          [5, 1, 1, 2, -1],[1, 1, -1, -5, 1]]).view(1,1,5,5)
        l = torch.Tensor([1, 0]).view(1,2)
        self.zero_grad()
        print('\n>>> forward:')
        y = self.forward(x)
        loss = self.L(y, l)
        print('\n>>> backward:')
        loss.backward()
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(1,4)
        y = self.fc(x)
        return y     
    
    def hook_layer(self):
        def _print(module, _in , _out):
            print('---', module.__class__.__name__,'---')
            print('in:',_in)
            print('out:',_out)

        def _forward(module, ten_in, ten_out):
            _print(module, ten_in, ten_out)
            if hasattr(module, 'weight'):
                print('weight:',module.weight)
    
        def _backward(module, grad_in, grad_out):
            _print(module, grad_out, grad_in)
        
        self.handles = []
        for modules in [self.conv, self.fc]:
            for module in modules:
                print(module)
                self.handles.append(module.register_forward_hook(_forward))
                self.handles.append(module.register_backward_hook(_backward))

Understand_Gradient().run()