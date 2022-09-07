# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

def nonlinear_act(Z, act_name):
    if act_name == 'Affine': return Z
    elif act_name == 'Square': return Z**2
    elif act_name == 'Tanh': return torch.tanh(Z)
    elif act_name == 'ReLU': return torch.relu(Z)
    elif act_name == 'Sigmoid': return torch.sigmoid(Z)
    elif act_name == 'Gaussian': return 1-torch.exp(-torch.pow(Z,2))

class PICP(Attribution):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        kwargs['if_hook'] = [True, True]
        kwargs['if_need_baseline'] = True
        Attribution.__init__(self, **kwargs)
    
    def _cal_picp_contribution(self, x, x_bl, act_name, W, b = 0):
        X = x.repeat(x.size(1),1)
        for i in range(x.size(1)): X[i,i] = x_bl[0,i]
        CP_mx = nonlinear_act(x @ W + b, act_name) - nonlinear_act(X @ W + b, act_name)
        CP_mx = CP_mx / CP_mx.sum(axis = 0, keepdim = True)
        CP_mx[CP_mx!=CP_mx] = 0
        return CP_mx
    
    def _modify_linear_grad(self, module, grad_x, grad_z):
        x =  module.temp_x
        x_bl = module.bl_x.to(self.model.dvc)
        act_name = module._behind_act_name
        W = module.weight.data.t()
        b = module.bias.data.view(1,-1)
        
        CP_mx = self._cal_picp_contribution(x, x_bl, act_name, W, b)
        
        # print(CP_mx.sum(axis = 0))
        if self.contributions is None:
            self.contributions = CP_mx
        else:
            self.contributions = CP_mx @ self.contributions

        return grad_x