# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

class LCP(Attribution):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        kwargs['if_hook'] = [True, True]
        kwargs['if_need_baseline'] = False
        Attribution.__init__(self, **kwargs)
    
    def _modify_linear_grad(self, module, grad_x, grad_z):
        x =  module.temp_x
        z =  module.temp_z
        h =  module.temp_h
        act_name = module._behind_act_name
        
        if act_name ==  'ReLU': sign = h.sign()
        elif act_name in ['Sigmoid','Softmax']: sign = 1
        elif act_name in ['Tanh', 'Gaussian', 'Affine', 'Square']: sign = z.sign()
        
        CP_mx = torch.relu( module.weight.data.t() * x.t() * sign)
        CP_mx = CP_mx / CP_mx.sum(axis = 0, keepdim = True)
        CP_mx[CP_mx!=CP_mx] = 0
        
        if self.contributions is None:
            self.contributions = CP_mx
        else:
            self.contributions = CP_mx @ self.contributions

        return grad_x