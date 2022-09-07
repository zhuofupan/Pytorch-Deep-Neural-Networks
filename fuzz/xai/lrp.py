# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

class LRP(Attribution):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        kwargs['if_hook'] = [True, False]
        kwargs['if_need_baseline'] = False
        Attribution.__init__(self, **kwargs)
    
    def _modify_linear_grad(self, module, grad_x, grad_z):
        _z = module.temp_z - module.bias.data
        x =  module.temp_x
        
        # weight \in (l, l-1)
        # forward: Z = (1, l-1) @ (l-1, l)
        # backward: CP_mx = (l-1, l) * (l-1, 1) = (l-1,l)
        CP_mx = module.weight.data.t() * x.t() / (_z + 1e-5 * torch.sign(_z))
        CP_mx[CP_mx!=CP_mx] = 0
        
        if self.contributions is None:
            self.contributions = CP_mx
        else:
            self.contributions = CP_mx @ self.contributions

        return grad_x
