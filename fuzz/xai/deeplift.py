# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

class DeepLIFT(Attribution):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        kwargs['if_hook'] = [True, False]
        kwargs['if_need_baseline'] = True
        Attribution.__init__(self, **kwargs)
    
    def _modify_linear_grad(self, module, grad_x, grad_z):
        x =  module.temp_x
        x_bl = module.bl_x
        
        CP_mx = module.weight.data.t() * (x - x_bl).t()
        CP_mx = CP_mx / CP_mx.sum(axis = 0, keepdim = True)
        CP_mx[CP_mx!=CP_mx] = 0
        
        # print(CP_mx.sum(axis = 0))
        if self.contributions is None:
            self.contributions = CP_mx
        else:
            self.contributions = CP_mx @ self.contributions

        return grad_x