# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

class LCG_BP(Attribution):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        Attribution.__init__(self, **kwargs)
    
    def _get_grad_z(self, grad_h):
        (z, h) = self.fp_z_h[-self.bp_act_cnt]
        
        if self._module_names[-self.bp_act_cnt] ==  'ReLU':
            sign = h.sign()
        elif self._module_names[-self.bp_act_cnt] in ['Sigmoid','Softmax']:
            sign = 1
        elif self._module_names[-self.bp_act_cnt] in ['Tanh','Gaussian', 'Affine']:
            sign = z.sign()
        
        self.bp_act_cnt += 1
        grad_h = torch.clamp(grad_h * h, min=0.0)
        grad_z = sign * grad_h
        return grad_z