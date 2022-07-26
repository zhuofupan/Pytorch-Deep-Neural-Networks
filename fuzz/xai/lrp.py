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
        Attribution.__init__(self, **kwargs)
    
    def _get_grad_z(self, grad_h):
        (z, h) = self.fp_z_h[-self.bp_act_cnt]
        self.bp_act_cnt += 1
        
        grad_z = h / (z + 1e-8 * torch.sign(z)) * grad_h
        # 防止分母为 0 
        loc = torch.where(grad_z != grad_z)
        grad_z[ loc ] = 0
        return grad_z