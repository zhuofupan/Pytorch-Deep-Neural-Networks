# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

class DeepLIFT(Attribution):
    def __init__(self,
                 **kwargs):
        kwargs['need_baseline'] = True
        self.name = self.__class__.__name__
        Attribution.__init__(self, **kwargs)
    
    def _get_grad_z(self, grad_h):
        (z, h) = self.fp_z_h[-self.bp_act_cnt]
        (z_bl, h_bl) = self.fp_z_h_bl[-self.bp_act_cnt]
        self.bp_act_cnt += 1
        delta_z = z - z_bl
        grad_z = (h - h_bl) / (delta_z + 1e-8 * torch.sign(delta_z)) * grad_h
        # 防止分母为 0 
        loc = torch.where(grad_z != grad_z)
        grad_z[ loc ] = 0
        return grad_z