# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

def _grad_f(name, z, h, matrix = True):
    z, h = z.view(-1), h.view(-1)
    e = 1e-6
    if name == 'Softmax':
        # hj = zj/∑z
        # ∂hj/∂zi = hj*((i==j) - hi)
        # 对角线恒大于0, 非对角线恒小于0
        D = h.diag()
        h = h.view(1,-1)
        Y = torch.mm(h.t(), h)
        grad_f = (D - Y)
        
        if matrix: return grad_f
        else: return grad_f.diag()
        
    elif name == 'Gaussian':
        grad_f = 2*z*torch.exp(-z*z)#/7
        grad_f = torch.clamp(grad_f.abs(), min =e) * grad_f.sign()
    elif name == 'Affine':
        grad_f = torch.ones_like(z)
    elif name == 'Sigmoid':
        grad_f = torch.clamp( (h*(1-h)), min =e)#/2
    elif name == 'Tanh':
        # tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
        grad_f = torch.clamp( (1-h*h), min =e)#/4
    elif name == 'ReLU':
        grad_f = torch.clamp(z, min = 0).sign()
    
    if matrix: return grad_f.diag()
    else: return grad_f

class Guided_BP(Attribution):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        kwargs['sensitivity'] = True
        Attribution.__init__(self, **kwargs)
    
    def _get_grad_z(self, grad_h):
        (z, h) = self.fp_z_h[-self.bp_act_cnt]
        self.bp_act_cnt += 1
        # grad_f = _grad_f(self.module_name, z, h)
        grad_f = torch.clamp(z, min=0.0).sign().view(-1).diag()
        grad_h = torch.clamp(grad_h, min=0.0)
        grad_z = torch.mm(grad_h, grad_f)
        return grad_z