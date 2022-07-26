# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 22:57:02 2021

@author: Fuzz4
"""
import torch
import numpy as np

class GaussianFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): 
        ctx.save_for_backward(x)
        return 1-torch.exp(-torch.pow(x,2))
    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_in = grad_out * 2*x*torch.exp(-x*x)
        return grad_in

class Gaussian(torch.nn.Module):
    def forward(self, x):
        return GaussianFunction.apply(x)
        
class Affine(torch.nn.Module):
    def forward(self, x):
        return x * 1.0

class Square(torch.nn.Module):
    def forward(self, x):
        return x ** 2

class Exp_F(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x)
    
class Exp_sq(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x) - x - 1

class Ln_sq(torch.nn.Module):
    def forward(self, x):
        return torch.log(x**2 + 1)

class Bottom_F(torch.nn.Module):
    def forward(self, x):
        return x**2 - torch.log(x**2 + 1)

class Oscillation_F(torch.nn.Module):
    def forward(self, x):
        return x * torch.sin(x)

# -----------------------------------------------------------------------------
class Ex0_sqd(torch.nn.Module):
    def forward(self, x):
        return x**2 - x**3

class Ex0_sqt(torch.nn.Module):
    def forward(self, x):
        return x**2 * torch.atan(x+1)

class Ex1_2e(torch.nn.Module):
    def forward(self, x):
        return (x**2 - x + 1) * torch.exp(x)

class Ex1_gaus(torch.nn.Module):
    def forward(self, x):
        return torch.exp(x/1000) * (2 - torch.exp(-x**2))