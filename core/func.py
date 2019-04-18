# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn

class Gaussian(torch.nn.Module):
    def forward(self, x):
        return 1-torch.exp(-torch.pow(x,2))
class Affine(torch.nn.Module):
    def forward(self, x):
        return x

class Func(object):
    def F(self, name, func = None, **kwargs):
        if func is None:
            func = self.hidden_func
        if type(name) == int:
            if type(func) is list:
                name = func[np.mod(name, len(func))]
            else:
                name = func
                
        if name == 'Gaussian':
            func = Gaussian()
        elif name == 'Affine':
            func = Affine()
        else:
            '''
                ReLU, ReLU6, ELU, PReLU, LeakyReLU, 
                Threshold, Hardtanh, Sigmoid, Tanh, LogSigmoid, 
                Softplus, Softshrink, Softsign, Tanhshrink, Softmin, Softmax, LogSoftmax
            '''
            func = eval('nn.'+name+'(**kwargs)')
        return func
    
    def get_loss(self, output, target):
        if hasattr(self, 'loss'):
            # 在 forword 里自定义了损失值，调用get_loss前调用 forword 以获取损失值
            return self.loss
        else:
            return self.L(output, target)
        
    def get_rmse(self, output, target):
        return torch.sqrt(nn.functional.mse_loss(output, target))
    
    def get_R2(self, output, target):
        total_error = torch.sum(torch.pow(target -  torch.mean(target),2))
        unexplained_error = torch.sum(torch.pow(target - output,2))
        R_squared = 1 - unexplained_error/ total_error
        return R_squared
    
    def get_accuracy(self, output, target):
        if target.size(-1)>1:
            output_arg = torch.argmax(output,1)
            target_arg = torch.argmax(target,1)
        else:
            output_arg = (output + 0.5).int()
            target_arg = target.int()
            
        return torch.mean(output_arg.eq(target_arg).float())