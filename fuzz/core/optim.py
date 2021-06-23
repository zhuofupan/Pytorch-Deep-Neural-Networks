# -*- coding: utf-8 -*-
import torch
import math

def get_index(data, index):
    if index is None: return data
    else: return data[index]

class RMSprop():     
    def __init__(self, lr=1e-3, alpha=0.9, eps=1e-10, momentum=0, centered=True, memory_size = None):
        self.lr, self.alpha, self.eps, self.momentum, self.centered = \
            lr, alpha, eps, momentum, centered
        if memory_size is not None:
            self.init(memory_size)
    
    def init(self, size):
        self.epoch = 0
        # memory_format=torch.preserve_format
        self.S = torch.zeros(size)
        if self.momentum > 0:
            self.M = torch.zeros(size)
        if self.centered:
            self.C = torch.zeros(size)
    
    def step(self, grad, index = None):
        # State initialization
        if hasattr(self, 'S') == False:
            self.init(grad.size())
            
        square_avg = get_index(self.S, index)
        alpha = self.alpha

        self.epoch += 1

        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if self.centered:
            grad_avg = get_index(self.C, index)
            grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(self.eps)
        else:
            avg = square_avg.sqrt().add_(self.eps)

        if self.momentum > 0:
            buf = get_index(self.M, index)
            buf.mul_(self.momentum).addcdiv_(grad, avg)
            return -self.lr * buf
        else:
            return -self.lr * grad / avg
        
class Adam():     
    def __init__(self, lr=1e-3, betas=(0.9, 0.999), eps=1e-10, amsgrad=False, memory_size = None):
        self.lr, self.betas, self.eps, self.amsgrad = \
            lr, betas, eps, amsgrad
        if memory_size is not None:
            self.init(memory_size)
    
    def init(self, size):
        self.epoch = 0
        # memory_format=torch.preserve_format
        self.ea = torch.zeros(size)
        self.eas = torch.zeros(size)
        if self.amsgrad:
            self.meas = torch.zeros(size)
    
    def step(self, grad, index = None):
        # State initialization
        if hasattr(self, 'ea') == False:
            self.init(grad.size())
            
        exp_avg = get_index(self.ea, index)
        exp_avg_sq = get_index(self.eas, index)
        beta1, beta2 = self.betas[0], self.betas[1]
        
        self.epoch += 1
            
        if self.amsgrad:
            max_exp_avg_sq = get_index(self.meas, index)

        bias_correction1 = 1 - beta1 ** self.epoch
        bias_correction2 = 1 - beta2 ** self.epoch

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if self.amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)

        step_size = self.lr / bias_correction1

        return -step_size * exp_avg /  denom 
