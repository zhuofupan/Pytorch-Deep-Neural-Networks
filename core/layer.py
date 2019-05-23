# -*- coding: utf-8 -*-

import torch
import math
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter

'''
    W: truncated_normal(stddev=np.sqrt(2 / (size(0) + size(1))))
    b: constant(0.0)
'''

def get_dvc(x):
    if x.is_cuda:
        dvc = torch.device('cuda')
    else:
        dvc = torch.device('cpu')
    return dvc

def make_noise(x, prob):
    dvc = get_dvc(x)
    rand_mat = torch.rand(x.size())
    noise_co = (rand_mat < prob).float().to(dvc)  # 噪声系数矩阵
    non_noise_co = (1-noise_co) # 保留系数矩阵
    output = x * non_noise_co
    return output, noise_co
            
class Linear2(torch.nn.Module):
    def __init__(self, weight, bias = None):
        super().__init__()
        self.weight = weight
        if bias is None:
            self.bias = Parameter(torch.Tensor(weight.size(0)))
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        else:
            self.bias = bias
        
    def forward(self, x):
        dvc = get_dvc(x)
        return F.linear(x, self.weight.to(dvc), self.bias.to(dvc))