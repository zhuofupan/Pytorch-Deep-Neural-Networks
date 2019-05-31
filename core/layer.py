# -*- coding: utf-8 -*-

import torch
import math
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from core.func import Func, act_dict

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
    
class ConvBlock(torch.nn.Module, Func):
    '''
        inputs: para_row, dropout[ conv, res ], func[ conv, res ]
        outputs: conv_layers, res_layers, act_layer, pool_layer
    '''
    
    def __init__(self, 
                 row, 
                 dropout = None, func = 'r', 
                 use_bias = False, batch_norm = 'B',
                 give_name = False):
        torch.nn.Module.__init__(self)
    
        if type(dropout) == list: 
            self.conv_dropout, self.res_dropout = dropout[0], dropout[1]
        else: self.conv_dropout, self.res_dropout = dropout, None
        
        if type(func) == list: 
            self.conv_func, self.res_func = func[0], func[1]
        else: self.conv_func, self.res_func = func, None
        
        self.use_bias, self.batch_norm = use_bias, batch_norm
        conv_para, times, pool_para, res_para = row[0], row[1], row[2], row[3]
        
        self.layers = []
        # Conv
        if conv_para != '-':
            self.conv_layers = []
            conv_cnt = 1
            for t in range(times):
                last_one = False
                if type(conv_para[0]) == int:
                    
                    if give_name: name = str(conv_cnt)
                    else: name = None
                    if t == 1: conv_para[0] = conv_para[1]
                    if t == times - 1: last_one = True
                    
                    self.construct_conv(conv_para, 'conv', name, last_one)
                    conv_cnt += 1
                else:
                    if t == 1: conv_para[0][0] = conv_para[-1][1] 
                    for para in conv_para:
                        if give_name: name = str(conv_cnt)
                        else: name = None
                        if t == times - 1 and para == conv_para[-1]: last_one = True
                        
                        self.construct_conv(para, 'conv', name, last_one)
                        conv_cnt += 1
                        
            self.layers += self.conv_layers
            if hasattr(self, 'act_layer'):
                self.layers += [self.act_layer]
        
        # Pool
        if pool_para != '-':
            pooling = pool_para[0] + 'Pool2d'
            self.pool_layer = eval('nn.'+pooling+'(*pool_para[1:])')
            self.layers += [self.pool_layer]
            
        # Res
        if res_para != '-':  
            self.res_layers = []
            self.res_adaptive = nn.AdaptiveAvgPool2d((row[5][1], row[5][2]))
        if type(res_para) != str: 
            if type(res_para[0]) == int:
                self.construct_conv(res_para, 'res', None, True)
            else:
                last_one = False
                for para in res_para:
                    if para == conv_para[-1]: last_one = True
                    self.construct_conv(para, 'res', None, last_one)
            self.downsample = nn.Sequential(*self.res_layers)
            
    def construct_conv(self, para, case = 'conv', name = None, last_one = True):
        if case == 'conv':
            dropout, func = self.conv_dropout, self.conv_func
            layers = self.conv_layers
        else:
            dropout, func = self.res_dropout, self.res_func
            layers = self.res_layers
        use_bias, batch_norm = self.use_bias, self.batch_norm
        # preprocess
        conv_para = []
        for x in para:
            if type(x) == str:
                if x in act_dict.keys(): func = x
                if x in ['B', 'N']: batch_norm = x
                elif x[0] == 'B': batch_norm = x
                elif x[0] == 'D': dropout = float(x[1:])
            elif type(x) == bool:
                use_bias = x
            else:
                conv_para.append(x)
        # Dropout
        if dropout is not None:
            Dropout = nn.Dropout2d(p = dropout)
            if name is not None:
                exec('self.drop' + name + ' = Dropout')
            layers.append( Dropout )
        # Conv
        Conv = nn.Conv2d(*conv_para, bias = use_bias)
        if name is not None:
            exec('self.conv' + name + ' = Conv')
        layers.append( Conv )
        # BatchNorm
        if batch_norm is not None and batch_norm != 'N':
            if batch_norm == 'B': affine, track_running_stats = True, True
            else: affine, track_running_stats = bool(int(batch_norm[1])), bool(int(batch_norm[2]))
            BN = nn.BatchNorm2d(conv_para[1],affine = affine, track_running_stats = track_running_stats)
            if name is not None:
                exec('self.bn' + name + ' = BN')
            layers.append( BN )
        # Activation
        if func is not None:
            Act = self.F(func)
            if last_one == False:
                if name is not None:
                    exec('self.act' + name + ' = Act')
                layers.append(Act)
            elif case == 'conv':
                self.act_layer = Act
        
    def forward(self, x):
        res = x
        if hasattr(self,'conv_layers'):
            for conv_layer in self.conv_layers:
                x = conv_layer(x)
        if hasattr(self,'res_layers'):
            if hasattr(self,'downsample'):
                res = self.downsample(res)
            try:
                x += res
            except RuntimeError:
                res = self.res_adaptive(res)
                x += res
                
        if hasattr(self,'act_layer'):
            x = self.act_layer(x)
        if hasattr(self,'pool_layer'):
            x = self.pool_layer(x)
        return x
    