# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import sys
sys.path.append('..')
from core.func import Func, act_dict, get_func

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

class Linear2(torch.nn.Linear):
    def __init__(self, weight, bias = None):
        super(Linear2, self).__init__(weight.size(1), weight.size(0))
        self.name = 'Linear2'
        self.weight = weight
        if bias is not None:
            self.bias = bias

class Reshape(torch.nn.Module):
    def __init__(self, size):
        self.name = 'Reshape'
        self.size = list(size)
        super().__init__()
        
    def forward(self, x):
        x = x.contiguous().view((-1, *self.size))
        return x
    
class Concat(torch.nn.Module):
    def __init__(self, module_list, dim, out_size = None):
        self.name = 'Concat'
        self.dim = dim
        self.module_list = module_list
        self.out_size = out_size
        super().__init__()
        
    def forward(self, x):
        out = []
        for i in range(len(self.module_list)):
            if type(self.module_list[i]) == str:
                out.append(x)
            else:
                out.append(self.module_list[i](x))
        x = torch.cat(out, self.dim)
        return x

class Square(torch.nn.Module):
    def __init__(self, size = None, func = 'a'):
        super().__init__()
        self.name = 'Square'
        if size is not None:
            self.weight = Parameter(torch.Tensor(*list(size)))
            init.uniform_(self.weight, 0, 1)
        self.func = get_func(func)
        
    def forward(self, x):
        if hasattr(self, 'weight'):
            x = torch.matmul( torch.matmul(x, self.weight), x.transpose(-1, -2) )
        else:
            x = torch.matmul( x, x.transpose(-1, -2) )
        x = self.func(x)
        return x

class ShuffleX(torch.nn.Module):
    def __init__(self, dim = 1, groups = 1):
        self.name = 'ShuffleX'
        super().__init__()
        self.dim = int(dim)
        self.groups = int(groups)
        
    def forward(self, x):
        N,C,H,W = x.size()
        D = x.size(self.dim)
        if self.groups > 1 and self.groups < D:
            groups = int(D/self.groups)
            size = list(x.size())
            size.pop(self.dim)
            size.insert(self.dim, int(D/groups))
            size.insert(self.dim, groups)
            loc = [0,1,2,3,4]
            loc[self.dim] += 1
            loc[self.dim+1] -= 1
            x = x.contiguous().view(*size).permute(*loc)
            x = x.contiguous().view(N,C,H,W)
        x = x.contiguous()
        return x

class ConvBlock(torch.nn.Module, Func):
    '''
        inputs: para_row, dropout[ conv, res ], func[ conv, res ]
        outputs: conv_layers, res_layers, act_layer, pool_layer
    '''
    
    def __init__(self, 
                 row, 
                 dropout = None, func = 'r', 
                 use_bias = False, batch_norm = 'N',
                 give_name = False):
        torch.nn.Module.__init__(self)
        self.name = 'ConvBlock'
        
        if type(dropout) == list: 
            self.conv_dropout, self.res_dropout = dropout[0], dropout[1]
        else: self.conv_dropout, self.res_dropout = dropout, None
        
        if type(func) == list: 
            self.conv_func, self.res_func = func[0], func[1]
        else: self.conv_func, self.res_func = func, None
        
        conv_para, times, res_para, pool_para = row[0], row[1], row[2], row[3]
        self.use_bias, self.batch_norm = use_bias, batch_norm
        
        self.layers = []
        # Conv
        self.layer_cnts = 0
        if conv_para != '-':
            self.conv_layers = []
            conv_cnt = 1
            for t in range(times):
                last_one = False
                if type(conv_para[0]) == list:
                    for para in conv_para:
                        if give_name: name = str(conv_cnt)
                        else: name = None
                        if t == 1: conv_para[0][0] = conv_para[-1][1] 
                        if t == times - 1 and para == conv_para[-1]: last_one = True
                        
                        self.construct_conv(para, 'conv', name, last_one)
                        conv_cnt += 1                    
                else:
                    if give_name: name = str(conv_cnt)
                    else: name = None
                    if t == 1: conv_para[0] = conv_para[1]
                    if t == times - 1: last_one = True
                    
                    self.construct_conv(conv_para, 'conv', name, last_one)
                    conv_cnt += 1
                        
            self.layers += self.conv_layers
            if hasattr(self, 'act_layer'):
                self.layers += [self.act_layer]
            
        # Res
        self.layer_cnts = 0
        if res_para != '-':
            self.res_layers = []
        
        if type(res_para) != str:
            if type(res_para[0]) == int:
                self.construct_conv(res_para, 'res', None, True)
            else:
                last_one = False
                for para in res_para:
                    if para == conv_para[-1]: last_one = True
                    self.construct_conv(para, 'res', None, last_one)
            self.downsample = nn.Sequential(*self.res_layers)
#        else:
#            self.sup = True
        
        # Pool
        if pool_para != '-':
            pooling = pool_para[0] + 'Pool2d'
            self.pool_layer = eval('nn.'+pooling+'(*pool_para[1:])')
            self.layers += [self.pool_layer]
            
    def construct_conv(self, para, case = 'conv', name = None, last_layer_in_block = True):
        if case == 'conv':
            dropout, func = self.conv_dropout, self.conv_func
            layers = self.conv_layers
        else:
            dropout, func = self.res_dropout, self.res_func
            layers = self.res_layers
        use_bias, batch_norm = self.use_bias, self.batch_norm
        # preprocess str
        conv_para = []
        shuffle, transpose, sup_factor = False, False, None
        for x in para:
            if type(x) == str:
                if x in act_dict.keys(): func = x
                elif x in ['B', 'N']: batch_norm = x
                elif 'SF' in x: 
                    shuffle = True
                    shuffle_dim = x[2]
                    shuffle_groups = x[3:]
                elif x == 'TS': transpose = True
                elif x[0] == 'B': batch_norm = x
                elif x[0] == 'D': dropout = float(x[1:])
                elif x[0] == 'Y': sup_factor = float(x[1:])
            elif type(x) == bool:
                use_bias = x
            else:
                conv_para.append(x)
        #print(conv_para)
        
        # Shuffle:
        if shuffle:
            Shuffle = ShuffleX(shuffle_dim, shuffle_groups)
            if name is not None:
                exec('self.shuffle' + name + ' = Shuffle')
            layers.append( Shuffle )
        
        # Dropout
        if dropout is not None and dropout > 0:
            Dropout = nn.Dropout2d(p = dropout)
            if name is not None:
                exec('self.drop' + name + ' = Dropout')
            layers.append( Dropout )
        # Conv
        if transpose:
            Conv = nn.ConvTranspose2d(*conv_para, bias = use_bias)
        else:
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
            Act = self.F(func, self.layer_cnts)
            if last_layer_in_block == False:
                if name is not None:
                    exec('self.act' + name + ' = Act')
                layers.append(Act)
            elif case == 'conv':
                self.act_layer = Act
        # Sup Loss
        if sup_factor is not None:
            try:
                from private._conv import SupLayer
                sup_layer = SupLayer(sup_factor)
                if last_layer_in_block == False:
                    layers.append( sup_layer )
                else:
                    self.sup_layer = sup_layer
            except ImportError:
                pass
            
        self.layer_cnts += 1
        
    def forward(self, x):
        _x = x
        if hasattr(self,'conv_layers'):
            for i in range(len( self.conv_layers )):
                layer = self.conv_layers[i]
                layer._target = self._target
                x = layer(x)
                if i == 0 and isinstance(layer, ShuffleX):
                    _x = x
            
        if hasattr(self,'res_layers'):# and hasattr(self,'sup') == False:
            res = _x
            if hasattr(self,'downsample'):
                res = self.downsample(_x)
            try:
                x += res
            except RuntimeError:
                if x.size(1) != _x.size(1):
                    if hasattr(self, 'res_conv_adaptive') == False:
                        self.res_conv_adaptive = nn.Conv2d(_x.size(1), x.size(1), (1,1))
                        self.res_conv_adaptive.to(get_dvc(_x))
                        print("Add res layer: {}".format(self.res_conv_adaptive))
                    res = self.res_conv_adaptive(res)
                    
                if x.size(2) != _x.size(2) or x.size(3) != _x.size(3):
                    if hasattr(self, 'res_pool_adaptive') == False:
                        self.res_pool_adaptive = nn.AdaptiveAvgPool2d((x.size(2), x.size(3)))
                        print("Add res layer: {}".format(self.res_pool_adaptive))
                    res = self.res_pool_adaptive(res)
                x += res
                
        if hasattr(self,'pool_layer'):
            x = self.pool_layer(x)
            
        if hasattr(self,'act_layer'):
            x = self.act_layer(x)
        
        if hasattr(self,'sup_layer'):
            self.sup_layer._target = self._target
            x = self.sup_layer(x)
        
        self.act_val = x

        return x
    