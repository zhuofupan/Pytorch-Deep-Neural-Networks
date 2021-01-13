# -*- coding: utf-8 -*-
import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from .func import Func, act_dict, get_func

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
    def __init__(self, w, b = None, use_bias = True):
        super(Linear2, self).__init__(w.size(1), w.size(0), bias = use_bias)
        self.name = 'Linear2'
        self.weight = w
        if b is not None and use_bias:
            self.bias = b

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
            
        if hasattr(self,'res_layers'):
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

# https://github.com/jadore801120/attention-is-all-you-need-pytorch
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, **kwargs):
        torch.nn.Module.__init__(self)
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
            
        d_hid = self.d_inner
        if type(d_hid) == str: d_hid = int(eval('d_model' + d_hid))
        self.w_1 = nn.Linear(d_model, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_model) # position-wise
        if self.dropout > 0:
            self.Dropout = nn.Dropout(self.dropout)
        if self.layer_norm:
            self.Layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        if self.dropout > 0:
            x = self.Dropout(x)
        x += residual
        if self.layer_norm:
            x = self.Layer_norm(x)
        return x

class MultiHeadAttention(torch.nn.Module):
    '''
           q_in(b, l, d_model), k_in & v_in(b, l, k_in) 
        -> q_out & k_out(b, n_head, l, k_out), v_out(b, n_head, l, v_out)
        -> alf(b, n_head, l, l), v_out(b, n_head, l, v_out)
        -> z(b, l, n_head * v_out)
        -> z(b, l, v_out)
    '''
    def __init__(self, d_model, v_out, **kwargs):
        torch.nn.Module.__init__(self)
        for key in kwargs.keys():
            setattr(self, key, kwargs[key]) 
        
        if self.k_in == 0: self.k_in = d_model
        if type(self.k_out) == str: self.k_out = int(eval('d_model' + self.k_out))
        if v_out == 0: v_out = self.k_out
        self.v_out = v_out
        n_head, k_out = self.n_head, self.k_out
        self.Q = nn.Linear(d_model, n_head * k_out, False)
        self.K = nn.Linear(self.k_in, n_head * k_out, False)
        self.V = nn.Linear(self.k_in, n_head * self.v_out, False)
        if self.dropout > 0:
            self.Drop_alf = nn.Dropout(self.dropout)
            self.Drop_z = nn.Dropout(self.dropout)
        # back to 'd_model'
        if self.fc:
            self.FC = nn.Linear(n_head * self.v_out, self.v_out, False)
        if self.layer_norm:
            self.Layer_norm = nn.LayerNorm(self.v_out, eps=1e-6)
    
    # self_attention: (I, I, I) -> O
    # encoder_attention: (O, E, E) -> I
    # attention_map: Q-K
    def forward(self, q, k, v):
        # q -> (batch_size, seq_len, d_model)
        # k, v -> (batch_size, seq_len, k_in)
        residual = q
        b, l = q.size(0), q.size(1)
        q = self.Q(q).view(b, l, self.n_head, self.k_out)
        k = self.K(k).view(b, l, self.n_head, self.k_out)
        v = self.V(v).view(b, l, self.n_head, self.v_out)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        # q, k -> (batch_size, n_head, seq_len, k_out)
        # v -> (batch_size, n_head, seq_len, v_out)
        alf = torch.matmul(q / self.k_out ** 0.5, k.transpose(2, 3))
        alf = torch.softmax(alf, dim = -1)
        # alf -> (batch_size, n_head, seq_len, seq_len)
        if self.dropout > 0:
            alf = self.Drop_alf(alf)
        z = torch.matmul(alf, v)
        z = z.transpose(1, 2).contiguous().view(b, l, -1)
        # z -> (batch_size, seq_len, n_head * v_out)
        if self.fc:
            # z -> (batch_size, seq_len, v_out)
            z = self.FC(z)
        if self.dropout > 0:
            z = self.Drop_z(z)
        if z.size() == residual.size():
            z += residual
        if self.layer_norm:
            z = self.Layer_norm(z)
        return z


class LSTMCell(torch.nn.Module):
    def __init__(self, in_dim, out_dim, stride = None):
        torch.nn.Module.__init__(self)
        self.hf = nn.Linear(out_dim, out_dim)
        self.hi = nn.Linear(out_dim, out_dim)
        self.hg = nn.Linear(out_dim, out_dim)
        self.ho = nn.Linear(out_dim, out_dim)
        self.xf = nn.Linear(in_dim, out_dim)
        self.xi = nn.Linear(in_dim, out_dim)
        self.xg = nn.Linear(in_dim, out_dim)
        self.xo = nn.Linear(in_dim, out_dim)
        self.H = nn.Sequential(*[self.hf, self.hi, self.hg, self.ho])
        self.X = nn.Sequential(*[self.xf, self.xi, self.xg, self.xo])
        self.stride = stride
        self.out_dim = out_dim
        
    # x.size() = batch, seq_len, in_dim
    # h_in.size() = batch, out_dim
    def forward(self, x, _in):
        batch, seq_len, _ = x.size()
        h_in, c_in = _in
        
        if hasattr(self, 'h') == False:
            self.h = torch.zeros((batch, seq_len, self.out_dim)).to(get_dvc(x))
            
        # 计算开始前将 计算起点的 计算图指向归 None
        h_in._grad_fn, c_in._grad_fn, self.h._grad_fn = None, None, None
        
        for t in range(seq_len):
            xt = x[:,t,:]
            ft = torch.sigmoid(self.hf(h_in) + self.xf(xt))
            it = torch.sigmoid(self.hi(h_in) + self.xi(xt))
            gt = torch.tanh(self.hg(h_in) + self.xg(xt))
            ot = torch.sigmoid(self.ho(h_in) + self.xo(xt))
            
            # 根据划分数据集时的 stride 得到下一个样本要用到的 h_in, c_in
            if self.stride is not None and t == self.stride:
                h_out, c_out = h_in , c_in
                
            c_in = c_in * ft + it * gt
            h_in = ot * torch.tanh(c_in)
            self.h[:,t,:] = h_in
            
        if self.stride is None or self.stride >= seq_len:
            h_out, c_out = h_in , c_in
        return self.h, (h_out, c_out)