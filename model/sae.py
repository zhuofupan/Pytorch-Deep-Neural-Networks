# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

import sys
sys.path.append('..')
from core.module import Module
from core.pre_module import Pre_Module
from core.layer import make_noise, Linear2


class AE(Module):
    def __init__(self, w, b, func, cnt, **kwargs):
        default = {'ae_type': 'AE',
                   'prob': 0.382,      # make_noise 概率
                   'sparse': 0.05,     # 稀疏编码中分布系数 
                   'share_w':False,    # 解码器是否使用编码器的权值
                   'alf': 0.5}         # 损失系数
        
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
        kwargs['task'] = 'usp'
        if 'pre_lr' not in kwargs.keys(): kwargs['pre_lr'] = kwargs['lr']
        if 'name' not in kwargs.keys(): kwargs['name'] = kwargs['ae_type'] + '-{}'.format(cnt+1)
        
        super().__init__(**kwargs)
        self.w, self.b = w, b
        self.encoder = nn.Sequential(Linear2(w, b),
                                     self.F(func[0]))
        if self.share_w:
            self.decoder = nn.Sequential(Linear2(w.t()),
                                         self.F(func[1]))
        else:
            self.decoder = nn.Sequential(nn.Linear(w.size(0),w.size(1)),
                                         self.F(func[1]))
        self.opt()
        
    def _feature(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        origin = x
        if self.ae_type == 'DAE':
            x, loc = make_noise(x, self.prob)
        feature = self.encoder(x)
        recon = self.decoder(feature)
        
        self.loss = self.L(recon, origin)
        if self.ae_type == 'SAE':
            avg = torch.mean(feature)
            epd = torch.ones_like(avg) * self.sparse
            KL = torch.sum(epd * torch.log(epd / avg) + (1 - epd) * torch.log((1 - epd)/(1 - avg)))
            self.loss = (1- self.alf) * self.loss + self.alf * KL
        if self.ae_type == 'CG-AE':
            try:
                from private.sup_loss import get_h_y
                _h, _y = get_h_y(feature, self._target)
                self.loss = (1- self.alf) * self.loss + self.alf * self.L(_h, _y)
            except ImportError:
                pass
        return recon
 
class SAE(Module, Pre_Module):  
    def __init__(self, **kwargs):
        if 'name' in kwargs.keys(): 
            kwargs['_name'] = kwargs['name']
            del kwargs['name']
        if '_name' not in kwargs.keys(): kwargs['_name'] = 'Stacked_'+kwargs['ae_type']
        
        if 'decoder_func' not in kwargs.keys(): kwargs['decoder_func'] = 'a'
            
        Module.__init__(self, **kwargs)
        
        self._feature, self._output = self.Sequential(out_number = 2)
        self.opt()
        self.Stacked()
        
    def forward(self, x):
        x = self._feature(x)
        x = self._output(x)
        return x
    
    '''
        ae_func 存在时用 ae_func
        encoder_func 同 hidden_func
        decoder_func 自定义
    '''
    def get_sub_func(self, cnt):
        if hasattr(self,'ae_func'):
            return self.ae_func
        
        if type(self.hidden_func) != list: 
            self.hidden_func = [self.hidden_func]
        encoder_func = self.hidden_func[np.mod(cnt, len(self.hidden_func))]
        
        if type(self.decoder_func) != list: 
            self.decoder_func = [self.decoder_func]
        decoder_func = self.decoder_func[np.mod(cnt, len(self.decoder_func))]
        return [encoder_func, decoder_func]
    
    def add_pre_module(self, w, b, cnt):
        ae_func = self.get_sub_func(cnt)
        ae = AE(w, b, ae_func, cnt, **self.kwargs)
        return ae
