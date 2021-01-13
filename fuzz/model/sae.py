# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from ..core.module import Module
from ..core.pre_module import Pre_Module
from ..core.layer import Linear2, make_noise


class AE(Module):
    def __init__(self, w, b, func, cnt, father, drop_rate, **kwargs):
        default = {'ae_type': 'AE',
                   'n_category': 10,    # 用于监督学习
                   'noise_prob': 0.382, # make_noise 概率
                   'sparse': 0.05,      # 稀疏编码中分布系数 
                   'share_w':False,     # 解码器是否使用编码器的权值
                   'alf': 0.5}          # 损失系数
        
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
        kwargs['task'] = 'usp'
        if 'pre_L' in kwargs.keys(): kwargs['L'] = kwargs['pre_L']
        if 'pre_lr' not in kwargs.keys(): kwargs['pre_lr'] = kwargs['lr']
        if 'name' not in kwargs.keys(): kwargs['name'] = kwargs['ae_type'] + '-{}'.format(cnt+1)
        
        super().__init__(**kwargs)
        self.w, self.b = w, b
        if self.ae_type != 'DAE' and drop_rate > 0:
            self.encoder = nn.Sequential(nn.Dropout(p = drop_rate),
                                         Linear2(w, b, use_bias = self.use_bias),
                                         self.F(func[0]))
        else:
            self.encoder = nn.Sequential(Linear2(w, b, use_bias = self.use_bias),
                                         self.F(func[0]))
        if self.share_w:
            self.decoder = nn.Sequential(Linear2(w.t(), use_bias = True),
                                         self.F(func[1]))
        else:
            self.decoder = nn.Sequential(nn.Linear(w.size(0),w.size(1), bias = True),
                                         self.F(func[1]))
        
        improve = False
        if self.ae_type == 'YSupAE':
            if cnt == len(father.pre_modules) - 1 and improve:
                # improved by me (直接用AE的输出层作为微调的输出层)
                liner = father._output[0]
            else:
                # in the original paper, the fine-turning parameters of output layer are 
                # different from the pre-trained ones  
                liner = nn.Linear(w.size(0), self.n_category)
            if drop_rate > 0:
                self.output_layer = nn.Sequential(nn.Dropout(p = drop_rate),
                                                  liner)
            else:
                self.output_layer = liner
            if isinstance(self.L, nn.CrossEntropyLoss) == False:
                self._output_func = self.F('o')
            
        self.opt()
        
    def _feature(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        origin = x.clone()
        if self.ae_type == 'DAE':
            x, loc = make_noise(x, self.noise_prob)
        feature = self.encoder(x)
        recon = self.decoder(feature)
        self.loss = nn.functional.mse_loss(recon, origin)
        
        _loss = None
        if self.ae_type == 'SAE':
            avg = torch.mean(feature)
            epd = torch.ones_like(avg) * self.sparse
            _loss = torch.sum(epd * torch.log(epd / avg) + (1 - epd) * torch.log((1 - epd)/(1 - avg)))
            
        elif self.ae_type == 'CG-AE':
            try:
                from private.sup_loss import get_h_y
                _h = get_h_y(feature, self._target)
                _loss = nn.functional.mse_loss(_h, self._target)
                # _loss = self.get_suploss(_h)
            except ImportError:
                pass
            
        elif self.ae_type == 'YSupAE':
            y_logits = self.output_layer(feature)
            if isinstance(self.L, nn.CrossEntropyLoss):
                _loss = self.L(y_logits, torch.argmax(self._target, 1).long())
            else:
                y = self._output_func(y_logits)
                _loss = nn.functional.mse_loss(y, self._target) 
            
        if _loss is not None:
            self.loss = (1- self.alf) * self.loss + self.alf * _loss
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
        ae_func 存在时用 ae_func 否则：
        encoder_func 同 hidden_func
        decoder_func 可自定义，默认为 'a'
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
        drop_rate = 0
        if hasattr(self,'pre_dropout') and self.pre_dropout == True:
            drop_rate = self.D(self.dropout, cnt)
            if cnt == 0 and self.__drop__[0] == False: drop_rate = 0
        ae = AE(w, b, ae_func, cnt, self, drop_rate, **self.kwargs)
        return ae
