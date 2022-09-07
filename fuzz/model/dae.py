# -*- coding: utf-8 -*-
import torch
import sys
import numpy as np

from ..core.module import Module
from ..core.impu_module import Impu_Module
from ..core.func import find_act
from ..core.layer import make_noise

def get_decoder_setting(struct,                 # encoder 的 struct
                        func,                   # encoder 的 func
                        decoder_func = None,    # 定义了 decoder_func 时，替代 func
                        expd = True             # 让 func 的个数和 struct 相等
                        ):          
    if type(func) == str: func = [func]
    _struct, _func = struct.copy(), func.copy()
    # struct
    _struct.reverse()

    for s in _struct:
        if type(s) == str:
            if s[0] == '/': s[0] = '*'
            elif s[0] == '*': s[0] = '/'
    # func
    if len(_func) > len(_struct):
        _func = _func[:len(_struct)]
    _func.reverse()
    if decoder_func is not None: _func = decoder_func
    if type(_func) != list: _func = list(_func)
    if expd:
        i, lengh = 0, len(_func)
        while len(_func) < len(_struct) - 1:
            _func.append(_func[np.mod(i, lengh)])
            i += 1  
    return _struct, _func

class Deep_AE(Module, Impu_Module):
    def __init__(self, **kwargs):
        default = {'ae_type': 'AE',
                   'dropout': 0.0,
                   'decoder_func': None,
                   'output_func': None,
                   'prob': 0.0,
                   'share_w':False,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        self._name = 'Deep_'+ kwargs['ae_type'].upper()
        Module.__init__(self, **kwargs)
        if self.task == 'impu':
            Impu_Module.__init__(self, **kwargs)
    
        # Encoder
        self.encoder = self.Sequential(output_func = None)

        # Decoder (struct 和 func 是 encoder 中的颠倒)
        decoder_struct, decoder_func = get_decoder_setting(self.struct, self.hidden_func, self.decoder_func )
        if self.share_w:
            paras = []
            for layer in self.encoder:
                if isinstance(layer, torch.nn.Linear):
                    paras.append((layer.weight, layer.bias))
            self.decoder = self.Sequential(struct = decoder_struct,
                                           hidden_func = decoder_func,
                                           output_func = self.output_func, paras = paras)
        else:
            self.decoder = self.Sequential(struct = decoder_struct,
                                           hidden_func = decoder_func,
                                           output_func = self.output_func)
        self.opt()
    
    def _get_latent(self, x):
        if hasattr(self, 'Add'):
            x = x + self.Add(x * (1 - self._nan)) * self._nan
        
        if self.ae_type == 'DAE':
            x, loc = make_noise(x, self.prob)
            self.noise_x, self.noise_loc = x, loc
        
        first = True
        for module in self.encoder:
            # print(x.device, next(module.parameters()).device)
            x = module(x)
            if find_act(module) is not None:
                if first and hasattr(self, 'Combine'):
                    first = False
                    x = x + self.Combine(self._nan)
        return x
    
    def forward(self, x):
        origin = x.clone()
        feature = self._get_latent(x)
        self._latent_variables = feature
        recon = self.decoder(feature)
            
        if self.task == 'impu':
            self.loss = self._get_impu_loss(recon, origin)
        else:
            self._loss_ = torch.sum((recon - x)**2, -1)
            self.loss = torch.mean(self._loss_)
        return recon  
