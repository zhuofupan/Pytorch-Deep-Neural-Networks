# -*- coding: utf-8 -*-
import torch
import sys
import numpy as np

from ..core.module import Module
from ..core.impu_module import Impu_Module
from ..core.func import find_act
from ..core.layer import make_noise

def get_decoder_setting(struct, func, decoder_func = None, expd = False):
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
    if expd:
        i, lengh = 0, len(_func)
        while len(_func) < len(_struct) - 1:
            _func.append(_func[np.mod(i, lengh)])
            i += 1  
    _func.reverse()
    if decoder_func is not None: _func = decoder_func
    return _struct, _func

class Deep_AE(Module, Impu_Module):
    def __init__(self, **kwargs):
        default = {'ae_type': 'AE',
                   'dropout': 0.0,
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
        
        # Decoder
        decoder_struct, decoder_func = get_decoder_setting(self.struct, self.hidden_func)
        self.__drop__.reverse()
        if self.share_w:
            paras = []
            for layer in self.encoder:
                if isinstance(layer, torch.nn.Linear):
                    paras.append((layer.weight, layer.bias))
            self.decoder = self.Sequential(struct = decoder_struct,
                                           hidden_func = decoder_func,
                                           output_func = None, paras = paras)
        else:
            self.decoder = self.Sequential(struct = decoder_struct,
                                           hidden_func = decoder_func,
                                           output_func = None)
        
        self.opt()

    def forward(self, x):
        origin = x.clone()
        if hasattr(self, 'Add'):
            x = x + self.Add(x * (1 - self._nan)) * self._nan
        
        if self.ae_type == 'DAE':
            x, loc = make_noise(x, self.prob)
            self.noise_x, self.noise_loc = x, loc
        
        first = True
        for module in self.encoder:
            x = module(x)
            if find_act(module) is not None:
                if first and hasattr(self, 'Combine'):
                    first = False
                    x = x + self.Combine(self._nan)
                    
        feature = x
        recon = self.decoder(feature)
            
        if self.task == 'impu':
            self.loss = self._get_impu_loss(recon, origin)
        else:
            self.loss = self.L(recon, origin)
        return recon  