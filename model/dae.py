# -*- coding: utf-8 -*-
import torch
import sys
import numpy as np
sys.path.append('..')

from core.module import Module
from core.layer import make_noise


class Deep_AE(Module):  
    def __init__(self, **kwargs):
        
        default = {'ae_type': 'AE',
                   'dropout': 0.0,
                   'prob': 0.8,
                   'share_w':False,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['dvc'] = torch.device('cpu')
        
        self._name = 'Deep_'+ kwargs['ae_type'].upper()
        super().__init__(**kwargs)
        
        struct = self.struct.copy()
        if struct[0] != struct[-1]:
            # 扩展结构
            extend = struct.copy()
            extend.pop(); extend.reverse()
            struct += extend
        elif self.share_w:
            # 检查是否对称
            for i in range(int(len(struct)/2)):
                if struct[i] != struct[-(i+1)]:
                    self.share_w = False
                    break
        # 特征层位置
        loc = np.argmin(np.array(struct))
        
        # Encoder
        self.struct = struct[:loc+1]
        self.encoder = self.Sequential()
        weights,_ = self._get_para()
        self.struct = struct[loc:]
        weights.reverse()
        
        if self.share_w:
            self.decoder = self.Sequential(weights = weights)
        else:
            self.decoder = self.Sequential()
  
        self.opt()
    
    def forward(self, x):
        origin = x
        if self.name == 'DAE':
            x, loc = make_noise(x, self.prob)
            self.noise_x, self.noise_loc = x, loc
        
        feature = self.encoder(x)
        recon = self.decoder(feature)
        
        self.loss = self.L(origin, recon)
        return recon  
