# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
sys.path.append('..')

from core.module import Module
from core.layer import make_noise, Linear2

import torch
import torch.nn as nn
from torchvision.utils import save_image

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
        
        super().__init__(**kwargs)
        self.name = 'Deep_'+ self.ae_type
        
        if self.struct[0] != self.struct[-1]:
            extend = self.struct.copy()
            extend.pop(); extend.reverse()
            self.struct += extend
        elif self.share_w:
            # 检查是否对称
            for i in range(int(len(self.struct)/2)):
                if self.struct[i] != self.struct[-(i+1)]:
                    self.share_w = False
                    break
        loc = np.argmin(np.array(self.struct))
        
        # Encoder
        struct = self.struct[:loc+1]
        self.encoder = nn.Sequential()
        weight_list = []
        
        for i in range(len(struct)-1):
            if self.dropout > 0:
                self.encoder.add_module('Dropout'+str(i),nn.Dropout(p = self.dropout))
            
            l_en = nn.Linear(struct[i], struct[i+1])
            weight_list.append(l_en.weight)
            
            self.encoder.add_module('Add_In'+str(i),l_en)
            self.encoder.add_module('Activation'+str(i),self.F('h',i))
        
        # Decoder
        struct = self.struct[loc:]
        weight_list.reverse()
        self.decoder = nn.Sequential()

        for i in range(len(struct)-1):
            if self.dropout > 0:
                self.decoder.add_module('Dropout'+str(i),nn.Dropout(p = self.dropout))
            
            if self.share_w:
                self.decoder.add_module('Add_In'+str(i),Linear2(weight_list[i].t()))
            else:
                self.decoder.add_module('Add_In'+str(i),nn.Linear(struct[i], struct[i+1]))
            
            if i < len(struct)-1:
                self.decoder.add_module('Activation'+str(i),self.F('h',int(i+loc-1)))
            else:
                self.decoder.add_module('Activation'+str(i),self.F('o'))

        self.opt()
    
    def forward(self, x):
        origin = x
        if self.name == 'DAE':
            x, loc = make_noise(x, self.prob)
            self.noise_x, self.noise_loc = x, loc
        
        feature = self.encoder(x)
        out = self.decoder(feature)
        
        self.loss = self.L(origin, out)
        return out
        
    def save(self, data, output):
        if not os.path.exists('../results'): os.makedirs('../results')
        n = min(data.size(0), 8)
        res = output-data
        save_list = [data.view(data.size(0), 1, 28, 28)[:n],
                     output.view(data.size(0), 1, 28, 28)[:n],
                     res.view(data.size(0), 1, 28, 28)[:n]]
        if self.name == 'DAE':
            save_list.insert(1,self.noise_x.view(data.size(0), 1, 28, 28)[:n])
        
        comparison = torch.cat(save_list)
        
        save_image(comparison.cpu(),
                   '../results/reconstruction_' + str(epoch) + '.png', nrow=n)


if __name__ == '__main__':
    
    parameter = {'struct': [784,400,100],
                 'hidden_func': ['Gaussian', 'Affine'],
                 'output_func': 'Affine',
                 'dropout': 0.0,
                 'task': 'prd',
                 'unsupervised': True,
                 'flatten': True}
    
    model = Deep_AE(**parameter)
    
    model.load_mnist('../data', 128)
    
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.batch_test()