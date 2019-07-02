# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

import sys
sys.path.append('..')

from core.module import Module
from core.pre_module import Pre_Module
from core.layer import make_noise, Linear2


class AE(Module):
    def __init__(self,w,b,cnt,**kwargs):
        default = {'ae_type': 'AE',
                   'act_func': ['Gaussian', 'Affine'],
                   'prob': 0.3,
                   'share_w':False,
                   'factor': 0.5,
                   'dvc': ''}
        
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
        kwargs['task'] = 'usp'
        if 'pre_lr' not in kwargs.keys(): kwargs['pre_lr'] = kwargs['lr']
        
        self.name = self.ae_type + '-{}'.format(cnt+1)
        super().__init__(**kwargs)

        self.encoder = nn.Sequential(Linear2(w, b),
                                     self.F('ae',0))
        if self.share_w:
            self.decoder = nn.Sequential(Linear2(w.t()),
                                         self.F('ae',1))
        else:
            self.decoder = nn.Sequential(nn.Linear(w.size(0),w.size(1)),
                                         self.F('ae',1))
        self.opt()
        
    def _feature(self, x):
        return self.encoder(x)
    
    def forward(self, x, y = None):
        origin = x
        if self.ae_type == 'DAE':
            x, loc = make_noise(x, self.prob)
        feature = self.encoder(x)
        recon = self.decoder(feature)
        
        self.loss = self.L(origin, recon)
        if self.ae_type == 'SAE':
            avrg = torch.mean(feature)
            expd = torch.ones_like(avrg) * self.prob
            KL = torch.sum(expd * torch.log(expd / avrg) + (1 - expd) * torch.log((1 - expd)/(1 - avrg)))
            self.loss = (1- self.factor) * self.loss + self.factor * KL
        if self.ae_type == 'CGAE':
            try:
                from private.cgae import cg_mean
                h_mean = cg_mean(feature, y)
                self.loss = (1- self.factor) * self.loss + self.factor * self.L(h_mean, y)
            except ImportError:
                pass
        return recon
 
class SAE(Module, Pre_Module):  
    def __init__(self, **kwargs):
        self._name = 'SAE'
        #kwargs['dvc'] =  torch.device('cpu')
        Module.__init__(self, **kwargs)
        self._feature, self._output = self.Sequential(out_number = 2)
        self.opt()
        self.Stacked()
        
    def forward(self, x, y = None):
        x = self._feature(x)
        x = self._output(x)
        return x
    
    def add_pre_module(self, w, b, cnt):
        if hasattr(self,'share_a') and self.share_a:
            act = self.F('h',cnt)
            self.kwargs['act_func'] = [act,act]
        ae = AE(w,b,cnt,**self.kwargs)
        return ae

if __name__ == '__main__':
    
    parameter = {'struct': [784,400,100,10],
                 'hidden_func': ['Gaussian', 'Affine'],
                 'output_func': 'Affine',
                 'ae_type': 'AE',
                 'dropout': 0.0,
                 'task': 'cls',
                 'flatten': True}
    
    model = SAE(**parameter)
    
    model.load_mnist('../data', 128)
    
    model.pre_train(3, 128)
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.test(epoch)