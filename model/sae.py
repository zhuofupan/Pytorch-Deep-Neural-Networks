# -*- coding: utf-8 -*-

import sys
sys.path.append('..')

from core.module import Module
from core.pre_module import Pre_Module
from core.layer import make_noise, Linear2

import torch
import torch.nn as nn

class AE(Module):
    def __init__(self,w,b,**kwargs):
        default = {'ae_type': 'AE',
                   'act_func': ['Gaussian', 'Affine'],
                   'prob': 0.3,
                   'lr': 1e-3,
                   'dvc': ''}
        
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
                
        super().__init__(**kwargs)
        self.name = self.ae_type
        self.task = 'usp'
        
        self.encoder = nn.Sequential(Linear2(w, b),
                                     self.F(0,self.act_func))
        self.decoder = nn.Sequential(Linear2(w.t()),
                                     self.F(1,self.act_func))
        self.opt()
        
    def feature(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        origin = x
        if self.name == 'DAE':
            x, loc = make_noise(x, self.prob)
        feature = self.encoder(x)
        out = self.decoder(feature)
        
        self.loss = self.L(origin, out)
        if self.name == 'SAE':
            avrg = torch.mean(feature)
            expd = torch.ones_like(avrg) * self.prob
            KL = torch.sum(expd * torch.log(expd / avrg) + (1 - expd) * torch.log((1 - expd)/(1 - avrg)))
            self.loss += KL
        return out
 
class SAE(Module, Pre_Module):  
    def __init__(self, **kwargs):
        self.name = 'SAE'
        #kwargs['dvc'] =  torch.device('cpu')
        
        self.kwargs = kwargs
        Module.__init__(self, **kwargs)
        self.Sequential()
        self.opt()
        self.Stacked()
        
    def forward(self, x):
        x = self.feature(x)
        x = self.output(x)
        return x
    
    def add_pre_module(self, w, b):
        ae = AE(w,b,**self.kwargs)
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
        model.test()