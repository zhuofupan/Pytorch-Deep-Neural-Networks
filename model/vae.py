# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('..')

from core.module import Module

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

class VAE(Module):
    def __init__(self, **kwargs):
        default = {'struct2': None, # 解码部分的结构，默认为编码部分反向
                   'hidden_func2':None,
                   'dropout': 0.0,
                   'exec_dropout': [True, True],
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['dvc'] = torch.device('cpu')
        
        self._name = 'VAE'
        super().__init__(**kwargs)
        
        # q(z|x)
        self.Q = self.Sequential(struct = self.struct[:-1], hidden_func = self.hidden_func)
        self.z_mu = self.Sequential(struct = self.struct[-2:], hidden_func = 'a')
        self.z_logvar = self.Sequential(struct = self.struct[-2:], hidden_func = 'a')
        
        # p(x|z)
        if self.struct2 is None:
            self.struct2 = self.struct.copy()
            self.struct2.reverse()
        if self.hidden_func2 is None:
            if type(self.hidden_func) == str: self.hidden_func = [self.hidden_func]
            self.hidden_func2 = self.hidden_func.copy()
            self.hidden_func2.reverse()
            
        self.P = self.Sequential(struct = self.struct2, hidden_func = self.hidden_func2)
        self.opt()
    
    def sample_z(self, z_mu, z_logvar):
        eps = Variable(torch.randn(z_mu.size()))
        return z_mu + torch.exp(z_logvar / 2) * eps

    def forward(self, x):
        # q(z|x)
        h = self.Q(x)
        z_mu, z_logvar = self.z_mu(h), self.z_logvar(h)
        z = self.sample_z(z_mu, z_logvar)
        # p(x|z)
        recon = self.P(z)
        # Loss
        recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kl_loss = torch.mean(torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1) /2 )
        #print('\n', recon_loss.data, kl_loss.data)
        self.loss = recon_loss + kl_loss
        return recon