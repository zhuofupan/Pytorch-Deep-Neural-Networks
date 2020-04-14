# -*- coding: utf-8 -*-
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import sys
sys.path.append('..')

from core.module import Module

class MMDGM_VAE(Module):
    def __init__(self, **kwargs):
        '''
            dis_type = [q(z|x,y), p(z), p(x|z,y)]
            q(z|x,y) <- 'gaussian', 'gaussianmarg'
            p(z)     <- 'gaussian', 'gaussianmarg', 'laplace'
            p(x|z,y) <- 'gaussian', 'laplace', 'bernoulli'
        '''
        default = {'struct2': None,
                   'dropout': 0.0,
                   'n_category': 0,
                   'dis_type': ['gaussianmarg','gaussianmarg','gaussian'],
                   'mean_z_prior': 0,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['dvc'] = torch.device('cpu')
        
        self._name = 'MMDGM_VAE'
        super().__init__(**kwargs)
        
        # Encode: q(z|x,y) -> mean_z, logvar_z
        self.open_dropout = False
        self.encode_x = self.Sequential(struct = self.struct[0,1])
        self.encode_y = self.Sequential(struct = self.struct[self.n_category,1])
        self.hidden_z, self.mean_z = self.Sequential(struct = self.struct[1,-1], out_number = 2)
        self.logvar_z = self.Sequential(struct = self.struct[-2,-1])
        
        if self.struct2 is None:
            self.struct2 = self.struct.copy().reverse()
        
        # Decode: p(x|z,y) -> mean_r, logvar_r
        self.open_dropout = True
        self.decode_z = self.Sequential(struct = self.struct2[0,1])
        self.decode_y = self.Sequential(struct = self.struct2[self.n_category,1])
        self.hidden_r, self.mean_r = self.Sequential(struct = self.struct2[1,-1], out_number = 2)
        self.logvar_r = self.Sequential(struct = self.struct2[-2,-1])
        self.opt()
        
    def forward(self,x):
        # q(z|x,y)
        h1_z = self.encode_x(x) + self.encode_y(self._target)
        h_z = self.hidden_z(h1_z)
        mean_z, logvar_z = self.mean_z(h_z), self.logvar_z(h_z)
        eps_z = Variable(torch.randn(self.batch_size, self.struct[-1]))
        z = mean_z + torch.exp(logvar_z / 2) * eps_z
        
        if self.dis_type[0] == 'gaussian':  
            logqx_z = torch.sum(- torch.log(2*math.pi)/2 - logvar_z/2 - (z - mean_z)**2 / (2 * torch.exp(logvar_z)), 0)
        elif self.dis_type[0] == 'gaussianmarg':
            logqx_z = torch.sum( - 1/2 * (torch.log(2 * math.pi) + 1 + logvar_z), 0)
        
        # p(z)
        if self.dis_type[1] == 'gaussian':
            logpz = torch.sum(- torch.log(2*math.pi)/2 - z**2 / 2, 0)
        elif self.dis_type[1] == 'gaussianmarg':
            logpz = torch.sum(-1/2* (torch.log(2*math.pi) + (mean_z - self.mean_z_prior)**2 + torch.exp(logvar_z)), 0)
        elif self.type_pz == 'laplace':
            logpz = torch.sum( torch.log(1/2) - torch.abs(z), 0)
        else:
            raise Exception("Unknown type of p(z)")
        
        # p(x|z,y)
        h1_r = self.decode_z(z) + self.decode_y(self._target)
        h_r = self.hidden_r(h1_r)
        mean_r, logvar_r = self.mean_r(h_r), self.logvar_r(h_r)
        
        if self.dis_type[2] == 'gaussian':
            logpz_x = - torch.log(2*math.pi)/2 - logvar_r/2 - (x - mean_r)**2 / (2 * torch.exp(logvar_r))
        elif self.dis_type[2] == 'laplace':
            sd = torch.exp(logvar_r/2)
            logpz_x = - torch.abs(x - mean_r) / sd - logvar_r/2 - torch.log(2)
        elif self.dis_type[2] == 'bernoulli':
            sig = 1/(1 + torch.exp(-mean_r))
            logpz_x = nn.functional.binary_cross_entropy(sig, x)
        else:
            raise Exception("Unknown type of p(x|z,y)")
            
        self.loss = torch.sum(logpz_x + logpz - logqx_z)
        return mean_r
    
    def features(self, x, layer_id = -1):
        h = self.encode_x(x) + self.encode_y(self._target)
        features = []
        features.append(h)
        for modules in self.hidden_z:
            h = modules(h)
            features.append(h)
        return torch.cat(features[layer_id:-1], 1)