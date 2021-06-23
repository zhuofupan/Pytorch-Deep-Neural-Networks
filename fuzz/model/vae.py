# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from .dae import get_decoder_setting
from ..core.module import Module
from ..core.impu_module import Impu_Module

class VAE(Module, Impu_Module):
    def __init__(self, **kwargs):
        default = {'decoder_struct': None, # 解码部分的结构，默认为编码部分反向
                   'decoder_func':None,
                   'dropout': 0.0,
                   'exec_dropout': ['h', None],
                   'L': 'BCE',
                   'alf': 1e3,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['output_func'] = None
        
        self._name = 'VAE'
        super().__init__(**kwargs)
        if self.task == 'impu':
            Impu_Module.__init__(self, **kwargs)
        
        # q(z|x)
        self.Q = self.Sequential(struct = self.struct[:-1], 
                                 hidden_func = self.hidden_func,
                                 dropout = self.exec_dropout[0])
        self.z_mu = self.Sequential(struct = self.struct[-2:], hidden_func = 'a')
        self.z_logvar = self.Sequential(struct = self.struct[-2:], hidden_func = 'a')
        
        # p(x|z)
        self.decoder_struct, self.decoder_func = \
            get_decoder_setting(self.struct, self.hidden_func, self.decoder_func)
        self.P = self.Sequential(struct = self.decoder_struct, 
                                 hidden_func = self.decoder_func, 
                                 dropout = self.exec_dropout[1])
        self.opt()

    def _feature(self, dataset = 'test'):
        if dataset == 'test':
            loader = self.test_loader
        elif dataset == 'train':
            loader = self.train_loader
        else:
            loader = dataset
        self.eval()
        self = self.to(self.dvc)
        
        feature = {}
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                data, target = data.to(self.dvc), target.to(self.dvc)
                self._target = target
                output = self.forward(data)
                output, _ = self.get_loss(output, target)
                
                if hasattr(self,'record_feature') == False:
                    return None
                
                if type(self.record_feature) != list:
                    self.record_feature = list(self.record_feature)
                
                for j, _feature in enumerate(self.record_feature):
                    if i == 0:
                        feature[str(j)] = _feature[j]
                    else:
                        feature[str(j)].append(_feature[j])
        
        for key in feature.keys():
            feature[key] = torch.cat(feature[key], 0)

        return feature
    
    def sample_z(self, z_mu, z_logvar):
        eps = Variable(torch.randn(z_mu.size())).to(self.dvc)
        return z_mu + torch.exp(z_logvar / 2) * eps

    def forward(self, x):
        # q(z|x)
        h = self.Q(x)
        z_mu, z_logvar = self.z_mu(h), self.z_logvar(h)
        z = self.sample_z(z_mu, z_logvar)
        # p(x|z)
        recon = self.P(z)
        # Loss
        recon = torch.clamp(recon, 0, 1)
        # print(recon.min(), recon.max(), x.min(), x.max())
        recon_loss = self.L(recon, x)
        # recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kl_loss = torch.mean(torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1) /2 )
        # print('\n', recon_loss.data, kl_loss.data)
        self.loss = recon_loss + kl_loss * self.alf
        return recon