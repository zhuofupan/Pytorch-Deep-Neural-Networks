# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import sys
sys.path.append('..')
from model.vae import VAE
from core.module import Module

class MMDGM_VAE(VAE):
    def __init__(self, **kwargs):
        default = {'struct2': None, # 解码部分的结构，默认为编码部分反向
                   'hidden_func2':None,
                   'n_category': None,
                   'dropout': 0.0,
                   'exec_dropout': [False, True],
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['dvc'] = torch.device('cpu')
        
        self._name = 'MMDGM_VAE'
        Module.__init__(self, **kwargs)
        
        # q(z|x)
        self.qx = self.Sequential(struct = self.struct[:2], hidden_func = self.hidden_func)
        self.qy = self.Sequential(struct = [self.n_category, self.struct[1]], hidden_func = self.hidden_func)
        self.Q = self.Sequential(struct = self.struct[1:-1], hidden_func = self.hidden_func[1:])
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
        
        self.pz = self.Sequential(struct = self.struct2[:2], hidden_func = self.hidden_func2)
        self.py = self.Sequential(struct = [self.n_category, self.struct2[1]], hidden_func = self.hidden_func2)
        self.P = self.Sequential(struct = self.struct2[1:], hidden_func = self.hidden_func2[1:])
        self.opt()

    def forward(self, x):
        # q(z|x)
        h = self.qx(x) + self.qy(self._target)
        h = self.Q(h)
        z_mu, z_logvar = self.z_mu(h), self.z_logvar(h)
        z = self.sample_z(z_mu, z_logvar)
        # p(x|z)
        h2 = self.pz(z) + self.py(self._target)
        recon = self.P(h2)
        # Loss
        recon_loss = nn.functional.binary_cross_entropy(recon, x, reduction='sum') / x.size(0)
        kl_loss = torch.mean(torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1) /2 )
        #print('\n', recon_loss.data, kl_loss.data)
        self.loss = recon_loss + kl_loss
        self.enc_feature = (z_mu, z_logvar)
        return recon