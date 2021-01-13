# -*- coding: utf-8 -*-
# https://arxiv.org/abs/1406.5298

import torch
import torch.nn as nn
from fuzz import VAE, Module

class Semi_VAE(VAE):
    def __init__(self, **kwargs):
        default = {'decoder_struct': None, # 解码部分的结构，默认为编码部分反向
                   'decoder_func':None,
                   'n_category': None,
                   'dropout': 0.0,
                   'exec_dropout': ['h', None],
                   'alf_r': 1e-2,
                   'alf_sup':0.1,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        self._name = 'Semi_VAE'
        Module.__init__(self, **kwargs)
        
        # q(z|x)
        self.qx_in = self.Sequential(struct = self.struct[:2], hidden_func = self.hidden_func)
        self.qy_in = self.Sequential(struct = [self.n_category, self.struct[1]], hidden_func = self.hidden_func)
        self.Q = self.Sequential(struct = self.struct[1:-1], hidden_func = self.hidden_func[1:], dropout = self.exec_dropout[0])
        self.z_mu = self.Sequential(struct = self.struct[-2:], hidden_func = 'a')
        self.z_logvar = self.Sequential(struct = self.struct[-2:], hidden_func = 'a')
        self.qy_out = self.Sequential(struct = [self.struct[-2], self.n_category], hidden_func = 'a')
        
        # p(x|z)
        self.check_decoder()
        self.pz_in = self.Sequential(struct = self.decoder_struct[:2], hidden_func = self.decoder_func)
        self.py_in = self.Sequential(struct = [self.n_category, self.decoder_struct[1]], hidden_func = self.decoder_func)
        self.P = self.Sequential(struct = self.decoder_struct[1:], hidden_func = self.decoder_func[1:], dropout = self.exec_dropout[1])
        self.opt()
    
    def forward(self, x):
        '''
            with label:
                q(x,y -> z,0), p(z,y -> x*) & q(x,0 -> 0,y*)
            miss label:
                q(x,0 -> z,y*), p(z,y* -> x*)
        '''
        # q(z|x)
        u_h = self.qx_in(x)
        if self._target is not None: 
            l_h = u_h + self.qy_in(self._target)
            l_h = self.Q(l_h)
        u_h = self.Q(u_h)
        
        if self._target is not None: h = l_h
        else: h = u_h
        
        z_mu, z_logvar = self.z_mu(h), self.z_logvar(h)
        z = self.sample_z(z_mu, z_logvar)
        y_logits = self.qy_out(u_h)
        y_pre = torch.softmax(y_logits, dim = 1)
        
        if self._target is not None: 
            y = self._target
        else:
            y = y_pre
        
        # p(x|z)
        ph = self.pz_in(z) + self.py_in(y)
        recon = self.P(ph)
        
        # Loss
        recon_loss = nn.functional.binary_cross_entropy(torch.sigmoid(recon), torch.sigmoid(x), reduction='sum') / x.size(0)
        kl_loss = torch.mean(torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1) /2 )
        self.loss = self.alf_r * recon_loss + kl_loss
        #print('\n', recon_loss.data, kl_loss.data, self.loss.data)
        
        if self._target is not None:
            sup_loss = nn.functional.cross_entropy(y_logits, torch.argmax(self._target,1).long())
            self.loss += self.alf_sup * sup_loss

        return y_pre