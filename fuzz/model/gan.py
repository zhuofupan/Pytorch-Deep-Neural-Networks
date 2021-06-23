# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from ..core.module import Module
from ..core.impu_module import Impu_Module

class GAN(Module, Impu_Module):
    def __init__(self, **kwargs):
        default = {'dicr_struct': None,
                   'dicr_func':None,
                   'dropout': 0.0,
                   'exec_dropout': ['h', None],
                   'alf': 1e3,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['output_func'] = None
        
        self._name = 'GAN'
        super().__init__(**kwargs)
        if self.task == 'impu':
            Impu_Module.__init__(self, **kwargs)
        
        # Generator
        self.Gene = self.Sequential(struct = self.struct, 
                                 hidden_func = self.hidden_func,
                                 dropout = self.exec_dropout[0])
        _, G_para = self._get_para('all', self.Gene)
        # Dicriminator
        if self.dicr_func is None: self.dicr_func = self.hidden_func.copy()
        self.Dicr = self.Sequential(struct = self.dicr_struct, 
                                 hidden_func = self.dicr_func, 
                                 dropout = self.exec_dropout[1])
        _, D_para = self._get_para('all', self.Dicr)
        
        if hasattr(self, 'batch_size'):
            self.ones_label = Variable(torch.ones(self.batch_size, 1))
            self.zeros_label = Variable(torch.zeros(self.batch_size, 1))
        
        self.opt(G_para, info = False)
        self.G_optim = self.optim
        self.opt(D_para)
        self.D_optim = self.optim
        self.jump_bp = True
    
    def forward(self, x):
        if self.task == 'impu': 
            return self.impu_forward(x)
        
        # Dicriminator forward-loss-backward-update
        z = Variable(torch.randn(self.batch_size, self.struct[0]))
        G_sample = self.Gene(z)
        D_real = self.Dicr(x)
        D_fake = self.Dicr(G_sample.detach())
        
        D_loss_real = nn.binary_cross_entropy(D_real, self.ones_label)
        D_loss_fake = nn.binary_cross_entropy(D_fake, self.zeros_label)
        D_loss = D_loss_real + D_loss_fake
        
        D_loss.backward()
        self.D_optim.step()
        
        # Housekeeping - reset gradient
        self.zero_grad()
        
        # Generator forward-loss-backward-update
        z = Variable(torch.randn(self.batch_size, self.struct[0]))
        G_sample = self.Gene(z)
        D_fake = self.Dicr(G_sample)
        
        G_loss = nn.binary_cross_entropy(D_fake, self.ones_label)
        
        G_loss.backward()
        self.G_optim.step()
        
        return G_sample
    
    def impu_forward(self, x):
        nan = self._nan
        mask = 1 - nan
        
        G_sample = self.Gene(x)
        G_recon = G_sample * nan + x * mask
        D_recon = self.Dicr(G_recon.detach())
        
        D_loss = -1 * torch.mean( (mask * torch.log(D_recon) + (1 - mask) * torch.log(1 - D_recon)))
        self.loss = D_loss
        D_loss.backward()
        self.D_optim.step()
        
        # Housekeeping - reset gradient
        self.zero_grad()
        
        # Generator forward-loss-backward-update
        G_sample = self.Gene(x)
        G_recon = G_sample * nan + x * mask
        D_recon = self.Dicr(G_recon)
        
        gene_loss =  -1 * torch.mean(( (1 - mask) * torch.log(D_recon)))
        recon_loss = torch.mean((G_sample * mask - x * mask)**2)
        G_loss = gene_loss + self.alf * recon_loss
        self.loss = G_loss
        G_loss.backward()
        self.G_optim.step()
        
        return G_recon