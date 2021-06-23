# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

from ...core.module import Module
from ...core.impu_module import Impu_Module

class GAIN(Module, Impu_Module):
    def __init__(self, **kwargs):
        default = {'dicr_struct': None,
                   'dicr_func':None,
                   'dropout': 0.0,
                   'exec_dropout': ['h', None],
                   '_inherit': False,
                   '_part': True,
                   'L': 'BCE',
                   'alf': 1e3,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['output_func'] = None
        
        self._name = 'GAIN'
        super().__init__(**kwargs)
        if self.task == 'impu':
            Impu_Module.__init__(self, **kwargs)
            
        # Generator
        self.G_x = self.Sequential(struct = self.struct[:2], 
                                 hidden_func = self.hidden_func,
                                 dropout = self.exec_dropout[0])
        self.G_z = self.Sequential(struct = self.struct[:2], 
                                 hidden_func = self.hidden_func,
                                 dropout = self.exec_dropout[0])
        self.G_m = self.Sequential(struct = self.struct[:2], 
                                 hidden_func = self.hidden_func,
                                 dropout = self.exec_dropout[0])
        self.Gene = self.Sequential(struct = self.struct[1:], 
                                 hidden_func = self.hidden_func[1:],
                                 dropout = self.exec_dropout[0],
                                 __drop__ = [True, self.__drop__[1]])
        _, G_para_1 = self._get_para('all', self.G_x)
        _, G_para_2 = self._get_para('all', self.G_z)
        _, G_para_3 = self._get_para('all', self.G_m)
        _, G_para_4 = self._get_para('all', self.Gene)
        G_para = G_para_1 + G_para_2 + G_para_3 + G_para_4
            
        # Dicriminator
        if self.dicr_func is None: self.dicr_func = self.hidden_func.copy()
        self.D_x = self.Sequential(struct = self.dicr_struct[:2], 
                                 hidden_func = self.dicr_func, 
                                 dropout = self.exec_dropout[1])
        self.D_h = self.Sequential(struct = self.dicr_struct[:2], 
                                 hidden_func = self.dicr_func, 
                                 dropout = self.exec_dropout[1])
        self.Dicr = self.Sequential(struct = self.dicr_struct[1:], 
                                 hidden_func = self.dicr_func[1:], 
                                 dropout = self.exec_dropout[1],
                                 __drop__ = [True, self.__drop__[1]])
        _, D_para_1 = self._get_para('all', self.D_x)
        _, D_para_2 = self._get_para('all', self.D_h)
        _, D_para_3 = self._get_para('all', self.Dicr)
        D_para = D_para_1 + D_para_2 + D_para_3
        
        self.opt(G_para, info = False)
        self.G_optim = self.optim
        self.opt(D_para)
        self.D_optim = self.optim
        self.update_module = 'D'
        self.jump_bp = True
    
    def forward(self, x):
        # _nan 标记缺失位置， mask 标记实值位置
        nan = self._nan
        mask = 1 - nan
        x_in = x * mask

        # Gene
        if self._inherit:
            z = x
        else:
            z = Variable(torch.randn(x.size(0), self.struct[0]).to(self.dvc), requires_grad=False)

        G_x = self.G_x(x_in) + self.G_z(z * nan) + self.G_m(mask)
        x_gene = self.Gene(G_x)
        x_recon = x_gene * nan + x_in
        if self.update_module == 'D': x_recon = x_recon.detach()

        # Dicr
        b = Variable(torch.ones_like(mask).to(self.dvc), requires_grad=False)
        rd = torch.rand(mask.size(0),mask.size(1))
        b[rd < 1/5] = 0
        h = b * mask + 0.5 * (1 - b)
        
        D_x = self.D_x(x_recon) + self.D_h(h)
        D_validity = self.Dicr(D_x)
        D_validity = torch.clamp(D_validity, 0, 1)
        
        if self._part:
            part = 1-b
        else:
            part = 1
        if self.training and self.update_module == 'D':
            self.update_module = 'G'
            D_loss = -1 * torch.mean( part *(mask * torch.log(D_validity) + (1 - mask) * torch.log(1 - D_validity)))
            self.loss = D_loss
            D_loss.backward()
            self.D_optim.step()
        elif self.training and self.update_module == 'G':
            self.update_module = 'D'
            
            gene_loss =  -1 * torch.mean(part * ( (1 - mask) * torch.log(D_validity)))
            recon_loss = torch.mean((x_gene * mask - x_in)**2)
            G_loss = gene_loss + self.alf * recon_loss
            # print('\nG:', gene_loss, 'R:', recon_loss)
            self.loss = G_loss
            G_loss.backward()
            self.G_optim.step()
            
        return x_recon