# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 14:32:16 2022

@author: Fuzz4
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import random
import os
import time
from torch.autograd import Variable
import torch.utils.data as Data

from ..vae import VAE
from ...data.load import _loader_kwargs
from ...core.module import Module
from ...core.fd_statistics import Statistics

def make_loader(X, Y, batch_size, shuffle, dvc):
    if type(X) == np.ndarray: 
        X, Y = torch.from_numpy(X).float(), torch.from_numpy(Y).float()
    dataset = Data.dataset.TensorDataset(X, Y)
    data_loader = Data.DataLoader(dataset, batch_size = batch_size, 
                                  shuffle = shuffle, drop_last = False, **_loader_kwargs(dvc))
    return dataset, data_loader

class Faulty_Dataset():
    def __init__(self, datasets, b, gene_f_min, gene_f_max, dvc, shuffle_Y):
        self.train_X, self.train_Y, self.test_X, self.test_Y = datasets
        self.batch_size = b
        self.gene_f_min, self.gene_f_max = gene_f_min, gene_f_max
        self.dvc, self.shuffle_Y = dvc, shuffle_Y
        
        self.x_std = np.std(self.train_X, axis = 0).reshape(1,-1)
        
        self.train_set, self.train_loader = \
            make_loader(self.train_X, self.train_Y, b, True, self.dvc)
        self.test_set, self.test_loader = \
            make_loader(self.test_X, self.test_Y, b, False, self.dvc)
    
    def transfer_data(self):
        # X 是故障数据，Y 是对应的正常数据
        Y = self.train_X
        X = np.copy(Y)
        # 对 Y 洗牌
        if self.shuffle_Y: np.random.shuffle(Y)
        n, m = X.shape[0], X.shape[1]
        
        switch = np.random.rand(n)
        add_smp = np.where(switch <= self.p_additive_fault)
        mul_smp = np.where(switch > self.p_additive_fault)
        X_add, X_mul = X[add_smp], X[mul_smp]
        
        for i, _X in enumerate( [X_add, X_mul] ):
            # 加入故障的维度
            ni = _X.shape[0]
            dim_p = np.random.rand(ni, m)
            p = 0.1
            indexs = np.where(dim_p <= p)
            
            rd = np.random.rand(ni, m)
            # rd = 2*(rd-0.5) # -1 to 1 (开启结果变差)
            # 故障系数 
            if i == 0:
                add_f = np.sign(rd) * (np.abs(rd)*(self.gene_add_f[1] - self.gene_add_f[0]) + self.gene_add_f[0]) 
                X[add_smp][indexs] = _X[indexs] + np.repeat(self.x_std, ni, axis=0)[indexs] * add_f[indexs]
            else:
                mul_f = np.sign(rd) * (np.abs(rd)*(self.gene_mul_f[1] - self.gene_mul_f[0]) + self.gene_mul_f[0]) 
                X[mul_smp][indexs] = _X[indexs] * mul_f[indexs]
        
        _, self.faulty_loader = \
            make_loader(X, Y, self.batch_size, True, self.dvc)
    
    def shuffle_data(self):
        dataset = self.faulty_loader.dataset
        X, Y =  dataset.tensors[0], dataset.tensors[1]
        Y = Y[torch.randperm(Y.size(0))]
        _, self.faulty_loader = \
            make_loader(X, Y, self.batch_size, True, self.dvc)

class VAE_FIdN(Module):
    def __init__(self, **kwargs):
        default = {'decp_struct': None,
                   'decp_func': None,
                   'struct': None,                       # 编码部分的结构
                   'hidden_func': None,
                   'decoder_struct': None,               # 解码部分的结构
                   'decoder_func': None,
                   'sample_times': 5,
                   'dropout': 0.0,
                   'dvd_epoch': 0.5,                     # 分割训练代数用于 before_transfer 和 transfer
                   'L': 'MSE',
                   'alf': 2e1,
                   'alf_mmd': 1,
                   'gene_add_f': [1.25, 12],
                   'gene_mul_f': [1.25, 3],
                   'p_additive_fault': 0.75,
                   'lr': 1e-3,
                   'lr_tl': 1e-2,
                   'gene_new_data': True,
                   'shuffle_Y': True,
                   'var_msg':['vae_loss', 'mmd_loss']}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        self._name = 'VAE_FIdN'
        Module.__init__(self, **kwargs)
        self._decoupling = self.Sequential(struct = self.decp_struct, 
                                           hidden_func = self.decp_func,
                                           output_func = 'a',
                                           dropout = None)
        
        self._vae = VAE(**kwargs)
        
        self.optimizer_vae = torch.optim.RMSprop(params = self._vae.parameters(),
                                                 lr = self.lr,
                                                 alpha=0.9, 
                                                 eps=1e-10)

        self.optimizer_decp = torch.optim.RMSprop(params = self._decoupling.parameters(),
                                                  lr = self.lr_tl,
                                                  alpha=0.9, 
                                                  eps=1e-10)
    
    def _get_latent(self, x):
        f = -self.decoupling(x)
        return f
    
    def decoupling(self, x):
        b, m = x.size(0), x.size(1)
        self.I_matrix = torch.eye(m).to(self.dvc)
        # b × m × m
        x_epd = x.view(b, 1, m) * self.I_matrix
        # b * m × m
        x_2d = x_epd.view(-1, m)
        h = self._decoupling(x_2d)
        return torch.diagonal(h.view(b,m,m), dim1 = 1, dim2 = 2)
    
    def _mmd(self, normal_x, faulty_x):
        mmd = torch.sum(torch.mean(normal_x - faulty_x, 0) ** 2)
        return mmd
    
    def load_data(self, datasets, b):
        self.train_X, self.train_Y, self.test_X, self.test_Y = datasets
        self.batch_size = b
        self.Data = Faulty_Dataset(datasets, b, 
                                   self.gene_f_min, self.gene_f_max, 
                                   self.dvc, self.shuffle_Y)
        self.train_set, self.train_loader, self.test_set, self.test_loader = \
            self.Data.train_set, self.Data.train_loader, self.Data.test_set, self.Data.test_loader
    
    def forward(self, x):
        if self.stage == 'train_vae' or self.training == False:
            recon_x = self._vae.forward(x)
            self.vae_loss = self._vae.loss
            self.mmd_loss = torch.tensor(float('nan')).to(self.dvc)
            self.loss = self.vae_loss
        elif self.stage == 'transfer_learning':
            input_xs, self.vae_loss = [], 0
            normal_x, faulty_x = self._target, x
            for data in [normal_x, faulty_x]:
                f = self.decoupling(data)
                input_x = data + f
                input_xs.append(input_x)
                recon_x = self._vae.forward(input_x)
                self.vae_loss += self._vae.loss
            self.mmd_loss = self._mmd(input_xs[0], input_xs[1])
            self.loss = self.vae_loss + self.alf_mmd * self.mmd_loss
        
        return recon_x
        
    def batch_training(self, epoch):
        self = self.to(self.dvc)
        self.train()
        # stage - before_transfer:
        if epoch <= int(self.n_epochs * self.dvd_epoch):
            dataloader = self.Data.train_loader
            optimizer = self.optimizer_vae
            self.stage = 'train_vae'
        # stage - transfer_learning:
        else:
            if hasattr(self.Data, 'faulty_loader') == False:
            # if hasattr(self, 'faulty_loader') == False or np.mod(epoch, 5)  == 0:
                print()
                self.Data.transfer_data()
            else:
                if self.gene_new_data: self.Data.transfer_data()
                elif self.shuffle_Y: self.Data.shuffle_data()

            dataloader = self.Data.faulty_loader
            # dataloader = self.train_loader
            optimizer = self.optimizer_decp
            self.stage = 'transfer_learning'
        
        # forward and backward:
        Loss, Vae_loss, Mmd_loss = 0, 0, 0
        for i, (X, Y) in enumerate(dataloader):
            X, Y = X.to(self.dvc), Y.to(self.dvc)
            self._target = Y
            
            # optimizer.zero_grad()
            optimizer.zero_grad()
            _ = self.forward(X)
            self.loss.backward()
            optimizer.step()
            
            loss, vae_loss, mmd_loss = self.loss.item(), self.vae_loss.item(), self.mmd_loss.item()
            Loss, Vae_loss, Mmd_loss = Loss + loss *X.size(0), Vae_loss + vae_loss *X.size(0), \
                Mmd_loss + mmd_loss *X.size(0)
            if (i+1) % 10 == 0 or i == len(dataloader) - 1:
                msg_str ="%s: [Epoch %d/%d | Batch %d/%d] VAE loss: %f, MMD loss: %f, Loss: %f"\
                    % (self.stage, epoch, self.n_epochs, i+1, len(dataloader), vae_loss, mmd_loss, loss)
                sys.stdout.write('\r'+ msg_str + '                                          ')
                sys.stdout.flush()
            
        msg_dict = {}
        N = dataloader.dataset.tensors[0].size(0)
        for key in ['loss'] + self.var_msg:
            msg_dict[key] = np.around(eval(key.capitalize())/N, 4)
        
        self.train_df = self.train_df.append(msg_dict, ignore_index=True)  