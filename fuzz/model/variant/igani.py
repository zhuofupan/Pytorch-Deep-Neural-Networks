# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd
from torch.autograd import Variable

from ...core.module import Module
from ...core.impu_module import Impu_Module

class IGANI(Module, Impu_Module):
    def __init__(self, **kwargs):
        default = {'dicr_struct': None,
                   'dicr_func':None,
                   'dropout': 0.0,
                   'exec_dropout': ['h', None],
                   'L': 'BCE',
                   'loss_compt': 1,
                   'd_alf': 1e1,
                   'g_alf': 1e2,
                   'n_critic': 5,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['output_func'] = None
        
        self._name = 'IGANI'
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
        
        self.opt(G_para)
        self.G_optim = self.optim
        self.opt(D_para)
        self.D_optim = self.optim
        self.jump_bp = True 
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        # Wasserstein GAN GP: https://github.com/eriklindernoren/PyTorch-GAN
        """Calculates the gradient penalty loss for WGAN GP"""
        # Random weight term for interpolation between real and fake samples
        alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1))).to(self.dvc)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.Dicr(interpolates)
        ones = Variable(torch.ones_like(d_interpolates).to(self.dvc), requires_grad=False)
        # Get gradient w.r.t. interp olates
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=ones,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def gene_sample(self, x):
        # _nan 标记缺失位置， mask 标记实值位置
        nan = self._nan
        mask = 1 - nan
        
        z = Variable(torch.randn(x.size(0), self.struct[0]).to(self.dvc), requires_grad=False)
        u = x * mask + z * nan
        y = self.Gene(u)
        v = x * mask + y * nan
        
        n = mask.cpu().numpy().reshape(-1,)
        np.random.shuffle(n)
        n = Variable(torch.from_numpy(n).view_as(mask).to(self.dvc), requires_grad=False)
        
        z = Variable(torch.randn(x.size(0), self.struct[0]).to(self.dvc), requires_grad=False)
        u_fake = v * n + z * (1 - n)
        y_fake = self.Gene(u_fake)
        v_fake = v * n + y_fake * (1 - n)
        return v, y, n, v_fake, y_fake
    
    def forward(self, x):
        nan = self._nan
        mask = 1 - nan
        # Dicriminator forward-loss-backward-update
        v, _, n, v_fake, _ = self.gene_sample(x)
        
        real_validity = self.Dicr(v.detach())
        fake_validity = self.Dicr(v_fake.detach())
        
        # z = Variable(torch.randn(x.size(0), self.struct[0]).to(self.dvc), requires_grad=False)
        # gradient_penalty = self.compute_gradient_penalty(v.data, self.Gene(z).data)
        gradient_penalty = self.compute_gradient_penalty(v.data, v_fake.data)
        
        if self.loss_compt == 1:
            n[mask == 0] = 0
            d_loss = self.L(real_validity, mask) + self.L(fake_validity, n)
        elif self.loss_compt == 2:
            d_loss = torch.mean(real_validity) - torch.mean(fake_validity)

        D_loss = d_loss + self.d_alf * gradient_penalty
        self.loss = D_loss
        # print('d_loss:', d_loss, 'W-GP:', gradient_penalty, 'D_loss:', D_loss)
        D_loss.backward()
        self.D_optim.step()
        self.zero_grad()
        
        if self.cnt_iter % self.n_critic == 0:
            
            # Generator forward-loss-backward-update
            v, y, n, v_fake, y_fake = self.gene_sample(x)
            
            real_validity = self.Dicr(v)
            fake_validity = self.Dicr(v_fake)
            
            if self.loss_compt == 1:
                g_loss = -1 * torch.mean( (1 - n) * torch.log(fake_validity)) - \
                    torch.mean( (1 - mask) * torch.log(real_validity))
            elif self.loss_compt == 2:
                g_loss = -torch.mean(fake_validity)
            
            r_loss = torch.mean((y * mask - x * mask)**2) + torch.mean((y_fake * n - v * n)**2)
            G_loss = g_loss + self.g_alf * r_loss
            # print('g_Loss:', G_loss, 'r_loss:', r_loss, 'G_loss:', G_loss)
            self.loss = G_loss
            G_loss.backward()
            self.G_optim.step()
        
        return v