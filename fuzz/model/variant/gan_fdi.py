# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:53:32 2022

@author: Fuzz4
"""
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import sys
import numpy as np

from ...core.module import Module
from ..dae import get_decoder_setting
from .vae_fidn import Faulty_Dataset

class GAN_FDI(Module):
    def __init__(self, **kwargs):
        default = {'decoder_func': None,
                   'disc_struct': None,
                   'disc_func':None,
                   'gene_add_f': [1.25, 12],
                   'gene_mul_f': [1.25, 3],
                   'p_additive_fault': 0.75,
                   'shuffle_Y': True,
                   'use_wgan': True,
                   'lambda_gp': 10,
                   'n_critic': 5,
                   'alf': 10,
                   'lr': 1e-3,
                   'var_msg':['g_loss', 'd_loss_r', 'd_loss_f', 'recon_loss']}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        self._name = 'GAN_FDI'
        
        Module.__init__(self, **kwargs)
        
        # Generator
        g_encoder = self.Sequential(output_func = None)
        decoder_struct, decoder_func = get_decoder_setting(self.struct, self.hidden_func, self.decoder_func)
        g_decoder = self.Sequential(struct = decoder_struct,
                                    hidden_func = decoder_func,
                                    output_func = 'a')
        self.Gene = nn.Sequential(*[g_encoder, g_decoder])
        
        # Dicriminator
        if self.disc_struct is None: self.disc_struct = self.struct.copy() + [1]
        if self.disc_func is None: self.disc_func = self.hidden_func.copy()
        self.Disc = self.Sequential(struct = self.disc_struct, 
                                    hidden_func = self.disc_func,
                                    output_func = 's')
        
        self.optimizer_g = torch.optim.Adam(params = self.Gene.parameters(),
                                            lr = self.lr)

        self.optimizer_d = torch.optim.SGD(params = self.Disc.parameters(),
                                           lr = self.lr)
    
    def load_data(self, datasets, b):
        self.train_X, self.train_Y, self.test_X, self.test_Y = datasets
        self.batch_size = b
        self.Data = Faulty_Dataset(datasets, b, 
                                   self.gene_f_min, 
                                   self.gene_f_max, 
                                   self.dvc, 
                                   self.shuffle_Y)
        self.train_set, self.train_loader, self.test_set, self.test_loader = \
            self.Data.train_set, self.Data.train_loader, self.Data.test_set, self.Data.test_loader
    
    def get_normal_data(self, x):
        x_g = self.Gene(x)
        r = x - x_g
        xn = x - torch.sign(r) * torch.sqrt(torch.abs(r))
        # return x_g, xn
        return xn, x_g
    
    def compute_gradient_penalty(self, real_samples, fake_samples):
        """Calculates the gradient penalty loss for WGAN GP"""
        real_samples, fake_samples = real_samples.to(self.dvc), fake_samples.to(self.dvc)
        Tensor = torch.cuda.FloatTensor if self.dvc == torch.device('cuda') else torch.FloatTensor
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), real_samples.size(1))))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_interpolates = self.Disc(interpolates)
        ones = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
        # Get gradient w.r.t. interpolates
        # ∂ outputs/ ∂ inputs * grad_outputs
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
    
    def g_forward(self, xn, xf):
        # trian G
        self.optimizer_g.zero_grad()
        xn_g = self.Gene(xn)
        xf_g = self.Gene(xf)
        self.recon_loss = self.L(xn_g, xn)
        self.g_loss = -torch.mean(torch.log(self.Disc(xn_g))) \
                      -torch.mean(torch.log(self.Disc(xf_g)))
                       
        (self.g_loss + self.recon_loss * self.alf).backward()
        self.optimizer_g.step()
    
    def forward(self, x):
        if self.training:
            xn, xf = self._target, x
            
            if self.use_wgan:
                # trian D
                d_loss = []
                for x_in in [xn, xf]:
                    self.optimizer_d.zero_grad()
                    x_out = self.Gene(x_in)
                    gradient_penalty = self.compute_gradient_penalty(xn.data, x_out.data)
                    _d_loss = -torch.mean(self.Disc(xn)) + torch.mean(self.Disc(x_out)) \
                        + self.lambda_gp * gradient_penalty
                    d_loss.append(_d_loss)
                    _d_loss.backward()
                    self.optimizer_d.step()
                self.d_loss_r, self.d_loss_f = d_loss[0], d_loss[1]
                
                # trian G
                if self.n_training_epoch % self.n_critic == 0:
                    self.g_forward(xn, xf)
                
                self.loss = self.d_loss_r + self.d_loss_f + self.g_loss + self.recon_loss * self.alf
            else:
                # trian D_real
                rd = (torch.rand(x.size(0)) * 0.3 + 0.7 ).to(self.dvc)
                self.optimizer_d.zero_grad()
                self.d_loss_r = -torch.mean(rd * torch.log(self.Disc(xn)))
                self.d_loss_r.backward()
                self.optimizer_d.step()
                
                # trian D_fake
                xn_g = self.Gene(xn)
                xf_g = self.Gene(xf)
                self.optimizer_d.zero_grad()
                self.d_loss_f = -torch.mean(torch.log(1-self.Disc(xn_g.detach()))) \
                                -torch.mean(torch.log(1-self.Disc(xf_g.detach())))
                self.d_loss_f.backward()
                self.optimizer_d.step()
                
                # trian G
                if self.n_training_epoch % self.n_critic == 0:
                    self.g_forward(xn, xf)
                    
                self.loss = self.d_loss_r + self.d_loss_f + self.g_loss + self.recon_loss * self.alf
        else:
            x_g = self.Gene(x)
            self.loss = torch.zeros(1)
            # 用 r = x - x_g 做检测， 用 x - xn 做辨识/重构
            return x_g
        
    def batch_training(self, epoch):
        self = self.to(self.dvc)
        self.train()
        
        self.Data.transfer_data()
        dataloader = self.Data.faulty_loader
        
        # forward and backward:
        G_loss, D_loss_r, D_loss_f, Recon_loss, Loss = 0, 0, 0, 0, 0
        self.n_training_epoch = 0
        for i, (X, Y) in enumerate(dataloader):
            X, Y = X.to(self.dvc), Y.to(self.dvc)
            # X 是故障数据， Y是正常数据
            self._target = Y
            
            self.forward(X)
            
            self.n_training_epoch += 1
            # 显示与记录
            g_loss, d_loss_r, d_loss_f, recon_loss, loss = self.g_loss.item(), self.d_loss_r.item(),\
                self.d_loss_f.item(), self.recon_loss.item(), self.loss.item()
            G_loss, D_loss_r, D_loss_f, Recon_loss, Loss = G_loss + g_loss * X.size(0), \
                D_loss_r + d_loss_r * X.size(0), D_loss_f + d_loss_f * X.size(0), \
                Recon_loss + recon_loss * X.size(0), Loss + loss * X.size(0)
                
            if (i+1) % 10 == 0 or i == len(dataloader) - 1:
                msg_str ="[Epoch %d/%d | Batch %d/%d] G_loss: %.4f, D_loss: %.4f (r), %.4f (f), Recon_loss: %.4f, Loss: %.4f"\
                    % (epoch, self.n_epochs, i+1, len(dataloader), g_loss, d_loss_r, d_loss_f, recon_loss, loss)
                sys.stdout.write('\r'+ msg_str + '                                  ')
                sys.stdout.flush()

        msg_dict = {}
        N = dataloader.dataset.tensors[0].size(0)
        for key in ['loss'] + self.var_msg:
            msg_dict[key] = np.around(eval(key.capitalize())/N, 4)
        
        self.train_df = self.train_df.append(msg_dict, ignore_index=True)