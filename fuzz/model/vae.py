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
        default = {'decoder_struct': None,               # 解码部分的结构，默认为编码部分反向
                   'decoder_func': None,
                   'latent_func': ['a', 'a'],
                   'is_logv2': True,
                   'output_func': None,
                   'v0_2': 1,
                   'sample_times': 5, 
                   'var_msg': ['recon_loss', 'kl_loss'], # 显示loss之外的额外信息
                   'if_vmap': False,
                   'dropout': 0.0,
                   'L': 'MSE',                           # 'MSE' or 'BCE' (要求变量在 0 至 1 之间)
                   'alf': 1e2,
                   'gamma': 1,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        self._name = 'VAE'
        Module.__init__(self, **kwargs)
        if self.task == 'impu':
            Impu_Module.__init__(self, **kwargs)
        
        # q_x(z): x -> mu, log_var (输出激活函数值域必须是 -∞ 到 +∞ 的)
        encoder_struct = self.struct.copy()
        self._encoder = self.Sequential(struct = encoder_struct[:-1], 
                                        hidden_func = self.hidden_func,
                                        output_func = None)
        self._u = self.Sequential(struct = encoder_struct[-2:], 
                                        output_func = self.latent_func[0])
        self._logv2 = self.Sequential(struct = encoder_struct[-2:], 
                                        output_func = self.latent_func[1])
        if self.latent_func[1] in ['ex', 'sp', 'sq', 'e2', 'l2', 'b', '1e', '1s']: 
            self.is_logv2 = False
        
        # p(x|z)
        self.decoder_struct, self.decoder_func = \
            get_decoder_setting(self.struct, self.hidden_func, self.decoder_func)
        self.decoder = self.Sequential(struct = self.decoder_struct, 
                                        hidden_func = self.decoder_func, 
                                        output_func = self.output_func)
        # sampling
        self.mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.struct[-1]), torch.eye(self.struct[-1]))
        self.opt()
    
    def encoder(self, x):
        h = self._encoder(x)
        u, logv2 = self._u(h), self._logv2(h)
        return u, logv2

    def _feature(self, dataset = 'test'):
        if hasattr(self,'record_feature') == False:
            return None
        
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
    
    def sample_z(self, u, v2, eps):
        # z = mu + (var)^(1/2) * eps
        return u + torch.sqrt(v2) * eps
    
    # 用于FD
    def _get_latent(self, x):
        u, logv2 = self.encoder(x)
        return torch.cat((u, logv2), 1)
    
    def forward(self, x):
        # q(z|x)
        u, logv2 = self.encoder(x)
        if self.is_logv2: v2 = torch.exp(logv2)
        else: v2 = logv2
        z_dim = self.struct[-1]
        recon = 0
        
        rd = self.mv_normal.sample(torch.Size([u.size(0), self.sample_times])).to(self.dvc)
        for k in range(self.sample_times):
            z = self.sample_z(u, v2, rd[:,k,:])
            # p(x|z)
            recon += self.decoder(z)
        recon /= self.sample_times
        
        # Loss
        _recon_loss_ = torch.sum((recon - x)**2, -1)
        _kl_loss_ = torch.sum(u**2/self.v0_2 + v2/self.v0_2 - torch.log(v2/self.v0_2), -1)/2 - z_dim /2 
        self._loss_ = _recon_loss_ * self.gamma + _kl_loss_ * self.alf
        
        self.recon_loss, self.kl_loss, self.loss = torch.mean(_recon_loss_),\
            torch.mean(_kl_loss_), torch.mean(self._loss_)
        
        return recon
    
    def forward_without_sampling(self, x, rd):
        recon = 0
        u, logv2 = self.encoder(x)
        if self.is_logv2: v2 = torch.exp(logv2)
        else: v2 = logv2
        self._latent_variables = torch.cat([u, logv2], dim = -1)
        if self.fdi == 'lv': 
            return self._latent_variables
        
        for k in range(self.sample_times):
            if len(rd.size()) == 2: eps = rd[k]
            else: eps = rd[:,k,:]
            z = u + torch.sqrt(v2) * eps
            recon += self.decoder(z)
        recon /= self.sample_times
        
        return recon