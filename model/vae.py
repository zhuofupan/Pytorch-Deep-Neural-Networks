# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import sys
sys.path.append('..')

from core.module import Module

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return Variable(torch.randn(*size) * xavier_stddev, requires_grad=True)

class VAE(Module):
    def __init__(self, **kwargs):
        default = {'struct2': None, # 解码部分的结构，默认为编码部分反向
                   'hidden_func2':None,
                   'dropout': 0.0,
                   'exec_dropout': [True, True],
                   'L':'BCE',
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        kwargs['dvc'] = torch.device('cpu')
        
        self._name = 'VAE'
        super().__init__(**kwargs)
        
        # q(z|x)
        print('\n>>> q(z|x) >>>')
        self.q_dropout, self.q_w, self.q_b, self.q_func = self.gene_para(self.struct, 
                                                                         self.hidden_func,
                                                                         self.exec_dropout[0],
                                                                         (1, 2)
                                                                         )
        
        # p(x|z)
        print('\n>>> p(x|z) >>>')
        if self.struct2 is None:
            self.struct2 = self.struct.copy()
            self.struct2.reverse()
        if self.hidden_func2 is None:
            if type(self.hidden_func) == str: self.hidden_func = [self.hidden_func]
            self.hidden_func2 = self.hidden_func.copy()
            self.hidden_func2.reverse()
            
        self.p_dropout, self.p_w, self.p_b, self.p_func = self.gene_para(self.struct2, 
                                                                         self.hidden_func2,
                                                                         self.exec_dropout[1],
                                                                         (1, 1)
                                                                         )
        
        paras = self.q_w + self.q_b + self.p_w + self.p_b
        self.opt(paras)
    
    def gene_para(self, struct, hidden_func, exec_dropout, _in_out = (1,1)):
        dropout, w, b, func = [], [], [], []
        for i in range(len(struct)-1):
            if exec_dropout and hasattr(self,'dropout') and i < len(struct)-1:
                dropout.append( nn.Dropout(p = self.D('h', i)) )
                print('   ',dropout[-1])
                
            if i == 0: repeat = _in_out[0]
            elif i < len(struct)-2: repeat = 1
            else: repeat = _in_out[1]
            
            for _ in range(repeat):
                w.append( xavier_init(size=[struct[i], struct[i+1]]) )
                b.append( Variable(torch.zeros(struct[i+1]), requires_grad=True) )
                func.append( self.F(hidden_func,i) )
                print('   ','Linear(weight = {}, bias = {})'.format(w[-1].size(), b[-1].size()))
                print('   ',func[-1])
                
        return dropout, w, b, func
    
    def Q(self, x):
        for i in range(len(self.struct)-2):
            try: x = self.q_dropout[i](x)
            except: pass
            x = x @ self.q_w[i] + self.q_b[i].repeat(x.size(0), 1)
            x = self.q_func[i](x)
        z_mu = x @ self.q_w[-2] + self.q_b[-2].repeat(x.size(0), 1)
        z_logvar = x @ self.q_w[-1] + self.q_b[-1].repeat(x.size(0), 1)
        return z_mu, z_logvar
    
    def P(self, x):
        for i in range(len(self.struct2)-1):
            try: x = self.p_dropout[i](x)
            except: pass
            x = x @ self.p_w[i] + self.p_b[i].repeat(x.size(0), 1)
            x = self.p_func[i](x)
        return x
    
    def sample_z(self, z_mu, z_logvar):
        eps = Variable(torch.randn(z_mu.size(0), z_mu.size(1)))
        return z_mu + torch.exp(z_logvar / 2) * eps
    
    def forward(self, x):
        # q(z|x)
        z_mu, z_logvar = self.Q(x)
        z = self.sample_z(z_mu, z_logvar)
        # p(x|z)
        recon = self.P(z)
        
        # Loss
        recon_loss = self.L(recon, x)
        kl_loss = torch.mean(torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1) /2 )
        #print('\n', recon_loss.data, kl_loss.data)
        self.loss = recon_loss + kl_loss
        return recon