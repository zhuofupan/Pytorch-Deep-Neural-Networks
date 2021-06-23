# -*- coding: utf-8 -*-
# https://github.com/zhenxuan00/mmdgm/blob/9422bdc88b913c09506944d3529f6f955b95687c/mlp-mmdgm/anglepy/models/GPUVAE_MM_Z_X.py

import torch
import torch.nn as nn
import torch.utils.data as Data
import sys

from ..vae import VAE
from ..dae import get_decoder_setting
from ...core.module import Module
from ...core.func import find_act
from ...visual.plot import t_SNE

class MMDGM_VAE(VAE):
    '''
                                 qy-> y_logits -qy-> y 
                      ┏-----------┿------┓
        x -Q_x-┯-Q-> h_z -z_mu-> z_mu ---┿--sample-> z -P_z-┯-P-> recon 
       (y) -Q_y┛      ┗---z_logvar-> z_logvar ┛     (y) -P_y┛
    '''
    def __init__(self, **kwargs):
        default = {'decoder_struct': None,   # 解码部分的结构，默认为编码部分反向
                   'decoder_func':None,
                   'n_category': None,
                   'coff': { 'r': 1e-3, 'kl': 1, 'w2': 0, 'sp': 0, 'e': 1},
                   'exec_dropout': ['h', 'h', 'h'],
                   'add_y_to_pre': True,
                   'add_z_to_logits': True,
                   'p_sp': 1e-2,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        self._name = 'MMDGM_VAE'
        Module.__init__(self, **kwargs)
        
        # q(z|x)
        self.Q_x = self.Sequential(struct = self.struct[:2], hidden_func = self.hidden_func, 
                                   output_func = None, dropout = self.exec_dropout[0])
        if self.add_y_to_pre:
            self.Q_y = self.Sequential(struct = [self.n_category, self.struct[1]], hidden_func = self.hidden_func, 
                                       output_func = None, dropout = self.exec_dropout[0])
        
        if type(self.hidden_func) == str: self.hidden_func = [self.hidden_func]
        hidden_func = self.hidden_func.copy()
        while len(hidden_func) < len(self.struct):
            hidden_func = hidden_func + self.hidden_func.copy()
        self.Q = self.Sequential(struct = self.struct[1:-1], hidden_func = self.hidden_func[1:], 
                                 output_func = None, dropout = self.exec_dropout[0])
        
        self.z_mu = self.Sequential(struct = self.struct[-2:], 
                                    output_func = 'a', dropout = None)
        self.z_logvar = self.Sequential(struct = self.struct[-2:], 
                                        output_func = 'a', dropout = None)
        
        # p(x|z)
        self.decoder_struct, self.decoder_func = \
            get_decoder_setting(self.struct, self.hidden_func)
        self.P_z = self.Sequential(struct = self.decoder_struct[:2], hidden_func = self.decoder_func, 
                                   output_func = None, dropout = self.exec_dropout[1])
        if self.add_y_to_pre:
            self.P_y = self.Sequential(struct = [self.n_category, self.decoder_struct[1]], hidden_func = self.decoder_func, 
                                       output_func = None, dropout = self.exec_dropout[1])
        
        hidden_func = self.decoder_func.copy()
        while len(hidden_func) < len(self.decoder_struct):
            hidden_func = hidden_func + self.decoder_func.copy()
        self.P = self.Sequential(struct = self.decoder_struct[1:], hidden_func = self.decoder_func[1:], 
                                 output_func = 'a', dropout = self.exec_dropout[1])
        
        # q(y|x)
        self.qy = []
        qy_func = 'o'
        for i, n_h in enumerate(self.struct):
            if i > 0 and i < len(self.struct) - 1:
                self.qy.append(self.Sequential(struct = [n_h, self.n_category], 
                                               output_func = qy_func, dropout = self.exec_dropout[2]))
        if self.add_z_to_logits:
            self.qy.append(self.Sequential(struct = [self.struct[-1], self.n_category], 
                                           output_func = qy_func, dropout = self.exec_dropout[2]))
            self.qy.append(self.Sequential(struct = [self.struct[-1], self.n_category], 
                                           output_func = qy_func, dropout = self.exec_dropout[2]))
        if hasattr(self, 'output_func'): self.qy.append(self.F('x'))
        self.Y = nn.Sequential(*self.qy)
        self.opt()
    
    def pre_test(self, data = 'train'):
        # 获取Q生成的z之前的特征
        if data == 'train':
            test_loader = self.train_loader
        elif data == 'test':
            test_loader = self.test_loader
        else:
            test_loader = data
            
        X = test_loader.dataset.tensors[0].data.cpu()
        Y = test_loader.dataset.tensors[1].data.cpu()
        
        test_set = Data.dataset.TensorDataset(X, Y)
        test_loader = Data.DataLoader(test_set, batch_size = self.batch_size, 
                                      shuffle = False, drop_last = False, **self.loader_kwargs)
        self.eval()
        self = self.to(self.dvc)
        
        feature = []
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                x = data.to(self.dvc)
                h = self.Q(self.Q_x(x))
                h = h.data.cpu()
                feature.append(h)
        feature = torch.cat(feature, 0).numpy()
        del test_set
        del test_loader
        return feature
    
    def _plot_pre_feature_tsne(self, loc = -1, data = 'train'):
        feature, Y = self.pre_test(data = data), self.train_loader.dataset.tensors[1]
        Y = Y.data.cpu().numpy()
        print(feature.shape, Y.shape)
        
        path ='../save/'+ self.name + self.run_id + '/'
        file_name = '['+ self.name + '] _' + data + ' (pre-trained feature).png'
        show_info = True
        if hasattr(self, 'tsne_info'):
            show_info = self.tsne_info
        t_SNE(feature, Y, path, file_name, show_info = show_info)
    
    def pre_batch_training(self, pre_e, pre_b):
        if pre_e > 0:
            task = self.task
            self.task = 'gnr'
            for epoch in range(1, pre_e + 1):
                self.batch_training(epoch)
            self.add_msg = ''
            self.task = task

    def _get_loss_and_logits(self, layer_out, layer):
        if self.task == 'cls' and isinstance(layer, nn.Linear):
            # y_logits
            self._y_logits += self.qy[self._pointer](layer_out)
            self._pointer += 1
            # w2_loss
            if self.coff['w2'] > 0:
                self._w2_loss += torch.sum(layer.weight **2)
        # sparsity_penalty
        if self.coff['sp'] > 0 and find_act(layer) is not None:
            a = self.p_sp
            h_s = torch.clamp(layer_out.abs(), 0, 1)
            # h_s = (h_z - h_z.min()) / (h_z.max() - h_z.min())
            self._sparsity_penalty += torch.mean( a*torch.log(a/h_s) + (1-a)*torch.log((1-a)/(1-h_s)))

    def forward(self, x):
        self._y_logits, self._pointer, self._w2_loss, self._sparsity_penalty = 0, 0, 0, 0
        
        origin = x.clone()
        # q(z|x)
        q_h = x
        for layer in self.Q_x:
            q_h = layer(q_h.clone())
            self._get_loss_and_logits(q_h, layer)
            
        if self.task == 'gnr' and self.add_y_to_pre and self.training:
            q_h += self.Q_y(self._target)

        for layer in self.Q:
            q_h = layer(q_h)
            self._get_loss_and_logits(q_h, layer)
        
        z_mu, z_logvar = self.z_mu(q_h), self.z_logvar(q_h)
        z = self.sample_z(z_mu, z_logvar)
        
        # p(x|z)
        p_h = self.P_z(z)
        if self.task == 'gnr' and self.add_y_to_pre and self.training:
            p_h += self.P_y(self._target)
        recon = self.P(p_h)

        # msg_info
        msg = ''
        w2_loss, sparsity_penalty = 0, 0
        
        if self.task == 'cls':
            self.coff['e'] *= 0.9999
        
        if self.coff['w2'] > 0:
            w2_loss = self.coff['w2'] * self._w2_loss
            msg += ', w2_loss = {:.4f}'.format(self.coff['e'] * w2_loss.data)
        if self.coff['sp'] > 0:
            sparsity_penalty = self.coff['sp'] * self._sparsity_penalty
            msg += ', sp_loss = {:.4f}'.format(self.coff['e'] * sparsity_penalty.data)
        
        # for 'gnr' task (pre-training)
        if self.task == 'gnr':
            # recon_Loss
            recon_loss = self.coff['r'] * nn.functional.binary_cross_entropy(torch.sigmoid(recon), torch.sigmoid(origin), reduction='sum') / origin.size(0)
            
            # kl_loss
            kl_loss = self.coff['kl'] * torch.mean(torch.sum(torch.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1) /2 )
            
            self.loss =  recon_loss + kl_loss + w2_loss + sparsity_penalty
            
            self.add_msg = ' (recon_loss = {:.4f}, kl_loss = {:.4f}'.format(recon_loss.data, kl_loss.data) + msg + ')    '
            return recon
        
        # for 'cls' task (fine-tuning)
        elif self.task == 'cls':
            if self.add_z_to_logits:
                self._y_logits += self.qy[self._pointer](z_mu) + self.qy[self._pointer+1](z_logvar)
            
            # sup_loss
            if isinstance(self.L, nn.CrossEntropyLoss):
                y = torch.softmax(self._y_logits, dim =1)
                sup_loss = nn.functional.cross_entropy(self._y_logits, torch.argmax(self._target, 1).long())
            else:
                y = self.qy[-1](self._y_logits)
                sup_loss = self.L(y, self._target)
                
            self.loss = sup_loss + self.coff['e'] * (w2_loss + sparsity_penalty)
            
            self.add_msg = ''
            if msg!= '':
                self.add_msg = ' (sup_loss = {:.4f}'.format(sup_loss.data) + msg + ')    '
            return y