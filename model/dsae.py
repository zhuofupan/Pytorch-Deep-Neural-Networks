# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np

import sys
sys.path.append('..')
from model.sae import AE, SAE
from core.layer import Linear2
from data.dsae_dataset import DSAEDataSet

class SupAE(AE):
    def __init__(self, w, b, func, cnt, **kwargs):
        default = {'share_w': False,
                   'lam': 10,
                   'cof': 1e-3,
                   'p0': torch.tensor(0.05),
                   'name': 'SupAE-{}'.format(cnt+1)}
        
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
                
        super().__init__(w, b, func, cnt, **kwargs) 
        self.lam = self.lam * (w.size(1)/w.size(0)) * np.power(10, cnt)
        self.cof = self.cof / np.power(10, cnt)
    
    def forward(self, x1, x2):
        p0 = self.p0
        f1, f2 = self.encoder(x1), self.encoder(x2)
        g2 = self.decoder(f2)
        
        p1, p2 = torch.mean(0.5* (f1 + 1),0), torch.mean(0.5* (f2 + 1),0)
        self.kl1 = torch.sum(p0 * torch.log(p0/p1) + (1-p0)* torch.log((1-p0)/(1-p1)))
        self.kl2 = torch.sum(p0 * torch.log(p0/p2) + (1-p0)* torch.log((1-p0)/(1-p2)))
        self.l1, self.l2 = self.L(g2, x1), self.L(f2, f1)
        self.loss = self.l1 + self.lam * self.l2 + self.cof * (self.kl1 + self.kl2)
        return g2
    
    def batch_training(self, epoch):
        if epoch == 1:
            print('\nTraining '+self.name+ ' in {}'.format(self.dvc) + self.dvc_info +':')
            
        self = self.to(self.dvc)
        self.train()
        train_loss = 0

        for batch_idx, (x1, x2, target) in enumerate(self.train_loader):
            x1, x2, target = x1.to(self.dvc), x2.to(self.dvc), target.to(self.dvc)
            self._target = target
            
            self.zero_grad()
            output = self.forward(x1, x2)
            output, loss = self.get_loss(output, target)
            loss.backward()
            self.optim.step()
            
            train_loss += (loss.detach() * x1.size(0))
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(self.train_loader):
                self.msg_str = 'Epoch: {} - {}/{} | l1 = {:.4f}, lam *l2 = {:.4f}, cof *kl1 = {:.4f}, cof *kl2 = {:.4f}, loss = {:.4f}'.format(
                        epoch, batch_idx+1, len(self.train_loader),
                        self.l1.detach(), self.lam * self.l2.detach(), 
                        self.cof * self.kl1.detach(), self.cof * self.kl2.detach(), loss.detach())
                for item in self.msg:
                    if hasattr(self, item):
                        self.msg_str += '   '+item+' = {:.4f}'.format(eval('self.'+item))
                sys.stdout.write('\r'+ self.msg_str)
                sys.stdout.flush()
                      
        train_loss = train_loss/ len(self.train_loader.dataset)

class DSAE(SAE):  
    def __init__(self, **kwargs):
        if 'name' in kwargs.keys(): 
            kwargs['_name'] = kwargs['name']
            del kwargs['name']
        if '_name' not in kwargs.keys(): kwargs['_name'] = 'Stacked_SupAE'
        
        super().__init__(**kwargs)
    
    def add_pre_module(self, w, b, cnt):
        ae_func = self.get_sub_func(cnt)
        if ae_func[0] in ['a', 'Linear']: ae_func[0] = 'r'
        ae = SupAE(w, b, ae_func, cnt, **self.kwargs)
        return ae

    def pre_batch_training(self, pre_epoch, pre_batch_size):
        # 预训练 所有子模型
        self.pre_batch_size = pre_batch_size
        X = self.train_loader.dataset.tensors[0].data.cpu()
        Y = self.train_loader.dataset.tensors[1].data.cpu()
        train_set = DSAEDataSet(X, Y)
        train_loader = Data.DataLoader(train_set, batch_size = self.pre_batch_size, 
                                       shuffle = True, drop_last = False, **self.loader_kwargs)
        
        features = []
        for k, module in enumerate(self.pre_modules):
            module.train_loader, module.train_set = train_loader, train_loader.dataset
            module.dvc_info = self.dvc_info
            
            if pre_epoch > 0:
                for i in range(1, pre_epoch + 1):
                    module.batch_training(i)
                    # 检测预训练是否正常进行 
                    # print('\n',module.w.is_leaf, module.w.is_cuda, module.w.mean(), module.w.grad.mean())
                print()
            
            X = train_loader.dataset.X.data.cpu()
            Y = train_loader.dataset.Y.data.cpu()
            # module 前向传播
            X = self._sub_module_test(module, X, Y).data.cpu()
            train_set = DSAEDataSet(X, Y)
            train_loader = Data.DataLoader(train_set, batch_size = self.pre_batch_size, 
                                           shuffle = True, drop_last = False, **self.loader_kwargs)
            features.append(X.numpy())
        self.pre_features = features
        return features, Y
    