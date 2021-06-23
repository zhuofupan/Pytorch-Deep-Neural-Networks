# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from ..data.imputation_dataset import ImpuDataSet

class Impu_Module(object):
    def __init__(self, **kwargs):
        default = {'loc_intr': '',          # 引入位置矩阵的方式 {'combine', 'add'}
                   'missing_rate': 0.3,     # 数据的丢失率
                   'how_impu': 'replace',   # 怎么插补 {'grad', 'mid', 'replace'}
                   'compt_loss': 'adv',     # 损失计算方式
                   'coff_grad': 1e3
                   }
        for key in default.keys():
            if key in kwargs.keys(): 
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
                
        if self.loc_intr == 'combine':
            self.Combine = nn.Linear(self.struct[0], self.struct[1])
        elif self.loc_intr == 'add':
            self.Add = nn.Linear(self.struct[0], self.struct[0])
    
    # load data
    def load_impu(self, impu_path, real_path, batch_size, **kwargs):
        self.batch_size = batch_size
        self.train_loader = ImpuDataSet(impu_path, real_path, batch_size, **kwargs)
        self.datasets = self.train_loader.dataset
        self.train_X, self.train_Y, self.test_X, self.test_Y = self.datasets
        return self.datasets
    
    # set input data
    # 这里 self._target 存储的是 self._nan
    # 模型训练的 loss 是在 forward 中计算的，而评价指标是直接对数据集进行计算的
    def _before_fp(self, data, nan):
        if self.task != 'impu':  return data.to(self.dvc), nan.to(self.dvc)
        
        data, nan = data.to(self.dvc), nan.to(self.dvc)
        # print('---',(data*nan).mean(), (data*nan).max(), (data*nan).min())
        if self.how_impu == 'grad':
            if hasattr(self, '_data'): del self._data
            self._data = Variable(data, requires_grad = True)
        else: self._data = data
        self._nan = nan
        
        # self._check_update()
        return self._data, self._nan
    
    # calculate loss
    def _get_impu_loss(self, recon, origin):
        if self.compt_loss == 'all':
            return self.L(recon, origin)
        elif self.compt_loss == 'complete':
            return self.L(recon * (1-self._nan), origin * (1-self._nan))
        elif self.compt_loss == 'missing':
            return self.L(recon * self._nan, origin * self._nan)
        elif self.compt_loss == 'adv':
            r = self.e_prop * 0.5  # 0 to 0.5, 1 to 0.5
            return self.L(recon * self._nan, origin * self._nan) * r *2 + \
                self.L(recon * (1-self._nan), origin * (1-self._nan)) * (1 - r) *2
    
    # update missing values
    def _after_bp(self, recon, origin):
        if self.task != 'impu': return
        if hasattr(self,'update_module') == False or self.update_module == 'D':
            self.update_impu(recon, origin)
              
    def update_impu(self, recon, origin):
        if self.how_impu == 'grad':
            # print((self._data.grad*self._nan).data.mean(), (self._data*self._nan).mean())
            impus = self._data - self._data.grad * self.coff_grad
        elif self.how_impu == 'mid':
            impus = (recon + origin) / 2
        elif self.how_impu == 'replace':
            impus = recon
        self.train_loader.__impu_data__(impus.to('cpu').data)        
    
    # def _check_update(self):
    #     if self._index[0] != 0: return
    #     X, Y, NAN = self.train_set.X, self.train_set.Y, self.train_set.nan
    #     for i in range(X.size(0)):
    #         for j in range(X.size(1)):
    #             if NAN[i,j] == 1:
    #                 print('X:', X[i,j], 'Y:', Y[i,j])
                
        # lack = self._data * self._nan
        # comp = self._data * (1 - self._nan)
        # print('>>> lacked values')
        # print(lack.mean(), lack.min(), lack.max())
        # print('>>> comped values')
        # print(comp.mean(), comp.min(), comp.max())
        
    # def pre_batch_training(self, pre_e, b):
    #     self.pre_training = True
    #     self.train_set.shuffle = True
    #     print('\nPre-training '+self.name+ ' in {}'.format(self.dvc) + self.dvc_info +':')
    #     for epoch in range(1, pre_e + 1):
    #         self.batch_training(epoch)
    #     self.pre_training = False
    #     # for p in self.parameters():
    #     #     p.requires_grad = False
    #     self.train_set.fill_mean()     
    
    # def _hook_input_layer(self, module):
    #     def _backward_linear_function(module, grad_out, grad_in):
    #         grad_out, grad_in = list(grad_out), list(grad_in)
    #         for grad in grad_out:
    #             if self._index[0] == 0:
    #                 print(grad.size(), grad.mean())
    #     module.register_backward_hook(_backward_linear_function)
        