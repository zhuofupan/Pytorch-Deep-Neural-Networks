# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data

from ..core.module import Module
from ..core.layer import LSTMCell
from ..data.gene_dynamic_data import ReadData

class LSTM(Module):
    def __init__(self, **kwargs):
        default = {'train_c0_h0': True,
                   'stride': None,
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        self._name = 'LSTM'
        Module.__init__(self, **kwargs)
        
        # lstm 模型
        self.lstm = []
        for i in range(len(self.struct)-1):
            self.lstm.append(LSTMCell(self.struct[i], self.struct[i+1], self.stride))
        self.lstm = nn.Sequential(*self.lstm)
        self.state = 'trian'
        self.opt()
    
    # 创建 DataLoader
    def load_data(self, path, batch_size, **kwargs):
        self.RD = ReadData(path, **kwargs)
        # 需要满足两个条件不然报错
        # 1. mod(len(X), batch_size) = 0
        # 2. batch_size * stride < seq_len
        
        self.train_X, self.train_Y, self.test_X, self.test_Y = dataset
        self.batch_size = batch_size   
        
        # 记录 t0 的 c,h 值
        self.h0, self.c0 = [], []
        self.h_out, self.c_out = [], []
        for i in range(len(self.struct)-1):
            self.h0.append( torch.randn(self.batch_size, self.struct[i+1]).to(self.dvc) )
            self.c0.append( torch.randn(self.batch_size, self.struct[i+1]).to(self.dvc) )
            self.h_out.append(self.h0[-1].clone().to(self.dvc) )
            self.c_out.append(self.c0[-1].clone().to(self.dvc) )

        self.train_loader = LSTMDataSet(self.train_X, self.train_Y, batch_size = batch_size)
        self.test_loader = LSTMDataSet(self.test_X, self.test_Y, batch_size = 1)
    
    # 预训练 c0, h0 (batch_size = 1)
    # def pre_batch_training(self, pre_epoch, pre_batch_size):
    #     if self.train_c0_h0 == False: return
    
    def forward(self, x):
        for k in range(len(self.lstm)):
            h_in, c_in = self.h_out[k], self.c_out[k]
            if self.training:
                # 测试之后回到 t0
                if self.state == 'test':
                    self.state = 'trian'
                    h_in, c_in = self.h0[k], self.c0[k]
            else:
                # 训练之后回到 t0
                if self.state == 'trian':
                    self.state = 'test'
                    h_in, c_in = self.h0[k], self.c0[k]
                    
            x, (self.h_out[k], self.c_out[k]) = self.lstm[k](x, (h_in, c_in)) 
        return x