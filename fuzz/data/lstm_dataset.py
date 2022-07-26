# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.utils.data as Data

class LSTMDataSet(Data.Dataset):
    def __init__(self, X, Y, batch_size, use_for = 'train'):
        self.X, self.Y = X, Y
        self.dataset = self
        self.batch_size = batch_size
        if torch.is_tensor(self.X) == False:
            self.X = torch.from_numpy(self.X)
            self.Y = torch.from_numpy(self.Y)
        self.len, self.seq_len, self.dim = self.X.size()
        _, _, self.dim_y = self.Y.size()
        
        self.use_for = use_for
        # 各线路偏移 index 的长度（起始设置为 0 ~ batch_size - 1）
        self.batch_bias = list(range(batch_size))
            
        # 用于训练/测试的 X 和 Y
        self.batch_X = torch.zeros((self.batch_size, self.seq_len, self.dim))
        self.batch_Y = torch.zeros((self.batch_size, self.seq_len, self.dim_y))
    
    def __getitem__(self, index):
        if index >= self.len:
            raise StopIteration
            
        # index 是 0 号线 编号
        self.index = index
        # 获取 batch data
        for b in range(self.batch_size):
            i = np.mod(self.batch_bias[b] + index, self.len)
            self.batch_X[b,:,:] = self.X[i,:,:]
            self.batch_Y[b,:,:] = self.Y[i,:,:]
        
        return self.batch_X, self.batch_Y
    
    def __len__(self):
        return self.len
        