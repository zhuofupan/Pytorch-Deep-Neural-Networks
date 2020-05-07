# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.utils.data as Data

class DSAEDataSet(Data.Dataset):
    def __init__(self, X, Y, _shuffle = True):
        self.X, self.Y = X, Y
        if torch.is_tensor(self.X) == False:
            self.X = torch.from_numpy(self.X)
            self.Y = torch.from_numpy(self.Y)
        self.len = self.X.size(0)
        if _shuffle: self.shuffle_data()
        self.p = 0
    
    def shuffle_data(self):
        index = torch.randperm(self.len)
        self.X = self.X[index, :]
        self.Y = self.Y[index, :]
        
    def __getitem__(self, index):
        while torch.argmax(self.Y[index]) != torch.argmax(self.Y[self.p]) or index == self.p:
            self.p = np.mod(self.p + 1, self.len)
        return self.X[index], self.X[self.p], self.Y[index]
    
    def __len__(self):
        return self.len