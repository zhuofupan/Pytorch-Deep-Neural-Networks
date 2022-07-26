# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:16:54 2021

@author: Fuzz4
"""
import torch
import numpy as np
from ..core.module import Module

def split_data(n_p, n_f, dim_u, datasets):
    dynamic = n_p + n_f
    train_X, train_L, test_X, test_L = datasets
    dim_dy = train_X.shape[1]
    dim_z = int( dim_dy / dynamic )
    train_U, train_Y, test_U, test_Y = [], [], [], []
    for t in range(dynamic):
        u_start, u_end = int(t*dim_z), int(t*dim_z+dim_u)
        y_start, y_end = int(t*dim_z+dim_u), int((t+1)*dim_z)
        train_U.append(train_X[:,u_start: u_end])
        train_Y.append(train_X[:,y_start: y_end])
        test_U.append(test_X[:,u_start: u_end])
        test_Y.append(test_X[:,y_start: y_end])
    train_U, train_Y, test_U, test_Y = np.concatenate(train_U, 1), np.concatenate(train_Y, 1),\
                                       np.concatenate(test_U, 1), np.concatenate(test_Y, 1)
    print(train_U.shape, train_Y.shape, test_U.shape, test_Y.shape)
    print(train_L.shape, test_L.shape)
    train_X, test_X = np.concatenate([train_U, train_Y], 1), np.concatenate([test_U, test_Y], 1)
    # split_point = dim_u * dynamic + dim_y * n_p
    # input = X[:, :split_point], label = X[:, split_point:]
    return train_X, train_L, test_X, test_L

class SupDynamic(Module):
    def __init__(self, **kwargs):
        default = {'dim_u': None,                # 未堆叠前u的维度
                   'dim_y': None,                # 未堆叠前y的维度
                   'n_p': None,                  # 用于预测x(k)的堆叠时刻数
                   'n_f': None,                  # 用于预测后n_f个y_f(k)
                   'struct': []
                   }
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        kwargs['split_p'] = int(kwargs['dim_u']*(kwargs['n_p'] + kwargs['n_f']) +\
                                kwargs['dim_y'] * kwargs['n_p'])
        kwargs['struct'][0] = kwargs['split_p']
        
        self._name = 'SupDynamic'
        Module.__init__(self, **kwargs)
        self.FCNN = self.Sequential()
        self.opt()
        
    def forward(self, dy_x):
        inputs, labels = dy_x[:,:self.split_p], dy_x[:,self.split_p:]
        pred_ys = self.FCNN(inputs)
        self.loss = self.L(pred_ys, labels)
        return labels - pred_ys