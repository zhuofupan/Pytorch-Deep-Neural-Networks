# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np

from ..sae import SAE
from ..dbn import DBN
from ...core.module import Module
from ...core.pre_module import Pre_Module
from ...core.layer import Linear2, make_noise


class OCON(Module):
    def __init__(self, **kwargs):
        default = {'sub_name': 'DBN',
                   'n_category': None}
        
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
                
        Module.__init__(self, **kwargs)
        
        if self.n_category is None: self.n_category = self.struct[-1]
        
        sub_struct = self.struct.copy()
        for i in range(1,len(sub_struct)):
            sub_struct[i] = int(sub_struct[i]/10)
        sub_struct[-1] = 2
        sub_kwargs = kwargs.copy()
        sub_kwargs['struct'] = sub_struct
        sub_kwargs['show_model_info'] = False
        
        self.subnets, self.output_linears = [], []
        for i in range(self.n_category):
            sub_kwargs['name'] = self.sub_name + '-sub' + str(i+1)
            self.subnets.append(eval(self.sub_name+"(**sub_kwargs)"))
            if i == self.n_category - 1: bias = True
            else: bias = False
            self.output_linears.append( nn.Linear(2, self.n_category, bias = bias))
            
        self.subnets = nn.Sequential(*self.subnets)
        self.output_linears = nn.Sequential(*self.output_linears)
        
        self.output_func_layer = self.F('o')
        self.opt()
    
    def forward(self, x):
        logits = 0
        for i in range(len(self.subnets)):
            h = self.subnets[i].forward(x)
            logits += self.output_linears[i](h)
        y = self.output_func_layer(logits)
        return y
    
    def pre_batch_training(self, pre_epoch, pre_batch_size):
        self._datasets = {}
        for i, sample in enumerate(self.train_X):
            lable = int(np.argmax(self.train_Y[i]))
            if str(lable) not in self._datasets.keys():
                self._datasets[str(lable)] = []
            else:
                self._datasets[str(lable)].append(sample.reshape(1,-1))
        for c in range(self.n_category):
            self._datasets[str(c)] = np.concatenate(self._datasets[str(c)], axis = 0)
            #print(self._datasets[str(c)].shape)
            
        _rank = ['th', 'st', 'nd', 'rd'] + ['th'] * 6
        for i in range(len(self.subnets)):
            if i + 1 > 10 and i + 1 < 20: rank = 'th'
            else: rank = _rank[np.mod(i+1, 10)]
            print("\n>>> Pre-training the {}{} subnets...".format(i + 1, rank))
            datasets = self.make_sub_datasets(i)
            self.subnets[i].task = '2cls'
            self.subnets[i].run(datasets = datasets, e = int(pre_epoch/1.5), b = int(pre_batch_size*2), pre_e = int(pre_epoch/4),
                                cpu_core = 0.8, num_workers = 0)
        
    def make_sub_datasets(self, index):
        data0_x = []
        for c in range(self.n_category):
            if c != index:
                data0_x.append(self._datasets[str(c)])
        data0_x = np.concatenate(data0_x, axis = 0)
        data0_y = np.array([[1,0]]).repeat(data0_x.shape[0], axis = 0)
        #print(data0_x.shape, data0_y.shape)
        
        data1_x = self._datasets[str(index)]
        length = data1_x.shape[0]
        while data1_x.shape[0] < data0_x.shape[0]:
            data_x = self._datasets[str(index)]
            np.random.shuffle(data_x)
            if data1_x.shape[0] + length > data0_x.shape[0]:
                data_x = data_x[:data0_x.shape[0] - data1_x.shape[0]]
            #print(data1_x.shape, data_x.shape)
            data1_x = np.concatenate([data1_x, data_x], axis = 0)
        
        data1_y = np.array([[0,1]]).repeat(data1_x.shape[0], axis = 0)
        #print(data1_x.shape, data1_y.shape)
        train_X = np.concatenate([data0_x, data1_x], axis = 0)
        train_Y = np.concatenate([data0_y, data1_y], axis = 0)
        return train_X, train_Y, self.test_X, self.test_Y
        