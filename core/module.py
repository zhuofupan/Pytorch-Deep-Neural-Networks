# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch import optim
from core.load import Load
from core.func import Func
from core.epoch import Epoch
from pandas import DataFrame

class Module(torch.nn.Module,Load,Func,Epoch):
    
    def default_setting(self):
        # default setting
        self.flatten = False
        self.unsupervised = False
        self.L = torch.nn.MSELoss()
        self.msg = []
    
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.default_setting()
        for key in kwargs.keys(): setattr(self, key, kwargs[key])
        
        if self.task == 'cls':
            head = ['loss', 'accuracy']
        elif self.task == 'prd':
            head = ['loss', 'rmse', 'R2']
        self.train_df = DataFrame(columns = head)
        self.test_df = DataFrame(columns = head)
        
    def __call__(self, **kwargs):
        return self.forward(**kwargs)
    
    def opt(self, lr = 1e-3):
        self.optimizer  = optim.Adam(self.parameters(), lr=lr)
    
    def Sequential(self, struct = None, is_drop = True):
        if struct is None:
            struct = self.struct
            
        self.feature = nn.Sequential() 
        for i in range(len(struct)-2):
            if is_drop and self.dropout > 0:
                self.feature.add_module('Dropout'+str(i),nn.Dropout(p = self.dropout))
            self.feature.add_module('Add_In'+str(i),nn.Linear(struct[i], struct[i+1]))
            self.feature.add_module('Activation'+str(i),self.F(i))
        
        self.output = nn.Sequential(nn.Linear(struct[-2],struct[-1]),
                                    self.F(self.output_func))
        
    def Convolutional(self, conv_struct = None, is_drop = True):
        in_channel = self.img_size[0]
        if conv_struct is None:
            conv_struct = self.conv_struct
            
        self.conv = nn.Sequential()
        for i in range(conv_struct.shape[0]):
            row = conv_struct.loc[i].values
            if is_drop and self.dropout > 0:
                self.conv.add_module('Dropout'+str(i),nn.Dropout(p = self.dropout))
            ''' 
                head = ['out_channel', 'conv_kernel_size', 'is_bn', 'pool_kernel_size'] 
            '''
            self.conv.add_module('Conv2d'+str(i),nn.Conv2d(in_channel, row[0], row[1]))
            if row[2] == 1:
                self.conv.add_module('BatchNorm2d'+str(i),nn.BatchNorm2d(row[0]))
            self.conv.add_module('Activation'+str(i),self.F(i,func = self.conv_func))
            if type(row[3]) is tuple or row[3] > 0:
                self.conv.add_module('MaxPool2d'+str(i),nn.MaxPool2d(row[3]))
            in_channel = row[0]
            