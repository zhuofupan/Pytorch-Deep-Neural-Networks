# -*- coding: utf-8 -*-

import torch
from torchvision import datasets, transforms
import torch.utils.data as Data

import sys
sys.path.append('..')
from data.gene_dynamic_data import ReadData, preprocess

class Load(object):
    
    # mnist
    def load_mnist(self, path, batch_size):
        self.batch_size = batch_size
        
        train_set = datasets.MNIST(path, train=True, download=True)
        test_set = datasets.MNIST(path, train = False)
        self.train_X, self.train_Y = train_set.data.numpy().astype(float), train_set.targets.numpy().astype(int)
        self.test_X, self.test_Y = test_set.data.numpy().astype(float), test_set.targets.numpy().astype(int)  
        self.train_X, self.test_X, self.scaler_x = preprocess( self.train_X, self.test_X, 'mm')
        self.train_Y, self.test_Y, self.scaler_y = preprocess( self.train_Y, self.test_Y, 'oh')
        
        self.get_loader(batch_size)
    
    # csv, txt, xls, xlsx
    def load_data(self, path, batch_size, **kwargs):
      
        self.train_X, self.train_Y, self.test_X, self.test_Y =  None, None, None, None
        if type(path) != str:
            # path 是一个 数据集
            self.train_X, self.train_Y, self.test_X, self.test_Y = path
        else:
            self.Data = ReadData(path, **kwargs)
            self.train_X, self.train_Y, self.test_X, self.test_Y = self.Data.dataset
        self.get_loader(batch_size) 

    def get_loader(self, batch_size):
        
        self.batch_size = batch_size
        
        # 扁平化
        if self.flatten:
            if len(self.train_X.shape)>2:
                self.train_X = self.train_X.reshape((self.train_X.shape[0],-1))
                self.test_X = self.test_X.reshape((self.test_X.shape[0],-1))
        # 转成规定图片大小
        elif hasattr(self, 'img_size'):
            if len(self.train_X.shape)<4:
                img_size = self.img_size.copy() 
                # [-1, channel, H, W]
                img_size = [-1] + img_size
                self.train_X = self.train_X.reshape(img_size)
                self.test_X = self.test_X.reshape(img_size)
        
        if self.unsupervised: 
            self.train_Y, self.test_Y = self.train_X, self.test_X
        
        kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
        
        self.train_set = Data.dataset.TensorDataset(torch.from_numpy(self.train_X).float(), 
                                                    torch.from_numpy(self.train_Y).float())
        self.train_loader = Data.DataLoader(self.train_set, batch_size = batch_size, 
                                            shuffle = True, drop_last = False, **kwargs)
        
        self.test_set = Data.dataset.TensorDataset(torch.from_numpy(self.test_X).float(), 
                                                   torch.from_numpy(self.test_Y).float())
        self.test_loader = Data.DataLoader(self.test_set, batch_size = batch_size, 
                                           shuffle = False, drop_last = False, **kwargs)
        
