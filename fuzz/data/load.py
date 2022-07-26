# -*- coding: utf-8 -*-

import torch
import numpy as np
from torchvision import datasets as mnist
import torch.utils.data as Data

from .gene_dynamic_data import ReadData, preprocess

def _flatten(X):
    return X.reshape((X.shape[0],-1))

def _loader_kwargs(dvc):
    if dvc == 'cpu':
        return {'pin_memory': False}
    else:
        return {'pin_memory': True, 'num_workers': 0}

class Load(object):
    
    # mnist
    def load_mnist(self, path, batch_size):
        print('\nLoading mnist...')
        self.batch_size = batch_size
        
        train_set = mnist.MNIST(path, train=True, download=True)
        test_set = mnist.MNIST(path, train = False)
        self.train_X, self.train_Y = train_set.data.numpy().astype(float), train_set.targets.numpy().astype(int)
        self.test_X, self.test_Y = test_set.data.numpy().astype(float), test_set.targets.numpy().astype(int)  
        if self.flatten:
            self.train_X, self.test_X = _flatten(self.train_X), _flatten(self.test_X)
        self.train_X, self.test_X, self.scaler_x = preprocess( self.train_X, self.test_X, 'mm')
        self.train_Y, self.test_Y, self.scaler_y = preprocess( self.train_Y, self.test_Y, 'oh')
        print('->  train_X{},  train_Y{}\n->  test_X{},  test_Y{}'.\
              format(self.train_X.shape, self.train_Y.shape, self.test_X.shape, self.test_Y.shape))
        self.get_loader(batch_size)
    
    # csv, txt, xls, xlsx
    def load_data(self, path, batch_size, **kwargs):
        self.batch_size = batch_size
        if self.task == 'impu': return
        self.train_X, self.train_Y, self.test_X, self.test_Y =  None, None, None, None
        if type(path) != str:
            # path 是一个 数据集
            self.train_X, self.train_Y, self.test_X, self.test_Y = path
        else:
            self.Data = ReadData(path, **kwargs)
            self.train_X, self.train_Y, self.test_X, self.test_Y = self.Data.dataset
        self.datasets = (self.train_X, self.train_Y, self.test_X, self.test_Y)
        self.get_loader(batch_size) 

    def get_loader(self, batch_size):
        # 扁平化
        if self.flatten:
            if len(self.train_X.shape)>2:
                self.train_X, self.test_X = _flatten(self.train_X), _flatten(self.test_X)
        # 转成规定图片大小
        elif hasattr(self, 'img_size'):
            if len(self.train_X.shape)<4:
                img_size = self.img_size.copy() 
                # [-1, channel, H, W]
                img_size = [-1] + img_size
                self.train_X = self.train_X.reshape(img_size)
                self.test_X = self.test_X.reshape(img_size)
        
        # 只需要正常样本用来训练
        if self.task == 'fd':
            normal_indexs = np.argwhere(self.test_Y.argmax(axis = 1) == 0).reshape(-1,)
            test_Y_n = self.test_Y[normal_indexs]
            print('\nNumber of train data:')
            print('->  Normal{}'.format(self.train_X.shape))
            print('Number of test samples:')
            print('->  Normal({}, {}),  faulty({}, {})'.format(test_Y_n.shape[0], self.test_X.shape[1],\
                  self.test_Y.shape[0] - test_Y_n.shape[0], self.test_X.shape[1]))
          
        self.train_set = Data.dataset.TensorDataset(torch.from_numpy(self.train_X).float(), 
                                                    torch.from_numpy(self.train_Y).float())
        self.train_loader = Data.DataLoader(self.train_set, batch_size = batch_size, 
                                            shuffle = True, drop_last = False, **_loader_kwargs(self.dvc))
        
        self.test_set = Data.dataset.TensorDataset(torch.from_numpy(self.test_X).float(), 
                                                   torch.from_numpy(self.test_Y).float())
        self.test_loader = Data.DataLoader(self.test_set, batch_size = batch_size, 
                                           shuffle = False, drop_last = False, **_loader_kwargs(self.dvc))
        
        # 获取原始数据集
        # dataset = dataloader.dataset
        # X, Y =  dataset.tensors[0], dataset.tensors[1]