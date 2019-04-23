# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler,StandardScaler

kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}

class Load(object):
    # preprocess
    def preprocess(self, train, test = None, prep = 'st'):
        reshape = False
        if len(train.shape) > 2:
            reshape = True
            train_size = train.shape
            test_size = test.shape
            train = train.reshape((train_size[0],-1))
            test = test.reshape((test_size[0],-1))
        
        if prep == 'one-hot':
            n_category = len(set(train))
            train = np.eye(n_category)[train]
            if test is not None:
                test = np.eye(n_category)[test]
            return train, test, None
        
        if prep == 'st': # 标准化
            scaler = StandardScaler() 
        if prep == 'mm': # 归一化
            scaler = MinMaxScaler()
        train = scaler.fit_transform(train)
        if test is not None:
            test = scaler.transform(test)
            
        if reshape:
            train = train.reshape(train_size)
            test = test.reshape(test_size)
        return train, test, scaler
    
    # mnist
    def load_mnist(self, path, batch_size, shuffle = True, drop_last = False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        train_set = datasets.MNIST(path, train=True, download=True)
        test_set = datasets.MNIST(path, train = False)
        self.train_X, self.train_Y = train_set.data.numpy().astype(float), train_set.targets.numpy().astype(int)
        self.test_X, self.test_Y = test_set.data.numpy().astype(float), test_set.targets.numpy().astype(int)  
        
        self.get_loader(batch_size, prep = ['mm','one-hot'], shuffle = shuffle, drop_last = drop_last)
    
    # csv, txt, xls, xlsx
    def load_data(self,path,form):
        def load_file(file):
            if form in ['csv','txt']:
                return np.loadtxt(file, dtype = np.float32, delimiter=',')
            elif form in ['xls','xlsx']:
                return pd.read_csv(file,sep=',',header=None).values
        
        def load_file_by_name(file, name1, name2):
            if name1 in file or name1.upper() in file:
                if name2 is None:
                    data = load_file(file)
                    X = data[:,:-1]
                    Y = data[:,-1]
                    return [X,Y]
                elif  name2 in file or name2.upper() in file:
                    return load_file(file)
                
        self.train_X, self.train_Y, self.test_X, self.test_Y =  None, None, None, None
        file_list = os.listdir(path)  #列出文件夹下所有的目录与文件
        for i in range(len(file_list)):
            file = os.path.join(path,file_list[i])
            if os.path.isfile(file):
                if '.'+form in file:
                    train_X = load_file_by_name(file, 'train', 'x')
                    train_Y = load_file_by_name(file, 'train', 'y')
                    if train_X is None:
                        train_X, train_Y = load_file_by_name(file, 'train', None)
                        
                    test_X = load_file_by_name(file, 'test', 'x')
                    test_Y = load_file_by_name(file, 'test', 'y')
                    if test_X is None:
                        test_X, test_Y = load_file_by_name(file, 'test', None)
        
        self.train_X, self.train_Y, self.test_X, self.test_Y = train_X, train_Y, test_X, test_Y

    def get_loader(self, batch_size, prep = None, shuffle = True, drop_last = False):
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        if self.flatten:
            self.train_X = self.train_X.reshape((self.train_X.shape[0],-1))
            self.test_X = self.test_X.reshape((self.test_X.shape[0],-1))
        elif len(self.train_X.shape)<4:
            self.train_X = self.train_X[:,np.newaxis,:,:]
            self.test_X = self.test_X[:,np.newaxis,:,:]
        
        if prep is not None:
            if type(prep) is list:
                self.train_X, self.test_X, _ = self.preprocess(self.train_X, test = self.test_X, prep = prep[0])
                self.train_Y, self.test_Y, self.scaler = self.preprocess(self.train_Y, test = self.test_Y, prep = prep[1])
            else:
                self.train_X, self.test_X, _ = self.preprocess(self.train_X, test = self.test_X, prep = prep)
        
        if self.unsupervised: 
            self.train_Y, self.test_Y = self.train_X, self.test_X
        
        self.train_set = Data.dataset.TensorDataset(torch.from_numpy(self.train_X).float(), 
                                                    torch.from_numpy(self.train_Y).float())
        self.train_loader = Data.DataLoader(self.train_set, batch_size = batch_size, 
                                            shuffle = shuffle, drop_last = drop_last, **kwargs)
        
        self.test_set = Data.dataset.TensorDataset(torch.from_numpy(self.test_X).float(), 
                                                   torch.from_numpy(self.test_Y).float())
        self.test_loader = Data.DataLoader(self.test_set, batch_size = batch_size, 
                                           shuffle = False, drop_last = False, **kwargs)
        
