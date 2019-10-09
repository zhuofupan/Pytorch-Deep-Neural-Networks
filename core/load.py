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
    def to_onehot(self, n, data):
        data = data.astype(np.int)
        data = data - data.min()
        return np.eye(n)[data]
    
    def preprocess(self, train, test = None, prep = 'st'):
        if prep is None: 
            return train, test, None
        
        reshape = False
        if len(train.shape) > 2:
            reshape = True
            train_size = train.shape
            test_size = test.shape
            train = train.reshape((train_size[0],-1))
            test = test.reshape((test_size[0],-1))
        
        if prep == 'onehot':
            n_category = len(set(train)) # 无序的不重复元素序列
            train = self.to_onehot(n_category, train)
            if test is not None:
                test = self.to_onehot(n_category, test)
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
    def load_mnist(self, path, batch_size, shuffle = True):
        self.batch_size = batch_size
        
        train_set = datasets.MNIST(path, train=True, download=True)
        test_set = datasets.MNIST(path, train = False)
        self.train_X, self.train_Y = train_set.data.numpy().astype(float), train_set.targets.numpy().astype(int)
        self.test_X, self.test_Y = test_set.data.numpy().astype(float), test_set.targets.numpy().astype(int)  
        
        self.get_loader(batch_size, prep = ['mm','onehot'], shuffle = shuffle)
    
    # csv, txt, xls, xlsx
    def load_data(self, path, batch_size, prep = None, shuffle = True):
        
        def load_file(file_name, file_path):
            print("Load data from '" + file_path + "'")
            suffix = file_name.split('.')[-1]
            if suffix in ['csv','txt']:
                return np.loadtxt(file_path, dtype = np.float32, delimiter=',')
            elif suffix in ['xls','xlsx']:
                return pd.read_csv(file_path,sep=',',header=None).values
            return None
        
        def load_file_by_name(file_name, file_path):
            name1 = ['train', 'test', 'Train', 'Test']
            name2 = ['x', 'X', 'y', 'Y']
            id1, id2 = None, None
            for i in range(4): 
                if name1[i] in file_name: 
                    id1 = name1[i].lower()
                if name2[i] in file_name:
                    id2 = name2[i].lower()
            if id1 is not None:
                if id2 is None:
                    data = load_file(file_name, file_path)
                    X = data[:,:-1]
                    Y = data[:,-1]
                    if id1 == 'train': return [ [0, X], [1, Y] ]
                    else: return [ [2, X], [3, Y] ]
                else:
                    D = load_file(file_name, file_path)
                    if id1 == 'train' and id2 == 'x': return [ [0, D] ]
                    if id1 == 'train' and id2 == 'y': return [ [1, D] ]
                    if id1 == 'test' and id2 == 'x': return [ [2, D] ]
                    if id1 == 'test' and id2 == 'y': return [ [3, D] ]
                
        self.train_X, self.train_Y, self.test_X, self.test_Y =  None, None, None, None
        if type(path) != str:
            # path 是一个 数据集
            self.train_X, self.train_Y, self.test_X, self.test_Y = path
        else:
            # 从文件夹 path 中读取数据文件
            dataset = [ None, None, None, None ]
            file_list = os.listdir(path)  #列出文件夹下所有的目录与文件
            for i in range(len(file_list)):
                file_name = file_list[i]
                file_path = path+'/'+file_list[i]
                if os.path.isfile(file_path):
                    _data = load_file_by_name(file_name, file_path)
                    for _d in _data:
                        dataset[ _d[0] ] = _d[1]
            
            self.train_X, self.train_Y, self.test_X, self.test_Y = dataset
        self.get_loader(batch_size, prep = prep, shuffle = shuffle) 

    def get_loader(self, batch_size, prep = None, shuffle = True):
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
                                            shuffle = shuffle, drop_last = False, **kwargs)
        
        self.test_set = Data.dataset.TensorDataset(torch.from_numpy(self.test_X).float(), 
                                                   torch.from_numpy(self.test_Y).float())
        self.test_loader = Data.DataLoader(self.test_set, batch_size = batch_size, 
                                           shuffle = False, drop_last = False, **kwargs)
        
