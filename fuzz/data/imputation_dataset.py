# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.utils.data as Data
import pandas as pd
import os

from fuzz.data.gene_dynamic_data import ReadData
from fuzz.data.read_hy_data import ReadHYData

def loader_kwargs(dvc):
    if dvc == 'cpu': return {'pin_memory': False}
    else: return {'pin_memory': True, 'num_workers': 0}

class ImpuDataSet(Data.Dataset):
    def __init__(self, 
                 impu_path, 
                 real_path, 
                 batch_size,
                 dynamic = 16,
                 missing_rate = 0, 
                 gene_new = False,
                 example = 'TE'):
        
        if example == 'HY': impu_path = '../data/'+ impu_path[impu_path.rfind('/')+1:impu_path.rfind('.')]
        if missing_rate > 0: impu_path+= ' [' +str(missing_rate) +']'
        self.seq_len, self.batch_size = dynamic, batch_size
        self.state = 'training'
        prep = (0.1, 0.9)
        if gene_new:
            # 生成新的数据集 <作为缺失数据集>
            print('Gene new incomplete data set in {}, as X ...'.format(impu_path))
            RD = self.read_data(real_path, prep = prep, missing_rate = missing_rate, 
                                is_del = True, example = example)
        else:
            # 读取 X_path 下数据集 <作为缺失数据集>
            print('Read incomplete data set from {}, as X ...'.format(impu_path))
            RD = self.read_data(impu_path, prep = prep, missing_rate = 0,
                                is_del = False, example = example)
        impu_datasets = RD.datasets
        scaler = RD.scaler_x
        X = impu_datasets[0]
        # 读取 Y_path 下数据集 <作为标签>
        print('\nRead complete data set from {}, as Y ...'.format(real_path))
        comp_datasets = self.read_data(real_path, prep = scaler, missing_rate = 0,
                                       is_del = True, example = example).datasets
        Y = comp_datasets[0]
        self.mean = torch.from_numpy(scaler.nanmean)
        
        # 生成动态样本的序号 d_indexs
        self.gene_dynamic_indexs(X) 
        X = np.concatenate(X, axis = 0)
        Y = np.concatenate(Y, axis = 0)
        print("\ntrain_X's shape is {}, train_Y's shape is {}, d_index's shape is {}\n".format(X.shape, Y.shape, self.d_indexs.shape))
        
        # To tensor
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)
        
        # Get nan loc
        nan = torch.isnan(X).int()
        X[X != X] = 0 
        self.X, self.Y, self.nan = X.float(), Y.float(), nan.float()
        self.len = self.X.size(0)
        
        # Get fill mean
        self.fill_init()
        
        # Count the number of missing
        self.count = torch.sum(nan, 0).data.numpy()
        missing_var = np.where(self.count > 0)
        self.is_missing_var = list(missing_var[0])
        
        self.missing_var_rate = list((self.count / self.X.size(0) * 100)[missing_var])
        self.missing_var_rate.append(np.sum(self.count)/ self.X.size(0) / self.X.size(1))
        self.missing_var_rate = np.array(self.missing_var_rate)
        
        # Shuffle indexs
        np.random.shuffle(self.d_indexs)
    
    def read_data(self, path, prep, missing_rate, is_del, example):
        if example == 'TE':
            return ReadData(path, prep = prep, dynamic = 0,
                            missing_rate = missing_rate,
                            task = 'impu', is_del = is_del,
                            example = example)
        elif example == 'HY':
            return ReadHYData(path, prep = prep, dynamic = 0, 
                              missing_rate = missing_rate,
                              task = 'impu')
    
    def fill_init(self):
        # Get fill mean
        for i in range(self.X.size(0)):
            self.X[i] = self.X[i] * (1 - self.nan[i]) + self.mean * self.nan[i]
        self.dataset = (self.X, self.Y, None, None)

    def gene_dynamic_indexs(self, X):
        start = 0
        del_set = []
        for x in X:
            end = x.shape[0] + start
            del_dynamic_smaples = np.arange(end - self.seq_len + 1, end).reshape(1,-1)
            # print(del_dynamic_smaples - start)
            del_set.append(del_dynamic_smaples)
            start = end 
        
        del_set = np.concatenate(del_set, 1).reshape(-1,)
        # 生成动态样本的乱序 d_indexs
        # 第 i 个动态样本为 [ i : i + seq_len ]，总计 (len - seq_len + 1) 个
        self.d_indexs = np.arange(end - self.seq_len + 1)
        # 但是 list 中两个元素 不是 连续的，中间不存在动态样本，需删除对应的序号
        self.d_indexs = np.setdiff1d(self.d_indexs, del_set)        
        
    def __getitem__(self, batch_idx):
        '''
            origin indexs:
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9 | d = 4, l-s+1 = 7
            self.d_indexs:
                0, 1, 2, 3, 4, 5, 6 | b = 3
            start, end:
                0-3, 3-6, 6
        '''
        # 将动态样本的 d_indexs 分批次提取
        # 单批次 batch 个样本的序号为 batch_indexs
        start, end = batch_idx * self.batch_size, min((batch_idx + 1) * self.batch_size, self.d_indexs.shape[0])
        if start >= self.d_indexs.shape[0]:
            raise StopIteration
        # 在开头 shuffle，确保只在训练的时候改变 d_index 的顺序
        if start == 0 and self.state == 'training':
            np.random.shuffle(self.d_indexs)
                
        self.batch_indexs = self.d_indexs[start:end]
        return self.__get_dynamic_data__()
    
    def __get_dynamic_data__(self):
        # 从原数据集中生成动态数据集
        X, NAN = [], []
        for index in self.batch_indexs:
            start, end = index, index + self.seq_len
            X.append(self.X[start:end].reshape((1,-1)))
            NAN.append(self.nan[start:end].reshape((1,-1)))
        return torch.cat(X, 0), torch.cat(NAN, 0)
    
    def __impu_data__(self, impu_batch):
        # 用重构的动态样本替换原来的样本
        for i, index in enumerate(self.batch_indexs):
            start, end = index, index + self.seq_len
            loc = self.nan[start:end]
            modify = impu_batch[i].view(self.seq_len, -1)
            self.X[start:end] = self.X[start:end] * (1 - loc) + modify * loc
    
    def __impu_data_set__(self, impu_set):
        # 用重构的动态样本集替换原来的数据集
        for i in range(self.len - self.seq_len + 1):
            index = self.d_indexs[i]
            start, end = index, index + self.seq_len
            loc = self.nan[start:end]
            modify = impu_set[i].view(self.seq_len, -1)
            self.X[start:end] = self.X[start:end] * (1 - loc) + modify * loc
            # if np.mod(i, 1000): print(i)
    
    # batch 数目
    def __len__(self):
        return int(np.ceil(self.d_indexs.shape[0]/self.batch_size))
    
    def save_best_impu_result(self, model_name):
        path = '../save/'+ model_name +'/'
        X = self.X.clone().data.numpy()
        X[self.nan == 0] = float('nan')
        df = pd.DataFrame(X)
        df.to_csv(path + model_name + '-impu.csv', header = None, index = None) 
        
        Y = self.Y.clone().data.numpy()
        Y[self.nan == 0] = float('nan')
        df = pd.DataFrame(Y)
        df.to_csv(path + 'real.csv', header = None, index = None) 
    
    def example(self):
        nan = float('nan')
        aa = np.array([[1,    nan,    2,    5    ],
                       [nan,  20,     2,    3    ],
                       [nan,  20,     nan,  2    ],
                       [3,    20,     1,    nan  ]])
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        scaler = MinMaxScaler()
        scaler.fit(aa)
        print(scaler.transform(aa))
        scaler = StandardScaler()
        scaler.fit(aa)
        print(scaler.var_)
        print(scaler.transform(aa))
        print(np.sum(aa, axis = 0))
        print(np.nansum(aa, axis = 0))
        aa[ aa != aa ] = 0
        print(np.sum(aa, axis = 0))

if __name__ == '__main__':
    dynamic, batch_size, epoch = 40, 4, 3
    missing_rate = 0

    datasets = ImpuDataSet('../data/Impu', '../data/TE', batch_size, 
                           dynamic = dynamic, missing_rate = missing_rate, export = '1d')
    datasets.example()
    data_loader = datasets.get_loader()
    
    for e in range(epoch):
        print(">>> Epoch: " + str(e+1))
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx == 0:
                index, x, nan = data
                y = target
                print(index)
                print(x.size())
                print(y.size())
                print(torch.sum((y-x)*(1-nan)))
