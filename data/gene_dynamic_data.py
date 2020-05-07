# -*- coding: utf-8 -*-
"""Functions for downloading and reading TE data."""
import os
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# preprocess
def to_onehot(n, data):
    data = data.astype(np.int)
    return np.eye(n)[data]

def preprocess(train, test = None, prep = 'st', feature_range=(0, 1)):
    
    def trans_data(_data):
        _data = np.array(_data)
        # shape
        reshape = False
        if len(_data.shape) > 2:
            reshape = True
            raw_size = _data.shape
            _data = _data.reshape((raw_size[0],-1))
        
        if prep == 'oh':
            _data = to_onehot(n_category, _data)
        else:
            _data = scaler.transform(_data)
        
        if reshape:
            _data = _data.reshape(raw_size)
        return _data

    if prep is None:
        return train, test, None
        
    if type(train) == list:
        fit = np.concatenate(train, axis = 0)
    else:
        fit = train
    
    scaler = None
    if prep == 'oh':
        if test is not None:
            if type(test) == list: trans = np.concatenate(test, aixs = 0)
            else: trans = test
            fit = np.concatenate([fit, trans], axis = 0)
        labels = list(set(fit))
        n_category = len(labels)
    elif prep == 'st': # 标准化
        scaler = StandardScaler() 
        scaler.fit(fit)
    elif prep == 'mm': # 归一化
        scaler = MinMaxScaler(feature_range)
        scaler.fit(fit)
    
    if type(train) == list:
        for i in range(len(train)):
            train[i] = trans_data(train[i])
    else:
        train = trans_data(train)
    if test is not None:
        if type(test) == list:
            for i in range(len(test)):
                test[i] = trans_data(test[i])
        else:
            test = trans_data(test)

    return train, test, scaler

def load_file(file_name, file_path):
    print("Load data from '" + file_path + "'")
    suffix = file_name.split('.')[-1]
    name = file_name[:int(-1*len(suffix))]
    if suffix in ['csv','txt']:
        return {name: np.loadtxt(file_path, dtype = np.float32, delimiter=',')}
    elif suffix in ['dat']:
        return {name: np.loadtxt(file_path, dtype = np.float32)}
    elif suffix in ['xls','xlsx']:
        return {name: pd.read_csv(file_path,sep=',',header=None).values}
    elif suffix == 'mat':
        data_dic = scio.loadmat(file_path)
        data_dic.pop('__header__')
        data_dic.pop('__version__')
        data_dic.pop('__globals__')
        return data_dic
        
class ReadData():
    def __init__(self, path, prep = None, dynamic = 0, stride = 1, 
                 set_normal = -1, set_for = [0,1], cut_mode = 'continue', example = ''):
        
        self.train_X, self.train_Y, self.test_X, self.test_Y, self.scaler = None, None, None, None, None
        
        # 读取原始数据
        if example == 'TE':
            self.laod_data(path, [1,3])
            self.del_data([22,41], ['03','09','15'])
            self.get_category_lables(160, [1])
        elif example == 'CSTR':
            self.laod_data(path, [7,0])
            self.get_category_lables(200, [0,1])
        else:
            self.laod_data(path)
            self.get_category_lables(set_normal, set_for)
            
        # X 预处理
        prep_y = None
        if prep is not None:
            prep_x = prep
            if type(prep) == list:
                prep_x, prep_y = prep[0], prep[1]
            self.train_X, self.test_X, self.scaler_x = preprocess(self.train_X, self.test_X, prep_x)
        
        # 生成动态数据
        if dynamic > 0:
            self.gene_dymanic_data(dynamic, stride, cut_mode)
        
        # 合并数据集
        if self.train_X is not None and type(self.train_X) == list:
            self.train_X = np.concatenate(self.train_X, axis = 0)
            self.train_Y = np.concatenate(self.train_Y, axis = 0)
        if self.test_X is not None and type(self.test_X) == list:
            self.test_X = np.concatenate(self.test_X, axis = 0)
            self.test_Y = np.concatenate(self.test_Y, axis = 0)
        
        # Y预处理
        if prep_y is not None:
            self.train_Y, self.test_Y, self.scaler_y = preprocess(self.train_Y, self.test_Y, prep_y)
        
        self.dataset = self.train_X, self.train_Y, self.test_X, self.test_Y
        print('Gene dataset with shape:\n->  train_X{},  train_Y{}\n->  test_X{},  test_Y{}'.\
              format(self.train_X.shape, self.train_Y.shape, self.test_X.shape, self.test_Y.shape))
            
    def laod_data(self, path, intercept = None):
        
        for _tp in ['train','test']:
            data_dic = {}
            X, Y = [], []
            
            # 读取数据存入字典
            if os.path.exists(path+'/'+_tp):
                file_list = os.listdir(path+'/'+_tp)  #列出文件夹下所有的目录与文件
                for i in range(len(file_list)):
                    file_name = file_list[i]
                    file_path = path+'/'+_tp+'/'+file_list[i]
                    if os.path.isfile(file_path):
                        D = load_file(file_name, file_path)
                        data_dic.update(D)
            
            # 找出 X 和 Y
            for key, data in data_dic.items():
                
                label = key
                if intercept is not None:
                    if intercept[1] == 0: label = label[intercept[0]:]
                    else: label = label[intercept[0]:intercept[1]]
                    
                if '_x' in key or '_X' in key :
                    X.append(data)
                    key = key.replace('_x', '_y')
                    key = key.replace('_X', '_Y')
                    if key in data_dic.keys():
                        label = data_dic[key]
                    else:
                        label = [label]*data.shape[0]
                    Y.append(np.array(label))
                elif '_y' in key or '_Y' in key:
                    continue
                else:
                    X.append(data)
                    label = [label]*data.shape[0]
                    Y.append(np.array(label))
            
            if _tp == 'train':
                self.train_X, self.train_Y = X, Y
            else:
                self.test_X, self.test_Y = X, Y
            
    def del_data(self, del_dim = None, del_lbs = None):
        
        # 删除维度
        for X in [self.train_X, self.test_X]:
            if X is None: continue
            for i in range(len(X)):
                X[i] = np.delete(X[i],range(del_dim[0],del_dim[1]),axis=1)
        
        # 删除类别
        for index, Y in enumerate([self.train_Y, self.test_Y]):
            if Y is None: continue
            i = 0 
            while i < len(Y):
                lb = Y[i][0]
                if lb in del_lbs:
                    if index == 0:
                        self.train_X.pop(i)
                        self.train_Y.pop(i)
                    else:
                        self.test_X.pop(i)
                        self.test_Y.pop(i)
                else:
                    i += 1
    
    def get_category_lables(self, set_normal = -1, set_for = [0,1]):
        lbs = []
        self.set_normal = set_normal
        self.set_for = set_for
        # 获取类别
        for index, Y in enumerate([self.train_Y, self.test_Y]):
            if Y is None: continue
            labels = list(set(np.concatenate(Y, axis = 0)))
            if index == 0:
                lbs = labels
            else:
                lbs += labels
                lbs = list(set(lbs))
                lbs.sort()
                for s in ['normal', 'Normal']:
                    if s in lbs:
                        lbs.remove(s)
                        lbs.insert(0, s)
                self.labels = lbs
    
        # 设置数字类别
        for index, Y in enumerate([self.train_Y, self.test_Y]):
            if Y is None: continue
            for i in range(len(Y)):
                for k in range(Y[i].shape[0]):
                    if index in set_for and k < set_normal:
                        Y[i][k] = 0
                    else:
                        Y[i][k] = self.labels.index(Y[i][k])
                        
    def gene_dymanic_data(self, dynamic, stride, cut_mode = 'continue', cut_for = 'all'):
        
        def get_dymanic_x(_x, _y):
            _dx, _dy = [], []
            for r in range(_x.shape[0]):
                if r+1>=dynamic:
                    start=r+1-dynamic
                    end=r+1
                    _dx.append(_x[start:end].reshape(1,-1))
                    _dy += [_y[r]]
                    r=r+stride
            _dx = np.array(np.concatenate(_dx,axis=0), dtype=np.float32)
            _dy = np.array(_dy, dtype=np.float32)
            return _dx, _dy
        
        for index, (X,Y) in enumerate([(self.train_X, self.train_Y), (self.test_X, self.test_Y)]):
            if X is not None:
                for i in range(len(X)):
                    x,y = X[i],Y[i]
                    if index in self.set_for and self.set_normal > -1 and i > 0:
                        if cut_mode == 'continue':
                            x1, y1 = get_dymanic_x(x[:self.set_normal], y[:self.set_normal])
                            x2, y2 = get_dymanic_x(x[self.set_normal - dynamic + 2:], y[self.set_normal - dynamic + 2:])
                        else:
                            x1, y1 = get_dymanic_x(x[:self.set_normal], y[:self.set_normal])
                            x2, y2 = get_dymanic_x(x[self.set_normal:], y[self.set_normal:])
                        X[i] = np.concatenate([x1, x2], axis = 0)
                        Y[i] = np.concatenate([y1, y2], axis = 0)
                    else:
                        X[i],Y[i] = get_dymanic_x( x, y )

if __name__ == "__main__": 
    X1, Y1, X2, Y2 = ReadData('../data/TE', ['st', 'oh'], 40, cut_mode = '', example = 'TE').dataset
#    X1, Y1, X2, Y2 = ReadData('../data/CSTR', ['st', 'oh'], 40, example = 'CSTR').dataset
