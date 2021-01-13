# -*- coding: utf-8 -*-
"""Functions for downloading and reading TE data."""
import os
import numpy as np
import pandas as pd
import scipy.io as scio
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# one-hot (*不适用于y中包含 np.nan)
def to_onehot(n, data):
    data = data.astype(np.int)
    return np.eye(n)[data]

# preprocess (train, test 可以是 list 或 np.array)
def preprocess(train, test = None, prep = 'st', feature_range=(0, 1)):
    
    def _transform(_data):
        _data = np.array(_data)
        # 大于2维的数据要先转换成 2维
        reshape = False
        if len(_data.shape) > 2:
            reshape = True
            raw_size = _data.shape
            _data = _data.reshape((raw_size[0],-1))
        
        # one-hot
        if prep == 'oh':
            _data = to_onehot(n_category, _data)
        else:
            _data = scaler.transform(_data)
            
        # 转回原来的尺寸
        if reshape:
            _data = _data.reshape(raw_size)
        return _data
    
    scaler = None
    if prep is None:
        return train, test, None
    elif type(prep) == int:
        feature_range = (-1*prep, prep)
        prep = 'mm'
    elif type(prep) == tuple:
        feature_range = (prep[0], prep[1])
        prep = 'mm'
    
    _train = train
    if type(train) == list:
        _train = np.concatenate(train, axis = 0)
    
    # fit
    if prep == 'oh':
        labels = list(set(_train))
        n_category = len(labels)
    elif prep == 'st': # 标准化
        scaler = StandardScaler() 
        scaler.fit(_train)
        scaler.name = 'st'
    elif prep == 'mm': # 归一化
        scaler = MinMaxScaler(feature_range)
        scaler.fit(_train)
        scaler.name = 'mm'
    else:
        scaler = prep
    
    # record
    if scaler is not None:
        _train = scaler.transform(_train)
    if prep == 'st':
        scaler.nanmean, scaler.nanvar = \
            np.array([0.0]*_train.shape[1]), np.array([1.0]*_train.shape[1])
        scaler.nanmin, scaler.nanmax = \
            np.nanmin(_train, axis = 0), np.nanmax(_train, axis = 0)
    if prep == 'mm':
        scaler.nanmean, scaler.nanvar = \
            np.nanmean(_train, axis = 0), np.nanvar(_train, axis = 0)
        scaler.nanmin, scaler.nanmax = \
            np.array([feature_range[0]*1.0]*_train.shape[1]), \
            np.array([feature_range[1]*1.0]*_train.shape[1])
    
    # transform
    if type(train) == list:
        for i in range(len(train)):
            train[i] = _transform(train[i])
    else:
        train = _transform(train)
        
    if test is not None:
        if type(test) == list:
            for i in range(len(test)):
                test[i] = _transform(test[i])
        else:
            test = _transform(test)

    return train, test, scaler

class Scaler():    
    def save(self, scaler, path, name = '', tp = 'x'):
        if scaler is None: return
        if hasattr(scaler, 'name'): name = scaler.name
        self.tp = tp
        self.name = name
        if not os.path.exists(path + '/scaler'): os.makedirs(path + '/scaler')
        if scaler.name == 'st':
            np.savetxt(path + '/scaler/[{}] scaler_{}.csv'.format(scaler.name, tp), 
                       np.concatenate([scaler.mean_.reshape(-1,1), scaler.var_.reshape(-1,1)], 1) , 
                       fmt='%f', delimiter=',')
        elif scaler.name == 'mm':
            np.savetxt(path + '/scaler/[{}] scaler_{}.csv'.format(scaler.name, tp), 
                       np.concatenate([scaler.scale_.reshape(-1,1), scaler.min_.reshape(-1,1)], 1) , 
                       fmt='%f', delimiter=',')
    
    def read(self, path, name, tp = 'x'):
        data = np.loadtxt(path + '/scaler/[{}] scaler_{}.csv'.format(name, tp), delimiter = ',')
        if name == 'st':
            self.mean_ = data[:,0]
            self.var_ = data[:,1]
        elif name == 'mm':
            self.scale_ = data[:,0]
            self.min_ = data[:,1]
        self.tp = tp
        self.name = name
    
    def transform(self, X):
        if self.name == 'st':
            return (X - self.mean_) / np.sqrt(self.var_)
        elif self.name == 'mm':
            return X * self.scale_ + self.min_
        
    def inverse_transform(self, X):
        if self.name == 'st':
            return X * np.sqrt(self.var_) + self.mean_
        elif self.name == 'mm':
            return (X - self.min_) / self.scale_
        
# load data
def load_file(file_name, file_path):
    # return {name: key}
    suffix = file_name.split('.')[-1]       # 文件格式
    name = file_name[:int(-1*len(suffix))-1]  # 去掉文件格式的文件名
    if suffix in ['csv','txt']:
        # 无表头时 加上 条件 header = None
        return {name: pd.read_csv(file_path, header = None).values}
    elif suffix in ['dat']:
        return {name: np.loadtxt(file_path)}
    elif suffix in ['xls','xlsx']:
        return {name: pd.read_excel(file_path).values}
    elif suffix == 'mat':
        data_dic = scio.loadmat(file_path)
        data_dic.pop('__header__')
        data_dic.pop('__version__')
        data_dic.pop('__globals__')
        return data_dic
     
class ReadData():
    def __init__(self, 
                 path,                  # 路径
                 prep = None,           # 预处理
                 dynamic = 0,           # 滑窗长度 (= seq_len)
                 stride = 1,            # 滑动步长 
                 bias = 0,              # x(t0 ~ td - 1) 对应 y(td - 1 + bias)
                 task = 'cls',          # 由于 'cls'/'prd'/'impu' 任务
                 export = '1d',         # 导出X的数据类型 '1d' / 'seq'
                 intercept = None,      # 将截取 [start, end] 的文件名作为标签名
                 is_del = True,         # 是否执行删除函数
                 impu_rate = 0,         # 将数据按概率填充为 nan
                 div_prop = None,       # 分割比例>0时，将后 1 - div_prop 作为测试集
                 div_shuffle = False,   # 分割前是否打乱
                 set_normal = -1,       # 用于故障诊断，将前 set_normal 个样本的标签设为 0
                 set_for = [0,1],       # set_normal 是否应用于 [训练集, 测试集]
                 cut_mode = 'seg',      # 滑窗到类别分界点时，continue滑窗 or seg 滑窗
                 example = '',          # 特定内设数据集
                 save_data = False      # 是否存储制作好的动态数据集
                 ):

        self.train_X, self.train_Y, self.test_X, self.test_Y, self.scaler = None, None, None, None, None
        self.dynamic, self.stride, self.bias = dynamic, stride, bias
        self.task = task
        self.impu_rate = impu_rate
        if type(prep) == list:
            prep_x, prep_y = prep[0], prep[1]
        else:
            prep_x, prep_y = prep, None
        
        # 读取原始数据
        if example == 'TE':
            self.laod_data(path, intercept = [1,3])
            if is_del: self.del_data(del_dim = [22,41], del_lbs = ['03','09','15'])
            set_normal, set_for = 160, [1]
            self.get_category_lables(set_normal = set_normal, set_for = set_for)
        elif example == 'CSTR':
            self.laod_data(path, intercept = [7,0])
            set_normal, set_for = 200, [0,1]
            self.get_category_lables(set_normal = set_normal, set_for = set_for)
        else:
            self.laod_data(path, intercept)
            if task == 'cls':
                self.get_category_lables(set_normal, set_for)
        
        if impu_rate > 0:
            self.drop_as_nan(impu_rate, industrial = True)
        # if task == 'impu': self.only_need_train_dataset()
        
        # 对 X 预处理
        if prep_x is not None and div_prop is None:
            self.train_X, self.test_X, self.scaler_x = preprocess(self.train_X, self.test_X, prep_x)
            Scaler().save(self.scaler_x, path, '', 'x')
        
        # 生成动态数据
        if dynamic > 1:
            self.gene_dymanic_data(dynamic, stride, export, set_normal, set_for, cut_mode)
        
        # 分割数据集
        if div_prop is not None:
            self.div_dataset(div_prop, div_shuffle)
            self.train_X, self.test_X, self.scaler_x = preprocess(self.train_X, self.test_X, prep_x)
            Scaler().save(self.scaler_x, path, '', 'x')
        
        # 合并 list 数据集
        if self.train_X is not None and type(self.train_X) == list:
            self.train_X = np.concatenate(self.train_X, axis = 0)
            self.train_Y = np.concatenate(self.train_Y, axis = 0)
        if self.test_X is not None and type(self.test_X) == list:
            self.test_X = np.concatenate(self.test_X, axis = 0)
            self.test_Y = np.concatenate(self.test_Y, axis = 0)
            
        # 对 Y 预处理
        if prep_y is not None:
            self.train_Y, self.test_Y, self.scaler_y = preprocess(self.train_Y, self.test_Y, prep_y)
            Scaler().save(self.scaler_y, path, '', 'y')
            
        # 保存制作好的动态数据集
        if save_data:
            if not os.path.exists(path + '/gene'): os.makedirs(path + '/gene')
            np.savetxt(path + '/gene/trian_X.csv', self.trian_X, '%f', ',')
            np.savetxt(path + '/gene/trian_Y.csv', self.trian_Y, '%f', ',')
            np.savetxt(path + '/gene/test_X.csv', self.test_X, '%f', ',')
            np.savetxt(path + '/gene/test_Y.csv', self.test_Y, '%f', ',')
        
        self.datasets = self.train_X, self.train_Y, self.test_X, self.test_Y
        shapes = []
        for data_set in [self.train_X, self.train_Y, self.test_X, self.test_Y]:
            if data_set is not None: shapes.append(data_set.shape)
            else: shapes.append(' None')
        if task != 'impu': print('Gene datasets with shape:')
        print('->  train_X{},  train_Y{}\n->  test_X{},  test_Y{}'.\
              format(shapes[0], shapes[1], shapes[2], shapes[3]))
    
    # 得到数据为 list， Y 可能为 str list
    def laod_data(self, 
                  path,                 # 路径
                  intercept = None      # 截取文件名 [start, end] 作为标签名
                  ):
        
        # 将数据放入 train 与 test 文件夹中
        for _tp in ['train','test']:
            data_dic = {}
            X_list, Y_list = [], []
            
            # 读取数据存入字典
            file_paths = []
            if os.path.exists(path+'/'+_tp):
                file_list = os.listdir(path+'/'+_tp)  #列出文件夹下所有的目录与文件
                for i in range(len(file_list)):
                    file_name = file_list[i]
                    file_path = path+'/'+_tp+'/'+file_list[i]
                    if os.path.isfile(file_path):
                        D = load_file(file_name, file_path)
                        file_paths.append(file_path)
                        data_dic.update(D)
            
            for key, data in data_dic.items():
                # 1、 文件名以 _x, _X 结尾
                if '_x' in key or '_X' in key:
                    X = data
                    y_key = key.replace('_x', '_y')
                    y_key = key.replace('_X', '_Y')
                    # 在字典中找到对应同名 key 且以 _y 或 _Y 结尾
                    if y_key in data_dic.keys():
                        label = data_dic[y_key]
                        Y = np.array(label)
                    else:
                        label = [key[:-2]]*data.shape[0]
                        Y = np.array(label)
                # 2、 文件名以 _D 结尾，用 X 中的后 col 列作为 Y
                elif '_d' in key or '_D' in key:
                    if '_d' in key: loc = key.find('_d')
                    else: loc = key.find('_D')
                    col = int(key[loc+2:]) * -1
                    X = data[:,:col]
                    Y = data[:,col:]
                # 3、 文件名不以 _x,_X 及 _y,_Y 结尾
                elif '_y' not in key and '_Y' not in key:
                    X = data
                    # 用截取的文件名作为标签
                    if intercept is not None:
                        if intercept[1] == 0: label_name = key[intercept[0]:]
                        else: label_name = key[intercept[0]:intercept[1]]
                        label = [label_name]*data.shape[0]
                    else:
                        label = [key]*data.shape[0]
                    Y = np.array(label)
                    
                # 需将y整体向左偏移 bias,并去掉 x最后bias个时刻样本，使得 x(t0 ~ td - 1) 对应 y(td - 1)
                if self.bias > 0:
                    X = X[:-self.bias]
                    Y = Y[self.bias:]
                
                # if self.task != 'impu':
                print("Load data from '{}' \t-> X{}, Y{}".format(key, X.shape, Y.shape))
                X_list.append(X)
                Y_list.append(Y)
                
            if _tp == 'train':
                self.train_X, self.train_Y = X_list, Y_list
                self.train_name = list(data_dic.keys())
            else:
                self.test_X, self.test_Y = X_list, Y_list
                self.test_name = list(data_dic.keys())
    
    def del_data(self, del_dim = None, del_lbs = None):
        # 删除维度
        for X_list in [self.train_X, self.test_X]:
            if X_list is None: continue
            for k in range(len(X_list)):
                X_list[k] = np.delete(X_list[k],range(del_dim[0], del_dim[1]), axis=1)
        
        # 删除类别
        for index, Y_list in enumerate([self.train_Y, self.test_Y]):
            if Y_list is None: continue
            k = 0
            while k < len(Y_list):                
                total = 0
                # 统计个数
                for del_lb in del_lbs:
                    total += np.where(Y_list[k] == del_lb, 1, 0).sum()
                # 删除整个
                if total == Y_list[k].shape[0]:
                    if index == 0:
                        self.train_X.pop(k)
                        self.train_Y.pop(k)
                        self.train_name.pop(k)
                    else:
                        self.test_X.pop(k)
                        self.test_Y.pop(k)
                        self.test_name.pop(k)
                # 删除部分
                elif total > 0:
                    i = 0
                    while i < Y_list[k].shape[0]:
                        if Y_list[k][i] in del_lbs: 
                            np.delete(X_list[k], i, 0)
                            np.delete(Y_list[k], i, 0)
                        else: i+=1
                    k+=1
                # 无删除
                else:
                    k+=1
    
    def get_category_lables(self, set_normal = -1, set_for = [0,1]):
        self.set_normal = set_normal
        self.set_for = set_for
        
        # 获取类别（去重）
        labels = []
        for index, Y_list in enumerate([self.train_Y, self.test_Y]):
            if Y_list is None: continue
            for k in range(len(Y_list)):
                for i in range(Y_list[k].shape[0]):
                    if Y_list[k][i] not in labels:
                        labels.append(Y_list[k][i])
        labels.sort()
        # 把normal的位置提到最前
        for s in ['normal', 'Normal']:
            if s in labels:
                labels.remove(s)
                labels.insert(0, s)
        # np.nan不是标签
        if np.nan in labels:
            labels.remove(np.nan)
        self.labels = labels
        if self.task != 'impu':
            print('The name of labels are:\n {}'.format(labels))
        
        # 设置数字类别 (str -> int)
        for index, Y_list in enumerate([self.train_Y, self.test_Y]):
            if Y_list is None: continue
            for k in range(len(Y_list)):
                for i in range(Y_list[k].shape[0]):
                    if index in set_for and i < set_normal:
                        Y_list[k][i] = 0
                    elif Y_list[k][i] in self.labels:
                        Y_list[k][i] = self.labels.index(Y_list[k][i])     
    
    def drop_as_nan(self, impu_rate, industrial = True):
        new_path = os.path.abspath(__file__)[:-20] + 'Impu'
        if not os.path.exists(new_path + '/train'): os.makedirs(new_path + '/train')
        if not os.path.exists(new_path + '/test'): os.makedirs(new_path + '/test')
        for _tp in ['train','test']:
            if _tp == 'train':
                _data, _name = self.train_X, self.train_name
            else:
                _data, _name = self.test_X, self.test_name
            #print(len(_data), len(_name), _name)
            for i in range(len(_data)):
                X, key = _data[i], _name[i]
                path_name = new_path + '/' + _tp + '/' + key + '.csv'
                # drop
                if industrial:
                    if hasattr(self, 'chosen_var') == False:
                        # 取 4n 个是 1/2 采样率的； 2n 个是 1/3 采样率的； n 个是 1/6 长条缺失的
                        m1, m2, m3 = 3, 2, 1
                        n =  int(X.shape[1] * impu_rate / (1/2* m1 + 2/3 * m2 + 1/6 * m3))
                        self.chosen_var = np.random.choice(X.shape[1], (m1 + m2 + m3)*n, replace = False)
                    rk1= np.setdiff1d(np.arange(X.shape[0]), np.arange(0, X.shape[0], 2), True)
                    rk2= np.setdiff1d(np.arange(X.shape[0]), np.arange(0, X.shape[0], 3), True)
                    loc_sum = int(X.shape[0] * X.shape[1] * impu_rate)
                    for j, k in enumerate(self.chosen_var):
                        if j < m1*n: X[rk1, k] = float('nan');  loc_sum -= len(rk1)
                        elif j < (m1 + m2)*n: X[rk2, k] = float('nan'); loc_sum -= len(rk2)
                        else:
                            missing_numbers = int(loc_sum / ((m1 + m2 + m3)*n - j))
                            if X.shape[0] - missing_numbers > 0:
                                rd = np.random.randint(0, X.shape[0] - missing_numbers)
                                rk3 = np.arange(rd, rd + missing_numbers)
                                loc_sum -= len(rk3)
                                X[rk3, k] = float('nan')
                else:
                    rd = np.random.rand(*list(X.shape))
                    loc = np.where(rd < impu_rate)
                    X[loc] = float('nan')
                
                # save as 'csv'
                _data[i] = X
                df = pd.DataFrame(X)
                df.to_csv(path_name, header = None, index = None)   
                
    def only_need_train_dataset(self):
        self.train_X += self.test_X
        self.train_Y += self.test_Y
        self.test_X = None
        self.test_Y = None
    
    def gene_dymanic_data(self, dynamic, stride, export, set_normal, set_for, cut_mode):
        # 制作动态数据集
        def get_dymanic_x(_x, _y):
            _dx, _dy = [], []
            r = dynamic
            # 取 [r - dynamic, r) 作为一个动态样本 dx, r - 1 时刻的标签为其 dy
            # r 是取不到的
            while r <= _x.shape[0]:
                start = r - dynamic
                end = r
                if export == '1d':
                    _dx.append(_x[start:end].reshape(1,-1))
                    _dy += [_y[r-1]]
                    r = r + stride
                elif export == 'seq':
                    _dx.append(_x[start:end].reshape(1, dynamic, -1))
                    _dy.append(_y[start:end].reshape(1, dynamic, -1))
                    r = r + stride
                    
            _dx = np.array(np.concatenate(_dx, axis=0), dtype=np.float32)
            if export == '1d':
                _dy = np.array(_dy, dtype=np.float32)
            elif export == 'seq':
                _dy = np.array(np.concatenate(_dy, axis=0), dtype=np.float32)
            return _dx, _dy
        # print(dynamic, stride, export, set_normal, set_for, cut_mode)
        for index, (X_list,Y_list) in enumerate([(self.train_X, self.train_Y), (self.test_X, self.test_Y)]):
            if X_list is not None:
                for k in range(len(X_list)):
                    X, Y = X_list[k],Y_list[k]
                    # 前 set_normal 个数据的标签被设置为了 0
                    if set_normal > -1 and index in set_for and k > 0 and cut_mode != 'continue':
                        x1, y1 = get_dymanic_x(X[:set_normal], Y[:set_normal])
                        x2, y2 = get_dymanic_x(X[set_normal:], Y[set_normal:])
                        # print(x1.shape, x2.shape)
                        X_list[k] = np.concatenate([x1, x2], axis = 0)
                        Y_list[k] = np.concatenate([y1, y2], axis = 0)
                    else:
                        X_list[k], Y_list[k] = get_dymanic_x( X, Y )
                        
    def div_dataset(self, div_prop = None, div_shuffle = False):
        # 分割数据集
        for k in range(len(self.train_X)):
            X, Y = self.train_X[k], self.train_Y[k]
            if div_shuffle:
                index=np.arange(X.shape[0])
                np.random.shuffle(index)
                Xr = X[index]
                Yr = Y[index]
                
            n = int(X.shape[0] * div_prop)
            self.train_X[k], self.train_Y[k] = Xr[:n], Yr[:n]
            
            # 使测试集按打乱前序排列（画图效果更好）
            index2 = index[n:]
            index2 = np.sort(index2, kind = 'heapsort')
            self.test_X.append(X[index2])
            self.test_Y.append(Y[index2]) 
            
if __name__ == "__main__": 
    X1, Y1, X2, Y2 = ReadData('../data/TE', ['st', 'oh'], 40, cut_mode = '', example = 'TE').datasets
    import sys
    sys.path.append('..')
    from _dynamic_data import ReadData as ReadData2
    _X1, _Y1, _X2, _Y2 = ReadData2('../data/TE', ['st', 'oh'], 40, cut_mode = '', example = 'TE').datasets
    print((X1-_X1).sum(), (Y1-_Y1).sum(), (X2-_X2).sum(), (Y2-_Y2).sum())
#    X1, Y1, X2, Y2 = ReadData('../data/CSTR', ['st', 'oh'], 40, example = 'CSTR').dataset
