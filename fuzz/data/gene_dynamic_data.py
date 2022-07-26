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
def preprocess(train, test = None, prep = 'st', feature_range=(0, 1), n_category = None):
    
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
    
    # 读取参数
    scaler = None
    if prep is None:
        return train, test, None
    elif type(prep) == int:
        feature_range = (-1*prep, prep)
        prep = 'mm'
    elif type(prep) == tuple:
        feature_range = (prep[0], prep[1])
        prep = 'mm'
    
    # 获取数组训练集
    _train = train
    if type(train) == list:
        _train = np.concatenate(train, axis = 0)
    
    # fit 训练集
    if prep == 'oh': 
        if n_category is None:
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
    
    # transform 训练集
    if scaler is not None:
        _train = scaler.transform(_train)
    
    # 记录 scaler 中的系数 
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
        
    # transform 训练集与测试集
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
            # print('st:', '\nmean = ', scaler.mean_, '\nvar = ', scaler.var_)
        elif scaler.name == 'mm':
            np.savetxt(path + '/scaler/[{}] scaler_{}.csv'.format(scaler.name, tp), 
                       np.concatenate([scaler.scale_.reshape(-1,1), scaler.min_.reshape(-1,1)], 1) , 
                       fmt='%f', delimiter=',')
            # print('mm:', '\nscale = ', scaler.scale_, '\nmin = ', scaler.min_)
    
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
    if suffix in ['csv']:
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

def cal_missing_number(X):
    return int(np.isnan(X).astype(int).sum()/X.shape[1])
    
class ReadData():
    def __init__(self, 
                 path,                  # 路径
                 prep = None,           # 预处理
                 dynamic = 0,           # 滑窗长度 (= seq_len), dynamic = 0 时返回 list
                 stride = 1,            # 滑动步长 
                 bias = 0,              # x(t0 ~ td - 1) 对应 y(td - 1 + bias)
                 task = 'cls',          # 可以是 'cls'/'prd'/'impu'/'fd' 任务
                 export = '1d',         # 导出X的数据类型 '1d' / 'seq'
                 seg_name = None,       # 将截取 [start, end] 的文件名作为标签名
                 is_del = True,         # 是否执行删除函数
                 missing_rate = 0,      # 将数据按概率填充为 nan
                 drop_label_rate = 0,   # 将标签按概率填充为 nan
                 div_prop = None,       # 分割比例>0时，将后 1 - div_prop 作为测试集
                 div_shuffle = False,   # 分割前是否打乱
                 set_normal = -1,       # 用于故障诊断，将前 set_normal 个样本的标签设为 0
                 set_for = [0,1],       # set_normal 是否应用于 [训练集, 测试集]
                 cut_mode = 'seg',      # 滑窗到类别分界点时，continue滑窗 or seg 滑窗
                 example = '',          # 特定内设数据集
                 single_mode = False,   # 单/双模态
                 save_data = False      # 是否存储制作好的动态数据集
                 ):

        self.train_X, self.train_Y, self.test_X, self.test_Y, self.scaler = None, None, None, None, None
        self.dynamic, self.stride, self.bias = dynamic, stride, bias
        self.is_del = is_del
        self.task = task
        self.missing_rate = missing_rate
        if type(prep) == list:
            prep_x, prep_y = prep[0], prep[1]
        else:
            prep_x, prep_y = prep, None
        
        # 读取原始数据
        self.example = example
        if example == 'TE':
            self.laod_data(path, seg_name = [1,3])
            if is_del: self.del_data(del_dim = np.arange(22,41), del_lbs = ['03','09','15'])
            # 将测试集前 160 个样本设为正常
            set_normal, set_for = 160, [1]
            self.get_category_lables(set_normal = set_normal, set_for = set_for)
        elif example == 'CSTR':
            if seg_name is None: seg_name = [7,-1]
            self.laod_data(path, seg_name = seg_name)
            if is_del: self.del_data(del_dim = np.array([3,4,8]), del_lbs = ['Fault04','Fault05','Fault06'])
            # 将训练集和测试集前 201 个样本设为正常
            set_normal = 201
            if task == 'cls':  set_for = [0,1]
            else: set_for = [1]
            self.get_category_lables(set_normal = set_normal, set_for = set_for)
        elif example == 'MFF':
            self.laod_data(path, seg_name = [0,4])
            set_normal = [(0, 1565, 5181, 5811), (0, 656, 3777, 4467), (0, 690, 3691, 4321),
                          (0, 2243, 6616, 9192), (0, 475, 2656, 3496), (0, 330, 2467, 3421),
                          (0, 1135, 8352, 9090), (0, 332, 5871, 6272), (0, 595, 9566, 10764),
                          (0, 952, 6294, 7208), (0, 850, 3851, 4451), (0, 240, 3241, 3661),
                          (0, 685, 1172, 1771, 2253, 2541), (0, 1632, 2955, 7030, 7553, 8056),
                          (0, 1722), (0, 1036)]
            n_samples = [14599, 16109, 26126, 15320, 13149, 7630]
            self.split_p_list = [0]
            sum_samples = 0
            for i in range(len(n_samples)):
                sum_samples += n_samples[i]
                self.split_p_list.append(sum_samples)
            self.plot_p_list = [(5811, 10278, 14599), (9192, 12688, 16109), (9090, 15362, 26126),
                                (7208, 11659, 15320), (2541, 13149), (2800, 7630)]
            self.get_category_lables(set_normal, set_for)
        else:
            self.laod_data(path, seg_name)
            if task == 'cls' or task == 'fd':
                self.get_category_lables(set_normal, set_for)
        
        # 删掉测试集的（纯）正常类别数据
        if self.task == 'fd':
            if self.example == 'TE':
                del self.test_X[0]
                del self.test_Y[0]
            # elif self.example == 'CSTR':
            #     if len(self.test_X) > 20:
            #         # add test model1_normal
            #         # self.train_X.insert(11, self.test_X[10].copy())
            #         # self.train_Y.insert(11, self.test_Y[10].copy())
            #         # self.train_X.append(self.test_X[21].copy())
            #         # self.train_Y.append(self.test_Y[21].copy())
            #         del self.test_X[21]
            #         del self.test_Y[21]
            #     if self.is_del == False:
            #         try:
            #             del self.test_X[10]
            #             del self.test_Y[10]
            #         except IndexError:
            #             pass
            #     else:
            #         del self.test_X[7]
            #         del self.test_Y[7]
        
        # 单模态时，去掉后半部分数据
        if example == 'CSTR' and single_mode and len(self.train_X) > 1:
            self.train_X = self.train_X[:int(len(self.train_X)/2)]
            self.train_Y = self.train_Y[:int(len(self.train_Y)/2)]
            self.test_X = self.test_X[:int(len(self.test_X)/2)]
            self.test_Y = self.test_Y[:int(len(self.test_Y)/2)]
        
        # 删除训练数据中的故障数据
        if self.task == 'fd' and len(self.train_X) > 1:
            i = 0
            while i < len(self.train_Y):
                normal_indexs = np.argwhere(self.train_Y[i].astype(int) == 0).reshape(-1,)
                if normal_indexs.shape[0] == 0:
                    del self.train_X[i]
                    del self.train_Y[i]
                else:
                    self.train_X[i] = self.train_X[i][normal_indexs].copy()
                    self.train_Y[i] = self.train_Y[i][normal_indexs].copy()
                    i += 1
        
        # 丢失 X (imputation)
        if missing_rate > 0:
            self.drop_as_nan(path, missing_rate, industrial = True)
            
        # if task == 'impu': self.only_need_train_dataset()
        self.make_dataset(path, prep_x, prep_y, dynamic, stride, 
                          task, drop_label_rate, div_prop, div_shuffle, 
                          export, set_normal, set_for, cut_mode, save_data)
        
    def make_dataset(self, path, prep_x, prep_y, dynamic, stride, 
                     task, drop_label_rate, div_prop, div_shuffle, 
                     export = '1d', set_normal = -1, set_for = [0,1], 
                     cut_mode = 'seg', save_data = False):
        
        # 对 X 预处理
        if prep_x is not None and (div_prop is None or div_prop == 0):
            self.train_X, self.test_X, self.scaler_x = preprocess(self.train_X, self.test_X, prep_x)
            Scaler().save(self.scaler_x, path, '', 'x')
        
        # 生成动态数据 (dynamic)
        if dynamic > 1:
            self.gene_dymanic_data(dynamic, stride, export, set_normal, set_for, cut_mode)
        
        # 分割训练/测试集 (shuffle)
        if div_prop is not None and div_prop > 0:
            self.div_dataset(div_prop, div_shuffle)
            self.train_X, self.test_X, self.scaler_x = preprocess(self.train_X, self.test_X, prep_x)
            Scaler().save(self.scaler_x, path, '', 'x')
        
        # 合并 list 数据集 (to array)
        if self.train_X is not None and type(self.train_X) == list:
            self.train_X = np.concatenate(self.train_X, axis = 0)
            self.train_Y = np.concatenate(self.train_Y, axis = 0)
        if self.test_X is not None and type(self.test_X) == list:
            self.test_X = np.concatenate(self.test_X, axis = 0)
            self.test_Y = np.concatenate(self.test_Y, axis = 0)
            
        # 对 Y 预处理
        if prep_y is not None:
            self.train_Y, self.test_Y, self.scaler_y = preprocess(self.train_Y, self.test_Y, prep_y, n_category = len(self.labels))
            Scaler().save(self.scaler_y, path, '', 'y')
        
        # 丢失 Y (semi-supervision)
        if drop_label_rate > 0:
            self.drop_label(path, drop_label_rate)
        
        # 保存制作好的动态数据集
        if dynamic > 1 and save_data:
            if not os.path.exists(path + '/gene'): os.makedirs(path + '/gene')
            np.savetxt(path + '/gene/trian_X.csv', self.trian_X, '%f', ',')
            np.savetxt(path + '/gene/trian_Y.csv', self.trian_Y, '%f', ',')
            np.savetxt(path + '/gene/test_X.csv', self.test_X, '%f', ',')
            np.savetxt(path + '/gene/test_Y.csv', self.test_Y, '%f', ',')
        
        self.datasets = self.train_X, self.train_Y, self.test_X, self.test_Y
        
        # 打印信息
        shapes = []
        for data_set in [self.train_X, self.train_Y, self.test_X, self.test_Y]:
            if data_set is None:
                shapes.append(' None')
            elif type(data_set) == list:
                shapes.append(' [list] * {}'.format(len(data_set)))
            else:
                shapes.append(data_set.shape)
        if task != 'impu': print('Gene datasets with shape:')
        print('->  train_X{},  train_Y{}\n->  test_X{},  test_Y{}'.\
              format(shapes[0], shapes[1], shapes[2], shapes[3]))
        if task == 'impu':
            print('Number of missing values:')
            print('->  train_X({}),  train_Y({})\n->  test_X({}),  test_Y({})'.\
                format(cal_missing_number(self.train_X), cal_missing_number(self.train_Y),
                       cal_missing_number(self.test_X), cal_missing_number(self.test_Y)))
    
    # 得到数据为 list， Y 可能为 str list
    def laod_data(self, 
                  path,                 # 路径
                  seg_name = None       # 截取文件名 [start, end] 作为标签名
                  ):
        
        # 将数据放入 train 与 test 文件夹中
        for _tp in ['train','test']:
            print('Load {} data ...'.format(_tp))
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
                if 'EvoFault' in key:
                    continue
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
                    if seg_name is not None:
                        if seg_name[1] == -1: label_name = key[seg_name[0]:]
                        else: label_name = key[seg_name[0]:seg_name[1]]
                        label_name = label_name.capitalize()
                        label = [label_name]*data.shape[0]
                    else:
                        label = [key]*data.shape[0]
                    Y = np.array(label)
                    
                # 需将y整体向左偏移 bias,并去掉 x最后bias个时刻样本，使得 x(t0 ~ td - 1) 对应 y(td - 1)
                if self.bias > 0:
                    X = X[:-self.bias]
                    Y = Y[self.bias:]

                print("->  from '{}'\t-> X{}, Y{}".format(key, X.shape, Y.shape))
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
                X_list[k] = np.delete(X_list[k],del_dim, axis=1)
        
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
                if type(set_normal) == list: 
                    normal_locs = set_normal[k]
                    if type(normal_locs) == tuple:
                        normal_loc_list = []
                        for p in range(int(len(list(normal_locs))/2)):
                            normal_loc_list.append(np.arange(normal_locs[int(2*p)],\
                                                             normal_locs[int(2*p+1)]).reshape(1,-1))
                        normal_locs = np.concatenate(normal_loc_list, axis = 1)
                    else:
                        normal_locs = np.arange(normal_locs)
                else:
                    normal_locs = np.arange(set_normal)
                        
                for i in range(Y_list[k].shape[0]):
                    if index in set_for and i in normal_locs:
                        Y_list[k][i] = 0
                    elif Y_list[k][i] in self.labels:
                        Y_list[k][i] = int(self.labels.index(Y_list[k][i])) 
    
    def drop_as_nan(self, path, missing_rate, industrial = True):
        new_path = path + ' [' + str(missing_rate) + ']'
        if not os.path.exists(new_path + '/train'): os.makedirs(new_path + '/train')
        if not os.path.exists(new_path + '/test'): os.makedirs(new_path + '/test')
        for _tp in ['train','test']:
            if _tp == 'train':
                _data, _name = self.train_X, self.train_name
            else:
                if self.test_X is None: continue
                _data, _name = self.test_X, self.test_name
            #print(len(_data), len(_name), _name)
            for i in range(len(_data)):
                X = _data[i]
                if _name is None: key = str(i+1)
                else: key = _name[i]
                path_name = new_path + '/' + _tp + '/' + key + '.csv'
                # drop
                if industrial:
                    if hasattr(self, 'chosen_var') == False:
                        # 取 3n 个是 1/2 采样率的； 2n 个是 1/3 采样率的； n 个是 1/6 长条缺失的
                        miss = [1/2, 2/3, 3/5]
                        if missing_rate <=0.3: n = [3, 2, 1]
                        elif missing_rate <=0.5: n = [2, 3, 1]
                        else: n = [1, 2, 3]
                        p = 0
                        while np.dot(n, miss) < X.shape[1] * missing_rate:
                            n[np.mod(p, len(n))] += 1
                            p += 1
                        if np.sum(np.array(n)) > X.shape[1]:
                            diff = int((np.sum(np.array(n)) - X.shape[1]) / 2)
                            n[1] -= diff
                            n[0] -= np.sum(np.array(n)) - X.shape[1]
                            
                        self.chosen_var = np.random.choice(X.shape[1], np.sum(np.array(n)), replace = False)
                        string = '{}, sum  = {}\n'.format(n, np.sum(np.array(n)))
                        string += 'variable {} missing with rate {}\n'.format(self.chosen_var[:n[0]], miss[0])
                        string += 'variable {} missing with rate {:.2f}\n'.format(self.chosen_var[n[0]:(n[0] + n[1])], miss[1])

                    loc_sum = int(X.shape[0] * X.shape[1] * missing_rate)
                    for j, k in enumerate(self.chosen_var):
                        if j < n[0]: 
                            rk1= np.setdiff1d(np.arange(X.shape[0]), np.arange(np.random.randint(0,2), X.shape[0], 2), True) # 1/2 采样
                            X[rk1, k] = float('nan');  loc_sum -= len(rk1)
                        elif j < (n[0] + n[1]): 
                            rk2= np.setdiff1d(np.arange(X.shape[0]), np.arange(np.random.randint(0,3), X.shape[0], 3), True) # 1/3 采样
                            X[rk2, k] = float('nan'); loc_sum -= len(rk2)
                        else:
                            missing_numbers = int(loc_sum / (len(self.chosen_var) - j))
                            if X.shape[0] - missing_numbers > 0:
                                rd = np.random.randint(0, X.shape[0] - missing_numbers)
                                rk3 = np.arange(rd, rd + missing_numbers)
                                rk3_mr = len(rk3)/ X.shape[0]
                                loc_sum -= len(rk3)
                                X[rk3, k] = float('nan')
                            else:
                                print('Error: Exceed the maximum number of samples !')
                    
                else:
                    rd = np.random.rand(*list(X.shape))
                    loc = np.where(rd < missing_rate)
                    X[loc] = float('nan')
                
                # save as 'csv'
                _data[i] = X
                df = pd.DataFrame(X)
                df.to_csv(path_name, header = None, index = None)
        
        string += 'variable {} missing with rate {:.2f}\n'.format(self.chosen_var[(n[0] + n[1]):], rk3_mr)
        string += 'Total missing rate = {}'.format( np.sum(np.isnan(X).astype(int))/ X.shape[0] / X.shape[1] )
        print('\n'+ string +'\n')
        fh = open(new_path + '/readme.txt', 'w', encoding='utf-8')
        fh.write(string)
        fh.close()
                
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
        self.test_X, self.test_Y = [], []
        for k in range(len(self.train_X)):
            X, Y = self.train_X[k].copy(), self.train_Y[k].copy()
            # 分割位置
            n = int(X.shape[0] * div_prop)
            if n==0 or n == X.shape[0]: continue
            # 取 list 中每个 matrix 的前 n 个留在训练集，后 n 个放入测试集
            if div_shuffle:
                index = np.arange(X.shape[0])
                np.random.shuffle(index)
                Xr = X[index]
                Yr = Y[index]
                self.train_X[k], self.train_Y[k] = Xr[:n], Yr[:n]
                # 使测试集按打乱前序排列（画图效果更好）
                test_index = index[n:]
                test_index = np.sort(test_index, kind = 'heapsort')
                self.test_X.append(X[test_index])
                self.test_Y.append(Y[test_index]) 
            else:
                self.train_X[k], self.train_Y[k] = X[:n], Y[:n]
                self.test_X.append(X[n:])
                self.test_Y.append(Y[n:])
    
    def drop_label(self, path, rate, data_set = [1,0], except_normal = True):
        _tp = ['train', 'test']
        for i, Y in enumerate([self.train_Y, self.test_Y]):
            if data_set[i] == 1:
                file = path + '/drop_' + _tp[i] + '_label_index [{}].csv'.format(rate)
                category = np.argmax(Y,1)
                if os.path.exists(file):
                    # read
                    chosen_sample = np.loadtxt(file, dtype=np.int, delimiter=',')
                else:
                    if except_normal:
                        Y_set = np.setdiff1d(np.arange(Y.shape[0]), np.where(category == 0), True)
                    else:
                        Y_set = Y.shape[0]
                    chosen_sample = np.random.choice(Y_set, int(Y.shape[0]*rate), replace = False)
                    # save
                    np.savetxt(file, chosen_sample, delimiter=',')
                # 统计各个类别丢失标签的比例
                drop_rate = np.round(np.bincount(category[chosen_sample])/np.bincount(category),2)
                drop_rate_total = np.round(chosen_sample.shape[0]/category.shape[0],2)
                print('The missing rates of labels in each category is:\n{}'.format(drop_rate))
                print('The total missing rate of label is {:.2f}'.format(drop_rate_total))
                file2 = path + '/drop_' + _tp[i] + '_label_rate [{}].csv'.format(drop_rate_total)
                np.savetxt(file2, drop_rate, delimiter=',')
                # 丢
                for k in chosen_sample:
                    Y[k] = float('nan')
            
if __name__ == "__main__": 
    X1, Y1, X2, Y2 = ReadData('../data/TE', ['st', 'oh'], 40, cut_mode = '', example = 'TE').datasets
    import sys
    sys.path.append('..')
    from _dynamic_data import ReadData as ReadData2
    _X1, _Y1, _X2, _Y2 = ReadData2('../data/TE', ['st', 'oh'], 40, cut_mode = '', example = 'TE').datasets
    print((X1-_X1).sum(), (Y1-_Y1).sum(), (X2-_X2).sum(), (Y2-_Y2).sum())
#    X1, Y1, X2, Y2 = ReadData('../data/CSTR', ['st', 'oh'], 40, example = 'CSTR').dataset
