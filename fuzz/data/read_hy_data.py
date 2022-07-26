# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from fuzz.data.gene_dynamic_data import ReadData, preprocess, Scaler

# for 'cls': dynamic = 18, div_prop = 0.7
    
class ReadHYData(ReadData):
    def __init__(self,
                 path = '../data/hydrocracking/hydrocracking.xls',
                 prep = None,           # 预处理
                 dynamic = 0,           # 滑窗长度 (= seq_len)
                 stride = 1,            # 滑动步长 
                 task = 'cls',          # 由于 'cls'/'prd'/'impu' 任务
                 missing_rate = 0,      # 将数据按概率填充为 nan
                 drop_label_rate = 0,   # 将标签按概率填充为 nan
                 div_prop = 0,          # 分割比例>0时，将后 1 - div_prop 作为测试集
                 div_shuffle = False,   # 分割前是否打乱
                 ):
        '''
            划分训练集和测试集：
            数据由不同时段运行记录得到
            1、 所有运行数据整合 -> 打乱 -> 按某一比例分段
                优：获取的数据集测试正确率往往会很高，这时训练集中包含测试集信息，它们是同源的
                缺：实际应用时时不可能包含测试集信息
            2、 按类别轮流将一次运行的数据分配至训练集或测试集（防止训练集/测试集中某类别无数据的情况）
                优：划分方式符合实际，获取的数据集在每两次运行差异不大的时候效果尚可
                缺：若数据存在多模态特性，训练集和测试集差异性较大导致训练效果很差（这体现了迁移学习的价值？）
            3、 用一个比例选择用上述两种方法中的一种（折中一下）
                然而训练结果的好坏严重依赖于这个比例的大小，用方法1的比例越高则训练效果越佳
            4、 对每一次运行的数据 —> 不打乱 -> 按某一比例分段
                优：这回结果是真的折中了
                缺：真实情况下，测试集里的数据有可能来自训练集中未出现过的模态，而这种划分方式忽视了这个问题
            尝试解决：
            是否能用一个模型先消除数据的多模态性，将它们投影到同一模态（投影前后变量维度保持不变）
            考虑用一个训练/测试样本与训练集中的所有/部分样本做对比，得到一个投影后的样本
        '''
        self.train_X, self.train_Y, self.test_X, self.test_Y, self.scaler = None, None, None, None, None
        self.dynamic, self.stride = dynamic, stride
        self.task = task
        self.missing_rate = missing_rate
        if type(prep) == list:
            prep_x, prep_y = prep[0], prep[1]
        else:
            prep_x, prep_y = prep, None
        
        # 从文件读取数据
        original_file = True
        if path.find('.xls'):
            hy = pd.read_excel(path, engine = 'xlrd').values
            path = '../data/'+ path[path.rfind('/')+1:path.rfind('.')]
        else:
            try:
                hy = pd.read_excel(path + '/hydrocracking.xls', engine = 'xlrd').values
            except PermissionError:
                hy = pd.read_csv(path + '/train/incomplete.csv', header = None).values
                original_file = False
            
        # 时间， 数据， 标签
        T, X, Y = hy[:,0], hy[:,1:-9], np.argmax(hy[:,-9:], axis = 1)
        # 丢失数据
        if missing_rate > 0:
            self.train_X, self.train_name = [np.array(X).astype(np.float32)], ['incomplete']
            self.drop_as_nan(path, missing_rate, industrial = True)
            X = self.train_X[0]
            path += ' [' +str(missing_rate) +']'
        
        # 数据分段
        self.train_X, self.train_Y = [], []
        if original_file:
            seg_loc = []
            # 参照组：时段，分段的 起始点，标签
            t0, p0, y0 = T[0][:9], 0, Y[0]
            for i in range(1,T.shape[0]):
                t, y = T[i][:9], Y[i]
                # 分段依据：日期改变，标签改变，遍历到最后
                # 遍历到不同的分段起点要将前面的分段保存
                if t != t0 or y != y0 or i == T.shape[0]-1:
                    if i == T.shape[0]-1: i+= 1
                    seg_loc.append(i)
                    
                    # 分段长是否大于 滑窗长
                    if i >= p0 + dynamic:
                        self.train_X.append( X[p0:i] )
                        self.train_Y.append( Y[p0:i] )
                    else:
                        print('{} < {}, its label is {}'.format(i - p0, dynamic, y0))
                    
                    # 设定分段起点
                    t0, p0, y0 = t, i, y
            
            # 保存分段位置 seg_loc
            np.savetxt(path + '/seg_loc.csv',np.array(seg_loc),fmt='%d',delimiter=',')
        else:
            # 读取分段位置 得到 list
            # print(hy.shape)
            seg_loc = list(np.loadtxt(path + '/seg_loc.csv'))
            self.train_Y = None
            start = 0
            for index in seg_loc:
                self.train_X.append(hy[start: int(index)])
                # print(start, int(index), int(index) - start, hy[start: int(index)].shape[0])
                start = int(index)
                
        
        self.get_category_lables(set_normal = -1, set_for = [0,0])
        
        self.make_dataset(path, prep_x, prep_y, dynamic, stride, 
                          task, drop_label_rate, div_prop, div_shuffle)
            
if __name__ == "__main__":
    rd = ReadHYData(path = 'hydrocracking/hydrocracking.xls',
                    prep = (0.1, 0.9),     # 预处理
                    task = 'impu',         # 由于 'cls'/'prd'/'impu' 任务
                    missing_rate = 0.1,    # 将数据按概率填充为 nan
                    )