# -*- coding: utf-8 -*-
import torch.nn as nn
import pandas as pd
import numpy as np
from pandas import DataFrame
import sys
sys.path.append('..')
from core.func import act_dict
from core.layer import ConvBlock

pool_dict = {'M': 'Max',
             'A': 'Avg',
             'AM': 'AdaptiveMax',
             'AA': 'AdaptiveAvg',
             'FM': 'FractionalMax'}

pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)
pd.set_option('display.max_rows', None)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大行


class Conv_Module(object):
    ''' 
        conv:
        [ in_channels, out_channels, kernel_size, stride = 1, padding=0, dilation=1, bias=True ] - default
        bn:
        [ affine = True, track_running_stats = True ] - default
        [ 64, 3, '/2', '+1', '#1', '%1', 'B01', 'r' ] - list
        
        pool:
        [ kernel_size, stride=None(=kernel_size), padding=0, dilation=1 ] - default
        [ 'M', 2, '/2', '+1', '#1' ] - list
        
        res:
        [ 'R', [conv], '*2', '|', [conv] ] - list
        
        set:
        [ 'S', [conv], [conv], '*2' ] - list
        
        repeat:
        [conv], '*2' - list
        
        dataframe: - df
                             Conv   *         Res            Pool   Loop
        0        [ 64, 3, 1, 'B']   2          []   [ 'M', 2, 2 ]      2
        1   [[ 128, 2],[ 128, 1]]   1   [ 64, 1 ]               -      -
        2            [ 64, '/2' ]   3           -     [ 'AM', 3 ]      -
    '''
    def __init__(self, **kwargs):
        default = {'conv_struct': None,
                   'conv_func': 'r',
                   'res_func': 'r',
                   'batch_norm': 'B',
                   'use_bias': True, 
                   'img_size': [3, 224, 224]
                   }
        for key in default.keys():
            if key in kwargs.keys(): 
                default[key] = kwargs[key]
            if hasattr(self, key) == False:
                setattr(self, key, default[key])
        if self.batch_norm == True: self.batch_norm = 'B'
        if self.batch_norm == False: self.batch_norm = 'N'
    
    # 构建 dataframe 中定义的结构
    def Convolutional(self, rt = 'blocks', auto_name = True):
        
        layers, blocks = [], []
        self.para_df = self.list2df(self.conv_struct)
        
        cnt = 0
        for i in range(len(self.para_df)):
            
            # conv_dropout, res_dropout
            if hasattr(self,'conv_dropout') and i > 0: conv_dropout = self.conv_dropout
            else: conv_dropout = None
            if hasattr(self,'res_dropout'): dropout = [conv_dropout, self.res_dropout]
            else: dropout = conv_dropout
            
            # conv_func, res_func
            if type(self.conv_func)!= list: self.conv_func = [self.conv_func]
            f = self.conv_func[np.mod(i, len(self.conv_func))]
            if hasattr(self,'res_func'): func = [f, self.res_func]
            else: func = f
            
            row = self.para_df.loc[i].values
            row = row.copy()
            
            # sub_module <融入 block 里>
            if type(row[0]) == str and row[0][0] == '@':
                index = int(row[1][1:])
                module = self.sub_module[index]
                blocks.append(module)
                layers.append(module)
                continue
            
            loop = row[4]
            for k in range(loop):
                if k > 0:
                    # 循环时，输入尺寸与上一个卷积层的输出尺寸保持一致（这时的 row[0] 一定是个list）
                    if type(row[0][0]) == int: row[0][0], in_channels = row[0][1], row[0][1]
                    else: row[0][0][0], in_channels  = row[0][-1][1], row[0][-1][1]
                    if row[2] not in ['-','[]']:
                        if type(row[2][0]) == int: row[2][0] = in_channels
                        else: row[2][0][0] = in_channels
                    
                block = ConvBlock(row, 
                                  dropout, 
                                  func, 
                                  self.use_bias, 
                                  self.batch_norm,
                                  auto_name)
                
                block.name = 'ConvBlock' + str(cnt)
                cnt += 1
                
                blocks.append(block)
                for _layer in block.layers:
                    layers.append(_layer)
            
        if rt == 'blocks':
            return nn.Sequential(*blocks)
        else:
            return nn.Sequential(*layers)    
    
    # 将输入的 list 转成 dataframe 格式
    def list2df(self, lst):
        '''
        list 中可添加的内容：
            单个整数（无核尺寸）：表示卷积输出通道大小。 默认 [ 输出通道, 核尺寸 = 3, 步长 = 1, 扩展 = 1 ]
            整数开头的list：表示卷积参数设定。 默认 [ 输出通道, 核尺寸, 步长 = '/1', 扩展 = '+0', 空洞 = '#1', 
                                                  分组 = '%1', 偏值 = self.use_bias ]
                           附加参数设定。 默认 [ 批次正则化 = 'N', 激活函数 = 'r', dropout = 'D0', 
                                               按某维度洗牌 = 'SF', 反卷积 = 'TS' ]
            大写字母开头：表示池化的有 {'M', 'A', 'AM', 'AA', 'FM'}
                           {'M', 'A'} 默认 [ 卷积类型, 核尺寸, 步长 = 核尺寸, 扩展 = 0, 空洞 = 1 ]
                           {'AA', 'AM', 'FM'} 默认 [ 卷积类型, 输出尺寸 ]
                        表示残差卷积。 ['R', 卷积参数, '|', 卷积参数(默认无) ]
                           无残差卷积参数的时候：若通道不符合，默认添加 1*1 卷积层；若尺寸不符合，默认添加 'AA'池化层
                        表示集合。 ['S', ...]
            'N*'：表示将后面一个元素重复N遍
            module：表示嵌入已定义好的module（需定义属性 name 和 out_size ）。
        '''
        df = DataFrame( columns = ['Conv', '*', 'Res', 'Pool', 'Loop', 'Out'] )
        self.sub_module = []
        times = 1
        cnt = 0
#        to_list = True
#        for l in lst:
#            if type(l) == list: to_list = False; break
#        if to_list:
#            lst = [lst]
            
        for i, v in enumerate(lst):
            # sub_module
            if isinstance(v, nn.Module):
                df.loc[cnt] = ['@'+v.name, '@'+str(len(self.sub_module)) , '-', '-', 1, 0]
                self.sub_module.append(v)
                cnt += 1
            # repeat
            elif type(v) == str and v[-1] == '*':
                times = int(v[:-1])
                continue
            # conv
            elif type(v) == int or type(v[0]) == int:
                df.loc[cnt] = [v, times, '-', '-', 1, 0]
                cnt += 1
            # pool
            elif ( type(v) == str and v in pool_dict.keys() ) \
            or ( type(v) == list and v[0] in pool_dict.keys() ):
                # 上一个也为 pool 或 为 sub_module
                if ( type(lst[i-1]) == str and lst[i-1] in pool_dict.keys() ) \
                or ( type(lst[i-1]) == list and lst[i-1][0] in pool_dict.keys() ) \
                or ( type(df.loc[cnt-1][0]) == str and df.loc[cnt-1][0][0] == '@' ):
                    df.loc[cnt] = ['-', 1, '-', v, 1, 0]
                    cnt += 1
                else:
                    df.iloc[cnt-1,3] = v
            # set/res
            elif v[0] in ['S','R']:
                if v[0] == 'S':
                    out = ['-', 1 , '-', '-', times, 0]
                else:
                    out = ['-', 1 , '[]', '-', times, 0]
                v = v[1:]
                conv, res = [], []
                to_res = False
                for x in v:
                    # repeat
                    if type(x) == str:
                        if x[-1] == '*': out[1] = int(x[:-1])
                        elif x == '|': to_res = True
#                        else: res.append(x)
                    # conv
                    elif type(x) == int or type(x[0]) == int:
                        if to_res: res.append(x)
                        else: conv.append(x)
                    # pool
                    elif x[0] in pool_dict.keys():
                        out[3] = x
                if len(conv) == 1: out[0] = conv[0]
                elif len(conv) > 0: out[0] = conv
                if len(res) == 1: out[2] = res[0]
                elif len(res) > 0: out[2] = res
                df.loc[cnt] = out
                cnt += 1
            times = 1
        
        #print(df)
        in_channel, size = self.img_size[0], self.img_size
        for i in range(len(df)):
            row = df.loc[i].values.copy()
            in_channel, df.loc[i] = self.check_row(in_channel, row)
            row = df.loc[i].values.copy()
            size, df.loc[i][5] = self.get_out_size(size, row)
        print(df)
        return df
    
    # 检查 dataframe 每行完备性
    def check_row(self, in_channel, row):
        
        def check_conv(in_dim, v):
            if type(v) == int or len(v) == 1:
                return [in_dim, v, 3, 1, 1]
            left = [in_dim, 'out_dim', 'k_size', 1, 0, 1, 1, self.use_bias]
            right = [] # batch_norm, func, dropout
            cnt = 1
            loc = 0
            for x in v:
                # out_dim, k_size, stride, padding, dilation
                if type(x) == int or type(x) == tuple:   left[cnt] = x;   cnt += 1
                # bias
                elif type(x) == bool and x != left[7]:   left[7] = x;     loc = max(7,loc)
                # batch_norm
                elif x in ['B', 'N'] and x != self.batch_norm:       right += [x]
                # func
                elif x in act_dict.keys():
                    if process_res:
                        if x != self.res_func:   right += [x]
                    else:
                        if x != self.conv_func:   right += [x]
                elif type(x) == str:
                    # stride
                    if x[0] == '/':     left[3] = eval(x[1:]);   loc = max(3,loc)
                    # padding
                    elif x[0] == '+':   left[4] = eval(x[1:]);   loc = max(4,loc)
                    # dilation
                    elif x[0] == '#':   left[5] = eval(x[1:]);   loc = max(5,loc)
                    # groups
                    elif x[0] == '%':   left[6] = eval(x[1:]);   loc = max(6,loc)
                    # batch_norm
                    elif x[0] == 'B' and x != self.conv_func:  
                        if len(x) == 2: x += '1'
                        right += [x]
                    # dropout
                    elif x[0] == 'D':
                        dropout = float(x[1:])
                        if process_res:
                            if hasattr(self, 'res_dropout') == False \
                            or self.res_dropout != dropout:
                                right += [x]
                        else:
                            if hasattr(self, 'conv_dropout') == False \
                            or self.conv_dropout != dropout:
                                right += [x]
                    else:
                        right += [x]
                        
            # no k_size, e.g. [64]
            if cnt-1 < 2:
                left[cnt] = 3;   cnt += 1
            loc = max(cnt-1, loc)
            out = left[:loc+1] + right
            return out
        
        def check_pool(v):
            if type(v) == str:
                return [pool_dict[v], 2, 2]
            out = [ 'pool_type', 'k_size', None, 0, 1 ]
            cnt = 1
            loc = 2
            for x in v:
                # pool_type
                if x in pool_dict.keys(): out[0] = pool_dict[x]
                # k_size, stride, padding, dilation
                elif type(x) == int or type(x) == tuple: out[cnt] = x; cnt += 1
                # stride
                elif x[0] == '/': out[2] = eval(x[1:]); loc = max(2, loc)
                # padding
                elif x[0] == '+': out[3] = eval(x[1:]); loc = max(3, loc)
                # dilation
                elif x[0] == '#': out[4] = eval(x[1:]); loc = max(4, loc)
            # stride = k_size
            if 'Adaptive' in out[0]:
                return out[:2]
            if out[2] is None: 
                if cnt == 2: cnt += 1
                out[2] = out[1]
            loc = max(cnt-1, loc)
            out = out[:loc+1]
            return out
    
        conv_para, res_para, pool_para = row[0], row[2], row[3]
        
        # sub_module
        if type(conv_para) == str and conv_para[0] == '@':
            index = int(row[1][1:])
            module = self.sub_module[index]
            out_channel = module.out_size[0]
            return out_channel, row
        
        # conv
        process_res = False
        out_channel = in_channel
        if conv_para != '-':
            if type(conv_para) == int or type(conv_para[0]) == int: 
                row[0] = check_conv(in_channel, conv_para)
                out_channel = row[0][1]
            else:
                for i in range(len(conv_para)):
                    row[0][i] = check_conv(out_channel, conv_para[i])
                    out_channel = row[0][i][1]
                    
        # res
        process_res = True
        if type(res_para) != str:
            if type(res_para) == int or type(res_para[0]) == int:
                row[2] = check_conv(in_channel, res_para)
                if row[2][1] != out_channel: row[2][1] = out_channel
            else:
                res_dim = in_channel
                for i in range(len(res_para)):
                    row[2][i] = check_conv(res_dim, res_para[i])
                    res_dim = row[2][i][1]
                if row[2][-1][1] != out_channel: row[2][-1][1] = out_channel
        
        # pool
        if pool_para != '-':
            row[3] = check_pool(pool_para)        
        
        return out_channel, row
    
    # 计算 dataframe 中每行操作后的 输出尺寸
    def get_out_size(self, in_size, row):
        ''' 
            conv_para: (in_channels(auto), out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            pool_para: (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        ''' 
        def to_2dim(x):
            if type(x) == tuple:
                x = list(x)
            if type(x) == list and len(x) <2: 
                return [x[0], x[0]]
            if type(x) == int or len(x) < 2:
                return [x, x]
            else:
                return x
        
        def cal_size(H_in, W_in, kernel_size, stride = 1, padding = 0, dilation = 1, groups = 1):
            kernel_size, stride, padding, dilation = to_2dim(kernel_size), to_2dim(stride), to_2dim(padding), to_2dim(dilation)
            H_out = int( (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1 )
            W_out = int( (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1 )
            return [H_out, W_out]
        
        def drop_str(_para):
            _cal_para = []
            for j in range(2, len(_para)):
                if type(_para[j]) == int or type(_para[j]) == tuple: 
                    _cal_para.append(_para[j])
            return _cal_para
        
        times, loop =  row[1], row[4]
        if type(row[0]) == list: conv_para = row[0].copy()
        else: conv_para = row[0]
        if type(row[3]) == list: pool_para = row[3].copy()
        else: pool_para = row[3]
        
        # sub_module
        if type(conv_para) == str and conv_para[0] == '@':
            index = int(times[1:])
            module = self.sub_module[index]
            out_size = module.out_size
            if out_size[1] == -1: out_size[1] = in_size[1]
            if out_size[2] == -1: out_size[2] = in_size[2]
            return out_size, out_size
        
        # drop str in conv_para, get out_channel
        out_channel, size = in_size[0], in_size[1:]
        if conv_para != '-': 
            if type(conv_para[0]) == list:
                out_channel = conv_para[-1][1]
                for i, para in enumerate(conv_para):   conv_para[i] = drop_str(para)
            else:
                out_channel = conv_para[1]
                conv_para = drop_str(conv_para)
        
        # get type and para of pool
        if pool_para != '-': 
            pool_type, pool_para = pool_para[0], pool_para[1:]
        
        for _ in range(loop):
            # conv
            if conv_para != '-': 
                for _ in range(times):
                    if type(conv_para[0]) == list:
                        for para in conv_para:
                            size = cal_size(size[0], size[1], *para)
                    else:
                        size = cal_size(size[0], size[1], *conv_para)
        
            # pool
            if pool_para != '-': 
                if 'Adaptive' in pool_type: # 只有一个 out_size 的参数
                    pool_para = to_2dim(pool_para)
                    if pool_para[0] is None: pool_para[0] = size[0]
                    if pool_para[1] is None: pool_para[1] = size[1]
                    size = pool_para
                else:
                    size = cal_size(size[0], size[1], *pool_para)
        
        out_size = [out_channel, size[0], size[1]]
        return out_size, out_size
    
if __name__ == '__main__':
    from model.resnet import cfgs as res_cfgs
    from model.vgg import cfgs as vgg_cfgs

    # resnet
    lst = []
    for l in res_cfgs['resnet50']: lst += l
    
    # vgg
    #lst = vgg_cfgs['vgg19']
    
    #print(lst)
    cm = Conv_Module()
    df = cm.list2df(lst)
    print(df)
    