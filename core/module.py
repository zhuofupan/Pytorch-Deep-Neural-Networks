# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from core.load import Load
from core.func import Func
from core.epoch import Epoch
from pandas import DataFrame
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Module(torch.nn.Module,Load,Func,Epoch):
    
    def default_setting(self, **kwargs):
        # default setting
        if 'img_size' in kwargs.keys(): flatten = False
        else: flatten = True
        default = {'flatten': flatten,
                   'unsupervised': False,
                   'L': torch.nn.MSELoss(),
                   'msg': [],
                   'dvc': device,
                   'best_acc': 0,
                   'best_rmse': float('inf')}
        for key in default.keys():
            setattr(self, key, default[key])
    
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.default_setting(**kwargs)
        for key in kwargs.keys(): setattr(self, key, kwargs[key])
        
        if self.task == 'cls':
            head = ['loss', 'accuracy']
        elif self.task == 'prd':
            head = ['loss', 'rmse', 'R2']
        else:
            head = ['loss']
            
        self.train_df = DataFrame(columns = head)
        self.test_df = DataFrame(columns = head)
        
    def __call__(self, **kwargs):
        return self.forward(**kwargs)
    
    def opt(self):
        '''
            Adadelta, Adagrad, Adam, SparseAdam, Adamax, ASGD, SGD, Rprop, RMSprop, Optimizer, LBFGS
        '''
        if hasattr(self, 'optim'): optim = self.optim
        else: optim = 'Adam'
        if hasattr(self, 'optim_setting'): 
            setting = self.optim_setting # 字符串
        elif hasattr(self, 'lr'): 
            setting = 'self.parameters(), lr = self.lr'
        else: 
            setting = 'self.parameters()'
        if isinstance(optim,str):
            self.optim  = eval('torch.optim.'+optim+'('+setting+')') 
        #print_module:
        print()
        print(self)
        #print_parameter:
        print("{}'s Parameters(".format(self.name))
        for para in self.state_dict():print('  {}'.format(para))
        print(')')
        
    def Sequential(self, struct = None, is_drop = True):
        if struct is None:
            struct = self.struct
        if len(struct) == 0: return
            
        self.feature = nn.Sequential() 
        for i in range(len(struct)-2):
            if is_drop and (isinstance(self.dropout,list) or self.dropout > 0):
                self.feature.add_module('Dropout'+str(i),nn.Dropout(p = self.take('Dh', i)))
            self.feature.add_module('Add_In'+str(i),nn.Linear(struct[i], struct[i+1]))
            self.feature.add_module('Activation'+str(i),self.F('h',i))
        
        self.output = nn.Sequential(nn.Linear(struct[-2],struct[-1]),
                                    self.F('o'))
        
    def Convolutional(self, conv_struct = None, is_drop = True):
        in_channel = self.img_size[0]
        if conv_struct is None:
            conv_struct = self.conv_struct
            
        self.conv = nn.Sequential()
        for i in range(conv_struct.shape[0]):
            row = conv_struct.loc[i].values
            if is_drop and (isinstance(self.conv_dropout,list) or self.conv_dropout > 0):
                self.conv.add_module('Dropout'+str(i),nn.Dropout(p = self.take('Dc', i)))
            ''' 
                head = ['conv_para', 'bn_type', 'pool_type', 'pool_para']
                conv_para: (in_channels(auto), out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
                bn_type: 0, 1
                pool_type: 0, 'Max', 'Avg', 'FractionalMax', 'AdaptiveMax', 'AdaptiveAvg'
                pool_para: (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
            ''' 
            conv_para = row[0].copy()
            conv_para.insert(0, in_channel)
            self.conv.add_module('Conv2d'+str(i),nn.Conv2d(*conv_para))
            if row[1] != 0:
                self.conv.add_module('BatchNorm2d'+str(i),nn.BatchNorm2d(conv_para[1]))
            if len(self.struct) == 0 and i == conv_struct.shape[0] - 1:
                self.conv.add_module('Activation'+str(i),self.F('o',i))
            else:
                self.conv.add_module('Activation'+str(i),self.F('c',i))
            
            if row[2] == '':
                if i == 0: pool_type = 'Max'
            else:
                pool_type = row[2]
            if row[3] != 0:
                if isinstance(row[3],int): pool_setting = 'row[3]'
                else: pool_setting = '*row[3]'
                pooling = pool_type + 'Pool2d'
                self.conv.add_module(pooling+str(i),eval('nn.'+pooling+'('+pool_setting+')'))
            in_channel = conv_para[1]
            
    def init_linear(self, init_w = 'xavier_normal_', init_b = 0):
        '''
            uniform_, normal_, constant_, ones_, zeros_, eye_, dirac_, 
            xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_
        '''
        def do_init(x,way):
            if isinstance(way, int):
                nn.init.constant_(x,way)
                return
            elif isinstance(way, list):
                setting = way[1] # 字符串
                way = way[0]
                eval('nn.init.'+way+'(x,'+setting+')')
            else:
                eval('nn.init.'+way+'(x)')
            
        def init_w_b(layer):
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight
                b = layer.bias
                do_init(w,init_w)
                do_init(b,init_b)
                
        self.apply(init_w_b)
                    
    def get_paras(self, name = None, prt = False):
        paras = []
        for named_para in self.named_parameters():
            if name is not None:
                if name in named_para[0]: 
                    paras.append(named_para[1])
                    if prt: print(named_para)
            else: 
                paras.append(named_para[1])
                if prt: print(named_para)
        if len(paras) == 1: paras = paras[0]
        return paras
           