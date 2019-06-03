# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from .load import Load
from .func import Func
from .epoch import Epoch
from pandas import DataFrame

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Module(torch.nn.Module,Load,Func,Epoch):
    
    def __default__(self, **kwargs):
        # default setting
        if 'img_size' in kwargs.keys(): flatten = False
        else: flatten = True
        default = {'flatten': flatten,
                   'unsupervised': False,
                   'L': torch.nn.MSELoss(),
                   'msg': [],
                   'dvc': device,
                   'best_acc': 0,
                   'best_rmse': float('inf'),
                   'task': 'cls'}
        for key in default.keys():
            setattr(self, key, default[key])
            
    def __print__(self):
        #print module
        print()
        print(self)
        #print parameters
        print("{}'s Parameters(".format(self.name))
        for key, v in self.state_dict().items():print('  {}:\t{}'.format(key,v.size()))
        print(')')
    
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.kwargs = kwargs
        self.__default__(**kwargs)
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
        else:
            setting = 'self.parameters()'
            if hasattr(self, 'lr'): setting += ',lr = self.lr'
            if hasattr(self, 'l2'): setting += ',weight_decay = self.l2'
        if type(optim) == str:
            self.optim  = eval('torch.optim.'+optim+'('+setting+')')
        self.__print__()
    
    def Sequential(self, out_number = 1):
        '''
            pre_setting: struct, dropout, hidden_func, output_func
        '''
        if len(self.struct) == 0: return
            
        if self.struct[0] == -1:
            size = self.para_df.iloc[-1,-1]
            self.struct[0] = size[0] * size[1] * size[2]
        
        features, outputs = [], []
        for i in range(len(self.struct)-1):
            if i < len(self.struct)-2: layers = features
            else: layers = outputs
            
            if hasattr(self,'dropout') and i>0:
                layers.append( nn.Dropout(p = self.D('h', i)) )
                
            layers.append( nn.Linear(self.struct[i], self.struct[i+1]) )
            if i < len(self.struct)-2:
                layers.append(self.F('h',i))
            elif hasattr(self,'output_func'):
                layers.append(self.F('o',i))
 
        if out_number == 1: 
            features += outputs
        else:   
            if len(outputs) == 1: outputs = outputs[0]
            else: outputs = nn.Sequential(*outputs)
        
        if len(features) == 1: features = features[0]
        else: features = nn.Sequential(*features)
        
        if out_number == 1: 
            return features
        else:
            return features, outputs
            
            
    def init_seq(self, init_w = 'xavier_normal_', init_b = 0):
        '''
            uniform_, normal_, constant_, ones_, zeros_, eye_, dirac_, 
            xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_
            default:
                W: truncated_normal(stddev=np.sqrt(2 / (size(0) + size(1))))
                b: constant(0.0)
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
       