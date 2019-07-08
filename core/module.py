# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from .load import Load
from .func import Func
from .epoch import Epoch
from pandas import DataFrame
import pandas as pd
import numpy as np
import os
import sys
sys.path.append('..')
from core.layer import Linear2
from core.func import _para
from core.plot import t_SNE, _save_img, _save_multi_img
from core.visual import Visual

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Module(torch.nn.Module,Load,Func,Epoch):
    
    def __default__(self, **kwargs):
        # default setting
        if 'img_size' in kwargs.keys(): flatten = False
        else: flatten = True
        default = {'flatten': flatten,
                   'unsupervised': False,
                   'msg': [],
                   'L': torch.nn.MSELoss(),
                   'dvc': device,
                   'best_acc': 0,
                   'best_rmse': float('inf'),
                   'task': 'cls'}
        # set module attr
        for key in default.keys():
            setattr(self, key, default[key])
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])
        # adjust
        if type(self.dvc) == str: self.dvc = torch.device(self.dvc)
        if hasattr(self, 'name') == False: self.name = self._name
            
    def __print__(self):
        #print module
        print()
        print(self)
        #print parameters
        print("{}'s Parameters(".format(self.name))
        for key, v in self.state_dict().items():print('  {}:\t{}'.format(key,v.size()))
        print(')')
        #print optimizer
        print("{}'s Optimizer: {}".format(self.name, self.optim))
    
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.__default__(**kwargs)
        self.kwargs = kwargs

        if self.task == 'cls':
            head = ['loss', 'accuracy']
            #self.L = torch.nn.CrossEntropyLoss()
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
            SGD,  Adam, RMSprop
            Adadelta, Adagrad, Adamax, SparseAdam, ASGD, Rprop, LBFGS
        '''
        if hasattr(self, 'optim'): _optim = self.optim
        else: _optim = 'Adam'

        if hasattr(self, 'l2'):  # L2 正则化
            weights, others = self._get_para()
            para = "[ \
            {'params': weights, 'weight_decay': self.l2}, \
            {'params': others, 'weight_decay':0} \
            ]"
        else:
            para = 'self.parameters()'
        if self.task == 'usp':
            if hasattr(self, 'pre_lr'): para += ',lr = self.pre_lr'
        else:
            if hasattr(self, 'lr'): para += ',lr = self.lr'
        if hasattr(self, 'optim_para'): para += ',' + self.optim_para
            
        if type(_optim) == str:
            self.optim  = eval('torch.optim.'+_optim+'('+para+')')
        if hasattr(self, 'decay_s'):
            self.scheduler = StepLR(self.optim, step_size=100, gamma=self.decay_s)
        elif hasattr(self, 'decay_r'):
            self.scheduler = ReduceLROnPlateau(self.optim, mode="min", patience=100, factor=self.decay_r)
            
        self.__print__()
    
    def Sequential(self, out_number = 1, weights = None, modules = None):
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
            
            # Dropout
            if hasattr(self,'dropout'):
                p = self.D('h', i)
                if p > 0: layers.append( nn.Dropout(p = p) )
            
            # Module
            if weights is not None:
                layers.append( Linear2(weights[i]) )
            elif modules is not None:
                layers.append( modules[i] )
            else:
                layers.append( nn.Linear(self.struct[i], self.struct[i+1]) )
            
            # Act
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
    
    def _save_load(self, do = 'save', stage = 'best', obj = 'para'):
        _para(self, do, stage, obj)
            
    def _init_para(self, init_w = 'xavier_normal_', init_b = 0):
        '''
            uniform_, normal_, constant_, ones_, zeros_, eye_, dirac_, 
            xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_
            default:
                W: truncated_normal(stddev=np.sqrt(2 / (size(0) + size(1))))
                b: constant(0.0)
        '''
        def do_init(x, init_ ):
            if init_ is None:
                return
            elif type(init_) == int:
                nn.init.constant_(x,init_)
            elif init_[-1] == ')':
                eval('nn.init.'+init_)
            else:
                eval('nn.init.'+init_+'(x)')
        
        for name, para in self.named_parameters():
            if 'weight' in name: do_init(para,init_w)
            if 'bias' in name: do_init(para,init_b) 
    
    def _get_para(self, para_name = 'weight', transpose = False):
        paras, others = [], []
        for name, para in self.named_parameters():
            if para_name in name:  
                if transpose:
                    paras.append(para.t())
                else:
                    paras.append(para)
            else: others.append(para)
        return paras, others
       
    def _plot_feature_tsne(self, data = 'train'):
        if hasattr(self, '_feature') == False: 
            return
        _para(self, 'load', 'best')
        if data == 'train':
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        Y = data_loader.dataset.tensors[1].cpu().numpy()
        self.eval()
        with torch.no_grad():
            X = self._feature(data_loader.dataset.tensors[0].cpu()).numpy()
        if not os.path.exists('../save/plot'): os.makedirs('../save/plot')
        path ='../save/plot/['+ self.name + '] _' + data + ' {best-layer'+str(len(self.struct)-2) + '}.png'
        t_SNE(X, Y, path)
            
    def _plot_weight(self):
        path = '../save/para/['+self.name + '] weights/'
        if not os.path.exists(path): os.makedirs(path)
        # scalar
        weights,_ = self._get_para()
        _min, _max = np.zeros(len(weights)), np.zeros(len(weights))
        for i in range(len(weights)):
            data = weights[i].data.cpu().numpy()
            _min[i], _max[i] = data.min(), data.max()
        _min = _min.min()
        _max = _max.max()
        # named_children 只返回最外层, named_modules 返回各层元素
        for (name, layer) in self.named_modules():
            if isinstance(layer, torch.nn.Linear):
                # 2d
                _save_img(layer.weight.data, [_min, _max], path + name)
            elif isinstance(layer, torch.nn.Conv2d):
                # 3d
                #print(layer.weight.data.size())
                data = layer.weight.data.cpu().numpy()
                _save_multi_img(data, data.shape[1], [_min, _max], path + name)
                
    def _visual_weight(self, layer_name = 'all', epoch = 30, reshape = None):
        if hasattr(self,'img_size'):
            input_dim = self.img_size
        else:
            input_dim = self.struct[0]
        vis = Visual(self,input_dim, layer_name, epoch = epoch, reshape = reshape)
        vis._weight()
        
    def _save_xlsx(self):
        # sheet_names
        if self.task == 'cls':
            sheet_names = ['model_info','epoch_curve','cls_result', 'FDR_FPR']
        else: 
            sheet_names = ['model_info','epoch_curve','prd_result']
        # model_info
        df1 = DataFrame({'keys': list(self.kwargs.keys()), 'vaules': list(self.kwargs.values())})
        # epoch_curve
        self.train_df.rename(columns=lambda x:'train_' + x, inplace=True)
        self.test_df.rename(columns=lambda x:'test_' + x, inplace=True)
        df2 = pd.concat([self.train_df, self.test_df], axis=1)
        df2.insert(0, 'Epoch', np.array(range(1,df2.shape[0] + 1)))
        # prd_result
        if self.task == 'prd':
            df3 = DataFrame({'real_Y': self.test_Y, 'pred_Y': self.pred_Y})
            dfs = [df1, df2, df3]
        # cls_result, FDR_FPR
        if self.task == 'cls':
            df3 = pd.concat( 
                    [DataFrame(self.pred_distrib[0], columns = self.categories_name),
                     DataFrame(self.pred_distrib[1], columns = self.categories_name)],
                     axis=0)
            df3.insert(0,'Categories',self.categories_name *2)
            df4 = DataFrame(self.FDR, columns = ['FDR', 'FPR'])
            df4.insert(0,'Categories',self.categories_name + ['Average'])
            dfs = [df1, df2, df3, df4]
        # writer
        writer = pd.ExcelWriter('../save/['+self.name+'] result.xlsx',engine='openpyxl')
        # save
        for i, sheet_name in enumerate(sheet_names):
            dfs[i].to_excel(excel_writer=writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()
        