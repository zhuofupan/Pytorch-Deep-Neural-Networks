# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from .func import Func 
from .epoch import Epoch, _save_module
from .layer import Linear2
from ..data.load import Load
from ..visual.plot import t_SNE, _save_img, _save_multi_img, _get_categories_name
from ..visual.visual_weight import VisualWeight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Module(torch.nn.Module, Load, Func, Epoch):
    
    def __set_attr__(self, **kwargs):
        # default setting
        if 'img_size' in kwargs.keys(): flatten = False
        else: flatten = True
        default = {'dvc': device,               # 定义cpu/gpu训练
                   'flatten': flatten,          # 是否将数据扁平化（拉成1维向量）
                   'msg': [],                   # 用于训练中显示有关训练的info
                   'add_msg': '',               # 显示的额外自定义信息
                   'show_model_info': True,     # 是否在控制台打印模型信息
                   'save_module_para': True,    # 是否存储训练好的模型参数
                   'run_id': '',                # 用于Run_N中记录模型运行第几遍了
                   'n_category': None,          # 类别数目
                   'L': 'MSE',                  # 默认MSE为损失函数
                   'use_bias': True,            # Linear中是否使用bias
                   '__drop__': [True, True],    # 是否丢struct[0](input)和struct[-1](output)
                   'label_name': None,          # 标签的名字用于绘图
                   'best_acc': 0,               # 分类任务下的最好正确率
                   'best_rmse': float('inf'),   # 预测任务下的最小rmse
                   'best_mape': float('inf'),   # 数据补全任务下的最小mape
                   'task': 'cls'}               # 指定模型执行的任务 cls/prd/usp/gnr
        # set module attr
        for key in default.keys():
            setattr(self, key, default[key])
        for key in kwargs.keys(): 
            setattr(self, key, kwargs[key])
        # adjust 'dvc', 'name', 'L', 'loader_kwargs', 'struct'
        if type(self.dvc) == str: self.dvc = torch.device(self.dvc)
        if hasattr(self, 'name') == False: self.name = self._name
        '''
            L1, NLL (NLLLoss2d), KLDiv, MSE, BCE, BCEWithLogits, \
            CosineEmbedding, CTC, HingeEmbedding, MarginRanking, \
            MultiLabelMargin, MultiLabelSoftMargin, MultiMargin, \
            SmoothL1, SoftMargin, CrossEntropy, TripletMargin, PoissonNLL
        '''
        self.L = eval('torch.nn.'+self.L+'Loss()')
        if self.n_category is None and self.task == 'cls' and hasattr(self,'struct'):
            self.n_category = self.struct[-1]
        
        if self.dvc == torch.device('cpu'):
            self.loader_kwargs = {'pin_memory': False}
        else:
            self.loader_kwargs = {'pin_memory': True, 'num_workers': 0}
            
        for i in range(len(self.struct)):
            if i>0 and type(self.struct[i]) == str:
                self.struct[i] = int(eval('self.struct[i-1]' + self.struct[i]))
            
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
        
    def __watch__(self, size = None):
        if size is None:
            if hasattr(self, 'img_size'):
                size = [1] + self.img_size
            elif hasattr(self, 'struct'):
                size = [1] + [self.struct[0]]
        #import tensorwatch as tw
        #tw.draw_model(self, input_shape=size)
    
    def __init__(self, **kwargs):
        torch.nn.Module.__init__(self)
        self.__set_attr__(**kwargs)
        self.kwargs = kwargs
        self.kwargs['dvc'] = self.dvc

        if self.task == 'cls':
            head = ['loss', 'accuracy']
            #self.L = torch.nn.CrossEntropyLoss()
            self.label_name = _get_categories_name(self.label_name, self.struct[-1])
        elif self.task == 'prd':
            head = ['loss', 'rmse', 'R2']
        elif self.task == 'impu':
            head = ['loss', 'rmse', 'mape']
        else:
            head = ['loss']
            
        self.train_df = DataFrame(columns = head)
        self.test_df = DataFrame(columns = head)
        
    def __call__(self, **kwargs):
        return self.forward(**kwargs)
    
    def opt(self, parameters = None, info = True):
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
        elif parameters is not None:
            para = 'parameters'
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
        
        if info and self.show_model_info: self.__print__()
    
    def Sequential(self, 
                   out_number = 1,      # hidden 和 output 作为 1个 seq 输出，或作为 2个 输出
                   struct = None,       # struct 网络结构
                   hidden_func = 'h',   # 隐层激活函数
                   output_func = 'o',   # 输出层激活函数，可以为 None
                   dropout = 'h',       # 使用的 dropout
                   paras = None):       # 采用已有参数
        '''
            pre_setting: struct, dropout, hidden_func, output_func
        '''
        # print(struct, hidden_func, output_func)
        if struct is None:
            if hasattr(self, 'struct'):
                struct = self.struct
            else:
                print("Error: miss attr 'struct'!")
                return
        
        for i in range(len(struct)):
            # for convnet
            if i==0 and struct[0] == -1:
                size = self.para_df.iloc[-1,-1]
                struct[0] = size[0] * size[1] * size[2]
            # for all
            if i>0 and type(struct[i]) == str:
                struct[i] = int(eval('struct[i-1]' + struct[i]))
        
        hidden, output = [], []
        for i in range(len(struct)-1):
            if i < len(struct)-2: 
                layers = hidden
            else: 
                layers = output
                if self.__drop__[1] == False: dropout = None
            
            # Dropout
            if dropout is not None and hasattr(self,'dropout'):
                p = self.D(dropout, i)
                if self.__drop__[0] == False and i == 0: pass
                elif p > 0: layers.append( nn.Dropout(p = p) )
            
            # Module
            if paras is not None:
                para = paras[i]
                if type(para) == tuple: weight, bias = para
                else: weight, bias = para, None
                layers.append( Linear2(weight, bias, bias = self.use_bias))
            else:
                layers.append( nn.Linear(struct[i], struct[i+1], bias = self.use_bias))
            
            # Act
            if i < len(struct)-2:
                # hidden_func
                layers.append(self.F(hidden_func,i))
            elif isinstance(self.L, nn.CrossEntropyLoss):
                # 这时的 output 输出的是 logits
                pass 
            elif output_func is not None and hasattr(self,'output_func') and \
                self.output_func is not None:
                # output_func
                layers.append(self.F(output_func))
            else:
                # hidden_func
                layers.append(self.F(hidden_func,i))
 
        if out_number == 1: 
            return nn.Sequential(*(hidden + output))

        if len(hidden) == 1: hidden = hidden[0]
        else: hidden = nn.Sequential(*hidden)
        
        if len(output) == 1: output = output[0]
        else: output = nn.Sequential(*output)
        
        return hidden, output
    
    def _save_load(self, do = 'save', stage = 'best', obj = 'para', path = '../save/'):
        _save_module(self, do, stage, obj, path)
            
    def _init_para(self, para_name = 'weight', init = 'xavier_normal_'):
        '''
            uniform_, normal_, constant_, ones_, zeros_, eye_, dirac_, 
            xavier_uniform_, xavier_normal_, kaiming_uniform_, kaiming_normal_, orthogonal_, sparse_
            default:
                W: truncated_normal(stddev=np.sqrt(2 / (size(0) + size(1))))
                b: constant(0.0)
        '''
        def do_init(x):
            if init is None:
                return
            elif type(init) == int:
                nn.init.constant_(x,init)
            elif init[-1] == ')':
                eval('nn.init.'+init)
            else:
                eval('nn.init.'+init+'(x)')
        
        paras, _ = self._get_para(para_name)
        for para in paras:
            if 'weight' in para_name:
                if len(para.size()) > 1:
                    do_init(para) 
            else:
                do_init(para) 
            
    def _get_para(self, para_name = 'weight', transpose = False):
        paras, others = [], []
        print("Find {} in module's paras".format(para_name))
        if type(para_name) != list: para_name = [para_name]
        for name, para in self.named_parameters():
            add_para = True
            for i in para_name:
                if i not in name: 
                    add_para = False
                    break
            if add_para:
                print(name, para.size())
                if transpose:
                    paras.append(para.t())
                else:
                    paras.append(para)
            else:
                others.append(para)
        return paras, others
    
    def _plot_feature_tsne(self, data = 'train'):
        if hasattr(self, '_feature') == False: 
            return
        _save_module(self, 'load', 'best')
        if data == 'train':
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        Y = data_loader.dataset.tensors[1].cpu().numpy()
        self.eval()
        self = self.cpu()
        with torch.no_grad():
            X = self._feature(data_loader.dataset.tensors[0].cpu()).numpy()
        path ='../save/'+ self.name + self.run_id +'/'
        file_name = '['+ self.name + '] _' + data + ' (best-layer-'+str(len(self.struct)-2) + ').png'
        t_SNE(X, Y, path, file_name)
            
    def _plot_weight(self, item = 'both', _min_max = None):
        path = '../save/'+ self.name + self.run_id + '/['+self.name + ']/'
        if not os.path.exists(path): os.makedirs(path)
        # scalar
        weights,_ = self._get_para()
        _min, _max = np.zeros(len(weights)), np.zeros(len(weights))
        for i in range(len(weights)):
            data = weights[i].data.cpu().numpy()
            _min[i], _max[i] = data.min(), data.max()
        _min, _max = _min.min(), _max.max()
        if _min_max == True:
            _min_max = [_min, _max]   
        
        # named_children 只返回最外层, named_modules 返回各层元素
        for (name, layer) in self.named_modules():
            if isinstance(layer, torch.nn.Linear) and item in ['both', 'linear']:
                # 2d
                sys.stdout.write('\r'+ "Plot weight: {}".format(name))
                sys.stdout.flush()
                _save_img(layer.weight.data, _min_max, path + name)
            elif isinstance(layer, torch.nn.Conv2d) and item in ['both', 'conv']:
                # 3d
                #print(layer.weight.data.size())
                sys.stdout.write('\r'+ "Plot weight: {}".format(name))
                sys.stdout.flush()
                data = layer.weight.data.cpu().numpy()
                _save_multi_img(data, data.shape[1], _min_max, path + name)
        print()       
                
    def _visual(self, item = 'category', layer_name = 'all', filter_id = None, epoch = 30, reshape = None):
        if hasattr(self, 'img_size'):
            if reshape == True: 
                reshape = (self.img_size[1],self.img_size[2])
            input_dim = self.img_size
        else:
            input_dim = self.struct[0]
        vis = VisualWeight(self, input_dim, layer_name,  filter_id = filter_id, epoch = epoch, reshape = reshape)
        if item == 'both':
            vis._weight()
            vis._get_input_for_category()
        elif item ==  'weight':
            vis._weight()
        elif item ==  'category':
            vis._get_input_for_category()
        
    def _save_xlsx(self):
        # sheet_names
        if self.task == 'cls':
            sheet_names = ['model_info','epoch_curve','cls_result', 'FDR_FPR']
        elif self.task == 'prd': 
            sheet_names = ['model_info','epoch_curve','prd_result']
        elif self.task == 'impu': 
            sheet_names = ['model_info','epoch_curve','RMSE_MAPE']
        # model_info
        df1 = DataFrame({'keys': list(self.kwargs.keys()), 'vaules': list(self.kwargs.values())})
        # epoch_curve
        self.train_df.rename(columns=lambda x:'train_' + x, inplace=True)
        self.test_df.rename(columns=lambda x:'test_' + x, inplace=True)
        df2 = pd.concat([self.train_df, self.test_df], axis=1)
        df2.insert(0, 'Epoch', np.array(range(1,df2.shape[0] + 1)))
        
        dfs = [df1, df2]
        # prd_result
        if self.task == 'prd':
            df3 = DataFrame({'real_Y': self.test_Y.reshape(-1,), 'pred_Y': self.pred_Y.reshape(-1,)})
            dfs += [df3]
        # cls_result, FDR_FPR
        if self.task == 'cls':
            df3 = pd.concat( 
                    [DataFrame(self.pred_distrib[0], columns = self.label_name),
                     DataFrame(self.pred_distrib[1], columns = self.label_name)],
                     axis=0)
            df3.insert(0,'Categories',self.label_name *2)
            df4 = DataFrame(self.FDR, columns = ['FDR', 'FPR'])
            df4.insert(0,'Categories',self.label_name + ['Average'])
            dfs += [df3, df4]
        # cls_result, FDR_FPR
        if self.task == 'impu':
            df3 = DataFrame(np.concatenate([self.RMSE.reshape(-1,1),
                                            self.MAPE.reshape(-1,1), 
                                            self.train_loader.missing_var_rate.reshape(-1,1)], 1), 
                            columns = ['RMSE', 'MAPE', 'missing_rate'])
            df3.insert(0,'Variable', self.train_loader.is_missing_var + ['Average'])
            dfs += [df3]
            
        # writer
        if not os.path.exists('../save/'+ self.name + self.run_id): os.makedirs('../save/'+ self.name + self. run_id)
        path = '../save/' + self.name + self.run_id + '/['+self.name+'] result.xlsx'
        writer = pd.ExcelWriter(path, engine='openpyxl')
        # save
        for i, sheet_name in enumerate(sheet_names):
            dfs[i].to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()
        