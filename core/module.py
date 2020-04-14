# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

sys.path.append('..')
from data.load import Load
from core.epoch import Epoch
from core.func import Func, _save_module
from core.layer import Linear2
from visual.plot import t_SNE, _save_img, _save_multi_img
from visual.visual_weight import VisualWeight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Module(torch.nn.Module,Load,Func,Epoch):
    
    def __default__(self, **kwargs):
        # default setting
        if 'img_size' in kwargs.keys(): flatten = False
        else: flatten = True
        default = {'flatten': flatten,
                   'open_dropout': True,
                   'unsupervised': False,
                   'msg': [],
                   'L': 'MSE',
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
        '''
            L1Loss, NLLLoss, KLDivLoss, MSELoss, BCELoss, BCEWithLogitsLoss, NLLLoss2d, \
            CosineEmbeddingLoss, CTCLoss, HingeEmbeddingLoss, MarginRankingLoss, \
            MultiLabelMarginLoss, MultiLabelSoftMarginLoss, MultiMarginLoss, \
            SmoothL1Loss, SoftMarginLoss, CrossEntropyLoss, TripletMarginLoss, PoissonNLLLoss
        '''
        self.L = eval('torch.nn.'+self.L+'Loss()')
            
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
        
        if info: self.__print__()
    
    def Sequential(self, out_number = 1, 
                   weights = None, struct = None, hidden_func = 'h',
                   contain_logits = True):
        '''
            pre_setting: struct, dropout, hidden_func, output_func
        '''
        if struct is None:
            if hasattr(self, 'struct'):
                struct = self.struct
            else:
                print("Error: miss attr 'struct'!")
                return
        
        for i in range(len(struct)):
            if i==0 and struct[0] == -1:
                size = self.para_df.iloc[-1,-1]
                struct[0] = size[0] * size[1] * size[2]
            if i>0 and type(struct[i]) == str:
                struct[i] = int(eval('struct[i-1]' + struct[i]))
        
        hidden, output = [], []
        for i in range(len(struct)-1):
            if i < len(struct)-2: layers = hidden
            else: layers = output
            
            # Dropout
            if self.open_dropout and hasattr(self,'dropout'):
                p = self.D('h', i)
                if p > 0: layers.append( nn.Dropout(p = p) )
            
            # Module
            if weights is not None and weights[i] is not None:
                layers.append( Linear2(weights[i]) )
            else:
                layers.append( nn.Linear(struct[i], struct[i+1]) )
            
            # Act
            if i < len(struct)-2:
                layers.append(self.F(hidden_func,i))
            elif contain_logits and isinstance(self.L, nn.CrossEntropyLoss):
                pass # 这时的 output 输出的是 logits
            elif hasattr(self,'output_func'):
                layers.append(self.F('o'))
 
        if out_number == 1: 
            return nn.Sequential(*(hidden + output))

        if len(hidden) == 1: hidden = hidden[0]
        else: hidden = nn.Sequential(*hidden)
        
        if len(output) == 1: output = output[0]
        else: output = nn.Sequential(*output)
        
        return hidden, output
    
    def _save_load(self, do = 'save', stage = 'best', obj = 'para'):
        _save_module(self, do, stage, obj)
            
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
       
#    def _add_loss(self, pred = None, target = None, rule = None):
#        pred = list(pred)
    
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
        with torch.no_grad():
            X = self._feature(data_loader.dataset.tensors[0].cpu()).numpy()
        if not os.path.exists('../save/plot'): os.makedirs('../save/plot')
        path ='../save/plot/['+ self.name + '] _' + data + ' {best-layer'+str(len(self.struct)-2) + '}.png'
        t_SNE(X, Y, path)
            
    def _plot_weight(self, item = 'both', _min_max = None):
        path = '../save/para/['+self.name + ']/'
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
        