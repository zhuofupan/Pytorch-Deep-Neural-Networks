# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
import sys
sys.path.append('..')
from visual.plot import loss_acc_curve, rmse_R2_curve, pred_real_curve, category_distribution, _get_categories_name
from data.gene_dynamic_data import to_onehot

class GaussianFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): 
        ctx.save_for_backward(x)
        return 1-torch.exp(-torch.pow(x,2))                      
    @staticmethod                            
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_in = grad_out * 2*x*torch.exp(-x*x)
        return grad_in

class Gaussian(torch.nn.Module):
    def forward(self, x):
        return GaussianFunction.apply(x)
        
class Affine(torch.nn.Module):
    def forward(self, x):
        return x * 1.0

inquire_dict = {'Dh':'dropout',
                'Dc':'conv_dropout',
                
                'Fh':'hidden_func',
                'Fo':'output_func',
                'Fc':'conv_func'}

act_additon = {'g': 'Gaussian',    'a': 'Affine',      }

act_dict = {'ts': 'Threshold',  'r': 'ReLU',        'ht': 'Hardtanh',   'r6': 'ReLU6',
            's': 'Sigmoid',     't': 'Tanh',        'x': 'Softmax',     'x2': 'Softmax2d',
            'lx': 'LogSoftmax', 'e': 'ELU',         'se': 'SELU',       'ce': 'CELU',
            'hs': 'Hardshrink', 'lr': 'LeakyReLU',  'ls': 'LogSigmoid', 'sp': 'Softplus',
            'ss': 'Softshrink', 'ma': 'MultiheadAttention', 'pr': 'PReLU', 'sn': 'Softsign',
            'si': 'Softmin',    'tk': 'Tanhshrink', 'rr': 'RReLU',      'gl': 'GLU',                
            }
act_dict = dict(act_dict, **act_additon)
        
def get_func(_var, i = 0): 
    if type(_var) == list:
        name = _var[np.mod(i, len(_var))]
    else:
        name = _var
    # func in act_dict
    if name in act_dict.keys():
        name = act_dict[name]
  
    if name in act_additon.values():
        func = eval(name + '()')
    elif name in ['Softmax','LogSoftmax']:
        func = nn.Softmax(dim = 1)          
    elif name[-1] == ')':
        func = eval('nn.'+name)
    else:
        '''
            ReLU, ReLU6, ELU, PReLU, LeakyReLU, 
            Threshold, Hardtanh, Sigmoid, Tanh, LogSigmoid, 
            Softplus, Softshrink, Softsign, Tanhshrink, Softmin, Softmax, LogSoftmax
        '''
        try:
            func = eval('nn.'+name+'(inplace = True)')
        except TypeError:
            func = eval('nn.'+name+'()')
    func.is_func = True
    return func

def find_act(module):
    Act = act_dict.values()
    act = None
    for act_str in Act:
        act_name = act_str
        if act_str not in act_additon.values():
            act_name = 'nn.'+act_str
        if isinstance(module, eval(act_name)):
            return act_str
    return act

class Func(object):
    def F(self, _var, i = 0):
        if type(_var) == str and 'F' + _var in inquire_dict.keys():
            _var = eval('self.'+inquire_dict['F' + _var])
        return get_func(_var, i)
    
    def D(self, _var, i = 0): 
        if type(_var) == str and 'D' + _var in inquire_dict.keys():
            _var = eval('self.'+inquire_dict['D' + _var])
        if type(_var) == list:
            value = _var[np.mod(i, len(_var))]
        else:
            value = _var
        return value
    
    def get_loss(self, output, target):
        if hasattr(self, 'loss'):
            # 在 forword 里自定义了损失值，直接返回定义的损失值
            loss = self.loss
        elif isinstance(self.L, nn.CrossEntropyLoss):
            # 这里的 output = logits
            loss = self.cross_entropy_loss(output, target)
        else:
            loss = self.L(output, target)
        if hasattr(self, '_loss') and self.training:
            # 在 forword 里自定义了附加损失值，加上附加的损失值
            loss += self._loss 
        
        if isinstance(self.L, nn.CrossEntropyLoss):
            # 损失为 CEL 时，将 output修正为 softmax(logits)
            output = nn.functional.softmax(output, 1, _stacklevel=5)
        return output, loss
    
    def cross_entropy_loss(self, logits, target):
        target = target.argmax(1).long()
        return self.L(logits, target)
        
    def get_rmse(self, output, target):
        mse = np.mean((output - target)**2)
        return np.sqrt(mse)
    
    def get_R2(self, output, target):
        total_error = np.sum(np.power(target -  np.mean(target),2))
        unexplained_error = np.sum(np.power(target - output,2))
        R_squared = 1 - unexplained_error/ total_error
        return R_squared
    
    def get_accuracy(self, output, target):
        if len(target.shape)>1:
            output_arg = np.argmax(output,1)
            target_arg = np.argmax(target,1)
        else:
            output_arg = np.array(output + 0.5, dtype = np.int)
            target_arg = np.array(target, dtype = np.int)
        
        return np.mean(np.equal(output_arg, target_arg).astype(np.float))
    
    def get_FDR(self, output, target):
        '''
            正分率:
            FDR_i = pred_cnt[i][i] / n_sample_cnts[i]
            
            误分率:
            FPR_i = ∑_j(pred_cnt[i]),j ≠ i / ∑_j(n_sample_cnts),j ≠ i
        '''
        if hasattr(self,'FDR') == False:
            self.statistics_number(target)
        if len(target.shape) > 1:
            output_arg = np.argmax(output,1)
            target_arg = np.argmax(target,1)
            
        pred_cnt = np.zeros((self.n_category, self.n_category))
        for i in range(self.n_sample):
            # 第 r 号分类 被 分到了 第 p 号分类
            p = output_arg[i]
            r = target_arg[i]
            pred_cnt[p][r] += 1
        pred_cnt_pro = pred_cnt / self.n_sample_cnts
        # array是一个1维数组时，形成以array为对角线的对角阵；array是一个2维矩阵时，输出array对角线组成的向量
        FDR = np.diag(pred_cnt_pro)
        FPR = [(self.n_sample_cnts[i]-pred_cnt[i][i])/
               (self.n_sample-self.n_sample_cnts[i]) for i in range(self.n_category)]
        
        self.pred_distrib = [pred_cnt, np.around(pred_cnt_pro*100, 2)]
        for i in range(self.n_category):
            self.FDR[i][0], self.FDR[i][1] = FDR[i], FPR[i]
        self.FDR[-1][0], self.FDR[-1][1] = self.best_acc, 1 - self.best_acc
        self.FDR = np.around(self.FDR*100, 2)
        
    def statistics_number(self,target):
        if len(target.shape) > 1:
            self.n_category = target.shape[1]
        else:
            self.n_category = len(set(target))
            target = to_onehot(self.n_category, target)
        
        self.FDR = np.zeros((self.n_category + 1, 2))
        self.n_sample_cnts = np.sum(target, axis = 0, dtype = np.int)
        self.n_sample = np.sum(self.n_sample_cnts, dtype = np.int)
    
    def result(self, categories_name = None):
        if hasattr(self, 'n_category') == False:
            return
        # best result
        print('\nShowing test result:')
        if self.task == 'cls':
            self.categories_name = _get_categories_name(categories_name, self.n_category)
            for i in range(self.n_category):
                print('Category {}:'.format(i))
                print('    >>> FDR = {}%, FPR = {}%'.format(self.FDR[i][0],self.FDR[i][1]))
            print('The best test average accuracy is {}%\n'.format(self.FDR[-1][0]))
            loss_acc_curve(self.train_df, self.test_df, self.name + self.run_id)
            category_distribution(self.pred_distrib[0], self.categories_name, self.name + self.run_id)
        else:
            print('The bset test rmse is {:.4f}, and the corresponding R2 is {:.4f}\n'.format(self.best_rmse, self.best_R2))
            rmse_R2_curve(self.train_df, self.test_df, self.name)
            pred_real_curve(self.pred_Y, self.test_Y, self.name)
        print("Save ["+self.name+"] 's test results")
        self._save_xlsx()