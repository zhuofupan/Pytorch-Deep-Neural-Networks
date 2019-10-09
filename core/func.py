# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
import sys
sys.path.append('..')
from core.plot import loss_acc_curve, rmse_R2_curve, pred_real_curve, category_distribution, _get_categories_name

class Gaussian(torch.nn.Module):
    def forward(self, x):
        return 1-torch.exp(-torch.pow(x,2))
class Affine(torch.nn.Module):
    def forward(self, x):
        return x

inquire_dict = {'Dh':'dropout',
                'Dc':'conv_dropout',
                
                'Fh':'hidden_func',
                'Fo':'output_func',
                'Fae':'act_func',
                'Fc':'conv_func'}

act_dict = {'s': 'Sigmoid',     't': 'Tanh',       'r': 'ReLU',        'e': 'ELU',     
            'r6': 'ReLU6',      'pr': 'PReLU',     'rr': 'RReLU',      'lr': 'LeakyReLU',
            'si': 'Softmin',  'sp': 'Softplus',    'sk': 'Softshrink', 'sn': 'Softsign',
            'ls': 'LogSigmoid','lx': 'LogSoftmax', 'ht': 'Hardtanh',   'tk': 'Tanhshrink', 
            'b': 'Threshold', 'a': 'Affine',       'g': 'Gaussian',    'x': 'Softmax',
            }

def _para(model = None, do = 'save', stage = 'best', obj = 'para'):
    if model is None:
        do, obj= 'load', 'model'
    if stage!= 'best' or do == 'load':
        print("\n{} [{}] 's {} in '{}'".format(do.capitalize(), model.name, obj, stage))
    if not os.path.exists('../save/para'): os.makedirs('../save/para')
    path = '../save/para/[{}] _{} _{}'.format(model.name, stage, obj) 

    if obj == 'para':
        if do == 'save': torch.save(model.state_dict(), path)
        else: model.load_state_dict(torch.load(path))
    elif obj == 'model':
        if do == 'save': torch.save(model, path)
        # model = access()
        else: return torch.load(path)
        
def get_func(lst, i = 0): 
    if type(lst) == list:
        name = lst[np.mod(i, len(lst))]
    else:
        name = lst
    # func in act_dict
    if name in act_dict.keys():
        name = act_dict[name]
  
    if name == 'Gaussian':
        func = Gaussian()
    elif name == 'Affine':
        func = Affine()
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
    
class Func(object):
    def F(self, lst, i = 0):
        if type(lst) == str and 'F' + lst in inquire_dict.keys():
            lst = eval('self.'+inquire_dict['F' + lst])
        return get_func(lst, i)
    
    def D(self, lst, i = 0): 
        if type(lst) == str and 'D' + lst in inquire_dict.keys():
            lst = eval('self.'+inquire_dict['D' + lst])
        if type(lst) == list:
            out = lst[np.mod(i, len(lst))]
        else:
            out = lst
        return out
    
    def is_cross_entropy(self, x):
        if hasattr(self, '_corss_entropy_softmax'):
            self._corss_entropy_in = x
            return self._corss_entropy_softmax(x)
        else:
            return x
    
    def get_loss(self, output, target):
        if hasattr(self, 'loss'):
            # 在 forword 里自定义了损失值，直接返回定义的损失值
            return self.loss
        else:
            if isinstance(self.L, nn.CrossEntropyLoss):
                target = target.argmax(1).long()
                output = self._corss_entropy_in
            loss = self.L(output, target)
        if hasattr(self, '_loss') and self.training:
            # 在 forword 里自定义了附加损失值，加上附加的损失值
            loss += self._loss 
        return loss
        
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
            target = self.to_onehot(self.n_category, target)
        
        self.FDR = np.zeros((self.n_category + 1, 2))
        self.n_sample_cnts = np.sum(target, axis = 0, dtype = np.int)
        self.n_sample = np.sum(self.n_sample_cnts, dtype = np.int)
    
    def result(self, categories_name = None):
        # best result
        print('\nShowing test result:')
        if self.task == 'cls':
            self.categories_name = _get_categories_name(categories_name, self.n_category)
            for i in range(self.n_category):
                print('Category {}:'.format(i))
                print('    >>> FDR = {}%, FPR = {}%'.format(self.FDR[i][0],self.FDR[i][1]))
            print('The best test average accuracy is {}%\n'.format(self.FDR[-1][0]))
            loss_acc_curve(self.train_df, self.test_df, self.name)
            category_distribution(self.pred_distrib[0], self.categories_name, self.name)
        else:
            print('The bset test rmse is {:.4f}, and the corresponding R2 is {:.4f}\n'.format(self.best_rmse, self.best_R2))
            rmse_R2_curve(self.train_df, self.test_df, self.name)
            pred_real_curve(self.pred_Y, self.test_Y, self.name)
        print("Save ["+self.name+"] 's test results")
        self._save_xlsx()