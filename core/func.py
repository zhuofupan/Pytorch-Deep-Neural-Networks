# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn
from pandas import DataFrame

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

act_dict = {'r': 'ReLU',      's': 'Sigmoid',      't': 'Tanh',        'x': 'Softmax',      
            'r6': 'ReLU6',    'e': 'ELU',          'pr': 'PReLU',      'lr': 'LeakyReLU',
            'si': 'Softmin',  'sp': 'Softplus',    'sk': 'Softshrink', 'sn': 'Softsign',
            'ls': 'LogSigmoid','lx': 'LogSoftmax', 'ht': 'Hardtanh',    'tk': 'Tanhshrink', 
            'b': 'Threshold',
            }

class Func(object):
    def F(self, name, i = 0):
        if type(name) == list:
            name = self.take(name, i)
        # func in inquire_dict
        elif 'F' + name in inquire_dict.keys():
            lst = eval('self.'+inquire_dict['F' + name])
            name = self.take(lst, i)
        # func in act_dict
        elif name in act_dict.keys():
            name = act_dict[name]
  
        if name == 'Gaussian':
            func = Gaussian()
        elif name == 'Affine':
            func = Affine()
        elif name == 'Softmax':
            func = nn.Softmax(dim = 1)          
        elif name[-1] == ')':
            func = eval('nn.'+name)
        else:
            '''
                ReLU, ReLU6, ELU, PReLU, LeakyReLU, 
                Threshold, Hardtanh, Sigmoid, Tanh, LogSigmoid, 
                Softplus, Softshrink, Softsign, Tanhshrink, Softmin, Softmax, LogSoftmax
            '''
            func = eval('nn.'+name+'(inplace = True)')
        return func
    
    def take(self, lst, i = 0): 
        if type(lst) == list:
            out = lst[np.mod(i, len(lst))]
        # dropout
        elif lst in inquire_dict:
            lst = eval('self.'+inquire_dict[lst])
            out = lst[np.mod(i, len(lst))]
        else:
            out = lst
        return out
    
    def get_loss(self, output, target):
        if hasattr(self, 'loss'):
            # 在 forword 里自定义了损失值，调用get_loss前调用 forword 以获取损失值
            return self.loss
        else:
            return self.L(output, target)
        
    def get_rmse(self, output, target):
        return torch.sqrt(nn.functional.mse_loss(output, target))
    
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
        
        self.pred_distri = [pred_cnt, pred_cnt_pro]
        for i in range(self.n_category):
            self.FDR[i][0], self.FDR[i][1] = FDR[i], FPR[i]
        self.FDR[-1][0], self.FDR[-1][1] = self.best_acc, 1 - self.best_acc
        
    def statistics_number(self,target):
        if len(target.shape) > 1:
            self.n_category = target.shape[1]
        else:
            self.n_category = len(set(target))
            target = self.to_onehot(self.n_category, target)
        
        self.FDR = np.zeros((self.n_category + 1, 2))
        self.n_sample_cnts = np.sum(target, axis = 0, dtype = np.int)
        self.n_sample = np.sum(self.n_sample_cnts, dtype = np.int)
        
    def show(self):
        # best result
        print('\nShowing test result:')
        if self.task == 'cls':
            for i in range(self.n_category):
                print('Category {}:'.format(i))
                print('    >>> FDR = {:.2f}%, FPR = {:.2f}%'.format(self.FDR[i][0]*100,self.FDR[i][1]*100))
            print('The best test average accuracy is {:.2f}%'.format(self.FDR[-1][0]*100))
        else:
            print('The bset test rmse is {:.4f}, and the corresponding R2 is {:.4f}'.format(self.best_rmse, self.best_R2))
        # plot loss & acc cure / rmse & R2 cure
        # plot category distribution / pred & real curve