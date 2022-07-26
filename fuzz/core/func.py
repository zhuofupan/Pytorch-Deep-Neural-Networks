# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
from pandas import DataFrame

from fuzz.visual.plot import loss_curve, loss_acc_curve, rmse_R2_curve, rmse_mape_curve,\
    pred_real_curve, var_impu_curve, stat_curve, category_distribution, _check_label_name
from fuzz.data.gene_dynamic_data import to_onehot
from fuzz.core.act import Gaussian, Affine, Square, Exp_F, Exp_sq, Ln_sq, Bottom_F, \
    Oscillation_F, Ex0_sqd, Ex0_sqt, Ex1_2e, Ex1_gaus

inquire_dict = {'Dh':'dropout',
                'Dc':'conv_dropout',
                'Fh':'hidden_func',
                'Fo':'output_func',
                'Fc':'conv_func'}

act_additon = {'g': 'Gaussian',    'a': 'Affine',        'q': 'Square',    
               'ex': 'Exp_F',      'e2':'Exp_sq',        'l2':'Ln_sq',
               'b':'Bottom_F',     'o': 'Oscillation_F', '0d': 'Ex0_sqd',  
               '0t': 'Ex0_sqt',    '1e': 'Ex1_2e',       '1s': 'Ex1_gaus'
               }

act_dict = {'ts': 'Threshold',  'r': 'ReLU',        'ht': 'Hardtanh',   'r6': 'ReLU6',
            's': 'Sigmoid',     't': 'Tanh',        'x': 'Softmax',     'x2': 'Softmax2d',
            'lx': 'LogSoftmax', 'e': 'ELU',         'se': 'SELU',       'ce': 'CELU',
            'hs': 'Hardshrink', 'lr': 'LeakyReLU',  'ls': 'LogSigmoid', 'sp': 'Softplus',
            'ss': 'Softshrink', 'ma': 'MultiheadAttention', 'pr': 'PReLU', 'sn': 'Softsign',
            'si': 'Softmin',    'tk': 'Tanhshrink', 'rr': 'RReLU',      'gl': 'GLU'                
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
        if self.task not in ['usp'] and torch.isnan(target).int().sum() > 0:
            # 标签中存在缺失值
            dim = target.size(1)
            loc = torch.where(target == target)
            output, target = output[loc], target[loc]
            output, target = output.reshape(-1, dim), target.reshape(-1, dim)
        
        if hasattr(self, 'loss') and self.loss is not None:
            # 在 forword 里自定义了损失值，直接返回定义的损失值
            loss = self.loss
        elif isinstance(self.L, nn.CrossEntropyLoss):
            # 这里的 output = logits
            loss = self.cross_entropy_loss(output, target)
            # 损失为 CEL 时，将 output修正为 softmax(logits)
            output = torch.softmax(output, dim = 1)
        else:
            loss = self.L(output, target)
            
        if hasattr(self, '_loss') and self.training:
            # 在 forword 里自定义了附加损失值，加上附加的损失值
            loss += self._loss 
            
        return output, loss
    
    def cross_entropy_loss(self, logits, target):
        target = torch.argmax(target,1).long()
        return nn.functional.cross_entropy(logits, target)
    
    def get_impu_eva(self, X, Y, NAN):
        loc = np.where(NAN == 1)
        pred_Y, test_Y = X[loc], Y[loc]
        rmse, mape = self.get_rmse(pred_Y, test_Y), self.get_mape(pred_Y, test_Y)
        return rmse, mape
    
    def get_var_impu_eva(self, X, Y, NAN):
        v = self.train_loader.is_missing_var
        self.RMSE, self.MAPE = np.zeros(len(v) + 1), np.zeros(len(v) + 1)
        for i, index in enumerate(v):
            x, y, nan = X[:,index], Y[:,index], NAN[:,index]
            loc = np.where(nan == 1)
            _x, _y = x[loc], y[loc]
            self.RMSE[i], self.MAPE[i] = self.get_rmse(_x, _y), self.get_mape(_x, _y)
        self.RMSE[-1], self.MAPE[-1] = self.best_rmse, self.best_mape
    
    def get_mape(self, output, target):
        delta = output - target
        loc = np.where(target == 0)
        target[loc] = 1e-6
        error = np.abs(delta/target)
        # max_error = 1
        # error = np.clip(error, 0, max_error)
        return np.mean(error) * 100         
    
    def get_rmse(self, output, target):
        mse = np.mean((output - target)**2)
        return np.sqrt(mse)
    
    def get_R2(self, output, target):
        total_error = np.sum(np.power(target -  np.mean(target),2))
        unexplained_error = np.sum(np.power(target - output,2))
        R_squared = 1 - unexplained_error/ total_error
        return R_squared
    
    def get_accuracy(self, output, target):
        # print(output.shape, target.shape)
        if np.isnan(target).astype(int).sum() > 0:
            # 标签中存在缺失值
            dim = target.shape[1]
            loc = np.where(target == target)
            target = target[loc]
            output, target = output.reshape(-1, dim), target.reshape(-1, dim)
            
        if len(target.shape)>1:
            output_arg = np.argmax(output,1)
            target_arg = np.argmax(target,1)
        else:
            output_arg = np.array(output + 0.5, dtype = np.int)
            target_arg = np.array(target, dtype = np.int)
        
        return np.mean(np.equal(output_arg, target_arg).astype(np.float)) * 100
    
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
        pred_cnt_pro = pred_cnt / self.n_sample_cnts * 100
        # array是一个1维数组时，形成以array为对角线的对角阵；array是一个2维矩阵时，输出array对角线组成的向量
        FDR = np.diag(pred_cnt_pro)
        FPR = [(self.n_sample_cnts[i]-pred_cnt[i][i])/
               (self.n_sample-self.n_sample_cnts[i]) * 100 for i in range(self.n_category)]
        
        self.pred_distrib = [pred_cnt, np.around(pred_cnt_pro, 2)]
        for i in range(self.n_category):
            self.FDR[i][0], self.FDR[i][1] = FDR[i], FPR[i]
        self.FDR[-1][0], self.FDR[-1][1] = self.best_acc, 100 - self.best_acc
        self.FDR = np.around(self.FDR, 2)
        
    def statistics_number(self,target):
        if len(target.shape) > 1:
            self.n_category = target.shape[1]
        else:
            self.n_category = len(set(target))
            target = to_onehot(self.n_category, target)
        
        self.FDR = np.zeros((self.n_category + 1, 2))
        self.n_sample_cnts = np.sum(target, axis = 0, dtype = np.int)
        self.n_sample = np.sum(self.n_sample_cnts, dtype = np.int)
    
    def result(self, label_name = None, _plot = True):
        # label_name
        if label_name is None and hasattr(self, 'label_name'):
            label_name = self.label_name
        else:
            self.label_name = label_name
        # plot
        print('\nShowing test result:')
        if self.task == 'cls':
            self.label_name = _check_label_name(label_name, self.n_category)
            self.get_FDR(self.best_pred, self.test_Y)
            for i in range(self.n_category):
                print('Category {}:'.format(i))
                print('    >>> FDR = {}%, FPR = {}%'.format(self.FDR[i][0],self.FDR[i][1]))
            print('The best test average accuracy is {}%\n'.format(self.FDR[-1][0]))
            if _plot:
                loss_acc_curve(self.train_df, self.test_df, self.name, self.add_info + self.run_id)
                category_distribution(self.pred_distrib[0], self.label_name, \
                                      self.name, self.add_info + self.run_id)
        elif self.task == 'prd':
            print('The best test rmse is {:.4f}, and the corresponding R2 is {:.4f}\n'.\
                  format(self.best_rmse, self.best_R2))
            if _plot:
                rmse_R2_curve(self.train_df, self.test_df, self.name, self.add_info + self.run_id)
                pred_real_curve(self.pred_Y, self.test_Y, self.name, self.add_info + self.run_id, 'prd')
        elif self.task == 'impu':
            d = self.train_loader
            X, Y, NAN = self.best_pred, d.Y.data.numpy(), d.nan.data.numpy()
            self.get_var_impu_eva(X, Y, NAN)
            # save 'csv' for plot
            d.X = torch.from_numpy(X)
            d.save_best_impu_result(self.name + self.add_info + self.run_id) 
            for i, index in enumerate(d.is_missing_var):
                print('Variable {}:'.format(index + 1))
                print('    >>> RMSE = {:.4f}, MAPE = {:.2f}%, missing_rate = {:.2f}%'.\
                      format(self.RMSE[i], self.MAPE[i], d.missing_var_rate[i]))
            print('The best test rmse is {:.4f}, and the best test mape is {:.2f}%'.\
                  format(self.best_rmse, self.best_mape))
            if _plot:
                rmse_mape_curve(self.train_df, self.name, self.add_info + self.run_id)
                var_impu_curve(X, Y, NAN, d.is_missing_var, self.name, self.add_info + self.run_id)
        elif self.task == 'fd':
            if _plot:
                __subplot__ = False
                if hasattr(self, '__subplot__'): __subplot__ = self.__subplot__
                try:
                    loss_curve(self.train_df, None, self.name, self.add_info + self.run_id)
                except ValueError: pass 
                for i, key in enumerate (self.stat_lists.keys()):
                    stat_list = self.stat_lists[key]
                    stat_curve(stat_list, self.switch_p_list, self.fd_thrd[i], 
                               self.name, 
                               self.add_info + self.run_id + '_(' +str(i+1) + ')', 
                               label_name, 
                               self.Stat.plot_p_list,
                               __subplot__)

        # loss_curve
        if len(self.var_msg) > 0:
            loss_curve(self.train_df, self.var_msg, self.name, self.add_info + self.run_id)
        # xlsx
        print("Save "+self.name+"'s test results in '/save/{}{}{}'".format(self.name, self.add_info, self.run_id))
        self._save_xlsx()