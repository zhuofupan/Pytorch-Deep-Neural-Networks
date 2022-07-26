# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:45:28 2022

@author: Fuzz4
"""
import numpy as np
from scipy.stats import norm, f, chi2
from fuzz.core.fd_statistics import Statistics

class PCA(Statistics):
    def __init__(self, **kwargs):
        default = {'fdi': 'res',                
                   'test_stat': 'T2',           # 统计指标 T2, GLR_VAE
                   'n_components': 2,           # 主元个数
                   'confidence': 1 - 0.005,     # 置信度
                   'if_use_chi2': True,         # 用卡方或者F分布
                   'if_times_sigma': True,      # 是否乘以sigma[-1]
                   'if_recal_test_stat': False, # 是否重新计算测试统计
                   
                   'thrd_sele': 'pdf',          # 阈值确定方法 pdf, ineq
                   'expt_FAR':  0.005,          # 预期的误报率
                   'esti_error': 0.01,          # 容许估计误差
                   'if_minus_mean': True,       # 算T2时是否减均值
                   '__interpolation__': False,  # 是否用插值计算T2
                   'kde_isf': [1,1],            # 估计cdf和isf的方法
                   
                   'split_p_list': None,        # fault 分割点
                   'plot_p_list': None,         # plot 分割点
                   '__subplot__': False,        # 绘图时是否用子视图
                   'label_name': None           # 绘图时用的标签
                   } 
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
        
        self.delta = 1 - self.confidence
        self.name = self.__class__.__name__
    
    # train PCA
    def fit(self, train_X):
        N = train_X.shape[0]
        # self.mean, self.var = np.mean(train_X, axis = 0), np.var(train_X, axis = 0)*self.N/(self.N-1)
        # X = (train_X - self.mean) / np.sqrt(self.var)
        X = train_X
        
        Cov_X = X.T @ X / (N - 1)
        self.sigma, self.P = np.linalg.eig(Cov_X)

        # P 是正交矩阵 P.T @ P = I; P^{-1} = P.T
        # P_pc (m × n_components) 负载矩阵
            
        sigma_pc = self.sigma[:self.n_components]
        self.P_pc = self.P[:,:self.n_components]
        
        # T_pc = X @ P_pc (N × n_components) 主元子空间;
        # /hat X = T_pc @ P_pc.T = X @ P_pc @ P_pc.T 得分矩阵
        # E = X - /hat X = X - X @ P_pc @ P_pc.T 残差子空间
 
        self.inv_sigma_pc = np.diag(1/sigma_pc)

        # np.set_printoptions(formatter={'float_kind':"{:.2e}".format})
        # print('Sigma = ', self.sigma)
        _, Recon_X = self.transform(train_X)
        print('Recon error = {:.4f}'.format(np.mean((X - Recon_X)**2)))
    
    # forward propagation
    def transform(self, test_X):
        # X = (test_X - self.mean) / np.sqrt(self.var)
        X = test_X
        T_pc = X @ self.P_pc
        Recon_X = T_pc @ self.P_pc.T
        return T_pc, Recon_X
    
    # get test statistics
    def _get_stat_vec(self, test_X, phase = 'offline'):
        # X = (test_X - self.mean) / np.sqrt(self.var)
        X = test_X
        m = test_X.shape[1]
        
        P_res = self.P[:,self.n_components:]
        sigma_res = self.sigma[self.n_components:]
        S_res = np.diag(sigma_res)
        S_res_inv = np.diag(1/sigma_res)
        if self.if_times_sigma:
            S_res = np.diag(sigma_res/self.sigma[-1])
            S_res_inv = np.diag(self.sigma[-1]/sigma_res)
        
        T_pc, Recon_X = self.transform(X)
        if self.fdi == 'res': FDI = X - Recon_X
        elif self.fdi == 'lv': FDI = T_pc
        elif self.fdi == 'lv_res': FDI = X @ P_res
        
        stat_vec = []
        # get test stat (by eig)
        if self.if_recal_test_stat == False:
            if self.fdi == 'res' and 'SPE' in self.test_stat:
                M = np.eye(test_X.shape[1]) - self.P_pc @ self.P_pc.T
                stat_vec.append( (X[:,np.newaxis,:] @ M @ X[:,:,np.newaxis]).reshape(-1,) )
            if self.fdi == 'res' and 'T2' in self.test_stat:
                R = np.eye(m) - self.P_pc @ self.P_pc.T
                M = R @ np.linalg.inv( P_res @ S_res @ P_res.T ) @ R.T
                stat_vec.append( (X[:,np.newaxis,:] @ M @ X[:,:,np.newaxis]).reshape(-1,) )
            if self.fdi == 'lv' and 'T2' in self.test_stat:
                M = self.P_pc @ self.inv_sigma_pc @ self.P_pc.T
                stat_vec.append( (X[:,np.newaxis,:] @ M @ X[:,:,np.newaxis]).reshape(-1,) )
            if self.fdi == 'lv_res' and 'T2' in self.test_stat:
                M = P_res @ S_res_inv @ P_res.T
                stat_vec.append( (X[:,np.newaxis,:] @ M @ X[:,:,np.newaxis]).reshape(-1,) )
            if self.fdi in ['lv','lv_res']  and 'SPE' in self.test_stat and self.thrd_sele != 'pdf':
                stat_vec.append( (FDI[:,np.newaxis,:] @ FDI[:,:,np.newaxis]).reshape(-1,) )
        
        # get test stat (recalculate)
        if len(stat_vec) == 0 or self.if_recal_test_stat:
            if phase == 'offline':
                self._Statistics__offline_stat(FDI)
            return self._Statistics__online_stat(FDI)
        
        return stat_vec
    
    def offline_modeling(self, train_X):
        # train
        self.fit(train_X)
        N, m, l = train_X.shape[0], train_X.shape[1], self.n_components
        _alf = 0
        
        self.fd_thrd = []
        # set threshold (with pdf)
        if self.thrd_sele == 'pdf':
            if self.fdi == 'res' and 'SPE' in self.test_stat:
                sigma_res = self.sigma[self.n_components:]
                theta = np.zeros(3)
                for i in range(3):
                    for j in range(len(sigma_res)):
                        theta[i] += np.power(sigma_res[j], i+1)
                h0 = 1 - 2*theta[0]*theta[2]/(3 * theta[1]**2 )
                # isf: 1-cdf 的反函数
                _alf = norm.isf(self.delta)
                self.fd_thrd.append( theta[0] * np.power( _alf*np.sqrt(2* theta[1] * h0**2)/theta[0] + 1 + \
                                                     theta[1]*h0*(h0-1)/theta[0]**2, 1/h0 ) )
            if self.fdi == 'res' and 'T2' in self.test_stat:
                _alf = chi2.isf(self.delta, m)
                fd_thrd = _alf
                if self.if_times_sigma:
                    fd_thrd = self.sigma[-1] * fd_thrd 
                self.fd_thrd.append( fd_thrd ) 
                
            if self.fdi == 'lv' and 'T2' in self.test_stat:
                if self.if_use_chi2:
                    _alf = chi2.isf(self.delta, l)
                    fd_thrd = _alf
                else:
                    _alf = f.isf(self.delta, l, N - l)
                    fd_thrd = l * (N**2 - 1) / (N * (N-l)) * _alf
                self.fd_thrd.append( fd_thrd ) 
                
            if self.fdi == 'lv_res' and 'T2' in self.test_stat:
                if self.if_use_chi2:
                    _alf = chi2.isf(self.delta, m - l)
                    fd_thrd = _alf
                else:
                    _alf = f.isf(self.delta, l, N - m + l)
                    fd_thrd = (m - l) * (N**2 - 1)/ (N * (N - m + l)) * _alf
                if self.if_times_sigma:
                    fd_thrd = self.sigma[-1] * self.fd_thrd
                self.fd_thrd.append( fd_thrd ) 
                
            if len(self.fd_thrd) == 0: 
                self.thrd_sele = 'ineq'
            else:
                print('Mean of stat: {:.2f}, Std of stat: {:.2f}, Threshold = {:.2f}'\
                  .format(np.mean(self.stat_vec_off), np.std(self.stat_vec_off), self.fd_thrd[0]))
        
        # set threshold (without pdf)
        if self.thrd_sele != 'pdf':
            self.stat_vec_off = self._get_stat_vec(train_X)
            self.fd_thrd = np.zeros(len(self.stat_vec_off))
            for i in range(len(self.stat_vec_off)):
                # threshold
                if self.thrd_sele == 'ineq':
                    self.fd_thrd[i] = self._get_thrd_ineq(i)
                elif self.thrd_sele == 'kde':
                    self.fd_thrd[i] = self._get_thrd_kde(i)
                      
        print("Thrd_sele = '{}', FDI = '{}', Test_stat = '{}', alf = {:.2f}".format(\
               self.thrd_sele, self.fdi, self.test_stat, _alf))
        return self.fd_thrd
        
    def online_monitoring(self, test_X, label_Y):
        self.stat_vec_on = self._get_stat_vec(test_X, phase = 'online')
        return self._get_decision(label_Y)

if __name__ == '__main__':
    from fuzz.data.gene_dynamic_data import ReadData
    path = '../../data/CSTR/fd'
    dataset_name = 'FD_CSTR'
    dynamic = 1
    n_components = 3
    datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                        is_del = False, example = 'CSTR', single_mode = True).datasets
    labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
              'Fault 08', 'Fault 09', 'Fault 10']
    train_X, train_Y, test_X, test_Y = datasets
    pca = PCA(n_components = n_components)
    for i in range(10):
        pca.n_components = i+1
        pca.fit(train_X)
    
    