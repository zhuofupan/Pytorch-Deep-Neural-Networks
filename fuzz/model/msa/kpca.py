# -*- coding: utf-8 -*-
"""
Created on Sat Apr  9 15:45:28 2022

@author: Fuzz4
"""
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import f, chi2
from scipy.spatial.distance import cdist
from fuzz.core.fd_statistics import Statistics

file_path = '../save/KPCA/'
file = file_path + 'selected_sample_ids.xlsx'
if not os.path.exists(file_path): os.makedirs(file_path)

class KPCA(Statistics):
    def __init__(self, **kwargs):
        default = {'fdi': 'lv',                
                   'test_stat': 'T2',           # 统计指标 T2, GLR_VAE
                   'n_components': 2,           # 主元个数
                   'kernel': 'g',               # Gaussian: g,  polynomial: p
                   'k_paras': None,             # 核参数 g (width), p(d0,d1)
                   'kernel_batch_size': 128,    # 批次计算核矩阵
                   'error_thrd': 0.09,          # 选样本时的误差阈值
                   'confidence': 1 - 0.005,     # 置信度
                   'if_recal_test_stat': False, # 是否重新计算测试统计
                   'select_sample_mthd': 'rd100', # 取代表性样本的方法
                   
                   'thrd_sele': 'kde',          # 阈值确定方法 kde, ineq
                   'kde_isf': [1,1],            # 估计cdf和isf的方法
                   'expt_FAR':  0.005,          # 预期的误报率
                   'esti_error': 0.01,          # 容许估计误差
                   'if_minus_mean': True,       # 算T2时是否减均值
                   
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
    
    # batch kernel opt
    def kernel_opt(self, X, Y):
        N, S = X.shape[0], Y.shape[0]
        bs = self.kernel_batch_size
        K = np.zeros((N, S))
        n_bs, s_bs = int(np.ceil(N/bs)), int(np.ceil(S/bs))
        for i in range(n_bs):
            start, end = i* bs, (i+1) * bs
            if end > N: end = N
            X_b = X[start: end]
            for j in range(s_bs):
                _start, _end = j* bs, (j+1) * bs
                if _end > S: _end = S
                Y_b = Y[_start: _end]
                # 运算
                if self.kernel == 'g':
                    # D_b = cdist(X_b, Y_b, 'sqeuclidean')
                    D_b = np.diag(X_b @ X_b.T).reshape(-1,1) - 2* X_b @ Y_b.T + np.diag(Y_b @ Y_b.T).reshape(1,-1)
                    K_b = np.exp(- D_b /self._w)
                    # print(K_b.mean(), K_b.max(), K_b.min())
                elif self.kernel == 'p':
                    D_b = X_b @ Y_b.T
                    K_b = np.power(D_b + self._d0, self._d1)
                K[start: end, _start: _end] = K_b
                # if j % 10 == 0 or j == s_bs-1:
                #     sys.stdout.write('\rCompute kernel matrix i:{}/{}, j:{}/{}   '.format(i+1, n_bs, j+1, s_bs))
                #     sys.stdout.flush()
        # print()
        return K
    
    def fitness(self, temp_indexs, K, N):
        # 算error
        K_SS = K[temp_indexs][:,temp_indexs]
        # print(K_SS)
        if len(temp_indexs) == 1: K_SS_inv = 1/K_SS
        else: K_SS_inv = np.linalg.inv(K_SS)
        K_NS = K[:,temp_indexs]
        K_SR = K_NS
        K_SL = K_NS/np.diag(K).reshape(-1,1)
        sum_error = np.sum( K_SL[:,np.newaxis,:] @ K_SS_inv @ K_SR[:,:,np.newaxis] )
        sum_error = 1 - sum_error/N
        return sum_error
    
    def RD(self, X, n_select):
        N = X.shape[0]
        K = self.kernel_opt( X, X )
        
        min_error = np.inf
        cnt = 0
        while cnt < 12:
            cnt += 1
            temp_indexs = np.random.choice(N, n_select, replace=False)
            error = self.fitness(temp_indexs, K, N)
            if error < min_error:
                min_error = error
                sample_indexs = temp_indexs
            sys.stdout.write('\rSelect times: {} | n_select = {}, min_error = {:.4f}'.format(cnt, n_select, min_error))
            sys.stdout.flush()
        
        print()            
        self.selected_samples = X[sample_indexs]
        self.K_SS = K[sample_indexs][:,sample_indexs]
    
    # offline - feature-samples’ selection (FSS)
    def FSS(self, X):
        N = X.shape[0]
        
        # 计算离线的全局核矩阵
        K = self.kernel_opt( X, X )
        # kd = np.diag(K)
        # print(kd.mean(), kd.max(), kd.min())
        
        sample_indexs = []
        min_errors = []
        error = np.inf
        while error > self.error_thrd and len(sample_indexs) < N:
            temp_i = []
            temp_error = []
            
            for i in range(N):
                if i in sample_indexs: continue
                # 加入临时样本 组成 临时子集
                temp_indexs = sample_indexs + [i]
                # 算error
                sum_error = self.fitness(temp_indexs, K, N)
                temp_i.append(i)
                temp_error.append(sum_error)
                if i % 10 == 0 or i == N-1:
                    sys.stdout.write('\rSelected: {} | Selecting: {}/{}, temp_error = {:.4f}'.format(len(sample_indexs), i+1, N, sum_error))
                    sys.stdout.flush()
                
            # 找error最小的加入 子样本集
            min_error = np.inf
            for k in range(len(temp_i)):
                if temp_error[k] < min_error:
                    min_i = temp_i[k]
                    min_error = temp_error[k]
            sample_indexs.append(min_i)
            min_errors.append(min_error)
            error = min_error
            
            # 存储
            save_data = np.concatenate((np.array(sample_indexs).reshape(-1,1), np.array(min_errors).reshape(-1,1)), axis = 1)
            writer = pd.ExcelWriter(file, engine='openpyxl')
            pd_data = pd.DataFrame(save_data)
            pd_data.to_excel(excel_writer = writer, encoding="utf-8", index=False)
            writer.save()
            writer.close()
            
        print()
        self.selected_samples = X[sample_indexs]
        self.K_SS = K[sample_indexs][:,sample_indexs]
    
    def fit(self, train_X):
        X = train_X
        # self.mean, self.var = np.mean(train_X, axis = 0), np.var(train_X, axis = 0)*self.N/(self.N-1)
        # X = (train_X - self.mean) / np.sqrt(self.var)
        
        # 核参数
        if self.kernel == 'g':
            if self.k_paras is None: self._w = 500 * X.shape[1] # 128**2
            else: self._w = self.k_paras
        elif self.kernel == 'p':
            if self.k_paras is None: self._d0, self._d1 = 100 * X.shape[1], 2
            else: self._d0, self._d1 = self.k_paras
        
        if 'rd' in self.select_sample_mthd:
            n_select = int(self.select_sample_mthd[2:])
            self.RD(X, n_select)
        elif self.select_sample_mthd == 'fss':
            # 求解 子样本集 并 存储
            self.FSS(X)
            # 读取 子样本集
        elif self.select_sample_mthd == 'read':
            save_data = pd.read_excel(file).values
            sample_indexs = list(save_data[:,0].astype(int))
            print(sample_indexs)
            self.selected_samples = X[sample_indexs]
            K = self.kernel_opt( X, X )
            self.K_SS = K[sample_indexs][:,sample_indexs]
        else:
            self.selected_samples = X
            self.K_SS = self.kernel_opt( X, X )
        
        K_SS = self.K_SS
        # 离线 中心化
        S = K_SS.shape[0]
        K_SS_c = K_SS - np.ones_like(K_SS)/S @ K_SS - K_SS @ np.ones_like(K_SS)/S + \
            np.ones_like(K_SS)/S @ K_SS @ np.ones_like(K_SS)/S
        
        _K_c = K_SS_c / (S - 1)
        # svd
        # U, sigma, VT = np.linalg.svd(_K)
        # self.sigma, self.A = sigma, U/ np.sqrt(S*sigma.reshape(1,-1))
        # eig
        self.sigma, self.A = np.linalg.eig(_K_c)
        self.sigma, self.A = np.real(self.sigma), np.real(self.A)
        
        if self.n_components < 1: self.n_components = int(self.n_components * S)
        
        # P 是正交矩阵 P.T @ P = I; P^{-1} = P.T
        # P_pc (m × n_components) 负载矩阵
            
        self.sigma_pc = self.sigma[:self.n_components]
        self.A_pc = self.A[:,:self.n_components]
        
        # T_pc = X @ P_pc (N × n_components) 主元子空间;
        # /hat X = T_pc @ P_pc.T = X @ P_pc @ P_pc.T 得分矩阵
        # E = X - /hat X = X - X @ P_pc @ P_pc.T 残差子空间
 
        self.inv_sigma_pc = np.diag(1/self.sigma_pc)
        
        # np.set_printoptions(formatter={'float_kind':"{:.2e}".format})
        # print('Sigma = ', self.sigma)
        _, Error2 = self.transform(X)
        print('\nRecon error = {:.2e}'.format(np.mean(Error2)))
    
    def transform(self, test_X):
        # X = (test_X - self.mean) / np.sqrt(self.var)
        X = test_X
        N, S = X.shape[0], self.K_SS.shape[0]
        # K = F(X) @ F(X).T
        K_NS = self.kernel_opt( X, self.selected_samples )
        # 在线 中心化
        K_NS_c = K_NS - np.ones_like(K_NS)/S @ self.K_SS - K_NS @ np.ones_like(self.K_SS)/S + \
            np.ones_like(K_NS)/S @ self.K_SS @ np.ones_like(self.K_SS)/S
        T_pc = K_NS_c @ self.A_pc
        # 残差 F(X) - \hat F(X) 是求不出来的，但其平方可以求出来
        T_res = K_NS_c @ self.A[:, self.n_components:]
        Error2 = (T_res[:,np.newaxis,:] @ T_res[:,:,np.newaxis]).reshape(-1,)
        return T_pc, Error2
    
    # get test statistics
    def _get_stat_vec(self, test_X, phase = 'offline'):
        X = test_X
        T_pc, Error2 = self.transform(X)
        
        if self.fdi == 'res': FDI = np.sqrt(Error2)
        elif self.fdi == 'lv': FDI = T_pc
        
        stat_vec = []
        # get test stat (by eig)
        if self.if_recal_test_stat == False:
            if self.fdi == 'res' and 'SPE' in self.test_stat:
                if phase == 'offline':
                    a = np.mean(Error2)
                    b = np.var(Error2)
                    self.SPE_g = b/(2*a)
                    self.SPE_h = 2*a*a/b
                stat_vec.append( Error2 )
            # if self.fdi == 'lv' and 'SPE' in self.test_stat:
            #     stat_vec.append( (T_pc[:,np.newaxis,:] @ T_pc[:,:,np.newaxis]).reshape(-1,) )
            if self.fdi == 'lv' and 'T2' in self.test_stat:
                stat_vec.append( (T_pc[:,np.newaxis,:] @ self.inv_sigma_pc @ T_pc[:,:,np.newaxis]).reshape(-1,) )
        
        # get test stat (recalculate)
        if len(stat_vec) == 0 or self.if_recal_test_stat:
            if phase == 'offline':
                self._Statistics__offline_stat(FDI)
            return self._Statistics__online_stat(FDI)
        
        return stat_vec
        
    def offline_modeling(self, train_X):
        # train
        self.fit(train_X)
        N, S = train_X.shape[0], self.K_SS.shape[0]
        self.stat_vec_off = self._get_stat_vec(train_X)
        
        self.fd_thrd = []
        # set threshold (with pdf)
        if self.thrd_sele == 'pdf':
            if self.fdi == 'res' and 'SPE' in self.test_stat:
                _alf = self.SPE_g * chi2.cdf(1-self.delta, self.SPE_h)
                self.fd_thrd.append( _alf ) 
            if self.fdi == 'lv' and 'T2' in self.test_stat:
                beta = self.n_components * (S - 1)/(S - self.n_components)
                _alf = beta * f.cdf(1-self.delta, self.n_components, S - self.n_components)
                self.fd_thrd.append( _alf ) 
                
            if len(self.fd_thrd) == 0: 
                self.thrd_sele = 'ineq'
            else:
                print('Mean of stat: {:.2e}, Std of stat: {:.2e}, Threshold = {:.2e}'\
                  .format(np.mean(self.stat_vec_off), np.std(self.stat_vec_off), self.fd_thrd[0]))
        
        # set threshold (without pdf)
        if self.thrd_sele != 'pdf':
            self.fd_thrd = np.zeros(len(self.stat_vec_off))
            for i in range(len(self.stat_vec_off)):
                # threshold
                if self.thrd_sele == 'ineq':
                    self.fd_thrd[i] = self._get_thrd_ineq(i)
                elif self.thrd_sele == 'kde':
                    self.fd_thrd[i] = self._get_thrd_kde(i)
        
        print("Thrd_sele = '{}', FDI = '{}', Test_stat = '{}'".format(\
               self.thrd_sele, self.fdi, self.test_stat))
        return self.fd_thrd
        
    def online_monitoring(self, test_X, label_Y):
        self.stat_vec_on = self._get_stat_vec(test_X, phase = 'online')
        return self._get_decision(label_Y)

if __name__ == '__main__':
    from fuzz.data.gene_dynamic_data import ReadData
    path = '../../data/FD_CSTR'
    dataset_name = 'FD_CSTR'
    dynamic = 1
    n_components = 3
    datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                        is_del = False, example = 'CSTR', single_mode = True).datasets
    labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
              'Fault 08', 'Fault 09', 'Fault 10']
    train_X, train_Y, test_X, test_Y = datasets
    kpca = KPCA(n_components = n_components)
    for i in range(10):
        kpca.n_components = i+1
        kpca.fit(train_X)
    
    