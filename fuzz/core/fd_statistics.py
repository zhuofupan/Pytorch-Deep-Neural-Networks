# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 13:13:33 2021

@author: Fuzz4
"""
import os
import torch
import pandas as pd
import numpy as np
from pandas import DataFrame
from collections import OrderedDict
from sklearn import manifold
from time import time
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from scipy import interpolate
from scipy.stats.distributions import chi2
from scipy.special import ndtr
from scipy.stats import gaussian_kde

def Save_excel(data, save_path, file_name, sheet_names = ['Sheet1']):
    print('Save {}.xlsx in {}'.format(file_name, save_path))
    if not os.path.exists(save_path): os.makedirs(save_path)
    if type(data) == dict:
        sheet_names = data.keys()
        data_list = []
        for i, key in enumerate(data.keys()):
            data_list.append(data[key])
        data = data_list
    elif type(data) == list: sheet_names = [str(i) for i in range(len(data))]
    else: data = [data]
    
    file_path = save_path + '/' + file_name + '.xlsx'
    writer = pd.ExcelWriter(file_path, engine='openpyxl')
    for i, sheet_name in enumerate(sheet_names):
        fd_data = DataFrame(np.array(data[i]))
        fd_data.to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
    writer.save()
    writer.close()

class Statistics():
    def __init__(self, **kwargs):
        default = {'fdi': 'res',                # 残差产生形式 re, abs_re, lv
                   'part_dim': None,            # 取后面几个维度作为残差
                   'test_stat': 'T2',           # 统计指标 T2, GLR_VAE
                   'thrd_sele': 'ineq',         # 阈值确定方法 pdf, ineq, kde
                   'expt_FAR':  0.005,          # 预期的误报率
                   'esti_error': 0.01,          # 容许估计误差
                   'confidence': 1 - 0.005,     # 置信度
                   '__interpolation__': False,  # 是否用插值计算T2
                   'kde_isf': [1,1],            # 估计cdf和isf的方法
                   'if_minus_mean': True,       # 算T2时是否减均值
                   'save_fdi': False,           # 保存残差为excel
                   'view_fdi': False,           # 查看残差的低维流行
                   'n_sampling': 100,           # 采样次数
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
    
    def offline_modeling(self, inputs, latents, outputs, custos = None, save_name = ''):
        print('\nOffline modeling:')
        fdi = self.fdi_generation(inputs, latents, outputs, custos)
        # offline
        if self.view_fdi:
            self._view_fdi_manifold(fdi, save_name)
        self.__offline_stat(fdi)
        # get_offline_stat
        self.stat_vec_off = self.__online_stat(fdi)
        self.fd_thrd = np.zeros(len(self.stat_vec_off))
        self.sorted_stat_vecs = []
        for i in range(len(self.stat_vec_off)):
            # threshold
            if self.thrd_sele == 'ineq':
                self.fd_thrd[i] = self._get_thrd_ineq(i)
            elif self.thrd_sele == 'kde':
                self.fd_thrd[i] = self._get_thrd_kde(i)
        self.save_sorted_stats_to_excel()
        return self.fd_thrd
    
    def online_monitoring(self, inputs, latents, outputs, custos, labels):
        print('\nOnline monitoring:')
        # generate_residual
        fdi = self.fdi_generation(inputs, latents, outputs, custos)
        # get_online_stat
        self.stat_vec_on = self.__online_stat(fdi)
        stat_lists, switch_p_list, FAR, MDR = self._get_decision(labels)
        # Save fdi to excel
        if self.save_fdi:
            self.__save_fdi(fdi)
        return stat_lists, switch_p_list, FAR, MDR
    
    def fdi_generation(self, inputs, latents, outputs, custos): 
        if self.fdi == 'res':
            fdi = inputs - outputs
        elif self.fdi == 'lv':
            fdi = latents
        elif self.fdi == 'out':
            fdi = outputs
        # 在 _get_customized_fdi 里自定义的 fdi
        elif self.fdi == 'custo':
            fdi = custos
        
        # 去掉最后几个维度
        if self.part_dim is not None:
            fdi = fdi[:, -self.part_dim:]
        # 打印信息： 输入 潜变量 输出
        print('\n\t\tmean\tstd')
        for i, data in enumerate([inputs, latents, outputs, inputs - outputs, custos]):
            if data is None: continue
            # 期望和方差的二范数
            mean = np.sqrt( np.sum( (np.mean(data, axis = 0))**2 ) )
            std = np.sqrt( np.sum( (np.std(data, axis = 0))**2 ) )
            # std = np.sqrt( np.sum( (np.std(data, axis = 0))**2 ) )
            print('{}:\t{:.4f}\t{:.4f}'.format(['input ', 'latent', 'output', 'res', 'custo'][i], mean, std))
        return fdi
    
    # 离线计算 协方差的逆
    def __offline_stat(self, fdi):
        if 'T2' in self.test_stat:
            self.t2_fdi_mean, self.t2_cov_inverse = self.__cov_inverse(fdi)
        if 'SPE' in self.test_stat:
            self.spe_fdi_mean = np.mean(fdi, axis = 0).reshape(1,-1)
    
    # 在线计算 测试统计
    def __online_stat(self, fdi):
        # msa 方法, fdi = lv 时, 用特征值作为协方差矩阵，也不用减均值
        stat_vec = []
        if 'T2' in self.test_stat:
            stat_vec.append( self.__quadratic_form(fdi, self.t2_fdi_mean, self.t2_cov_inverse) )
        if 'SPE' in self.test_stat:
            if self.if_minus_mean:
                fdi -= self.spe_fdi_mean
            stat_vec.append( np.array(np.sum(fdi **2, 1) , dtype = np.float32).reshape(-1) ) 
        return stat_vec
    
    # 学习 阈值    
    def _get_thrd_ineq(self, index):
        # ϵ, δ, E[FAR] =  error, delta, self.expt_FAR
        stat_vec = self.stat_vec_off[index]
        # 计算统计量的均值、方差
        print('\nMean of stat: {:.2f}, Std of stat: {:.2f}'\
              .format(np.mean(stat_vec), np.std(stat_vec)))
        # 从小到大
        sorted_stat_vec = np.sort(stat_vec)
        # print('\nSorted test statistics in descending order:')
        # print(stat_vec[np.argsort(-stat_vec)])
        error, delta = self.esti_error, 1- self.confidence
        # median
        
        N = sorted_stat_vec.shape[0]
        gama = N * (1 - self.expt_FAR)
        k = int(np.floor(gama))
        if hasattr(self, '__interpolation__') and self.__interpolation__:
            thrd = sorted_stat_vec[k-1] + (gama - k) * (sorted_stat_vec[k] - sorted_stat_vec[k-1])
        else:
            thrd = sorted_stat_vec[k-1]

        print('Threshold = {:.2f}'.format(thrd))
        # min_inf = int(np.ceil((1 - self.expt_FAR - error)*sorted_stat_vec.shape[0]))
        # thrd = sorted_stat_vec[min_inf - 1]
        v2, lb = self._compt_v2_lb(thrd)
        lg = np.log(2/delta)
        print('\nSample size: N = ({})'.format(stat_vec.shape[0]))
        # One-sided Chernoff bound
        N_min = lg/(2*error**2)
        print("Chernoff's minimum requirements: N >= ({})".format(int(np.ceil(N_min))))
        # Freedman inequalities with [ΔM], ΔM >= -lb, lb > 0
        # exp(-(Nε)^2/(2*(v2+Nεb))) <= δ
        N_min = (lg*lb + np.sqrt((lg*lb)**2 + 2*v2*lg)) / error
        print("Freedman's minimum requirements: N >= ({})".format(int(np.ceil(N_min))))
        # De la Peña inequalities with [ΔM]
        # exp(-(Nε)^2/(2*v2)) <= δ
        N_min = np.sqrt(2*v2*lg) / error
        print("De la Pena's minimum requirements: N >= ({})".format(int(np.ceil(N_min))))
        # Bernstein inequalities with [ΔM], ΔM >= -1
        # (1+Nε/v2)^v2*exp(-Nε) <= δ
        N_min = self._stride_search_threshold(error, v2, delta)
        print("Bernstein's minimum requirements: N >= ({})".format(N_min))
        
        self.plot_sorted_stats(index, sorted_stat_vec, thrd)
        if hasattr(self, 'sorted_stat_vecs'):
            self.sorted_stat_vecs.append(sorted_stat_vec.reshape(-1,1))
        return thrd
    
    def _get_thrd_kde(self, index):
        stat_vec = self.stat_vec_off[index]
        # print(stat_vec.shape)
        # 计算统计量的均值、方差
        print('\nMean of stat: {:.2e}, Std of stat: {:.2e}'\
              .format(np.mean(stat_vec), np.std(stat_vec)))
        
        x = np.sort(stat_vec)
        # x = np.insert(x, 0, 0)
        # x = np.append(x, x[-1] + (x[-1] - x[-2]) )
        
        # x = np.insert(x, 0, np.arange(x[0], x[1], 100))
        # x = np.append(x, np.arange(x[-2], x[-1], 50))
        x = np.unique(x)
        
        # create kde
        kde = gaussian_kde(stat_vec)
        # evaluate pdfs
        kde_pdf = kde.evaluate(x)
        
        # save fig
        zoom = 0.03
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].plot(x, kde_pdf, c='r')
        ax[0].set_title('PDF')
        ax[0].set_xscale('symlog')
        ymin, ymax = np.min(kde_pdf), np.max(kde_pdf)
        ax[0].set_ylim(ymin - (ymax - ymin) * zoom, ymax + (ymax - ymin) * zoom)
        
        # methods of estimate cdf
        if self.kde_isf[0] == 1:
            cdf_func = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
            cdf = cdf_func(x)
        elif self.kde_isf[0] == 2:
            cdf = tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
                        for item in x)
        elif self.kde_isf[0] == 3:
            stdev = np.sqrt(kde.covariance)[0, 0]
            cdf = ndtr(np.subtract.outer(x, stat_vec)/stdev).mean(axis=1)

        ax[1].plot(x, cdf, c='r')
        ax[1].set_title('CDF')
        ax[1].set_xscale('symlog')
        ymin, ymax = np.min(cdf), np.max(cdf)
        ax[1].set_ylim(ymin - (ymax - ymin) * zoom, ymax + (ymax - ymin) * zoom)
        
        # methods of estimate inv cdf
        if self.kde_isf[1] == 1:
            isf_curve = interpolate.InterpolatedUnivariateSpline(x, 1-cdf-self.expt_FAR)
            thrd = isf_curve.roots()
        elif self.kde_isf[1] == 2:
            isf_curve = interpolate.interp1d(1-cdf-self.expt_FAR, x, kind='cubic', bounds_error=False)
            thrd = isf_curve.roots()
        
        if type(thrd) == np.ndarray:
            print('Solutions = ', thrd)
            thrd = np.min(thrd)
        
        y = isf_curve(x) + self.expt_FAR
        ax[2].plot(x, y, c='r')
        ax[2].set_title("ISF")
        ax[2].set_xscale('symlog')
        # ax[2].plot(x, np.ones_like(x) * self.expt_FAR, c='gray', linestyle = '--')
        # ax[2].set_yscale('symlog')
        ymin, ymax = np.min(y), np.max(y)
        ax[2].set_ylim(ymin - (ymax - ymin) * zoom, ymax + (ymax - ymin) * zoom)

        file_path = self.save_path + '_('+ str(index+1) +')/'
        if not os.path.exists(file_path): os.makedirs(file_path)
        plt.savefig( file_path + 'KDE.pdf', bbox_inches='tight')
        plt.savefig( file_path + 'KDE.svg', bbox_inches='tight')
        plt.close(fig)
        
        # save excel
        print('\nSave KDE to excel in '+ self.save_path)
        file_name = file_path + '/['+self.name+'] kde.xlsx'
        writer = pd.ExcelWriter(file_name, engine='openpyxl')
        fd_data = np.concatenate([x.reshape(-1,1), kde_pdf.reshape(-1,1), cdf.reshape(-1,1), 
                                  isf_curve(x).reshape(-1,1)], axis = 1 )
        fd_data = pd.DataFrame(fd_data, columns = ['x', 'pdf', 'cdf', 'isf'])
        fd_data.to_excel(excel_writer = writer, encoding="utf-8", index=False)
        writer.save()
        writer.close()
        
        if hasattr(self, 'sorted_stat_vecs'):
            sorted_stat_vec = np.sort(stat_vec)
            self.sorted_stat_vecs.append(sorted_stat_vec.reshape(-1,1))
        
        return thrd
            
    def _get_decision(self, labels):
        # labels
        test_Y = (labels.argmax(axis = 1) != 0).astype(np.int32)
        
        if self.split_p_list is None:
            # split data sets according to fault type
            start = 0
            self.split_p_list = [start] # 包括头0和尾length-1
            for p in range(test_Y.shape[0]):
                if p == test_Y.shape[0] - 1 or (test_Y[p] == 1 and test_Y[p+1] == 0):
                    self.split_p_list.append(p+1)
                    start = p+1
                    
        # find switch points
        self.switch_p_list = []
        for i in range(len(self.split_p_list)-1):
            test_y = test_Y[self.split_p_list[i]:self.split_p_list[i+1]].reshape(-1,1)
            switch_p = []
            for p in range(test_y.shape[0]-1):
                if test_y[p+1] != test_y[p]:
                    switch_p.append(p+1)
            switch_p = np.array(switch_p).reshape(-1,1)
            self.switch_p_list.append(switch_p)
        
        # Loop stat type
        preds = []
        self.stat_lists = {}
        for i in range(len(self.stat_vec_on)):
            # stat_list
            self.stat_list = []
            stat_vec = self.stat_vec_on[i]
            max_lenth = 0
            for k in range(len(self.split_p_list) - 1): 
                lenth = self.split_p_list[k+1] - self.split_p_list[k]
                if lenth > max_lenth: max_lenth = lenth
                self.stat_list.append(stat_vec[self.split_p_list[k]: self.split_p_list[k+1]])
            self.stat_lists[str(i)] = self.stat_list
            self.stat_array = np.zeros((max_lenth, len(self.stat_list)))
            for k in range(len(self.stat_list)):
                lenth = self.stat_list[k].shape[0]
                self.stat_array[:lenth, k] = self.stat_list[k]
            # pred
            pred_Y = (stat_vec > self.fd_thrd[i]).astype(np.int32)
            preds.append(pred_Y.reshape(1,-1))
            # get_FAR_MDR
            FAR, MDR = self._get_FDR_MDR(pred_Y, test_Y)
            self.save_test_excel_plot(i, self.fd_thrd[i], FAR, MDR)
        
        if len(self.stat_vec_on) > 1:
            # Together
            preds = np.concatenate(preds, axis = 0)
            pred_Y = np.max(preds, axis = 0)
            FAR, MDR = self._get_FDR_MDR(pred_Y, test_Y)
            self.save_test_excel_plot(len(self.stat_vec_on), self.fd_thrd, FAR, MDR, False)
        return self.stat_lists, self.switch_p_list, FAR, MDR
    
    def _get_FDR_MDR(self, pred_Y, test_Y):
        FAR, MDR = [], []
        sum_fa, sum_normal = 0, 0
        sum_md, sum_faulty = 0, 0
        for k in range(len(self.split_p_list) - 1):
            # split data sets according to fault type
            p1, p2 = self.split_p_list[k], self.split_p_list[k+1]
            pred_y = pred_Y[p1:p2]
            test_y = test_Y[p1:p2]
            # lables norml and faulty location
            normal_loc = np.where(test_y == 0)[0]
            faulty_loc = np.where(test_y != 0)[0]
            # false alarm & missed detection
            n_fa = (pred_y[normal_loc] == 1).astype(np.int32).sum()
            n_md = (pred_y[faulty_loc] == 0).astype(np.int32).sum()
    
            sum_fa += n_fa; sum_normal += normal_loc.shape[0]
            sum_md += n_md; sum_faulty += faulty_loc.shape[0]
            far = np.round(n_fa * 100.0 / normal_loc.shape[0], 2)
            mdr = np.round(n_md * 100.0 / faulty_loc.shape[0], 2)
            FAR.append(far)
            MDR.append(mdr)
        FAR.append(np.round(sum_fa * 100.0/sum_normal, 2))
        MDR.append(np.round(sum_md * 100.0/sum_faulty, 2))
        
        return FAR, MDR
    
    def __cov_inverse(self, fdi):
        # centered
        fdi_mean = np.mean(fdi, axis = 0).reshape(1,-1)
        fdi_cen = fdi
        if self.if_minus_mean:
            fdi_cen -= fdi_mean
        # cov_matrix
        fdi_cen = torch.from_numpy(fdi_cen).float()
        TT = torch.bmm(fdi_cen.unsqueeze(-1), fdi_cen.unsqueeze(1))
        self.cov_matrix = torch.mean(TT, axis = 0)
        cov_inverse = torch.inverse(self.cov_matrix)
        return fdi_mean, cov_inverse
    
    def __mul_fdi(self, X, V2, mean = None):
        X = torch.from_numpy(X).float()
        V2 = torch.from_numpy(V2).float()
        
        if mean is None:
            X2 = torch.diag( X @ X.t() ).reshape(-1,1)
            mul_fdi = V2/X2
            self.x2_mean = torch.mean(X2)
            print('x2_mean:', self.x2_mean)
        else:
            mul_fdi = V2/self.x2_mean
        return mul_fdi.data.cpu().numpy()

    def __quadratic_form(self, fdi, fdi_mean, cov_inverse):
        # centered
        if self.if_minus_mean:
            fdi -= fdi_mean
        stat_vec = []
        fdi = torch.from_numpy(fdi).float()
        for r in fdi:
            T2 = torch.mm(torch.mm(r.view(1,-1), cov_inverse), r.view(-1,1))
            stat_vec.append(T2.data.cpu().numpy())
        stat_vec = np.array(stat_vec, dtype = np.float32).reshape(-1)
        return stat_vec

    def _compt_v2_lb(self, thrd):
        FAR = self.expt_FAR
        indicator = np.array(self.stat_vec_off > thrd).astype(np.int32)
        X = indicator - FAR
        mean = np.mean(X)
        v2 = np.sum(X**2)
        # lb > 0 
        lb = FAR
        print('Martingale difference: mean_[X] = {:.4f}, sum_[X**2] = {:.4f}'.format(mean, v2))   
        return v2, lb

    
    def plot_sorted_stats(self, index, sorted_stat_vec, thrd):
        # save plot
        N = sorted_stat_vec.shape[0]
        N_left = int(N * 0.9)
        x = np.arange(N_left + 1, N + 1)
        thrds = np.array([thrd] * (N - N_left))
        
        fig = plt.figure(figsize=[24,15])

        ax = fig.add_subplot(111)
        
        ax.plot(x, sorted_stat_vec[N_left:], linewidth = 5, c = 'b', label = '$T^2$')
        ax.plot(x, thrds, c = 'black', linestyle = '--', linewidth = 3, label = 'Threshold')
        
        ax.tick_params('x', labelsize = 48)
        ax.set_xlabel('Samples', fontsize = 58)  # 设置x坐标轴
        ax.tick_params('y', labelsize = 48)
        ax.set_ylabel('$T^2$', fontsize = 58)    # 设置y坐标轴
        plt.xlim(N_left, N + 1)
        plt.yscale('log')
        
        lgd = ax.legend(loc = 'upper right', fontsize=48)
        lgd.get_frame().set_alpha(0.5)
        
        plt.tight_layout()
        file_path = self.save_path + '_('+ str(index+1) +')/'
        if not os.path.exists(file_path): os.makedirs(file_path)
        plt.savefig( file_path + 't2Training_set.pdf', bbox_inches='tight')
        plt.savefig( file_path + 't2Training_set.svg', bbox_inches='tight')

        plt.close(fig)
        
    def save_sorted_stats_to_excel(self):
        # save excel
        print('\nSave normal sorted statistics to excel at '+ self.save_path)
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        file_name = self.save_path + '/['+self.name+'] Sorted_Statistics.xlsx'
        writer = pd.ExcelWriter(file_name, engine='openpyxl')
        sheet_names = ['({}) {:.2f}'.format(i, self.fd_thrd[i-1]) for i in range(1, len(self.sorted_stat_vecs)+1)]
        for i, sheet_name in enumerate(sheet_names):
            fd_data = DataFrame(np.array(self.sorted_stat_vecs[i]))
            fd_data.to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()

    def save_test_excel_plot(self, index, fd_thrd, FAR, MDR, _plot = True):
        print('\n>>> {}:'.format(index + 1))
        print('\nThe threshold is {}'.format(np.round(np.array(fd_thrd),2)))
        print('\nThe test false alarm rates (FARs) are:\n{}(%)'.format(FAR[:-1]))
        print('\nThe test missed detection rates (MDRs) are:\n{}(%)'.format(MDR[:-1]))
        print('\nThe average test FAR is {}%.'.format(FAR[-1]))
        print('The average test MDR is {}%.\n'.format(MDR[-1]))
        # writer
        print("Save "+self.name+"'s statistic results in '/save/{}{}{}'".\
              format(self.name, self.add_info, self.run_id))
        file_name = self.save_path + '/['+self.name+'] FD_result_('+ str(index+1) +').xlsx'
        writer = pd.ExcelWriter(file_name, engine='openpyxl')
        sheet_names = ['stat_list', 'switch_p_list', 'fd_thrd']
        length = 0
        for i in range(len(self.switch_p_list)):
            length = max(length, self.switch_p_list[i].shape[0])
        self.switch_p_array = np.zeros((length, len(self.switch_p_list)))
        for i in range(len(self.switch_p_list)):
            self.switch_p_array[:self.switch_p_list[i].shape[0],i] = self.switch_p_list[i].reshape(-1,)
        
        fd_data = [DataFrame(self.stat_array), 
                   DataFrame(self.switch_p_array),
                   DataFrame(np.array(fd_thrd).reshape(1,-1))]
        for i, sheet_name in enumerate(sheet_names):
            fd_data[i].to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()
    
    def __save_fdi(self, fdi):
        print('\nSave monitoring indicators to excel in '+ self.save_path)
        file_name = self.save_path + '/['+self.name+'] Monitoring_Indicators.xlsx'
        writer = pd.ExcelWriter(file_name, engine='openpyxl')
        sheet_names = [str(i) for i in range(1, len(self.switch_p_list)+1)]
        fd_data = []
        # print(self.split_p_list)
        for k in range(len(self.split_p_list) - 1): 
            start, end = self.split_p_list[k], self.split_p_list[k+1]
            fd_data.append(DataFrame(np.array(fdi[start: end])))
        for i, sheet_name in enumerate(sheet_names):
            fd_data[i].to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()
    
    def _view_fdi_manifold(self, fdi, save_name):   
        # fdi = MinMaxScaler().fit_transform(fdi)
        
        n_neighbors = 10
        n_components = 2
        
        # Create figure
        fig = plt.figure(figsize=(16, 9))
        fig.suptitle("Manifold Learning with %i points, %i neighbors"
                     % (fdi.shape[0], n_neighbors), fontsize=14)
        
        # Set-up manifold methods
        methods = OrderedDict()
        methods['Isomap'] = manifold.Isomap(n_neighbors=n_neighbors,
                                            n_components=n_components)
        methods['MDS'] = manifold.MDS(n_components, max_iter=50, n_init=1)
        # methods['t-SNE'] = manifold.TSNE(n_components=n_components, init='pca',
        #                                  random_state=0)
        
        # Plot results
        for i, (label, method) in enumerate(methods.items()):
            t0 = time()
            Y = method.fit_transform(fdi)
            t1 = time()
            print("%s: %.2g sec" % (label, t1 - t0))
            ax = fig.add_subplot(1, 2, i+1)
            ax.scatter(Y[:, 0], Y[:, 1], cmap=plt.cm.Spectral)
            ax.set_title("%s (%.2g sec)" % (label, t1 - t0))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            ax.axis('tight')
        
        save_path = '../save/'+save_name
        if not os.path.exists(save_path): os.makedirs(save_path)
        plt.savefig(save_path + '/fD_fdi.png', bbox_inches = 'tight', format = 'png')
        plt.savefig(save_path + '/fD_fdi.svg', bbox_inches='tight')
        # plt.show()
        plt.close(fig)

    # 二分查找
    def _stride_search_threshold(self, error, v2, delta):
        def Bernstein(n):
            return (1+n*error/v2)**v2 * np.exp(-n*error) - delta
        left_N, stride = 0, 50
        while Bernstein(left_N) > 0:
            left_N += stride
        for i in range(left_N - stride, left_N):
            if Bernstein(i+1) < 0:
                return i+1
    