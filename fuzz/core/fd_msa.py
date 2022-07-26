# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 16:42:14 2021

@author: Fuzz4
"""
# Multivariate Statistical Analysis

import sys
import time
import os
import numpy as np
import traceback
from IPython.core.ultratb import ColorTB

from ..model.msa.pca import PCA
from ..model.msa.kpca import KPCA
from ..core.log import Logger
from ..visual.plot import stat_curve


mthd_dict = {'pca':'PCA',
             'kpca':'KPCA'}
decomposition_mthds = ['pca','kpca']

class MSA():
    def __init__(self, 
                 mthd = 'pca',
                 add_info = '_fd',
                 **kwargs):
        
        self.name = mthd_dict[mthd]
        self.add_info = add_info
        self.Stat = eval(self.name)(**kwargs)
        # 输出类的所有属性
        # for item in dir(self.Stat):
        #     print(item)
    
    def run(self, datasets, **kwargs):
        try:
            log_path = '../Logs/[fd] ' + self.name + self.add_info + '/'
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            log_file_name = log_path + 'log - ' + time.strftime("%Y.%m.%d - %Hh %Mm %Ss", time.localtime()) + '.log'

            self.logger = Logger(log_file_name)
            self._run(datasets, **kwargs)
        except:
            # save Error to file
            self.logger.to_file(traceback.format_exc())
            self.logger.reset()
            # show Error in console
            color = ColorTB()
            exc = sys.exc_info()
            for _str in color.structured_traceback(*exc):
                print(_str)
        finally:
            self.logger.reset()
    
    def _run(self, datasets, **kwargs):
        self.train_X, self.train_Y, self.test_X, self.test_Y = datasets
        
        normal_indexs = np.argwhere(self.test_Y.argmax(axis = 1) == 0).reshape(-1,)
        test_Y_n = self.test_Y[normal_indexs]
        print('\nNumber of train data:')
        print('->  Normal{}'.format(self.train_X.shape))
        print('Number of test samples:')
        print('->  Normal({}, {}),  faulty({}, {})'.format(test_Y_n.shape[0], self.test_X.shape[1],\
              self.test_Y.shape[0] - test_Y_n.shape[0], self.test_X.shape[1]))
        
        if type(self.train_X) == tuple:
            self.train_X, self.train_Y = self.train_X
            self.test_X, self.test_Y = self.test_X
            
        # train
        t0 = time.perf_counter()
        print('\nFit ' + self.name + '(n_components = {}) ...'.format(self.Stat.n_components))
        
        self.Stat.name, self.Stat.add_info, self.Stat.run_id, self.Stat.save_path = \
            self.name, '', '', '../save/' + self.name + self.add_info
        if not os.path.exists(self.Stat.save_path):
                os.makedirs(self.Stat.save_path)
            
        # offline
        self.fd_thrd = self.Stat.offline_modeling(self.train_X)
        
        t1 = time.perf_counter()
        print('Finish fit, cost {} seconds'.format(int(t1 - t0)))
        
        # online
        # print(inputs.shape, latents.shape, outputs.shape)
        self.stat_lists, self.switch_p_list, self.FAR, self.MDR = \
            self.Stat.online_monitoring(self.test_X, self.test_Y)
            
    
    def result(self, label_name, plot):
        __subplot__ = False
        if hasattr(self, '__subplot__'): __subplot__ = self.__subplot__
        for i, key in enumerate (self.stat_lists.keys()):
            stat_list = self.stat_lists[key]
            stat_curve(stat_list, self.switch_p_list, self.fd_thrd[i], 
                       self.name, 
                       self.add_info + '_(' +str(i+1) + ')', 
                       label_name, 
                       self.Stat.plot_p_list,
                       __subplot__)