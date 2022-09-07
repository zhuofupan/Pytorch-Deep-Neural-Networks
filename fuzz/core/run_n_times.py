# -*- coding: utf-8 -*-
"""
Created on Mon May 30 21:11:35 2022

@author: Fuzz4
"""
import time
import torch
import numpy as np

class Run_N_Times(object):
    def __init__(self, **kwargs):
        
        default = {'import_str': 'from fuzz.model.vae import VAE', 
                   'class_name': 'VAE',
                   'parameter': None,
                   'run_times': -1,
                   
                   'datasets': None,
                   'e': 12,
                   'b': 16,
                   'labels_name': None,
                   'if_plot': None
                   }
        
        for key in kwargs.keys():
            if key in default.keys():
                default[key] = kwargs[key]
        for key in default.keys():
            setattr(self, key, default[key])
            
        self._add_info = self.parameter['add_info']
        self.model = None
        
    def run(self):
        if type(self.import_str) != list: self.import_str = [self.import_str]
        if type(self.class_name) != list: self.class_name = [self.class_name]
        rank_info = ['th', 'st', 'nd', 'rd'] + ['th'] * 6
        # change model
        for i in range(len(self.import_str)):
            eval(self.import_str[i])
            # run n times
            for n in range(self.run_times):
                if n + 1 > 10 and n + 1 < 20: rank = 'th'
                else: rank = rank_info[np.mod(i+1, 10)]
                print("\n---------------------------------------")
                print(" ★ Running the {} for the {}{} time ---".format(self.class_name[i], n + 1, rank))
                print("---------------------------------------------")
                
                rd_sd = torch.randint(0, 10000, (1,))
                torch.manual_seed(rd_sd[0])
                print('Random seed = {}'.format(rd_sd[0]))
                
                self.parameter['add_info'] = self._add_info + '_{}{}'.format(n + 1, rank)
                if n > 0:
                    if hasattr(self, 'model'): print('厚礼谢')                
                    del self.model
                    time.sleep(1.0)
                    
                    self.parameter['show_model_info'] = False
                    self.parameter['save_module_para'] = False
                    self.model = eval(self.class_name[i] + '(**self.parameter)')
                    time.sleep(1.0)
                    
                    self.model._init_para()
                    self.model.run(datasets = self.datasets, e = self.e, b = self.b,
                                   load = '', cpu_core = 0.8, num_workers = 0)
                    
                    if self.if_plot: self.model.result(self.labels, True)