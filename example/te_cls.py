# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from model.sae import SAE
from data.gene_dynamic_data import ReadData

datasets = ReadData('../data/TE', ['st', 'oh'], 40, cut_mode = '', example = 'TE').dataset
parameter = {'name':'TE',
                'struct': [40*33,600,400,200,19],
                'hidden_func': ['g', 'a', 'a'],
                'output_func': 'g',
                'ae_type': 'AE',
                'sup_factor': 1,
                'act_func': ['g', 'a'],
                'dropout': 0.382,
                'share_w': False,
                'task': 'cls',
                'optim':'RMSprop',
                'optim_para': 'alpha=0.9, eps=1e-10',
                'lr': 1e-4,
                'pre_lr': 1e-3}
model = SAE(**parameter)
labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 04','Fault 05', 'Fault 06', 'Fault 07', 'Fault 08', 
          'Fault 10', 'Fault 11', 'Fault 12', 'Fault 13', 'Fault 14', 'Fault 16', 'Fault 17', 'Fault 18',
          'Fault 19', 'Fault 20', 'Fault 21']
    
model.run(datasets, e = 120, b = 16, pre_e = 15, load = '')     
model.result(labels)