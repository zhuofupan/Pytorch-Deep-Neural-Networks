# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:35:45 2021

@author: Fuzz4
"""
import numpy as np
from fuzz.model.supdy import SupDynamic, split_data
from fuzz.data.gene_dynamic_data import ReadData
from fuzz.core.system_model import System

def get_fcnn_model(n_p = 20, 
                   n_f = 20, 
                   dataset_id = 1,
                   model_id = 1,
                   struct_id = 1,
                   dropout = 0.0):

    # modle id
    class_name = ['SupDynamic', '']
    _class = class_name[model_id - 1]
        
    # data set
    if dataset_id == 1:
        dim_u, dim_y = 3, 7
        dynamic = n_p + n_f
        dataset_name = 'CSTR'
        path = '../data/FD_CSTR'
        datasets = ReadData(path, [None, 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            example = 'CSTR', single_mode = True).datasets
        datasets = split_data(n_p, n_f, dim_u, datasets)
        labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08', 'Fault 09', 'Fault 10']
    elif dataset_id == 2:
        dim_u, dim_y = 1, 2
        dynamic = n_p + n_f
        dataset_name = 'SYS'
        sys = System()
        datasets, labels = sys.demo(dynamic)
    
    in_dim = int(dim_u * dynamic + dim_y * n_p)
    out_dim = int(dim_y * n_f)
    
    fd_dict={'res_generator': 'o',
             'test_stat': 'T2',
             'thrd_sele': 'ineq',
             'ineq_rv': 'indi_mean',
             'if_minus_mean': False,
             'confidence': 1 - 0.5/100}
    
    output_func = ['a']
    if struct_id == 1:
        struct = [in_dim, '/2', '/1', out_dim]
        hidden_func = ['g','a']
    elif struct_id == 2:
        struct = [in_dim, '/2', '/2', out_dim]
        hidden_func = ['g','g']
    elif struct_id == 3:
        struct = [in_dim, '/2', '/2', '/2', out_dim]
        hidden_func = ['g','a','g']
    elif struct_id == 4:
        struct = [in_dim, '/2', '/2', '/2', out_dim]
        hidden_func = ['a','g','a']
    
    parameter = {'add_info':'_fd_' + dataset_name,
                 'dvc': 'cuda', 
                 'dim_u': dim_u,
                 'dim_y': dim_y,
                 'n_p': n_p,
                 'n_f': n_f,
                 'struct': struct,
                 'hidden_func': hidden_func,
                 'output_func': output_func,
                 'dropout': dropout,
                 'task': 'fd',
                 'expt_FAR': 0.5/100,
                 'esti_error': 0.005,
                 'optim':'RMSprop',
                 'optim_para': 'alpha=0.9, eps=1e-10',
                 'lr': 1e-4,
                 'label_name': labels
                 }
    
    parameter.update(fd_dict)
    
    model = eval(_class + '(**parameter)')
    return model, datasets, labels

if __name__ == '__main__':
    
    model, datasets, labels = get_fcnn_model(n_p = 5, 
                                             n_f = 3, 
                                             dataset_id = 2,
                                             model_id = 1,
                                             struct_id = 1,
                                             dropout = 0)
 
    model.run(datasets = datasets, e = 12, b = 16, load = '', cpu_core = 0.8, num_workers = 0)
    model.result(labels, True)
    