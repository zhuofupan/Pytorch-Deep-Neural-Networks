# -*- coding: utf-8 -*-
"""
Created on Fri Dec 24 14:35:45 2021

@author: Fuzz4
"""
import numpy as np
from fuzz.model.supdy import split_data
from fuzz.model.dae import Deep_AE
from fuzz.model.vae import VAE
from fuzz.data.gene_dynamic_data import ReadData
from fuzz.core.system_model import System

def get_fcnn_model(n_p = 20, 
                   n_f = 20, 
                   dataset_id = 1,
                   model_id = 1,
                   struct_id = 1,
                   dropout = 0.0):

    # modle id
    class_name = ['Deep_AE', 'VAE']
    _class = class_name[model_id - 1]
        
    # data set
    if dataset_id == 1:
        dim_u, dim_y = 3, 4
        dynamic = n_p + n_f
        dataset_name = 'CSTR'
        path = '../data/FD_CSTR'
        datasets = ReadData(path, [None, 'oh'], dynamic, is_del = True, task = 'fd', cut_mode = '', 
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
    
    in_dim = int((dim_u  + dim_y) * dynamic)
    out_dim = in_dim
    
    fd_dict={'res_generator': 're',
             'part_dim': int(dim_y * n_f),
             'test_stat': 'T2',
             'thrd_sele': 'ineq',
             'ineq_rv': 'indi_mean',
             'if_minus_mean': False,
             'confidence': 1 - 0.5/100}
    
    # sturct id
    decoder_func = 'a'
    output_func = 'a'
    latent_func = ['a','a']
    if struct_id == 1:
        struct = [in_dim, '/2']
        hidden_func = ['g','a']
        decoder_func = ['g','a']
    elif struct_id == 2:
        struct = [in_dim, '/2']
        hidden_func = ['g','a']
        decoder_func = ['a','a']
    elif struct_id == 3:
        struct = [in_dim, '/2','/2']
        hidden_func = ['g','a','a']
        decoder_func = ['g','a','a']
    elif struct_id == 4:
        struct = [in_dim, '/2','/2']
        hidden_func = ['s','t','a']
        decoder_func = ['g','a','a']
    
    parameter = {'add_info':'_fd_' + dataset_name,
                 'dvc': 'cuda', 
                 'struct': struct,
                 'label_name': labels,
                 'hidden_func': hidden_func,
                 'decoder_func': decoder_func,
                 'latent_func': latent_func,
                 'output_func': output_func,
                 'dropout': dropout,
                 'share_w': False,
                 'task': 'fd',
                 'view_res': False,
                 'expt_FAR': 0.5/100,
                 'esti_error': 0.005,
                 'alf': 1,
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
    