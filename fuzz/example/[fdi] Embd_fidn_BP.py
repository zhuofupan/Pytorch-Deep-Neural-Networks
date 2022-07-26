# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:21:40 2022

@author: Fuzz4
"""
from fuzz.data.gene_dynamic_data import ReadData
from fuzz.model.variant.embd_fidn import Embd_DNet

from fuzz.model.dae import Deep_AE
from fuzz.model.vae import VAE
from fuzz.core.fd_msa import MSA
from fuzz.data.gene_dynamic_data import ReadData

def get_sae_model(data_set = 1, dynamic = 1, model_id = 1, fdi = 'lv',
                  struct_id = 1, dropout = 0.382, alf = 1, fd_mode = 1):

    # modle id
    class_name = ['Embd_DNet', 'Embd_DNet', 'VAE', 'Deep_AE']
    _class = class_name[model_id - 1]
    if model_id == 1: basic_module = 'VAE'
    if model_id == 2: basic_module = 'DAE'
    
    if data_set == 1:
        v_dim = 7
        in_dim = dynamic * v_dim
        # dropout = 0
        dropout = dropout
        dataset_name = 'CSTR'
        path = '../data/CSTR/fd_sensor'
        datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, example = 'CSTR', seg_name = [0, -1]).datasets
        labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08']
    elif data_set == 2:
        v_dim = 7
        in_dim = dynamic * v_dim
        # dropout = 0
        dropout = dropout
        dataset_name = 'CSTR'
        path = '../data/CSTR/fd_close'
        datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, example = 'CSTR', seg_name = [0, -1]).datasets
        labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08']
    
    if fd_mode == 1:
        thrd_sele = 'ineq'
    elif fd_mode == 2:
        thrd_sele = 'kde'
    fd_dict={'fdi': fdi,
             'test_stat': 'T2',
             'thrd_sele': thrd_sele,
             'ineq_rv': 'indi_mean',
             'confidence': 1 - 0.5/100}
    
    # sturct id
    decoder_func = 'a'
    if struct_id == 1:
        struct = [in_dim, v_dim * 20, 20]
        hidden_func = ['a']
        decoder_func = ['g','a']
    elif struct_id == 2:
        struct = [in_dim, v_dim * 20, 20]
        hidden_func = ['s']
        decoder_func = ['g','a']
    elif struct_id == 3:
        struct = [in_dim, v_dim * 20, 20]
        hidden_func = ['g']
        decoder_func = ['g','a']
    elif struct_id == 4:
        struct = [in_dim, v_dim * 20, 20]
        hidden_func = ['q']
        decoder_func = ['g','a']
    elif struct_id == 5:
        struct = [in_dim, v_dim * 20, 50, 10]
        hidden_func = ['q','q']
        decoder_func = ['s','s','a']
    elif struct_id == 6:
        struct = [in_dim, v_dim * 20, 50, 10]
        hidden_func = ['q','a']
        decoder_func = ['s','s','a']
    elif struct_id == 7:
        struct = [in_dim, v_dim * 20, 50, 10]
        hidden_func = ['q','s']
        decoder_func = ['s','s','a']
    else:
        struct = [in_dim, v_dim * 20, 50, 10]
        hidden_func = ['g','a']
        decoder_func = ['s','s','a']
    

    parameter = {'add_info':'_fd_' + dataset_name + '_' + str(struct_id),
                 'dvc': 'cuda', 
                 'basic_module': basic_module,
                 'label_name': labels,
                 
                 'struct': struct,
                 'hidden_func': hidden_func,
                 'decoder_func': decoder_func,
                 'output_func': 'a',
                 
                 'n_used_variables': 2,
                 # 'input_noise': 1e-3,
                 'if_inner': True,
                  # 'if_inner': False,
                 'toeplitz_mode': '1',
                 # 'if_times_input_dim': True,
                 
                 'v0_2': 1e-3,
                 'sample_times': 5, 
                 'dropout': 0,
                 'task': 'fd',
                 'view_res': False,
                 'expt_FAR': 0.5/100,
                 'esti_error': 0.005,
                 'alf': 1,
                 'gamma': 1,
                 'gamma_out': 1,
                 # 'optim':'Adam',
                 'optim':'RMSprop',
                 'optim_para': 'alpha=0.9, eps=1e-10',
                 'lr': 1e-4
                 }
    
    parameter.update(fd_dict)
    
    model = eval(_class + '(**parameter)')
    return model, datasets, labels

if __name__ == '__main__':
    model, datasets, labels = get_sae_model(data_set = 1,
                                            model_id = 1,         # Embd VAE DAE
                                            fd_mode = 1,          # ineq kde
                                            fdi = 'res',
                                            # fdi = 'custo',
                                            
                                            struct_id = 3,        # 结构 1 至 8

                                            dropout = 0.0         # dropout
                                            # dropout = 0.05,
                                            )
    
    load_ = True
    # load_ = False
    if load_:
        model._save_load('load','last')
        estimated_f = model._bp_update_inputs(datasets = datasets, b = 16, dvc = 'cuda')
    else:
        model.run(datasets = datasets, e = 20, b = 16, load = '', cpu_core = 0.8, num_workers = 0)
        model.result(labels, True)
    