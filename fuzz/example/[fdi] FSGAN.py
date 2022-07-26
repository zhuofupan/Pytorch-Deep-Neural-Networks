# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 15:16:23 2022

@author: Fuzz4
"""

from fuzz.data.gene_dynamic_data import ReadData
from fuzz.model.variant.gan_fdi import GAN_FDI

def get_sae_model(dynamic = 1, struct_id = 1, dropout = 0, alf = 1):
        
    
    v_dim = 10
    in_dim = dynamic * v_dim
    dropout = dropout
    dataset_name = 'CSTR_Y'
    
    path = '../data/CSTR/fi'
    datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                        is_del = False, example = 'CSTR', single_mode = True).datasets
    labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 
              'Fault 07', 'Fault 08']
    
    fd_dict = {'res_generator': 're',
               'test_stat': 'T2',
               'thrd_sele': 'ineq',
               'ineq_rv': 'indi_mean',
               'save_res': True,
               # 'if_minus_mean': False,
               'confidence': 1 - 0.5/100}
    
    # sturct id
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
        hidden_func = ['g','a']
        decoder_func = ['g']
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
    elif struct_id == 8:
        struct = [in_dim, v_dim * 20, 50, 10]
        hidden_func = ['g','a']
        decoder_func = ['s','s','a']
    
    
    parameter = {'add_info':'_' + dataset_name,
                 'dvc': 'cuda',
                 'struct': struct,
                 'hidden_func': hidden_func,
                 'decoder_func': decoder_func,
                 'dropout': dropout,
                 'task': 'fd',
                 'expt_FAR': 0.5/100,
                 'esti_error': 0.005,
                 'lr': 1e-4,
                 'L': 'MSE',
                 'alf': alf,
                 'gene_f_max': 12,
                 'gene_f_min': 1.25,
                 'lambda_gp': 1,
                 'n_critic': 1,
                 'shuffle_Y': True,
                 'use_wgan': False,
                 'label_name': labels}
    
    parameter.update(fd_dict)
    
    model = GAN_FDI(**parameter)
    return model, datasets, labels

if __name__ == '__main__':
    model, datasets, labels = get_sae_model(struct_id = 3,          # 结构
                                            alf = 10,               # Recon系数
                                            dropout = 0.382         # dropout
                                            )

    model.run(datasets = datasets, e = 3, b = 16)
    model.result(labels, True)
    