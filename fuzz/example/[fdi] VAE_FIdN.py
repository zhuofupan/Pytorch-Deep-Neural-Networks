# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 17:21:40 2022

@author: Fuzz4
"""
from fuzz.data.gene_dynamic_data import ReadData
from fuzz.model.variant.vae_fidn import VAE_FIdN

def get_sae_model(dynamic = 1, struct_id = 1, dropout = 0, alf = 1):
        
    
    v_dim = 10
    in_dim = dynamic * v_dim
    dropout = dropout
    dataset_name = 'CSTR'
    
    path = '../data/CSTR/fi'
    datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                        is_del = False, example = 'CSTR', single_mode = True).datasets
    labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
              'Fault 08', 'Fault 09', 'Fault 10']
    
    fd_dict = {'res_generator': 'lv',
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
    elif struct_id == 8:
        struct = [in_dim, v_dim * 20, 50, 10]
        hidden_func = ['g','a']
        decoder_func = ['s','s','a']
    
    
    parameter = {'add_info':'_' + dataset_name + '_' + str(struct_id),
                 'dvc': 'cuda',
                 'decp_struct': [in_dim, in_dim*20, in_dim*20, in_dim],
                 'decp_func': ['a', 'q'],
                 'struct': struct,
                 'hidden_func': hidden_func,
                 'decoder_func': decoder_func,
                 'dropout': dropout,
                 'task': 'fd',
                 'expt_FAR': 0.5/100,
                 'esti_error': 0.005,
                 'lr': 1e-4,
                 'lr_tl': 1e-4,
                 'dvd_epoch': 0.36,         # 分割训练代数用于 before_transfer 和 transfer
                 'L': 'MSE',
                 'alf': alf,
                 'alf_mmd': 1,
                 'gene_f_max': 12,
                 'gene_f_min': 1.25,
                 'gene_new_data': True,
                 'shuffle_Y': True,
                 'label_name': labels}
    
    parameter.update(fd_dict)
    
    model = VAE_FIdN(**parameter)
    return model, datasets, labels

if __name__ == '__main__':
    model, datasets, labels = get_sae_model(struct_id = 3,        # 结构
                                            alf = 1              # D_KL系数
                                            # dropout = 0.382       # dropout
                                            )
 
    model.run(datasets = datasets, e = 30, b = 16)
    model.result(labels, True)
    