# -*- coding: utf-8 -*-
import sys

from fuzz.data.gene_dynamic_data import ReadData
from fuzz.data.read_hy_data import ReadHYData

from fuzz.model.sae import SAE
from fuzz.model.variant.sup_sae import SupSAE
from fuzz.model.variant.mmdgm_vae import MMDGM_VAE
from fuzz.model.variant.ocon import OCON
from fuzz.model.dbn import DBN
from fuzz.model.dnn import DNN
from fuzz.core.run_n import Run_N

def get_sae_model(model_id = 1, struct_id = 1, data_set = 1, dynamic = 40, 
                  drop_label_rate = 0):
    loss_func = 1
    dropout = 0.382
    if dynamic == 1: dropout = 0.
    pre_dropout = False
    if loss_func == 1:
        L = 'MSE'
        pre_L = 'MSE'
    elif loss_func == 2:
        L = 'MSE'
        pre_L = 'CrossEntropy'
    elif loss_func == 3:
        L = 'CrossEntropy'
        pre_L = 'MSE'
    elif loss_func == 4:
        L = 'CrossEntropy'
        pre_L = 'CrossEntropy'
        dropout = 0.316
    
    # modle id
    if model_id == 1:
        name, ae_type, _class = 'CG-SAE', 'CG-AE', 'SAE'
    elif model_id == 2:
        name, ae_type, _class = 'SAE', 'AE', 'SAE'
    elif model_id == 3:
        name, ae_type, _class = 'DBN', None, 'DBN'
    elif model_id == 4:
        name, ae_type, _class = 'SupSAE', 'SupAE', 'SupSAE'
    elif model_id == 5:
        name, ae_type, _class = 'MMDGM_VAE', None, 'MMDGM_VAE'
    elif model_id == 6:
        pre_dropout = True
        name, ae_type, _class = 'YSupSAE', 'YSupAE', 'SAE'
    elif model_id == 7:
        name, ae_type, _class = 'DNN', None, 'DNN'
    elif model_id == 8:
        name, ae_type, _class = 'OCON', None, 'OCON'
        
    # data set
    if data_set == 1:
        in_dim, out_dim = dynamic * 33, 19
        datasets = ReadData('../data/TE', ['st', 'oh'], dynamic, cut_mode = '', example = 'TE', 
                            drop_label_rate = drop_label_rate).datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 04','Fault 05', 'Fault 06', 'Fault 07', 'Fault 08', 
                  'Fault 10', 'Fault 11', 'Fault 12', 'Fault 13', 'Fault 14', 'Fault 16', 'Fault 17', 'Fault 18',
                  'Fault 19', 'Fault 20', 'Fault 21']
    elif data_set == 2:
        in_dim, out_dim = dynamic * 10, 11
        dropout = 0.082
        datasets = ReadData('../data/CSTR', ['st', 'oh'], dynamic, cut_mode = '', example = 'CSTR',
                            drop_label_rate = drop_label_rate).datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08', 'Fault 09', 'Fault 10']
    elif data_set == 3:
        # dynamic = 18
        in_dim, out_dim = dynamic * 61, 9
        div_prop = 0.7
        if div_prop > 0:
            path = '../data/hydrocracking/hydrocracking.xls'
            datasets = ReadHYData(path,['st', 'oh'], dynamic, div_prop = div_prop).datasets
        else:
            path = '../data/hydrocracking/gene'
            datasets = ReadData(path,['st', 'oh'], dynamic).datasets
        
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08']
        
    # sturct id
    if struct_id == 1:
        struct = [in_dim,600,out_dim]
        hidden_func, output_func = 'g', 'g'
    elif struct_id == 2:
        struct = [in_dim,600,200,out_dim]
        hidden_func, output_func = ['g','a'], 'g'
    elif struct_id == 3:
        struct = [in_dim,600,200,out_dim]
        hidden_func, output_func = ['g','g'], 'g'
    elif struct_id == 4:
        struct = [in_dim,600,400,200,out_dim]
        hidden_func, output_func = ['g','a','a'], 'g'
    elif struct_id == 5:
        struct = [in_dim,600,400,200,out_dim]
        hidden_func, output_func = ['g','a','g'], 'g'
    elif struct_id == 6:
        struct = [in_dim,600,400,200,out_dim]
        hidden_func, output_func = ['g','g','g'], 'g'
    elif struct_id == 7:
        struct = [in_dim,600,400,200,100,out_dim]
        hidden_func, output_func = ['g','a','g','a'], 'g'
    elif struct_id == 8:
        struct = [in_dim,600,400,200,100,out_dim]
        hidden_func, output_func = ['g','g','g','g'], 'g'
    if model_id == 5:
        struct[-1] = 100
    
    alf = 0.9
    name += ' dy-{}, alf-{}, dlr-{}'.format(dynamic, alf, drop_label_rate)
    parameter = {'name': name,
                 'ae_type': ae_type,
                 '__drop__': [True, True],
                 'dvc': 'cuda', 
                 'n_category': 19,
                 'label_name': labels,
                 'struct': struct,
                 'hidden_func': hidden_func,
                 'output_func': output_func,
                 'alf': alf,
                 'task': 'cls',
                 'optim':'RMSprop',
                 'optim_para': 'alpha=0.9, eps=1e-10',
                 'L': L, 'pre_L': pre_L,
                 'lr': 1e-4, 'pre_lr': 1e-3,
                 'dropout': dropout, 'pre_dropout': pre_dropout,
                 'tsne_info': False}
    model = eval(_class + '(**parameter)')
    return model, datasets, labels


if __name__ == '__main__':
    model_id, struct_id, data_set, dynamic, drop_label_rate = 1, 4, 1, 40, 0.5
    model, datasets, labels = get_sae_model(model_id, struct_id, data_set, dynamic, 
                                            drop_label_rate)
    
    if data_set == 1: e = 240
    else: e = 180
    model.run(datasets = datasets, e = e, b = 16, pre_e = 15, load = '', tsne = False, 
              cpu_core = 0.8, num_workers = 0)
    model.result(labels)
    
    # run_info = ' {}s'.format(struct_id)
    # Run_N(model, 3, run_info).run(datasets = datasets, e = 120, b = 16, pre_e = 15, load = '', 
    #                                 cpu_core = 0.8, num_workers = 0)