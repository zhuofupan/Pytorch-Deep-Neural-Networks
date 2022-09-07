# -*- coding: utf-8 -*-

from fuzz.model.vae import VAE
from fuzz.model.dae import Deep_AE
from fuzz.data.gene_dynamic_data import ReadData
from fuzz.data.read_hy_data import ReadHYData
from fuzz.xai.lrp import LRP
from fuzz.xai.deeplift import DeepLIFT
from fuzz.xai.int_grad import Int_Grad
from fuzz.xai.guided_bp import Guided_BP
from fuzz.xai.picp import PICP
from fuzz.xai.lcp import LCP
from fuzz.xai.grad_times_input import Grad_T_Input
from fuzz.xai.sensitivity_analysis import Sens_Anal

def get_dae_model(data_set = 1, model_id = 1, struct_id = 1, 
                  dynamic = 40, dropout = 0.382):
    # modle id
    class_name = ['VAE', 'Deep_AE']
    _class = class_name[model_id - 1]
    
    # TE
    if data_set == 1:
        v_dim = 33
        in_dim = dynamic * v_dim
        dropout = dropout
        dataset_name = 'TE'
        datasets = ReadData('../data/TE', ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', example = 'TE').datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 04','Fault 05', 'Fault 06', 'Fault 07', 'Fault 08', 
                  'Fault 10', 'Fault 11', 'Fault 12', 'Fault 13', 'Fault 14', 'Fault 16', 'Fault 17', 'Fault 18',
                  'Fault 19', 'Fault 20', 'Fault 21']
    # CSTR
    elif data_set == 2 or data_set == 3:
        dataset_name = 'CSTR'
        if data_set == 2:
            v_dim = 10
            path = '../data/CSTR/fd'
            datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, example = 'CSTR').datasets
            labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                      'Fault 08', 'Fault 09', 'Fault 10']
            variables = ['n','m','m','m', 3, 4, 8, 5, 6, 9, 7]
        elif data_set == 3:
            v_dim = 7
            path = '../data/CSTR/fd_close'
            datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, example = 'CSTR', seg_name = [0, -1]).datasets
            labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                      'Fault 08']
            variables = ['n','m','m', 0, 1, 2, (3,5), 4, 6]
            
        in_dim = dynamic * v_dim
        dropout = dropout
        
    # HY
    elif data_set == 4:
        v_dim = 61
        in_dim = dynamic * v_dim
        dropout = dropout
        datasets = ReadHYData('../data/hydrocracking/hydrocracking.xls',
                              ['st', 'oh'], dynamic).datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08']
        dataset_name = 'HY'
    
    # struct
    output_func = ['a']
    if struct_id == 1:
        act_name = 'gas'
        struct = [in_dim, '*20', '/2']
        hidden_func, decoder_func = ['g','a'], ['s']
    elif struct_id == 2:
        act_name = 'rtg'
        struct = [in_dim, '*20', '/2']
        hidden_func, decoder_func = ['r','t'], ['g']
    elif struct_id == 3:
        act_name = 'rss'
        struct = [in_dim, '*20', '/2']
        hidden_func, decoder_func = ['r','s'], ['s']
    elif struct_id == 4:
        act_name = 'gagag'
        struct = [in_dim, '*20', '/2', '/2']
        hidden_func, decoder_func = ['g','a','g'], ['a','g']
    elif struct_id == 5:
        act_name = 'rrgrg'
        struct = [in_dim, '*20', '/2', '/2']
        hidden_func, decoder_func = ['r','r','g'], ['r','g']
    elif struct_id == 6:
        act_name = 'tatat'
        struct = [in_dim, '*20', '/2', '/2']
        hidden_func, decoder_func = ['t','a','t'], ['a','t']
    
    # FD
    fd_dict={'fdi': 'res',
             'test_stat': 'T2',
             'thrd_sele': 'ineq',
             'ineq_rv': 'indi_mean',
             'confidence': 1 - 0.5/100}
    
    parameter = {'add_info':'_attri_' + dataset_name + '_' + act_name,
                 'dvc': 'cuda', 
                 '__drop__': 70,
                 'struct': struct,
                 'label_name': labels,
                 'hidden_func': hidden_func,
                 'decoder_func': decoder_func,
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
                 'lr': 1e-4
                 }
        
    parameter.update(fd_dict)
    model = eval(_class + '(**parameter)')
    
    return model, datasets, labels, variables

def _run_dae(data_set, model_id, struct_id, dynamic, dropout, e, b, obj, attri_id):
    if attri_id == 1: _class = 'LRP'
    elif attri_id == 2: _class = 'DeepLIFT'
    elif attri_id == 3: _class = 'Int_Grad'
    elif attri_id == 4: _class = 'PICP'
    elif attri_id == 5: _class = 'LCP'
    
    model, datasets, labels, variables = \
        get_dae_model(data_set = data_set, model_id = model_id,\
                      struct_id = struct_id, dynamic = dynamic,\
                      dropout = dropout)
    if obj == 'train':
        model.run(datasets = datasets, e = e, b = b, load = '', cpu_core = 0.8, num_workers = 0)
        model.result(labels, True)
    elif obj == 'attri':
        model._save_load('load', 'last')
        eval(_class)(model = model, datasets = datasets, batch_size = b, 
            labels = labels, real_root_cause = variables
            # if_show_debug_info = [False, True]
            )
    
'''
    故障1：C Sensor Bias (4)
    故障2：T Sensor Bias (5)
    故障3：Tc Sensor Bias (6)
    故障4：Ci and Ti Sensor Bias (1,2)
    故障5：Tci and Qc Sensor Bias (3,7)
    故障6：Ci, Ti and Tci Sensor Bias (1,2,3)
    故障7：Catalyst Fault (4,5,6,7)
    故障8：HTC Fault (4,5,6,7)
'''
if __name__ == '__main__':
    data_set, model_id, dynamic, struct_id = 3, 2, 1, 6
    n_interpolation, attri_id = 30, 5
    dropout = 0.3
    e, b = 15, 16
    # task = 'fd'
    obj = 'attri'
    
    _run_dae(data_set = data_set, model_id = model_id, struct_id = struct_id,
              dynamic = dynamic, dropout = dropout, 
              e = e, b = b, obj = obj, attri_id = attri_id)
    
    # attri_dae(data_set = data_set, model_id = model_id, struct_id = struct_id, 
    #           dynamic = dynamic, dropout = dropout,
    #           judge_rule = 1, n_interpolation = n_interpolation, attri_id = attri_id
    #           )