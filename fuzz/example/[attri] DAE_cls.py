# -*- coding: utf-8 -*-

from fuzz.model.sae import SAE
from fuzz.data.gene_dynamic_data import ReadData
from fuzz.data.read_hy_data import ReadHYData
from fuzz.xai.lrp import LRP
from fuzz.xai.deeplift import DeepLIFT
from fuzz.xai.int_grad import Int_Grad
from fuzz.xai.guided_bp import Guided_BP
from fuzz.xai.lcg_bp import LCG_BP
from fuzz.xai.grad_times_input import Grad_T_Input
from fuzz.xai.sensitivity_analysis import Sens_Anal

def get_sae_model(model_id = 1, dynamic = 40, ae_type = 'AE'):
    if dynamic > 1:
        __drop__ = [True, True]
    else:
        __drop__ = [False, False]
    
    # TE
    if model_id == 0:
        n_v, n_category = 33, 19
        if dynamic > 1: dropout = 0.382
        else: dropout = 0.082
        struct = [dynamic*n_v, 660, 440, 220, n_category]
        datasets = ReadData('../data/TE', ['st', 'oh'], dynamic, cut_mode = '', example = 'TE').datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 04','Fault 05', 'Fault 06', 'Fault 07', 'Fault 08', 
                  'Fault 10', 'Fault 11', 'Fault 12', 'Fault 13', 'Fault 14', 'Fault 16', 'Fault 17', 'Fault 18',
                  'Fault 19', 'Fault 20', 'Fault 21']
        
        hidden_func = ['g', 'a', 'a']
        output_func = 'g'
        name = 'TE_'+ae_type+'_'+str(dynamic)
        
    # CSTR
    elif model_id < 10:
        n_v, n_category = 10, 11
        dropout = 0.082
        datasets = ReadData('../data/CSTR', ['st', 'oh'], dynamic, cut_mode = '', example = 'CSTR').datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08', 'Fault 09', 'Fault 10']
        
        if model_id == 1:
            name = 'GAS'
            struct = [dynamic*n_v, 200, 100, n_category]
            hidden_func = ['g', 'a']
            output_func = 's'
        elif model_id == 2:
            name = 'RTG'
            struct = [dynamic*n_v, 200, 100, n_category]
            hidden_func = ['r', 't']
            output_func = 'g'
        elif model_id == 3:
            name = 'RSX'
            struct = [dynamic*n_v, 200, 100, n_category]
            hidden_func = ['r', 's']
            output_func = 'x'
        elif model_id == 4:
            name = 'GAAG'
            struct = [dynamic*n_v, 200, 100, 50, n_category]
            hidden_func = ['g', 'a', 'a']
            output_func = 'g'
        elif model_id == 5:
            name = 'RRRG'
            struct = [dynamic*n_v, 200, 100, 50, n_category]
            hidden_func = ['r', 'r', 'r']
            output_func = 'g'
        elif model_id == 6:
            name = 'TATA'
            struct = [dynamic*n_v, 200, 100, 50, n_category]
            hidden_func = ['t', 'a', 't']
            output_func = 'a'
        elif model_id == 7:
            name = 'STSRX'
            struct = [dynamic*n_v, 200, 150, 100, 50, n_category]
            hidden_func = ['s', 't', 's', 'r']
            output_func = 'x'
        elif model_id == 8:
            name = 'SSSAG'
            struct = [dynamic*n_v, 200, 150, 100, 50, n_category]
            hidden_func = ['s', 's', 's', 'a']
            output_func = 'g'
        elif model_id == 9:
            name = 'ARARA'
            struct = [dynamic*n_v, 200, 150, 100, 50, n_category]
            hidden_func = ['a', 'r', 'a', 'r']
            output_func = 'a'
        name = 'CSTR_'+ae_type+'_'+name+'_'+str(dynamic)
    
    # HY
    elif model_id == 10:
        n_v, n_category = 61, 9
        dropout = 0.282
        datasets = ReadHYData('../data/hydrocracking/hydrocracking.xls',
                              ['st', 'oh'], dynamic).datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08']
        
        ae_type = 'CG-AE'
        struct = [dynamic*n_v, 500, 300, 100, n_category]
        hidden_func = ['g','a','a']
        output_func = 'g'
        name = 'HY_'+ae_type+'_'+'GAAG'+'_'+str(dynamic)
        

    parameter = {'name':name,
                 'ae_type': ae_type,
                 '__drop__': __drop__,
                 'dvc': 'cuda', 
                 'n_category': n_category,
                 'label_name': labels,
                 'struct': struct,
                 'hidden_func': hidden_func,
                 'output_func': output_func,
                 'alf': 1,
                 'task': 'cls',
                 'optim':'RMSprop',
                 'optim_para': 'alpha=0.9, eps=1e-10',
                 'lr': 1e-4, 
                 'pre_lr': 1e-3,
                 'dropout': dropout,
                 'tsne_info': False}
        
    model = SAE(**parameter)
    return model, datasets, labels

def train_sae(model_id = 1, dynamic = 40, ae_type = 'AE'):
    model, datasets, labels = get_sae_model(model_id, dynamic = dynamic, ae_type = ae_type)
    model.run(datasets = datasets, e = 180, b = 16, pre_e = 15, load = '', tsne = False, 
              cpu_core = 0.8, num_workers = 0)
    model.result(labels)
    
def attri_sae(model_id = 1, dynamic = 40, ae_type = 'AE', 
              judge_rule = 1, n_interpolation = 10, _class_id = 1):
    model, datasets, labels = get_sae_model(model_id, dynamic = dynamic, ae_type = ae_type)
    parameter = {'model': model,
                 'judge_rule': judge_rule,
                 'basaline_goal': 'input',
                 # 'basaline_goal': 'output',
                 'manual_cal': True,
                 'show_op_info': False}
    if _class_id == 6: parameter['n_interpolation'] = n_interpolation
    
    if _class_id == 1: _class = 'Sens_Anal'
    elif _class_id == 2: _class = 'Grad_T_Input'
    elif _class_id == 3: _class = 'Guided_BP'
    elif _class_id == 4: _class = 'LRP'
    elif _class_id == 5: _class = 'DeepLIFT'
    elif _class_id == 6: _class = 'Int_Grad'
    elif _class_id == 7: _class = 'LCG_BP'
    
    attri_md = eval(_class + '(**parameter)')
    attri_md.test_attri(datasets = datasets, 
                        dynamic = dynamic, 
                        labels = labels,
                        dvc = 'cpu')

if __name__ == '__main__':
    # train_sae(model_id = 8, 
    #           dynamic = 1
    #           )
    
    attri_sae(model_id = 8, 
              dynamic = 40, 
              ae_type = 'AE', 
              judge_rule = 1, 
              n_interpolation = 30,
              _class_id = 7
              )