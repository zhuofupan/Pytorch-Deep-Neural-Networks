from fuzz.model.dae import Deep_AE
from fuzz.model.vae import VAE
from fuzz.core.fd_msa import MSA
from fuzz.data.gene_dynamic_data import ReadData

def get_sae_model(data_set = 1, dynamic = 1, model_id = 1, del_3 = False,
                  struct_id = 1, dropout = 0.382, alf = 1, gamma = 1):

    # modle id
    class_name = ['VAE', 'Deep_AE', 'pca', 'kpca']
    _class = class_name[model_id - 1]
        
    # data set
    split_p_list = None
    plot_p_list = None
    if data_set == 1:
        dropout = dropout
        dataset_name = 'TE'
        datasets = ReadData('../data/TE', ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', example = 'TE').datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 04','Fault 05', 'Fault 06', 'Fault 07', 'Fault 08', 
                  'Fault 10', 'Fault 11', 'Fault 12', 'Fault 13', 'Fault 14', 'Fault 16', 'Fault 17', 'Fault 18',
                  'Fault 19', 'Fault 20', 'Fault 21']
    elif data_set <=4:
        
        # dropout = 0
        dropout = dropout
        
        seg_name = None
        if data_set == 2:
            path = '../data/CSTR/fd'
            dataset_name = 'CSTR_FD'
        elif data_set == 3:
            path = '../data/CSTR/fi'
            dataset_name = 'CSTR_FI'
        elif data_set == 4:
            path = '../data/CSTR/fd_close'
            dataset_name = 'CSTR_Close'
            seg_name = [0,-1]
            
        datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, seg_name = seg_name, example = 'CSTR').datasets
        if del_3:
            train_x, train_y, test_x, test_y = datasets
            datasets = train_x[:,3:], train_y, test_x[:,3:], test_y
        
        labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08', 'Fault 09', 'Fault 10']
    
    elif data_set == 5:
        path = '../data/Multiphase_Flow_Facility'
        dataset_name = 'MFF'
        RD = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, example = 'MFF')
        datasets, split_p_list, plot_p_list = RD.datasets, RD.split_p_list, RD.plot_p_list
        labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06']
        
    v_dim = datasets[0].shape[1]
    in_dim = dynamic * v_dim
        
    fd_dict={'fdi': 'res',
             'test_stat': 'T2',
             'thrd_sele': 'ineq',
             'n_components': struct_id,
             'split_p_list': split_p_list,
             'plot_p_list': plot_p_list,
             'confidence': 1 - 0.5/100}
    
    # sturct id
    decoder_func = 'a'
    output_func = 'a'
    if struct_id == 1:
        struct = [in_dim, v_dim * 10, '/2']
        hidden_func = ['a','a']
        decoder_func = ['a','a']
    elif struct_id == 2:
        struct = [in_dim, v_dim * 10, '/2']
        hidden_func = ['s','s']
        decoder_func = ['s','a']
    elif struct_id == 3:
        struct = [in_dim, v_dim * 10, '/2']
        hidden_func = ['g','s']
        decoder_func = ['s','a']
    elif struct_id == 4:
        struct = [in_dim, v_dim * 10, '/2', '/2']
        hidden_func = ['q','s','s']
        decoder_func = ['q','s','a']
    elif struct_id == 5:
        struct = [in_dim, v_dim * 10, '/2', '/2']
        hidden_func = ['g','g','s']
        decoder_func = ['g','a','a']
    elif struct_id == 6:
        struct = [in_dim, v_dim * 10, '/2', '/2']
        hidden_func = ['t','t','s']
        decoder_func = ['t','s','a']
    
    if struct_id <= 6:
        parameter = {'add_info':'_S{}_'.format(struct_id) + dataset_name,
                     'dvc': 'cuda', 
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
                     'alf': alf,
                     'gamma': gamma,
                     'optim':'RMSprop',
                     'optim_para': 'alpha=0.9, eps=1e-10',
                     'lr': 1e-4
                     }
        
        parameter.update(fd_dict)
    
    if model_id <= 2:
        model = eval(_class + '(**parameter)')
    else:
        # fd_dict['fdi'] = 'lv_pc'
        # fd_dict['fdi'] = 'lv_res'
        fd_dict['test_stat'] = 'SPE'
        # fd_dict['thrd_sele'] = 'pdf'
        model = MSA(mthd = _class, 
                    add_info = '_S{}_{}_{}_{}'.format(struct_id, fd_dict['fdi'].title(), \
                                fd_dict['test_stat'].title(), fd_dict['thrd_sele'].title()) \
                                + dataset_name,
                    **fd_dict)
    return model, datasets, labels

if __name__ == '__main__':
    model, datasets, labels = get_sae_model(data_set = 4,
                                            # del_3 = True,
                                            
                                            model_id = 2,         # 1.VAE 2.DAE
                                            struct_id = 5,        # 结构
                                            
                                            alf = 1,
                                            gamma = 1,
                                            dropout = 0        # dropout
                                            # dropout = 0.382
                                            )
 
    model.run(datasets = datasets, e = 30, b = 16, load = '', cpu_core = 0.8, num_workers = 0)
    model.result(labels, True)