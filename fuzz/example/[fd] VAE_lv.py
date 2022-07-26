from fuzz.model.dae import Deep_AE
from fuzz.model.vae import VAE
from fuzz.core.fd_msa import MSA
from fuzz.data.gene_dynamic_data import ReadData

def get_sae_model(data_set = 1, dynamic = 1, model_id = 1, fdi = 'lv',
                  struct_id = 1, dropout = 0.382, alf = 1, fd_mode = 1, 
                  select_sample_mthd = 'rd500'):

    # modle id
    class_name = ['VAE', 'Deep_AE', 'pca', 'kpca']
    _class = class_name[model_id - 1]
        
    # data set
    if data_set == 1:
        v_dim = 33
        in_dim = dynamic * v_dim
        dropout = dropout
        dataset_name = 'TE'
        datasets = ReadData('../data/TE', ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', example = 'TE').datasets
        labels = ['Normal', 'Fault 01', 'Fault 02', 'Fault 04','Fault 05', 'Fault 06', 'Fault 07', 'Fault 08', 
                  'Fault 10', 'Fault 11', 'Fault 12', 'Fault 13', 'Fault 14', 'Fault 16', 'Fault 17', 'Fault 18',
                  'Fault 19', 'Fault 20', 'Fault 21']
    elif data_set == 2 or data_set == 3:
        v_dim = 10
        in_dim = dynamic * v_dim
        # dropout = 0
        dropout = dropout
        dataset_name = 'CSTR'
        if data_set == 2:
            path = '../data/CSTR/fd'
        else:
            path = '../data/CSTR/fi'
        datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, example = 'CSTR').datasets
        labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08', 'Fault 09', 'Fault 10']
    elif data_set == 4:
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
    elif data_set == 5:
        v_dim = 5
        in_dim = dynamic * v_dim
        # dropout = 0
        dropout = dropout
        dataset_name = 'TTS'
        path = '../data/Three_Tank_System'
        datasets = ReadData(path, ['st', 'oh'], dynamic, task = 'fd', cut_mode = '', 
                            is_del = False, example = 'TTS', set_normal = 201, set_for = [1]).datasets
        labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08', 'Fault 09', 'Fault 10', 'Fault 11', 'Fault 12']
    
    if fd_mode == 1:
        thrd_sele = 'ineq'
    elif fd_mode == 2:
        thrd_sele = 'kde'
    elif fd_mode == 3:
        thrd_sele = 'pdf'
    fd_dict={'n_components': struct_id,
             'fdi': fdi,
             'select_sample_mthd': select_sample_mthd,
             'test_stat': 'T2&SPE',
             'thrd_sele': thrd_sele,
             'ineq_rv': 'indi_mean',
             'confidence': 1 - 0.5/100}
    
    # sturct id
    decoder_func = 'a'
    output_func = 'a'
    
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
                 'optim':'RMSprop',
                 'optim_para': 'alpha=0.9, eps=1e-10',
                 'lr': 1e-4
                 }
    
    parameter.update(fd_dict)
    
    if model_id <= 2:
        model = eval(_class + '(**parameter)')
    else:
        model = MSA(mthd = _class, 
                    add_info = '_fd_'+ dataset_name + '_' + str(struct_id),
                    **fd_dict)
    return model, datasets, labels

if __name__ == '__main__':
    model, datasets, labels = get_sae_model(data_set = 4,
                                            model_id = 1,         # VAE 1, 3, 4
                                            fd_mode = 1,          # 统计量 model_id = 1 (1 和 2)
                                                                  # model_id = 3 和 4 (1, 2, 3)
                                            fdi = 'lv',
                                            select_sample_mthd = 'rd1000',
                                            
                                            struct_id = 6,        # 结构 model_id = 1, 1 至 8
                                            
                                            # struct_id = 0.15,   # 结构 model_id = 4, 0.1 到 0.5
                                                                  # 结构 model_id = 3, 1 至 5
                                            alf = 1e2,            # D_KL系数
                                            dropout = 0.0         # dropout
                                            # dropout = 0.05,
                                            )

    model.run(datasets = datasets, e = 12, b = 16, load = '', cpu_core = 0.8, num_workers = 0)
    model.result(labels, True)
    