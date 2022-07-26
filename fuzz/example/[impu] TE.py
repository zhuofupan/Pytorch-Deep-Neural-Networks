# -*- coding: utf-8 -*-
from fuzz.model.dae import Deep_AE
from fuzz.model.gan import GAN
from fuzz.model.variant.trdae import TRDAE
from fuzz.model.variant.gain import GAIN
from fuzz.model.variant.igani import IGANI
from fuzz.core.run_n import Run_N

def get_model(model_id, dynamic, n_v, dropout, batch_size, missing_rate, 
              path = '../data/TE', example = 'TE'):
    
    struct = [dynamic *  n_v, '/2', '*1']
    
    if model_id == 1:
        # Deep_AE
        _class = 'Deep_AE'
        name = 'Deep_AE'
        how_impu = 'replace'
        compt_loss = 'all'
        hidden_func = ['lr', 'lr']
        decoder_func = hidden_func
    elif model_id == 2:
        # SM-DAE
        _class = 'Deep_AE'
        name = 'SM_DAE'
        how_impu = 'mid'
        compt_loss = 'adv'
        hidden_func = ['lr', 'lr']
        decoder_func = hidden_func
    elif model_id == 3:
        # SDAi
        _class = 'Deep_AE'
        name = 'SDAi'
        how_impu = 'replace'
        compt_loss = 'complete'
        hidden_func = ['lr', 'lr']
        decoder_func = hidden_func
    elif model_id == 4:
        # TRAE
        _class = 'TRDAE'
        struct = [dynamic *  n_v, '/2']
        name = 'TRAE'
        how_impu = 'grad'
        compt_loss = 'all'
        hidden_func = ['lr', 'lr']
        decoder_func = hidden_func
    elif model_id == 5:
        # GAIN
        _class = 'GAIN'
        name = 'GAIN'
        how_impu = 'replace'
        hidden_func = ['lr', 'lr']
    elif model_id == 6:
        # GAN
        _class = 'GAN'
        name = 'GAN'
        how_impu = 'replace'
        hidden_func = ['lr', 'lr']
    
    name = name + ' [' +str(missing_rate) +']'
    if model_id <= 4:
        __drop__ = [False, True]
        coff_grad = 1e3
        # coff_grad = 3e2
        parameter = {'dvc': 'cuda', 
                     'name': name,
                     'struct': struct,
                     'hidden_func': hidden_func,
                     'decoder_func': decoder_func,
                     'how_impu': how_impu,
                     'compt_loss': compt_loss,
                     'coff_grad': coff_grad,
                     'task': 'impu',
                     'optim':'RMSprop',
                     'optim_para': 'alpha=0.9, eps=1e-10', # centered=True',
                     'lr': 1e-4, 
                     'dropout': dropout,
                     '__drop__': __drop__}
    else:
        __drop__ = [False, True]
        struct = [dynamic *  n_v, '/2', '*1', '*1', dynamic *  n_v]
        dicr_func = ['lr', 'lr', 'lr', 's']
        alf = 1e3
        parameter = {'dvc': 'cuda',
                     'name': name,
                     'struct': struct,
                     'hidden_func': hidden_func,
                     'dicr_struct': struct,
                     'dicr_func': dicr_func,
                     'how_impu': how_impu,
                     'alf': alf,
                     'task': 'impu',
                     'optim':'Adam',
                     'optim_para': 'betas=(0.5, 0.999)',
                     # 'optim':'RMSprop',
                     # 'optim_para': 'alpha=0.9, eps=1e-10', # centered=True',
                     'lr': 1e-4, 
                     'dropout': dropout,
                     '__drop__': __drop__}
        
    model = eval(_class + '(**parameter)')
    gene_new = False
    # gene_new = True
    model.load_impu(path, path, batch_size, dynamic = dynamic,
                    missing_rate = missing_rate, gene_new = gene_new,
                    example = example)
    return model
    
if __name__ == '__main__':
    dataset = 2
    model_id = 5
    missing_rate = 0.1
    if dataset == 1:
        n_v, dynamic = 33, 16
        dropout = 0.1
        batch_size = 32
        path = '../data/TE'
        example = 'TE'
    else:
        n_v, dynamic = 61, 16
        dropout = 0.1
        batch_size = 16
        path = '../data/hydrocracking/hydrocracking.xls'
        example = 'HY'
    model = get_model(model_id, dynamic, n_v, dropout, batch_size, missing_rate,
                      path = path, example = example)
    
    model._init_para()
    model.run(datasets = None,
              # pre_e = 15, load = 'pre',
              e = 120, b = batch_size, 
              tsne = False, 
              cpu_core = 0.8, num_workers = 0)
    model.result()

    # Run_N(model, 3).run(datasets = None, e = 120, b = batch_size, load = '', 
    #                     cpu_core = 0.8, num_workers = 0)