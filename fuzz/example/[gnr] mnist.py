# -*- coding: utf-8 -*-

import sys
from fuzz.model.vae import VAE
# from fuzz.model.variant.mmdgm_vae import MMDGM_VAE

def get_model(name):
    para = {'struct': [784,256,128,100],
        'hidden_func': ['r','s'],
        'decoder_func': ['r','r','s'],
        'n_category': 10,
        'dropout': 0.0,
        'task': 'gnr',
        'reshape_size': [28,28],
        'flatten': True}
    if name == 'vae':
        return VAE(**para)
    # elif name == 'mmd':
    #     return MMDGM_VAE(**para)
    else:
        raise Exception("Enter a correct model name!")

model = get_model('vae')
model.load_mnist('../data', 64)
model.run(e = 10, pre_e = 0, load = '', n_sampling = 9)

