# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from model.vae import VAE
from model.mmdgm_vae import MMDGM_VAE

def get_model(name):
    para = {'struct': [784,128,100],
        'hidden_func': ['r','a'],
        'hidden_func2': ['r','s'],
        'dropout': 0.0,
        'task': 'gnr',
        'img_size': [28,28],
        'flatten': True}
    if name == 'vae':
        return VAE(**para)
    elif name == 'mmd':
        return MMDGM_VAE(**para)
    else:
        raise Exception("Enter a correct model name!")

model = get_model('vae')
model.load_mnist('../data', 64)
model.run(e = 10, pre_e = 3, load = 'pre', n_sampling = 9)

