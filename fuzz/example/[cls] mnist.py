# -*- coding: utf-8 -*-
from fuzz.model.dbn import DBN
from fuzz.model.sae import SAE
from fuzz.model.cnn import CNN

def get_model(name):
    if name == 'dbn':
        # DBN
        para = {'struct': [784,400,100,10],
                'h_type': ['Gaussian', 'Binary'],
                'output_func': 'Affine',
                'v_type': ['Binary', 'Binary'],
                'dropout': 0.0,
                'task': 'cls',
                'reshape_size': [28,28],
                'flatten': True}
        return DBN(**para)
    elif name == 'sae':
        para = {'struct': [784,400,100,10],
                'hidden_func': ['Gaussian', 'Affine'],
                'output_func': 'Affine',
                'decoder_func': ['Affine', 'Affine'],
                'ae_type': 'AE',
                'dropout': 0.0,
                'task': 'cls',
                'lr': 1e-3,
                'reshape_size': [28,28],
                'flatten': True}
        return SAE(**para)
    elif name == 'cnn':
        ''' 
            conv_para: (in_channels(auto), out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
            pool_type: 'Max', 'Avg', 'FractionalMax', 'AdaptiveMax', 'AdaptiveAvg'
            pool_para: (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        ''' 
    
        para = {'img_size': [1,28,28],
                'conv_struct': [[3, 8], ['M', 2], [6, (6,6)]],
                'conv_func': 'ReLU',
                'conv_dropout': 0.0,
                'struct': [150, 10],
                'hidden_func': ['Gaussian', 'Affine'],
                'output_func': 'Affine',
                'dropout': 0.0,
                'task': 'cls'}
        return CNN(**para)
    else:
        raise Exception("Enter a correct model name!")

model = get_model('sae')
model.load_mnist('../data', 128)
model.run(e = 3, pre_e = 3, load = '', n_sampling = 9)