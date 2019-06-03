# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from model.sae import SAE
from model.cnn import CNN

def run(name, **para):
    model = eval(name.upper()+'(**para)')
    
    model.load_mnist('../data', 128)
    
    model.pre_train(3, 128)
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.test()
    model.result()

def sae():
    # DBN
    para = {'struct': [784,400,100,10],
            'hidden_func': ['g', 'a'],
            'output_func': 'a',
            'dropout': 0.0,
            'task': 'cls',
            'flatten': True}
    run('sae',**para)

def cnn():
    # CNN
    ''' 
        conv_para: (in_channels(auto), out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        pool_type: 'Max', 'Avg', 'FractionalMax', 'AdaptiveMax', 'AdaptiveAvg'
        pool_para: (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    ''' 

    para = {'img_size': [1,28,28],
            'conv_struct': [[3, 8], ['M', 2], [6, (6,6)]],
            'conv_func': 'r',
            'conv_dropout': 0.0,
            'struct': [150, 10],
            'hidden_func': ['g', 'a'],
            'output_func': 'a',
            'dropout': 0.0,
            'task': 'cls'}
    run('cnn',**para)

sae()
#cnn()