# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from model.dbn import DBN
from model.cnn import CNN
import torch
from pandas import DataFrame
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run(name, para):
    model = eval(name.upper()+'(**para)')
    model = model.to(device)
    
    model.load_mnist('../data', 128)
    
    model.pre_train(3, 128)
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.test()

def dbn():
    # DBN
    para = {'struct': [784,400,100,10],
            'hidden_func': ['Gaussian', 'Affine'],
            'output_func': 'Affine',
            'dropout': 0.0,
            'task': 'cls',
            'flatten': True}
    run('dbn',**para)

def cnn():
    # CNN
    ''' 
        head = ['conv_para', 'bn_type', 'pool_type', 'pool_para']
        conv_para: (in_channels(auto), out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        bn_type: 0, 1
        pool_type: 0, 'Max', 'Avg', 'FractionalMax', 'AdaptiveMax', 'AdaptiveAvg'
        pool_para: (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    ''' 
    conv = DataFrame(
            columns = ['conv_para', 'bn_type', 'pool_type', 'pool_para']
            )
    conv.loc[0] = [[3, 8],     1, '', [2]]
    conv.loc[1] = [[6, (6,6)], 1, '',  0 ]
    
    para = {'img_size': [1,28,28],
            'conv_struct': conv,
            'conv_func': 'ReLU',
            'conv_dropout': 0.0,
            'struct': [150, 10],
            'hidden_func': ['Gaussian', 'Affine'],
            'output_func': 'Affine',
            'dropout': 0.0,
            'task': 'cls'}
    run('cnn',**para)

dbn()
#cnn()