# -*- coding: utf-8 -*-
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
sys.path.append('..')
from model.dbn import DBN
from model.cnn import CNN

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
    run('cnn',**para)

dbn()
#cnn()