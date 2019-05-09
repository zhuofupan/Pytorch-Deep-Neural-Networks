# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from model.dbn import DBN
from model.cnn import CNN
import torch
from pandas import DataFrame

# DBN
dbn_parameter = {'struct': [784,400,100,10],
             'hidden_func': ['Gaussian', 'Affine'],
             'output_func': 'Affine',
             'dropout': 0.0,
             'task': 'cls',
             'flatten': True}
# CNN
conv = DataFrame(
        columns = ['out_channel', 'conv_kernel_size', 'is_bn', 'pool_kernel_size']
        )
conv.loc[0] = [3, 8, 1, 2]
conv.loc[1] = [6, (6,6), 1, 0]

cnn_parameter = {'img_size': [1,28,28],
             'conv_struct': conv,
             'conv_func': 'ReLU',
             'struct': [150, 10],
             'hidden_func': ['Gaussian', 'Affine'],
             'output_func': 'Affine',
             'dropout': 0.0,
             'task': 'cls'}

def run(name):
    model = eval(name.upper()+'(**'+name.lower()+'_parameter)')
    
    model.load_mnist('../data', 128)
    
    model.pre_train(3, 128)
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.test()

run('dbn')
#run('cnn')