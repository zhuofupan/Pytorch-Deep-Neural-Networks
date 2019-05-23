# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from core.module import Module
from pandas import DataFrame


class CNN(Module):  
    def __init__(self, **kwargs):
        self.name = 'CNN'
        super().__init__(**kwargs)
        self.Convolutional()
        self.Sequential()
        self.opt()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        if len(self.struct) == 0:
            return x
        else:
            x = self.feature(x)
            x = self.output(x)
            return x

if __name__ == '__main__':
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
    conv.loc[0] = [[3, 8],     1, '', 2]
    conv.loc[1] = [[6, (6,6)], 1, '', 0]
    
    parameter = {'img_size': [1,28,28],
                 'conv_struct': conv,
                 'conv_func': ['ReLU'],
                 'struct': [150, 10],
                 'hidden_func': ['Gaussian', 'Affine'],
                 'output_func': 'Affine',
                 'dropout': 0.0,
                 'task': 'cls'}
    
    model = CNN(**parameter)
    
    model.load_mnist('../data', 128)
    
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.test()
