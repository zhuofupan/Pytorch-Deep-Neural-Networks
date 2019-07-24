# -*- coding: utf-8 -*-
import torch.nn as nn
import sys
sys.path.append('..')
from core.module import Module

class Connect(Module):  
    def __init__(self, module_list, benchmark = -1, **kwargs):
        _kwargs = module_list[benchmark].kwargs
        for key in kwargs.keys():
            _kwargs[key] = kwargs[key]

        Module.__init__(self, **_kwargs)
        
        self.modules = nn.Sequential(*module_list)
        self.module_list = module_list
        
        self._name = ''
        for i in range(len(module_list)):
            self._name += module_list[i]._name
            if i < len(module_list) -1:
                self._name +=  '-'
        self.opt(False)

    def forward(self, x, y = None):
        for module in self.module_list:
            x = module.forward(x, y)
        return x
        
    
if __name__ == '__main__':
    from private.read_te_data import gene_net_datas
    from core.layer import Square, Reshape
    from model.cnn import CNN
    
    datasets = gene_net_datas(
            data_dir='../private/TE',
            preprocessing='st', # gauss单元用‘st’, bin单元用'mm'
            one_hot=True,
            shuffle=False,
            # 考虑动态数据集
            dynamic=40,
            stride=0,
            load_data=False)
    
    # CNN
    conv_struct = [ #['A', 3, 1, 1],
                    '1*', [16, (13,10), 'B01'], ['M', 2, 1], 
                    '2*', [32, (6,4), 'B01'], ['M', 2, 1], 
                    '3*', [32, 3, 'B01'], 
                    '1*', [19, 6, 'B01'], ['AM', 1 ]]
                  
    cnn_parameter = {'name': 'CNN_TE',
                     'img_size': [1,40,40],
                     'conv_struct': conv_struct,
                     'conv_func': ['r', 'r', 'r', 'x'], # ReLU > Sigmoid > Affine > Gaussian
                     'conv_dropout': 0,
#                     'struct': [-1, 100, 19], # Output > Adaptive
#                     'hidden_func': ['r', 'r', 'r'], # ReLU > Sigmoid > Affine > Gaussian
#                     'output_func': 'g',
                     'dropout': 0.216,
                     'use_bias': False,
#                     'optim': 'Adam',
                     'optim': 'RMSprop',
                     'optim_para': 'alpha=0.9, eps=1e-10', #momentum=0.5',
                     'task': 'cls',
                     'lr':1e-4} 
    
    re = Reshape((1,40,33))
    sq = Square((1,33,33), 'g')
    cnn = CNN(**cnn_parameter)
    
    model = Connect([re, sq, cnn], img_size = [1,40,33])
    model.load_data(datasets, 32)
    #model._save_load('load', 'best')
    for epoch in range(1, 50 + 1):
        model.batch_training(epoch)
        model.test(epoch)
    model.result()
    model._plot_weight()
    model._visual('category', epoch = 100, reshape = True)
