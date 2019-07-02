# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
from core.module import Module
from core.conv_module import Conv_Module

class CNN(Module, Conv_Module):  
    def __init__(self, **kwargs):
        self._name = 'CNN'
        
        Module.__init__(self,**kwargs)
        Conv_Module.__init__(self,**kwargs)

        self.layers = self.Convolutional()
        self.fc = self.Sequential()
        self.opt()

    def forward(self, x, y = None):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
        
if __name__ == '__main__':
    # CNN    
    conv_struct = [[3, 8], ['M', 2], [6, (6,6)]]
    
    parameter = {'img_size': [1,28,28],
                 'conv_struct': conv_struct,
                 'conv_func': ['ReLU'],
                 'batch_norm': True,
                 
                 'struct': [150, 10],
                 'hidden_func': ['Gaussian', 'Affine'],
                 'output_func': 'Affine',
                 'dropout': 0.0,
                 'task': 'cls'}
    
    model = CNN(**parameter)
    
    model.load_mnist('../data', 128)
    
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.test(epoch)
