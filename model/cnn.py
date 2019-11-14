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

    def forward(self, x):
        self._loss = 0
        for layer in self.layers:
            layer._target = self._target
            x = layer(x)
            if hasattr(layer, '_loss') and self.training:
                self._loss += layer._loss
            
#            if self.training and hasattr(self,'sup_factor'):
#                h = layer.act_val
#                try:
#                    from private.sup_loss import get_h_y
#                    _h, _y = get_h_y(h, self._target)
#                    self._loss  += self.L(_h, _y) * self.sup_factor 
#                except ImportError:
#                    pass
            
        x = x.contiguous().view(x.size(0),-1)
        if hasattr(self, 'struct'):
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
