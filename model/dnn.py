# -*- coding: utf-8 -*-
import torch
import sys
sys.path.append('..')
from core.module import Module

class DNN(Module):  
    def __init__(self, **kwargs):
        self._name = 'DNN'
        Module.__init__(self, **kwargs)
        self._feature, self._output = self.Sequential(out_number = 2)
        self.opt()

    def forward(self, x):
        
#        self._loss =0
#        if self.training and hasattr(self,'sup_factor'):
#            for module in self._feature.children():
#                x = module(x)
#                if isinstance(module, torch.nn.Linear) == False\
#                and isinstance(module, torch.nn.Dropout) == False:
#                    # print(module)    
#                    try:
#                        from private.sup_loss import get_h_y
#                        _h, _y = get_h_y(x, self._target)
#                        self._loss += self.L(_h, _y) * self.sup_factor 
#                    except ImportError:
#                        pass
        
        x = self._feature(x)
        x = self._output(x)
        return x

if __name__ == '__main__':
    
    parameter = {'struct': [784,400,100,10],
                 'hidden_func': ['g', 'a'],
                 'output_func': 'x',
                 'dropout': 0.0,
                 'task': 'cls',
                 'flatten': True}
    
    model = DNN(**parameter)
    
    model.load_mnist('../data', 128)
    
    for epoch in range(1, 3 + 1):
        model.batch_training(epoch)
        model.test(epoch)