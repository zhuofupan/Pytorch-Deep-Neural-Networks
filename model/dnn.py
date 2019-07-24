# -*- coding: utf-8 -*-

import sys
sys.path.append('..')
from core.module import Module

class DNN(Module):  
    def __init__(self, **kwargs):
        self._name = 'DNN'
        Module.__init__(self, **kwargs)
        self._feature, self._output = self.Sequential(out_number = 2)
        self.opt()

    def forward(self, x, y = None):
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