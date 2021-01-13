# -*- coding: utf-8 -*-
import torch.nn as nn
from ...core.module import Module

class Connect(Module):  
    def __init__(self, module_list, **kwargs):
        ''' 
            _loss: 附加损失
            loss: 全部损失
        '''
        name = ''
        for i in range(len(module_list)):
            if hasattr(module_list[i], 'name'):
                name += module_list[i].name
            if i < len(module_list) -1:
                name +=  '-'
        self.name = name
        
        Module.__init__(self, **kwargs)
        
        self.modules = nn.Sequential(*module_list)
        self.module_list = module_list
        
        self.opt(False)

    def forward(self, x):
        self._loss = 0
        for i in range(len(self.module_list)):
            module = self.module_list[i]    
            x = module.forward(x)
            if hasattr(module, '_loss') and self.training:
                self._loss += module._loss
        return x