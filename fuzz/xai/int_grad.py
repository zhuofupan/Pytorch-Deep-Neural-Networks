# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
from ..core.attribution import Attribution

# Integrated gradients
class Int_Grad(Attribution):
    def __init__(self, 
                 n_interpolation = 10,
                 **kwargs):
        kwargs_change = {'ignore_wrong_pre': False,
                         'need_baseline': True,
                         'hook_fp': [False, False],
                         'hook_bp': [False, False]
                         }
        kwargs.update(kwargs_change)
        self.name = self.__class__.__name__
        Attribution.__init__(self, **kwargs)
        self.n_interpolation = n_interpolation
    
    def _get_integrated_gradient(self, data, class_int):
        # Interpolation from x to baseline
        grad_input = 0
        for i in range(1, self.n_interpolation + 1):
            input_data = self.x_bl + i/self.n_interpolation * (data - self.x_bl)
            self._get_gradient(input_data, class_int)
            grad_input += self.grad_input
        self.grad_input = grad_input/self.n_interpolation