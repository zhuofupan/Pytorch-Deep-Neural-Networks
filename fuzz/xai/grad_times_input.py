# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 21:29:41 2021

@author: owner
"""
import torch
from ..core.attribution import Attribution

class Grad_T_Input(Attribution):
    def __init__(self, **kwargs):
        self.name = self.__class__.__name__
        Attribution.__init__(self, **kwargs)