# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:30:30 2022

@author: Fuzz4
"""
import os
import numpy as np
import matplotlib.pyplot as plt

def affine(x): return x
def sigmoid(x): return 1/(1+np.exp(-x))
def tanh(x): return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
def square(x): return x**2
def gaussian(x): return 1-np.exp(-x**2)
func_dict = {'s': sigmoid, 't': tanh, 'q': square, 'g': gaussian, 'a': affine}

def cf(_str):
    def func(x):
        for s in _str:
            x = func_dict[s](x)
        return x
    return func

class Plot_Curves():
    def __init__(self, **kwargs):
        default = {'func_list': None, 
                   'x_range': 3,
                   'n_point': 20000}
        for key in default.keys():
            if key not in kwargs.keys():
                setattr(self, key, default[key])
            else: 
                setattr(self, key, kwargs[key])
        self.plot()
    
    def plot(self):
        if type(self.x_range) != tuple: self.x_range = (-self.x_range, self.x_range)
        x = np.linspace(self.x_range[0], self.x_range[1], self.n_point)
        
        fig, ax = plt.subplots(1,1, figsize=(32,18), dpi= 80)
        ax.set_facecolor("white")  # 设置背景色
        
        for i in range(len(self.func_list)):
            func = self.func_list[i]
            if func in func_dict.keys(): func = func_dict[func]
            else: func = cf(func)
            y = func(x)
            ax.plot(x, y, linewidth = 5)
        
        ax.set_xlabel('$x$', fontsize=72)
        ax.tick_params(axis='x', labelsize=72, pad = 0.01)
        ax.set_ylabel('$y$', fontsize=72)
        ax.tick_params(axis='y', labelsize=72, pad = 0.01)
        ax.grid(color = 'k', alpha = 0.2)
        ax.grid(None)
        
        plt.legend([self.func_list[i] for i in range(len(self.func_list))], 
                   fontsize=60, loc = 1, frameon=False)

        save_path, file_name = '../plot', '/curves.pdf'
        if not os.path.exists(save_path): os.makedirs(save_path)
        plt.savefig(save_path + file_name, bbox_inches='tight')
        plt.show()
        plt.close(fig)

if __name__ == '__main__':
    para = {'func_list':['sssa', 'gssa', 'qssqsa', 'ggsgaa', 'ttstsa']}
    Plot_Curves(**para)