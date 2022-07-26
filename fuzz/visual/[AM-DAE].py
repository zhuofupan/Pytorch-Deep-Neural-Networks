# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 13:28:58 2022

@author: Fuzz4
"""
import numpy as np
import pandas as pd
from fuzz.visual.plot import epoch_comparing_curve, var_impu_curve, _color_bar

def comparing(name = 'train_rmse'): # train_rmse, train_mape
    # compare_curve
    epoch_comparing_curve('../save/Impu_curve', (name,''), 
                          color_cmap = _color_bar[5],
                          # title, ax_label, ax_ticks, legend, text
                          fontsize = [54, 42, 40, 38, 38],
                          linewidth = (4, 4),
                          marker = (['o','^','s','*','d'], ['o','^','s','*','d']),
                          markersize = (18, 18),
                          yscale = 'log')
    
def var_impu(mr = 0.2):
    # var_impu_curve(pred, real, locationi, missing_var_id, model_name, add_info)
    path = '../save/AM_DAE [{}]'.format(mr)
    pred = path + '/AM_DAE [{}]-impu.csv'.format(mr)
    pred_X = pd.read_csv(pred, header = None).values
    real = path + '/real.csv'
    real_X = pd.read_csv(real, header = None).values
    nan = 1-np.isnan(pred_X).astype(int)
    sum_nan = nan.sum(axis = 0)
    missing_var_id = np.where(sum_nan != 0)[0].tolist()
    print(missing_var_id)
    nan = nan.astype(np.float32)
    var_impu_curve(pred_X, real_X, nan, missing_var_id, 'AM_DAE_{}'.format(mr), '')
    
if __name__ == '__main__':
    var_impu(0.2)