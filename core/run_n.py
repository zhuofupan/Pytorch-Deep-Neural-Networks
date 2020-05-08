# -*- coding: utf-8 -*-
import torch
import os
import numpy as np
import time
import pandas as pd
from pandas import DataFrame

class Run_N(object):
    '''
        让模型了连续跑 N 次
        使用例子：
            Run_N(model, 2).run(datasets = datasets, e = 1, b = 16, pre_e = 1, load = '', tsne = True, 
                            cpu_core = 0.8, num_workers = 0)
    '''
    def __init__(self, model, run_times, categories_name = None):
        self.model = model
        self.run_times = run_times
        self.categories_name = categories_name

    def run(self, *args, **kwargs):
        model = self.model
        kwargs['tsne'] = False
        
        cost_time = []
        if model.task == 'cls':
            best_FDR, best_FPR = [], []
        else:
            best_rmse, best_R2 = [], []
        
        _rank = ['th', 'st', 'nd', 'rd'] + ['th'] * 6
        for i in range(self.run_times):
            if i + 1 > 10 and i + 1 < 20: rank = 'th'
            else: rank = _rank[np.mod(i+1, 10)]
            
            print("\n---------------------------------------")
            print(" ★ Running the model for the {}{} time ---".format(i + 1, rank))
            print("---------------------------------------------")
            
            if i > 1:
                # 参数初始化
                # print("\nReset model's parameters ......")
                # for layer in model.modules():
                #     if hasattr(layer, 'reset_parameters'):
                #         layer.reset_parameters()

                # 属性初始化
                for attr in ['pre_modules', 'optim', 'train_df', 'test_df', 'train_loader', 'test_loader']:
                    if hasattr(model, attr): 
                        eval('del model.' + attr)
                model.best_acc = 0
                model.best_rmse = float('inf')

                # 等一哈
                time.sleep(0.1)
                model.__init__(model.kwargs)
                
            model.run_id = '-' + str(i + 1)    
            model.run(*args, **kwargs)
            model._save_xlsx()
            
            cost_time.append(model.cost_time)
            if model.task == 'cls':
                best_FDR.append(model.FDR[-1][0])
                best_FPR.append(model.FDR[-1][1])
            else:
                best_rmse.append(model.best_rmse)
                best_R2.append(best_R2)
        
        time.sleep(0.1)
        
        # sheet_names
        sheet_names = ['epoch_indi','mean_var']
        run_id = list(range(1, self.run_times + 1))
        
        # epoch_indi
        if model.task == 'cls':
            df1 = DataFrame({'Runid': run_id, 'best_acc': best_FDR, '1 - best_acc': best_FPR, 'cost_time': cost_time})
        else:
            df1 = DataFrame({'Runid': run_id, 'best_rmse': best_rmse, 'best_R2': best_R2, 'cost_time': cost_time})

        # mean_var
        res = np.zeros((4, 3))
        index = ['mean', 'var', 'max-mean', 'min-mean']
        if model.task == 'cls':
            columns = ['best_acc', '1 - best_acc', 'cost_time']
            datas = [best_FDR, best_FPR, cost_time]
        else:
            columns = ['best_rmse', 'best_R2', 'cost_time']
            datas = [best_rmse, best_R2, cost_time]
            
        for j, data in enumerate(datas):
            data = np.array(data)
            res[0][j] = np.mean(data)
            res[1][j] = np.var(data)
            res[2][j] = np.max(data) - np.mean(data)
            res[3][j] = np.min(data) - np.mean(data)
            
        df2 = DataFrame(res, columns = columns)
        df2.insert(0,'Indicators', index)

        # writer
        dfs = [df1, df2]
        if not os.path.exists('../save/Run_N'): os.makedirs('../save/Run_N')
        writer = pd.ExcelWriter('../save/Run_N/['+model.name+'] result.xlsx',engine='openpyxl')
        # save
        for i, sheet_name in enumerate(sheet_names):
            dfs[i].to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()
