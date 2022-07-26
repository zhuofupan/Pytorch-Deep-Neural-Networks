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
        Run_N(model, 2).run(datasets = datasets, e = 1, b = 16, pre_e = 1, load = '', tsne = False, 
                            cpu_core = 0.8, num_workers = 0)
    '''
    def __init__(self, model, run_times, run_info = '', categories_name = None):
        self.model = model
        self.run_times = run_times
        self.run_info = run_info
        self.categories_name = categories_name

    def run(self, *args, **kwargs):
        model = self.model
        kwargs['tsne'] = False
        
        cost_time = []
        if model.task == 'cls':
            best_FDR, best_FPR = [], []
        elif model.task == 'prd':
            best_rmse, best_R2 = [], []
        elif model.task == 'impu':
            best_rmse, best_mape = [], []
        
        _rank = ['th', 'st', 'nd', 'rd'] + ['th'] * 6
        for i in range(self.run_times):
            if i + 1 > 10 and i + 1 < 20: rank = 'th'
            else: rank = _rank[np.mod(i+1, 10)]
            
            print("\n---------------------------------------")
            print(" ★ Running the model for the {}{} time ---".format(i + 1, rank))
            print("---------------------------------------------")
            
            rd_sd = torch.randint(0, 10000, (1,))
            torch.manual_seed(rd_sd[0])
            print('Random seed = {}'.format(rd_sd[0]))
            if i > 0:
                # 等一哈
                time.sleep(1.0)

                # 属性初始化
                for attr in ['pre_modules', 'optim', 'train_df', 'test_df', 'test_loader']:
                    if hasattr(model, attr): 
                        delattr(model, attr)
                if model.task != 'impu' and hasattr(model, 'train_loader'): 
                    delattr(model, 'train_loader')

                model.best_acc = 0
                model.best_rmse = float('inf')
                if model.task == 'impu':
                    model.best_mape = float('inf')
                    model.train_loader.fill_init()
                    np.random.shuffle(model.train_loader.d_indexs)

                # 等一哈
                time.sleep(1.0)
                model.kwargs['show_model_info'] = False
                model.kwargs['save_module_para'] = False
                model.__init__(**model.kwargs)
            
            model._init_para()
            model.run_id = self.run_info + ' {}{}'.format(i + 1, rank)
            model.run(*args, **kwargs)
            
            if model.task == 'cls':
                print('The best test average accuracy is {}%\n'.format(model.FDR[-1][0]))
            elif model.task == 'prd':
                print('The bset test rmse is {:.4f}, and the corresponding R2 is {:.4f}\n'.format(model.best_rmse, model.best_R2))
            elif model.task == 'impu':
                print('The bset test rmse is {:.4f}, and the best mape is {:.2f}%\n'.format(model.best_rmse, model.best_mape))
            
            model.result(plot = False)
            
            cost_time.append(model.cost_time)
            if model.task == 'cls':
                best_FDR.append(model.FDR[-1][0])
                best_FPR.append(model.FDR[-1][1])
            elif model.task == 'prd':
                best_rmse.append(model.best_rmse)
                best_R2.append(best_R2)
            elif model.task == 'impu':
                best_rmse.append(model.best_rmse)
                best_mape.append(model.best_mape)
        
        # save run_n results
        time.sleep(1.0)
        
        # sheet_names
        sheet_names = ['epoch_indi','mean_var']
        run_id = list(range(1, self.run_times + 1))
        
        # epoch_indi
        if model.task == 'cls':
            df1 = DataFrame({'Runid': run_id, 'best_acc': best_FDR, '1 - best_acc': best_FPR, 'cost_time': cost_time})
        elif model.task == 'prd':
            df1 = DataFrame({'Runid': run_id, 'best_rmse': best_rmse, 'best_R2': best_R2, 'cost_time': cost_time})
        elif model.task == 'impu':
            df1 = DataFrame({'Runid': run_id, 'best_rmse': best_rmse, 'best_mape': best_mape, 'cost_time': cost_time})

        # mean_var
        res = np.zeros((4, 3))
        index = ['mean', 'var', 'max-mean', 'min-mean']
        if model.task == 'cls':
            columns = ['best_acc', '1 - best_acc', 'cost_time']
            datas = [best_FDR, best_FPR, cost_time]
        elif model.task == 'prd':
            columns = ['best_rmse', 'best_R2', 'cost_time']
            datas = [best_rmse, best_R2, cost_time]
        elif model.task == 'impu':
            columns = ['best_rmse', 'best_mape', 'cost_time']
            datas = [best_rmse, best_mape, cost_time]
            
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
        writer = pd.ExcelWriter('../save/'+ model.name + model.run_id +\
                                '/['+ model.name + self.run_info +'] run_n result.xlsx',engine='openpyxl')
        # save
        for i, sheet_name in enumerate(sheet_names):
            dfs[i].to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()
