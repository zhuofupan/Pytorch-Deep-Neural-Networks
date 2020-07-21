# -*- coding: utf-8 -*-
import torch
import torch.multiprocessing as mp

# 初始化随机数种子
torch.manual_seed(1)

import numpy as np
import sys
import os
import time

def _to2d(x):
    x = x.data
    if len(x.size()) < 2:
       x = x.view(-1, 1) 
    return x

def _get_subplot_size(N):
    A = []
    for i in range(N):
        if np.mod(N, i+1) == 0: A.append(i+1)
    l = len(A)
    if np.mod(l,2) == 1: n_row, n_col = A[int(l/2)], A[int(l/2)]
    elif A[int(l/2)-1]/A[int(l/2)] > 0.5: n_row, n_col = A[int(l/2)-1], A[int(l/2)]
    else: 
        n_col = int(np.sqrt(N)) + 1
        n_row = int(N/n_col) + 1
    return n_row, n_col

def _save_module(model = None, do = 'save', stage = 'best', obj = 'para'):
    if model is None:
        do, obj= 'load', 'model'
    if stage!= 'best' and do == 'save': print()
    if stage!= 'best' or do == 'load':
        print("{} [{}] 's {} as '{}'".format(do.capitalize(), model.name, obj, stage))

    path = '../save/'+ model.name + model.run_id +'/[{}] _{} _{}'.format(model.name, stage, obj) 

    if obj == 'para':
        if do == 'save': torch.save(model.state_dict(), path)
        else: model.load_state_dict(torch.load(path))
    elif obj == 'model':
        if do == 'save': torch.save(model, path)
        # model = access()
        else: return torch.load(path)

class Epoch(object):
    
    def run(self, 
            datasets = None,     # 数据集路径或数据集
            e = 100,             # 微调迭代次数
            b = None,            # 批次大小
            pre_e = 0,           # 预训练迭代次数
            cpu_core = 0.8,       # 设置最大可用的 cpu 核数
            gpu_id = '0',        # 使用的 gpu 编号
            num_workers = 0,     # data_loader参数，加载时的线程数
            load = '',           # 加载已训练的模型
            tsne = False,        # 是否绘制预训练后的 t-sne 图
            n_sampling = 0,      # 设置测试时的采样个数 - 用于可视化 
            run_id = -1):        # 多次运行时的 id
    
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id      # 设置使用的 gpu 编号
        
        max_cpu_core = int(os.cpu_count()/2)
        if cpu_core == -1: cpu_core = max_cpu_core
        elif cpu_core < 1: cpu_core = int(max_cpu_core * cpu_core)
        torch.set_num_threads(cpu_core)                  # 设置 cpu 核使用数
        
            
        if num_workers == -1: num_workers = cpu_core     # 默认线程数 = 设置的核心数
        self.loader_kwargs['num_workers'] = num_workers
        
        if self.dvc == torch.device('cpu'): 
            self.dvc_info = ' ({} core, {} threads)'.format(cpu_core, num_workers)
        else:
            self.dvc_info = ':'+ gpu_id + ' ({} core, {} threads)'.format(cpu_core, num_workers)
            
        if datasets is not None:
            self.load_data(datasets, b)
        if b is None:
            b = self.batch_size   
        
        if load == 'pre':
            try:
                self._save_load('load', 'pre')
                pre_e = -1
            except FileNotFoundError:
                print("\nCannot find 'pre' para, exec pre-training...\n")
        
        if not os.path.exists('../save/'+ self.name + self.run_id): os.makedirs('../save/'+ self.name + self.run_id)
        start = time.clock()
        time0 = start
        
        # 开始预训练
        if load != 'best' and pre_e > 0:
            # try:
                self.pre_batch_training(pre_e, b)
                self._save_load('save', 'pre')
                
                pre_time = time.clock()
                print("Finish pre-training, cost {} seconds".format(int(pre_time - start)))
                start = pre_time
            # except AttributeError:
            #     print("No need pre-training, exec training...\n")
            
        # 绘制特征 t-sne 图
        if tsne:
            print('\nPlot t-SNE for last feature layer')
            self._plot_pre_feature_tsne()
            
            tsne_time = time.clock()
            print("Finish ploting t-SNE, cost {} seconds (totally use {} seconds)".format(
                int(tsne_time - start), int(tsne_time - time0)))
            start = tsne_time
            
        if load == 'best':
            try:
                self._save_load('load', 'best')
                self.test(1, n_sampling = n_sampling)
                e = -1
            except FileNotFoundError:
                print("\nCannot find 'best' para, exec training...\n")
                
        # 开始微调
        if e > 0:
            for epoch in range(1, e + 1):
                self.batch_training(epoch)
                self.test(epoch, n_sampling = n_sampling)
                
            ft_time = time.clock()
            self.cost_time = int(ft_time - time0)
            print("\nFinish fine-tuning, cost {} seconds (totally use {} seconds)".format(
                int(ft_time - start), int(ft_time - time0)))
            
        if self.task in ['cls', 'prd']:
            print("Save [{}] 's para as 'best'".format(self.name))                                                                           
    
    def batch_training(self, epoch):
        if epoch == 1:
            print('\nTraining '+self.name+ ' in {}'.format(self.dvc) + self.dvc_info +':')
            
        self = self.to(self.dvc)
        self.train()
        
        train_loss = 0
        outputs, targets = [], []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            # if self.dvc == torch.device('cuda') and hasattr(torch.cuda, 'empty_cache'): 
            #     torch.cuda.empty_cache()
            data, target = data.to(self.dvc), target.to(self.dvc)
            self._target = target
            
            self.zero_grad()
            output = self.forward(data)
            output, loss = self.get_loss(output, target)
            loss.backward()
            self.optim.step()
            
            train_loss += (loss.detach() * data.size(0))
            if hasattr(self, 'decay_s'):
                self.scheduler.step()
            elif hasattr(self, 'decay_r'):
                self.scheduler.step(loss.detach())
                
            outputs.append(_to2d(output))
            targets.append(_to2d(target))
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(self.train_loader):
                self.msg_str = 'Epoch: {} - {}/{} | loss = {:.4f}'.format(
                    epoch, batch_idx+1, len(self.train_loader), loss.detach())
                for item in self.msg:
                    if hasattr(self, item):
                        self.msg_str += '   '+item+' = {:.4f}'.format(eval('self.'+item))
                sys.stdout.write('\r'+ self.msg_str)
                sys.stdout.flush()
                      
        train_loss = train_loss/ len(self.train_loader.dataset)
        outputs = torch.cat(outputs, 0)
        targets = torch.cat(targets, 0)
        
        self.evaluation('train', outputs, targets, train_loss)

    def test(self, epoch, dataset = 'test', n_sampling = 0):
        if dataset == 'test':
            loader = self.test_loader
        elif dataset == 'train':
            loader = self.train_loader
        else:
            loader = dataset

        self = self.to(self.dvc)
        self.eval()
        
        test_loss = 0
        outputs, targets = [], []
        
        self.n_sampling = n_sampling
        with torch.no_grad():
            if n_sampling > 0:
                batch_id = []
                if n_sampling < len(loader): 
                    batch_id = np.random.choice(len(loader), n_sampling, replace = False) 
                self._sampling = {'img':[], 'label':[], 'name':['in','out']}
                
            for i, (data, target) in enumerate(loader):
                data, target = data.to(self.dvc), target.to(self.dvc)
                self._target = target
                
                output = self.forward(data)
                output, loss = self.get_loss(output, target)
                
                test_loss += loss.data * data.size(0)
                outputs.append(_to2d(output))
                targets.append(_to2d(target))
                
                # save img
                if n_sampling > 0:
                    sample_id = []
                    if len(batch_id) > 0:
                        if i in batch_id: sample_id.append(np.random.randint(data.size(0)))
                    else:
                        if i == 0: n_sample_in_batch = n_sampling - int(n_sampling/len(loader))*(len(loader)-1)
                        else: n_sample_in_batch = int(n_sampling/len(loader))
                        sample_id = np.random.choice(data.size(0), n_sample_in_batch, replace = False)
                        
                    for k in sample_id:
                        if data.size(1) == output.size(1):
                            self._sampling['img'].append([_to2d(data[k]).cpu().numpy(), _to2d(output[k]).cpu().numpy()])
                        else:
                            self._sampling['img'].append([_to2d(data[k]).cpu().numpy()])
                        self._sampling['label'].append(_to2d(target[k]).cpu().numpy())
        
        if n_sampling > 0:         
                self._save_sample_img(epoch)
        test_loss = test_loss/ len(loader.dataset)
        outputs = torch.cat(outputs, 0)
        targets = torch.cat(targets, 0)
        
        self.evaluation('test', outputs, targets, test_loss)
        
                
    def evaluation(self, phase, output, target, loss):
        output, target, loss = output.cpu().numpy(), target.cpu().numpy(), loss.cpu().numpy()
        if self.task in ['usp','gnr']: 
            return
        elif self.task == 'cls':
            accuracy = self.get_accuracy(output, target)
            msg_dict = {'accuracy':np.around(accuracy*100,2)}
            if phase == 'test' and accuracy > self.best_acc:
                self.best_acc = accuracy
                self.get_FDR(output, target)
                self._save_load('save')
        elif self.task == 'prd':
            rmse = self.get_rmse(output, target)
            R2 = self.get_R2(output, target)
            msg_dict = {'rmse':np.around(rmse,4), 'R2':np.around(R2,4)}
            if phase == 'test' and rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_R2 = R2
                self.pred_Y = output
                self._save_load('save')
        
        if phase == 'train':
            msg_str = '\n    >>> Train: loss = {:.4f}   '.format(loss)
        else:
            msg_str = '    >>> Test: loss = {:.4f}   '.format(loss)
        
        for key in msg_dict.keys():
            if key == 'accuracy':
                msg_str += key+'(%) = {}   '.format(msg_dict[key])
            else:
                msg_str += key+' = {:.4f}   '.format(msg_dict[key])
                
        if phase == 'test' and self.n_sampling > 0:
             msg_str += '- plot img ({} n_sampling)'.format(self.n_sampling)
        print(msg_str)
        
        msg_dict['loss'] = np.around(loss,4)
        # 存入DataFrame
        exec('self.'+phase+'_df = self.'+phase+'_df.append(msg_dict, ignore_index=True)')
    
    def _save_sample_img(self, epoch):
        # 一个epoch存一张图，子图个数为 n_sampling
        import matplotlib.pyplot as plt
        
        reshape = None
        if hasattr(self, 'img_size'): reshape = self.img_size
        
        n_plot_per_sample = len(self._sampling['img'][0])
        n_sampling = len(self._sampling['img'])
        N = int(n_sampling * n_plot_per_sample)
            
        n_row, n_col = _get_subplot_size(N)
        
        fig = plt.figure(figsize=[n_row*4, n_col*4])
        for i in range(n_sampling):
            x, y = self._sampling['img'][i], self._sampling['label'][i]
            
            if y.shape[0] > 1: y = np.argmax(y, 0)
            else: y = np.round(y, 2)

            for j in range(n_plot_per_sample):
                ax = fig.add_subplot(n_row, n_col, n_plot_per_sample*i+j+1)
                ax.set_title('{}, y = {}'.format(self._sampling['name'][j], y))
                img = x[j]
                if reshape is not None: img = img.reshape((reshape[0],reshape[1]))
                ax.imshow(img)
            
        file_name = 'Epoch {} ({})'.format(epoch, N)
        if not os.path.exists('../save/'+ self.name + self. run_id + '/sampling/'): os.makedirs('../save/'+ self.name + self. run_id + '/sampling/')
        plt.savefig('../save/'+ self.name + self. run_id + '/sampling/'+ file_name +'.png', bbox_inches='tight')
        plt.close()
    
#    def _save_test_img(self, data, _add_data = None, epoch = None, target = None):
#        '''
#            _img_to_save: ['res','...']
#        '''
#        
#        path = '../save/img/['+self.name+']/'
#        if not os.path.exists(path): os.makedirs(path)
#        path += 'Epoch = {}'.format(epoch)
#        
#        data_list = []
#        n = 8
#        
#        # [data, output]
#        for _d in data: 
#            data_list.append(_d[:n])
#
#        # _add_data
#        if _add_data is not None:
#            if type(_add_data) != list: _add_data = [_add_data]
#            for _s in _add_data:
#                if _s == 'res':  _d = data[1] - data[0]
#                else:  _d = eval('self.'+_s)
#                data_list.append(_d[:n])
#            
#        # _add_info
#        if self.task == 'cls':
#            target = target[:n].cpu().numpy()
#            if target.ndim >1: target = np.argmax(target, 1)
#            path += ' ,label = ['
#            for i in range(n):
#                if i < n - 1: path += str(target[i]) + ' '
#                else: path += str(target[i]) + ']'
#        
#        _save_multi_img(data_list, data_list[0].shape[0], path = path)
