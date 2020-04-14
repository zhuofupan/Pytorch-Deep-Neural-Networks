# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys
import os

torch.manual_seed(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'

def to_np(x):
    x = x.data.cpu().numpy()
    if len(x.shape) < 2:
       x = x.reshape(-1, 1) 
    return x

def _get_fit_size(N):
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

class Epoch(object):
    
    def run(self, data_path = None, e = 100, b = None, pre_e = 0, load = '', n_sampling = 0):
        if data_path is not None:
            self.load_data(data_path, b)
        if b is None:
            b = self.batch_size
        
        if load == 'pre':
            try:
                self._save_load('load', 'pre')
                pre_e = -1
            except FileNotFoundError:
                print("\nCannot find 'pre' para, exec pre-training...\n")
                
        if load != 'best' and pre_e > 0:
            try:
                self.pre_batch_training(pre_e, b)
                self._save_load('save', 'pre')
            except AttributeError:
                print("No need pre-training, exec training...\n")
            
        if load == 'best':
            try:
                self._save_load('load', 'best')
                self.test(1, n_sampling = n_sampling)
                e = -1
            except FileNotFoundError:
                print("\nCannot find 'best' para, exec training...\n")
        if e > 0:
            for epoch in range(1, e + 1):
                self.batch_training(epoch)
                self.test(epoch, n_sampling = n_sampling)
        if self.task in ['cls', 'prd']:
            print("\nSave [{}] 's para as 'best'".format(self.name))
    
    def batch_training(self, epoch):       
        if epoch == 1:
            print('\nTraining '+self.name+ ' in {}:'.format(self.dvc))
            
        self.train()
        self = self.to(self.dvc)
        train_loss = 0
        outputs, targets = [], []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.dvc == torch.device('cuda') and hasattr(torch.cuda, 'empty_cache'): 
                torch.cuda.empty_cache()
            data, target = data.to(self.dvc), target.to(self.dvc)
            self._target = target
            self.zero_grad()
            output = self.forward(data)
            
            output, loss = self.get_loss(output, target)
            loss.backward()
            train_loss += (loss.data.cpu().numpy() * data.size(0))
            self.optim.step()
            if hasattr(self, 'decay_s'):
                self.scheduler.step()
            elif hasattr(self, 'decay_r'):
                self.scheduler.step(loss.data)
            outputs.append(to_np(output))
            targets.append(to_np(target))
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(self.train_loader):
                self.msg_str = 'Epoch: {} - {}/{} | loss = {:.4f}'.format(epoch, batch_idx+1, len(self.train_loader), loss.data)
                for item in self.msg:
                    if hasattr(self, item):
                        self.msg_str += '   '+item+' = {:.4f}'.format(eval('self.'+item))
                sys.stdout.write('\r'+ self.msg_str)
                sys.stdout.flush()
                      
        train_loss = train_loss/ len(self.train_loader.dataset)
        outputs = np.concatenate(outputs, 0)
        targets = np.concatenate(targets, 0)
        
        self.evaluation('train', outputs, targets, train_loss)

    def test(self, epoch, dataset = 'test', n_sampling = 0):
        if dataset == 'test':
            loader = self.test_loader
        elif dataset == 'train':
            loader = self.train_loader
        else:
            loader = dataset
        
        self.eval()
        self = self.to(self.dvc)
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
                test_loss += loss.data.cpu().numpy() * data.size(0)
                outputs.append(to_np(output))
                targets.append(to_np(target))
                
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
                            self._sampling['img'].append([data[k], output[k]])
                        else:
                            self._sampling['img'].append([data[k]])
                        self._sampling['label'].append(target[k])
                    
        self._save_sample_img(epoch)
        test_loss = test_loss/ len(loader.dataset)
        outputs = np.concatenate(outputs, 0)
        targets = np.concatenate(targets, 0)
        
        self.evaluation('test', outputs, targets, test_loss)
    
    def evaluation(self, phase, output, target, loss):
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
        print(msg_str)
        
        msg_dict['loss'] = np.around(loss,4)
        # 存入DataFrame
        exec('self.'+phase+'_df = self.'+phase+'_df.append(msg_dict, ignore_index=True)')
    
    def _save_sample_img(self, epoch):
        # 存单张图片（1通道）
        import matplotlib.pyplot as plt
        
        reshape = None
        if hasattr(self, 'img_size'): reshape = self.img_size
        
        n_plot_per_sample = len(self._sampling['img'][0])
        n_sampling = len(self._sampling['img'])
        N = int(n_sampling * n_plot_per_sample)
            
        n_row, n_col = _get_fit_size(N)
        print(" - plot img ({} n_sampling)".format(N))
        
        fig = plt.figure(figsize=[n_row*4, n_col*4])
        for i in range(n_sampling):
            x, y = self._sampling['img'][i], self._sampling['label'][i]
            
            if y.size(0) > 1: y = torch.argmax(y, 0)
            else: y = torch.round(y, 2)

            for j in range(n_plot_per_sample):
                ax = fig.add_subplot(n_row, n_col, n_plot_per_sample*i+j+1)
                ax.set_title('{}, y = {}'.format(self._sampling['name'][j], y.data.numpy()))
                img = x[j].data.numpy()
                if reshape is not None: img = img.reshape((reshape[0],reshape[1]))
                ax.imshow(img)
            
        file_name = 'Epoch {} ({})'.format(epoch, N)
        plt.savefig('../save/plot/'+ file_name +'.png', bbox_inches='tight')
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