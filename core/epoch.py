# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys
import os
sys.path.append('..')
from core.plot import _save_multi_img

torch.manual_seed(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'

def to_np(x):
    x = x.data.cpu().numpy()
    if len(x.shape) < 2:
       x = x.reshape(-1, 1) 
    return x

class Epoch(object):
    
    def run(self, datasets, e = 100, b = 64, pre_e = 0, load = ''):
        self.load_data(datasets, b)
        
        if load == 'pre':
            self._save_load('load', 'pre')
        elif pre_e > 0:
            self.pre_train(pre_e, b)
            
        if load == 'best':
            self._save_load('load', 'best')
        else:
            for epoch in range(1, e + 1):
                self.batch_training(epoch)
                self.test(epoch)
    
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
            loss = self.get_loss(output, target)
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

    def test(self, epoch):
        self.eval()
        self = self.to(self.dvc)
        test_loss = 0
        outputs = []
        with torch.no_grad():
            k = np.random.randint(len(self.test_loader))
            for i, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.dvc), target.to(self.dvc)
                self._target = target
                output = self.forward(data)
                loss = self.get_loss(output, target)
                test_loss += loss.data.cpu().numpy() * data.size(0)
                outputs.append(to_np(output))
                if i == k and hasattr(self, '_img_to_save'):
                    self._save_test_img([data, output], self._img_to_save, epoch, target)

        test_loss = test_loss/ len(self.test_loader.dataset)
        outputs = np.concatenate(outputs, 0)
        targets = self.test_Y
        
        self.evaluation('test', outputs, targets, test_loss)
    
    def evaluation(self, phase, output, target, loss):
        if self.task == 'usp': 
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
        
    def _save_test_img(self, data, _add_data = None, epoch = None, target = None):
        '''
            _img_to_save: ['res','...']
        '''
        
        path = '../save/img/['+self.name+']/'
        if not os.path.exists(path): os.makedirs(path)
        path += 'Epoch = {}'.format(epoch)
        
        data_list = []
        n = 8
        
        # [data, output]
        for _d in data: 
            data_list.append(_d[:n])

        # _add_data
        if _add_data is not None:
            if type(_add_data) != list: _add_data = [_add_data]
            for _s in _add_data:
                if _s == 'res':  _d = data[1] - data[0]
                else:  _d = eval('self.'+_s)
                data_list.append(_d[:n])
            
        # _add_info
        if self.task == 'cls':
            target = target[:n].cpu().numpy()
            if target.ndim >1: target = np.argmax(target, 1)
            path += ' ,label = ['
            for i in range(n):
                if i < n - 1: path += str(target[i]) + ' '
                else: path += str(target[i]) + ']'
        
        _save_multi_img(data_list, data_list[0].shape[0], path = path)