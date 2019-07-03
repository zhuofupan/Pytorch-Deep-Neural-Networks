# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys
import os
from torchvision.utils import save_image

torch.manual_seed(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'

def to_np(x):
    x = x.data.cpu().numpy()
    if len(x.shape) < 2:
       x = x.reshape(-1, 1) 
    return x

class Epoch(object):
    
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
            self.zero_grad()
            output = self.forward(data, target)
            loss = self.get_loss(output, target)
            loss.backward()
            
            train_loss += (loss.data.cpu().numpy() * data.size(0))
            self.optim.step()
            if hasattr(self, 'decay_s'):
                self.scheduler.step()
            elif hasattr(self, 'decay_r'):
                self.scheduler.step(loss)
            outputs.append(to_np(output))
            targets.append(to_np(target))
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(self.train_loader):
                self.msg_str = 'Epoch: {} - {}/{} | loss = {:.4f}'.format(epoch, batch_idx+1, len(self.train_loader), loss)
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
                output = self.forward(data)
                loss = self.get_loss(output, target)
                test_loss += loss.data.cpu().numpy() * data.size(0)
                outputs.append(to_np(output))
                if i == k and hasattr(self, '_img_to_save'):
                    self._save_image(epoch, data, output, target)

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
                msg_str += key+' = {}   '.format(msg_dict[key])
        print(msg_str)
        
        msg_dict['loss'] = np.around(loss,4)
        # 存入DataFrame
        exec('self.'+phase+'_df = self.'+phase+'_df.append(msg_dict, ignore_index=True)')
    
    def _save_image(self,epoch, data, output, target):
        if not os.path.exists('../save/img/['+self.name+']'): os.makedirs('../save/img/['+self.name+']')
        n = min(data.size(0), self._img_to_save[0])
        save_list = []
        for _img in self._img_to_save:
            if _img in ['data', 'output']: 
                save_list.append(eval(_img+'.view_as(data)[:n]'))
            elif _img == 'res':
                res = output-data
                save_list.append(res.view_as(data)[:n])
            elif type(_img) == str:
                save_list.append(eval('self.'+_img+'.view_as(data)[:n]'))
                
        comparison = torch.cat(save_list)
        
        _str = ''
        if self.task == 'cls': 
            target = torch.argmax(target, 1)
            _str = ',label = ['
            for i in range(n):
                if i < n - 1: _str += str(target[i]) + ', '
                else: _str += str(target[i]) + ']'

        save_image(comparison.cpu(),
                   '../save/img/[{}]/Epoch = {} {}'.format(self.name, epoch, _str) +'.png', nrow=n)