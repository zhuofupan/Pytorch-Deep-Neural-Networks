# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import os

torch.manual_seed(1)
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Epoch(object):
    
    def batch_training(self, epoch, *args):
        if epoch == 1:
            print('\nTraining '+self.name+ ' in {}:'.format(self.dvc))
        self.train()
        self = self.to(self.dvc)
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if self.dvc == torch.device('cuda') and hasattr(torch.cuda, 'empty_cache'): 
                torch.cuda.empty_cache()
            data, target = data.to(self.dvc), target.to(self.dvc)
            self.zero_grad()
            output = self.forward(data, *args)
            loss = self.get_loss(output, target)
            loss.backward()
            train_loss += (loss.data.cpu() * data.size(0)).item()
            self.optim.step()
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(self.train_loader):
                self.msg_str = 'Epoch: {} - {}/{} | loss = {:.4f}'.format(epoch, batch_idx+1, len(self.train_loader), loss)
                for item in self.msg:
                    if hasattr(self, item):
                        self.msg_str += '   '+item+' = {:.4f}'.format(eval('self.'+item))
                sys.stdout.write('\r'+ self.msg_str)
                sys.stdout.flush()
        if hasattr(self,'drop_last') and self.drop_last: 
            train_loss = train_loss/ len(self.train_loader) / self.batch_size
        else: train_loss = train_loss/ len(self.train_loader.dataset)
        
        self.evaluation('train', train_loss)

    def batch_test(self, *args):
        self.eval()
        self = self.to(self.dvc)
        test_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                data, target = data.to(self.dvc), target.to(self.dvc)
                output = self.forward(data, *args)
                loss = self.get_loss(output, target)
                test_loss += loss.data.cpu() * data.size(0)

        test_loss = test_loss/ len(self.test_loader.dataset)
        
        if hasattr(self, 'save'):
            self.save(data, output)
        
        self.evaluation('test', test_loss)
    
    def test(self, *args):
        self.evaluation('test', None)
    
    def evaluation(self, phase, loss, *args):
        if self.task == 'usp':
            #print()
            return
        self.eval()
        self.to('cpu')
        if phase == 'train':
            dataset = self.train_set
        else:
            dataset = self.test_set
        
        with torch.no_grad():
            data, target = dataset.tensors[0].to('cpu'), dataset.tensors[1].to('cpu')
            output = self.forward(data, *args)
            if loss is None:
                loss = self.get_loss(output, target).data.numpy()
            if self.task == 'prd':
                rmse = self.get_rmse(output, target).data.numpy()
            
        output, target = output.data.numpy(), target.data.numpy()
        if self.task == 'cls':
            accuracy = self.get_accuracy(output, target)
            msg_dict = {'accuracy':accuracy}
            if phase == 'test' and accuracy > self.best_acc:
                self.best_acc = accuracy
                self.get_FDR(output, target)
                #self.save_model()
        elif self.task == 'prd':
            R2 = self.get_R2(output, target)
            msg_dict = {'rmse':rmse, 'R2':R2}
            if phase == 'test' and rmse < self.best_rmse:
                self.best_rmse = rmse
                self.best_R2 = R2
                #self.save_model()
        
        if phase == 'train':
            msg_str = '\n    >>> Train: loss = {:.4f}   '.format(loss)
        else:
            msg_str = '    >>> Test: loss = {:.4f}   '.format(loss)
        
        for key in msg_dict.keys():
            msg_str += key+' = {:.4f}   '.format(msg_dict[key])
        print(msg_str)
        
        msg_dict['loss'] = loss
        # 存入DataFrame
        exec('self.'+phase+'_df = self.'+phase+'_df.append(msg_dict, ignore_index=True)')
        