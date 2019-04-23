# -*- coding: utf-8 -*-
import sys
import torch
import os

torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Epoch(object):  
    def batch_training(self, epoch, *args):
        self.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if device == "cuda" and hasattr(torch.cuda, 'empty_cache'): 
                torch.cuda.empty_cache()
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.forward(data, *args)
            loss = self.get_loss(output, target)
            loss.backward()
            train_loss += loss * data.size(0)
            self.optimizer.step()
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(self.train_loader):
                self.msg_str = 'Epoch: {} - {}/{} | loss = {:.4f}'.format(epoch, batch_idx+1, len(self.train_loader), loss)
                for item in self.msg:
                    if hasattr(self, item):
                        self.msg_str += '   '+item+' = {:.4f}'.format(eval('self.'+item))
                sys.stdout.write('\r'+ self.msg_str)
                sys.stdout.flush()
        if self.drop_last: train_loss = train_loss/ len(self.train_loader) / self.batch_size
        else: train_loss = train_loss/ len(self.train_loader.dataset)
        
        self.get_evaluation('train', train_loss)

    def test(self, *args):
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_loader):
                data, target = data.to(device), target.to(device)
                output = self.forward(data, *args)
                loss = self.get_loss(output, target)
                test_loss += loss * data.size(0)

        test_loss = test_loss/ len(self.test_loader.dataset)
        
        if hasattr(self, 'save'):
            self.save(data, output)
        
        self.get_evaluation('test', test_loss)
    
    def get_evaluation(self, phase, loss, *args):
        self.eval()
        if phase == 'train':
            dataset = self.train_set
            msg_str = '\n    >>> Train: loss = {:.4f}   '.format(loss)
        else:
            dataset = self.test_set
            msg_str = '    >>> Test: loss = {:.4f}   '.format(loss)
        
        with torch.no_grad():
            data, target = dataset.tensors[0].to(device), dataset.tensors[1].to(device)
            output = self.forward(data, *args)
            if self.task == 'cls':
                accuracy = self.get_accuracy(output, target).cpu().numpy()
                msg_dict = {'accuracy':accuracy}
            elif self.task == 'prd':
                rmse = self.get_rmse(output, target).cpu().numpy()
                R2 = self.get_R2(output, target).cpu().numpy()
                msg_dict = {'rmse':rmse, 'R2':R2}
        
        for key in msg_dict.keys():
            msg_str += key+' = {:.4f}   '.format(msg_dict[key])
        print(msg_str)
        
        msg_dict['loss'] = loss.detach().cpu().numpy()
        exec('self.'+phase+'_df = self.'+phase+'_df.append(msg_dict, ignore_index=True)')
        