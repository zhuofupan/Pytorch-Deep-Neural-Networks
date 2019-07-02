# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import os
import sys
sys.path.append('..')
from core.plot import t_SNE

class Pre_Module(object):
    def Stacked(self):
        self.pre_modules = []
        cnt = 0
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear) and cnt < len(self.struct) - 2:
                w = layer.weight
                b = layer.bias
                self.pre_modules.append(self.add_pre_module(w.to(self.dvc), b.to(self.dvc), cnt).to(self.dvc))
                cnt+=1

    def pre_train(self, epoch, batch_size):
        self.batch_size = batch_size
        self._get_pre_feature(epoch)
        self._save_load('save', 'pre')
                
    def _get_pre_feature(self, epoch = 0, data = 'train'):
        if data == 'train':
            data_loader = self.train_loader
        else:
            data_loader = self.test_loader
        Y = data_loader.dataset.tensors[1].cpu()
        
        features = []
        for i, module in enumerate(self.pre_modules):
            if data == 'train':
                module.train_loader, module.train_set = data_loader, data_loader.dataset
                if epoch > 0:
                    for k in range(1, epoch + 1):
                        module.batch_training(k)
            else:
                module.test_loader, module.test_set = data_loader, data_loader.dataset
            
            with torch.no_grad():
                module.cpu()
                X = data_loader.dataset.tensors[0].cpu()
                X = module._feature(X).data
                data_set = Data.dataset.TensorDataset(X, Y)
                data_loader = Data.DataLoader(data_set, batch_size = self.batch_size, 
                                              shuffle = True, drop_last = False)
                features.append(X.numpy())
        return features, Y
        
    
    def _draw_pre_feature_tsne(self, loc = -1, data = 'train'):
        self._save_load('load', 'pre')
        features, Y = self._get_pre_feature(data = data)
        if not os.path.exists('../save/plot'): os.makedirs('../save/plot')
        if loc == 0:
            for i in range(len(features)):
                path ='../save/plot/['+ self.name + '] _' + data + ' {pre-layer'+ str(i+1) +'}.png'
                t_SNE(features[i], Y, path)
        else:
            path ='../save/plot/['+ self.name + '] _' + data + ' {pre-layer'+ str(len(features)) +'}.png'
            t_SNE(features[-1], Y, path)
            
        