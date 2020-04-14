# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
import os
import sys
sys.path.append('..')
from visual.plot import t_SNE
from core.epoch import to_np

'''
    Module.modules(): get each level modules in Module's tree 
    Module.children(): get sub-level modules
    Module._modules.items(): tuple('name': sub-level Module)
'''

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
    
    def _module_feature(self, module, X, Y):
        test_set = Data.dataset.TensorDataset(X, Y)
        test_loader = Data.DataLoader(test_set, batch_size = self.pre_batch_size, 
                                              shuffle = False, drop_last = False)
        module.eval()
        module = module.to(module.dvc)
        feature = []
        
        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                data, target = data.to(module.dvc), target.to(module.dvc)
                module._target = target
                output = module._feature(data)
                feature.append(to_np(output))
        feature = np.concatenate(feature, 0)
        return torch.from_numpy(feature)

    def pre_batch_training(self, pre_epoch, pre_batch_size):
        self.pre_batch_size = pre_batch_size
        train_loader = self.train_loader
        Y = train_loader.dataset.tensors[1].cpu()
        features = []
        for k, module in enumerate(self.pre_modules):
            module.train_loader, module.train_set = train_loader, train_loader.dataset
            if pre_epoch > 0:
                for i in range(1, pre_epoch + 1):
                    module.batch_training(i)
            
            with torch.no_grad():
                X = train_loader.dataset.tensors[0].cpu()
                X = self._module_feature(module, X, Y)
                #print('\n',X.shape, type(X), Y.shape, type(Y))
                train_set = Data.dataset.TensorDataset(X, Y)
                train_loader = Data.DataLoader(train_set, batch_size = self.pre_batch_size, 
                                              shuffle = True, drop_last = False)
                features.append(X.numpy())
        return features, Y
                
    def pre_test(self, data = 'train'):
        if data == 'train':
            test_loader = self.train_loader
        else:
            test_loader = self.test_loader
        X = test_loader.dataset.tensors[0].cpu()
        Y = test_loader.dataset.tensors[1].cpu()
        
        features = []
        for k, module in enumerate(self.pre_modules):
            with torch.no_grad():
                X = self._module_feature(module, X, Y)
                features.append(X.numpy())
        return features, Y
        
    
    def _plot_pre_feature_tsne(self, loc = -1, data = 'train'):
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
            
        