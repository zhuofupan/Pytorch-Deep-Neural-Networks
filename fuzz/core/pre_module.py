# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data
import numpy as np
import os
from ..visual.plot import t_SNE

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
                self.pre_modules.append(self.add_pre_module(w, b, cnt))
                cnt += 1
    
    def _sub_batch_test(self, module, X, Y):
        # 子模型 module 的前向传递: module._feature(data)
        test_set = Data.dataset.TensorDataset(X, Y)
        test_loader = Data.DataLoader(test_set, batch_size = self.pre_batch_size, 
                                      shuffle = False, drop_last = False, **self.loader_kwargs)
        module.eval()
        module = module.to(module.dvc)
        feature = []
        
        for (data, _) in test_loader:
            data = data.to(module.dvc)
            hidden = module._feature(data)
            feature.append(hidden.data.cpu())
            
        feature = torch.cat(feature, 0)
        del test_set
        del test_loader
        return feature

    def pre_batch_training(self, pre_epoch, pre_batch_size):
        # 预训练 所有子模型
        self.pre_batch_size = pre_batch_size
        train_loader = self.train_loader
        X = self.train_loader.dataset.tensors[0].data.cpu()
        Y = self.train_loader.dataset.tensors[1].data.cpu()
        features = []
        for module in self.pre_modules:
            module.train_loader, module.train_set = train_loader, train_loader.dataset
            module.dvc_info = self.dvc_info
            
            # 训练 sub_module
            print('\nTraining '+module.name+ ' in {}'.format(module.dvc) + module.dvc_info +':')
            for i in range(1, pre_epoch + 1):
                module.batch_training(i)
            print()

            # 前向特征提取
            X = self._sub_batch_test(module, X, Y)
            train_set = Data.dataset.TensorDataset(X, Y)
            train_loader = Data.DataLoader(train_set, batch_size = self.pre_batch_size, 
                                           shuffle = True, drop_last = False, **self.loader_kwargs)
            features.append(X.numpy())
        self.pre_features = features
        # self._check_pre_training_effect()
        return features, Y
    
    def _check_pre_training_effect(self):
        print('\nTest pre-training effect...')
        for module in self.pre_modules:
            print('Parameters in {}:'.format(module.name))
            for key, v in module.state_dict().items():
                print('  {}:\t{:.6f}\t{:.6f}'.format(key, v.mean(), v.var()))
        print('Parameters in {}:'.format(self.name))
        for key, v in self.state_dict().items():
            print('  {}:\t{:.6f}\t{:.6f}'.format(key, v.mean(), v.var()))
    
    def pre_test(self, data = 'train'):
        # 获取 各层特征
        if data == 'train':
            test_loader = self.train_loader
        elif data == 'test':
            test_loader = self.test_loader
        else:
            test_loader = data
            
        X = test_loader.dataset.tensors[0].data.cpu()
        Y = test_loader.dataset.tensors[1].data.cpu()
        
        features = []
        for _, module in enumerate(self.pre_modules):
            X = self._sub_batch_test(module, X, Y).data.cpu()
            features.append(X.numpy())
        return features, Y
    
    def _plot_pre_feature_tsne(self, loc = -1, data = 'train'):
        if data == 'train':
            features, Y = self.pre_features, self.train_loader.dataset.tensors[1]
        else:
            self._save_load('load', 'pre')
            features, Y = self.pre_test(data = data)
        Y = Y.data.cpu().numpy()
            
        path = '../save/'+ self.name + self.run_id + '/'
        file_name = '['+ self.name + '] _' + data + ' (pre-layer-'+ str(len(features)) +').png'
        show_info = True
        if hasattr(self, 'tsne_info'):
            show_info = self.tsne_info
        t_SNE(features[loc], Y, path, file_name, show_info = show_info)
            
        