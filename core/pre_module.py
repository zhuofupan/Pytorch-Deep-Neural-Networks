# -*- coding: utf-8 -*-
import torch
import torch.utils.data as Data

class Pre_Module(object):
    def Stacked(self):
        self.pre_modules = []
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                w = layer.weight
                b = layer.bias
                self.pre_modules.append(self.add_pre_module(w.to(self.dvc), b.to(self.dvc)).to(self.dvc))
    
    def get_pre_loader(self, batch_size, train_X):
        train_set = Data.dataset.TensorDataset(train_X, train_X)
        train_loader = Data.DataLoader(train_set, batch_size = batch_size, 
                                       shuffle = True, drop_last = False)
        return train_loader

    def pre_train(self, epoch, batch_size):
        train_loader = self.get_pre_loader(batch_size, self.train_set.tensors[0])
        
        for i, module in enumerate(self.pre_modules):
            module.name += '-{}'.format(i+1)
            module.train_loader = train_loader
            module.train_set = train_loader.dataset
            for k in range(1, epoch + 1):
                module.batch_training(k)
            with torch.no_grad():
                module.cpu()
                train_X = module.feature(train_loader.dataset.tensors[0].cpu())
                train_loader = self.get_pre_loader(batch_size, train_X)