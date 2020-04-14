# -*- coding: utf-8 -*-
import sys
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torch.utils.data as Data
from sklearn.preprocessing import MinMaxScaler

class Run():
    def _get_loader(self, batch_size):
        train_set = datasets.MNIST('../data', train=True, download=True)
        test_set = datasets.MNIST('../data', train = False)
        self.train_X, self.train_Y = train_set.data.numpy().astype(float), train_set.targets.numpy().astype(int)
        self.test_X, self.test_Y = test_set.data.numpy().astype(float), test_set.targets.numpy().astype(int)  
        self.train_X = self.train_X.reshape((-1,28*28))
        self.test_X = self.test_X.reshape((-1,28*28))
        scaler = MinMaxScaler()
        scaler.fit(self.train_X)
        self.train_X, self.test_X = scaler.transform(self.train_X), scaler.transform(self.test_X)
        
        self.train_set = Data.dataset.TensorDataset(torch.from_numpy(self.train_X).float(), 
                                                    torch.from_numpy(self.train_Y).float())
        self.train_loader = Data.DataLoader(self.train_set, batch_size = batch_size, 
                                            shuffle = True, drop_last = False)
        self.test_set = Data.dataset.TensorDataset(torch.from_numpy(self.test_X).float(), 
                                                   torch.from_numpy(self.test_Y).float())
        self.test_loader = Data.DataLoader(self.test_set, batch_size = batch_size, 
                                           shuffle = False, drop_last = False)
        return self.train_loader, self.test_loader
        
    def _train(self, epoch, dataloader):
        print('Training...')
        self.train()
        for e in range(1, epoch + 1):
            train_loss = 0.0
            for batch_idx, (data, target) in enumerate(dataloader):
                self.zero_grad()
                self.forward(data)
                loss = self.loss
                loss.backward()
                train_loss += (loss.data.cpu().numpy() * data.size(0))
                self.optim.step()
                
                if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(dataloader):
                    msg_str = 'Epoch: {} - {}/{} | loss = {:.4f}'.format(e, batch_idx+1, len(dataloader), loss.data)
                    sys.stdout.write('\r'+ msg_str)
                    sys.stdout.flush()
                          
            train_loss = train_loss/ len(dataloader.dataset)
            print()
            
    def _test(self, dataloader):
        print('Test...')
        self.eval()
        test_loss = 0.0
        idx = np.random.randint(len(dataloader))
        for batch_idx, (data, target) in enumerate(dataloader):
            self.zero_grad()
            feature, recon = self.forward(data)
            loss = self.loss
            test_loss += (loss.data.cpu().numpy() * data.size(0))
            if batch_idx == idx:
                k = np.random.randint(data.size(0))
                self.img_sample = [data[k], feature[k], recon[k]]
                      
        test_loss = test_loss/ len(dataloader.dataset)
        print('loss = {:.4f}'.format(loss.data))
    
    def show_img(self):
        fig = plt.figure(figsize=[12.5,12.5])
        title = ['Input', 'Hidden', 'Output']
        for idx, img in enumerate(self.img_sample):
            i = 131 + idx
            ax = fig.add_subplot(i)
            ax.set_title(title[idx])
            n = int(np.sqrt(img.size(0)))
            img = img.data.numpy().reshape((n,n))
            ax.imshow(img)
        plt.show()

class AE(nn.Module, Run):
    def __init__(self,_in, _out):
        nn.Module.__init__(self)
        self.encoder = nn.Sequential(nn.Linear(_in,_out),
                                         nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(_out,_in),
                                         nn.Sigmoid())
        self.optim = torch.optim.Adam(self.parameters())
        self.L = torch.nn.MSELoss()
    
    def forward(self, x):
        feature = self.encoder(x)
        recon = self.decoder(feature)
        self.loss = self.L(recon, x)
        return feature, recon
        
module = AE(28*28, 10*10)
train_data, test_data = module._get_loader(64)
module._train(1, train_data)
module._test(test_data)
module.show_img()
