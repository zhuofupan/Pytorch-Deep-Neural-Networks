# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('..')
import torch
from torchvision.utils import save_image
from core.module import Module

noise_prob = 0.8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_noise(x, prob):
    rand_mat = torch.rand(x.size())
    noise_co = (rand_mat < prob).float().to(device)  # 噪声系数矩阵
    non_noise_co = (1-noise_co) # 保留系数矩阵
    output = x * non_noise_co
    return output, noise_co

class DAE(Module):  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Sequential()
        self.opt()
        self.unsupervised = True
        self.flatten = True
        self.msg = ['recon_loss','transfer_loss']
    
    def forward(self, x):
        
        noise_x, noise_loc = make_noise(x, noise_prob)
        self.noise_x = noise_x; self.noise_loc = noise_loc
        
        
        feature = self.feature(noise_x)
        out = self.output(feature)
        if self.training == False:
            out = noise_x + noise_loc * out
        self.loss = self.L(x, out)
        return out
        
    def save(self, data, output):
        if not os.path.exists('../results'): os.makedirs('../results')
        n = min(data.size(0), 8)
        res = output-data
        comparison = torch.cat([data.view(data.size(0), 1, 28, 28)[:n],
                                self.noise_x.view(data.size(0), 1, 28, 28)[:n],
                                output.view(data.size(0), 1, 28, 28)[:n],
                                res.view(data.size(0), 1, 28, 28)[:n]])
        
        save_image(comparison.cpu(),
                   '../results/reconstruction_' + str(epoch) + '.png', nrow=n)


if __name__ == '__main__':
    
    parameter = {'struct': [784,400,100,400,784],
                 'hidden_func': ['Gaussian', 'Affine'],
                 'output_func': 'Affine',
                 'dropout': 0.0,
                 'task': 'prd'}
    
    model = DAE(**parameter)
    model = model.to(device)
    print(model)
    
    model.load_mnist('../data', 128)
    
    for epoch in range(1, 3 + 1):
        model.train_batch(epoch)
        model.test_batch(epoch)