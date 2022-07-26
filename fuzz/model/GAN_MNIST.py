# -*- coding: utf-8 -*-

"""
    @Name:GAN of MNIST

    @Time: 2022/1/10

    @Author: Yccc7
"""

import torch
from torch.autograd import Variable
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from fuzz.core.module import Module

class GAN_mnist(Module):

    def __init__(self, **kwargs):

        default = {
            '_name': 'GAN_mnist',
            'var_msg': ['G_loss', 'D_loss'],
            'G_struct': None,   
            'G_func': None,  
            'D_struct': None,  
            'D_func': None
        }

        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]

        Module.__init__(self, **kwargs) 

  
        self.generator = self.Sequential(struct = self.G_struct, 
                                         hidden_func = self.G_func,
                                         output_func = 't',
                                         add_bn = 0.8)
       
        self.discriminator = self.Sequential(struct=self.D_struct, 
                                             hidden_func = self.D_func,
                                             output_func = 's')
        
        self.opt(parameters = self.generator.parameters(), info=False)
        self.G_optim = self.optim
        
        self.opt(parameters = self.discriminator.parameters(), info=False)
        self.D_optim = self.optim
        
        self.L = torch.nn.BCELoss()
        
        self.__print__()
        
        # sampling
        self.mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.G_struct[0]), torch.eye(self.G_struct[0]))
    
    def forward(self, x):
        valid = Variable(torch.Tensor(x.size(0), 1).fill_(1.0), requires_grad=False).to(self.dvc)
        fake = Variable(torch.Tensor(x.size(0), 1).fill_(0.0), requires_grad=False).to(self.dvc)
        # Housekeeping - reset gradient
        self.G_optim.zero_grad()
        
        # Generator forward-loss-backward-update
        z = Variable(torch.Tensor(np.random.normal(0, 1, (x.shape[0], self.G_struct[0])))).to(self.dvc)
        # z = self.mv_normal.sample(torch.Size([x.size(0)])).to(self.dvc)
        G_fake = self.generator(z)
        D_fake = self.discriminator(G_fake)
        
        # G_loss
        # self.G_loss = -torch.mean(torch.log(D_fake))
        self.G_loss = self.L(D_fake, valid)
        if self.training:
            self.G_loss.backward()
            self.G_optim.step()
        
        # Housekeeping - reset gradient
        self.D_optim.zero_grad()
        
        # Dicriminator forward-loss-backward-update
        # z = Variable(torch.Tensor(np.random.normal(0, 1, (x.shape[0], self.G_struct[0])))).to(self.dvc)
        # z = self.mv_normal.sample(torch.Size([x.size(0)])).to(self.dvc)
        # G_fake = self.generator(z)
        self.G_fake = G_fake
        D_fake = self.discriminator(G_fake.detach())
        D_real = self.discriminator(x)
        
        # D_loss
        # self.D_loss = -1/2 * torch.mean( torch.log(D_real) + torch.log(1-D_fake))
        real_loss = self.L(D_real, valid)
        fake_loss = self.L(D_fake, fake)
        self.D_loss = (real_loss + fake_loss) / 2
        
        if self.training:
            self.D_loss.backward()
            self.D_optim.step()
        
        return D_fake
    
    def batch_training(self, epoch):
        # .data 不能被 autograd 追踪求微分，.detach()可以
        self = self.to(self.dvc)
        self.train()
        
        D_loss, G_loss, sample_count = 0, 0, 0 
        outputs = []
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if data.size(1) != 784:
                data = data.view((-1, 784))
            data, target = data.to(self.dvc), target.to(self.dvc)
            D_fake = self.forward(data)
            
            sample_count += data.size(0)
            G_loss += (self.G_loss.data.cpu() * data.size(0))
            D_loss += (self.D_loss.data.cpu() * data.size(0))
            for k in range(len(self.var_msg)):
                self.var_msg_value[k] += (eval('self.'+self.var_msg[k]).data.cpu() * data.size(0))
            
            outputs.append(D_fake.data.cpu())
            if (batch_idx+1) % 10 == 0 or (batch_idx+1) == len(self.train_loader):
                self.msg_str = 'Epoch: {} - {}/{} | G_loss = {:.4f}, D_loss = {:.4f}'.format(
                    epoch, batch_idx+1, len(self.train_loader), self.G_loss.data, self.D_loss.data)
                sys.stdout.write('\r'+ self.msg_str)
                sys.stdout.flush()
                      
        G_loss = G_loss/ sample_count
        D_loss = D_loss/ sample_count
        for k in range(len(self.var_msg)):
            self.var_msg_value[k] /= sample_count
        outputs = torch.cat(outputs, 0)
        self.save_img(epoch)
        return outputs
    
    def save_img(self, epoch):
        gen_imgs = self.G_fake.view((self.G_fake.size(0), 1, 28, 28))
        save_image(gen_imgs.data[:25], "gan/%d.png" % epoch, nrow=5, normalize=True)
        print('\nSave img in gan/%d.png' % epoch)
    
    def test(self, epoch = 0, dataset = 'test', n_sampling = 0):
        if hasattr(self, 'test_loader'):
            loader = self.test_loader
        else:
            loader = self.train_loader
        self = self.to(self.dvc)
        self.eval()
        
        D_loss, G_loss, sample_count = 0, 0, 0
        outputs = []
        
        with torch.no_grad():
            if n_sampling > 0:
                batch_id = []
                # 选 n_sampling 个批次，从每个批次采一个样本
                if n_sampling < len(loader): 
                    batch_id = np.random.choice(len(loader), n_sampling, replace = False)
                
            for i, (data, target) in enumerate(loader):
                if data.size(1) != 784:
                    data = data.view((-1, 784))
                data, target = data.to(self.dvc), target.to(self.dvc)
                
                D_fake = self.forward(data)
                
                sample_count += data.size(0)
                G_loss += (self.G_loss.data.cpu() * data.size(0))
                D_loss += (self.D_loss.data.cpu() * data.size(0))
                
                outputs.append(D_fake.data.cpu())
                # save img
                if n_sampling > 0 and i in batch_id:
                    k = np.random.randint(0, data.size(0))
                    # + data, + feature
                    imgs = [data[k].data.cpu().numpy(), self.G_fake[k].data.cpu().numpy()]
                    # + label
                    labels = [target[k].data.cpu().numpy().argmax(), D_fake[k].data.cpu().numpy()]
        
        if n_sampling > 0:         
            self.show_img(epoch, imgs, labels)
        G_loss = G_loss/ sample_count
        D_loss = D_loss/ sample_count

        return outputs
    
    def _load_mnist(self, path = '../data/MNIST', batch_size = 64):
        # os.makedirs(path, exist_ok=True)
        
        train_set = datasets.MNIST(
                path,
                train=True,
                download=False,
                transform=transforms.Compose(
                    [transforms.Resize(28), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            )
        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        )
        self.train_X, self.train_Y = train_set.data.numpy().astype(float), train_set.targets.numpy().astype(int)       
        print('->  train_X{},  train_Y{}'.\
              format(self.train_X.shape, self.train_Y.shape))
    
    def show_img(self, epoch, imgs, labels):
        fig = plt.figure(figsize=[12.5,12.5])
        for idx in range(len(imgs)):
            img = imgs[idx]
            title = ['Real-'+str(labels[0]), 'Fake-'+str(np.round(labels[1],2))]
            i = 121 + idx
            ax = fig.add_subplot(i)
            ax.set_title(title[idx])
            print(img.shape)
            n = int(np.sqrt(img.shape[0]))
            img = img.reshape((n,n))
            ax.imshow(img)
        file_name = 'Epoch {}'.format(epoch)
        path = '../save/'+ self.name + self.run_id + '/sampling/'
        if not os.path.exists(path): 
            os.makedirs(path)
        print('\nSave img in '+ path)
        plt.savefig(path + file_name +'.pdf', bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    img_size = 784
    parameter = {
        'G_struct': [100, 128, 256, 512, 1024, img_size], 
        'G_func': 'LeakyReLU(negative_slope=0.2, inplace=True)',  
        'D_struct': [img_size, 512, 256, 1],  
        'D_func': 'LeakyReLU(negative_slope=0.2, inplace=True)',  
        'task': 'gnr',
        'optim':'Adam',
        'optim_para': 'betas=(0.5, 0.999)',
        'lr': 2e-4,
        'dropout': 0,
        'flatten': True,
        'n_sampling': 0
    }
    
    model = GAN_mnist(**parameter)
    model._load_mnist()
    # model.load_mnist('../data', 64)
    model.run(e = 30, b = 64)