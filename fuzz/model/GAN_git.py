# -*- coding: utf-8 -*-

import os
import sys
import argparse

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

import torch
from torch.autograd import Variable

from fuzz.core.module import Module

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

class GAN_MNIST(Module):

    def __init__(self, **kwargs):

        default = {
            '_name': 'GAN_MNIST',
            'var_msg': ['G_loss', 'D_loss'],
            'G_struct': [100, 128, 256, 512, 1024, 784], 
            'G_func': 'LeakyReLU(negative_slope=0.2, inplace=True)',  
            'D_struct': [784, 512, 256, 1],  
            'D_func': 'LeakyReLU(negative_slope=0.2, inplace=True)',  
            'task': 'gnr',
            'optim':'Adam',
            'optim_para': 'betas=(%f, %f)'%(opt.b1, opt.b2),
            'lr': opt.lr,
            'dropout': 0,
            'flatten': True,
            'n_sampling': 0
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
        
        self.optimizer_G = self.opt(parameters = 'self.generator.parameters()', info=False)

        self.optimizer_D = self.opt(parameters = 'self.discriminator.parameters()', info=False)
        
        self.L = torch.nn.BCELoss()
        
        self.__print__()
        
        # sampling
        self.mv_normal = torch.distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(self.G_struct[0]), torch.eye(self.G_struct[0]))
    
    def load_mnist(self, path = '../data/MNIST'):
        self.dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                path,
                train=True,
                download=False,
                transform=transforms.Compose(
                    [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=opt.batch_size,
            shuffle=True,
        )
    
    def run(self):
        if cuda:
            self.cuda()
        
        print('\nTraining...')
        for epoch in range(opt.n_epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                
                real_disc, fake_disc = self.forward(imgs)
                
                msg_str ="[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"\
                    % (epoch + 1, opt.n_epochs, i, len(self.dataloader), self.d_loss.item(), self.g_loss.item())
                sys.stdout.write('\r'+ msg_str)
                sys.stdout.flush()
                
                # batches_done = epoch * len(self.dataloader) + i
            msg = 'images/%d (R %.2f, F %.2f).png' % (epoch + 1, torch.mean(real_disc[:25]).item(), \
                                               torch.mean(fake_disc[:25]).item())
            print('\nSave img in ' + msg)
            img = self.gen_imgs.view(self.gen_imgs.size(0), *img_shape)
            save_image(img.data[:25], msg, nrow=5, normalize=True)
    
    def forward(self, imgs):
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).view(imgs.size(0), -1)

        # -----------------
        #  Train Generator
        # -----------------

        self.optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(self.mv_normal.sample(torch.Size([imgs.shape[0]])).to(self.dvc))
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        self.gen_imgs = self.generator(z)

        # Loss measures generator's ability to fool the discriminator
        self.g_loss = self.L(self.discriminator(self.gen_imgs), valid)

        self.g_loss.backward()
        self.optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        self.optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_disc = self.discriminator(real_imgs)
        fake_disc = self.discriminator(self.gen_imgs.detach())
        real_loss = self.L(real_disc, valid)
        fake_loss = self.L(fake_disc, fake)
        self.d_loss = (real_loss + fake_loss) / 2

        self.d_loss.backward()
        self.optimizer_D.step()
        return real_disc, fake_disc
    
if __name__ == '__main__':
    model = GAN_MNIST()
    model.load_mnist()
    model.run()