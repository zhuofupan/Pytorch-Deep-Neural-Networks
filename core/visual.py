# -*- coding: utf-8 -*-
import torch
import copy
import os
import numpy as np
from torch.optim import Adam, RMSprop
from torch.autograd import Variable
import sys
sys.path.append('..')
from core.plot import _save_img, _save_multi_img

def preprocess_image(pil_im, ImageNet=False):
    if ImageNet:
        # Resize image
        pil_im.thumbnail((512, 512))
    im_as_arr = np.float32(pil_im)
    if im_as_arr.ndim >2:
        im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    if ImageNet:
        # mean and std list for channels (Imagenet)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # Normalize the channels
        for channel, _ in enumerate(im_as_arr):
            im_as_arr[channel] /= 255
            im_as_arr[channel] -= mean[channel]
            im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def recreate_image(im_as_var, ImageNet=False, reshape = None):
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    if ImageNet:
        reverse_mean = [-0.485, -0.456, -0.406]
        reverse_std = [1/0.229, 1/0.224, 1/0.225]
        for c in range(3):
            recreated_im[c] /= reverse_std[c]
            recreated_im[c] -= reverse_mean[c]
        recreated_im[recreated_im > 1] = 1
        recreated_im[recreated_im < 0] = 0
        recreated_im = np.round(recreated_im * 255)
    if reshape is not None:
        recreated_im = recreated_im.reshape(reshape)
    if recreated_im.ndim == 1:
        recreated_im = recreated_im.reshape(1,-1)
    #recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    return recreated_im

class Visual():
    def __init__(self,
                 model,
                 input_dim,
                 layer_name = 'all',
                 filter_id = 0,
                 epoch = 30,
                 ImageNet = False,
                 reshape = None):
        self.model = model
        self.model.eval()
        self.input_dim = input_dim
        self.layer_name = layer_name
        self.filter_id = filter_id
        self.epoch = epoch
        self.ImageNet = ImageNet
        self.reshape = reshape
        
    def hook_layer(self):
        def hook_function(module, _in, _out):
            if isinstance(module, torch.nn.Linear):
                '''
                    in: (N, x_in)
                    out: (N, x_out)
                    weight: (x_out, x_in)
                '''
                self._output = _out
            elif isinstance(module, torch.nn.Conv2d):
                '''
                    in: (N, C_in, H_in, W_in)
                    out: (N, C_out, H_out, W_out)
                    weight: (C_out, C_in, H_kernel, W_kernel)
                '''
                self._output = _out[0, self.filter_id]                
        return self._layer.register_forward_hook(hook_function)

    def _weight(self):
        if self.layer_name != 'all':
            self._layer_name = self.layer_name
            self._layer = self.model.named_modules()[self.layer_name]
            x = self._get_input()
            self._save(x)
        else:
            for (name, layer) in self.model.named_modules():
                if hasattr(layer, 'weight') and isinstance(layer, torch.nn.BatchNorm2d) == False:
                    self._layer_name, self._layer = name, layer
                    #print(name, layer)
                    if isinstance(layer, torch.nn.Conv2d):
                        x = []
                        _loss = 0
                        self._n = layer.weight.size(0)
                        for i in range(self._n):
                            self.filter_id = i
                            x.append(self._get_input())
                            _loss += self._loss
                        self._loss = _loss/self._n
                        self._save(x)
                    else:
                        x = self._get_input()
                        self._save(x)
                    
    def _get_input(self):
        handle = self.hook_layer()
        _msg = "Visual '{}'".format(self._layer_name)
        if isinstance(self._layer, torch.nn.Conv2d):
            _msg += " , filter = {}".format(self.filter_id+1)
        # Generate a random image
        if type(self.input_dim) == int:
            random_image = np.uint8(np.random.uniform(0, 1, (self.input_dim,)))
        else:
            random_image = np.uint8(np.random.uniform(0, 1, (self.input_dim[1], self.input_dim[2], self.input_dim[0])))
        # Process image and return variable
        processed_image = preprocess_image(random_image, self.ImageNet)
        # Define optimizer for the image
#        optimizer = Adam([processed_image], lr=1e-1)
        self.optimizer = RMSprop([processed_image], lr=1e-2, alpha=0.9, eps=1e-10)
        for i in range(self.epoch):
            self.optimizer.zero_grad()
            # Assign create image to a variable to move forward in the model
            # Assign create image to a variable to move forward in the model
            self.model.forward(processed_image)
            # Loss function is the mean of the output of the selected layer/filter
            # We try to minimize the mean of the output of that specific filter
            loss = -torch.mean(self._output)
            # Backward
            loss.backward()
            self._loss = loss.item()
            # Update image
            self.optimizer.step()
            # Recreate image
            _str = _msg + " | Epoch: {}/{}, Loss = {:.4f}".format(i+1,self.epoch,self._loss)
            sys.stdout.write('\r'+ _str)
            sys.stdout.flush()

        handle.remove()
        self.model.zero_grad()
        # Save image
        processed_image.requires_grad_(False)
        created_image = recreate_image(processed_image, self.ImageNet, self.reshape)
        return created_image
    
    def _save(self, x):
        path = '../save/para/['+self.model.name + '] weights/'
        if not os.path.exists(path): os.makedirs(path)
        if type(x) == list:
            file = self._layer_name + ' (vis), loss = {:.4f}'.format(self._loss)
            _save_multi_img(x, int(np.sqrt(self._n)), path = path + file)
        else:
            file = self._layer_name + ' (vis), loss = {:.4f}'.format(self._loss)
            _save_img(x, path = path + file)
            
        sys.stdout.write('\r')
        sys.stdout.flush()
        print("Visual saved in: " + file + " "*25)