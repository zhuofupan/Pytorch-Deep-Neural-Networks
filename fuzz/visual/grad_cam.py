# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 10:10:17 2019

@author: Administrator
"""
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from misc_functions import (convert_to_grayscale,
                            save_gradient_images,
                            get_positive_negative_saliency)
import sys
sys.path.append('..')
from core.func import find_act
from _test.cls_sae import get_sae_model
from visual.plot import _get_rgb_colors, category_distribution

class GradCAM():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, last_layer_name, show_info = False):
        self.model = model
        self.last_layer = eval('self.model.'+last_layer_name)
        self.gradients = None
        self.handles = []
        self.show_info = show_info
        # Put model in evaluation mode
        self.model.eval()
        self.hook_layers()
    
    def hook_layers(self):
        def forward_function(module, ten_in, ten_out):
            self.act_value = ten_out[0]
        
        def backward_function(module, grad_in, grad_out):
            self.act_value = torch.clamp(grad_out[0] * self.act_value, min=0.0)
            print(self.act_value)

        # Register hook to the first layer, get backward values saved in 'self.gradients'
        self.layers = []
        for i, module in enumerate(self.model.modules()):
            if find_act(module) is not None: self.layers.append(module)
            if isinstance(module, nn.Linear): 
                self.layers.append(module)
                print(module.weight.size())
                module.weight_pinv = np.linalg.pinv(module.weight.data.numpy().T)
            if module == self.last_layer:
                self.handles.append(self.last_layer.register_forward_hook(forward_function))
                self.handles.append(self.last_layer.register_backward_hook(backward_function))
                break
        self.layers.reverse()
        
    def back_pass(self, x):
        for layer in self.layers:
            if layer.__call__.__name__ == 'Gaussian':
                x = np.sqrt(-1*np.log(1-x))
            if isinstance(layer, nn.Linear):
                x = np.matmul(x-layer.bias.data.numpy(), layer.weight_pinv)
        return x
        
    def generate_cam(self, input_image, target_class):
        # Input (tensor)
        input_image = torch.from_numpy(input_image).float()
        input_image.unsqueeze_(0)
        input_image = Variable(input_image, requires_grad=True)
        # Forward pass
        model_output = self.model.forward(input_image)
        # Zero gradients
        self.model.zero_grad()
        # Target for backprop (to onehot)
        if hasattr(self, 'n_category') == False:
            self.n_category = model_output.size()[-1]
        one_hot_output = torch.FloatTensor(1, self.n_category).zero_()
        one_hot_output[0][target_class] = 1
        ''' 
            Backward pass
            y.backward(arg) => x.grad = arg * ∑_yi (∂yi/∂x )T
            loss.backward = y.backward(∂loss/∂y)
        '''
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        cam = self.act_value.data.numpy()
        cam = self.back_pass(cam)
        return cam
    
    def del_handles(self):
        for handle in self.handles:
            handle.remove()
            
if __name__ == '__main__':
    model, datasets, _ = get_sae_model(2)
    model._save_load('load', 'best')
    # Guided backprop
    CAM = GradCAM(model, '_feature[8]')
    cam = CAM.generate_cam(datasets[0][4000], datasets[1][4000])
    cam = cam.reshape((40,10))
    label = np.argmax(cam, axis = 1)
    print(label.shape)
    label = np.bincount(label, minlength = 10).reshape(1,-1)
    print(label.shape)
    print(label)
    label = np.argmax(label, axis = 1)