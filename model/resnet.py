# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
import sys
sys.path.append('..')
from core.module import Module
from core.conv_module import Conv_Module

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'
}

def get_cfg(arch, base = 1):
    ''' 
        conv: (in_channels(auto), out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
        pool: (kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    ''' 
    layers = [[64,7,2,3], ['M',3,2,1]]
    k = [64, 128, 256, 512, 1024, 2048]
    if base == 1:
        layers += [
                str(arch[0]) + '*',
                ['R', '2*', 64 ]
                 ]
        for i in range(1,4):
            layers += [
                ['R', [k[i],3,2,1], k[i], '|', [k[i],1,2] ], 
                str(arch[i]-1) + '*',
                ['R', '2*', k[i] ]
                ]
    else:
        layers += [
                ['R', [k[0],1], k[0], [k[2],1], '|', [k[2],1,1] ],
                str(arch[0]-1) + '*',
                ['R', [k[0],1], k[0], [k[2],1] ]
                ]
        for i in range(1,4):
            layers += [
                ['R', [k[i],1], [k[i],3,2,1], [k[i+2],1], '|', [k[i+2],1,1] ], 
                str(arch[i]-1) + '*',
                ['R', [k[i],1], k[i], [k[i+2],1] ]
                ]
    layers += [['AA',1]]
    return layers

cfgs = {
    'resnet18': get_cfg([2,2,2,2], 1),
    'resnet34': get_cfg([3,4,6,3], 1), 
    'resnet50': get_cfg([3,4,6,3], 2), 
    'resnet101': get_cfg([3,4,23,3], 2), 
    'resnet152': get_cfg([3,8,36,3], 2) 
    }
block_id = {
    'resnet18': [1,3,5,7,9],
    'resnet34': [1,4,7,13,16], 
    'resnet50': [1,4,8,14,17], 
    'resnet101': [1,4,8,31,34], 
    'resnet152': [1,4,12,48,51]
    }
        
class ResNet(Module, Conv_Module):
    
    def __init__(self, cfg = None, 
                 batch_norm = True, use_bias = False, 
                 load_pre = None, init_weights=True, **kwargs): 
        
        if type(cfg) == str: arch = cfgs[cfg]
        else: arch = cfg
        default = {'img_size': [3, 224, 224],
                   'conv_struct': arch,
                   'conv_func': 'ReLU(True)',
                   'res_func': 'ReLU(True)',
                   'struct': [-1, 1000],
                   'dropout': 0,
                   'hidden_func': 'ReLU(True)'
                   }
        
        for key in default.keys():
            if key not in kwargs.keys():
                kwargs[key] = default[key]
                
        self.batch_norm = batch_norm
        self.use_bias = use_bias
        if type(cfg) == str:
            self._name = cfg.upper()
        else:
            self._name = 'ResNet'
        Module.__init__(self,**kwargs)
        Conv_Module.__init__(self,**kwargs)
        
        if type(cfg) == str:
            blocks = self.Convolutional()
            index = block_id[cfg]
            self.conv1, self.bn1, self.relu, self.maxpool = \
            blocks[0].conv_layers[0], blocks[0].conv_layers[1], blocks[0].act_layer, blocks[0].pool_layer
            
            self.layer1 = nn.Sequential(*blocks[index[0]:index[1]])
            self.layer2 = nn.Sequential(*blocks[index[1]:index[2]])
            self.layer3 = nn.Sequential(*blocks[index[2]:index[3]])
            self.layer4 = nn.Sequential(*blocks[index[3]:index[4]])
            
            self.layers = blocks.children()
            
        elif type(cfg) == list:
            self.conv_struct = cfg
            self.layers = self.Convolutional()
        
        self.fc = self.Sequential()
        self.opt()
        
        if init_weights:
            self._initialize_weights()
        if load_pre == True or type(load_pre) == str:
            self.load_pre(cfg, batch_norm, load_pre)
    
    def forward(self, x):
        '''
            ResBlock:
            x -> conv1 -> bn1 -> relu1 -> conv2 -> bn2 => conv_add_in
            x (-> conv3 -> bn3) => res_add_in
            conv_add_in + res_add_in -> relu2 => out 
        '''
        for layer in self.layers:
            x = layer(x)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        x = self.is_cross_entropy(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def load_pre(self, cfg, batch_norm, load_pre = True):
        if type(load_pre) == str:
            print("Load pre-trained model from ../save/para")
            self._save_load('load',load_pre)
            pre = torch.load(load_pre)
        elif load_pre == True and batch_norm: 
            if cfg in model_urls.keys():
                pre = load_state_dict_from_url(model_urls[cfg], progress=True)
            else:
                print("Cannot load pre-trained model. There is no such a model in the 'urls' list.")
                return
            print("Load pre-trained model from url")
            self.load_state_dict(pre)

if __name__ == '__main__':
    ResNet('resnet18', load_pre = True)