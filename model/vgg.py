# -*- coding: utf-8 -*-
import sys
sys.path.append('..')
import torch.nn as nn
import numpy as np
from core.module import Module
from core.conv_module import Conv_Module
from PIL import Image
from torchvision import transforms
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
cfgs = {
    'vgg11': ['1*', 64, 'M', '1*', 128, 'M', '2*', 256, 'M', '2*', 512, 'M', '2*', 512, 'M', ['AM',7]],
    'vgg13': ['2*', 64, 'M', '2*', 128, 'M', '2*', 256, 'M', '2*', 512, 'M', '2*', 512, 'M', ['AM',7]],
    'vgg16': ['2*', 64, 'M', '2*', 128, 'M', '3*', 256, 'M', '3*', 512, 'M', '3*', 512, 'M', ['AM',7]],
    'vgg19': ['2*', 64, 'M', '2*', 128, 'M', '4*', 256, 'M', '4*', 512, 'M', '4*', 512, 'M', ['AM',7]],
}

class VGG(Module, Conv_Module):
    def __init__(self, cfg = None, 
                 batch_norm = False, use_bias = True, 
                 load_pre = None, init_weights=True, **kwargs): 

        if type(cfg) == str: arch = cfgs[cfg]
        else: arch = cfg
        
        default = {'img_size': [3, 224, 224],
                   'conv_struct': arch,
                   'conv_func': 'ReLU(True)',
                   'struct': [-1, 4096, 4096, 1000],
                   'dropout': [0, 0.5, 0.5],
                   'hidden_func': 'ReLU(True)'
                   }
        
        for key in default.keys():
            if key not in kwargs.keys():
                kwargs[key] = default[key]
        
        self.batch_norm = batch_norm
        self.use_bias = use_bias
        if type(cfg) == str:
            self._name = cfg.upper()         
        elif type(cfg) == list:
            self._name = 'VGG'
            self.conv_struct = cfg
            
        Module.__init__(self,**kwargs)
        Conv_Module.__init__(self,**kwargs)
            
        self.features = self.Convolutional('layers', auto_name = False)
        self.classifier = self.Sequential()
        self.opt()
        
        if init_weights:
            self._initialize_weights()
        if load_pre == True or type(load_pre) == str:
            self.load_pre(cfg, batch_norm, load_pre)
    
    def forward(self, x):
        x = self.features(x)
        if hasattr(self,'adaptive'):
            x = self.adaptive(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.is_cross_entropy(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def load_pre(self, cfg, batch_norm, load_pre = True):
        if type(load_pre) == str:
            print("Load pre-trained model from ../save/para")
            self._save_load('load',load_pre)
        elif load_pre == True:
            if batch_norm: cfg += '_bn'
            if cfg in model_urls.keys():
                pre = load_state_dict_from_url(model_urls[cfg], progress=True)
            else:
                print("Cannot load pre-trained model. There is no such a model in the 'urls' list.")
                return
            print("Load pre-trained model from url")
            self.load_state_dict(pre)
    
    def img2tensor(self, x):
        trans = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], #这是imagenet數據集的均值
                                     std=[0.229, 0.224, 0.225])
                ])
        x = trans(x)
        x.unsqueeze_(dim=0)
        return x
    
    def test_a_img(self, x):
        if isinstance(x, str):
            x = Image.open(x)
        self.img2tensor(x)
        x.show()
        print('Img has a shape of {}'.format(x.shape))
        out=self(x)
        outnp=out.data[0]
        ind=int(np.argmax(outnp))
        print('Test class is {}'.format(ind))

if __name__ == '__main__':
    VGG('vgg11', batch_norm = True, load_pre = True)
#    VGG([[64, 3, 2], 'B', 'M', 64, ['B', '01'], ['', 0]])
