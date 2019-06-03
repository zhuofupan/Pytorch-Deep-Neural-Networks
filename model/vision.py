# -*- coding: utf-8 -*-
from torchvision.models.vgg import *
from torchvision.models.resnet import *
'''
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19'
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d'
'''
def print_model(model, para = True):
    print()
    print(model)
    #print parameters
    if para:
        print('Parameters(')
        for key, v in model.state_dict().items():
            print('  {}:\t{}'.format(key,v.size()))
        print(')')

print_model(resnet18(), True)