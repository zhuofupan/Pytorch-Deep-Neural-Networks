# -*- coding: utf-8 -*-
import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.autograd import Variable
import numpy as np

import sys
sys.path.append('..')
from core.module import Module
from core.pre_module import Pre_Module
from core.layer import Linear2

class RBM(torch.nn.Module):
    def __init__(self, w, b, unit_type, cnt, **kwargs):
        default = {'cd_k': 1, 
                   'lr': 1e-3}
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])
        
        kwargs['task'] = 'usp'
        self.name = 'RBM-{}'.format(cnt+1)
        self.unit_type = unit_type
        self.dvc = kwargs['dvc']
        super().__init__()
        
        self.w, self.b = w, b
        self.b2 = Parameter(torch.Tensor(w.size(1)))
        init.constant_(self.b2, 0)
        
        #print_module:
        print()
        #print_parameter:
        print("{}'s Parameters(".format(self.name))
        print('  unit\t{}-{}'.format(unit_type[0], unit_type[1]))
        for name, para in self.named_parameters(): print('  {}\t{}'.format(name, para.size()))
        print(')')
    
    def transfrom(self, x, direction):
        if direction == 'v2h':
            i = 0
            z = x @ self.w.t() + self.b
        else:
            i = 1
            z = x @ self.w + self.b2
        if self.unit_type[i] in ['Binary', 'b']:
            p = F.sigmoid(z)
            s = (torch.rand(p.size())< p).float().to(self.dvc)
            return p, s
        elif self.unit_type[i] in ['Gaussian', 'g']:
            u = z
            s = u
            return u, s
    
    def _feature(self, x):
        _, out = self.transfrom(x,'v2h')
        return out
    
    def forward(self, x):
        v0 = x
        ph0, h0 = self.transfrom(v0,'v2h') 
        pvk, vk = self.transfrom(h0,'h2v')
        for k in range(self.cd_k-1):
            phk, hk = self.transfrom(vk,'v2h')
            pvk, vk = self.transfrom(hk,'h2v')
        phk, hk = self.transfrom(vk,'v2h')
        vk = pvk
        hk = phk
        return v0, h0, vk, hk
    
    def _update(self, v0, h0, vk, hk):
        positive = torch.bmm(h0.unsqueeze(-1),v0.unsqueeze(1))
        negative = torch.bmm(hk.unsqueeze(-1),vk.unsqueeze(1))
        
        delta_w = positive - negative
        delta_b = h0 - hk
        delta_a = v0 - vk
        
        self.w += (torch.mean(delta_w, 0) * self.lr)
        self.b += (torch.mean(delta_b, 0) * self.lr)
        self.b2 += (torch.mean(delta_a, 0) * self.lr)
        
        l1_w, l1_b, l1_a = torch.mean(torch.abs(delta_w)), torch.mean(torch.abs(delta_b)), torch.mean(torch.abs(delta_a))
        return l1_w, l1_b, l1_a
    
    def batch_training(self, epoch):
        if epoch == 1:
            print('\nTraining '+self.name+ ' in {}'.format(self.dvc) + self.dvc_info +':')
        self = self.to(self.dvc)
        self.eval()
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data = data.to(self.dvc)
                v0,h0,vk,hk = self.forward(data)
                # print(v0.mean(), h0.mean(),vk.mean(),hk.mean())
                l1_w, l1_b, l1_a = self._update(v0, h0, vk, hk)
                if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(self.train_loader):
                    msg_str = 'Epoch: {} - {}/{} | |Δw| = {:.4f}, |Δb| = {:.4f}, |Δa| = {:.4f}'.format(
                            epoch, batch_idx + 1, len(self.train_loader), l1_w, l1_b, l1_a)
                    sys.stdout.write('\r'+ msg_str)
                    sys.stdout.flush()
        
class DBN(Module, Pre_Module):  
    def __init__(self, **kwargs):
        if 'name' in kwargs.keys(): 
            kwargs['_name'] = kwargs['name']
            del kwargs['name']
        if '_name' not in kwargs.keys(): kwargs['_name'] = 'DBN'
        
        # 检测是否设置单元类型 - 用于预训练
        if 'v_type' not in kwargs.keys():
            kwargs['v_type'] = ['Gaussian']
        if 'h_type' not in kwargs.keys():
            kwargs['h_type'] = ['Gaussian']
        
        Module.__init__(self, **kwargs)
        
        if type(self.h_type) != list:
            self.h_type = [self.h_type]
        
        # 如果未定义 hidden_func 则按单元类型给定 - 用于微调
        if hasattr(self,'hidden_func') == False:
            self.hidden_func = []
            for tp in self.h_type:
                if tp in ['Gaussian', 'g']: self.hidden_func.append('a')
                elif tp in ['Binary', 'b']: self.hidden_func.append('s')
                else: raise Exception("Unknown h_type!")
            
        self._feature, self._output = self.Sequential(out_number = 2)
        self.opt()
        self.Stacked()

    def forward(self, x):
        x = self._feature(x)
        x = self._output(x)
        return x
    
    def add_pre_module(self, w, b, cnt):
        if type(self.v_type) != list: 
            self.v_type = [self.v_type]
        v_type = self.v_type[np.mod(cnt, len(self.v_type))]
        h_type = self.h_type[np.mod(cnt, len(self.h_type))]
        
        rbm = RBM(w, b, [v_type, h_type], cnt, **self.kwargs)
        return rbm
