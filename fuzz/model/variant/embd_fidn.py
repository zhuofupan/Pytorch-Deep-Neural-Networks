# -*- coding: utf-8 -*-
"""
Created on Mon May  9 14:46:30 2022

@author: Fuzz4
"""

import os
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.autograd import Variable

from fuzz.core.fd_statistics import Statistics
from fuzz.data.gene_dynamic_data import ReadData
from fuzz.core.module import Module
from fuzz.model.vae import VAE
from fuzz.model.dae import Deep_AE
from scipy.linalg import toeplitz

def Tensor(x, dvc):
    tensor = torch.cuda.FloatTensor if dvc == torch.device('cuda') else torch.FloatTensor
    return tensor(x)

class Embd_DNet(Module):
    def __init__(self, **kwargs):
        default = {'basic_module': 'DAE',
                   'decoder_struct': None,               # 解码部分的结构，默认为编码部分反向
                   'decoder_func': None,
                   'latent_func': ['a', 'a'],
                   
                   'input_noise': 0,
                   'n_used_variables': 6,
                   'if_inner': True,
                   'toeplitz_mode': '1',
                   'if_times_input_dim': False,
                   
                   'is_logv2': True,
                   'output_func': None,
                   'sample_times': 5, 
                   'dropout': 0.0,
                   'L': 'MSE',                           # 'MSE' or 'BCE' (要求变量在 0 至 1 之间)
                   'alf': 1,
                   'gamma': 1,
                   'gamma_out': 1,
                   'var_msg': ['out_recon_loss'],
                   'lr': 1e-3}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        if 'basic_module' in kwargs.keys() and kwargs['basic_module'] == 'VAE':
            kwargs['var_msg'] = ['out_recon_loss', 'vae_loss', 'vae_recon_loss', 'vae_kl_loss']
        
        self._name = 'Embd_DNet'
        Module.__init__(self, **kwargs)
        
        if self.basic_module == 'DAE':
            self.basic_nn = Deep_AE(**kwargs)
        elif self.basic_module == 'VAE':
            self.basic_nn = VAE(**kwargs)
        
        # self.I_matrix = torch.eye(self.struct[0]).to(self.dvc)
        # self.E_matrix = torch.ones((self.struct[0], self.struct[0])).to(self.dvc)
        print('\nIn toeplitz matrix:')
        self.In_matrix = self.get_toeplitz_matrix(self.n_used_variables, self.toeplitz_mode, opt = 'pre')
        print('\nOut toeplitz matrix:')
        self.Out_matrix = self.get_toeplitz_matrix(self.n_used_variables, self.toeplitz_mode, opt = 'post')
        self.optimizer = self.basic_nn.optimizer
        
        if self.input_noise >0:
            self.input_noise_maker = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(self.struct[0]), torch.eye(self.struct[0])* self.input_noise )
    
    def get_toeplitz_matrix(self, nv, toeplitz_mode = '1', opt = 'pre'):
        m = self.struct[0]
        if opt == 'pre':
            l, r = np.ones(m), np.ones(m)
            right = int(np.ceil(nv/2))
            left = int(np.ceil((nv+1)/2))
            for i in range(right, m-(nv-right)): r[i] = 0
            for i in range(left, m-(nv-left)): l[i] = 0
            if toeplitz_mode == 'p':
                for k, data in enumerate( [l,r] ):
                    left, right = np.ceil(nv/2), 1
                    switch = False
                    for i in range(data.shape[0]):
                        if (data[i] == 0 or left == 0) and switch == False: switch = True
                        if data[i] == 1:
                            if switch:
                                data[i] = right; right += 1
                            else:
                                data[i] = left
                                if k == 0 and i == 0 and np.mod(nv,2) == 0: continue
                                left -= 1
            if toeplitz_mode == 'd':
                l, r = np.zeros(m), np.linspace(nv, nv-m+1, m)
                l[0] = r[0]
                for i in range(nv, m): r[i] = 0
                for i in range(m-nv+1, m): l[i] = i+1 - (m-nv+1)
            toeplitz_matrix = toeplitz(l, r)
            _row_sum = np.sum(toeplitz_matrix, axis = 1)[0]
            if self.if_times_input_dim:
                print('{}/{}*{}'.format(toeplitz_matrix, _row_sum, m))
                self.toeplitz_matrix = toeplitz_matrix/_row_sum * nv / (1 - (m - nv)/ m)
            else:
                print('{}/{}'.format(toeplitz_matrix, _row_sum))
                self.toeplitz_matrix = toeplitz_matrix/_row_sum
            return Tensor(self.toeplitz_matrix, self.dvc)
        else:
            self.toeplitz_matrix[self.toeplitz_matrix != 0] = 1
            if self.if_times_input_dim:
                print('{}/{}'.format(self.toeplitz_matrix, m))
                self.toeplitz_matrix /= m
            else:
                print('{}'.format(self.toeplitz_matrix))
            return Tensor(self.toeplitz_matrix/m, self.dvc)
    
    def decoupling(self, x):
        b, m = x.size(0), x.size(1)
        # b × m × m
        x_epd = x.view(b, m, 1) * self.In_matrix
        # b * m × m
        x_2d = x_epd.view(-1, m)
        if self.input_noise > 0 and self.training:
            rd = self.input_noise_maker.sample(torch.Size([x_2d.size(0)]))
            rd = Variable(rd, requires_grad = False).to(self.dvc)
            x_2d += rd
        recon_matrix = self.basic_nn.forward(x_2d)
        self.inner_res = (x_2d - recon_matrix).view(b,m,m)[:,self.toeplitz_matrix!=0]
        # self.inner_res = (x_2d - recon_matrix).view(b,-1)
        # return (torch.max(recon_matrix.view(b,m,m), 1).values + torch.min(recon_matrix.view(b,m,m), 1).values)/2
        # return torch.mean(recon_matrix.view(b,m,m), axis = 1)
        return torch.sum(recon_matrix.view(b,m,m) * self.Out_matrix, axis = 1)
        # return torch.diagonal(recon_matrix.view(b,m,m), dim1 = 1, dim2 = 2)
    
    def forward(self, x):
        recon = self.decoupling(x)
        self._out_recon_loss_ = torch.sum((recon - x)**2, 1)
        self.out_recon_loss = torch.mean(self._out_recon_loss_)
        if self.basic_module == 'VAE':
            self.vae_loss, self.vae_recon_loss, self.vae_kl_loss = \
                self.basic_nn.loss, self.basic_nn.recon_loss, self.basic_nn.kl_loss
        if self.if_inner:
            self.loss = self.basic_nn.loss
        else:
            self.loss = self.out_recon_loss
            if self.basic_module == 'VAE':
                self.loss = self.out_recon_loss * self.gamma_out + \
                    self.basic_nn.kl_loss * self.basic_nn.alf
        return recon
    
    def _get_customized_fdi(self, x):
        self.decoupling(x)
        return self.inner_res
    
    def save_estimated_f(self, Estimated_F, dvc):
        if not os.path.exists(self.save_path): os.makedirs(self.save_path)
        file_name = self.save_path + '/['+self.name+'] Estimated_fault (' + dvc + ').xlsx'
        writer = pd.ExcelWriter(file_name, engine='openpyxl')
        fd_data = pd.DataFrame(Estimated_F)
        fd_data.to_excel(excel_writer = writer, encoding="utf-8", index=False)
        writer.save()
        writer.close()
    
    def _bp_update_inputs(self, datasets, b, dvc = 'cuda'):
        self.eval()
        self.dvc = dvc
        self.basic_nn.dvc = self.dvc
        self.In_matrix = self.In_matrix.to(self.dvc)
        self.Out_matrix = self.Out_matrix.to(self.dvc)
        self = self.to(self.dvc)
        self.load_data(datasets, b)
        self.fd_signals = None
        with torch.no_grad():
            self.Stat = Statistics(**self.kwargs)
            self.Stat.name, self.Stat.add_info, self.Stat.run_id, self.Stat.save_path = \
                self.name, self.add_info, self.run_id, self.save_path
            if hasattr(self,'label_name'): self.Stat.label_name = self.label_name
            # offline
            inputs, latents, outputs = self._get_fdi('train')
            if hasattr(self, '_get_customized_fdi') and self.fdi == 'custo':
                self.fd_signals = self._get_fdi('train', '_get_customized_fdi')
            self.fd_thrd = \
                self.Stat.offline_modeling(inputs, latents, outputs, self.fd_signals, self.name + self.add_info)
            fd_thrd = self.fd_thrd[0]
            
        # backpropagation
        print('\nEstimating fault signals...')
        self = self.to(self.dvc)
        X, Y =  datasets[2], datasets[3]
        L = (Y.argmax(axis = 1) != 0).astype(np.int32)
        XF = X[L == 1]
        XF = XF[2000:]
        self.fdi_mean, self.cov_inverse = torch.from_numpy(self.Stat.t2_fdi_mean).float().to(self.dvc), \
                self.Stat.t2_cov_inverse.float().to(self.dvc)
        if dvc == 'cpu':
            self.bp_update_inputs_cpu(XF, fd_thrd)
        else:
            self.bp_update_inputs_gpu(XF, fd_thrd)
            
    def bp_update_inputs_cpu(self, XF, fd_thrd, learning_rate = 1e-4, per_save = 300):
        XN = []
        N = XF.shape[0]
        start = time.perf_counter()
        for i in range(N):
            xf = XF[i]
            xf = torch.from_numpy(xf).float()
            xf.unsqueeze_(0)
            xf = xf.to(self.dvc)
            # input_x = Variable(Tensor(input_x, self.dvc), requires_grad=True)
            
            input_x = torch.nn.parameter.Parameter(data = xf, requires_grad = True)
            optim = torch.optim.Adam([input_x], lr = 1e-2)
            # optim = torch.optim.RMSprop([input_x], lr = 1e-2, momentum=0.3, alpha=0.9, eps=1e-10)
            # ExpLR = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.98, last_epoch = -1)
            optim.state[optim.param_groups[0]['params'][0]] = {}
            
            T2 = np.inf
            cnt = 1
            while T2 > fd_thrd:
                self.zero_grad()
                optim.zero_grad()
                recon_x = self.forward(input_x)
                if hasattr(self, '_get_customized_fdi') and self.fdi == 'custo':
                    fdi = self.inner_res
                else:
                    fdi = input_x - recon_x
                
                if self.Stat.if_minus_mean:
                    fdi -= self.fdi_mean
                T2 = torch.mm(torch.mm(fdi.view(1,-1), self.cov_inverse), fdi.view(-1,1))[0,0]
                
                # loss = T2 - fd_thrd
                loss = self.out_recon_loss
                    
                loss_data = loss.data.cpu().numpy()
                    
                if T2 > fd_thrd:
                    loss.backward()
                    optim.step()
                    # ExpLR.step()
                    # input_x.data -= learning_rate * input_x.grad.data
                    # input_x.grad.data.zero_()
                
                if np.mod(cnt,10) == 0:
                    sys.stdout.write('\r{}/{} | cnt = {}, loss = {:.4f}                       '\
                                     .format(i+1, N, cnt, loss_data))
                    sys.stdout.flush()
                
                cnt += 1
            
            sys.stdout.write('\r{}/{} | cnt = {}, loss = {:.4f}                       '\
                                     .format(i+1, N, cnt-1, loss_data))
            sys.stdout.flush()
            
            XN.append(input_x.data.cpu().numpy().reshape(1,-1))
            del optim
            
            if np.mod(i+1, per_save) == 0:
                _XN = np.concatenate(XN, axis = 0)
                Estimated_F = XF[:_XN.shape[0]] - _XN
                self.save_estimated_f(Estimated_F, 'cpu')
                end = time.perf_counter()
                print('\n Saved {} times, taking {} seconds'.format(int((i+1)/per_save), int(end-start)))
                start = end
                
        _XN = np.concatenate(XN, axis = 0)
        Estimated_F = XF[:_XN.shape[0]] - _XN
        self.save_estimated_f(Estimated_F, 'cpu')
        end = time.perf_counter()
        print('\n Saved {} times, taking {} seconds'.format(int(i/per_save), int(end-start)))
        
        return Estimated_F
    
    def bp_update_inputs_gpu(self, XF, fd_thrd, b = 20, per_save = 10):
        N = XF.shape[0]
        max_x = np.max(np.abs(XF))
        XN = []
        sum_time = 0
        start_time = time.perf_counter()
        for i in range(int(np.ceil(N/b))):
            if (i+1) * b <= N:
                _end = int((i+1)*b)
            else:
                _end = N
            indexs = np.arange(int(i*b),_end)
            XFb = XF[indexs]
            min_max_x = np.min(np.max(np.abs(XFb), axis = 1))
            
            Nb = XFb.shape[0]
            mark = np.zeros(Nb)
            
            XFb = torch.from_numpy(XFb).float()
            XFb = XFb.to(self.dvc)
            X = torch.nn.parameter.Parameter(data = XFb, requires_grad = True)

            lr = 10/max_x**2 * min_max_x **2 + 0.01
            optim = torch.optim.Adam([X], lr = lr)
            # optim = torch.optim.RMSprop([X], lr = 1e-2, momentum=0.3, alpha=0.9, eps=1e-10)
            optim.state[optim.param_groups[0]['params'][0]] = {}
            
            cnt = 1
            while np.sum(mark) < Nb:
                self.zero_grad()
                optim.zero_grad()
                
                Recon_X = self.forward(X)
                if hasattr(self, '_get_customized_fdi') and self.fdi == 'custo':
                    FDI = self.inner_res
                else:
                    FDI = X - Recon_X

                if self.Stat.if_minus_mean:
                    FDI -= self.fdi_mean
                T2 = (FDI.unsqueeze(1) @ self.cov_inverse @ FDI.unsqueeze(-1)).view(-1,)
                mark[T2.data.cpu().numpy() < fd_thrd] = 1
                
                # loss = torch.sum((T2 - fd_thrd)*(T2 >= fd_thrd).int())
                # loss = torch.sum(self.basic_nn._loss_.view(Nb,-1) * (T2 >= fd_thrd).int() )
                loss = torch.sum(self._out_recon_loss_ * (T2 >= fd_thrd).int() )
                
                loss.backward()
                optim.step()
                
                if np.mod(cnt,10) == 0:
                    sys.stdout.write('\r{}/{} [{}/{}] | cnt = {}, loss = {:.4f}                       '\
                                     .format(int(i*b+np.sum(mark)), N, int(np.sum(mark)), Nb, cnt, loss.data.cpu().numpy()))
                    sys.stdout.flush()
            
                cnt += 1
            
            sys.stdout.write('\r{}/{} [{}/{}] | cnt = {}, loss = {:.4f}                       '\
                             .format(int(i*b+np.sum(mark)), N, int(np.sum(mark)), Nb, cnt-1, loss.data.cpu().numpy()))
            sys.stdout.flush()
            
            XN.append(X.data.cpu().numpy())
            del optim
            if np.mod(i+1, per_save) == 0:
                _XN = np.concatenate(XN, axis = 0)
                Estimated_F = XF[:_XN.shape[0]] - _XN
                self.save_estimated_f(Estimated_F, 'cuda')
                end_time = time.perf_counter()
                sum_time += int(end_time-start_time)
                print('\n Saved {} times, taking {} seconds'.format(int((i+1)/per_save), int(end_time-start_time)))
                start_time = end_time
        
        XN = np.concatenate(XN, axis = 0)
        Estimated_F = XF - XN
        end_time = time.perf_counter()
        sum_time += int(end_time-start_time)
        self.save_estimated_f(Estimated_F, 'cuda')
        print('\nFinish estimating fault signals, taking {} seconds'.format(sum_time))
        
        return Estimated_F
    
if __name__ == '__main__':
    path = '../../data/CSTR/fi'
    datasets = ReadData(path, ['st', 'oh'], dynamic = 1, task = 'fd', cut_mode = '', 
                        is_del = False, example = 'CSTR').datasets
    labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
                  'Fault 08']
    fd_dict={
              'fdi': 'res',
             # 'fdi': 'custo',
             'test_stat': 'T2',
             'thrd_sele': 'ineq',
             'confidence': 1 - 0.5/100}
    L = 'LeakyReLU(negative_slope=0.1, inplace=True)'
    parameter = {'dvc': 'cuda', 
                 'basic_module': 'DAE',
                 # 'basic_module': 'VAE',
                 
                 'label_name': labels,
                 # 'struct': [10, 100, 20],
                   'struct': [10, 200, 100, 50],
                 'hidden_func': ['g', 's', 's'],
                 'decoder_func': ['s'],
                 'output_func': 'a',
                 
                 # 'transfer_e_prop': 1.1,
                 'n_used_variables': 3,
                 # 'input_noise': 1e-3,
                 # 'if_inner': True,
                 'if_inner': False,
                 'toeplitz_mode': '1',
                 'if_to_1': True,
                 'v0_2': 1e-2,
                 'sample_times': 3, 
                 'dropout': 0,
                 'task': 'fd',
                 'view_res': False,
                 'expt_FAR': 0.5/100,
                 'esti_error': 0.005,
                 'alf': 1,
                 'gamma': 1,
                 'gamma_out': 1,
                 # 'optim':'Adam',
                 'optim':'RMSprop',
                 'optim_para': 'alpha=0.9, eps=1e-10',
                 'lr': 1e-4
                 }
    parameter.update(fd_dict)
    model = Embd_DNet(**parameter)
    # load_ = True
    load_ = False
    if load_:
        model._save_load('load','last')
        estimated_f = model.bp_update_inputs(datasets = datasets)
    else:
        model.run(datasets = datasets, e = 20, b = 16, load = '', cpu_core = 0.8, num_workers = 0)
        model.result(labels, True)
    