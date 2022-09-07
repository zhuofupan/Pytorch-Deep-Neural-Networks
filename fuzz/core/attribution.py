# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 20:37:51 2021

@author: owner
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

import sys
import time
import numpy as np

from .func import find_act
from .fd_statistics import Statistics, Save_excel
from ..data.load import _loader_kwargs
from ..visual.plot import category_distribution
from ..model.variant.embd_fidn import Tensor

def view_info(name, v):
    if type(v) != list: v = list(v)
    print('\n>>>  {}: len = {}'.format(name, len(v)))
    np.set_printoptions(precision=4, suppress=True)
    for i, var in enumerate(v):
        if var is None:
            print('{}: {}'.format(i, var) )
        else:
            print('{}: size({}), mean({:.4f}), min({:.4f}), max({:.4f})'.format(i, var.size(), 
                                                                         var.mean().data.cpu().numpy(),
                                                                         var.min().data.cpu().numpy(),
                                                                         var.max().data.cpu().numpy()))

class TestAttri():

    def save_data(self, record_result, label, plot_text):
        if self.ignore_0: 
            record_result = record_result[:, 1:]

        print()
        add_info = ''
        # for CSTR
        if record_result.shape[1] == 10:
            real_cause_order = [0,1,2,3,4,8,5,6,9,7]
            prd_result = np.diag(record_result[real_cause_order])[3:]
            total = record_result.sum(axis = 0)[3:]
            attri_acc = np.round(prd_result/total*100, 2)
            avg_attri_acc = np.round(prd_result.sum()/total.sum()*100, 2)
            add_info = '{}'.format(avg_attri_acc)
            print(avg_attri_acc)
            print(attri_acc)
        
        # plot result
        if self.show_op_info == False:
            category_distribution(record_result, label = label, name = self.model.name,
                                  add_info = ' {} {}'.format(self.name, add_info),
                                  text = plot_text, diag_cl = False)
        self.result = record_result
    
    def test_attri_for_cls(self, datasets, dynamic = 1, labels = None, dvc = 'cpu'):
        # define data loader for test acc
        train_set = Data.dataset.TensorDataset(torch.from_numpy(datasets[0]).float(), 
                                               torch.from_numpy(datasets[1]).float())
        train_loader = Data.DataLoader(train_set, batch_size = 32, 
                                       shuffle = False, drop_last = False, **_loader_kwargs(dvc))
        
        test_set = Data.dataset.TensorDataset(torch.from_numpy(datasets[2]).float(), 
                                              torch.from_numpy(datasets[3]).float())
        test_loader = Data.DataLoader(test_set, batch_size = 32, 
                                      shuffle = False, drop_last = False, **_loader_kwargs(dvc))
        
        X = np.concatenate((datasets[0], datasets[2]), axis = 0)
        Y = np.concatenate((datasets[1], datasets[3]), axis = 0)
        
        con_data = Data.dataset.TensorDataset(torch.from_numpy(X).float(), 
                                              torch.from_numpy(Y).float())
        con_loader = Data.DataLoader(con_data, batch_size = 32, 
                                     shuffle = False, drop_last = False, **_loader_kwargs(dvc))
        
        # define data loader for test attri
        Y = np.argmax(Y, axis = 1)
        attri_data = Data.dataset.TensorDataset(torch.from_numpy(X).float(), 
                                                torch.from_numpy(Y).float())
        attri_loader = Data.DataLoader(attri_data, batch_size = 1, 
                                       shuffle = False, drop_last = False, **_loader_kwargs(dvc))
              
        # set label name
        if X.shape[1] == dynamic * 33:
            dynamic_size = (dynamic, 33)
            var_name = []
            for i in range(1, 23):
                var_name.append('V_{'+str(i)+'}')
            for i in range(42, 53):
                var_name.append('V_{'+str(i)+'}')
            label = ( labels[1:], var_name )
            plot_text = 'pro'
        else: 
            dynamic_size = (dynamic, 10)
            label = ( labels[1:], 
                     ['C_i','T_i','T_{ci}','C_i^{(s)}','T_i^{(s)}','C^{(s)}','T^{(s)}','Q_c^{(s)}','T_{ci}^{(s)}','T_c^{(s)}'])
            plot_text = 'cnt'
        print('label = ',label)
        if dynamic > 1:
            self.dynamic_size = dynamic_size
        
        # test acc
        model = self.model
        model._save_load('load', 'best')
        model.dvc = torch.device(dvc)
        model = model.to(dvc)
        model.n_category = model.struct[-1]
        model.eval()
        model.batch_size = 32
        model.test(dataset = train_loader)
        model.test(dataset = test_loader)
        model.test(dataset = con_loader)
        model.batch_size = 1
        
        # test attri
        # init
        self.zero_grad_cnt = np.zeros(model.n_category)
        record_result = np.zeros((dynamic_size[-1], model.n_category))
        
        start = time.perf_counter()
        # find baseline
        if self.need_baseline: self._find_baseline()
        
        # register_hook
        self.handles = []
        self._weights_mx = []      # record weight matrix
        self._module_names = []    # record act name
        self.hook_fr_layer()
        if self.manual_cal == False or hasattr(self, '_get_integrated_gradient'):
            self.hook_bk_layer()
        
        # baseline_fp
        if self.need_baseline: 
            self.is_bl_fp = True
            self._get_gradient(self.x_bl, 1)
            self.fp_z_h_bl = self.fp_z_h.copy()
        self.is_bl_fp = False
        
        # run
        for i, (data, target) in enumerate(attri_loader):
            data, target = data.to(model.dvc), target.to(model.dvc)
            class_int = int(target[0].data.cpu().numpy())
            self.sample_index, self.target_class = i, class_int
            
            # info
            msg_str = '{}/{} Calculate attribution in class-{}   '.format(i+1, len(attri_loader), class_int)
            sys.stdout.write('\r'+ msg_str)
            sys.stdout.flush()
            
            # Pass 0-category
            if self.ignore_0 and class_int == 0: continue
        
            # get gradient
            if hasattr(self, '_get_integrated_gradient'):
                self._get_integrated_gradient(data, class_int)
            else:
                self._get_gradient(data, class_int)
            # pred != class_int
            if self.grad_input is None: continue
        
            # get attri variable
            attri_variable = self._get_attri_variable()
            
            record_result[attri_variable, class_int] += 1
        
        end = time.perf_counter()
        print('\nFinish testing attribution, cost {} seconds'.format(int(np.round(end-start,0))))
        
        if self.zero_grad_cnt.sum()>0:
            print('\nThere are a total of {} samples with zero gradients:\n{}'.\
                  format(self.zero_grad_cnt.sum().astype(np.int32), self.zero_grad_cnt[1:].astype(np.int32)))
               
        self.save_data(record_result, label, plot_text)
        self.del_handles()
    
    def test_attri_for_fd(self, datasets, dynamic = 1, labels = None, dvc = 'cpu'):
        model = self.model
        # cov matrix
        if hasattr(model, 'Stat') == False:
            model.fd_signals = None
            with torch.no_grad():
                model.Stat = Statistics(**self.kwargs)
                model.Stat.name, model.Stat.add_info, model.Stat.run_id, model.Stat.save_path = \
                    model.name, model.add_info, model.run_id, model.save_path
                if hasattr(model,'label_name'): model.Stat.label_name = model.label_name
                # offline
                inputs, latents, outputs = model._get_fdi('train')
                if hasattr(model, '_get_customized_fdi') and model.fdi == 'custo':
                    model.fd_signals = model._get_fdi('train', '_get_customized_fdi')
                model.fd_thrd = \
                    model.Stat.offline_modeling(inputs, latents, outputs, model.fd_signals, model.name + model.add_info)

        cov_matrix = model.Stat.cov_matrix
        # A = QΛQ^(-1), A^(-1/2) = QΛ^(-1/2)Q^(-1)
        eigvalues, eigvectors = np.linalg.eig(cov_matrix)
        # neg half
        self.neg_half = eigvectors @ np.diag(1/np.sqrt(eigvalues)) @ eigvectors.T
        # residual-based FI: fdi_scores = (Σ^(-1/2) @ fdi)^2
        
        # data
        _, _, X, Y = datasets
        # set label name
        dynamic_size = (dynamic, 7)
        label = ( labels[1:], 
                 ['C_i','T_i','T_{ci}','C','T','T_c','Q_c'])
        plot_text = 'cnt'
        print('label = ',label)
        if dynamic > 1:
            self.dynamic_size = dynamic_size
        
        start = time.perf_counter()
        # find baseline
        if self.need_baseline: self._find_baseline()
        
        # register_hook
        self.handles = []
        self._weights_mx = []      # record weight matrix
        self._module_names = []    # record act name
        self.hook_fr_layer()       # hook forward
        if self.manual_cal == False or hasattr(self, '_get_integrated_gradient'):
            self.hook_bk_layer()   # hook backward
        
        # baseline activation
        if self.need_baseline: 
            self.is_bl_fp = True
            self._get_gradient(self.x_bl, 1)
            self.fp_z_h_bl = self.fp_z_h.copy()
        self.is_bl_fp = False
        
        # generate_residual
        inputs, latents, outputs = model._get_xyz_for_fd('test')
        res = model.residual_generation(inputs, latents, outputs)

        # labels
        Y = (Y.argmax(axis = 1) != 0).astype(np.int32)
        # split data sets according to fault type
        start = 0
        split_p_list = [start] # 包括头0和尾length-1
        switch_p_list = []     # 记录每个切换点
        test_X = []
        for p in range(Y.shape[0]):
            if p == Y.shape[0] - 1 or (Y[p] == 1 and Y[p+1] == 0):
                split_p_list.append(p+1)
                test_X.append(X[switch_p_list[-1]:split_p_list[-1],:])
                print(test_X[-1].shape)
                start = p+1
            elif Y[p] == 0 and Y[p+1] == 1:
                switch_p_list.append(p+1-start)
        
        # model
        model._save_load('load', 'last')
        model = model.to(dvc)
        model.eval()
        model.batch_size = 1
        # run  
        for f in range(len(test_X)):
            _X = test_X[f]
            for i in range(_X.shape[0]):
                data = _X[i].view(1,-1)

                # info
                if np.mod(i, 10) == 0 or i == _X.shape[0] - 1:
                    msg_str = '{}/{} Calculate attribution of fault-{}'.format(i+1, _X.shape[0], f)
                    sys.stdout.write('\r'+ msg_str)
                    sys.stdout.flush()
            
                # get gradient
                attri_variable = self._get_T2_attr(data)
            
        end = time.perf_counter()
        print('\nFinish testing attribution, cost {} seconds'.format(int(np.round(end-start,0))))
        
        self.save_data()
        self.del_handles()
    
    def _get_T2_attr(self, data):
        # init
        self._init_before_fp()
        self.grad_input = None

        # Input (tensor)
        data = Variable(data, requires_grad = True).to(self.model.dvc)
        output = self.model.forward(data)
        # contribution(component) of r to T2
        res = data - output
        Stat = self.model.Stat
        if Stat.if_minus_mean:
            res -= torch.from_numpy(Stat.res_mean).float()
        T2 = torch.mm(torch.mm(res.view(1,-1), Stat.cov_inverse), res.view(-1,1))
        component = torch.mm(res.view(1,-1),torch.from_numpy(self.neg_half).float()) * T2
        component = component.to(self.model.dvc)
        
        # Zero gradients
        self.model.zero_grad()
        # Baseline only need forward
        if self.is_bl_fp: return
        # Backward
        if self.manual_cal and hasattr(self, '_get_integrated_gradient') == False:
            self._cal_gradient(gradient = component)
        else:
            output.backward(gradient = component)
        # cal attr
        
        return 
    
class Attribution(object):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, **kwargs):     
        
        default = {'model': None,                # 用于归因任务的已训练好的分类模型
                   'datasets': None,             # 数据集
                   'batch_size': 16,             # 批次大小
                   'stacked_size': 1,            # 堆叠的样本尺度
                   'labels': None,               # 类别标签
                   'real_root_cause': None,      # 原因变量标签
                   'judge_rule': 1,              # 评判多时刻堆叠样本的归因的法则
                   'basaline_obj': 'output',     # 选取基线的标准
                   'sensitivity': False,         # 是否是基于敏感绝对值的方法
                   'if_hook': [True, False],     # 勾住前向和反向 [线性操作,激活操作]
                   'if_test_all': False,         # 测试包括训练集在内的所有样本
                   'if_ignore_tp0': True,        # 忽略类别0
                   'if_ignore_wrong': False,     # 忽略预测错误的样本
                   'if_need_baseline': True,     # 是否需要基线
                   'if_full_bp_hook': True,      # 新版的torch更新了bp的hook
                   'if_show_debug_info': [False, False]   # 是否显示debug
                   }
        
        for key in default.keys():
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, default[key])

        # 数据集
        self.model.train_X, self.model.train_Y, self.model.test_X, self.model.test_Y = self.datasets
        train_set = Data.dataset.TensorDataset(torch.from_numpy(self.datasets[0]).float(), 
                                               torch.from_numpy(self.datasets[1]).float())
        self.model.train_loader = Data.DataLoader(train_set, batch_size = self.batch_size,
                                                  shuffle = False, drop_last = False, **_loader_kwargs(self.model.dvc))
        
        # 计算协方差矩阵
        if self.model.task == 'fd':
            self.model._exec_fd(False)
            Stat = self.model.Stat
            if Stat.test_stat == 'T2':
                self._fdi_mean = torch.from_numpy(Stat.t2_fdi_mean).to(self.model.dvc)
                self._cov_inverse = Stat.t2_cov_inverse.to(self.model.dvc)
                cov_matrix = Stat.cov_matrix.data.cpu().numpy()
                # A = QΛQ^(-1), A^(-1/2) = QΛ^(-1/2)Q^(-1)
                eigvalues, eigvectors = np.linalg.eig(cov_matrix)
                # neg half
                self._neg_half = eigvectors @ np.diag(1/np.sqrt(eigvalues)) @ eigvectors.T
                # residual-based FI: fdi_scores = (Σ^(-1/2) @ fdi)^2
            else:
                self._fdi_mean = Stat.spe_fdi_mean.to(self.model.dvc)
            self._thrd = Stat.fd_thrd[0]
        
        # 先找 baseline
        self.x_bl_output = 0
        if self.if_need_baseline:
            self._find_baseline()
            # 保存 baseline 前向 activation
            self.if_hook[0] = True
            
        # register_hook
        self.handles = []
        self._weights_data = []     # record weight matrix
        self._hook_fp()             # hook forward
        self._hook_bp()             # hook backward
        
        # 保存 baseline 前向 activation
        if self.if_need_baseline: 
            self._only_hook_fp = True
            self._cal_contribution(self.x_bl, 1)
        self._only_hook_fp = False
        
        # 对数据集遍历，进行贡献评估与归因
        self._loop_data_set()
        
        # 删除句柄
        self.del_handles()
        
        # 保存结果
        self._save_attri_results()
    
    def del_handles(self):
        for handle in self.handles:
            handle.remove()
    
    def _hook_fp(self):
        def _fp_linear(module, ten_in, ten_out):
            # ten_in: Size([1, x])
            # ten_out: Size([z])
            if self.if_show_debug_info[0]:
                view_info('FP Lineear [{}]{} x'.format(module._index, module._module_name), ten_in) 
                view_info('FP Lineear [{}]{} z(x)'.format(module._index, module._module_name), ten_out) 
            x, z = ten_in[0].data, ten_out[0].data
            module.temp_x = x
            module.temp_z = z.view(1,-1)
            if self._only_hook_fp: module.bl_x = module.temp_x
        
        def _fp_act(module, ten_in, ten_out):
            # ten_in: Size([1, z])
            # ten_out: Size([h])
            if self.if_show_debug_info[0]:
                view_info('FP {} [{}]{} z'.format(module.__class__.__name__, module._index, module._module_name), ten_in) 
                view_info('FP {} [{}]{} h(z)'.format(module.__class__.__name__, module._index, module._module_name), ten_out)
            z, h = ten_in[0].data, ten_out[0].data
            module.temp_z = z
            module.temp_h = h.view(1,-1)
            module._front_linear.temp_h = module.temp_h
            if self._only_hook_fp: module.bl_h = module.temp_h

        print('\nLayers in the model:')
        '''
            self.model.named_modules(): generator([(module_name, module), ...])
            self.model.encoder._modules.items(): odict_items([('0', module), ...])
            self.model.encoder: Sequential([module, ...])
            self.model.encoder[id]: module
        '''
        last_wighting_opt = None
        act_id, linear_id = 0, 0
        for named_module in self.model.named_modules():
            module_name, module = named_module[0], named_module[1]
            act = find_act(module)
            if act is not None:
                print(act_id, '[a] ->', module_name+',', module)
                module._index = act_id
                module._module_name = module_name
                act_id += 1
                module._front_linear = last_wighting_opt
                last_wighting_opt._behind_act_name = act
                if self.if_hook[1]:
                    # _fp_act(self, module, ten_in, ten_out)
                    self.handles.append(module.register_forward_hook(_fp_act))
                    
            if isinstance(module, nn.Linear):
                if linear_id == 0: self._first_layer = module
                print(linear_id, '[l] ->', module_name+',', module)
                module._index = linear_id
                module._module_name = module_name
                linear_id += 1
                self._weights_data.append(module.weight.data.cpu())
                last_wighting_opt = module
                if self.if_hook[0]:
                    # _fp_linear(self, module, ten_in, ten_out)
                    self.handles.append(module.register_forward_hook(_fp_linear))
    
    def _hook_bp(self):
        def _bp_act(module, grad_out, grad_in):
            # grad_out = ∂y/∂z: Size([1, z])
            # grad_in = ∂y/∂h: Size([1, h])
            if self._only_hook_fp: return grad_out
            grad_out, grad_in = list(grad_out), list(grad_in)
            if self.if_show_debug_info[1]:
                view_info('BP {} [{}]{} ∂y/∂h'.format(module.__class__.__name__, module._index, module._module_name), grad_in) 
                view_info('BP {} [{}]{} ∂y/∂z'.format(module.__class__.__name__, module._index, module._module_name), grad_out) 
                
            grad_z, grad_h = grad_out[0], grad_in[0]
            with torch.no_grad():
                grad_z = self._modify_act_grad(module, grad_z, grad_h)
            
            grad_out[0] = grad_z
            return tuple(grad_out)
    
        def _bp_linear(module, grad_out, grad_in):
            # grad_out: Size([1, x]) or Size([b], [1, x], [w])
            # grad_in: Size([1, z])
            if self._only_hook_fp: return grad_out
            grad_out, grad_in = list(grad_out), list(grad_in)
            if self.if_show_debug_info[1]:
                view_info('BP Linear [{}]{} ∂y/∂z'.format(module._index, module._module_name), grad_in) 
                if self.if_full_bp_hook:
                    view_info('BP Linear [{}]{} ∂y/∂x'.format(module._index, module._module_name), grad_out) 
                else:
                    view_info('BP Linear [{}]{} ∂y/∂b, ∂y/∂x, ∂y/∂w'.format(module._index, module._module_name), grad_out) 
            
            if self.if_full_bp_hook: grad_x, grad_z = grad_out[0], grad_in[0]
            else: grad_x, grad_z = grad_out[1], grad_in[0]
            if hasattr(self, '_modify_linear_grad'):
                with torch.no_grad():
                    grad_x = self._modify_linear_grad(module, grad_x, grad_z)
            
            if self._first_layer == module:
                self.grad_x = grad_x
            
            if self.if_full_bp_hook: grad_out[0] = grad_x
            else: grad_out[1] = grad_x
            return tuple(grad_out)
        
        for i, module in enumerate(self.model.modules()):
            act = find_act(module)
            if act is not None:
                if self.if_hook[1] and hasattr(self, '_modify_act_grad'):
                    # _bp_act(self, module, grad_out, grad_in) = grad_out
                    if self.if_full_bp_hook:
                        self.handles.append(module.register_full_backward_hook(_bp_act))
                    else:
                        self.handles.append(module.register_backward_hook(_bp_act))    
            
            if isinstance(module, nn.Linear):
                if self._first_layer == module or \
                    (self.if_hook[0] and hasattr(self, '_modify_linear_grad')):
                    # _bp_linear(self, module, grad_out, grad_in) = grad_out
                    if self.if_full_bp_hook:
                        self.handles.append(module.register_full_backward_hook(_bp_linear))
                    else:
                        self.handles.append(module.register_backward_hook(_bp_linear))
    
    def _loop_data_set(self):
        print('\nLoop test set to calculate attributions...')
        self.model.dvc = 'cpu'
        self.model = self.model.to(self.model.dvc)
        self.model.eval()
        if self.if_test_all: test_X, test_Y = np.concatenate([self.datasets[0], self.datasets[2]]), np.concatenate([self.datasets[1], self.datasets[3]])
        else: test_X, test_Y = self.datasets[2], self.datasets[3]
        test_set = Data.dataset.TensorDataset(torch.from_numpy(test_X).float(), torch.from_numpy(test_Y).float())
        self.model.test_loader = Data.DataLoader(test_set, batch_size = 1,
                                                 shuffle = False, drop_last = False, **_loader_kwargs(self.model.dvc))
        
        self._attri_dict = {}
        N = self.model.test_X.shape[0]
        for batch_idx, (data, target) in enumerate(self.model.test_loader):
            data, target = data.to(self.model.dvc), target.to(self.model.dvc)
            class_int = target.argmax().data.cpu().numpy()
            
            if np.mod(batch_idx + 1, 10) == 0 or batch_idx == N - 1:
                msg_str = 'Calculate attributions: {}/{}, cls = {}'.format(batch_idx + 1, N, class_int)
                sys.stdout.write('\r'+ msg_str + '                                    ')
                sys.stdout.flush()
            
            if self.model.task == 'cls': self._cal_contribution(data, class_int)
            else: self._cal_contribution(data, class_int)
            
            if self.contributions is None: continue
            else: self._get_root_cause()
            
            if str(class_int) not in self._attri_dict.keys(): self._attri_dict[str(class_int)] = []
            self._attri_dict[str(class_int)].append(self._attri.reshape(1,-1))
        
        N_evaluate = 0
        N_right_pred = 0
        FIR = []
        for key in self._attri_dict.keys():
            self._attri_dict[key] = np.concatenate(self._attri_dict[key], axis = 0)
            real_v = self.real_root_cause[int(key)]
            pred_mx = self._attri_dict[key]
            
            if real_v == 'n' or real_v == 'm': continue
            n_cause = 1
            if type(real_v) == tuple: 
                n_cause = len(list(real_v))
                real_v = np.sort( np.array(real_v) )
            # 最后n_cause个为预测的原因变量的索引
            order = np.argsort(pred_mx, axis= 1)[:,-n_cause:]
            if n_cause > 1:
                order = np.sort(order, axis = 1)
            n_right_pred = np.sum((order == real_v).astype(int))
            n_evaluate = pred_mx.shape[0]*n_cause
            N_right_pred += n_right_pred
            N_evaluate += n_evaluate
            FIR.append(n_right_pred/n_evaluate)
        FIR.append(np.round(N_right_pred/N_evaluate,4))
        print('\nThe FIR of {} for {} is (tested {} samples):\n{}'.format(self.name, self.model.name, N_evaluate, FIR))
        
    def _save_attri_results(self):
        # Save excel
        Save_excel(self._attri_dict, self.model.save_path, '[{}] Attribution_results'.format(self.name))
        # Save plot
        
    
    def _cal_contribution(self, 
                          input_data,      # Size: [1, dynamic * v_dim]
                          class_int = 0    # Type: int
                          ):
        # init
        self.contributions = None
        if self.if_ignore_tp0 and class_int == 0: return

        # Input (tensor)
        input_data = Variable(input_data.to(self.model.dvc), requires_grad = True)
        self.input_x = input_data
        # Target for backprop (to onehot)
        if self.model.task == 'cls':
            one_hot_output = Tensor(np.zeros((1, self.model.n_category)), self.model.dvc)
            one_hot_output[0, class_int] = 1

        # Forward
        model_output = self.model.forward(input_data)
        self.model_output = model_output
        
        # no need backward
        if self._only_hook_fp: return
        
        # Filter: pred != real
        if self.if_ignore_wrong:
            if self.model.task == 'cls' and model_output.argmax() != class_int: return 
            if self.model.task == 'fd' and self._get_test_stat(input_data, model_output) < self._thrd: return 
            
        # Zero gradients
        self.model.zero_grad()
        
        # Backward
        ''' 
            y.backward(arg) => x.grad = arg * ∑_yi (∂yi/∂x )T
            loss.backward = y.backward(∂loss/∂y)
        ''' 
        if self.model.task == 'cls':
            model_output.backward(gradient = one_hot_output)
        else:
            ones_output = Tensor(np.ones((1, model_output.size(1))), self.model.dvc)
            model_output.backward(gradient = ones_output)
            
    def _get_root_cause(self):
        input_x = self.input_x.data.cpu().numpy()
        output_x = self.model_output.data.cpu().numpy()
        CP_mx = self.contributions.data.cpu().numpy()
        grad_x = self.grad_x.data.cpu().numpy()
        
        # C: x -> \hat x or y (0,L) * (1,L) = (0,L)
        C_ve = np.sum(CP_mx * (output_x - self.x_bl_output), axis = 1)
        if self.model.task == 'fd':
            # CP: x -> r
            Res_CP_mx = input_x.T * np.eye(input_x.shape[1]) - CP_mx * (output_x - self.x_bl_output)
            Res_CP_mx = Res_CP_mx / np.sum(Res_CP_mx, axis = 0, keepdims = True)
            # C: x -> r
            Res_C_ve = np.sum(Res_CP_mx * (input_x - output_x), axis = 1)
            # a unique layer
            # t2: r -> t2 = ( Σ^{-1/2} r )^2 = (1,L) @ (L,L) = (1,L)
            t2 = ( (input_x - output_x) @ self._neg_half.T )**2
            if self.name not in ['PICP','LCP']:
                ''' LRP, DeepLIFT, ... '''
                # R_CP_mx: r -> t2 = (L,L) * (L,1) = (L,L)
                R_CP_mx = self._neg_half.T * (input_x - output_x).T
                R_CP_mx = R_CP_mx / np.sum(R_CP_mx, axis = 0, keepdims = True)
                
            elif self.name == 'PICP':
                ''' PICP '''
                R_CP_mx = self._cal_picp_contribution(torch.from_numpy(input_x - output_x),
                                                      torch.zeros((1, input_x.shape[1])),
                                                      'Square',
                                                      torch.from_numpy(self._neg_half)
                                                      ).data.cpu().numpy()
            elif self.name == 'LCP':
                ''' LCP '''
                R_CP_mx = self._neg_half.T * (input_x - output_x).T * np.sign(t2)
                R_CP_mx[R_CP_mx < 0] = 0
                R_CP_mx = R_CP_mx / np.sum(R_CP_mx, axis = 0, keepdims = True)
                R_CP_mx[R_CP_mx!=R_CP_mx] = 0
            
            # C: x -> T^2 = (0,L) @ (L,L) * (1,L)
            self._attri = np.sum(Res_CP_mx @ R_CP_mx * t2, axis = 1)
            
            ''' LRP, DeepLIFT, ... '''
            # self._attri = np.abs(self._attri)
            if self.name not in ['PICP','LCP']: self._attri = np.abs(self._attri)
            
            # print('neg_half >>>\n',self._neg_half)
            # print('C_mx >>>\n', CP_mx * output_x)
            # print('Res_CP_mx >>>\n', Res_CP_mx)
            # print('output_x >>>\n', output_x)
            # print('C_ve >>>\n', C_ve)
            # print('Res_C_ve >>>\n', Res_C_ve)
            # print('sum_ve >>>\n', np.sum(Res_CP_mx @ R_CP_mx, axis = 0) )
            # print('Attri >>>\n', self._attri)
            # print('t2 >>>\n', t2)
            # print('T2 >>>', np.sum(t2), self._get_test_stat(self.input_x, self.model_output))
        
        if self.stacked_size > 1:
            if grad_x.abs().max() == 0: 
                self.zero_grad_cnt[self.target_class] += 1 
            if self.need_baseline:
                input_x = input_x - self.x_bl
            if self.sensitivity:
                attri = torch.abs(grad_x)
            else:
                attri = grad_x * input_x
            # judge_rule
            judge_rules = ['Frequency', 'Max', 'Weighted']
            judge_rule = judge_rules[self.judge_rule - 1]
                    
            if hasattr(self, 'dynamic_size') and self.dynamic_size is not None:
                # 2-d
                attri = attri.view(self.dynamic_size[0], self.dynamic_size[1])
                
                # cal_dynamic_attri
                if judge_rule == 'Frequency':
                    attri_for_each_interval = attri.argmax(axis = 1)
                    attri = torch.bincount(attri_for_each_interval)
                elif judge_rule == 'Max':
                    attri = attri.max(axis = 0).values
                elif judge_rule == 'Weighted':
                    attri = attri.sum(axis = 0)
                    
            attri_variable = attri.argmax()
            return attri_variable.data.cpu().numpy()
    
    def _get_test_stat(self, x, output):
        Stat = self.model.Stat
        
        fdi = x - output
        if Stat.if_minus_mean: fdi -= self._fdi_mean
        if Stat.test_stat == 'T2':
            test_stat = fdi.view(1,-1) @ self._cov_inverse @ fdi.view(-1,1)
        else:
            test_stat = torch.sum(fdi**2)
        return test_stat
        
    def _find_baseline(self):
        self.model.eval()
        task = self.model.task

        x_bl = Tensor(np.zeros((1, self.model.struct[0])), self.model.dvc)
        # zero baseline
        if self.basaline_obj == 'zero':
            self.x_bl = x_bl; return
        # load baseline
        elif self.basaline_obj == 'load':
            return
        
        # backward find baseline
        x_bl = Variable(x_bl, requires_grad = True)
        optimizer = RMSprop([x_bl], lr=1e-2, alpha=0.9, eps=1e-10)
        loss_data = np.inf
        epoch = 0
        print('\nSelecting baseline ...')
        while loss_data > 1e-6 and epoch <= 1e2:
            epoch += 1
            self.model.zero_grad()
            optimizer.zero_grad()
            output = self.model.forward(x_bl)
            
            if task == 'cls': loss = torch.sum(output**2)
            else: loss = self._get_test_stat(x_bl, output)
            
            loss_data = loss.data.cpu().numpy()[0,0]
            loss.backward()
            optimizer.step()
            msg = ">>> Epoch: {}, Loss = {:.4f}".format(epoch, loss_data)
            sys.stdout.write('\r'+ msg)
            sys.stdout.flush()
        
        print('\n{}'.format(x_bl.data.cpu().numpy()))
        view_info('baseline', x_bl)
        self.x_bl = x_bl.data
        self.x_bl_output = output.data.cpu().numpy()
        # save baseline
        