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
from .fd_statistics import Statistics
from ..data.load import _loader_kwargs
from ..visual.plot import category_distribution

def view_info(name, v):
    if type(v) != list: v = list(v)
    print('\n>>>  {}: len = {}'.format(name, len(v)))
    for i, var in enumerate(v):
        if var is None:
            print('{}: {}'.format(i, var) )
        else:
            print('{}: size({}), mean({}), min({}), max({})'.format(i, var.size(), 
                                                                    var.mean().data.cpu().numpy(),
                                                                    var.min().data.cpu().numpy(),
                                                                    var.max().data.cpu().numpy()))
    print('<<<')

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
        
        
        
        # data
        _, _, X, Y = datasets
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
    
class Attribution(TestAttri):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, 
                 model,                      # 用于归因任务的已训练好的分类模型
                 judge_rule = 1,             # 评判多时刻堆叠样本的归因的法则
                 ignore_0 = True,            # 忽略类别0
                 ignore_wrong_pre = True,    # 忽略预测错误的样本
                 need_baseline = False,      # 是否需要基线
                 basaline_goal = 'output',   # 选取基线的标准
                 sensitivity = False,        # 是否是基于敏感绝对值的方法
                 hook_fp = [False, True],    # 勾住前向 [线性操作,激活操作]
                 hook_bp = [False, True],    # 勾住反向 [线性操作,激活操作]
                 manual_cal = True,          # 是否启用手算（不用自带的反向传播）
                 show_op_info = False        # 是否显示debug
                 ):     
        
        # 已训练好的分类模型
        self.model = model
        # 进入eval模式
        self.model.eval()
        
        # [fp_l, fp_a]
        self.hook_fp = hook_fp
        # [bp_l, bp_a]
        self.hook_bp = hook_bp
        
        self.judge_rule = judge_rule
        self.show_op_info = show_op_info
        self.ignore_0 = ignore_0
        self.ignore_wrong_pre = ignore_wrong_pre
        self.need_baseline = need_baseline
        self.basaline_goal = basaline_goal
        self.sensitivity = sensitivity
        self.manual_cal = manual_cal
        # baseline 需要记录前向 act
        if need_baseline: 
            self.hook_fp[1] = True
    
    def del_handles(self):
        for handle in self.handles:
            handle.remove()
    
    def hook_fr_layer(self):
        def _fp_act(module, ten_in, ten_out):
            # ten_in: Size([1, z_(l+1)])
            # ten_out: Size([h_(l+1)])
            z, h = ten_in[0].data, ten_out[0].data
            if self.show_op_info:
                view_info('act_ten_in', ten_in) 
                view_info('act_ten_out', ten_out) 
            self.fp_z_h.append((z, h.view(1,-1)))
        
        def _fp_linear(module, ten_in, ten_out):
            # ten_in: Size([1, z_(l+1)])
            # ten_out: Size([h_(l+1)])
            x, z = ten_in[0].data, ten_out[0].data
            if self.show_op_info:
                view_info('act_ten_in', ten_in) 
                view_info('act_ten_out', ten_out) 
            self.fp_x_z.append((x,z))
        
        for i, module in enumerate(self.model.modules()):
            act = find_act(module)
            if act is not None:
                self._module_names.append(module.__class__.__name__)
                if self.hook_fp[1]:
                    # _fp_act(self, module, ten_in, ten_out)
                    self.handles.append(module.register_forward_hook(_fp_act))
                    
            if isinstance(module, nn.Linear):
                self._weights_mx.append(module.weight.data.cpu())
                if self.hook_fp[0]:
                    # _fp_linear(self, module, ten_in, ten_out)
                    self.handles.append(module.register_forward_hook(_fp_linear))
    
    def hook_bk_layer(self):
        def _bp_act(module, grad_out, grad_in):
            # grad_out: Size([1, z_(l+1)])
            # grad_in: Size([1, h_(l+1)])
            if self.is_bl_fp: return grad_out
            grad_out, grad_in = list(grad_out), list(grad_in)
            grad_z, grad_h = grad_out[0], grad_in[0]
            if self.show_op_info:
                view_info('act_grad_out', grad_out) 
                view_info('act_grad_in', grad_in) 
                
            with torch.no_grad():
                grad_z = self._get_grad_z(grad_h)
            
            grad_out[0] = grad_z
            return tuple(grad_out)
    
        def _bp_linear(module, grad_out, grad_in):
            # grad_out: Size([b_(l+1)], [1, x_(l)], [w_(l), w_(l+1)])
            # grad_in: Size([1, z_(l+1)])
            grad_out, grad_in = list(grad_out), list(grad_in)
            grad_x, grad_z = grad_out[1], grad_in[0]
            if self.show_op_info:
                view_info('linear_grad_out', grad_out) 
                view_info('linear_grad_in', grad_in) 
            
            if hasattr(self, '_get_grad_x'):
                with torch.no_grad():
                    grad_x = self._get_grad_x(grad_z)
            
            if self.first_layer == module:
                self.grad_input = grad_x
            
            grad_out[1] = grad_x
            return tuple(grad_out)
        
        first_one = True
        self.first_layer = None
        for i, module in enumerate(self.model.modules()):
            act = find_act(module)
            if act is not None:
                if self.hook_bp[1] and hasattr(self, '_get_grad_z'):
                    # _bp_act(self, module, grad_out, grad_in) = grad_out
                    self.handles.append(module.register_backward_hook(_bp_act))
            
            if isinstance(module, nn.Linear):
                if first_one:
                    first_one = False
                    self.first_layer = module
                    # record the result of the 1st layer
                    self.handles.append(module.register_backward_hook(_bp_linear)) 
                elif self.hook_bp[0] and hasattr(self, '_get_grad_x'):
                    # _bp_linear(self, module, grad_out, grad_in) = grad_out
                    self.handles.append(module.register_backward_hook(_bp_linear))
    
    def _init_before_fp(self):
        # fp_linear
        if self.hook_fp[0]:
            self.fp_x_z = []
            self.bp_linear_cnt = 1

        # fp_act
        if self.hook_fp[1]:
            self.fp_z_h = []
            self.bp_act_cnt = 1
      
    def _get_gradient(self, 
                      input_data,   # Size: [1, dynamic * v_dim]
                      class_int     # Type: int
                      ):
        # init
        self._init_before_fp()
        self.grad_input = None

        # Input (tensor)
        input_data = Variable(input_data, requires_grad = True).to(self.model.dvc)
        self.input_data = input_data
        # Target for backprop (to onehot)
        one_hot_output = torch.FloatTensor(1, self.model.n_category).zero_()
        one_hot_output[0, class_int] = 1
        one_hot_output = one_hot_output.to(self.model.dvc)
        self.one_hot_output = one_hot_output
        ''' 
            Backward pass
            y.backward(arg) => x.grad = arg * ∑_yi (∂yi/∂x )T
            loss.backward = y.backward(∂loss/∂y)
        ''' 
        # Forward
        model_output = self.model.forward(input_data)
        
        # Filter: pred != real
        if self.ignore_wrong_pre and model_output.argmax() != one_hot_output.argmax(): 
            return 

        # Zero gradients
        self.model.zero_grad()
        
        if self.is_bl_fp: return
        # Backward
        if self.manual_cal and hasattr(self, '_get_integrated_gradient') == False:
            self._cal_gradient(gradient = one_hot_output)
        else:
            model_output.backward(gradient = one_hot_output)
        # Get self.grad_input in module.register_backward_hook(self._bp_linear)
    
    def _cal_gradient(self, component):
        grad_h = component
        with torch.no_grad():
            for i in range(len(self._weights_mx)):
                grad_z = self._get_grad_z(grad_h)
                W = self._weights_mx[-(i+1)].to(self.model.dvc)
                grad_h = torch.mm(grad_z, W)
                # print(grad_z.mean(), W.mean(), grad_h.mean())
        self.grad_input = grad_h
        
    def _get_attri_variable(self):
        grad_input, input_data = self.grad_input, self.input_data
        if grad_input.abs().max() == 0: 
            self.zero_grad_cnt[self.target_class] += 1 
        
        if self.need_baseline:
            input_data = input_data - self.x_bl
        
        if self.sensitivity:
            attri = torch.abs(grad_input)
        else:
            attri = grad_input * input_data
        
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
    
    def _find_baseline(self):
        if hasattr(self, 'dynamic_size'):
            x_bl = torch.FloatTensor(1, self.dynamic_size[0]*self.dynamic_size[1]).zero_()
        else:
            x_bl = torch.FloatTensor(1, self.model.struct[0]).zero_()
        if self.basaline_goal == 'input':
            self.x_bl = x_bl; return
        # x_bl = torch.randn(1, self.dynamic_size[0]*self.dynamic_size[1])
        zero = torch.FloatTensor(1, self.model.n_category).zero_()
        x_bl = Variable(x_bl, requires_grad = True).to(self.model.dvc)
        optimizer = RMSprop([x_bl], lr=1e-2, alpha=0.9, eps=1e-10)
        loss_data = 1.0
        epoch = 0
        print('\nFind baseline ...')
        while loss_data > 1e-6 and epoch <= 5e3:
            epoch += 1
            optimizer.zero_grad()
            output = self.model.forward(x_bl)
            loss = torch.sqrt( torch.mean( (output )**2) )
            loss_data = loss.data.cpu().numpy()
            loss.backward()
            optimizer.step()
            msg = " | Epoch: {}, Loss = {:.4f}".format(epoch, loss_data)
            sys.stdout.write('\r'+ msg)
            sys.stdout.flush()
        view_info('baseline', x_bl)
        print('{:.4f}'.format(output))
        self.x_bl = x_bl.data