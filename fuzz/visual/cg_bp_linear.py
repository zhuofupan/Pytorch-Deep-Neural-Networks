"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import torch
import matplotlib.cm as mpl_color_map
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
import sys
sys.path.append('..')
from core.func import find_act
from _test.bp_cls_sae import get_sae_model
from visual.plot import category_distribution

class RunData():
    def save_img(self, img, index, file_name_to_export, pic):
        # save_img
        if not os.path.exists('../results/' + file_name_to_export + '/'): os.makedirs('../results/' +  file_name_to_export + '/')
        file_name_to_export = file_name_to_export + '/{}-'.format(index) + self.model.name
        # img = np.abs(img)
        if pic in ['gray', 'all']:
            # Convert to grayscale
            grayscale_guided_grads = convert_to_grayscale(img)
            # Save grayscale gradients
            save_gradient_images(grayscale_guided_grads, file_name_to_export + '_Guided_BP_gray')
        if pic in ['pos_neg', 'all']:
            # Positive and negative saliency maps
            if img.max() > 0:
                pos_sal = (np.maximum(0, img) / img.max())
                save_gradient_images(pos_sal, file_name_to_export + '_pos_sal')
            if -img.min() > 0:
                neg_sal = (np.maximum(0, -img) / -img.min())
                save_gradient_images(neg_sal, file_name_to_export + '_neg_sal')
        if pic in ['color', 'all']:
            if img.shape[0] == 1:
                save_gradient_images(img, file_name_to_export + '_Guided_BP_color[origin]')
                img = img.reshape((self.reshape[0], self.reshape[1]))
                img = (img - np.min(img)) / (np.max(img) - np.min(img))  # Normalize between 0-1
                # Get colormap
                color_map = mpl_color_map.get_cmap('hsv')
                img = color_map(img)
                img = img.transpose((2,0,1))[:3,:,:]
            # Save colored gradients
            save_gradient_images(img, file_name_to_export + '_Guided_BP_color')

    def save_data(self, X, Y, label, text):
        X, Y = X.reshape((-1,)), Y.reshape((-1,))
        n, m = self.reshape[1], self.model.n_category - 1
        array = np.zeros((n,m))
        for i in range(X.shape[0]):
            c, v = Y[i] - 1, X[i]
            if self.show_info: print('\n variable = {}, class = {}'.format(v,c))
            array[v][c] += 1
        print()
        if array.shape[1] == 10:
            real = [5,5,5,3,4,8,5,6,9,7]
            print(np.round(np.diag(array[real])/array.sum(axis = 0)*100, 2))
        if self.show_info == False:
            category_distribution(array, label = label, name = self.model.name, 
                                  text = text, diag_cl = False)
        self.result = array
    
    def test_multi_sample(self, X, Y, n_sampling = 0, 
                          save_type = 'img', pic = 'all', 
                          label_name = None,
                          text = 'cnt',
                          single = None):
        Y = np.argmax(Y, axis = 1)
        # choose sample
        if single is not None:
            n_sampling = 1
            index = [single]
            self.show_info = True
        elif n_sampling > 0:
            index = np.random.choice(X.shape[0], n_sampling, replace = False)
        else:
            n_sampling = X.shape[0]
            index = range(n_sampling)
            
        # loop sample
        data_x, data_y = [], []
        self.save_cnt = 0
        self.zero_grad_index = []
        for i in index:
            if self.dbug: return
            self.sample_index = i
            self.save_cnt += 1
            # info
            msg_str = '{}/{} Test sample index {}'.format(self.save_cnt,n_sampling,i+1)
            sys.stdout.write('\r'+ msg_str)
            sys.stdout.flush()
            # take x, y
            x, y = X[i], Y[i]
            # get grad_x
            guided_grads = self.generate_gradients(x, y)
            if guided_grads is None: continue
            # get visualization
            data_y.append(y.reshape(1,-1))
            if save_type == 'img':
                try:
                    self.save_img(guided_grads, i, str(y), pic)
                except TypeError:
                    pass
            if save_type == 'data':
                data_x.append(guided_grads.reshape((1,-1)))
        
        if len(self.zero_grad_index)>0:
            print('\nThere are a total of {} samples with zero gradients:\n'.\
                  format(len(self.zero_grad_index)),self.zero_grad_index)
        
        print('\nIgnore {} samples, {} samples tested.'.format(n_sampling - len(data_x), len(data_x)))
        
        if save_type != 'img' and single is None:
            data_x = np.concatenate(data_x, axis = 0) # cause variable
            data_y = np.concatenate(data_y, axis = 0) # class
            self.save_data(data_x, data_y, label_name, text)
        
        self.del_handles()

def value_info(name, x):
    print('\n{}: '.format(name), x.size(), x.abs().argmax(), '\n{}'.format(x))

def grad_info(module, grad_in, grad_out):
    print()
    print(module)
    name_in, name_out = 'in', 'out'
     
    for grad in list(grad_in):
        if grad is not None and grad.size(0) == 1:
            print('>>> grad_'+name_in+':', grad.size(), grad.min(), grad.max())

    for grad in list(grad_out):
        if grad is not None:
            print('<<< grad_'+name_out+':', grad.size(), grad.min(), grad.max())
            
    if hasattr(module, 'weight'):
        weight, bias = module.weight.data, module.bias.data
        print('--- weight:', weight.size(), weight.min(), weight.max())
        print('--- bias:', bias.size(), bias.min(), bias.max())        
        
def get_grad_f(name, z, h):
    z, h = z.view(-1), h.view(-1)
    e = 1e-6
    if name == 'Gaussian':
        grad_f = torch.clamp( 2*z*torch.exp(-z*z), min =e)
        grad_f = (grad_f.abs() * z.sign()).diag()
    elif name == 'Affine':
        grad_f = (torch.ones_like(z)).diag()
    elif name == 'Sigmoid':
        grad_f = torch.clamp( (h*(1-h)).diag(), min =e)
    elif name == 'Tanh':
        grad_f = torch.clamp( (1-h*h).diag(), min =e)
    elif name == 'ReLU':
        grad_f = torch.clamp(z, min = 0)
        grad_f[grad_f>0] = 1
        grad_f = (grad_f).diag()
    elif name == 'Softmax':
        D = h.diag()
        h = h.view(1,-1)
        Y = torch.mm(h.t(), h)
        grad_f = (D - Y)
    return grad_f
    
class IC_BP(RunData):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, reshape = None, show_info = False, standard = 1):
        self.model = model
        self.model.eval()
        
        self.reshape = reshape
        self.show_info = show_info
        self.standard = 2
        self.test_category = 9
        self.test_variable = 9
        self.dbug = False
        
        self.handles = []
        self.hook_act_layer()
        self.hook_linear_layer()
    
    def _get_v(self, grad_x):
        if grad_x.abs().max() == 0: 
            self.zero_grad_index.append((self.sample_index, self.target_class)) 
        
        grad_x = grad_x.view(self.reshape[0], self.reshape[1])
        x = self.input_image.view(self.reshape[0], self.reshape[1])
        # -------------------------- judge --------------------------#
        methods = ['bincount', 'weighted', 'abs_max']
        method = methods[self.standard]
        
        if method == 'bincount':
            if self.reshape[0] == 1:
                v = grad_x.argmax()
            else:
                grad_x = grad_x.argmax(axis = 1)
                v = torch.bincount(grad_x).argmax()
        elif method == 'weighted':
            if self.reshape[0] == 1:
                v = (grad_x * x).max()
            else:
                grad_x = (grad_x * x).sum(axis = 0)
                v = grad_x.argmax()
        elif method == 'abs_max':
            if self.reshape[0] == 1:
                v = grad_x.abs().argmax()
            else:
                grad_x = grad_x.abs().max(axis = 0).values
                v = grad_x.argmax()
        # -------------------------- judge --------------------------#
        return v
    
    def hook_linear_layer(self):
        def _backward_linear_function(module, grad_out, grad_in):
            grad_out, grad_in = list(grad_out), list(grad_in)
            grad_x, grad_z = grad_out[1], grad_in[0]
            (x, z) = self.linear[-1]
            (_, h) = self.z_h[-1]
            w, b = module.weight.data, module.bias.data
            # print(torch.mm(grad_z, w) - grad_x)
            
            # -------------------------- grad_x --------------------------#
            with torch.no_grad():
                # grad_x = torch.mm(grad_z*z.sign(), w)
                
                x, z = x.view(-1,1), z.view(1,-1)
                h = h.view(1,-1)
                z[z == 0] = 1
                mat = x * w.t() * h / z
                # print('nan in x * w * h:',torch.isnan(x * w.t() * h).int().sum())
                sum = mat.sum(axis = 0)
                sum[sum == 0] = 1
                print('sum.size():', sum.size() )
                print('0 in z:', z[z == 0].sum(), z)
                print('sum:',sum)
                print('nan in mat:',torch.isnan(mat).int().sum(), torch.isnan(mat))
                mat = mat / sum

                # print('nan in mat:',torch.isnan(mat).int().sum(), torch.isnan(mat).size())
                # grad_z = torch.clamp(grad_z, min=0.0)
                grad_z = grad_z / grad_z.sum()

                # mat_plus = torch.cat( (mat, b.view(1,-1)), axis = 0)
                # dnm = mat_plus.abs().max(axis = 0).values
                # if self.show_info:
                    # value_info('z',z)
                    # value_info('z-b', z-b)
                    # value_info('grad_z * w', grad_x)
                # dnm = mat.abs().sum(axis = 0) + b.abs()
                # mat = mat / dnm

                # mat = torch.clamp(mat, min=0.0)
                
                grad_x = torch.mm(grad_z, mat.t())
                if self.show_info:
                    value_info('x',grad_x)
                    print('\nisnan\n{}'.format(torch.isnan(grad_x)))
                    print('\nsum\n{}'.format(torch.isnan(grad_x).int().sum()))
                
                # x, z = x.view(-1,1), z.view(1, -1)
                # mat = torch.mm(x.sign(), z.sign())*w.t().sign()
                # # mat = torch.clamp(mat, min=0.0)
                # grad_x = torch.mm(grad_z, w*mat.t())
            # -------------------------- grad_x --------------------------#
                
            del self.linear[-1]
            del self.z_h[-1]
            grad_out[1] = grad_x
            if module == self.model.first_layer:
                self.gradients = self._get_v(grad_x)
                if self.show_info:
                    self.dbug = True
                    value_info('grad_x',grad_x)
                    value_info('grad_x*x',grad_x*x.view(1,-1))
                    print(self.gradients)
                    
            return tuple(grad_out)
            
        def _forward_linear_function(module, ten_in, ten_out):
            ten_in, ten_out = list(ten_in), list(ten_out)
            self.linear.append((ten_in[0], ten_out[0]))
            
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(_forward_linear_function))
                self.handles.append(module.register_backward_hook(_backward_linear_function))
    
    def hook_act_layer(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """      
        def _backward_act_function(module, grad_out, grad_in):
            """
            If there is a negative gradient, change it to zero
            backward_hook(module, grad_input, grad_output) -> Tensor or None
            """
            # if self.show_info: grad_info(module, grad_out, grad_in)
            grad_out, grad_in = list(grad_out), list(grad_in)
            grad_z, grad_h = grad_out[0], grad_in[0]
            grad_out[0] = grad_h.data
            return tuple(grad_out)
            
            (z, h) = self.z_h[-1]
            
            with torch.no_grad():
                if self.show_info:
                    value_info('grad_x = grad_h',grad_h)
                
                # grad_f = get_grad_f(module.__class__.__name__, z, h)
                # -------------------------- grad_z --------------------------#
                grad_h = torch.clamp(grad_h, min=0.0)
                # grad_h[grad_h != grad_h.max()] = 0
                grad_z = grad_h * h * z.sign()
                
                # grad_z = grad_z / grad_z.abs().sum()
                
                if self.show_info:
                    value_info('grad_z',grad_z)
                
                # grad_z = torch.mm(grad_h,  grad_f)
                # grad_z = torch.mm(grad_h, grad_f.sign()) * h
                # -------------------------- grad_z --------------------------#
            
            del self.z_h[-1]
            grad_out[0] = grad_z
            return tuple(grad_out)
        '''
        ########################################################################################
        '''
        
        def _forward_act_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            forward_hook(module, input, output) -> None
            """
            if module == self.model._output[-1]:
                ten_out[0] = self.one_hot_output
            self.z_h.append((ten_in[0], ten_out[0]))
             
        for i, module in enumerate(self.model.modules()):
            act = find_act(module)
            if act is not None:
                self.handles.append(module.register_forward_hook(_forward_act_function))
                self.handles.append(module.register_backward_hook(_backward_act_function))

    def generate_gradients(self, input_image, target_class):
        self.linear = []
        self.z_h = []
        self.gradients = None
        if target_class == 0: return None
        # Input (tensor)
        if type(input_image) == np.ndarray:
            input_image = torch.from_numpy(input_image).float()
            input_image.unsqueeze_(0)
            input_image = Variable(input_image, requires_grad = True)
        input_image = input_image.to(self.model.dvc)
        self.input_image = input_image
        # Target for backprop (to onehot)
        one_hot_output = torch.FloatTensor(1, self.model.n_category).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output = one_hot_output.to(self.model.dvc)
        self.target_class = target_class
        self.one_hot_output = one_hot_output
        ''' 
            Backward pass
            y.backward(arg) => x.grad = arg * ∑_yi (∂yi/∂x )T
            loss.backward = y.backward(∂loss/∂y)
        ''' 
        # Forward pass
        model_output = self.model.forward(input_image)
        # real = 0 or pred != real
        if model_output.argmax() != target_class: return None
        elif self.show_info: value_info('real_y', model_output)
        # Zero gradients (把权值梯度变为0)
        self.model.zero_grad()
        model_output.backward(gradient = one_hot_output)
        
        # Convert Pytorch variable to numpy array
        gradients_as_arr = self.gradients.data.cpu().numpy()
        return gradients_as_arr
    
    def del_handles(self):
        for handle in self.handles:
            handle.remove()
            
def image_visualization(index = 1, dvc = 'cpu'):
    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(index)

    print(pretrained_model)
    pretrained_model.dvc = torch.device(dvc)
    pretrained_model = pretrained_model.to(dvc)
    pretrained_model.name = 'AlexNet'
    pretrained_model.n_category = 1000
    pretrained_model.first_layer = list(pretrained_model.features._modules.items())[0][1]
    ic_bp = IC_BP(pretrained_model, show_info = True)
    # Get gradients
    guided_grads = ic_bp.generate_gradients(prep_img, target_class)
    ic_bp.save_img(guided_grads, target_class, file_name_to_export, 'all')
    print('Guided backprop completed')            
    
def cause_tracing(index = 1, dynamic = 1, standard = 1, dvc = 'cpu'):
    model, datasets, labels = get_sae_model(index, dynamic = dynamic)
    model._save_load('load', 'best')
    model.dvc = torch.device(dvc)
    model = model.to(dvc)
    model.load_data(datasets, 32)
    model.test(0, 'train')
    # Guided backprop
    if index == 1: 
        reshape = (dynamic,33)
        _x = []
        for i in range(33):
            _x.append('X_{'+str(i+1)+'}')
        label = ( labels[1:], _x )
        text = 'pro'
    else: 
        reshape = (dynamic,10)
        label = ( labels[1:], 
                 ['Ci^{(in)}','Ti^{(in)}','Tci^{(in)}','Ci','Ti','C','T','Qc','Tci','Tc'])
        text = 'cnt'
    
    print('label = ',label)
    if dynamic == 1:
        model.first_layer = model._feature[0]
    else:
        model.first_layer = model._feature[1]
    model.n_category = model.struct[-1]
    ic_bp = IC_BP(model, reshape, standard = standard)
    X = np.concatenate((datasets[0], datasets[2]), axis = 0)
    Y = np.concatenate((datasets[1], datasets[3]), axis = 0)
    ic_bp.test_multi_sample(X, Y, 0, 'data', 
                            # single = None, 
                            single = 9832, 
                            label_name = label, text = text)
    return ic_bp


if __name__ == '__main__':
    ic_bp = cause_tracing(1, 40, 0)
    # image_visualization(0)
    