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

from fuzz.visual.plot import category_distribution
from fuzz.core.func import find_act
from fuzz._test.bp_cls_sae import get_sae_model

def div(a, b):
    if type(a) == np.ndarray:
        if a.shape != b.shape:
            b = np.repeat(1,a.shape[1]) 
    elif type(a) == torch.Tensor:
        if a.size() != b.size():
            b = b.repeat(1,a.size(1))
    else:
        if b == 0: return 0
        else: return a/b
            
    b[b == 0] = 1
    a = a/b
    a[b == 0] = 0
    return a

def d_axis(x, axis = 0, opt = 'sum'):
    _x = x
    if 'abs' in opt:
        _x = _x.abs()
        
    if 'max' in opt:
        _x = _x.max(axis = axis).values
    elif 'sum' in opt:
        _x = _x.sum(axis = axis)
        
    if len(x.size()) == 1:
        if _x == 0: _x = 1
        return x/_x
    
    if axis == 1: _x = _x.view(-1,1)
    return div(x, _x)

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

    def save_data(self, label, text):
        n, m = self.reshape[1], self.model.n_category - 1
        self.V = np.concatenate(self.V, axis = 0)
        self.result = []
        for s in range(self.V.shape[1]):
            print('\nstandard = {}:'.format(self.judge[s]))
            array = np.zeros((n,m))
            for i in range(self.V.shape[0]):
                v, c = self.V[i,s], self.Y[i]
                array[v,c] += 1
            
            if m == 10:
                real = [5,5,5,3,4,8,5,6,9,7]
                print(np.round(div(np.diag(array[real]), array.sum(axis = 0))*100, 2))
                print(np.round(div(np.diag(array[real])[3:].sum(), array[:,3:].sum())*100, 2))
            if self.show_info == False:
                category_distribution(array, label = label, name = self.model.name, info = '-s' + str(s+1), 
                                      text = text, diag_cl = False)
            self.result.append(array)
    
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
        self.V, self.Y = [], []
        self.save_cnt = 0
        self.zero_grad_index = []
        for i in index:
            self.sample_index = i
            self.save_cnt += 1
            # info
            msg_str = '{}/{} Test sample index {}'.format(self.save_cnt,n_sampling,i)
            sys.stdout.write('\r'+ msg_str)
            sys.stdout.flush()
            # take x, y
            x, y = X[i], Y[i]
            # get grad_x
            v = self.generate_gradients(x, y)
            if v is None: continue
            # get visualization
            if save_type == 'img':
                try:
                    self.save_img(v, i, str(y), pic)
                except TypeError:
                    pass
            if save_type == 'data':
                self.V.append(v)
                self.Y.append(y-1)
        
        print('\nIgnore {} samples ({} zero gradient samples), {} samples tested.'.format(
            n_sampling - len(self.V), len(self.zero_grad_index), len(self.V)))
        
        if save_type != 'img' and single is None:
            self.save_data(label_name, text)
            gxs = np.sign(self.gxs_cnt[2] - self.gxs_cnt[0])
            np.savetxt('../save/'+self.model.name+'/gxs.csv', gxs, delimiter=',')
        
        self.del_handles()

def value_info(name, x, mode = 1):
    if mode == 1:
        print('\n{}: '.format(name), x.size(), x.abs().argmax(), '\n{}'.format(x))
    elif mode == 2:
        print('\n{}: '.format(name), x.size(), x.min(), x.max(), x.mean())

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
        
def get_grad_f(name, z, h, dim = 1):
    z, h = z.view(-1), h.view(-1)
    e = 1e-6
    if name == 'Softmax':
        # hj = zj/∑z
        # ∂hj/∂zi = hj*((i==j?) - hi)
        # 对角线恒大于0, 非对角线恒小于0
        D = h.diag()
        h = h.view(1,-1)
        Y = torch.mm(h.t(), h)
        grad_f = (D - Y)
        
        if dim == 1: return grad_f.diag()
        else: return grad_f
        
    elif name == 'Gaussian':
        grad_f = 2*z*torch.exp(-z*z)#/7
        grad_f = torch.clamp(grad_f.abs(), min =e) * grad_f.sign()
    elif name == 'Affine':
        grad_f = torch.ones_like(z)
    elif name == 'Sigmoid':
        grad_f = torch.clamp( (h*(1-h)), min =e)#/2
    elif name == 'Tanh':
        # tanh(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
        grad_f = torch.clamp( (1-h*h), min =e)#/4
    elif name == 'ReLU':
        grad_f = torch.clamp(z, min = 0).sign()
        
    if dim == 1: return grad_f
    else: return grad_f.diag()
    
class IC_BP(RunData):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, reshape = None, show_info = False):
        self.model = model
        self.model.eval()
        
        self.reshape = reshape
        self.show_info = show_info
        
        self.handles = []
        # self._pinv()
        self.hook_fr_layer()
        self.hook_bk_layer()
    
    def _pinv(self):
        self.w_pinv = []
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Linear):
                w = module.weight.data
                if module.bias is not None:
                    b = module.bias.data.view(-1,1)
                    A = torch.cat((w, b), axis = 1).cpu().numpy()
                    B = np.linalg.pinv(A)[:-1]
                else:
                    B = np.linalg.pinv(w)
                
                self.w_pinv.append(torch.from_numpy(B).to(self.model.dvc))
    
    def _get_v(self, grad_x):
        if self.reshape[0] > 1:
            grad_x = grad_x.view(self.reshape[0], self.reshape[1])
            x = self.input_image.view(self.reshape[0], self.reshape[1])
        else:
            grad_x = grad_x[0]
            x = self.input_image[0]
        # -------------------------- judge --------------------------#
        grad = []
        clamp_gx = torch.clamp(grad_x, min =0.0)
        if self.reshape[0] == 1:
            self.judge = ['gx*x', 'clamp(gx*x)', '(gx*x).abs']
            # s1
            grad.append( grad_x )
            # s2 <-
            grad.append( clamp_gx )
            # s3
            grad.append( grad_x.abs() )
        else:
            self.judge = ['bincount', 'clamp(bincount)', '∑(gx*x)', '∑clamp(gx*x)', 
                          '(gx*x).max', 'clamp(gx*x).max']
            # s1
            _grad_x = grad_x.argmax(axis = 1)
            grad.append( torch.bincount(_grad_x) )
            # s2
            _grad_x = clamp_gx.argmax(axis = 1)
            grad.append( torch.bincount(_grad_x) )
            # s3
            _grad_x = grad_x.sum(axis = 0)
            grad.append( _grad_x )
            # s4 <-
            _grad_x = clamp_gx.sum(axis = 0)
            grad.append( _grad_x )
            # s5
            _grad_x = grad_x.max(axis = 0).values
            grad.append( _grad_x )
            # s6
            _grad_x = clamp_gx.max(axis = 0).values
            grad.append( _grad_x )
        # -------------------------- judge --------------------------#
        v = []
        for g in grad:
            v.append( g.argmax().data.cpu().numpy() )
        v = np.array(v).reshape(1,-1)
        return  v
    
    def hook_fr_layer(self):
        def _forward_linear_function(module, ten_in, ten_out):
            x, z = list(ten_in)[0].data, list(ten_out)[0].data
            self.x_z.append((x, z))
            
            # w = module.weight.data
            # b = module.bias.data
            # self.alf = div(x.view(1,-1) * w, z.view(-1,1))
            # # self.alf = torch.clamp(self.alf, min = 0.0)
            # self.alf = div(self.alf, (self.alf.abs().sum(axis = 1) + b.abs()).view(-1,1))
        
        def _forward_act_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            forward_hook(module, input, output) -> None
            """
            z, h = list(ten_in)[0].data, list(ten_out)[0].data
            self.z_h.append((z, h))
            
            # if module not in self.model._output:
            #     self.coe.append(self.alf )
            # else:
            #     # --------------------------- last layer ------------------------#
            #     self.coe.append(self.alf * self.one_hot_output.view(-1,1))
            #     con = self.coe[0]
            #     for i in range(len(self.coe)-1):
            #         con = torch.mm(self.coe[i+1], con)
            #     con = con[self.target_class]
            #     self.gradients = self._get_v(con)
            #     # --------------------------- last layer ------------------------#    
        
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(_forward_linear_function))
            act = find_act(module)
            if act is not None:
                self.handles.append(module.register_forward_hook(_forward_act_function))     
    
    def hook_bk_layer(self):
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
            grad_z, grad_h = grad_out[0].data, grad_in[0].data
            (z, h) = self.z_h[-1]

            grad_f = get_grad_f(module.__class__.__name__, z, h, 2)
            
            # -------------------------- grad_z --------------------------#
            grad_z = torch.clamp(grad_h * h, min=0.0)
            if module.__class__.__name__ ==  'ReLU':
                grad_z = grad_z * h.sign()
            elif module.__class__.__name__ not in  ['Sigmoid','Softmax']:
                grad_z = grad_z * z.sign()
            # else:
            #     grad_z = torch.mm(grad_z, grad_f)
            
            # if module.__class__.__name__ in  ['Relu','Affine']:
            #     grad_z = div(grad_z, z.abs())

            # grad_z = grad_h * grad_f #* z.sign() * h.abs()
            
            # if module.__class__.__name__ not in ['Affine','ReLU','Softmax']:
            #     grad_z *= z.sign() #(z.sign()*h.sign()*grad_f.sign()).sign() #* (h.abs() < 0.25).float()
            # else:
            #     grad_z *= grad_f
            # -------------------------- grad_z --------------------------#
            
            grad_out[0] = grad_z
            return tuple(grad_out)
        
        def _backward_linear_function(module, grad_out, grad_in):
            grad_out, grad_in = list(grad_out), list(grad_in)
            if module.bias is None: index = 0
            else: index = 1
            
            grad_x, grad_z = grad_out[index].data, grad_in[0].data
            (x, z) = self.x_z[-1]
            (_, h) = self.z_h[-1]
            w = module.weight.data
            if module.bias is not None:
                b = module.bias.data
            # print(torch.mm(grad_z, w) - grad_x)
            
            # -------------------------- grad_x --------------------------#
            with torch.no_grad():
                x, z, h = x.view(1,-1), z.view(-1,1), h.view(1,-1)
                # w = torch.clamp(x * w * z, min = 0.0)
                
                # if module == self.model.first_layer:
                 
                # w = w * x#.sign()

                grad_x = torch.mm(grad_z, w)
                if self.show_info:
                    value_info('grad_x',grad_x)
                    
            # -------------------------- grad_x --------------------------#
                
            del self.x_z[-1]
            del self.z_h[-1]
            grad_out[index] = grad_x
            
            if module == self.model.first_layer:
                grad_x = grad_x * x
                gxs = (grad_x.sign() + 1)[0]
                xs = (self.input_image.sign() + 1)[0]
                for i in range(self.reshape[1]):
                    gs, s = gxs[i], xs[i]
                    self.gxs_cnt[int(gs), i, self.target_class] += 1
                    self.xs_cnt[int(s), i, self.target_class] += 1
                
                # --------------------------- 1st layer ------------------------#
                if grad_x.abs().max() == 0: 
                    self.zero_grad_index.append((self.sample_index, self.target_class))
                    self.gradients =  None
                else:
                    self.gradients = self._get_v(grad_x)
                # --------------------------- 1st layer ------------------------#
                    
            return tuple(grad_out)
             
        for i, module in enumerate(self.model.modules()):
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_backward_hook(_backward_linear_function))
            act = find_act(module)
            if act is not None:
                self.handles.append(module.register_backward_hook(_backward_act_function))

    def generate_gradients(self, input_image, target_class):
        self.coe = []
        self.p_index = 0
        self.x_z = []
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
        one_hot_output[0,target_class] = 1
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
        return self.gradients
    
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
    
def cause_tracing(index = 1, dynamic = 1, dvc = 'cpu', ae_type = 'AE'):
    model, datasets, labels = get_sae_model(index, dynamic = dynamic, ae_type = ae_type)
    model._save_load('load', 'best')
    model.dvc = torch.device(dvc)
    model = model.to(dvc)
    model.load_data(datasets, 16)
    model.test(0, 'train')
    model.test(0, 'test')
    X = np.concatenate((datasets[0], datasets[2]), axis = 0)
    Y = np.concatenate((datasets[1], datasets[3]), axis = 0)
    import torch.utils.data as Data
    data_set = Data.dataset.TensorDataset(torch.from_numpy(X).float(), 
                                          torch.from_numpy(Y).float())
    data_loader = Data.DataLoader(data_set, batch_size = 16, 
                                  shuffle = False, drop_last = False, pin_memory = False)
    model.test(0, data_loader)
    # Guided backprop
    # TE
    if index == 0: 
        reshape = (dynamic,33)
        _x = []
        for i in range(22):
            _x.append('V_{'+str(i+1)+'}')
        for i in range(41,52):
            _x.append('V_{'+str(i+1)+'}')
        label = ( labels[1:], _x )
        text = 'pro'
    # CSTR
    elif index < 10: 
        reshape = (dynamic,10)
        label = ( labels[1:], 
                 ['C_i','T_i','T_{ci}','C_i^{(s)}','T_i^{(s)}','C^{(s)}','T^{(s)}',
                  'Q_c^{(s)}','T_{ci}^{(s)}','T_c^{(s)}'])
        text = 'cnt'
    # HY
    elif index == 10: 
        reshape = (dynamic,61)
        _x = []
        for i in range(61):
            _x.append('V_{'+str(i+1)+'}')
        label = ( labels[1:], _x)
        text = 'pro'
    print('label = ',label)
    if dynamic == 1:
        model.first_layer = model._feature[0]
    else:
        model.first_layer = model._feature[1]
    model.n_category = model.struct[-1]
    ic_bp = IC_BP(model, reshape)
    
    ic_bp.grad_x_sign = np.zeros((3, X.shape[1], Y.shape[1]))
    ic_bp.gxs_cnt = np.zeros((3, reshape[1], Y.shape[1]))
    ic_bp.xs_cnt = np.zeros((3, reshape[1], Y.shape[1]))
    file = '../save/'+model.name+'/gxs.csv'
    if os.path.exists(file):
        ic_bp.gxs = np.loadtxt(file, delimiter = ',')
    else:
        ic_bp.gxs = None
        
    ic_bp.test_multi_sample(X, Y, 0, 'data', 
                            single = None, 
                            # single = 9832, 
                            label_name = label, text = text)
    return ic_bp

if __name__ == '__main__':
    ic_bp = cause_tracing(index = 4, dynamic = 40, ae_type = 'AE')
    result = ic_bp.result
    gxs_cnt = ic_bp.gxs_cnt
    xs_cnt = ic_bp.xs_cnt
    # print(look)
    # image_visualization(0)