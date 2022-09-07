"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
import matplotlib.cm as mpl_color_map
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from misc_functions import (convert_to_grayscale,
                            save_gradient_images)
import sys
sys.path.append('..')
from core.func import find_act
from _test.cls_sae import get_sae_model
from visual.plot import category_distribution
            
class RunData():
    def save_img(self, img, index, file_name_to_export, pic):
        # save_img
        file_name_to_export = '({})-[{}] '.format(index, file_name_to_export) + self.model.name
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

    def save_data(self, X, Y):
        X, Y = X.reshape((-1,)), Y.reshape((-1,))
        n, m = self.n_category, self.reshape[1]
        array = np.zeros((m,n))
        for i in range(X.shape[0]):
            c, k = Y[i], X[i]
            if self.show_info:
                print('\n',k,c)
            array[k][c] += 1
        array = array[:,1:]
        label = (['Fault 01','Fault 02','Fault 03','Fault 04','Fault 05','Fault 06',
                  'Fault 07','Fault 08','Fault 09','Fault 10'],
                 ['Ci^{(in)}','Ti^{(in)}','Tci^{(in)}','Ci','Ti','C','T','Qc','Tci','Tc'])
        print()
        if self.show_info == False:
            category_distribution(array, label = label, name = self.model.name, 
                                  text = 'cnt', diag_cl = False)
        self.result = array
    
    def test_multi_sample(self, X, Y, n_sampling = 0, 
                          save_type = 'img', pic = 'all', 
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
        zero_grad_index = []
        for i in index:
            self.save_cnt += 1
            # info
            msg_str = '{}/{} Test sample index {}'.format(self.save_cnt,n_sampling,i)
            sys.stdout.write('\r'+ msg_str)
            sys.stdout.flush()
            # take x, y
            x, y = X[i], Y[i]
            data_y.append(y.reshape(1,-1))
            # get grad
            guided_grads = self.generate_gradients(x, y)
            if save_type == 'img' or self.show_info:
                try:
                    self.save_img(guided_grads, i, str(y), pic)
                except TypeError:
                    pass
            if save_type == 'data':
                if self.show_info: 
                    print('\nguided_grads:', guided_grads.shape, guided_grads.min(), guided_grads.max())
                x = np.squeeze(guided_grads, axis=0) # delete first dim
                self.grad_array = x
                if x.max() == 0:
                    zero_grad_index.append((i, y))
                    
                ''' rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr '''
                x = np.abs(x)
#                x = np.clip(x, 0.0, np.inf)
                if self.show_info: 
                    print('x_2d:', x.shape, x.min(), x.max())
                
                x = np.argmax(x, axis = 1)
                x = np.bincount(x, minlength = self.reshape[1])
             
                if self.show_info: 
                    print('x_1d:', x)
                data_x.append(x.reshape((1,-1)))
        
        if len(zero_grad_index)>0:
            print('\nThere are a total of {} samples with zero gradients:\n'.\
                  format(len(zero_grad_index)),zero_grad_index)
        
        if save_type != 'img':
            data_x = np.concatenate(data_x, axis = 0)
            data_x = np.argmax(data_x, axis = 1)
            data_y = np.concatenate(data_y, axis = 0)
            self.save_data(data_x, data_y)

        self.del_handles()

def grad_info(module, grad_in, grad_out):
    print()
    print(module)
    name_in, name_out = 'in', 'out'
     
    for grad in list(grad_in):
        if grad is not None and grad.size(0) == 1:
            print('grad_'+name_in+':', grad.size(), grad.min(), grad.max())

    for grad in list(grad_out):
        if grad is not None:
            print('grad_'+name_out+':', grad.size(), grad.min(), grad.max())
            
    if hasattr(module, 'weight'):
        weight, bias = module.weight.data, module.bias.data
        print('weight:', weight.size(), weight.min(), weight.max())
        print('bias:', bias.size(), bias.min(), bias.max())        
        
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
    
class GuidedBackprop(RunData):
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model, first_layer_name,
                 reshape = None, show_info = False):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.reshape = reshape
        self.z_h = []
        self.handles = []
        self.show_info = show_info
        self.grad_array = None
        # Put model in evaluation mode
        self.guided_bp_hooks(first_layer_name)
        
        self.modi_layers = []
        for i in [2,5,8]: # [2, 5, 8]
            self.modi_layers.append(self.model._feature[i])
 
    def guided_bp_hooks(self, first_layer_name):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """      
        def _backward_act_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            backward_hook(module, grad_input, grad_output) -> Tensor or None
            """
            grad_in, grad_out = list(grad_in), list(grad_out)
            grad_z, grad_h = grad_in[0].data, grad_out[0].data
            (z, h) = self.z_h[-1]
            grad_f = get_grad_f(module.__class__.__name__, z, h)
            
            if self.show_info: 
                grad_info(module, grad_in, grad_out)
#                print('grad_f:', grad_f.size(), grad_f.min(), grad_f.max()) 
#                diff = grad_z - torch.mm( grad_h,  grad_f)
#                print('diff:', diff.size(), diff.min(), diff.max())
                print('z:', z.size(), z.min(), z.max())        
                print('h:', h.size(), h.min(), h.max())
            
            grad_h = torch.clamp(grad_h, min=0.0)
#            grad_z = z.sign() * h * grad_h
            grad_z = h * torch.mm(grad_h,  grad_f.sign())
          
            if self.show_info: print('modified_grad_in:', grad_z.size(), grad_z.min(), grad_z.max())
            
            del self.z_h[-1]
            grad_in[0] = grad_z
            
#            if module.act_index < len(self.act_module):
            return tuple(grad_in)
        '''
        ########################################################################################
        '''
        
        def _forward_act_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            forward_hook(module, input, output) -> None
            """
            self.z_h.append((ten_in[0].data, ten_out[0].data))
        
        def _backward_linear_function(module, grad_in, grad_out):
            if self.show_info: grad_info(module, grad_in, grad_out)
            if module == self.first_layer:
                self.gradients = grad_in[1]
        
        # Loop through layers, hook up ReLUs   
        self.first_layer = eval('self.model.'+first_layer_name)
        self.act_module = []
            
        for i, module in enumerate(self.model.modules()):
            act = find_act(module)
            if act is not None:
                self.act_module.append(module)
                module.act_index = len(self.act_module)
                self.handles.append(module.register_forward_hook(_forward_act_function))
                self.handles.append(module.register_backward_hook(_backward_act_function))
                
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_backward_hook(_backward_linear_function))

    def generate_gradients(self, input_image, target_class):
        # Input (tensor)
        input_image = torch.from_numpy(input_image).float()
        input_image.unsqueeze_(0)
        input_image = Variable(input_image, requires_grad = True)
        # Forward pass
        self.model.zero_grad()
        model_output = self.model.forward(input_image)
        if self.show_info:
            print('\nReal Y:',target_class)
            print('Predicted Y:', model_output, torch.argmax(model_output, axis = 1))
        # Zero gradients
        
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
        gradients_as_arr = self.gradients.data.numpy()
        if gradients_as_arr.ndim == 4:
            # [0] to get rid of the first channel (1,3,224,224) -> (3,224,224)
            gradients_as_arr = gradients_as_arr[0]
        elif gradients_as_arr.ndim == 2 and self.reshape is not None:
            gradients_as_arr = gradients_as_arr.reshape((1,self.reshape[0],self.reshape[1]))
        return gradients_as_arr
    
    def del_handles(self):
        for handle in self.handles:
            handle.remove()
            

if __name__ == '__main__':
    model, datasets, _ = get_sae_model(3)
    model._save_load('load', 'best')
    model.load_data(datasets, 32)
    model.test(0, 'train')
    # Guided backprop
    GBP = GuidedBackprop(model, '_feature[1]', (40,10))
    GBP.test_multi_sample(datasets[0], datasets[1], 0, 'data', single = None)
    grad_array = np.abs(GBP.grad_array)