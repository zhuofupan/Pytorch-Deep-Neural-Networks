# -*- coding: utf-8 -*-
import cv2
import sys
import numpy as np
import sympy as sp
from scipy.linalg import orth
sys.path.append('..')
from core.plot import _save_img, _save_multi_img
from private.read_te_data import gene_net_datas

def _to_even(x):
    x1 = x.astype('float')
    N, M = x.shape
    if np.mod(N, 2) == 1: 
        x1 = np.concatenate((x1, np.zeros((1,M))), axis = 0)
    if np.mod(M, 2) == 1:
        x1 = np.concatenate((x1, np.zeros((x1.shape[0],1))), axis = 1)
    return x1

def _blur(x, _filter = 'gauss', size = (3,3)):
    if _filter == 'gauss':
        return cv2.GaussianBlur(x, size, 0)
    elif _filter == 'mean':
        return cv2.blur(x, size)
    elif _filter == 'kernel':
        kernel = np.ones(size, np.float32)/(np.mean(np.array(size))**2)
        return cv2.filter2D(x, -1, kernel)

def dct(x):
    '''
    进行离散余弦变换
    矩阵的行数和列数必须是偶数
    '''
    x1 = _to_even(x)
    
    x_dct = cv2.dct(x1)   
    x_dct_log = np.log(abs(x_dct) + 1e-10) 
    return x_dct, x_dct_log

def idct(x_dct):
    x_dct = _to_even(x_dct)
    return cv2.idct(x_dct)

def dft(x, mask = False):
    x = _to_even(x)
    # 傅里叶变换
    x_dft = cv2.dft(np.float32(x), flags=cv2.DFT_COMPLEX_OUTPUT)
    # 移频
    x_dft_shift = np.fft.fftshift(x_dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(x_dft_shift[:, :, 0] + 1e-10, x_dft_shift[:, :, 1] + 1e-10))
     
    if mask:
        rows, cols = x.shape
        crow, ccol = int(rows/2), int(cols/2)
        # 创建一个掩膜，中间方形为1，其余为0
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1
        # 使用掩膜
        x_dft_shift = x_dft_shift*mask
        magnitude_spectrum = 20 * np.log(cv2.magnitude(x_dft_shift[:, :, 0] + 1e-10, x_dft_shift[:, :, 1] + 1e-10))
    return x_dft, x_dft_shift, magnitude_spectrum
    
def idft(x_dft, ishift = False):
    if ishift:
        # 逆移频
        x_dft = np.fft.ifftshift(x_dft)

    # 逆傅里叶变换
    x_idft = cv2.idft(x_dft)
    x_idft = cv2.magnitude(x_idft[:, :, 0], x_idft[:, :, 1])
    return x_idft

def x2(x):
    return np.dot(x,  x.T)

def gs(x, m = 'sp', e = False):
    if m == 'sp':
        _x = []
        for j in range(x.shape[1]):
            _x.append(sp.Matrix(x[:,j].tolist()))
        x = np.array(sp.GramSchmidt(_x, e), dtype= np.float32)  
    elif m == 'or':
        x = orth(x)
    return x

def _trans(data, trans = 'dct', reshape = None, pro = None):
    _data = []
    for i in range(data.shape[0]):
        x = data[i]
        if reshape is not None:
            x = x.reshape(*list(reshape))
        if pro is not None:
            pro = list(pro)
            for p in pro:
                x = eval(p+'(x)')
        x = eval(trans+'(x)')
        if type(x) == tuple:
            x = x[-1]
        _data.append(x.reshape(1,-1))
    _data = np.concatenate(_data, axis = 0)
    return _data

if __name__ == '__main__':
    datasets = gene_net_datas(
               data_dir='../private/TE',
               preprocessing='mm', # gauss单元用‘st’, bin单元用'mm'
               one_hot=True,
               shuffle=False,
               # 考虑动态数据集
               dynamic=40,
               stride=0,
               load_data=False)
    train_x, train_y, test_x, test_y = datasets
    samples = []
    j = 0
    cnt = 0
    for i in range(train_y.shape[0]):
        if j >= train_y.shape[1]: break
        if train_y[i,j] == 1:
            x = train_x[i].reshape((40,33))
            #w = np.random.rand(33,33)
            #x = np.dot( np.dot(x, w) ,  x.T)
#            x = gs(x.T, 'sp')
#            x = x.T
#            print(x.shape)
#            if x.shape != (40,33):
#                continue
            #x = np.dot(x,  x.T)
            #print(x)
#            x = _blur(x, 'mean', (3,3))
            samples += [ x ]
            cnt += 1
            if cnt > 1:
                cnt = 0
                j += 1
    imgs = []
    for i in range(len(samples)):
        x = samples[i]
        # 原始
        imgs += [_to_even(x)]
        # dct
        x_dct, x_dct_log = dct(x)
        imgs += [x_dct_log]
        # dct + idct
        imgs += [idct(x_dct)]
        # dct + dft
        _, x_dft_shift, magnitude_spectrum = dft(x_dct)
        imgs += [magnitude_spectrum]
        x_idft = idft(x_dft_shift, True)
        imgs += [x_idft]
        # dct + dct
        _, x_dct_log2 = dct(x_dct)
        imgs += [x_dct_log2]
        # dft , no mask
        _, x_dft_shift, magnitude_spectrum = dft(x)
        imgs += [magnitude_spectrum]
        x_idft = idft(x_dft_shift, True)
        imgs += [x_idft]
        # dft , mask
        _, x_dft_shift, magnitude_spectrum = dft(x, True)
        imgs += [magnitude_spectrum]
        x_idft = idft(x_dft_shift, True)
        imgs += [x_idft]
        
    _save_multi_img(imgs,10)