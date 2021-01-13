# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import pandas as pd
from pandas import DataFrame
from PIL import Image
from torchvision.utils import save_image
from sklearn.preprocessing import MinMaxScaler

import matplotlib.colors as __colors__
import matplotlib.cm as __cmx__

_color_bar = ['nipy_spectral',  # 0 广色域-多色
              'jet',            # 1 广色域-平滑
              'gist_ncar',      # 2 广色域-中间艳丽
              'gist_rainbow',   # 3 鲜艳-七彩虹
              'hsv',            # 4 鲜艳-七彩虹
              'rainbow',        # 5 鲜艳-反七彩虹
              'cool', 'Wistia'  # 6 冷，7 暖
              'spring','summer','autumn','winter', # 8 春，9 夏，10 秋，11 冬
              ]

def _get_rgb_colors(data = None, scalar = None, cmap = 'nipy_spectral'):
    # 将1通道映射到3通道（在给定颜色条上取色）
    '''
        cmap: https://matplotlib.org/users/colormaps.html
        color: jet, gist_ncar, nipy_spectral, rainbow
        weight: hsv, gray
    '''
    import matplotlib.pyplot as plt
    if type(cmap) == int: cmap = _color_bar[cmap]
    cmap = plt.get_cmap(cmap)
    
    if type(data) == int:
        values = range(data)
        cNorm  = __colors__.Normalize(vmin=0, vmax=values[-1])
        scalarMap = __cmx__.ScalarMappable(norm=cNorm, cmap=cmap)
        colors = []
        for idx in range(data):
            colorVal = scalarMap.to_rgba(values[idx])
            colors.append(colorVal)
    else:
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if scalar is not None:
            _min, _max = scalar[0], scalar[1]
        else:
            _min, _max = data.min(), data.max()
        cNorm  = __colors__.Normalize(vmin=_min, vmax=_max)
        scalarMap = __cmx__.ScalarMappable(norm=cNorm, cmap=cmap)
        if data.ndim == 2:
            colors = scalarMap.to_rgba(data)[:,:,:-1]
        if data.ndim == 3:
            data = data[np.newaxis,:,:,:]
        if data.ndim == 4:
            shape = data.shape
            data = data.reshape(shape[0]*shape[1],shape[2],shape[3])
            data_list = []
            for i in range(data.shape[0]):
                _d = scalarMap.to_rgba(data[i])[:,:,:-1].transpose(2,0,1)
                data_list.append(_d[np.newaxis,:])
            colors = np.concatenate(data_list, axis = 0)
    return colors

def _save_img(data, scalar = None, path = ''):
    # 存单张图片（3通道）
    data = _get_rgb_colors(data, scalar, cmap = 'hsv')
    if np.max(data) <= 1:
        data = (data*255).astype(np.uint8)
    im = Image.fromarray(data)
    im.save(path + '.jpg')
    
def _save_multi_img(data, nrow, scalar = None, path = ''):
    # 存多张图片（3通道）
    if type(data) == list:
        data_list = []
        for _d in data:
            if _d.ndim == 2: _d = _d[np.newaxis,np.newaxis,:,:]
            data_list.append( _get_rgb_colors(_d, scalar, cmap = 'hsv') )
        data = np.concatenate(data_list, axis = 0)
    else:
        data = _get_rgb_colors(data, scalar, cmap = 'hsv')
          
    data = torch.from_numpy(data).cpu()
    save_image(data, path +'.png', nrow= nrow)
    
#-----------------------------------分割线-------------------------------------#

def t_SNE(X=None, y=None, 
          save_path = '../save/', 
          file_name = '1.png',
          color_cmap = 'nipy_spectral',
          show_info = True):
    import matplotlib.pyplot as plt
    from sklearn import manifold
    from matplotlib.ticker import NullFormatter
    
    # preprocess
    X = MinMaxScaler().fit_transform(X)
    if len(y.shape)>1 and y.shape[1]>1:
        y = np.array(np.argmax(y,axis=1).reshape(-1, 1),dtype=np.float32)
    
    # do transform
    X = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
    
    # split
    datas = []
    n_class = int(np.max(y)) - int(np.min(y)) + 1
    colors = _get_rgb_colors(n_class, color_cmap)
    for i in range(n_class):
        datas.append([])
    for i in range(y.shape[0]):
        datas[int(y[i])].append(X[i].reshape(1,-1))
    
    plt.style.use('default')
    fig = plt.figure(figsize=[32,18])
    ax = fig.add_subplot(111)
    
    # (x1, x2) with color 'y'
    for i in range(len(datas)):
        data_X = np.concatenate(datas[i], axis = 0)
        colors[i] = np.array(colors[i]).reshape(1,-1)
        plt.scatter(data_X[:, 0], data_X[:, 1], label = str(i + 1),
                    cmap=plt.cm.Spectral,
                    c =  colors[i] 
                    )
    if show_info: 
        # text
        for i in range(len(datas)):
            mean_x = np.mean( np.concatenate(datas[i], axis = 0), axis = 0)
            plt.text(mean_x[0], mean_x[1], str(i + 1), 
                     ha='center', va='bottom', 
                     fontdict={'family':'serif',
                               'style':'italic',
                               'weight':'normal',
                               'color': [0.21, 0.21, 0.21],
                               'size':26}
                     )
        # legend
        plt.legend(fontsize=28, loc = 1)
        # axis
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.yaxis.set_major_formatter(NullFormatter())
        plt.axis('tight')
    else:
        plt.axis('off')
    
    if not os.path.exists(save_path): os.makedirs(save_path)
    plt.savefig(save_path + file_name, bbox_inches = 'tight', format = 'png')
    plt.close(fig)

#-----------------------------------分割线-------------------------------------#
def _s(s):
    if s is not None: 
        s = eval(repr(s).replace(' ', '\;'))
        return '$'+s+'$'
    else: return s

def _draw(ax, b, x, y, c, l, ms = 1):
    if b is not None and b:
        ax.scatter(x, y, color = c, label= l)
    else:
        ax.plot(x, y, color = c, marker='o', markersize = 1, linestyle='-', linewidth=4, label= l )

def plot_curve(y,                   # [n_point × n_line] numpy.array
               y_twin = None,       # [n_point × n_line] numpy.array
               label = None,        # [ax_x, ax_y1, ax_y2] str list
               legend = '',         # [legend] str list
               name = '',           # model name
               curve_type = '',     # curve name
               x = None,            # [n_point] data_x
               text = False,        # show point loc by text
               title = None,        # img title
               style = 'classic',   # plot style
               color_cmap = _color_bar[3], # color bar
               max_line = True,     # plot max line
               legend_loc = [1, 4], # [(ax1_x, ax1_y),(ax2_x, ax2_y)] set legend loc
               scatter = None
               ): 
    '''
        plot: https://blog.csdn.net/dss_dssssd/article/details/84430024
    ''' 
    
    import matplotlib.pyplot as plt
    if type(style) == int:
        style = plt.style.available[style]
    print('Plot {} {}_curve with style: {}'.format(name, curve_type, style))
    plt.style.use(style)
    
    # [n × m] <- [n_point × n_line(y+y_twin)] 
    y = np.array(y)
    n = y.shape[0]
    if x is None:
        x = range(1,n+1)
    if y.ndim == 1: m = m1 = 1
    else: m = m1 = y.shape[1]
    if y_twin is not None:
        y_twin = np.array(y_twin)
        if y_twin.ndim == 1: m2 = 1
        else: m2 = y_twin.shape[1]
        m += m2
    
    # colors
    if type(color_cmap) == list:
        colors = color_cmap
    else:
        color_bais = 0
        if type(color_cmap) == tuple:
            color_cmap, color_bais = color_cmap
        colors = _get_rgb_colors(m + int(color_bais*2), cmap = color_cmap)
        if color_bais > 0: colors = colors[color_bais:-color_bais]
        colors.reverse()
        if len(colors) == 1: colors = ['r']

    if type(legend)!=list: legend = [legend]
        
    # markersize (circle)
    markersize = int(24/m)
    if y_twin is not None: markersize = int(markersize+2)
    
    fig = plt.figure(figsize=[32,18])
    ax1 = fig.add_subplot(111)
    
    if type(scatter) != list: scatter = [scatter]
    
    # plot axis y
    if y.ndim == 1:
        _draw(ax1, scatter[0], x, y, colors[0], _s(legend[0]), markersize)
    else:
        for i in range(m1):
            if len(legend)==1:
                lgd = legend[0]+'-'+str(i+1) 
            else:
                lgd = legend[np.mod(i, len(legend))]
            _draw(ax1, scatter[np.mod(i, len(scatter))], x, y[:,i], colors[i], _s(lgd), markersize)
    
    # plot max line
    if max_line and 'loss_acc' in curve_type:
        max_y = [y.max()] * y.shape[0]
        ax1.plot(x, max_y, color='black', linestyle='--', linewidth=1)
    
    # title
    if title is not None:
        ax1.set_title(title,fontsize=36)
        
    # set axis y's label
    if label is not None:
        ax1.set_xlabel(_s(label[0]),fontsize=48)
        ax1.set_ylabel(_s(label[1]),fontsize=48)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    
    # plot axis y_twin
    if y_twin is not None:
        ax2 = ax1.twinx()
        if y_twin.ndim == 1:
            ax2.plot(x, y_twin, color=colors[m1], marker='o', markersize = markersize, linestyle='-', linewidth=4, label=_s(legend[m1]))
        else:
            for i in range(m2):
                lgd = legend[np.mod(m1 + i, len(legend))]
                ax2.plot(x, y_twin[:,i], color=colors[m1 + i], marker='o', markersize = markersize, linestyle='-', linewidth=4, label=_s(lgd))
        
        # set axis y_twin's label
        if label is not None:
            ax2.set_ylabel(_s(label[2]),fontsize=48)
            
        # set axis y_twin's legend
        if legend[0] != '':
            lgd2 = ax2.legend(loc = legend_loc[1], ncol= int(np.sqrt(m2)+0.5), fontsize=24)
            lgd2.get_frame().set_alpha(0.5)
        plt.yticks(fontsize=40)
        # plt.yscale('logit')
    
    # set axis y's legend 
    if legend[0] != '':
        lgd1 = ax1.legend(loc = legend_loc[0], ncol= int(np.sqrt(m1)+0.5), fontsize=24)
        lgd1.get_frame().set_alpha(0.5)  
        # frame.set_facecolor('none')  
    
    # plot text
    if text:
        for a, b in zip(x, y):
            plt.text(a, b+0.05, '%.2f' % b, ha='center', va='bottom', fontsize=38, color = 'b')
    
    if not os.path.exists('../save/'+ name): os.makedirs('../save/'+ name)
    plt.savefig('../save/'+ name + '/['+name+'] '+ curve_type +'.png',bbox_inches='tight')
    #plt.show()
    plt.close(fig)
    
def _concatenate(x1, x2):
    x1 = np.array(x1).reshape(-1,1)
    x2 = np.array(x2).reshape(-1,1)
    return np.concatenate((x1,x2), axis = 1)

# plot loss & acc curve / rmse & R2 curve / pred & real curve
    
def loss_acc_curve(train_df, test_df, name):
    train_loss, train_acc = train_df['loss'], train_df['accuracy']
    test_loss, test_acc = test_df['loss'], test_df['accuracy']
    data = _concatenate(train_acc, test_acc)
    data_twin = _concatenate(train_loss, test_loss)
    plot_curve(data, 
               data_twin,
               ['Epochs', 'Average\;\;FDR\;(\%)', 'Loss'],
               ['Training\;\;FDR', 'Test\;\;FDR', 'Training\;\;loss', 'Test\;\;loss'],
               name,
               'loss_acc'
               )

def rmse_R2_curve(train_df, test_df, name):
    train_rmse, train_R2 = train_df['rmse'], train_df['R2']
    test_rmse, test_R2 = test_df['rmse'], test_df['R2']
    data = _concatenate(train_rmse, test_rmse)
    data_twin = _concatenate(train_R2, test_R2)
    plot_curve(data, 
               data_twin,
               ['Epochs', 'RMSE', 'R^{2}'],
               ['Training\;\;RMSE', 'Test\;\;RMSE', 'Training\;\;R^{2}', 'Test\;\;R^{2}'],
               name,
               'rmse_R2'
               )    

def rmse_mape_curve(train_df, name):
    test_rmse, test_mape = train_df['rmse'], train_df['mape']
    plot_curve(test_rmse, 
               test_mape,
               ['Epochs', 'RMSE', 'MAPE\;(\%)'],
               ['Test\;\;RMSE', 'Test\;\;MAPE'],
               name,
               'loss_mape'
               )
    
def pred_real_curve(pred, real, name, task = 'prd'):
    if task == 'prd': 
        axis_label, legend, curve_name = \
            ['Sampels', 'Prediction\;\;result'], ['pred', 'label'], 'pred_real', None
    elif task == 'impu': 
        axis_label, legend, curve_name = \
            ['Missing\;\;variable', 'Imputation\;\;result'], ['imputed', 'real'], 'imputed_real'
    data = _concatenate(pred, real)
    plot_curve(data,
               None,
               axis_label,
               legend,
               name,
               curve_name,
               color_cmap = ['b','r'],
               scatter = [True, False]
               )
               
# 把要对比的 result.xlsx 放入 'path'文件夹
def compare_curve(path = '../save/compare', 
                  curve_type = 1, 
                  use_data = [True, True], 
                  ax_y_name = None, 
                  color_cmap = 3,
                  legend_loc = None
                  ):
    model_name, y1, y2 = [], [], []
    ax_label_x = ['Epochs']
    _curve_type = ['','loss_acc', 'rmse_R2', 'pred_real']
    legend_name_y1, legend_name_y2 = [], []
    
    # read data from 'xlsx'
    file_list = os.listdir(path)  #列出文件夹下所有的目录与文件
    for file_name in file_list:
        if '.xlsx' in file_name:
            # get model name
            model_name.append( file_name[1 : file_name.index(']')] )
            model_name[-1] = model_name[-1].replace('-','$-$')
            
            # read data
            file_path = path+'/' + file_name
            epoch_curve = pd.read_excel(file_path, sheet_name = 'epoch_curve')
            
            if curve_type in [1, _curve_type[1]]:
                y1.append( epoch_curve['test_accuracy'].values.reshape(-1,1) )
                y2.append( epoch_curve['train_loss'].values.reshape(-1,1) )
                ax_label_y1, ax_label_y2 = ['Test\;\;average\;\;FDR\;(\%)'], ['Training\;\;loss']
                legend_name_y1.append('test\;\;\overline{FDR} (' + model_name[-1] + ')')
                legend_name_y2.append('training\;\;loss (' + model_name[-1] + ')')
            elif curve_type in [2, _curve_type[2]]:
                y1.append( epoch_curve['test_rmse'].values.reshape(-1,1) )
                y2.append( epoch_curve['test_R2'].values.reshape(-1,1) )
                ax_label_y1, ax_label_y2 = ['Test\;\;RMSE'], ['Test\;\;R^{2}']
                legend_name_y1.append('test\;\;\RMSE (' + model_name[-1] + ')')
                legend_name_y2.append('test\;\;R^{2} (' + model_name[-1] + ')')
            elif curve_type in [3, _curve_type[3]]:
                y1.append( epoch_curve['pred_Y'].values.reshape(-1,1) )
                if ax_y_name is not None: ax_label_y1 = ax_y_name
                else: ax_label_y1 = ['Y']
                legend_name_y1.append(model_name[-1])
                ax_label_y2 = []
    
    if curve_type in [3, _curve_type[3]]:
        y1.append( epoch_curve['real_Y'].values.reshape(-1,1) )
    else:
        y2 = np.concatenate(y2, axis =1)
    y1 = np.concatenate(y1, axis =1)
    
    legend_name = []
    if use_data[0] == False:                     # [False, ?]    y, y_twin = y2, None
        y1 = y2 
        y2 = None
        ax_label = ax_label_x + ax_label_y2
        legend_name = legend_name_y2
    elif use_data[1] == False:                   # [Ture, False] y, y_twin = y1, None
        y2 = None
        ax_label = ax_label_x + ax_label_y1
        legend_name = legend_name_y1
    else:                                        # [True, True]  y, y_twin = y1, y2 (default)
        ax_label = ax_label_x + ax_label_y1 + ax_label_y2
        legend_name = legend_name_y1 + legend_name_y2
    
    if type(curve_type) == int: curve_name = _curve_type[curve_type]
    else: curve_name = curve_type
    
    plot_curve(y1, 
               y2,
               ax_label,
               legend_name,
               'compare',
               str(len(model_name)) + '_' + curve_name,
               legend_loc = legend_loc,
               color_cmap = color_cmap
               )

#-----------------------------------分割线-------------------------------------#
def _get_categories_name(label, N):
    _labels = []
    if type(label)!= list: label = [label]
    for i in range(N):
        if label[0] is not None: 
            if len(label) < N  and i >= len(label) - 1:
                _labels.append(label[-1] + ' ' + str(i - len(label) + 2))
            else:
                _labels.append(label[i])
        else:
            _labels.append('Category ' + str(i + 1))
    return _labels

def category_distribution(prd_cnt, label = None, name = '', info = '',
                          text = 'cnt', diag_cl = True, plot_size = None):
    import matplotlib.pyplot as plt
    plt.style.use('default')
    if info == '':
        print('Plot [{}]{} pred_distrib with style: default'.format(name,info))
    
    if label is None:
        axis_x, axis_y = np.arange(prd_cnt.shape[1]), np.arange(prd_cnt.shape[0])
    elif type(label) == list:
        axis_x, axis_y = label.copy(), label.copy()
        for i in range(len(axis_x)): axis_x[i] = _s(axis_x[i] + '_r')
        for i in range(len(axis_y)): axis_y[i] = _s(axis_y[i] + '_p')
    elif type(label) == tuple:
        axis_x, axis_y = label[0].copy(), label[1].copy()
        for i in range(len(axis_x)): axis_x[i] = _s(axis_x[i])
        for i in range(len(axis_y)): axis_y[i] = _s(axis_y[i])
    
    prd_cnt = np.array(prd_cnt, dtype = np.int32)
    n_sample_cnts = np.sum(prd_cnt, axis = 0, dtype = np.int)
    n_sample_cnts[n_sample_cnts == 0] = 1
    prd_pro = prd_cnt / n_sample_cnts
    imshow = np.array(np.around(prd_pro*100,0), dtype = np.int32)
    
    #mat = np.round(mat,decimals=0)
    #mat = np.transpose(mat)
    
    size = 32
    max_size = max(prd_cnt.shape[1], prd_cnt.shape[0]) 
    ticksize = size/max_size*18   # label
    fontsize = size/max_size*18   # text
    
    figsize = [prd_cnt.shape[0] * 2, prd_cnt.shape[1] * 2] 
    if plot_size is not None:
        figsize = plot_size
    
    #print(figsize, ticksize, fontsize)
    
    fig =  plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    #cmap = "magma_r"
    #cmap = "Blues"
    cmap = "gist_yarg"

    im = ax.imshow(imshow, cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    
    plt.xticks(fontsize=ticksize)
    plt.yticks(fontsize=ticksize)
    
    ax.set_xticks(np.arange(len(axis_x)))
    ax.set_yticks(np.arange(len(axis_y)))

    ax.set_xticklabels(axis_x)
    ax.set_yticklabels(axis_y)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor") # 旋转
    
    # print(prd_cnt.shape, len(axis_y), len(axis_x))
    
    for i in range(len(axis_y)):
        for j in range(len(axis_x)):
            x = prd_cnt[i,j]
            p = int(np.round(prd_pro[i,j]*100.0, 0))
            if text == 'cnt': t = x
            else: t = p
            
            if diag_cl:
                if i == j:
                    cl = 'w'
                elif p > 0: cl = 'b'
                else: cl = 'black'
            
            else:
                if p < 38.2: cl = 'black'
                else: cl = 'w'
            
            ax.text(j, i, t, ha="center", va="center", color=cl, fontsize=fontsize)
    
    if not os.path.exists('../save/'+ name): os.makedirs('../save/'+ name)
    plt.savefig('../save/'+ name +'/['+name+']'+info+' pred_distrib.png',bbox_inches='tight')
    
    plt.close(fig)

if __name__ == '__main__':
    # compare_curve
    # compare_curve(color_cmap = [3,0], legend_loc = [(0.565, 0.887),(0.524, 0.152)])
    
    # beta
    # data = np.array([92.49, 93.62, 93.84, 94.08, 94.15, 94.42, 94.49, 95.12, 95.84, 96.02])
    # x = np.linspace(0.1,1.0,10)
    # plot_curve(data, 
    #             None,
    #             ['Loss\;\;coefficient\;\;\\beta','Test\;\;average\;\;FDR\;(\%)'],
    #             x = x,
    #             text = True,
    #             style = 1, 
    #             color_cmap = 'rainbow'
    #             )
    
    # category_distribution
    label = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
             'Fault 08']
    cls_result = pd.read_excel('../save/CG-SAE-5 8t/[CG-SAE] result.xlsx', sheet_name = 'cls_result').values
    cls_result = cls_result[:len(label),1:]
    category_distribution(cls_result, label)
    
