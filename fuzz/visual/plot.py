# -*- coding: utf-8 -*-
import os
import torch
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from PIL import Image
from torchvision.utils import save_image
from sklearn.preprocessing import MinMaxScaler

import matplotlib.colors as __colors__
import matplotlib.cm as __cmx__
import matplotlib.pyplot as plt

from fuzz.core.epoch import _get_subplot_size

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

class Plot_Curve():
    '''
        plot: https://blog.csdn.net/dss_dssssd/article/details/84430024
    ''' 
    def __init__(self, 
                 y,                            # 主轴 [subnumber, n_point, n_y] 
                 y_twin = None,                # 副轴 [subnumber, n_point, n_y] 
                 model_name = '',              # 模型名
                 curve_name = '',              # 曲线名
                 figsize = [32, 18],           # 画布大小
                 style = 'classic',            # 绘制风格
                 color_cmap = _color_bar[3],   # 颜色条
                 ):
        # 用于绘图的数据
        self.y, self.y_twin = y, y_twin
        # 各轴绘制的线条数
        self.n_ytwin = 0
        if type(y) == list: 
            self.subnumber = len(y)
            self.n_y = y[0].shape[1]
            if y_twin is not None: self.n_ytwin = y_twin[0].shape[1]
        else:
            self.subnumber = 1
            self.n_y = y.shape[1]
            if y_twin is not None: self.n_ytwin = y_twin.shape[1]
            self.y, self.y_twin = [self.y], [self.y_twin]
            
        self._markersize = int(24/self.n_y)
        if y_twin is not None: self._markersize = int(self._markersize+2)
        
        # 子视窗数
        self.n_row, self.n_col = 1, 1
        if self.subnumber > 1:
            self.n_row, self.n_col = _get_subplot_size(self.subnumber)
        
        # 初始化画布
        self.figsize = [figsize[0]* self.n_row, figsize[1]* self.n_col]
        self.font_enlarge = min(self.n_row, self.n_col)
        self.fig = plt.figure(figsize=self.figsize)
        
        # 绘图风格
        if type(style) == int:
            style = plt.style.available[style]
        self.style = style
        plt.style.use(style)
        self.model_name, self.curve_name = model_name, curve_name
        print('Plot {} {}_curve with style: {}'.format(model_name, curve_name, style))
        
        # 颜色条
        if type(color_cmap) == list:
            colors = color_cmap
        else:
            color_bais = 0
            if type(color_cmap) == tuple:
                color_cmap, color_bais = color_cmap
            colors = _get_rgb_colors(self.n_y + self.n_ytwin + int(color_bais*2), cmap = color_cmap)
            if color_bais > 0: colors = colors[color_bais:-color_bais]
            colors.reverse()
            if len(colors) == 1: colors = ['r']
        self.colors = colors

    def _distribute(self, a, add_id = False):
        if type(a) == tuple: a, b = a
        else: a, b = a, None
        for i, (x, n) in enumerate([(a, self.n_y), (b, self.n_ytwin)]):
            if x is None: continue
            if type(x) != list:
                x = [x] * n
                if add_id:
                    for i in range(n):
                        x[i] += '-' + str(i+1)
                if i == 0: a = x
                else: b = x
        return a, b
        
    # tuple(主轴设置[list], 副轴设置[list]), 非list时默认为前项变动
    ''' 
    https://blog.csdn.net/htuhxf/article/details/82863630
    ls - linestyle：['solid', 'dashed', 'dashdot', 'dotted'] ['-', '--', '-.', ':']
    lw - linewidth
    ms - markersize：设定 marker 大小
    mec - markeredgecolor：设定 marker 边框颜色
    mew - markeredgewidth：设定 marker 边框粗细
    mfc - markerfacecolor：设定 marker 填充颜色 
    '''
    def plot(self,
             legend = (None, None),
             linestyle = ('-', '-'),
             linewidth = (4, 4),
             marker = ('o', 'o'),
             markersize = (8, 8),
             markersetting = ['none', 0],      # mec, mew
             max_line = False,
             title = None,
             axis_yname = (None, None),
             axis_xname = None,
             fontsize = [54, 42, 40, 24, 38],  # title, ax_label, ax_ticks, legend, text
             legend_loc = [1, 4],
             ax_scale = (None, None),
             add_text = False
             ):
        fontsize = (np.array(fontsize) *np.power(1.2, self.font_enlarge)).astype(int)
        legend, legend_t = self._distribute(legend, True)
        linestyle, linestyle_t = self._distribute(linestyle)
        linewidth, linewidth_t = self._distribute(linewidth)
        marker, marker_t = self._distribute(marker)
        markersize, markersize_t = self._distribute(markersize)
        title, _ = self._distribute(title, True)
        axis_yname, axis_yname_t = self._distribute(axis_yname)
        axis_xname, _ = self._distribute(axis_xname)
        self.markersetting = markersetting
        for sub_id in range(self.subnumber):
            ax = self.fig.add_subplot(self.n_row, self.n_col, sub_id + 1)
            # plot y
            Y = self.y[sub_id]
            for i in range(self.n_y):
                y = Y[:,i]
                self.plot_in_fig(ax,
                                 y, 
                                 self.colors[np.mod(i, len(self.colors))], 
                                 linestyle[np.mod(i, len(linestyle))], 
                                 linewidth[np.mod(i, len(linewidth))],
                                 marker[np.mod(i, len(marker))],
                                 markersize[np.mod(i, len(markersize))],
                                 legend[np.mod(i, len(legend))]
                                 )
            # plot max line
            if max_line:
                max_y = [y.max()] * y.shape[0]
                self.plot_in_fig(max_y, 'black', '--', 1, 0, None)
            
            # set title
            if title is not None:
                ax.set_title(_s(title[np.mod(sub_id, len(title))]),fontsize=fontsize[0])
                
            # set y's label
            if axis_xname is not None:
                ax.set_xlabel(_s(axis_xname[np.mod(sub_id, len(axis_xname))]), fontsize=fontsize[1])
            if axis_yname is not None:
                ax.set_ylabel(_s(axis_yname[np.mod(sub_id, len(axis_yname))]), fontsize=fontsize[1])
            plt.xticks(fontsize=fontsize[2])
            plt.yticks(fontsize=fontsize[2])
            if ax_scale[0] is not None: plt.yscale(ax_scale[0])
            
            # limit range
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xmin, xmax = 1,  Y.shape[0] + 1
                plt.xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.02)
                loc = np.where(Y == Y)
                ymin, ymax = np.min(Y[loc]), np.max(Y[loc])
                ax.set_ylim(ymin - (ymax - ymin) * 0.02, ymax + (ymax - ymin) * 0.02)
            
            if self.n_ytwin > 0:
                ax_twinx = ax.twinx()
                # plot y_twin
                Y_Twin = self.y_twin[sub_id]
                for i in range(self.n_ytwin):
                    y_twin = Y_Twin[:,i]
                    self.plot_in_fig(ax_twinx,
                                     y_twin, 
                                     self.colors[np.mod(i + self.n_y, len(self.colors))], 
                                     linestyle_t[np.mod(i, len(linestyle_t))], 
                                     linewidth_t[np.mod(i, len(linewidth_t))],
                                     marker_t[np.mod(i, len(marker_t))],
                                     markersize_t[np.mod(i, len(markersize_t))],
                                     legend_t[np.mod(i, len(legend_t))]
                                     )
                
                # set y_twin's label
                if axis_yname_t is not None:
                    ax_twinx.set_ylabel(_s(axis_yname_t[np.mod(sub_id, len(axis_yname_t))]),fontsize=fontsize[1])
            
                # set y_twin's legend
                if legend_t is not None:
                    lgd = ax_twinx.legend(loc = legend_loc[1], ncol= int(np.sqrt(self.n_ytwin)+0.5), fontsize=fontsize[3])
                    lgd.get_frame().set_alpha(0.5)
                plt.yticks(fontsize=fontsize[2])
                if ax_scale[1] is not None: plt.yscale(ax_scale[1])
                # plt.yscale('logit')
                
                # limit range
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loc = np.where(Y_Twin == Y_Twin)
                    ymin, ymax = np.min(Y_Twin[loc]), np.max(Y_Twin[loc])
                    ax_twinx.set_ylim(ymin - (ymax - ymin) * 0.02, ymax + (ymax - ymin) * 0.02)
           
            # set axis y's legend 
            if legend is not None:
                lgd = ax.legend(loc = legend_loc[0], ncol= int(np.sqrt(self.n_y)+0.5), fontsize=fontsize[3])
                lgd.get_frame().set_alpha(0.5)  
                # frame.set_facecolor('none')
                
            # plot text
            if add_text:
                x = range(1, 1 + Y.shape[0])
                for i in range(self.n_y):
                    y = Y[:,i]
                    for a, b in zip(x, y):
                        plt.text(a, b+0.05, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize[4], color = 'b')
        
        # save
        if not os.path.exists('../save/'+ self.model_name): os.makedirs('../save/'+ self.model_name)
        plt.savefig('../save/'+ self.model_name + '/['+self.model_name+'] '+ self.curve_name +'.png',bbox_inches='tight')
        #plt.show()
        plt.close(self.fig)               
    
    def plot_in_fig(self, ax, y, color, linestyle, linewidth, marker, markersize, legend):
        dafault = {'color': color, 'linestyle': linestyle, 'linewidth': linewidth,
                   'marker': marker, 'markersize': markersize, 'label': _s(legend), 'mec': 'none', 'mew': 0}
        for key, value in dafault.items():
            if key == 'markersize' and value == 0: dafault[key] = self._markersize
            if value is None: del dafault[key]
        
        x = range(1, 1 + y.shape[0])
        if linestyle == '':
            del dafault['linestyle']
            del dafault['markersize']
            del dafault['mec']
            del dafault['mew']
            ax.scatter(x, y, **dafault)
        else:
            ax.plot(x, y, **dafault)
        
def _concatenate(x_list):
    if len(x_list) == 0: return None
    y_list = []
    for x in x_list:
        y_list.append(np.array(x).reshape(-1,1)) 
    return np.concatenate(y_list, axis = 1)

# plot loss & acc curve / rmse & R2 curve / pred & real curve
    
def loss_acc_curve(train_df, test_df, name):
    train_loss, train_acc = train_df['loss'], train_df['accuracy']
    test_loss, test_acc = test_df['loss'], test_df['accuracy']
    data = _concatenate([train_acc, test_acc])
    data_twin = _concatenate([train_loss, test_loss])
    Plot_Curve(data, 
               data_twin,
               name,
               'loss_acc'
               ).plot(axis_yname = ('Average\;\;FDR\;(\%)', 'Loss'),
                      axis_xname = 'Epochs',
                      legend = (['Training\;\;FDR', 'Testing\;\;FDR'], 
                                ['Training\;\;loss', 'Testing\;\;loss'])
                      )

def rmse_R2_curve(train_df, test_df, name):
    train_rmse, train_R2 = train_df['rmse'], train_df['R2']
    test_rmse, test_R2 = test_df['rmse'], test_df['R2']
    data = _concatenate([train_rmse, test_rmse])
    data_twin = _concatenate([train_R2, test_R2])
    Plot_Curve(data, 
               data_twin,
               name,
               'rmse_R2'
               ).plot(axis_yname = ('RMSE', 'R^{2}'),
                      axis_xname = 'Epochs',
                      legend = (['Training\;\;RMSE', 'Testing\;\;RMSE'], 
                                ['Training\;\;R^{2}', 'Testing\;\;R^{2}'])
                      )

def rmse_mape_curve(train_df, name):
    test_rmse, test_mape = train_df['rmse'], train_df['mape']
    data, data_twin = np.array(test_rmse).reshape(-1,1),\
        np.array(test_mape).reshape(-1,1)
    Plot_Curve(data, 
               data_twin,
               name,
               'loss_mape',
               color_cmap = ['b','r'],
               ).plot(axis_yname = ('RMSE', 'MAPE\;(\%)'),
                      axis_xname = 'Epochs',
                      legend = (['Testing\;\;RMSE'], 
                                ['Testing\;\;MAPE']),
                      markersize = (8, 8)
                      )
    
def pred_real_curve(pred, real, name, task = 'prd'):
    data = _concatenate([pred, real])
    Plot_Curve(data, 
               None,
               name,
               'pred_real',
               color_cmap = ['b','r'],
               ).plot(axis_yname = 'Prediction\;\;result',
                      axis_xname = 'Samples',
                      legend = ['pred', 'real'],
                      linestyle = ['', '-']
                      )

def var_impu_curve(X, Y, NAN, missing_var_id, name):
    data, title = [], []
    for i, index in enumerate(missing_var_id):
        title.append('Variable {}'.format(index + 1))
        x, y, nan = X[:,index], Y[:,index], NAN[:,index]
        loc = np.where(nan == 1)
        _x, _y = x[loc], y[loc]
        order = np.argsort(_y)
        _x, _y = _x[order], _y[order]
        data.append(np.concatenate([_x.reshape(-1,1), _y.reshape(-1,1)], 1))
    Plot_Curve(data, 
               None,
               name,
               'var_impu',
               color_cmap = ['b','r'],
               figsize = [40, 16]
               ).plot(axis_yname = 'Values',
                      axis_xname = 'Incomplete\;\;samples',
                      title = title,
                      fontsize = [54, 45, 38, 30, 38], # title, ax_label, ax_ticks, legend, text
                      legend_loc = ('upper left',),
                      legend = ['imputed', 'real'],
                      linestyle = ['', '-'],
                      linewidth = 6,
                      markersize = [4, 2]
                      )
    

# 把要对比的 result.xlsx 放入 'path'文件夹
read_col_dict = {'train_loss': ['Training\;\;loss', 'training\;\;loss'],
                 'test_accuracy': ['Test\;\;average\;\;FDR\;(\%)', 'test\;\;\overline{FDR}'],
                 'test_rmse': ['Test\;\;RMSE', 'test\;\;RMSE'],
                 'test_R2': ['Test\;\;R^{2}', 'test\;\;R^{2}'],
                 'train_rmse': ['Test\;\;RMSE', 'test\;\;RMSE'],
                 'train_mape': ['Test\;\;MAPE\;(\%)', 'test\;\;MAPE']
    }
def epoch_comparing_curve(path = '../save/compare', 
                         read_col = ('', ''), 
                         color_cmap = _color_bar[5],
                         legend_loc = (1,),
                         **kwargs
                         ):
    y_label_name, basic_legend_name = read_col_dict[read_col[0]]
    has_twin = False
    if read_col[1] is not None and read_col[1]!= '':
        y_label_name_t, basic_legend_name_t = read_col_dict[read_col[1]]
        has_twin = True
    
    model_name = []
    y_data, y_data_t = [], []
    legend_name, legend_name_t = [], []
    # read data from 'xlsx'
    file_list = os.listdir(path)  #列出文件夹下所有的目录与文件
    for file_name in file_list:
        if '.xlsx' in file_name:
            # get model name
            model_name.append( file_name[1 : min(file_name.index(' '),file_name.index(']'))] )
            model_name[-1] = model_name[-1].replace('-','$-$')
            
            # read data
            file_path = path+'/' + file_name
            epoch_curve = pd.read_excel(file_path, sheet_name = 'epoch_curve')
            y_data.append( epoch_curve[read_col[0]].values.reshape(-1,1) )
            if has_twin:
                legend_name.append( basic_legend_name + ' (' +model_name[-1] + ')' )
                y_data_t.append( epoch_curve[read_col[1]].values.reshape(-1,1) )
                legend_name_t.append( basic_legend_name_t + ' (' +model_name[-1] + ')' )
            else:
                legend_name.append( model_name[-1] )
    if has_twin == False:
        y_label_name_t, legend_name_t = None, None
    
    save_name = read_col[0]
    if has_twin:  save_name += ' & ' + read_col[1]
                
    Plot_Curve(_concatenate(y_data), 
               _concatenate(y_data_t),
               path[path.rfind('/')+1:],
               save_name,
               color_cmap = color_cmap
               ).plot(legend = (legend_name, legend_name_t),
                      axis_yname = (y_label_name, y_label_name_t),
                      axis_xname = 'Epoch',
                      legend_loc = legend_loc,
                      **kwargs
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
    epoch_comparing_curve('../save/Impu_curve', ('train_mape',''), 
                          fontsize = [54, 42, 40, 30, 38],
                          linewidth = (4, 0),
                          marker = ('o', ''),
                          markersize = (8, 0),)
    
    # category_distribution
    # label = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
    #          'Fault 08']
    # cls_result = pd.read_excel('../save/CG-SAE-5 8t/[CG-SAE] result.xlsx', sheet_name = 'cls_result').values
    # cls_result = cls_result[:len(label),1:]
    # category_distribution(cls_result, label)
    
