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

# recommend: 3 for twins, 5 for single
_color_bar = ['nipy_spectral',  # 0 广色域-多色
              'jet',            # 1 广色域-平滑
              'gist_ncar',      # 2 广色域-中间艳丽
              'gist_rainbow',   # 3 鲜艳-七彩虹 *
              'hsv',            # 4 鲜艳-七彩虹
              'rainbow',        # 5 鲜艳-反七彩虹 *
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
    save_image(data, path +'.pdf', nrow= nrow)
    
#-----------------------------------分割线-------------------------------------#

def t_SNE(X=None, y=None, 
          save_path = '../save/', 
          file_name = '1.pdf',
          color_cmap = 'nipy_spectral',
          show_info = True,
          save_svg = True):
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
    plt.savefig(save_path + file_name, bbox_inches = 'tight', format = 'pdf')
    if save_svg:
        plt.savefig(save_path + '/' + file_name[:file_name.rfind('.')] + '.svg', bbox_inches='tight')
    plt.close(fig)

#-----------------------------------分割线-------------------------------------#
def _s(s):
    if s is None:
        return s
    elif '^' not in s and '_' not in s and '\;' not in s:
        return s
    else:
        # repr: 变成 string （与 eval 功能相反）
        # s = eval(repr(s).replace(' ', '\;'))
        s = s.replace(' ', '\;')
        return '$'+s+'$'

class Plot_Curve():
    '''
        plot: https://blog.csdn.net/dss_dssssd/article/details/84430024
    ''' 
    def __init__(self, 
                 y,                            # 主轴 [subnumber, n_point, n_y] 
                 y_twin = None,                # 副轴 [subnumber, n_point, n_y] 
                 model_name = 'model',              # 模型名
                 add_info = '',                # 模型后缀
                 curve_name = 'curve',              # 曲线名
                 figsize = [32, 18],           # 画布大小
                 style = 'classic',            # 绘制风格
                 color_cmap = _color_bar[3],   # 颜色条
                 given_n_row = None            # 给定的子视图行数
                 ):
        # 用于绘图的数据
        self.y, self.y_twin = y, y_twin
        # 各轴绘制的线条数 n_y, n_ytwin
        self.n_ytwin = 0
        if type(y) == list: 
            # subnumber 是子视图个数（当数据为list时）
            self.subnumber = len(y)
            try:
                self.n_y = y[0].shape[1]
            except:
                self.n_y = 1
            if y_twin is not None: self.n_ytwin = y_twin[0].shape[1]
        else:
            self.subnumber = 1
            self.n_y = y.shape[1]
            if y_twin is not None: self.n_ytwin = y_twin.shape[1]
            self.y, self.y_twin = [self.y], [self.y_twin]
            
        self._markersize = int(24/self.n_y)
        if y_twin is not None: self._markersize = int(self._markersize+2)
        
        # 子视窗横纵排数
        self.n_row, self.n_col = 1, 1
        self.last_row_loc_add = 0
        if self.subnumber > 1:
            self.n_row, self.n_col = _get_subplot_size(self.subnumber, given_n_row)
            if np.mod(self.n_row * self.n_col - self.subnumber, 2) == 0:
                self.last_row_loc_add = int((self.n_row * self.n_col - self.subnumber) / 2)
        
        # 初始化画布
        self.figsize = [figsize[0]* self.n_row, figsize[1]* self.n_col]
        self.font_enlarge = min(self.n_row, self.n_col)
        self.fig = plt.figure(figsize=self.figsize)
        
        # 绘图风格
        if type(style) == int:
            style = plt.style.available[style]
        self.style = style
        plt.style.use(style)
        self.model_name, self.add_info, self.curve_name = model_name, add_info, curve_name
        
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
    
    def _distribute(self, x, add_id = False, to_list = True):
        if type(x) == tuple: a, b = x
        else: a, b = x, None
        y_list, y_n_list = [a, b], [self.n_y, self.n_ytwin]
        for i in range(len(y_list)):
            y, n = y_list[i], y_n_list[i]
            if y is None: continue
            if add_id and type(y) != list:
                name = []
                for k in range(n):
                    name.append(y + '-' + str(k+1)) 
                print('>??', name, a, b)
                if i == 0: a = name
                else: b = name
        if to_list:
            if type(a) != list: a = list([a])
            if type(b) != list: b = list([b])
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
             legend = ([''], ['']),
             linestyle = ('-', '-'),
             linewidth = (4, 4),
             marker = ('o', 'o'),
             markersize = (8, 8),
             markersetting = ['none', 0],      # mec, mew
             hori_line = None,
             switch_p = None,
             title = None,
             axis_yname = (None, None),
             axis_xname = None,
             yscale = None,
             fontsize = [54, 42, 40, 24, 38],  # title, ax_label, ax_ticks, legend, text
             legend_loc = [1, 4],
             ax_scale = (None, None),
             add_text = False,
             save_svg = True,
             zoom = [False, True],
             if_tight_layout = True,
             frameon = True,                   # legend 边框
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
        yscale, yscale_t = self._distribute(yscale, to_list = False)
        self.markersetting = markersetting
        if type(switch_p)!= list: switch_p = [switch_p]
        for sub_id in range(self.subnumber):
            sub_loc = sub_id + 1
            if (sub_id + 1) > self.n_col * (self.n_row - 1) and self.last_row_loc_add > 0:
                sub_loc += self.last_row_loc_add
            ax = self.fig.add_subplot(self.n_row, self.n_col, sub_loc)
            # plot y
            self.plots = []
            Y = self.y[sub_id]
            for i in range(self.n_y):
                if self.n_y == 1: y = Y
                else: y = Y[:,i]
                self.plot_in_fig(ax,
                                 y, 
                                 self.colors[np.mod(i, len(self.colors))], 
                                 linestyle[np.mod(i, len(linestyle))], 
                                 linewidth[np.mod(i, len(linewidth))],
                                 marker[np.mod(i, len(marker))],
                                 markersize[np.mod(i, len(markersize))],
                                 legend[np.mod(i, len(legend))],
                                 switch_p[np.mod(i, len(switch_p))]
                                 )
            # plot max line
            if hori_line is not None:
                if hori_line == 'max': 
                    max_y = [y.max()] * y.shape[0]
                    self.plot_in_fig(ax, max_y, 'black', '--', 1, 0, None)
                else: 
                    max_y = [hori_line] * y.shape[0]
                    self.plot_in_fig(ax, max_y, 'black', '--', 6, 0, legend = 'Threshold')
            
            self.y_plots = self.plots
            
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
            # yscale: 'log', 'logit'
            if yscale is not None:
                plt.yscale(yscale)
            
            # limit range
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xmin, xmax = 1,  Y.shape[0] + 1
                if zoom[0]:
                    plt.xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.02)
                else:
                    plt.xlim(xmin -1, xmax -1)
                if zoom[1]:
                    loc = np.where(Y == Y)
                    ymin, ymax = np.min(Y[loc]), np.max(Y[loc])
                    ax.set_ylim(ymin - (ymax - ymin) * 0.02, ymax + (ymax - ymin) * 0.02)
            
            if self.n_ytwin > 0:
                self.plots = []
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
                
                self.ytwin_plots = self.plots
                
                # set y_twin's label
                if axis_yname_t is not None:
                    ax_twinx.set_ylabel(_s(axis_yname_t[np.mod(sub_id, len(axis_yname_t))]),fontsize=fontsize[1])
            
                plt.yticks(fontsize=fontsize[2])
                if ax_scale[1] is not None: plt.yscale(ax_scale[1])
                # yscale: 'log', 'logit'
                if yscale_t is not None:
                    plt.yscale(yscale_t)
                    
                # limit range
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    loc = np.where(Y_Twin == Y_Twin)
                    ymin, ymax = np.min(Y_Twin[loc]), np.max(Y_Twin[loc])
                    ax_twinx.set_ylim(ymin - (ymax - ymin) * 0.02, ymax + (ymax - ymin) * 0.02)
           
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # set axis y's legend 
                if legend is not None: 
                    y_legends = [plot.get_label() for plot in self.y_plots]
                    lgd = ax.legend(self.y_plots, y_legends, loc = legend_loc[0], frameon = frameon,\
                                    ncol= int(np.sqrt(self.n_y)+0.5), fontsize=fontsize[3])
                    lgd.get_frame().set_alpha(0.5)  
                    # frame.set_facecolor('none')
                # set y_twin's legend
                if legend_t is not None and hasattr(self, 'ytwin_plots'):
                    ytwin_legends = [plot.get_label() for plot in self.ytwin_plots]
                    lgd = ax_twinx.legend(self.ytwin_plots, ytwin_legends, loc = legend_loc[1], frameon = frameon,\
                                          ncol= int(np.sqrt(self.n_ytwin)+0.5), fontsize=fontsize[3])
                    lgd.get_frame().set_alpha(0.5)    
                
            # plot text
            if add_text:
                x = range(1, 1 + Y.shape[0])
                for i in range(self.n_y):
                    y = Y[:,i]
                    for a, b in zip(x, y):
                        plt.text(a, b+0.05, '%.2f' % b, ha='center', va='bottom', fontsize=fontsize[4], color = 'b')
            
        # 子图间留间隙
        if if_tight_layout:
            plt.tight_layout()
            
        # save
        save_path, file_name = '../save/'+ self.model_name + self.add_info, self.curve_name +'.pdf'
        if not os.path.exists(save_path): os.makedirs(save_path)
        plt.savefig(save_path + '/' + file_name, bbox_inches='tight')
        #plt.show()
        if save_svg:
            plt.savefig(save_path + '/' + file_name[:file_name.rfind('.')] + '.svg', bbox_inches='tight')
        plt.close(self.fig)
        print('Plot {}\{}'.format(self.model_name + self.add_info, file_name[:file_name.rfind('.')]))
    
    def plot_in_fig(self, ax, y, color, linestyle, linewidth, marker, markersize = 0, legend = None, switch_p = None):
        dafault = {'color': color, 'linestyle': linestyle, 'linewidth': linewidth,
                   'marker': marker, 'markersize': markersize, 'label': _s(legend), 'mec': 'none', 'mew': 0}
        delete_keys = []
        for key, value in dafault.items():
            if key == 'markersize' and value == 0: dafault[key] = self._markersize
            if value is None: delete_keys.append(key)
        for key in delete_keys: 
            del dafault[key]
        
        if type(y) == list: y = np.array(y)
        x = range(1, 1 + y.shape[0])
        if linestyle == '':
            # scatter 
            del dafault['linestyle']
            del dafault['markersize']
            del dafault['mec']
            del dafault['mew']
            ax.scatter(x, y, **dafault)
        else:
            if dafault['linestyle'] == '--':
                dafault['dashes'] = (25, 18)
            # plot    
            if switch_p is None:
                self.plots += ax.plot(x, y, **dafault)
            # fd plot
            else:
                dafault['color'] = 'b'
                dafault['label'] = _s('Normal')
                start = 0
                for k in range(switch_p.shape[0]):
                    p = switch_p[k,0]
                    x = np.arange(start, p) + 1
                    self.plots += ax.plot(x, y[start: p], **dafault)
                    if dafault['color'] == 'b':
                        dafault['color'] = 'r'
                        if k == 0: dafault['label'] = _s('Faulty')
                        else: dafault['label'] = None
                    else:
                        dafault['color'] = 'b'
                        dafault['label'] = None
                    start = p
                x = np.arange(start, y.shape[0]) + 1
                self.plots += ax.plot(x, y[start:], **dafault)
        
def _concatenate(x_list):
    if len(x_list) == 0: return None
    y_list = []
    for x in x_list:
        y_list.append(np.array(x).reshape(-1,1)) 
    return np.concatenate(y_list, axis = 1)

# plot loss & acc curve / rmse & R2 curve / pred & real curve
def loss_curve(train_df, add_var_names, name, add_info):
    losses = [train_df['loss']]
    legends = ['Loss']
    if add_var_names is not None:
        for var_name in add_var_names:
            losses += [train_df[var_name]]
            var_name = var_name.replace("_", " ");
            legends += [var_name.capitalize()]
    data = _concatenate(losses)
    Plot_Curve(data, 
               None,
               name,
               add_info,
               'loss['+str(len(legends))+']'
               ).plot(axis_yname = 'Loss',
                      axis_xname = 'Epochs',
                      legend = legends,
                      yscale = 'log',                   # 'log', 'logit'
                      legend_loc = ('upper right',),
                      fontsize = [54, 45, 38, 30, 38]
                      )

def loss_acc_curve(train_df, test_df, name, add_info):
    train_loss, train_acc = train_df['loss'], train_df['accuracy']
    test_loss, test_acc = test_df['loss'], test_df['accuracy']
    data = _concatenate([train_acc, test_acc])
    data_twin = _concatenate([train_loss, test_loss])
    Plot_Curve(data, 
               data_twin,
               name,
               add_info,
               'loss_acc'
               ).plot(axis_yname = ('Average\;\;FDR\;(\%)', 'Loss'),
                      axis_xname = 'Epochs',
                      legend = (['Training\;\;FDR', 'Testing\;\;FDR'], 
                                ['Training\;\;loss', 'Testing\;\;loss'])
                      )

def rmse_R2_curve(train_df, test_df, name, add_info):
    train_rmse, train_R2 = train_df['rmse'], train_df['R2']
    test_rmse, test_R2 = test_df['rmse'], test_df['R2']
    data = _concatenate([train_rmse, test_rmse])
    data_twin = _concatenate([train_R2, test_R2])
    Plot_Curve(data, 
               data_twin,
               name,
               add_info,
               'rmse_R2'
               ).plot(axis_yname = ('RMSE', 'R^{2}'),
                      axis_xname = 'Epochs',
                      legend = (['Training\;\;RMSE', 'Testing\;\;RMSE'], 
                                ['Training\;\;R^{2}', 'Testing\;\;R^{2}'])
                      )

def rmse_mape_curve(train_df, name, add_info):
    test_rmse, test_mape = train_df['rmse'], train_df['mape']
    data, data_twin = np.array(test_rmse).reshape(-1,1),\
        np.array(test_mape).reshape(-1,1)
    Plot_Curve(data, 
               data_twin,
               name,
               add_info,
               'loss_mape',
               color_cmap = ['b','r'],
               ).plot(axis_yname = ('RMSE', 'MAPE\;(\%)'),
                      axis_xname = 'Epochs',
                      legend = (['Testing\;\;RMSE'], 
                                ['Testing\;\;MAPE']),
                      markersize = (8, 8)
                      )
    
def pred_real_curve(pred, real, name, add_info, task = 'prd'):
    data = _concatenate([pred, real])
    Plot_Curve(data, 
               None,
               name,
               add_info,
               'pred_real',
               color_cmap = ['b','r'],
               ).plot(axis_yname = 'Prediction\;\;result',
                      axis_xname = 'Samples',
                      legend = ['pred', 'real'],
                      linestyle = ['', '-']
                      )

def var_impu_curve(X, Y, NAN, missing_var_id, name, add_info):
    data, title = [], []
    for i, index in enumerate(missing_var_id):
        title.append('Variable\;\;{}'.format(index + 1))
        x, y, nan = X[:,index], Y[:,index], NAN[:,index]
        loc = np.where(nan == 1)
        _x, _y = x[loc], y[loc]
        order = np.argsort(_y)
        _x, _y = _x[order], _y[order]
        data.append(np.concatenate([_x.reshape(-1,1), _y.reshape(-1,1)], 1))
    # data 是 list, 其中 data[i] = array(X = pred, Y = real) 为 dx * 2 的矩阵
    Plot_Curve(data, 
               None,
               name,
               add_info,
               'var_impu',
               color_cmap = ['b','r'],
               figsize = [45, 16]
               ).plot(axis_yname = '$Values$',
                      axis_xname = 'Incomplete\;\;samples',
                      title = title,
                      fontsize = [54, 45, 38, 35, 38], # title, ax_label, ax_ticks, legend, text
                      legend_loc = ('upper left',),
                      # legend_loc = ('lower right',),
                      legend = ['imputed', 'real'],
                      linestyle = ['', '-'],
                      linewidth = 6,
                      markersize = [4, 2],
                      if_tight_layout = False
                      )
    
def stat_curve(stat_list, switch_p_list, threshold, name, 
             add_info, label_name, plot_p_list = None, subplot = False):
    if subplot:
        title = []
        for i in range(len(switch_p_list)):
            title.append('Fault {:02d}'.format(i + 1))
        Plot_Curve(stat_list,
                   None,
                   name,
                   add_info,
                   't2_subplot',
                   figsize = [40, 13]
                   ).plot(axis_yname = 'T^2',
                          axis_xname = 'Samples',
                          hori_line = threshold,
                          switch_p = switch_p_list,
                          title = title,
                          linewidth = 4,
                          # 'log', 'logit'
                          yscale = 'log',         
                          legend_loc = ('upper left',),
                          # title, ax_label, ax_ticks, legend, text
                          fontsize = [54, 45, 38, 34, 38],
                          marker = None,
                          markersize = None
                          )
    else:
        if plot_p_list is not None:
            stats, switch_ps, label_names = [], [], []
            # for each fault
            for k in range(len(plot_p_list)):
                stat, switch_p = stat_list[k].reshape(-1,1), switch_p_list[k].reshape(-1,1)
                plot_ps = plot_p_list[k]
                start = 0
                temp_p = 0
                plot_switch_p = []
                # for each plot
                for p, plot_p in enumerate( list(plot_ps)):
                    label_names.append(label_name[k] + '-' + str(p+1))
                    stats.append(stat[start:plot_p])
                    while temp_p < switch_p.shape[0] and switch_p[temp_p] < plot_p:
                        if p == 0: 
                            plot_switch_p.append(switch_p[temp_p])
                        else:
                            plot_switch_p.append(switch_p[temp_p] - plot_ps[p-1])
                        temp_p += 1
                    switch_ps.append(np.array(plot_switch_p))
                    plot_switch_p = []
                    start = p
            stat_list, switch_p_list, label_name = stats, switch_ps, label_names
            
        for i in range(len(stat_list)):
            stat, switch_p = stat_list[i].reshape(-1,1), switch_p_list[i].reshape(-1,1)
            # stat 是 array
            Plot_Curve(stat,
                       None,
                       name,
                       add_info,
                       't2' + label_name[i],
                       figsize = [40, 16]
                       ).plot(axis_yname = 'T^2',
                              axis_xname = 'Samples',
                              hori_line = threshold,
                              switch_p = switch_p,
                              linewidth = 4,
                              # 'log', 'logit'
                              yscale = 'log',         
                              legend_loc = ('upper left',),
                              # title, ax_label, ax_ticks, legend, text
                              fontsize = (np.array([54, 40, 38, 34, 38])*2).astype(int),
                              marker = None,
                              markersize = None
                              # frameon = False
                              )
                              
def read_stat(file_path):
    stat_list = pd.read_excel(file_path, sheet_name = 'stat_list').values
    switch_p_list = pd.read_excel(file_path, sheet_name = 'switch_p_list').values
    fd_thrd = int(pd.read_excel(file_path, sheet_name = 'fd_thrd').values)
    labels = ['Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
              'Fault 08', 'Fault 09', 'Fault 10']
    stat_curve(stat_list, switch_p_list, fd_thrd, 'VAE', 
               '_fd_CSTR', labels, subplot = False)
                      
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
                         color_cmap = _color_bar[0],
                         legend_loc = (1,4),
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
            model_name[-1] = model_name[-1].replace('_','$-$')
            
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
def _check_label_name(label, N):
    _labels = []
    if type(label)!= list: label = [label]
    for i in range(N):
        if label[0] is None or len(label) < N:
            _labels.append('Category ' + str(i + 1))
        else:
            _labels.append(label[i])
    return _labels

def category_distribution(prd_cnt, label = None, name = '', add_info = '',
                          text = 'cnt', diag_cl = True, plot_size = None, 
                          fontsize = (18,14), 
                          save_svg = True):
    import matplotlib.pyplot as plt
    plt.style.use('default')
    print('Plot {}[{}] pred_distrib with style: default'.format(add_info, name))
    
    if label is None:
        axis_x, axis_y = np.arange(prd_cnt.shape[1]), np.arange(prd_cnt.shape[0])
    elif type(label) == list:
        axis_x, axis_y = label.copy(), label.copy()
        for i in range(len(label)): axis_x[i] = _s(axis_x[i] + '_r')
        for i in range(len(label)): axis_y[i] = _s(axis_y[i] + '_p')
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
    ticksize = size/max_size*fontsize[0]   # label
    textsize = size/max_size*fontsize[1]   # text
    
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
            
            ax.text(j, i, t, ha="center", va="center", color=cl, fontsize=textsize)
    
    save_path, file_name = '../save/'+ name + add_info, add_info+'['+name+']'+' pred_distrib.pdf'
    if not os.path.exists(save_path): os.makedirs(save_path)
    plt.savefig(save_path + '/' + file_name, bbox_inches='tight')
    if save_svg:
        plt.savefig(save_path + '/' + file_name[:file_name.rfind('.')] + '.svg', bbox_inches='tight')
    plt.close(fig)

''' legend location
    best            0
    upper right     1
    upper left      2
    lower left      3
    lower right     4
    right           5
    center left     6
    center right    7
    lower center    8
    upper center    9
    center          10 '''

if __name__ == '__main__':
    # read_stat('../save/VAE_fd_CSTR/[VAE] FD_result.xlsx')
    
    # compare_curve
    # epoch_comparing_curve('../save/【HY】 CG-SAE', ('test_accuracy','train_loss'), 
    #                       color_cmap = _color_bar[3],
    #                       # legend_loc = ((0.414, 0.711),(0.354, 0.09)), # TE
    #                       legend_loc = ((0.414, 0.751),(0.354, 0.16)),
    #                       fontsize = [54, 42, 40, 30, 38],
    #                       linewidth = (4, 4),
    #                       marker = ('o', 'o'),
    #                       markersize = (8, 8))
    
    # compare_curve (AM-DAE)
    epoch_comparing_curve('../save/Impu_curve', ('train_rmse',''), # train_rmse, train_mape
                          color_cmap = _color_bar[5],
                          # title, ax_label, ax_ticks, legend, text
                          fontsize = [54, 42, 40, 38, 38],
                          linewidth = (4, 4),
                          marker = (['o','^','s','*','d'], ['o','^','s','*','d']),
                          markersize = (18, 18),
                          yscale = 'log')
    
    # plot
    # data = pd.read_csv('../save/【Beta】 CG-SAE/Beta-FDR.csv').values
    # print(data)
    # Plot_Curve(data,
    #            color_cmap = _color_bar[3],
    #            model_name = '【Beta】 CG-SAE',
    #            curve_name = 'Beta-FDR', 
    #            ).plot(axis_yname = ('Test\;\;average\;\;FDR\;(\%)',None),
    #                   axis_xname = r'Loss\;\;coefficient\;\;\beta',
    #                   legend_loc = (4,),
    #                   fontsize = [54, 42, 40, 30, 38],
    #                   legend = (['Complete\;data\;set', '60\%\;label\;missing','70\%\;label\;missing'],None),
    #                   add_text = True
    #                   )
    
    # category_distribution
    # label = ['Normal', 'Fault 01', 'Fault 02', 'Fault 03','Fault 04', 'Fault 05', 'Fault 06', 'Fault 07', 
    #           'Fault 08']
    # cls_result = pd.read_excel('../save/CG-SAE-5 8t/[CG-SAE] result.xlsx', sheet_name = 'cls_result').values
    # cls_result = cls_result[:len(label),1:]
    # category_distribution(cls_result, label)
    
