# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.colors as cls
import matplotlib.cm as cmx
from sklearn.preprocessing import MinMaxScaler
if not os.path.exists('../save/plot'): os.makedirs('../save/plot')

def _get_colors(N):
    '''
        cmap: https://matplotlib.org/users/colormaps.html
    '''
    import matplotlib.pyplot as plt
    values = range(N)
    # jet, gist_ncar, nipy_spectral, hsv, rainbow
    if N <= 10:
        cmap = plt.get_cmap('rainbow')
    else:
        cmap = plt.get_cmap('nipy_spectral')
    cNorm  = cls.Normalize(vmin=0, vmax=values[-1])
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
    
    colors = []
    for idx in range(N):
        colorVal = scalarMap.to_rgba(values[idx])
        colors.append(colorVal)
    return colors

#-----------------------------------分割线-------------------------------------#

def t_SNE(X=None, y=None, save_path=None):
    import matplotlib.pyplot as plt
    from sklearn import manifold
    from time import time
    from matplotlib.ticker import NullFormatter
    
    t0 = time()
    # preprocess
    X = MinMaxScaler().fit_transform(X)
    if len(y.shape)>1 and y.shape[1]>1:
        y = np.array(np.argmax(y,axis=1).reshape(-1, 1),dtype=np.float32)
    
    # do transform
    X = manifold.TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)
    t1 = time()
    print("t-SNE cost time: {:.2f} sec".format(t1 - t0))
    
    # split
    datas = []
    n_class = int(np.max(y)) - int(np.min(y)) + 1
    colors = _get_colors(n_class)
    for i in range(n_class):
        datas.append([])
    for i in range(y.shape[0]):
        datas[int(y[i])].append(X[i].reshape(1,-1))
    
    plt.style.use('default')
    fig = plt.figure(figsize=[32,18])
    ax = fig.add_subplot(111)
    
    for i in range(len(datas)):
        data_X = np.concatenate(datas[i], axis = 0)
        colors[i] = np.array(colors[i]).reshape(1,-1)
        plt.scatter(data_X[:, 0], data_X[:, 1], label = str(i + 1),
                    cmap=plt.cm.Spectral,
                    c =  colors[i] 
                    )
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
        
    plt.legend(fontsize=28, loc = 1)
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.savefig(save_path,bbox_inches='tight')
    plt.close(fig)
    
    t2 = time()
    print("plot cost time: {:.2f} sec. img saved in {}".format(t2 - t1, save_path))

#-----------------------------------分割线-------------------------------------#
def _s(s):
    if s is not None: 
        s = eval(repr(s).replace(' ', '\;'))
        return '$'+s+'$'
    else: return s

def plot_curve(y, 
               y_twin = None, 
               label = None, 
               legend = None, 
               name = '',
               x = None, text = False, title = None, style = 'classic'):
    '''
        plot: https://blog.csdn.net/dss_dssssd/article/details/84430024
    ''' 
    
    import matplotlib.pyplot as plt
    if type(style) == int:
        style = plt.style.available[style]
    print('Plot {}_curve with style: {}'.format(name, style))
    plt.style.use(style)

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
    colors = _get_colors(m)
    if len(colors) == 1: colors = ['r']
    
    fig = plt.figure(figsize=[32,18])
    ax1 = fig.add_subplot(111)
    
    if type(legend)!=list: legend = [legend]
    
    if y.ndim == 1:
        ax1.plot(x, y, color=colors[0], marker='o', markersize=12, linestyle='-', linewidth=4, label=_s(legend[0]))
    else:
        for i in range(m1):
            if len(legend)==1:
                lgd = legend+'-'+str(i+1) 
            else:
                lgd = legend[np.mod(i, len(legend))]
            ax1.plot(x, y[:,i], color=colors[i], marker='o', markersize=12, linestyle='-', linewidth=4, label=_s(lgd))
    if title is not None:
        ax1.set_title(title,fontsize=36)
    if label is not None:
        ax1.set_xlabel(_s(label[0]),fontsize=48)
        ax1.set_ylabel(_s(label[1]),fontsize=48)

    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    
    if y_twin is not None:
        ax2 = ax1.twinx()
        if y_twin.ndim == 1:
            ax2.plot(x, y_twin, color=colors[m1], marker='o', markersize=12, linestyle='-', linewidth=4, label=_s(legend[m1]))
        else:
            for i in range(m2):
                lgd = legend[np.mod(m1 + i, len(legend))]
                ax2.plot(x, y_twin[:,i], color=colors[m1 + i], marker='o', markersize=12, linestyle='-', linewidth=4, label=_s(lgd))
        if label is not None:
            ax2.set_ylabel(_s(label[2]),fontsize=48)
        if legend[0] is not None:
            lgd2 = ax2.legend(loc = 1, ncol= 1, fontsize=24) 
            lgd2.get_frame().set_alpha(0.5)
            
        plt.yticks(fontsize=40)
        # plt.yscale('logit')
        
    if legend[0] is not None:
        lgd1 = ax1.legend(loc = 2, ncol= 1, fontsize=24)
        lgd1.get_frame().set_alpha(0.5)  
        # frame.set_facecolor('none')  
    
    if text:
        for a, b in zip(x, y):
            plt.text(a, b+0.05, '%.1f' % b, ha='center', va='bottom', fontsize=38, color = 'b')
    
    plt.savefig('../save/plot/'+name+'_curve.png',bbox_inches='tight')
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
               ['Train\;\;FDR', 'Test\;\;FDR', 'Train\;\;loss', 'Test\;\;loss'],
               '['+name+'] loss_acc'
               )

def rmse_R2_curve(train_df, test_df, name):
    train_rmse, train_R2 = train_df['rmse'], train_df['R2']
    test_rmse, test_R2 = test_df['rmse'], test_df['R2']
    data = _concatenate(train_rmse, test_rmse)
    data_twin = _concatenate(train_R2, test_R2)
    plot_curve(data, 
               data_twin,
               ['Epochs', 'RMSE', 'R^{2}'],
               ['Train\;\;RMSE', 'Test\;\;RMSE', 'Train\;\;R^{2}', 'Test\;\;R^{2}'],
               '['+name+'] rmse_R2'
               )    
    
def pred_real_curve(pred, real, name):
    data = _concatenate(pred, real)
    plot_curve(data,
               None,
               ['Sampels', 'Prediction\;\;result'],
               ['pred', 'label'],
               '['+name+'] pred_real'
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

def category_distribution(prd_cnt, label = None, name = ''):
    import matplotlib.pyplot as plt
    plt.style.use('default')
    print('Plot [{}] pred_distrib with style: default'.format(name))
    
    axis_x, axis_y = label.copy(), label.copy()
    for x in axis_x: x = _s(x + '_r')
    for y in axis_y: y = _s(y + '_p')
    
    prd_cnt = np.array(prd_cnt, dtype = np.int32)
    n_sample_cnts = np.sum(prd_cnt, axis = 0, dtype = np.int)
    prd_pro = prd_cnt / n_sample_cnts
    imshow = np.array(np.around(prd_pro*100,0), dtype = np.int32)
    
    #mat = np.round(mat,decimals=0)
    #mat = np.transpose(mat)
    
    size = 16
    ticksize = size/24*26
    fontsize = size/24*23
    
    fig =  plt.figure(figsize=[size,size])
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
    
    
    
    for i in range(len(axis_y)):
        for j in range(len(axis_x)):
            x = prd_cnt[i,j]
            p = prd_pro[i,j]
            if i == j:
                if p > 0.618:  cl = 'w'
                else:  cl = 'b'
            elif x == 0 : cl = 'black'
            else: cl = 'red'
            ax.text(j, i, x, ha="center", va="center", color=cl, fontsize=fontsize)
    
    plt.savefig('../save/plot/['+name+'] pred_distrib.png',bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    data = np.array([0.9418, 0.9455, 0.9417, 0.9437, 0.9468, 0.9482, 0.9515, 0.956, 0.9608, 0.9615])*100
    x = np.linspace(0.1,1.0,10)
    plot_curve(data, 
               None,
               ['Coefficient\;\;\\beta\;\;in\;\;loss\;\;function','Test\;\;average\;\;FDR\;(\%)'],
               x = x,
               text = True,
               style = 1
               )
