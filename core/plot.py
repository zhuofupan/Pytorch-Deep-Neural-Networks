# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
    cmap: https://matplotlib.org/users/colormaps.html
'''

def polyline(y, x = None, legend = None, label = None, title = None, style = 'classic'):
    import matplotlib.pyplot as plt
    if type(style) == int:
        style = plt.style.available[style]
    print('Plot Polyline with style: {}'.format(style))
    if type(y) == str:
        y =  np.loadtxt(y,delimiter=',')
    plt.style.use(style)

    if data.ndim == 1:
        n = len(data)
        color = 'r'
    else:
        n = y.shape[0]
        m = y.shape[1]
        color = MinMaxScaler().fit_transform(range(m)).reshape(-1,)

    fig = plt.figure(figsize=[32,18])
    
    if x is None:
        x = range(1,n+1)

    ax1 = fig.add_subplot(111)
    '''
        plot: https://blog.csdn.net/dss_dssssd/article/details/84430024
    '''
    if data.ndim == 1:
        ax1.plot(x, y,color=color,marker='o',markersize=12,linestyle='-',linewidth=4,label=legend)
    else:
        for i in range(m):
            if type(legend)==str:
                lgd = legend+'-'+str(i+1)
            elif legend is not None:
                lgd = legend[np.mod(i,len(legend))]
            ax1.plot(x, y[:,i],color=color[i],marker='o',markersize=12,linestyle='-',linewidth=4,label=lgd)
    if title is not None:
        ax1.set_title(title,fontsize=36)
    if label[0] is not None:
        ax1.set_xlabel('$'+label[0]+'$',fontsize=48)
    if label[1] is not None:
        ax1.set_ylabel('$'+label[1]+'$',fontsize=48)
    if legend is not None:
        ax1.legend(loc = 1, ncol=int(np.sqrt(m)),fontsize=24)

    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    
    for a, b in zip(x, y):
        plt.text(a, b+0.05, '%.1f' % b, ha='center', va='bottom', fontsize=38, color = 'b')
    
    if not os.path.exists('../results/img'): os.makedirs('../results/img')
    plt.savefig('../results/img/polyline.png',bbox_inches='tight')
    plt.show()
    plt.close(fig)

if __name__ == '__main__':
    data = np.array([0.9418, 0.9455, 0.9417, 0.9437, 0.9468, 0.9482, 0.9515, 0.956, 0.9608, 0.9615])*100
    x = np.linspace(0.1,1.0,10)
    polyline(data, x,label = ['Coefficient\;\;\\beta\;\;in\;\;loss\;\;function','Test\;\;average\;\;FDR\;(\%)'], style = 1)