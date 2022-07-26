# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:41:40 2021

@author: Fuzz4
"""
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_color_bar = ['nipy_spectral',  # 0 广色域-多色
              'jet',            # 1 广色域-平滑
              'gist_ncar',      # 2 广色域-中间艳丽
              'gist_rainbow',   # 3 鲜艳-七彩虹 *
              'hsv',            # 4 鲜艳-七彩虹
              'rainbow',        # 5 鲜艳-反七彩虹 *
              'cool', 'Wistia'  # 6 冷，7 暖
              'spring','summer','autumn','winter', # 8 春，9 夏，10 秋，11 冬
              ]

def Deff(error, delta, v2 = 59.75):
    lg = np.log(2/delta)
    # One-sided Chernoff bound
    N_C = lg/(2*error**2)
    N_C = np.ceil(N_C).astype(int)
    N_D = np.sqrt(2*v2*lg) / error
    N_D = np.ceil(N_D).astype(int)
    return N_C/N_D

#定义三维数据
xx = np.arange(0.001,0.010, 0.0001)
yy = np.arange(0.002,0.011, 0.002)
# z = xx; xx = yy; yy = z
X, Y = np.meshgrid(xx, yy)
Z = Deff(X,Y)

print(Z.shape)

#作图
# ax3.plot_surface(X,Y,Z,cmap='rainbow')
# ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow',linewidth=0)
# # ax3.contour(X,Y,Z, zdim='z',offset=-2, cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
# ax3.tick_params('x', labelsize = 48, labelrotation = -15)
# ax3.tick_params('y', labelsize = 48, labelrotation = 40)
# ax3.tick_params('z', labelsize = 48)
# ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.002))
# ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.002))
# ax3.set_xticklabels(['0','2','4','6','8','10'])
# ax3.set_yticklabels(['0','2','4','6','8','10'])
# # ax3.set_zticklabels(['0','5','10','15','20','25','30','35'])
# ax3.set_xlabel('$ϵ \;( × 10^{-3})$', fontsize = 68, labelpad = 48)  # 设置x坐标轴
# ax3.set_ylabel('$δ  \;( × 10^{-3})$', fontsize = 68, labelpad = 48)  # 设置y坐标轴
# ax3.set_zlabel('The ratio of $N_{min}$', fontsize = 68, labelpad = 48)  # 设置z坐标轴

fig = plt.figure(figsize=[24,15])  #定义新的三维坐标轴


ax = fig.add_subplot(111)
# ax.plot(yy, Z)

color=plt.get_cmap('Blues')(np.linspace(0.3, 1, 5))
for i in range(5):
    legend = '$δ = {}$'.format(yy[i])
    ax.plot(xx, Z.T[:,i], linewidth = 3, c = color[i], label = legend)

ax.tick_params('x', labelsize = 58)
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.002))
ax.set_xlabel('$ϵ \;( × 10^{-3})$', fontsize = 68)  # 设置x坐标轴
ax.set_xticklabels(['0','2','4','6','8','10'])
ax.tick_params('y', labelsize = 58)
ax.set_ylabel('$N_{min,C}/N_{min,D}$', fontsize = 68)  # 设置y坐标轴

ax.set_xlim(0.0001, 0.01)
ax.set_ylim(0, 122)

lgd = ax.legend(loc = 'upper right', fontsize=58)
lgd.get_frame().set_alpha(0.5)

plt.tight_layout()
plt.savefig( '../save/VAE_fd_CSTR/N_min_C_D.pdf', bbox_inches='tight')
plt.show()
plt.close(fig)
