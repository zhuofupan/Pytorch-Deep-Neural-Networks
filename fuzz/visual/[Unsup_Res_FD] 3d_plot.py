# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:41:40 2021

@author: Fuzz4
"""
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

fig = plt.figure(figsize=[24,24])  #定义新的三维坐标轴
ax3 = plt.axes(projection='3d')

def Deff(error, delta):
    lg = np.log(2/delta)
    # Two-sided Chernoff bound
    N1 = lg/(2*error**2)
    N1 = np.ceil(N1).astype(int)
    # Freedman inequalities with [ΔM], ΔM >= -lb, lb > 0
    # exp(-(Nε)^2/(2*(v2+Nεb))) <= δ
    lb = 0.005
    v2 = 60.6903
    N2 = (lg*lb + np.sqrt((lg*lb)**2 + 2*v2*lg)) / error
    N2 = np.ceil(N2).astype(int)
    return N1/N2

#定义三维数据
xx = np.arange(0.001,0.010, 0.0001)
yy = np.arange(0.001,0.010, 0.0001)
X, Y = np.meshgrid(xx, yy)
Z = Deff(X,Y)

#作图
# ax3.plot_surface(X,Y,Z,cmap='rainbow')
ax3.plot_surface(X,Y,Z,rstride = 1, cstride = 1,cmap='rainbow',linewidth=0)
# ax3.contour(X,Y,Z, zdim='z',offset=-2, cmap='rainbow')   #等高线图，要设置offset，为Z的最小值
ax3.tick_params('x', labelsize = 48, labelrotation = -15)
ax3.tick_params('y', labelsize = 48, labelrotation = 40)
ax3.tick_params('z', labelsize = 48)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(0.002))
ax3.yaxis.set_major_locator(ticker.MultipleLocator(0.002))
ax3.set_xticklabels(['0','2','4','6','8','10'])
ax3.set_yticklabels(['0','2','4','6','8','10'])
# ax3.set_zticklabels(['0','5','10','15','20','25','30','35'])
ax3.set_xlabel('$ϵ \;( × 10^{-3})$', fontsize = 68, labelpad = 48)  # 设置x坐标轴
ax3.set_ylabel('$δ  \;( × 10^{-3})$', fontsize = 68, labelpad = 48)  # 设置y坐标轴
ax3.set_zlabel('$N_{min,C}/N_{min,F}$', fontsize = 68, labelpad = 48)  # 设置z坐标轴
plt.tight_layout()

if not os.path.exists('img'): os.makedirs('img')
plt.savefig( 'img/ratio_of_N_3d.pdf', bbox_inches='tight')
plt.show()
plt.close(fig)
