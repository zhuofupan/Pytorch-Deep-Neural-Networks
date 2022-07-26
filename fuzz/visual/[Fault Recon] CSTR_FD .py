# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 00:33:38 2022

@author: Fuzz4
"""

from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

_color_bar = ['nipy_spectral',  # 0 广色域-多色
              'jet',            # 1 广色域-平滑
              'gist_ncar',      # 2 广色域-中间艳丽
              'gist_rainbow',   # 3 鲜艳-七彩虹 *
              'hsv',            # 4 鲜艳-七彩虹
              'rainbow',        # 5 鲜艳-反七彩虹 *
              'cool', 'Wistia'  # 6 冷，7 暖
              'spring','summer','autumn','winter', # 8 春，9 夏，10 秋，11 冬
              ]

def plot_fidn():
    # 变量的顺序乱了，故障的顺序没乱
    v_index = [0,1,2,3,4,8,5,6,9,7]
    file_path = '../save/VAE_FIdN_CSTR/[VAE_FIdN] Monitoring_Indicators.xlsx'
    est_fs = []
    scaler = pd.read_csv('../data/FD_CSTR/scaler/[st] scaler_x.csv', header = None).values
    scaler = scaler[v_index]
    n_f = 10
    for i in range(n_f):
        est_fs.append( pd.read_excel(file_path, sheet_name=i).values )
    
    n, m = est_fs[0].shape[0], est_fs[0].shape[1]
    
    legend = ['$\hat f_{C_i}$', '$\hat f_{T_i}$', '$\hat f_{T_{ci}}$', '$\hat f_{C_i^{(s)}}$',\
              '$\hat f_{T_i^{(s)}}$', '$\hat f_{T_{ci}^{(s)}}$', '$\hat f_{C^{(s)}}$',\
              '$\hat f_{T^{(s)}}$', '$\hat f_{T_c^{(s)}}$', '$\hat f_{Q_c^{(s)}}$']
    legend_plus = ['', '', '', '$f_{C_i^{(s)}}$',\
                   '$f_{T_i^{(s)}}$', '$f_{T_{ci}^{(s)}}$', '$f_{C^{(s)}}$',\
                   '$f_{T^{(s)}}$', '$f_{T_c^{(s)}}$', '$f_{Q_c^{(s)}}$']
    
    f_mag = [0,0,0,0.001,0.05,0.05,0.001,0.05,0.05,-0.1]
    
    c4= ['lime','darkorange','fuchsia','aqua']
    
    RMSE, RMSE_F = np.zeros(8), np.zeros(8)
    for c in range(n_f):
        print('Plot Recon_{}.pdf'.format(c+1))
        x = np.arange(1, 1202)
        color=plt.get_cmap('Blues')(np.linspace(0.05, 0.95, m))
        # 预测的故障信号
        data = est_fs[c]
        # 调整变量为对应的顺序
        data = data[:,v_index]

        # data = est_fs[c]
        fig = plt.figure(figsize=[26,15])
        ax = fig.add_subplot(111)
        for j in range(m):
            if c > 2 and c == j: continue
            if c <=2 and j>m-5:
                ax.plot(x, data[:,j], linewidth = 3, c = c4[m-j-1], label = legend[j])
            else:
                ax.plot(x, data[:,j], linewidth = 2, c = color[j], label = legend[j])
        # 加性故障
        if c > 2:
            # pred fault signal
            ax.plot(x, data[:,c], linewidth = 3, c = 'b', label = legend[c])
            # real fault signal
            f = np.zeros((n,))
            f_signal = np.linspace(f_mag[c], f_mag[c]*1000, 1000)
            # f[201:] = f_signal
            # 生成真实故障信号
            f[201:] = f_signal/np.sqrt( scaler[c,1] )
            
            RMSE_F[c - 3] = np.sum((data[:, c] - f)**2)
            RMSE_F[-1] += RMSE_F[c - 3]
            RMSE_F[c - 3] = np.sqrt( RMSE_F[c - 3]/1201 )
            
            RMSE[c - 3] = np.sum(data[:, :c] **2) + np.sum(data[:, c+1:] **2)
            RMSE[-1] += RMSE[c - 3]
            RMSE[c - 3] = np.sqrt( RMSE[c - 3]/1201 )
             
            ax.plot(x, f, linewidth = 3, c = 'r', label = legend_plus[c])
        
        if c > 2:
            lgd = ax.legend(loc = 'upper right',  fontsize=43)
        else:
            lgd = ax.legend(loc = 'upper right',  fontsize=46)
        lgd.get_frame().set_alpha(0.5)
        ax.tick_params('x', labelsize = 48)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.set_xlabel('Samples', fontsize = 58)  # 设置x坐标轴
        ax.tick_params('y', labelsize = 48)
        ax.set_ylabel('Fault signal', fontsize = 58)  # 设置y坐标轴
        plt.tight_layout()
        plt.savefig('../save/VAE_FIdN_CSTR/Recon_{}.pdf'.format(c+1), bbox_inches='tight')
        plt.savefig('../save/VAE_FIdN_CSTR/Recon_{}.svg'.format(c+1), bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    RMSE_F[-1] = np.sqrt( RMSE_F[-1]/(1201 * 7) )
    RMSE[-1] = np.sqrt( RMSE[-1]/(1201 * 7) )
    print('RMSE_F:', np.round(RMSE_F, 4))
    print('RMSE:  ', np.round(RMSE, 4))
    
if __name__ == '__main__':
    plot_fidn()

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


# fig = plt.figure(figsize=[24,15])


# ax = fig.add_subplot(111)
# # ax.plot(yy, Z)

# color=plt.get_cmap('Blues')(np.linspace(0.3, 1, 5))
# for i in range(5):
#     legend = '$δ = {}$'.format(yy[i])
#     ax.plot(xx, Z.T[:,i], linewidth = 3, c = color[i], label = legend)



# ax.set_xlim(0.0001, 0.01)
# ax.set_ylim(0, 122)

# lgd = ax.legend(loc = 'upper right', fontsize=58)
# lgd.get_frame().set_alpha(0.5)

# plt.tight_layout()
# plt.savefig( '../save/VAE_fd_CSTR/N_min_C_D.pdf', bbox_inches='tight')
# plt.show()
# plt.close(fig)
