# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 16:21:05 2022

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
    file_path = '../save/GAN_FDI_CSTR_Y/[GAN_FDI] Monitoring_Indicators.xlsx'
    scaler = pd.read_csv('../data/CSTR/fi/scaler/[st] scaler_x.csv', header = None).values
    n_f = 8
    n_mf, n_af = 2, 6
    
    esti_f = []
    for i in range(n_f):
        # FDI = Res = x - x_g
        r = pd.read_excel(file_path, sheet_name=i).values
        # f = x - xn = sign(r)sqrt(abs(r))
        # 对于经过开方处理的故障信号
        # esti_f.append(np.sign(r) * np.sqrt(np.abs(r))) 
        # 对于未经开方处理的故障信号
        esti_f.append(r)
    
    n, m = esti_f[0].shape[0], esti_f[0].shape[1]
    f_mag = [0.001,0.05,0.05,0.001,0.05,0.05,-0.1]
    real_f = np.zeros((n,m))
    for v in range(len(f_mag)):
        f_signal = np.linspace(f_mag[v], f_mag[v]*1000, 1000)
        # 生成真实故障信号
        real_f[201:,v+3] = f_signal/np.sqrt( scaler[v+3,1] )
    
    # 变量标签
    # Ci, Ti, Tci, C, T, Tc, Qc
    legend = ['$\hat f_{C_i}$', '$\hat f_{T_i}$', '$\hat f_{T_{ci}}$', '$\hat f_{C_i^{(s)}}$',\
              '$\hat f_{T_i^{(s)}}$', '$\hat f_{T_{ci}^{(s)}}$', '$\hat f_{C^{(s)}}$',\
              '$\hat f_{T^{(s)}}$', '$\hat f_{T_c^{(s)}}$', '$\hat f_{Q_c^{(s)}}$']
    # 故障变量标签
    legend_plus = ['', '', '', '$f_{C_i^{(s)}}$',\
                   '$f_{T_i^{(s)}}$', '$f_{T_{ci}^{(s)}}$', '$f_{C^{(s)}}$',\
                   '$f_{T^{(s)}}$', '$f_{T_c^{(s)}}$', '$f_{Q_c^{(s)}}$']
    # 故障位置
    '''
        故障1：Catalyst Fault (6,7,8,9)
        故障2：HTC Fault (6,7,8,9)
        故障3：Ci Sensor Bias (3)
        故障4：Ti Sensor Bias (4)
        故障5：Tci Sensor Bias (5)
        故障6：C, T Sensor Bias (6,7)
        故障7：Tci, Tc Sensor Bias (5,8)
        故障8：Qc, T Sensor Bias (9,7)
    '''
    index = {'3':[3], '4':[4], '5':[5], '6':[6,7], '7':[5,8], '8':[9,7]}
    color4= ['lime','darkorange','fuchsia','aqua']
    blues= ['b','darkblue']
    reds = ['r', 'crimson']
    RMSE_N, RMSE_F, RMSE = np.zeros(n_af + 1), np.zeros(n_af + 1), np.zeros(n_af + 1)
    
    for c in range(n_f):
        print('Plot Recon_{}.pdf'.format(c+1))
        x = np.arange(1, 1202)
        color=plt.get_cmap('Blues')(np.linspace(0.05, 0.95, m))
        # 预测的故障信号
        data = esti_f[c]

        fig = plt.figure(figsize=[26,14])
        ax = fig.add_subplot(111)
        for v in range(m):
            if c >= n_mf and v in index[str(c+1)]: continue
            # 乘性的后4个变量用彩色绘制
            if c < n_mf and v>m-5:
                ax.plot(x, data[:,v], linewidth = 3, c = color4[m-v-1], label = legend[v])
            # 加性/乘性的其他变量用浅蓝色
            else:
                ax.plot(x, data[:,v], linewidth = 2, c = color[v], label = legend[v])
        # 加性故障
        if c >= n_mf:
            for k in range(len(index[str(c+1)])):
                # pred fault signal
                v = index[str(c+1)][k]
                ax.plot(x, data[:, v], linewidth = 3, c = blues[k], label = legend[v])
            for k in range(len(index[str(c+1)])):
                # real fault signal
                v = index[str(c+1)][k]
                ax.plot(x, real_f[:, v], linewidth = 3, c = reds[k], label = legend_plus[v])
            
            for v in range(m):
                # faulty variables
                if v in index[str(c+1)]:
                    RMSE_F[c - n_mf] += np.sum((data[:, v] - real_f[:, v])**2)
                # fault-free variables
                else:
                    RMSE_N[c - n_mf] += np.sum(data[:, v] **2)
            
            # 0~-2 是各类故障的， -1 是总的
            RMSE_F[-1] += RMSE_F[c - n_mf]
            RMSE_N[-1] += RMSE_N[c - n_mf]
            RMSE[-1] += RMSE_F[c - n_mf] + RMSE_N[c - n_mf]
            RMSE_F[c - n_mf] = np.sqrt( RMSE_F[c - n_mf]/1201 )
            RMSE_N[c - n_mf] = np.sqrt( RMSE_N[c - n_mf]/1201 )
            RMSE[c - n_mf] = np.sqrt( (RMSE_F[c - n_mf] + RMSE_N[c - n_mf])/1201 )
                
        if c >= n_mf and c <= 4:
            lgd = ax.legend(loc = 'upper left',  fontsize=39)
        elif c >= n_mf:
            lgd = ax.legend(loc = 'upper left',  fontsize=36)
        else:
            lgd = ax.legend(loc = 'upper left',  fontsize=42)
        lgd.get_frame().set_alpha(0.5)
        ax.tick_params('x', labelsize = 48)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        
        ax.set_xlabel('Samples', fontsize = 58)  # 设置x坐标轴
        ax.tick_params('y', labelsize = 48)
        ax.set_ylabel('Fault signal', fontsize = 58)  # 设置y坐标轴
        
        xmin, xmax = np.min(x), np.max(x)
        plt.xlim(xmin - (xmax - xmin) * 0.02, xmax + (xmax - xmin) * 0.02)
        plt.tight_layout()
        plt.savefig('../save/GAN_FDI_CSTR_Y/Recon_{}.pdf'.format(c+1), bbox_inches='tight')
        plt.savefig('../save/GAN_FDI_CSTR_Y/Recon_{}.svg'.format(c+1), bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    RMSE_F[-1] = np.sqrt( RMSE_F[-1]/(1201 * n_af) )
    RMSE_N[-1] = np.sqrt( RMSE_N[-1]/(1201 * n_af) )
    RMSE[-1] = np.sqrt( RMSE[-1]/(1201 * n_af) )
    print('RMSE_F:', np.round(RMSE_F[:-1], 4), '\nARMSE_F:', np.round(RMSE_F[-1], 4))
    print('RMSE_N:', np.round(RMSE_N[:-1], 4), '\nARMSE_N:', np.round(RMSE_N[-1], 4))
    print('RMSE:', np.round(RMSE[:-1], 4), '\nARMSE:', np.round(RMSE[-1], 4))
    
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
