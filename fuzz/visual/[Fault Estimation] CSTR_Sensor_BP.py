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

def _get_rmse():
    model = 'Embd_DNet_fd_CSTR_3'
    file = '[Embd_DNet] Estimated_fault (cuda) loss.xlsx'
    # file = '[Embd_DNet] Estimated_fault (cuda).xlsx'
    file_path = '../save/'+model+'/'+file
    scaler = pd.read_csv('../data/CSTR/fd_sensor/scaler/[st] scaler_x.csv', header = None).values
    n_af = 6
    N, m = 1000, 7
    
    esti_f = []
    esti_f_data = pd.read_excel(file_path).values
    for i in range(6):
        esti_f.append(esti_f_data[int(i*N): int((i+1)*N)] )
    
    
    f_mag = [0.002,0.15,0.1,0.001,0.05,0.05,-0.1]
    real_f = np.zeros((N,m))
    for v in range(len(f_mag)):
        f_signal = np.linspace(f_mag[v], f_mag[v]*N, N)
        # 生成真实故障信号
        real_f[:,v] = f_signal/np.sqrt( scaler[v,1] )
    
    # 变量标签
    # Ci, Ti, Tci, C, T, Tc, Qc
    legend = ['$\hat f_{C_i^{(s)}}$','$\hat f_{T_i^{(s)}}$', '$\hat f_{T_{ci}^{(s)}}$', '$\hat f_{C^{(s)}}$',\
              '$\hat f_{T^{(s)}}$', '$\hat f_{T_c^{(s)}}$', '$\hat f_{Q_c^{(s)}}$']
    # 故障变量标签
    legend_plus = ['$f_{C_i^{(s)}}$', '$f_{T_i^{(s)}}$', '$f_{T_{ci}^{(s)}}$', '$f_{C^{(s)}}$',\
                   '$f_{T^{(s)}}$', '$f_{T_c^{(s)}}$', '$f_{Q_c^{(s)}}$']
    # 故障位置
    '''
        Fault start = 200
        故障1：Catalyst Fault    (4,5,6,7),  exp(-0.0005*t), in [0,1]
        故障2：HTC Fault         (4,5,6,7),  exp(-0.001*t),  in [0,1]
        故障3：Ci Actuator Bias  (1),        0.002*t
        故障4：Ti Actuator Bias  (2),        0.15*t
        故障5：Tci Actuator Bias (3),        0.1*t
        故障6：C, Tc Sensor Bias (4,6),      0.001*t, 0.05*t
        故障7：T Sensor Bias     (5),        0.05*t
        故障8：Qc Sensor Bias    (7),        -0.1*t
    '''
    index = {'1':[0], '2':[1], '3':[2], '4':[3,5], '5':[4], '6':[6]}
    color4= ['lime','darkorange','fuchsia','aqua']
    blues= ['b','darkblue']
    reds = ['r', 'crimson']

    RMSE_N, RMSE_F, RMSE = np.zeros(n_af + 1), np.zeros(n_af + 1), np.zeros(n_af + 1)
    
    for c in range(n_af):
        print('Plot Recon_{}.pdf'.format(c+1))
        x = np.arange(1, N + 1)
        color=plt.get_cmap('Blues')(np.linspace(0.05, 0.95, m))
        # 预测的故障信号
        data = esti_f[c]

        fig = plt.figure(figsize=[26,15])
        ax = fig.add_subplot(111)
        for v in range(m):
            if v in index[str(c+1)]: continue
            # 加性的非故障变量用浅蓝色
            ax.plot(x, data[:,v], linewidth = 2, c = color[v], label = legend[v])
        # 加性故障
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
                RMSE_F[c] += np.sum((data[:, v] - real_f[:, v])**2)
            # fault-free variables
            else:
                RMSE_N[c] += np.sum(data[:, v] **2)
            
            # 0~-2 是各类故障的， -1 是总的
            RMSE_F[-1] += RMSE_F[c]
            RMSE_N[-1] += RMSE_N[c]
            RMSE[-1] += RMSE_F[c] + RMSE_N[c]
            RMSE_F[c] = np.sqrt( RMSE_F[c]/N )
            RMSE_N[c] = np.sqrt( RMSE_N[c]/N )
            RMSE[c] = np.sqrt( (RMSE_F[c] + RMSE_N[c])/N )
                
        lgd = ax.legend(loc = 'upper right',  fontsize=46)
        lgd.get_frame().set_alpha(0.5)
        ax.tick_params('x', labelsize = 48)
        # ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
        ax.set_xlabel('Samples', fontsize = 58)  # 设置x坐标轴
        ax.tick_params('y', labelsize = 48)
        ax.set_ylabel('Fault signal', fontsize = 58)  # 设置y坐标轴
        plt.tight_layout()
        plt.savefig('../save/'+model+'/Recon_{}.pdf'.format(c+1), bbox_inches='tight')
        plt.savefig('../save/'+model+'/Recon_{}.svg'.format(c+1), bbox_inches='tight')
        plt.show()
        plt.close(fig)
    
    RMSE_F[-1] = np.sqrt( RMSE_F[-1]/(N * n_af) )
    RMSE_N[-1] = np.sqrt( RMSE_N[-1]/(N * n_af) )
    RMSE[-1] = np.sqrt( RMSE[-1]/(N * n_af) )
    print('RMSE_F:', np.round(RMSE_F[:-1], 4), '\nARMSE_F:', np.round(RMSE_F[-1], 4))
    print('RMSE_N:', np.round(RMSE_N[:-1], 4), '\nARMSE_N:', np.round(RMSE_N[-1], 4))
    print('RMSE:', np.round(RMSE[:-1], 4), '\nARMSE:', np.round(RMSE[-1], 4))
    
if __name__ == '__main__':
    _get_rmse()

