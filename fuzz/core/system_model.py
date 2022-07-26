# -*- coding: utf-8 -*-
"""
Created on Thu Dec 23 22:41:02 2021

@author: Fuzz4
"""
import os
import numpy as np
import pandas as pd
from pandas import DataFrame

def to_array(X):
    return np.array(X).astype(np.float32)

def Gx(X):
    g = np.zeros_like(X)
    g[0] = 0.2* np.sin(X[1])
    g[1] = -0.2* np.sin(X[0])
    g[2] = 0.1* np.cos(X[2])
    return g

_A = np.array([[0.7, 0.5, 0.4], 
               [0.6, 0.3, 0.7], 
               [-0.2, 0, 0.3]])
_B = np.array([[1], 
               [0], 
               [0.3]])
_C = np.array([[0, 1, 1], 
               [1, 0, 1]])
_D = np.array([[1], 
               [0]])

class System():
    def __init__(self, 
                 para = None,
                 nonlinear_func = Gx):
        if para is None:
            self.A, self.B, self.C, self.D = _A, _B, _C, _D
        else:
            A, B, C, D = para
            self.A, self.B, self.C, self.D = to_array(A), to_array(B), to_array(C), to_array(D)
        self.NF = nonlinear_func
    
    def gene_data(self, x0, U, N):
        x0 = to_array(x0).reshape((-1,1))
        U = to_array(U).reshape((N,-1))
        X = np.zeros((N + 1, x0.shape[0]))
        Y = np.zeros((N, self.C.shape[0]))
        dim_x, dim_y = X.shape[1], Y.shape[1]
        W = np.random.multivariate_normal(np.zeros(dim_x), np.eye(dim_x), N)
        V = np.random.multivariate_normal(np.zeros(dim_y), np.eye(dim_y), N)
        X[0] = x0.reshape((-1,))
        for k in range(N):
            Y[k] = self.C @ X[k] + self.D @ U[k] + V[k]
            g = 0
            if self.NF is not None:
                g = self.NF(X[k])
            X[k+1] = g + self.A @ X[k] + self.B @ U[k] + W[k]
        return X[:N], Y
    
    def demo(self, dynamic, N = 10000):
        # normal
        train_U = np.linspace(-10, 10, N)
        _, train_Y = self.gene_data([0, 0, 0], train_U, N)
        train_Us, train_Ys = self.stacked_data(train_U, train_Y, N, dynamic)
        train_Ls = np.tile(np.array([[1, 0]]), (train_Us.shape[0],1))
        # sys.save_data(train_Ls, train_Us, train_Ys, name = 'gene_train')
        
        # fault
        test_U = np.zeros_like(train_U)
        n_step = 5
        n_samples = int(N/n_step)
        for i in range(n_step):
            start, end = i*n_samples, (i+1)*n_samples
            test_U[start: end] = [(train_U[start] + train_U[end - 1])/2] * (end - start)
        _, test_Y = self.gene_data([0, 0, 0], test_U, N)
        test_Y[int(N/2):,:] += np.array([[10, 2]])
        test_Us, test_Ys = self.stacked_data(test_U, test_Y, N, dynamic)
        test_Ls = np.tile(np.array([[1, 0]]), (test_Us.shape[0],1))
        test_Ls[int(N/2)-dynamic+1:] = np.array([[0, 1]])
        # sys.save_data(test_Ls, test_Us, test_Ys, name = 'gene_test')
        
        train_X = np.concatenate([train_Us, train_Ys], 1)
        test_X = np.concatenate([test_Us, test_Ys], 1)
        datasets = (train_X, train_Ls, test_X, test_Ls)
        labels = ['Fault 01']
        return datasets, labels
    
    def stacked_data(self, U, Y, N, n):
        U = to_array(U).reshape((N,-1))
        Us = np.zeros((N - n + 1, n * U.shape[1]))
        Ys = np.zeros((N - n + 1, n * Y.shape[1]))
        for i in range(N - n + 1):
            Us[i] = U[i:i+n].reshape(-1)
            Ys[i] = Y[i:i+n].reshape(-1)
        return Us, Ys
    
    def save_data(self, X, U, Y, name = 'gene'):
        sheet_names = ['X','U','Y']
        dfs = [DataFrame(X), DataFrame(U), DataFrame(Y)]
        # writer
        path = '../data/system/'+name
        if not os.path.exists(path): os.makedirs(path)
        file = path + '/XUY.xlsx'
        writer = pd.ExcelWriter(file, engine='openpyxl')
        # save
        for i, sheet_name in enumerate(sheet_names):
            dfs[i].to_excel(excel_writer = writer, sheet_name = sheet_name, encoding="utf-8", index=False)
        writer.save()
        writer.close()
    
if __name__ == '__main__':
    N = 1000
    U = np.linspace(-10, 10, N)
    sys = System()
    X, Y = sys.gene_data([0, 0, 0], U, N)
    sys.save_data(X, U, Y)
    Us, Ys = sys.stacked_data(U, Y, N, 6)
    sys.save_data(X, Us, Ys, 'stacked')