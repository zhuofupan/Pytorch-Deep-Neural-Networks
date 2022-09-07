# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:14:37 2022

@author: Fuzz4
"""

import numpy as np
from scipy.linalg import toeplitz

A = toeplitz([5,4,3,2,1], [5,4,3,2,1])
sigma, P = np.linalg.eig(A)
S = np.diag(sigma)

np.set_printoptions(precision=3, suppress=True)
# P@P.T = I, P^{-1} = P.T
print(P@P.T)
# A = P@S@P.T
print(P@S@P.T)
# A^{-1} = P@S^{-1}@p.T
print(P@np.diag(1/sigma)@P.T)
print(np.linalg.inv(A))