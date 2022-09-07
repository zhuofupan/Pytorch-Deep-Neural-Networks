# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 18:10:55 2022

@author: Fuzz4
"""

import torch
from functorch import vmap, jacrev, hessian

x = torch.randn(5,4)
w = torch.randn(4,3)

def f(_x):
    # (?,4) -> (?,3)
    return (_x @ w)**2

def g(_x):
    # (?,4) -> (?,3)
    return torch.softmax(_x @ w, dim = -1)

J = vmap(jacrev(f))(x)
print(J.size())

J = vmap(jacrev(g))(x)
print(J.size())

with torch.no_grad():
    J = vmap(jacrev(g))(x)
    print((J**2).mean())
    H = vmap(hessian(g))(x)
    print((H**2).mean())