# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import math
CE = nn.CrossEntropyLoss()

'''
    参考： https://blog.csdn.net/tmk_01/article/details/80839810
    CrossEntropyLoss(logits, label) = - w_{label} * log( softmax(logits_{label}) )
'''
# 手算 CrossEntropyLoss - 单样本
print(">>> 单样本 >>>")
print("pred:")
logits = torch.randn(1, 5, requires_grad=True)
print(logits)
print("label:")
label = torch.empty(1, dtype=torch.long).random_(5)
print(label)
print("CrossEntropyLoss:")
loss = CE(logits, label)
print(loss)

print("CalculateLoss:")
first = 0
for i in range(1):
    first -= logits[i][label[i]]
second = 0
for i in range(1):
    for j in range(5):
        second += math.exp(logits[i][j])
loss = first + math.log(second)
print(loss,'\n')

# 手算 CrossEntropyLoss - 多样本
print(">>> 多样本 >>>")
print("pred:")
logits = torch.randn(3, 5, requires_grad=True)
print(logits)
print("label:")
label = torch.empty(3, dtype=torch.long).random_(5)
print(label)
print("CrossEntropyLoss:")
loss = CE(logits, label)
print(loss)

print("CalculateLoss:")
first = [0,0,0]
for i in range(3):
    first[i] -= logits[i][label[i]]
second = [0,0,0]
for i in range(3):
    for j in range(5):
        second[i] += math.exp(logits[i][j])
loss = 0
for i in range(3):
    loss += first[i] + math.log(second[i])
print(loss/3,'\n')

'''
    参考： https://www.jianshu.com/p/6049dbc1b73f
    CrossEntropyLoss(logits, label) = NLLLoss(LogSoftmax(logits), label)
'''
# pytorch 中 NLLLoss，CrossEntropyLoss 的使用方法

print(">>> nn.Softmax + nn.Log >>>")
logits = torch.randn(3,3) # 随机生成输入
print('logits:\n',logits)
label= torch.tensor([1,2,0]) # 设置输出具体值
print('label\n',label)

#计算输出的 log_softmax(logits)
softmax = nn.Softmax(dim=1)
pred = softmax(logits)
log_pred=torch.log(pred)

log_softmax = nn.LogSoftmax(dim=1)
log_pred = log_softmax(logits)

# NLLLoss 默认参数：reducetion=True，size_average=True
NLL = nn.NLLLoss()
nllloss = NLL(log_pred, label)
print('LogSoftmax + NLLLoss:\n',nllloss,'\n')

print(">>> nn.CrossEntropyLoss >>>")
# CrossEntropyLoss(logits, label) = NLLLoss (LogSoftmax(logits), label)
CE = nn.CrossEntropyLoss()
crossentropyloss=CE(logits, label)
print('CrossEntropyLoss:')
print(crossentropyloss, '\n')

'''
    参考： https://pytorch.org/docs/stable/nn.html#kldivloss
    KLDivLoss(log(Q), P) = P * log( P / Q )
'''
# pytorch 中 KLDivLoss 的使用方法
p0, p1 = torch.tensor(0.05), pred
print(">>> nn.KLDivLoss >>>")
print("CalculateLoss:")
loss = torch.sum(p0 * torch.log(p0/p1) + (1-p0)* torch.log((1-p0)/(1-p1)))
print(loss,'\n')

print("KLDivLoss:") 
KL = nn.KLDivLoss(reduction='sum')
loss = KL(p1.log(), p0) + KL((1-p1).log(), (1 - p0))
print(KL(p1.log(), p0), KL((1-p1).log(), (1 - p0)))
print(loss,'\n')