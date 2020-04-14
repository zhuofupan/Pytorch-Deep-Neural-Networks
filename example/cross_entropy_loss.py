# -*- coding: utf-8 -*-
'''
    转自： https://blog.csdn.net/tmk_01/article/details/80839810
'''
import torch
import torch.nn as nn
import math
CE = nn.CrossEntropyLoss()

print(">>> 单样本 >>>")
print("pred:")
pred = torch.randn(1, 5, requires_grad=True)
print(pred)
print("label:")
target = torch.empty(1, dtype=torch.long).random_(5)
print(target)
print("CrossEntropyLoss:")
loss = CE(pred, target)
print(loss)

print("CalculateLoss:")
first = 0
for i in range(1):
    first -= pred[i][target[i]]
second = 0
for i in range(1):
    for j in range(5):
        second += math.exp(pred[i][j])
loss = first +math.log(second)
print(loss,'\n')

print(">>> 多样本 >>>")
print("pred:")
pred = torch.randn(3, 5, requires_grad=True)
print(pred)
print("label:")
target = torch.empty(3, dtype=torch.long).random_(5)
print(target)
print("CrossEntropyLoss:")
loss = CE(pred, target)
print(loss)

print("CalculateLoss:")
first = [0,0,0]
for i in range(3):
    first[i] -= pred[i][target[i]]
second = [0,0,0]
for i in range(3):
    for j in range(5):
        second[i] += math.exp(pred[i][j])
loss = 0
for i in range(3):
    loss += first[i] +math.log(second[i])
print(loss/3,'\n')

'''
    转自： https://www.jianshu.com/p/6049dbc1b73f
'''
print(">>> nn.Softmax + nn.Log >>>")
logits = torch.randn(3,3)#随机生成输入
print('logits:\n',logits)
target= torch.tensor([1,2,0])#设置输出具体值
print('label\n',target)

#计算输入softmax，此时可以看到每一行加到一起结果都是1
softmax = nn.Softmax(dim=1)
pred = softmax(logits)
print('Softmax:\n',pred)

#在softmax的基础上取log
log_pred=torch.log(pred)
print('Softmax + Log:\n',log_pred, '\n')

print(">>> nn.LogSoftmax + nn.NLLLoss >>>")
#对比softmax与log的结合与nn.LogSoftmax(负对数似然损失)的输出结果，发现两者是一致的。
log_softmax = nn.LogSoftmax(dim=1)
log_pred = log_softmax(logits)
print('LogSoftmax:\n',log_pred)

#pytorch中关于NLLLoss的默认参数配置为：reducetion=True、size_average=True
NLL = nn.NLLLoss()
nllloss = NLL(log_pred, target)
print('LogSoftmax + NLLLoss:\n',nllloss,'\n')

print(">>> nn.CrossEntropyLoss >>>")
#直接使用pytorch中的loss_func=nn.CrossEntropyLoss()看与经过NLLLoss的计算是不是一样
CE = nn.CrossEntropyLoss()
crossentropyloss=CE(logits, target)
print('CrossEntropyLoss:\n',crossentropyloss)