# > 包含的网络模型：
*Deep Belief Network* **(DBN)** </br>
*Deep Autoencoder* **(DAE)** </br>
*Stacked Autoencoder* **(sAE)** </br>
*Stacked Sparse Autoencoder* **(sSAE)**  </br>
*Stacked Denoising Autoencoder* **(sDAE)** </br>
*Convolutional Neural Network* **(CNN)** </br>
*Visual Geometry Group* **(VGG)** </br>
*Residual Network* **(ResNet)**  </br>
# > 模型结构表示：
## List
用一个列表表示CNN的结构：</br>
如`[[3, 8], ['M', 2], ['R', [6, (6,6)], '|', [3, 1, 1] ]`表示 </br>
**1)** 3@8×8 - 3个8乘8的卷积核
**2)** MaxPool - 核大小为 2×2 的池化层（默认stride = kernel_size） </br>
**3)** 残差块 - 主体为6个6乘6的卷积核，残差部分为3个1乘1的卷积核 </br>
列表还有很多灵活的用法，如：</br>
`'/2'` 表示 `stride = 2` </br>
`'+1'` 表示 `padding = 1` </br>
`'#2'` 表示 `dilation = 2` </br>
`'2*'` 表示将后面一个元素循环2次 </br>
## DataFrame
包的内部会自动将列表转换为DataFrame以进一步构建模型 </br>
DataFrame中有6列： `Conv`, `*`, `Pool`, `Res`, `Loop`, `Out` </br>
分别表示`卷积结构`（可以是列表），`卷积循环次数`，`池化结构`，`残差结构`，`整个块循环次数`，`输出尺寸`
## Parameter
网络的`可优化参数`及`参数尺寸`将自动在Console中展示
# > 快速搭建模型！
## Step 1. 创建模型类
```python
class CNN(Module, Conv_Module):  
    def __init__(self, **kwargs):
        self.name = 'CNN'
        
        Module.__init__(self,**kwargs)
        Conv_Module.__init__(self,**kwargs)

        self.layers = self.Convolutional()
        self.fc = self.Sequential()
        self.opt()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
```
## Step 2. 实例化模型
```python  
parameter = {'img_size': [1,28,28],
             'conv_struct': [[3, 8], ['M', 2], [6, (6,6)]],
             'conv_func': 'ReLU',
             'struct': [-1, 10],
             'hidden_func': ['Gaussian', 'Affine'],
             'output_func': 'Affine',
             'dropout': 0.0,
             'task': 'cls'}
    
model = CNN(**parameter)
```
## Step 3. 加载数据集
```python
model.load_mnist('../data', 128)
```
## Step 4. 训练与测试模型
```python
for epoch in range(1, 3 + 1):
    model.batch_training(epoch)
    model.test()
```
# > 结果展示：
```python
Structure:
             Conv  *         Pool Res Loop          Out
0       [1, 3, 8]  1  [Max, 2, 2]   -    1  [3, 10, 10]
1  [3, 6, (6, 6)]  1            -   -    1    [6, 5, 5]

CNN(
  (L): MSELoss()
  (layers): Sequential(
    (0): ConvBlock(
      (conv1): Conv2d(1, 3, kernel_size=(8, 8), stride=(1, 1))
      (bn1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_layer): ReLU(inplace)
      (pool_layer): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    )
    (1): ConvBlock(
      (conv1): Conv2d(3, 6, kernel_size=(6, 6), stride=(1, 1))
      (bn1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (act_layer): ReLU(inplace)
    )
  )
  (fc): Sequential(
    (0): Linear(in_features=150, out_features=10, bias=True)
    (1): Affine()
  )
)
CNN's Parameters(
  layers.0.conv1.weight:        torch.Size([3, 1, 8, 8])
  layers.0.conv1.bias:  torch.Size([3])
  layers.0.bn1.weight:  torch.Size([3])
  layers.0.bn1.bias:    torch.Size([3])
  layers.0.bn1.running_mean:    torch.Size([3])
  layers.0.bn1.running_var:     torch.Size([3])
  layers.0.bn1.num_batches_tracked:     torch.Size([])
  layers.1.conv1.weight:        torch.Size([6, 3, 6, 6])
  layers.1.conv1.bias:  torch.Size([6])
  layers.1.bn1.weight:  torch.Size([6])
  layers.1.bn1.bias:    torch.Size([6])
  layers.1.bn1.running_mean:    torch.Size([6])
  layers.1.bn1.running_var:     torch.Size([6])
  layers.1.bn1.num_batches_tracked:     torch.Size([])
  fc.0.weight:  torch.Size([10, 150])
  fc.0.bias:    torch.Size([10])
)
Epoch: 1 - 469/469 | loss = 0.0259
    >>> Train: loss = 0.0375   accuracy = 0.9340   
    >>> Test: loss = 0.0225   accuracy = 0.9389   
Epoch: 2 - 469/469 | loss = 0.0200
    >>> Train: loss = 0.0215   accuracy = 0.9484   
    >>> Test: loss = 0.0191   accuracy = 0.9525   
Epoch: 3 - 469/469 | loss = 0.0204
    >>> Train: loss = 0.0192   accuracy = 0.9537   
    >>> Test: loss = 0.0179   accuracy = 0.9571   
```
