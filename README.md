# > 包含的网络模型：
*Deep Belief Network* **(DBN)** </br>
*Deep Autoencoder* **(DAE)** </br>
*Stacked Autoencoder* **(sAE)** </br>
*Stacked Sparse Autoencoder* **(sSAE)**  </br>
*Stacked Denoising Autoencoder* **(sDAE)** </br>
*Convolutional Neural Network* **(CNN)** </br>
*Visual Geometry Group* **(VGG)** </br>
*Residual Network* **(ResNet)**  </br>
## [模型详细介绍见README.md](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/model/README.md)</br>
# > 开始学习：
Pytorch初学: 建议看看 [官网教程](https://pytorch.org/tutorials/) 和 [网络模型codes](https://github.com/rusty1s/pytorch_geometric/tree/master/examples) </br>
理解本package：看看这个不依赖其他文件运行的 [简单AE](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/example/simple_ae.py)
# > 用于任务：
`task == 'cls'` 用于分类任务 </br>
`task == 'prd'` 用于预测任务 </br>
# > 读入数据集：
## 建立 `'ReadData'` 类来读入数据集 —— 详见 [gene_dynamic_data.py](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/data/gene_dynamic_data.py) </br>
## 输入网络前一般还需将数据集转换成 `DataLoader` 以便批次训练 —— 详见 [load.py](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/data/load.py) </br>
- 自动加载文件： 需要定位到根目录，目录下建立`trian`和`test`文件夹，文件名中包含`_x`或`_y`来区分输入和输出, 支持后缀为`csv`,`txt`,`dat`,`xls`,`xlsx`,`mat` 的文件 </br>
- 数据预处理：类初始化中设置 `prep = ['prep_x', 'prep_y']`, prep 方式包括 `'st'`标准化, `'mm'`归一化, `'oh'`01编码 </br>
- 制作动态数据：可设置动态滑窗边长`'dynamic'`, 步长 `'stride'` </br>
# > CNN快速建模： 
## List
用一个列表表示CNN的结构：</br>
如`[[3, 8], ['M', 2], ['R', [6, (6,6)], '|', [3, 1, 1] ]`表示 </br>
**1、** 3@8×8 - 3个8乘8的卷积核 </br>
**2、** MaxPool - 核大小为 2×2 的池化层（默认stride = kernel_size） </br>
**3、** 残差块 - 主体为6个6乘6的卷积核，残差部分为3个1乘1的卷积核 </br></br>
列表还有很多灵活的用法，如：</br>
- `'/2'` 表示 `stride = 2` </br>
- `'+1'` 表示 `padding = 1` </br>
- `'#2'` 表示 `dilation = 2` </br>
- `'2*'` 表示将后面一个元素循环2次 </br>
更多详见 [README.md](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/model/README.md) </br>
## DataFrame
包的内部会自动将List转换为DataFrame以进一步构建模型 </br>
DataFrame中有6列： `'Conv'`, `'*'`, `'Pool'`, `'Res'`, `'Loop'`, `'Out'` </br>
分别表示`“卷积结构”`（可以是列表），`“卷积循环次数”`，`“池化结构”`，`“残差结构”`，`“整个块循环次数”`，`“输出尺寸”`
## Parameter
模型构建好后，网络的`“可优化参数”`及`“参数尺寸”`将自动在Console中展示
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

    def forward(self, x, y = None):
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
             'conv_func': 'r',
             'struct': [-1, 10],
             'hidden_func': ['g', 'a'],
             'output_func': 'a',
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
    model.test(epoch)
```
# > 结果展示：
```python
model.result()
```
Console：
```python
Structure:
             Conv  * Res         Pool Loop          Out
0       [1, 3, 8]  1   -  [Max, 2, 2]    1  [3, 10, 10]
1  [3, 6, (6, 6)]  1   -            -    1    [6, 5, 5]

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
# My blog
[知乎](https://www.zhihu.com/people/fu-zi-36-41/posts), 
[CSDN](https://blog.csdn.net/fuzimango/article/list/) </br>
QQ群：640571839

# paper
[EDBN](https://www.sciencedirect.com/science/article/pii/S0019057819302903?via%3Dihub) 欢迎引用
