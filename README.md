# > 包含的网络模型：
*Deep Belief Network (DBN) </br>
Deep Autoencoder (DAE) </br>
Stacked Autoencoder (sAE) </br>
Stacked Sparse Autoencoder (sSAE)  </br>
Stacked Denoising Autoencoder (sDAE) </br>
Convolutional Neural Network (CNN) </br>
Visual Geometry Group (VGG) </br>
Residual Network (ResNet)*  </br>

# > 快速搭建模型！
## Step 1. 创建模型类
```python
class CNN(Module):  
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.Convolutional()
        self.Sequential()
        self.opt()

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.feature(x)
        x = self.output(x)
        return x
```
## Step 2. 实例化模型
```python  
parameter = {'img_size': [1,28,28],
             'conv_struct': [[3, 8], ['M', 2], [6, (6,6)]],
             'conv_func': 'ReLU',
             'struct': [150, 10],
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
    model.train_batch(epoch)
    model.test_batch(epoch)
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
