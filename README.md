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
conv = DataFrame(
    columns = ['out_channel', 'conv_kernel_size', 'is_bn', 'pool_kernel_size']
    )
conv.loc[0] = [3, 8, 1, 2]
conv.loc[1] = [6, (6,6), 1, 0]
    
parameter = {'img_size': [1,28,28],
             'conv_struct': conv,
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
CNN(
  (L): MSELoss()
  (conv): Sequential(
    (Conv2d0): Conv2d(1, 3, kernel_size=(8, 8), stride=(1, 1))
    (BatchNorm2d0): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Activation0): ReLU()
    (MaxPool2d0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (Conv2d1): Conv2d(3, 6, kernel_size=(6, 6), stride=(1, 1))
    (BatchNorm2d1): BatchNorm2d(6, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (Activation1): ReLU()
  )
  (feature): Sequential()
  (output): Sequential(
    (0): Linear(in_features=150, out_features=10, bias=True)
    (1): Affine()
  )
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
