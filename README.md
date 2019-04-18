# >>> 快速搭建模型！
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
