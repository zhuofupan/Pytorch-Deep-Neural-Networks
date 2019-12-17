# > Deep Belief Network (DBN)：
- DBN由RBM堆叠而成，每两个隐层（包括输入层）构成一个RBM</br>
- 每个RBM由可见层和隐层构成，根据CD-k算法进行训练</br>
- RBM逐个训练后堆叠在一起构成DBM，作为预训练后的模型，可以提取预训练后的深度特征</br>
- 在DRM的最后一层加入输出层构成DBN，随后用BP微调</br>
![DBN示意图](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/image/EDBN.jpg)

# > Restricted Boltzmann Machine (RBM)：
- 根据单元类型不同，分为高斯/二值/...，对应能量函数E和条件概率p(|)计算公式不同</br>
- 网络的参数W，b包含在E和p(|)中</br>
- 采用CD-k算法训练，即 v → p(h|v) → sample h → p(v|h) → v → ...</br>
- 通过极大化p(v)概率来训练W，b，而不是最小化重构误差</br>
- 具体原理可参考[我的论文](https://www.sciencedirect.com/science/article/pii/S0019057819302903?via%3Dihub) </br>
![CD-k示意图](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/image/CD-K.jpg)

# > Stacked Autoencoder (SAE)：
- 方式和DBN基本一致，区别在于SAE的AE是确定性模型，DBN的RBM是概率模型</br>

# > Autoencoder (AE)：
- AE是一个三层结构，前两层为编码器（也将用于微调），后两层为解码器（不参与微调）</br>
- 最小化输入与重构的误差来训练W，b</br>
- 代码中通过设置 ae_type = 'AE'/'SAE'/'DAE' 来切换自编码器/稀疏自编码器/去噪自编码器</br>

# > Deep Autoencoder (DAE)：
- 这里的DAE不是堆叠的多个AE，而是一个多层的AE</br></br>
- 编码器和解码器由多层网络构成，而不是两层</br>

# > Convolutional Neural Network (CNN)：
卷积神经网络有很多操作，在这里我们用一个list的形式来定义CNN的结构</br>

list 中可添加的内容：</br>

<table>
<style> table th:first-of-type { width: 68px;} table th:nth-of-type(2) { width: 100px;} </style>
| 类型 | 说明 | 默认值
| :- | :- | :- 
| 整数开头</br>(int) | 仅提供 `Conv2d` 层输出通道大小 | `[ 输出通道 = ?, 核尺寸 = 3, 步长 = 1, 扩展 = 1 ]`
| 整数开头</br>(list) | 提供了 `Conv2d` 层卷积尺寸 | `[ 输出通道 = ?, 核尺寸 = ?, 步长 = '/1', 扩展 = '+0', 空洞 = '#1', 分组 = '%1', 偏值 = self.use_bias ]`
|  | 其他可设定项</br>(加入list中即可) | `批次正则化 = 'N', 激活函数 = 'r', dropout = 'D0', 按某维度洗牌 = 'SF', 反卷积 = 'TS'`
| 字母开头</br>(str 或 list) | `'M'`, `'A'`, `'AM'`, `'AA'`, `'FM'` 表示池化层 |  `[ 卷积类型 = 'M' / 'A', 核尺寸 = ?, 步长 = 核尺寸, 扩展 = 0, 空洞 = 1 ]`</br> `[ 卷积类型 = 'AM' / 'AA' / 'FM' , 输出尺寸 = ? ]`
| | `'R'` 表示带残差层的卷积 | `['R', 卷积参数 = int / list, '|', 卷积参数 = int / list ]`
| | `'S'` 表示一个块 | `['S', ...]`
| `'int*'` | 表示将后面一个元素重复`int`遍 | `'int*', ...`
|`Module`类 | 嵌入自定义的`Module` 类</br> (需定义 `name` 和 `out_size` 属性) | `Module, ... / [ Module, ... ]`
</table>

通过list定义的CNN结构，在代码执行时会自动转化成dataframe（自动计算经过各层后的out_size）并print出来</br>

# > Visual Geometry Group (VGG)：
- 这里我用上述list定义CNN的方式构建了VGG</br>

# > Residual Network (ResNet)：
- 这里我用上述list定义CNN的方式构建了ResNet</br>

# > Connect：
- 这是一个连接模型，将定义的几个子网络堆叠成一个网络</br>
