# > 包含的网络模型：
*Deep Belief Network* **(DBN)** </br>
*Restricted Boltzmann Machine* **(RBM)** </br>
*Stacked Autoencoder* **(sAE)** </br>
*Stacked Sparse Autoencoder* **(sSAE)**  </br>
*Stacked Denoising Autoencoder* **(sDAE)** </br>
*Deep Autoencoder* **(DAE)** </br>
*Variational Autoencoder* **(VAE)** </br>
*Convolutional Neural Network* **(CNN)** </br>
*Visual Geometry Group* **(VGG)** </br>
*Residual Network* **(ResNet)**  </br>
*Long Short-Term Memory* **(LSTM)**  </br>

# > Deep Belief Network (DBN)：
- DBN 由 RBM 堆叠而成，每两个隐层（包括输入层）构成一个 RBM</br>
- 每个 RBM 由可见层和隐层构成，根据极大似然法则及 CD-k 算法进行训练</br>
- 逐个训练 RBM 称为 DBN 的预训练过程，将训练好的 RBM + Output Layer 用 BP 再训练称为 DBN 的微调过程</br>
![DBN示意图](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/image/EDBN.jpg)

# > Restricted Boltzmann Machine (RBM)：
- 根据单元类型不同，分为高斯/二值/...，对应不同的能量函数 E 和条件概率 p(|) 表达式</br>
- RBM 的参数 W，b 同时为 E 和 p(|) 的参数</br>
- 采用 CD-k 算法训练：`v → p(h|v) → sample h → p(v|h) → sample v → ...`</br>
- 通过极大化 p(v) 概率来训练 W，b，而不是最小化重构误差</br>
- 具体原理可参考[我的论文](https://www.sciencedirect.com/science/article/pii/S0019057819302903?via%3Dihub)，CD-K 的图示略有错误，更正如下 </br>
![CD-k示意图](https://github.com/fuzimaoxinan/Pytorch-Deep-Neural-Networks/blob/master/image/CD-K.jpg)

# > Stacked Autoencoder (SAE)：
- 训练流程和 DBN 基本一致，区别在于 sAE 及 AE 是确定性模型，而 DBN 及 RBM 是概率模型</br>

# > Autoencoder (AE)：
- AE 是一个三层结构，前两层为编码器（参与微调），后两层为解码器（不参与微调）</br>
- 最小化输入与重构的误差来训练 W，b</br>
- 代码中通过设置 `ae_type = 'AE'/'SAE'/'DAE'` 分别构建自编码器/稀疏自编码器/去噪自编码器</br>

# > Deep Autoencoder (DAE)：
- 这里的 DAE 是一个有多个隐层的 AE</br>
- 编码器和解码器由 ≥ 2 的多层网络构成</br>

# > Variational Autoencoder (VAE)：
- 生成模型，用于生成新样本或提取分布特征</br>

# > Convolutional Neural Network (CNN)：
这里我们可用一个 `list` 来定义 CNN 的结构，代码将自动将其转化为一个 `pd.DataFrame` 结构（自动计算各层后的 `out_size`）</br>

list 中可添加的元素：</br>

| 类型 | 说明 | 默认值
| :- | :- | :-
| 整数开头 (**int**) | 提供 `Conv2d` 层输出通道大小 | `[ 输出通道 = ?, 核尺寸 = 3, 步长 = 1, 扩展 = 1 ]`
| 整数开头 (**list**) | 提供 `Conv2d` 层输出通道大小及卷积尺寸 | `[ 输出通道 = ?, 核尺寸 = ?, 步长 = '/1', 扩展 = '+0', 空洞 = '#1', 分组 = '%1', 偏值 = True ]`
|  | 其他可设定项</br>(加入list中即可) | `批次正则化 = 'B11', 激活函数 = 'r', dropout = 'D0'`
| 字母开头 (**str** 或 **list**) | `'M'`, `'A'`, `'AM'`, `'AA'`, `'FM'` 为不同池化层 |  `[ 卷积类型 = 'M' / 'A', 核尺寸 = ?, 步长 = 核尺寸, 扩展 = 0, 空洞 = 1 ]`</br> `[ 卷积类型 = 'AM' / 'AA' / 'FM' , 输出尺寸 = ? ]`
|`'R'`开头 (**list**) | `'|'`之后为残差层参数 | `['R', 卷积参数 = int / list, '|', 卷积参数 = int / list ]`
|`'S'`开头 (**list**)| 表示一个集合 | `['S', ...]`
| `'int*'` (**str**)| 表示将后面一个元素重复`int`遍 | `'int*', ...`
| `Module` 类 (**class**)| 嵌入自定义的`Module` 类</br> (需定义 `name` 和 `out_size` 属性) | `Module, ... `

更多用法见源代码</br>

# > Visual Geometry Group (VGG)：
- 用 list 构建 CNN 的方式重建了 VGG </br>

# > Residual Network (ResNet)：
- 用 list 构建 CNN 的方式重建了 ResNet</br>

# > Long Short-Term Memory (LSTM)：
- 经典多变量时序模型，输入输出格式为 `output, (hn, cn) = lstm(input, (h0, c0))`</br>
- 训练与预测具有连续性，即后续运算需要前面的结果作为其中一部分输入
