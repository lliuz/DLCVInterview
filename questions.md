#### Q1 : 简介常见初始化方法及其思路

![tag](https://img.shields.io/badge/DL-1-brightgreen) 

**关键点: ** 方差公式，方差不稳定性，设计思路，Xavier 和 He 的实现方法与特性。

1. 完全随机

   方差会不断变大或变小，可以由公式推出来：
   $$
   \begin{aligned}
   \operatorname{Var}\left[a^{l}\right] &=\operatorname{Var}\left[\sum_{i=1}^{n^{(l-1)}} w_{i}^{l} a_{i}^{(l-1)}\right] \\
   &=\sum_{i=1}^{n^{(l-1)}} \operatorname{Var}\left[w_{i}^{l}\right] \operatorname{Var}\left[a_{i}^{(l-1)}\right] \\
   &={n^{(l-1)} \operatorname{Var}\left[w_{i}^{l}\right]} \operatorname{Var}\left[a_{i}^{(l-1)}\right]
   \end{aligned}
   $$

2. Xavier ['zʌvɪə] 初始化

设计思路：根据完全随机中的方差公式，<u>保持输入和输出的方差一致</u>。

实现：参数初始化为均值为0，方差为 $$\frac{1}{n}$$， $$n$$ 为输入个数

```python
W = tf.Variable(np.random.randn(node_in, node_out)) / np.sqrt(node_in)
```

特性：会对用 tanh 的网络保持各层输出分布相似，但对于 ReLU 还是会存在越来越接近 0 的分布。

3. He Initialization

设计思路：在 ReLU 网络中，假定每一层有一半的神经元被激活，另一半为 0，所以，要保持方差不变，只需要在 Xavier 的方差基础上再除以2

```
W = tf.Variable(np.random.randn(node_in,node_out)) / np.sqrt(node_in/2)
```

特性：基本保持输入输出方差一致。

#### Q2: 初始化能否完全随机或全0

![tag](https://img.shields.io/badge/DL-1-brightgreen) 

**关键点: ** 对称性，方差公式，方差不稳定性，设计思路
$$
\begin{aligned}
\operatorname{Var}\left[a^{l}\right] &=\operatorname{Var}\left[\sum_{i=1}^{n^{(l-1)}} w_{i}^{l} a_{i}^{(l-1)}\right] \\
&=\sum_{i=1}^{n^{(l-1)}} \operatorname{Var}\left[w_{i}^{l}\right] \operatorname{Var}\left[a_{i}^{(l-1)}\right] \\
&={n^{(l-1)} \operatorname{Var}\left[w_{i}^{l}\right]} \operatorname{Var}\left[a_{i}^{(l-1)}\right]
\end{aligned}
$$
绝对不行，两方面：

1. 如果所有的参数都是0，那么所有神经元的输出都将是相同的，那在back propagation的时候同一层内所有神经元的行为也是相同的 --- gradient相同，weight update也相同，这显然是一个不可接受的结果。
2. 完全随机的话，由公式可以知道高层的方差是不稳定的。也可以试验一下，设计一个简单的多层神经网络，激活函数为tanh，每一层的参数都是均值为0，标准差为 0.01 的随机正态分布，对每层的输出做直方图统计，会发现分布在高层会向0聚拢。将会导致梯度为0，参数难以更新。如果标准差改为1，每一层的输出都会集中在-1，1之间，神经元饱和，梯度无法更新。也就是说**参数的初始化要使得每一层的输出不能太大，也不能太小，最合理的方法就是使输出的标准差保持不变。**

#### Q3: 卷积扩张卷积的输出 size 的计算

![tag](https://img.shields.io/badge/DL-1-brightgreen) 

**关键点: ** size 公式，理解方式，Dilation Conv size 公式

公式为 (W+2P - K) / S 向下取整 + 1，也就是向上取整。
$$
o = \left \lfloor \frac{(W + 2P - K )} {S} \right \rfloor + 1
$$
理解方式：减掉 k 之后 能被 stride 整除几次(也就是还能滑几次)

![1583232870263](questions.assets/1583232870263.png)

另外如果有 Dilation Conv，需要修改以上公式为，直接以更大的卷积核(K) 来模拟 Dilated 卷积的滑动就行了，等效 Kernel size 为 K + (K-1) * (D-1)。
$$
o= \left \lfloor \frac{W + 2P - K - (K-1)(D-1)}{s} \right \rfloor + 1
$$
即除了单独考虑一次以外，还要考虑删掉 dilation 跳过的部分。

#### Q4: 为什么需要白化

![tag](https://img.shields.io/badge/ML-1-brightgreen) 

一般在把数据喂给机器学习模型之前，“**白化（whitening）**”是一个重要的数据预处理步骤。

由于特征之间很可能具有很强的相关性，所以用于训练时输入是冗余的。白化的目的就是降低输入的冗余性；

白化一般包含两个目的：

1. *去除特征之间的相关性* —> 独立；
2. *使得所有特征具有相同的均值和方差* —> 同分布。

#### Q5: 如何进行白化

![tag](https://img.shields.io/badge/ML-1-brightgreen) 

白化中第一步是使特性之间的相关性降低，这一步 PCA 就可以做到了。至于第二步，只需要将方差变为相同就好。由于 PCA 的第一步计算了协方差矩阵，而其对角阵就是方差，因此对 PCA 中获得的新特征向量，如果把每个数值都除以他的标准方差就可以使全部的数值的方差变为 1 了，这就是 PCA 白化。

#### Q6: 为什么要进行 BN

![tag](https://img.shields.io/badge/DL-1-brightgreen) 

关键点：原文的解释 ICS，为什么会 ICS，ICS 会怎样，后续的研究证明不是ICS，更合理的解释。

1. BN 的原论文中说是解决了 **深度学习中的 Internal Covariate Shift** (协方差漂移) 的问题。

   > ICS是什么: 网络中每一层都会导致上层的输入数据分布发生变化，通过层层叠加，高层的输入分布变化会非常剧烈。Google 将这一现象总结为 Internal Covariate Shift，简称 ICS。
   >
   > ICS 会怎么样: 
   >
   > 1. 底层的参数需要不断适应新的输入数据分布，降低学习速度。
   > 2. 高层的输入的变化可能趋向于变大或者变小，导致上层落入饱和区，使得学习过早停止。
   >
   > 3. 由于每层的更新都会影响到其它层，因此每层的参数更新策略需要尽可能的谨慎 (lr 减小)。

2. 实际上后续研究表明，**尽管** ICS 问题在较深的网络中确实是普遍存在的，**但是并非** 导致深层网络难以训练的根本原因。实验一方面证明：即使是应用了 BN，网络隐层中的输出仍然存在严重的ICS问题；另一方面也证明了：在 BN 层输出后人工加入噪音模拟 ICS 现象，并不妨碍 BN 的优秀表现。

3. 实际上更多的研究倾向于认为 **BN 改善了优化过程**，通过 Normalization 操作，使得网络参数重整 ( Re-parametrize ），它**对于非线性非凸问题复杂的损失曲面有很好的平滑作用**，有实验证明了用 BN 后损失曲面的平滑程度得到了很大提升。

4. Goodfollow 的**花书**里有一些直观的解释：如果没有 BN 更改某一个权重会影响后续的很多层，**BN 使得我们能够独立的控制每一层激活值的幅度和均值。**也就是"reduces second-order relationships between parameters of different layers than a method to reduce covariate shift." **减少层间参数的二阶影响**。

#### Q7: PCA 的流程

![tag](https://img.shields.io/badge/ML-1-brightgreen) 

1. 计算矩阵的协方差矩阵（多个输入需要求平均）
2. 计算协方差矩阵的特征值和特征向量
3. 特征向量的装置左乘以原特征的数值得到新的特征的数值
4. 如果是想降维的话，特征值从大到小排列，选取一定信息量的特征值进行转换
5. 将降维后的新特征的值左乘特征向量，恢复到原始形式
这里第三步已经得到了不相关的特征的值，如果想要进一步白化，只需要将方差变为相同就好。

#### Q8: 深度学习的输入特征是否需要独立

![tag](https://img.shields.io/badge/DL-1-brightgreen) 

机器学习任务一般都要求 训练数据(特征)满足独立同分布 (IID)，但在深度学习模型或高级机器学习模型中，**由于算法本身的先进性**，即使 Non-IID，训练结果仍然较好。

机器学习模型一般对输入进行白化操作。但对于深度学习模型，标准的白化操作代价高昂，特别是我们还希望白化操作是可微的，保证白化操作可以通过反向传播来更新梯度。

因此是一个 trade-off。

但对于某些应用场景，使用 Non-IID 数据训练会出现意想不到的负面效果，比如模型准确度低、模型无法收敛等。

[补充阅读](https://zhuanlan.zhihu.com/p/81726974)



#### Q: 如何进行 BN

![tag](https://img.shields.io/badge/DL-1-brightgreen) 





#### Q: BN 的实现中有哪些参数，有什么用

以 PyTorch 为例

```python
torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
```

- `num_feat` 指明输入特征维度，因为 PyTorch 是动态图模型；
- `eps` 是在除标准差的时候防止溢出；
- `track_running_stats`  和 `momentum` 参数用于指明是否需要计算统计量的滑动平均，以及更新幅度。
- `affine` 参数用于指示是否需要 re-shift 和 re-scale。



#### Q: BN 存在什么问题，如何解决

1. 对 batch size 有个下界，如果太小会导致统计量估不准，更甚如果 batch size 为1，会因为方差为0，不能用BN。
2. RNN 中，因为每个 time step 的激活值的统计量是不一样的，不能共用一个 BN 层，也就是说需要对每个 time step 加 BN，会很复杂，而且需要存储每个 time step 的统计量。

#### Q: 介绍 BN 及其衍生，以及特性

首先目的都是通过 normalization 和 reshift 与 rescale 来重新参数化网络。BN：在 Batch 中计算每个特征的统计量，获得的统计量都是 C 维向量(对应 C 个特征)。WN：通过对层权重的重新参数化来获得与 BN 中除标准差类似的效果，与 "mean-only BN" 配合，解决 BN 在 batch size 较小时的问题。LN：对单个样本的所有特征计算统计量，获得的统计量是 B 维向量(对应 B 个特征)，只与样本本身有关，因此可以用于 RNN 之类的结构。IN：认为网络应该与原始图像的对比度无关，所以对单个样本的每个特征在 HW 方向上计算统计量，获得的统计量是 BxC 的矩阵，适用于风格迁移，GANs，不适合于一维情况(绝大多数 NLP 问题)。GN：一方面认为 LN 把所有 channel 混为一谈忽略了 channel 间的独立性(减少了模型的表现力)，另一方面认为 IN 把所有 channel 视作独立 忽略了 channel 间的相关性，所以提出折中的方案。

#### Q: 多 GPU 时，BN 如何处理

因为通常多 GPU 训练过程是将网络复制到不同的 gpu 上，然后进行 forward 和 backward，之后只需要 collect gradient，再更新主 gpu 上的网络，然后下一个 iteration 再复制一遍。相当于缩小了mini－batch size，也就是说BN使用的均值和标准差以及反传关于均值方差的梯度都是单个 gpu 算的，需要用 synchronzie BN，在卡之间进行通信，对统计量和梯度进行同步。

#### Q: BN 一般加在哪，与激活函数的位置如何

BN 原文是放在前面的，但这个问题仍然有争论。不管 BN 和激活函数的位置关系，dropout 肯定是紧跟着激活函数，而且一般用了 BN 是不用 dropout 的。

#### Q: 迁移学习时 BN 的均值与方差如何处理

whether we should use the mean and variance computed on the **original dataset** or use the mean and variance of the mini-batches.

这个问题没有明确的答案，尽管绝大多数深度学习框架都选择重新计算统计量，但重新计算的话会导致新的统计量和网络其它参数不匹配，也许保持旧统计量会更好，或者可以考虑两者折中。









#### Q10: 分类网络发展史





