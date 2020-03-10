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

#### Q3: 卷积层输出 size 计算

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

#### Q4: BN 原理与存在的问题



#### Q5: BN 参数解释



#### Q6: 介绍 BN 及其衍生，以及特性



#### Q7: 多 GPU 时，BN 如何处理



#### Q8: BN 与激活函数的位置



#### Q9: 迁移学习时 BN 的均值与方差如何处理



#### Q10: 分类网络发展史





