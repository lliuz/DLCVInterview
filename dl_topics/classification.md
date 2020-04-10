### VGGNet

首个堆叠卷积层取得突破性提升的网络，主要由 3x3 卷积组成。

#### 7x7卷积核和3x3的卷积核的区别

第一，增加非线性而不改变感受野。利用三个非线性的激活层来代替一个，可以增加网络的鉴别能力；第二，单个7x7卷积核的参数量为$$7^2C^2=49C^2$$，而三个级联的 3x3 卷积核参数量为 $3(3^2C^2)=27C^2$ 参数量更少。

### GoogLeNet

也就是 inception v1，是包含 Inception 模块的全卷积结构，与 ResNet 同期。核心思想是：通过构建由多个子模块组成的**复杂卷积核**来提高卷积核的学习能力和抽象能力。

(图 inception 模块)

还使用了辅助分类器来提高稳定性和收敛速度，在不同层执行分类任务，不同层都可以计算梯度，然后使用这些梯度来优化训练。（因为层太深了，为了解决梯度消失）

```python
# The total loss used by the inception net during training.
total_loss = real_loss + 0.3 * aux_loss_1 + 0.3 * aux_loss_2
```

### Inception

Inception v2 和v3 发于同一篇论文，借鉴了 GoogLeNet 和 VGGNet，比如卷积核分解，把 5x5 分成两个 3x3，并且把 3x3 分解成 1x3 和 3x1(有并联和级联两种分法)，在 v2 中三种结构都有用到。

v3 分解了 7x7 的卷积核，并且使用了 BN 和标签平滑，标签平滑指每个类都分配一些权重，而不是将全权重分配给 ground truth 标签，防止网络过于自信，减少过拟合。

### ResNet

Motivation: 很深的网络的前几层可以被一个浅网络(功能完全一致)代替，其它层就只需要实现恒等映射(identity map)，

![1558193084224](../../../Sync/Notes/%E6%B1%82%E8%81%8C%E7%AC%94%E8%AE%B0/18%E5%B9%B4%E5%AE%9E%E4%B9%A0%E5%87%86%E5%A4%87/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C.assets/1558193084224.png)

所以如果我们让额外的层去学浅层以外的东西，是不是能获得更好的效果。

所以 Residual block 由一个 identity 和学习的残差组成

![Residual Block](../../../Sync/Notes/%E6%B1%82%E8%81%8C%E7%AC%94%E8%AE%B0/18%E5%B9%B4%E5%AE%9E%E4%B9%A0%E5%87%86%E5%A4%87/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C.assets/1558193250763.png)

（当维度不一致时，使用 1x1 的卷积核去调整维度以及 feature map 的大小。）

#### 预激活

在初版的 ResNet 中，1k 层的效果不如 100层，但在之后的一篇论文中，He 讨论了激活函数的位置的影响，并表示预激活(卷积前加 ReLU，也就是说在 Identity 加法前没有 ReLU)的效果最好，因为梯度能更直接的传到卷积层。在这版里 ResNet 1001 层的结构取得了最好的效果。

![1558193571322](../../../Sync/Notes/%E6%B1%82%E8%81%8C%E7%AC%94%E8%AE%B0/18%E5%B9%B4%E5%AE%9E%E4%B9%A0%E5%87%86%E5%A4%87/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C.assets/1558193571322.png)

更详细的解释是激活函数一般用于高维空间增加非线性，而addition操作只是简单的线性操作，使用ReLU没有帮助反而导致信息缺失。

#### 为什么能解决梯度消失

实际上，使用 ReLU 的网络中，梯度消失(由于 Sigmoid 等激活函数在饱和区梯度几乎为0) 现象已经很少，但 Skip Connection 能更直接的将梯度传递到浅层。

### DenseNet

采用 concat 代替 resnet 中的相加，设置增长率 k 表示每层只输出 k 个channels，一般 k 取很小(12 或 32)，下一层的输入为前一层输出与再之前的输出的 concat 结果。

(dense connect 图)

直接 concat 会导致通道数特别多，因此只是在 dense block 中使用，具体实现中还有一些细节：

1. 设计 bottleneck 层，先用 1x1 的卷积降低通道数为 4k，再通过 3x3 卷积。即 BN-ReLU-C1x1-BN-ReLU-C3x3
2. 在每个 dense block 连接处，即 transition layers，加入一个 1x1 卷积+池化输出 m 个 feature map。文中称之为 compression，用于减少 $$\theta$$ 的通道数。

(dense net 结构图)

#### 为什么要用 concat 代替相加

文中称相加阻碍了网络中的信息流。而 concat 使每一层都与其他层有关系，都有”沟通“，这种方式使得信息流最大化。

### SENet

SE(Sequeeze and Excitation) 并不是一个完整的网络结构，而是一个子模块，可以嵌入到其他的分类或者检测模型。所以有 SE-ResNet 之类的结构

核心思想为：对Channel之间的信息进行显式建模。通过网络根据 loss 函数去学习特征权重，使得有效的 feature map 权重大，而无效的小一些，从而使得模型精度更高。

Sequeeze and Excitation Block：

(图)

对 CHW 的 feature map 先做 global average pooling 得到 C11 的向量，通过一个维度不变的 FC (Excitation) 进行线性变换，把这个 C11 乘到原来的 CHW 上。

### MobileNet

把卷积操作分解为一个 Depth-wise 卷积和 一个 1x1 的 Point-wise 卷积，称作 depth-wise separable convolution。

(图 三种卷积核对比)

其中 1x1 卷积就是标准卷积，而 Depth-wise 卷积与传统卷积实现不同，其每组参数的大小为 $$k \times k \times 1$$，而不是 $k \times k \times C_{in} $，且组数为 $C_{in}$，也就是说每一组卷积核都只与输入的一个 channel 一一对应，而非传统卷积(参考卷积笔记)中每组参数中的每个卷积核在各个维度上卷积再求和得到一个输出 channel。

例子：输入 HxWx3，原来经过 3x3x3x16 的标准卷积卷成 16 通道，现在拿 3x3x1x3 的 3x3 深度卷积，卷成3通道，再通过 1x1 卷积卷成16通道。

传统卷积的时间复杂度为 $$HW D_k^2 C_{in} C_{out}$$，而DSC为 $$HWD_k^2C_{in} + HWC_{in}C_{out}$$，两者之比为
$$
\frac{HWD_k^2C_{in} + HWC_{in}C_{out}}{HW D_k^2 C_{in} C_{out}} = \frac{1}{C_{out}} + \frac{1}{D_k^2}
$$
对 3x3 的卷积核来说 一般计算量减少约8~9倍。

另外 MobileNet 中没有采用 Pooling 层，而是将 Depth-wise Conv 的 stride 设为 2。

#### 进一步减少网络 size

文中通过设置缩小因子 $\alpha$ 统一的减少 channel 数到 $\alpha C$。

#### MobileNet v2

- **在DW之前增加了一个PW卷积。** 这么做的原因，是因为 DW 卷积由于本身的计算特性决定它自己没有改变通道数的能力，上一层给它多少通道，它就只能输出多少通道。所以如果上一层给的通道数本身很少的话，DW 也只能很委屈的在低维空间提特征，因此效果不够好。现在 V2 为了改善这个问题，给每个 DW 之前都配备了一个 PW，专门用来升维。因此有一个升维系数。
- **去掉了第二个 PW 的激活函数**。作者认为激活函数在高维空间能够有效的增加非线性，而在低维空间时则会破坏特征，不如线性的效果好。由于第二个 PW 的主要功能就是降维，因此按照上面的理论，降维之后就不宜再使用 ReLU6 了。
- MobileNet V2 借鉴 ResNet，同样使用 Shortcut 将输出与输入相加。

对比 ResNet 与 MovileNet v2 的基本模块

![1558162702294](../../../Sync/Notes/%E6%B1%82%E8%81%8C%E7%AC%94%E8%AE%B0/18%E5%B9%B4%E5%AE%9E%E4%B9%A0%E5%87%86%E5%A4%87/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C.assets/1558162702294.png)

最大的区别在于 ResNet 在 3x3 卷积前是降维，而 MobileNet 是升维。

### ShuffleNet

在 MobileNet v2 的基础上进行改进，将 MobileNet 中的 point-wise convolution 换成了 channel shuffle。

#### Group Convolution

在 AlexNet 中提出，将卷积操作分成 g 个组，独立计算。首先把 $$H\times W \times C_{in}$$ 的输入分成 g 组，每组为$$H\times W \times C_{in}/g$$，经过 $$C_{out}$$ 组卷积，每组参数为 $$k\times k \times C_{in} / g$$，输出 $$H^\prime \times W^\prime \times C_{out}$$。当组数 g 和 $C_{in} $ 相等时称作 Depth-wise Convolution.

#### ShuffleNet Unit

![preview](../../../Sync/Notes/%E6%B1%82%E8%81%8C%E7%AC%94%E8%AE%B0/18%E5%B9%B4%E5%AE%9E%E4%B9%A0%E5%87%86%E5%A4%87/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E5%9F%BA%E7%A1%80/%E5%88%86%E7%B1%BB%E7%BD%91%E7%BB%9C.assets/v2-d2f36404c00c82af1b616fa0f1be7c13_r.jpg)

图 (a) 就是 MobileNet v2 的基本单元，复杂度为 $$HW(2 C_{in} C_{out} + D_k^2 C_{out} )$$

图 (b) 是 ShuffleNet 的基本单元，复杂度为 $$HW(2C_{in}C_{out}/g + D_k^2C_{out}) + \text{shuffle cost}$$ 

图 (c) 是 stride 为 2 的 Shuffle Unit.

#### 为什么要 Channel Shuffle

Group Convolution 导致模型的信息流限制在各个Group内，组与组之间没有信息交换，Channel Shuffle 引入组间信息交换的机制。



