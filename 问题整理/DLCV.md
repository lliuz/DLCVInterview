求感受野

BN层的moving mean怎么求得

BN层反向传播，怎么求导

视频分类难点有哪些，untrimmed的难点

attention起源是用在哪里？pixel还是frame，是soft还是hard

SlowFast 的流程，为了解决什么问题

Non-local 的原理

inception v1, v2、v3区别，用来解决什么问题，如何减少参数量

介绍一下 momentum

resnet好处

余弦距离和欧氏距离的区别，其它距离度量

mobilenet v1 v2, shuffleNet怎么实现


介绍残差网络

有没有自己写过层，反向传播之类的

反向传播求导，给了个例子链式求导，pool如何反向传播

经典光流怎么计算

weight decay 如何设置

为什么使用1x1卷积核

最近的激活函数，为什么relu替代sigmoid

学习率调整策略

二阶优化器

实现卷积操作

最新的 data augmentation 方法

### FLOPs计算:

输出的HW乘参数量，参数量为`Cout*(K1*K2*Cin + 1)`

### Q: SENet, SKNet



### FCOS

Retinanet 结构，回归三个分支：分类，回归，centerness。

在 gt box 分配到的 feature level 的每个正样本位置都回归 l t r b 四个位置值，最后 nms 时的置信度为分类得分 × centerness值。

Centerness回归目标为 min/max 形式，即以 GT 中心点为峰值点的 heatmap。

> 为了解决 gt box 存在的重叠问题, FCOS提出两种方法：
>
> 1. FCOS 不像基于 anchor 的检测器在不同的特征层中使用不同 size 的 anchor，在 FCOS 中直接对每个 feature level 分配了回归范围 $[m_{i-1}, m_i]$，$m_i$ 为第 i level 能回归的最大 size。m2~m7分别为 0, 64, 128, 256, 512, inf。
>
>    分配 anchor 的原则是: 如果一个 gt box 的 $\max(l, t, r, b) > m_i \or \min(l, t, r, b) < m_{i-1}$ 则对于这个 gt box 对于这个 level 设为负样本。 
>
> 2.	按 level 分配的方法已经能够一定程度上缓解重叠问题，但如果size相近的重叠还是没办法。直接选取最小的ground truth作为其回归目标。

> Trick:
>
> 根据论文，在不同的特征层中共享head，不仅使得检测器的参数更有效，同时提升了检测性能。不同的特征层级要求回归不同的size范围，因此在不同的特征层中使用相同的head是不合理的，因此使用exp(six)替换原先的exp(x)，si是一个可训练的标量，不同的层级中可以自动调整，最终提高一点性能。

常见 anchor 正负样本分配原则

### FreeAnchor, ATSS

### Q: AutoAssign:

之前目标检测的问题：

- anchor 的 label assignment 都基于 center，但是 center 不一定是最具表现力的
- 超参巨多
- scale 和 spatial assignment 不确定性多。

提出 baseline，FCOS 最简单的版本，所有在 box 内的点都算正样本。

基于 baseline 的改进：

- 改进 FCOS 的 center prior 分支，修改为每类学习一个均值方差，通过高斯核进行距离的度量。问题: 仅与类别有关，可能会造成对旋转的不match，固定与否或是否类别共享影响都不大 0.2 mAP左右。
- reweighting 解决样本不平衡，如P3 中大物体会有非常多的正样本。
- 分离判断是否是物体 objectness 和分类分支。用一个隐式的前景背景二分类对分类预测做一个乘性叠加。注意这个分支是没有额外监督的（也没法有），就是单纯地去scale一下分类的预测。

思考：

如果检测的类别只有1类，也就是说只有前景和背景，那么这个分支还需要么？如果答案是是，那么它就应该还扮演其他的功能。

1） 重叠框的问题是否的确如我分析所说，被优雅地解决掉了？

> FCOS 中重叠框是通过分配给面积小的 gt，AutoAssign 不需要管你重叠不重叠，我就每个gt框各自算就行，反正到时候不同gt框对应的w+会慢慢学习慢慢演化出来重叠区域的location究竟归属为哪个。

2） 训练的时候是否有一些变量需要detach掉？比如w+, w- 么？

3） ImpObj分支是否有更深入的理解？

### Q: MOT 最新工作:

1. ReID 改进，tracking-by-segmentation (百度 ECCV oral)，把segmentation内的点作为2d点云进行MLP+pooling
2. learnable matching。End-to-End Multi-Object Tracking with Global Response Map，Chained-Tracker，Multi-object Tracking via End-to-end Tracklet Searching and Ranking
3. 多帧输入，CenterTrack, ChainTrack等。

### YOLO系列的发展历史

YOLO V1 

\-     卷积输出7x7的格子，每个格子预测2个box，30维，分别是8(offset) + 2(objectness) + 20(num_class)。一共输出7x7x2=98个box，

\-     对相互靠的很近的物体，还有很小的群体，检测效果不好，这是因为一个网格中只预测了两个框，并且只属于一类。

YOLO V2 

\-     加入BN，更高分辨率输入(224->448)。

\-     引入anchor，提升了召回率，基于IoU的kmeans聚成5类anchor，得到13x13x5个box。

\-     多尺度训练，无论最后得到的10x10xCout或者20x20xCout，都是经过1x1x(5+C)，统计计算loss。

YOLO V3具有多尺度检测，更强大的特征提取器网络以及损失功能的某些变化。

\1.    Backbone 是Darknet53(v1是19层，类似resnet的残差结构)；

\2.    多尺度，assuming the input is 416x416, so three scale vectors would be 52x52, 26x26, and 13x13, corase-to-fine，在浅层时结合深层的特征，每个grid放3个anchor，每个anchor预测4(offset)+1(objectness)+C(num_class)。

![Image for post](D:\Notes\Interview\clip_image014.jpg)

![Image for post](D:\Notes\Interview\clip_image016.jpg)

\3.    Loss 类别loss不再用softmax，因为有些数据集存在重叠标签，而是用多个bce，进行多分类，精度不会下降。对于object的txty计算mse

不使用focal loss来处理正负样本不平衡问题，而是直接过滤掉objectness < 0.1的负样本，再根据正负样本比例对loss直接加权。

 

### 如何用Kmeans来聚类Anchor

\1.    随机选取K个box(样本)作为初始anchor(类中心)；

\2.    使用IoU度量(1 - IoU)，将每个box分配给与其距离最近的anchor；

\3.    计算每个簇中所有box宽和高的均值，更新anchor；

\4.    重复2、3步，直到anchor不再变化，或者达到了最大迭代次数。

读取所有的boxes，对K在[2,10]这个区间内进行多次聚类，然后画出平均IoU随K值的变化曲线，从中发现最佳的anchor数量。找出anchor后再人为的把anchor布置到不同的feature level上。

 



 

### YOLO 中anchor和其它的检测方案里anchor的区别

YOLOv3 里对每个gt box做了scale的分配，根据和anchor的重叠率直接分配到对应的scale的对应位置的对应anchor上。而像FRCNN, ssd等都是通过计算anchor和GT box的IoU，用阈值分配正负样本的。这就导致了yolo中的正负样本更加不平衡(但也加入了人为先验，避免搜索空间过大)。

YOLO的每个anchor box会单独预测一个objectness，ssd中是把这个和到普通类别中的 (每个anchor box直接输出4+21维) ，FRCNN中预测的是4 + 2维，因为只需要分前后背景二分类。

 ### IoU loss

https://zhuanlan.zhihu.com/p/143747206

```python
IoU_loss = 1 - IoU	# 问题1. IoU 等于 0 时不可导， 问题2. IoU 相同时对应的框调整方案是一对多的
GIoU_loss = 1 - (IoU - 差集/最小外接矩形)	# Generalized IoU 问题. 在框内且面积一样时，对应的情况是一对多的
DIoU_Loss = 1 - (IoU - D2^2 / Dc^2) # Distance IoU: Dc 最小外接矩形对角线的长度, D2 中心点距离. 问题. 没有考虑到长宽比，中心点一致的情况下一对多。
CIoU_loss = DIoU_loss + 长宽比因子	# NMS时不需要用 CIoU，一般用的还是 DIoU, 因为没有GT, 不用考虑长宽比
```



### 目标检测中预处理阶段要干什么

要对真值进行编码，

 

### 一阶段二阶段anchor free各自的优缺点

一阶段定位性能比较差，体现在IoU阈值调高后mAP会下降很严重(见YOLOv3)，原因主要是特征没有区分性。

**Anchor free:** https://www.zhihu.com/question/364639597

one-stage 由于没有这降低候选框的步骤，因此，候选区域的数量大大超过 two-stage 方法，因此，在精度上，two-stage 仍然优于 one-stage 方法，但是在速度和模型复杂度上， one-stage 占优势。

优化和稳定性上，ab总是要表现得好。

又可以分为基于关键点检测(centernet, cornernet)和基于密集点检测(FCOS)，前者一般通过在高分辨率层输出prediction，从而提升box数目，取得较高的recall，但容易拼接错误或定位错误，精度较低。后者直接预测非常多的box，recall较高，需要用像centerness的估计去过滤多余的box。

### FasterRCNN

https://zhuanlan.zhihu.com/p/61659240



### 如何解决anchor free中小目标和大目标的检测问题

YOLOv3: 在1/8, 1/16, 1/32 分别设置anchor 3个anchor，输出10k量级个boxes

SSD: 在不同scale上设置anchor数不同，1/8~1/256 6个尺度，分别设置[4,6,6,6,4,4] 个anchors，每个anchor表示多个类别，一共预测数是 N_anchor_box * (21+4)

FRCNN: faster rcnn只在con5_3 (1/32)进行roi pooling (NMS + top 2000 from 12000)，因此无法很好的检测小目标，需要加FPN，在M2~M5(1/4~1/32)上进行预测。

Centernet: 在1/4 scale预测，512x512输入，输出10k量级的boxes。

FCOS: 1/8到1/128 5个尺度预测，800x1024输入，

### Hourglass 结构的好处

Hourglass 结构的设计主要是源于想要抓住每个尺度信息的需求。 例如一些局部信息对识别一些特征（例如脸，手等）很重要，而对于最后姿态的估计需要对整个身体有一个好的理解，这就要抓住很多局部的特征信息并结合起来。 人的朝向，他们四肢的排列，相邻关节的关系都是在不同尺度图像中最好辨认的。

### RoI Pooling 和 RoI align

平均的切割成 k*k 个bin，做 max Pooling。在Mask R-CNN中，采用了浮点数切割并使用了双线性插值，即RoI Align。

 

**为什么回归BBox****的宽高要采用Log****的形式**

**一方面为了确保tw th****大于0****，**

### Q: Detection 未来

1. 无监督自监督的方法提供 backbone(Moco,SimCLR, BYOL, SimCLR v2)，从而引出很多问题，比如如何从 noisy label 中学习(temporal average, muture distillation)
2. DETR 等，目前训练开销太大，500 epoch

### Q. Keypoints based 和 Dense prediction based anchor 的关联

分类分支上：Dense prediction 的方法通常需要额外预测一个 centerness / objectness (FCOS, AutoAssign)，而keypoint based方法通常是直接预测 objectness，在使用focal loss 的形式上 可以参考 GFocal loss(把Dense prediction based anchor 和 keypoints based 在分类分支上拉进)。



### Q: RepPointsV1, RepPointsV2

由卷积预测每个点的一组offsets，作为代表性点，

PAA：用于目标检测的IoU预测的概率Anchor分配

### Q: 最新自监督学习方法

BYOL, SimCLR, MOCO, SimCLR++, MOCOv2

### 自监督学习

早期(Pretext Task, 委托任务): 如果没有得到 Label ，就利用 Rule-Based 的方法生成一些 Label 。预测旋转、拼图、上色。

突破与定调(Contrastive Learning, 对比学习): 目标是 score(f(x), f(x+)) >> score(f(x), f(x-))。这里x+指的是与x 相似的数据（正样本），x−指的是与x 不相似的数据（负样本）。score 函数是一个度量函数，评价两个特征间的相似性。x通常称为“anchor” 数据。为了解决这个问题，我们可以构建一个softmax分类器。类似的，对于N-way softmax 分类器，我们构建一个交叉熵损失，一般被称为InfoNCE 损失。为了最小化InfoNCE损失，可参考Poole的文章。

\-     Contrastive Learning能从Data 中获得相当丰富的信息，不需要拘泥在 Patch 上。

\-     使用ResNet这种 Backbone （而非早期 paper 强调VGG 能得到更好的 representation）、

接下来的文章，都基于这样的前提来Improve 。

MOCO

![preview](D:\Notes\Interview\clip_image018.jpg)

, SimCLR

https://www.cnblogs.com/gaopursuit/p/12242946.html

https://www.ershicimi.com/p/5bf00bd701bfee866ccc4d8cc1f3df75

https://www.ershicimi.com/p/a430485adef59e46bb63ca248bd799d5

 

### 自监督学习的特性:

\-     表示特征h到计算loss的特征g之间增加一个非线性的MLP至关重要，避免 Visual Representation 直接丢给 Contrastive Loss Function 计算，原因是这种比较 Similarity 的 Loss Function 可能会把一些信息给丢掉。

\-     模型扩展可以显著提高性能，SimCLR指出更宽或更深的网络，甚至是更久的训练时间(800epoch on Imagenet)带来的增益比监督学习更明显。

\-     大batchsize对自监督学习的增益比监督学习更明显。



### Q: 自监督学习的经验

不同网络结构在不同pretext task上学习的性能差异大，通常都用R50

不带skip-connection的模型在alexnet高层会有性能下降，但带skip-connection则不会。



### 光流做超分的基本思路

### 光流做插帧的基本思路

### Q. 3D 卷积如何加速

完全等效：通过分组后用2d卷积来实现，实际上也是调的im2col，gemm，col2im。

近似：用二维和一维的组合来近似。

### C3D, P3D, I3D的区别以及改进的原因

C3D不是第一个提出3d卷积的论文，但是首个使用3d卷积处理较长视频帧序列的，

I3D，即inception-bn C2D，拓展版本的C3D。对C3D的改进

### I3D 改进和motivation

I3D: 提出Kinectics，因为3D卷积网络，没有特别好的pretrained model，所以作者提出使用2D inceptionv1的结构进行膨胀(例如3x3卷积变成3x3x3，但部分maxpool在时间维度上stride不为2)，由此，预训练的参数直接来自2d网络复制。另外实验发现在Kinetics预训练，再去小数据集finetune有效。和其它文章不同，two-tream的两个分支是单独训练的，测试时融合它们的预测结果(后续文章几乎都这样)。

![img](D:\Notes\Interview\clip_image004.jpg)

 

*在InceptionV1**中，第一个conv**层stride**是2**，然后是4**个max-pooling**，stride=2**，最后的7\*7average-pooling**层。输入25**帧每秒，发现，在前两个max-pooling**层中不加入空间pooling**比较好，(**我们使用了1\*3*3de kernel**，stride=1)**同时使用了对称的kernels**和stride**在所有其他的max-pooling**层，最后一个average-pooling**层使用2\*7*7**的kernel*

 

motivation: 视频可以通过复制图片序列得到，且视频上的pooling激活值应该与单张图片相同。由于是线性的，可以将2D滤波器沿着时间维度重复N次。

### P3D 改进及其motivation

由于I3D把分类做到98%之后已经又没有什么提升空间了，所以人们转向提升效率，著名的有P3D（伪3D），S3D（分开3D），R(2+1)D（把3D变成2+1），这三种模型都考虑把3D CNN拆开，以求得到更好的效率。

P3D用一个1×3×3的空间方向卷积和一个3×1×1的时间方向卷积近似原3×3×3卷积，通过组合三种不同的模块结构，进而得到P3D ResNet，2D卷积的预训练参数来自2D imagenet。

![img](D:\Notes\Interview\clip_image006.jpg)

![img](D:\Notes\Interview\clip_image008.jpg)

![img](D:\Notes\Interview\clip_image010.jpg)

R (2+1)D，和P3D非常相似，不同点在于在所有层中都使用单一类型的(2d+1d)时空残差块，并且不包括bottlenecks，因为为了保持参数量一致2d卷积是升维的。

![img](D:\Notes\Interview\clip_image012.jpg)

为什么要拆分3D卷积

\1.    计算量参数量

\2.    增加表示能力(增加了非线性激活层)

\3.    迫使三维卷积分离空间和时间成分，使优化更容易。

\4.    复用预训练参数

双流的视频识别网络，如何避免使用光流

可以用KD的方法，也可以用OFF之类的学习一种更容易获得的输入(最直观的就是输入RGBDiff)。TSM之类的卷积shift也可以结合时序信息。



### Q. GAN的几种损失函数形式

https://zhuanlan.zhihu.com/p/72195907

![img](D:\Notes\Interview\clip_image002.jpg)

### Q. 如果在光流网络上加attention，怎么加。

https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhu_An_Empirical_Study_of_Spatial_Attention_Mechanisms_in_Deep_Networks_ICCV_2019_paper.pdf

### Q: 常见正负样本处理策略:

https://ranmaosong.github.io/2019/07/20/cv-imbalance-between-easy-and-hard-examples/

OHEM: 首先计算出每个预测框的loss， 然后按loss从高到低来排列每个预测框， 然后为每张图片选择k个损失最高的预测框作为Hard Examples，只对这些样本的梯度进行反传。存在问题，会采到位置相似的框，所以先进行一次NMS再排序。

Focal loss: 代价函数改为，-a y(1-p)^gamma ln(p)，其中y和p都是one-hot编码，如果是二分类，需要展开为两种情况，a为类别平衡因子一般不需要加，gamma为参数一般取2。

其它简单方法: 直接加权，或者直接按比例采样负样本。具体实现

https://github.com/fudannlp16/focal-loss/blob/master/focal_loss.py

其它新方法: GHM-R Loss, PISA…

### 简介联邦学习

https://zhuanlan.zhihu.com/p/87858287

- **什么是联邦学习** **联邦学习可以理解为数据相互不可知的前提下的分布式训练**。具体怎么实现不可知的方法可以是加密，差分隐私，可以是训练时传递模型参数而不传递模型梯度的方法，甚至可以是不共享模型结构的方法。众所周知，AI最主要的是数据，但对于大规模的AI应用，数据的深度(数据量)和广度(数据维度)很难兼顾。由于行业竞争，隐私安全和复杂的管理程序，即使同一公司的不同部门之间的数据集成也面临着巨大的阻力。**联邦学习就是打破数据源之间的障碍，在保证数据独立性的前提下进行模型训练。**

- **联邦学习分类** 横向联邦学习、纵向联邦学习和联邦迁移学习。

  横向联邦的特点是特征(业务)重叠多，用户重叠少的场景，比如不同医院、不同银行之间。方法有梯度加密的 PS 形式(各个参与者本地训练，梯度上传，模型聚合，模型下载)。联邦平均(Federated Average)，直接通过模型平均，只需传递模型，无需传递梯度信息。横向联邦下每台机器下都是相同且完整的模型。

  纵向联邦的特点是特征(业务)重叠少，用户重叠多的场景，如银行、购物、社交app之间进行联邦学习。在整个过程中参与方都不知道另一方的数据和特征，且训练结束后参与方只得到自己侧的模型参数，即半模型。

  联邦迁移学习指当参与者间特征和样本重叠都很少时可以考虑使用联邦迁移学习，如不同地区的银行和商超间的联合。主要适用于以深度神经网络为基模型的场景。联邦迁移学习的步骤与纵向联邦学习相似，只是中间传递结果不同。

- **联邦学习应用** 90% 的场景是纵向联邦学习的问题，即 ID 相同、在数据维度上联合建模。以广告为例，一方有用户的基本画像，主要是静态标签，而另一方则有其购买行为、兴趣等信息，需求是如何安全有效地联合两方的数据。

- **联邦学习挑战** 
  - 用户普遍关心安全问题，即学习过程是否安全。除了最终产出的模型参数，过程中不应该泄露任何一方的数据信息，也不可以反推数据信息的中间结果。
  - 参与者掉线或网络延迟的情况；
  - 加密带来的计算量和通信带宽剧增；
  - 深度学习等复杂算法如何改造成联邦学习模式；

### 简介同态加密

同态加密是指具有特殊代数结构的一系列加密方案，该结构允许**直接对加密数据执行计算而无需解密密钥**。

挑战：实现更简单，更快的同态加密方案

### 迁移学习的核心

迁移学习的核心是，**找到源领域和目标领域之间的相似性**，举一个杨强教授经常举的例子来说明：我们都知道在中国大陆开车时，驾驶员坐在左边，靠马路右侧行驶。这是基本的规则。然而，如果在英国、香港等地区开车，驾驶员是坐在右边，需要靠马路左侧行驶。那么，如果我们从中国大陆到了香港，应该如何快速地适应 他们的开车方式呢？诀窍就是找到这里的不变量：不论在哪个地区，驾驶员都是紧靠马路中间。这就是我们这个开车问题中的不变量。 找到相似性 (不变量)，是进行迁移学习的核心。

### 简介 few shot learning



### Word2vector

https://zhuanlan.zhihu.com/p/59396559

### Transformer



### 分类网络

#### ResNet 和 DenseNet 及其不同

那么ResNet解决了什么问题呢？

训练深层的神经网络，会遇到梯度消失的问题，影响了网络的收敛，但是这很大程度已经被标准初始化（normalized initialization）和BN（Batch Normalization）所处理。

当深层网络能够开始收敛，会引起网络退化（degradation problem）问题，即随着网络深度增加，准确率会饱和，甚至下降。这种退化不是由过拟合引起的，因为在适当的深度模型中增加更多的层反而会导致更高的训练误差。

ResNet就通过引入深度残差连接来解决网络退化的问题，从而解决深度CNN模型难训练的问题。

#### 相同层数，densenet和resnet哪个好，为什么？

#### resnet两种结构具体怎么实现，bottleneck的作用，为什么可以降低计算量，resnet参数量和模型大小

#### Inception系列的演化

#### Mobilenet v1 v2，shufflenet

### 卷积层

#### 卷积特点

局部连接，权值共享

#### 卷积, 反卷积, 扩张卷积, 可变形卷积, 计算过程

#### 深度可分离卷积, 原理，参数量计算量，口述计算，减少了多少

#### deformable conv 怎么做

具体怎么学的，对偏移有没有什么限制

#### 1x1卷积作用

变换 channel 数

通道融合



tensorrt怎么量化，float32怎么变int8，怎么变回来