### Detection Tethods

#### RCNN

具体流程：

- 用 [Selective Search 算法](https://zhuanlan.zhihu.com/p/27467369) 从每张图选出的 ~2k 个 proposal；
- 用 AlexNet 对 proposal (裁剪+拉伸后)提取特征；
- 对 proposal  的特征送入 SVM 分类；
- NMS 去重；
- 去重后的结果送入回归网络，微调位置。

也就是说， RCNN 需要依次训练三个部分：backbone Network，SVM，位置回归网络。

**Backbone 网络：**

RCNN 使用 ImageNet 上预训练的 AlexNet 作为 backbone，再在检测的数据集上 finetune 最后一层全连接层(也就是直接分类，正负样本以 IoU = 0.5 为界)，通过实验发现拿最后一层作为特征效果最好。

**SVM 分类器：**

SVM 是二分类模型，多分类时实际上是通过每一类训练一个 SVM 实现的。

正负样本的选取规则：通过反复的实验，RCNN 的 SVM 训练将 ground truth 样本作为正样本，而  IoU < 0.3 的样本作为负样本，中间的样本丢弃。

**位置回归网络：**

从 backbone 中 (pool5) 引出分支，FC 输出每个 proposal 的位置。

回归形式为：proposal 的**中心坐标偏移量**和 proposal 的**长宽变化量**，

为了方便训练和调参，对不同 size 的 proposal 一视同仁，采用归一化的形式，具体的回归目标为：
$$
\begin{aligned}
t_{x} &=\left(G_{x}-P_{x}\right) / P_{w} \\
t_{y} &=\left(G_{y}-P_{y}\right) / P_{h} \\
t_{w} &=\log \left(G_{w} / P_{w}\right) \\
t_{h} &=\log \left(G_{h} / P_{h}\right)
\end{aligned}
$$
其中 $G$ 为真值的中心位置与长宽，$P $ 为 Proposal 的中心位置与长宽。

可以注意到，对于长宽的变化量，采用了对数域，理由有：

1. 最重要的原因：从预测的 $t$ 恢复出 size 的过程为 $w = w_p \exp^t$， 这样可以确保预测的 bbox 的 size 是正的；

2. 当预测的 $t < 0$  时，size 的缩小将会变慢(在预测量变化很大时，size 变化很小)，

   反之，$t > 0$ 时，size 的增大会变快，能够加快收敛。

> Q: R-CNN 中正负样本选择的方式
>
> Backbone 网络的 finetune 中，采用 IoU = 0.5 为界进行训练，因此正负样本数会比较多。
>
> 而 SVM 训练的时候，只以 GT 作为正样本，IoU < 0.3 为负样本，控制了样本量。
>
> Q: 为什么 RCNN 中有了 CNN 分类，还要再用 SVM 分类?
>
> 文中在 SVM 分类训练的时候是有用手动的 hard negative mining 的，结合 SVM 适用于小样本的特性，能够给出更精确的分类结果。
>
> Q: RCNN 中 bbox 回归的具体形式
>
> 一方面为了方便调参，对不同 size 的 proposal 一视同仁，采用归一化坐标的形式；
> $$
> \begin{aligned}
> t_{x} &=\left(G_{x}-P_{x}\right) / P_{w} \\
> t_{y} &=\left(G_{y}-P_{y}\right) / P_{h} \\
> t_{w} &=\log \left(G_{w} / P_{w}\right) \\
> t_{h} &=\log \left(G_{h} / P_{h}\right)
> \end{aligned}
> $$
> 另一方面对于长宽的变化量，采用对数域转换后再回归的方式，原因：
>
> 1. 从预测的 $t$ 恢复出 size 的过程为 $w = w_p \exp^t$， 因此可以保证预测的 bbox 的 size 是正的；
>
> 2. 当预测的 $t < 0$  时，size 的缩小将会变慢(在预测量变化很大时，size 变化很小)，
>
> 反之，$t > 0$ 时，size 的增大会变快，能够加快收敛。
>
> Q: 手写 NMS 并分析复杂度以及优化
>
> 对单类候选框结果按得分降序排序，依次遍历每个候选框，每次把当前框加入保留集合，删除所有 IoU 超过阈值的框。涉及两次遍历，时间复杂度为 O(n^2)。 其中 IoU 计算过程的内循环中一般可以优化为先把所有的面积都求出来，每次比较 IoU 就只需要计算交叠部分面积，IoU = 交叠 / (面积1 + 面积2 - 交叠)。
>
> Soft-NMS:当两个 GT 重叠度很高时，NMS 会将具有较低置信度的框去掉。Soft-NMS 将置信度改为 IoU 的函数，从而较低的值不至于删去，但得到的得分不能太高。
>
> Q: 什么是 anchor

#### **SPP-Net**

- 提出通过 Spatial Pyramid Pooling 替换 RCNN 中的裁剪 + 拉伸。

  ![1585924595496](../../questions.assets/1585924595496.png) 

#### Fast R-CNN

- 将 SPP 换成了 RoI Pooling，SPP 的特征(网格)是多尺寸 cat 的，而 RoI Pooling 是单一尺度的 max pooling，因为实验发现多尺度提升并不大，单尺度节省时间。

- 用网络代替了 SVM，在最后的卷积层上设置 RoI Pooling 后接两个分支回归位置和分类。

- 用 Smooth-l1 而不是 l2 代价：
  $$
  \operatorname{smooth}_{L_{1}}(x)=\left\{\begin{array}{ll}{0.5 x^{2}} & {\text { if }|x|<1} \\ {|x|-0.5} & {\text { otherwise }}\end{array}\right.
  $$

> Q23: l1 和 l2 loss 的区别

#### Faster R-CNN

- 提出 RPN 取代 Selective Search，
- Faster R-CNN 的正负样本选取：
- 

#### HyperNet



#### FPN



#### SSD, YOLO, DSSD



#### EfficientDet

### Common Topics

#### NMS, Soft-NMS



