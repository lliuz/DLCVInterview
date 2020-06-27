## DLCV Interview

tag:

CV, Computer Vision

DL: Deep Learning

CI: Code Interview



https://www.julyedu.com/question/index/type/1



### TODO

#### Ring all-reduce 流程介绍

#### 数据并行框架总结

#### 常见目标检测网络(FRCNN, RetinaNet, SSD, YOLOv1~v4, [anchor free](https://zhuanlan.zhihu.com/p/62103812))

- NMS 算法实现, 复杂度

- 目标检测中的 Anchor

  anchor 算是对物体 size 的先验 的引入，降低了回归问题的难度，但也降低了算法的鲁棒性。

#### 最新 3D 目标检测方法

- PointPainting

- PV-RCNN
- https://zhuanlan.zhihu.com/p/149752359

#### 3D 检测基础论文

- PointNet， PointNet ++ ，Frustum PointNet，PointR-CNN，STD ，VoxelNet，SECOND, [PointPillars](https://blog.csdn.net/u011507206/article/details/89381872)，HVNet, MonoDIS ，CenterNet ，TTFNet ，RTM3D

#### 2D、3D目标检测趋势

#### 光流和场景流的应用点

##### 对于 2d tracker 的改进: OF + IoU tracker

> c.f. ICCV2019 workshop VisDrone-MOT2019: A.2. Multiple Object Tracking with Motion and Appearance Cues (Flow-Tracker)).
>
> 用光流网络预测两帧之间的运动，根据 OF 估计当前帧的 track 的位置，再和 Detecter 的 bbox 计算 IoU，如果 IoU 大于阈值，则直接 match 上，否则对未匹配上的框提取特征(类似ReID)，在外观和位置角度进行匹配判断。 
>
> 一方面可以消除相机运动的影响提升精度，另一方面可以减少对目标的特征提取的算力。

##### 对 3d tracker 的改进: 

投到车体下的3d坐标，2d 坐标，极易受到2d bbox 的大小变化以及车辆姿态的变化的影响尤其是pitch角的影响。用光流在图像里估计scale的变化，约束车体下跟踪的结果，确保车体下跟踪的scale一致性和图像中的scale变化一致，有一个闭环，做到稳定的 tracking。

**多传感器(多个 cam + LiDAR) 2d光流和3d场景流结合提升检测跟踪: **

google 多帧点云 检测 带预测物体方向(类似场景流)

F-pointnet flownet3d 每个点的场景流估计 但更实用的是 类似于目标检测，

特征层面: 点云处理经过早期 pointnet, 再之后有 vovel, pillar, 类似的，从特征层面可以根据 pillar的特征去提场景流。

框架方面: 2d centernet 放到点云空间。

光流和场景流怎么去结合还没有人去探索

#### 常见 MOT 方法

#### 常见 3D tracking 方法总结

#### 卡尔曼滤波总结

#### 常见网络参数量和精度

#### 对自监督学习的看法

#### 拓扑排序 计算图

https://www.bookstack.cn/read/huaxiaozhuan-ai/spilt.1.3d6a8774ef33a12e.md