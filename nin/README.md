# Network in network

[toc]

## 1 Paper

> * 李沐：https://zh.d2l.ai/chapter_convolutional-modern/nin.html
> https://developer.aliyun.com/article/1057504
> https://juejin.cn/post/7064475270417219598


## 2 Project

<!-- ![AlexNet](./assets/model.jpg) -->

### 2.1. Overview

1. 直观理解可能不容易，可以这样想。
   1. 一个 3x3 的卷积，输入的一个像素是 3x3 大小的感受野。一个 1x1 的卷积，输入的就是一个像素的感受野，所以它不改变图片分辨率。
   2. 一个 1x1 的卷积，如果输出维度是 1，那么就是 m 层特征图的每一个通道进行加权求和，得到新的一个特征。输出 n 个特征图的话，参数量就是 $m\cdot n\cdot 1\cdot 1$
