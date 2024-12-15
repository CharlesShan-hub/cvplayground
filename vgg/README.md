# AlexNet

[toc]

## 1 Paper

> * 李沐：https://zh.d2l.ai/chapter_convolutional-modern/vgg.html


## 2 Project

![VGG16](./assets/model.png)

### 2.1. Overview

1. VGG 就是“内卷”版的 AlexNet。提出 “卷积-卷积-池化”这种结构，通过变化卷积的次数以及整个块的次数，改变网络结构。
2. 本项目目的是学习 VGG 网络并且提供一个可以调用的模型。所以，仅提供 `test.sh` 并使用 torchvision 的模型和参数进行推理。数据集选择更小的 TinyImageNet。


wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d ./tiny-imagenet-200