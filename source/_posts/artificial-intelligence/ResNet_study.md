---
title: ResNet神经网络（PyTorch）
date: 2024-02-26 15:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### ResNet神经网络

`ResNet` 全名`Residual Network`，诞生于`2015`年，由微软研究院的研究者提出，为解决随着神经网络深度加深而出现的梯度消失和网络退化问题提供了颠覆性的解决方案。其核心思想是引入“残差块”构造深层网络，让信息直接由浅层传递到深层，绕过复杂的非线性变换，从而使得模型能够训练更深且更有效的层级结构。想象一下，你正在攀登一座陡峭的山峰，但是山路崎岖难行，每走一步都异常艰辛。这时，如果山路上每隔一段距离就有一个休息平台，你是不是会觉得轻松很多？`ResNet`就是这样的“超级楼梯”，它通过在神经网络中添加“休息平台”（残差块），让网络在学习的过程中能够轻松“喘气”，从而能够构建更深、更强大的模型。残差块就是一个小小的“助推器”，帮助神经网络的在训练的过程中更好地学习数据的特征。这种设计使得`ResNet`能够轻松应对深度神经网络中的“梯度消失”和“梯度爆炸”问题，让我们能够构建出更深层次的模型，实现更精准，更高效的预测。实际工程应用场景：
<!-- more -->
- 图像分类：`ResNet`应用于大估摸图像数据集（如`ImageNet`）上的物体识别和分类任务。
- 目标检测：在`Faster R-CNN、YOLOv3`等目标检测框架中，`ResNet`作为特征提取器用于定位和识别图像中的不同对象。
- 语义分割：结合全卷积网络（`FCN`），`ResNet`被用来从图像中精确划分出每个像素所属的类别。
- 实例分割：在`Mask R-CNN`等模型中，`ResNet`助力区分图像中单个物体的边界进行实例级别的标注。
- 迁移学习：预训练的`ResNet`模型广泛用于医疗影响分析、遥感图像识别等领域，通过微调适应特定任务。
- 行为识别和人体姿态估计：`ResNet`在视频分析和理解人类行为以及估计人体关节位置方面提供有效的特征表示。

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride =4, padding = 1, bias=False):
        super(ResidualBlock, self).__init__()
        
        # 残差块的基本组件
        self.conv1 = nn.Conv2d(in_channels, out_channels,  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 当输入和输出通道不同时，添加一个1 x 1卷积层用于调整通道数
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = x # 保存原始输入作为残差部分
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加快捷连接（shoutcut connection），即输入与经过残差映射后的输出相加
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    
# 定义一个简单的ResNet
class SimpleResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SimpleResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], strife=1)
        # 进一步添加更多的层和残差块
        self.avgpool  = nn.AvgPool2d(kernel_size=4)
        self.fc= nn.Linear(512, num_classes)  # 假设最后一层为512
    
    def make_layer(self, block, out_channels, num_blacks, stride):
        strides = [stride] + [1] * (num_blacks -1)
        # 第一块使用指定步长
        layers =[]
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            # 更新输入通道数
            return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        # 继续前向传播过程
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    # 创建一个实际的ResNet-18模型
    model = SimpleResNet(ResidualBlock, [2, 2,2,2]) # 这里仅表示简化版本的层配置
    
```