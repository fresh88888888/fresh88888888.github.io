---
title: 残缺块（PyTorch）
date: 2024-02-20 19:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 残缺块

残缺块是深度学习中一个重要的概念，尤其在卷积神经网络中。它的主要目的是帮助网络更好地学习输入和输出之间的差异，从而提高模型的性能。残差块通过引入跳跃连接，是网络能直接学习输入和输出之间的差异，从而避免了梯度消失或梯度爆炸的问题。这种设计有助于网络更好地学习输入和输出之间的差异，从而提高模型的性能和泛化能力。
<!-- more -->

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # 卷积层1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride) 
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 卷积层2
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False) 
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 跳跃连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.shortcut(out)         # 跳跃连接
        out = self.relu(out)             # 非线性激活函数
        
        return out

# 残差块模型实例化
model = ResidualBlock(24, 24)
```