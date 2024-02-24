---
title: 反向传播（PyTorch）
date: 2024-02-24 20:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 反向传播

当我们构建一个机器学习模型，例如神经网络，我们需要告诉模型如何根据输入的数据得到正确的输出。这个过程通常涉及到向前传播（或者说前向传播）：输入数据进入模型的输入层，然后经过一系列的数学运算和变换，最终得到输出结果。然而，仅仅向前传播是不够的。因为我们不仅想知道输出是什么，更想知道如果输入稍微改变一下，输出会如何变化。就需要反向传播发挥作用了。简单来说，反向传播就是一种计算误差的方法，它通过比较模型的输出结果和真实结果之间的差异（即误差），来决定如何调整模型的参数（例如权重和偏置项）以便在下一次前向传播时得到更接近真实结果的输出。为什么每个算法模型都有一个反向传播函数呢？因为只有通过反向传播，我们才知道模型在哪方面做得好，哪些方面需要改进。换句话说，反向传播是模型训练过程中不可或缺的一环。它不仅帮助我们评估模型的性能，还指导我们调整模型的参数使模型在未知的数据上表现得更好。想象以下，如果我们有一个模型但是不知道如何调整它的参数，那么无论怎么使用这个模型，它的性能都很难得到提升。而有了反向传播，我们就像有了一双“指南针”，知道该如何引导模型向更好的方向发展。

所以，反向传播是机器学习和深度学习的一项核心技术。它不仅帮助我们构建更强大的模型，还使得我们可以理解和解释模型的决策过程。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 顶一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.layer1 = nn.Linear(in_features=10, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=2)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
         
        return x

# 创建一个模型实例
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成一些随机数据作为输入和目标
inputs = torch.randn(64, 10)
targets = torch.randint(0,2, (64,))

# 前向传播
ouputs = model(inputs)
loss = criterion(ouputs, targets)

# 反向传播
optimizer.zero_grad()  # 清楚之前的梯度
loss.backward()        # 反向传播计算梯度
optimizer.step()       # 更新模型参数

print('Loss after backward propagation: ', loss.item())
```