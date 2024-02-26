---
title: 基于梯度的学习（PyTorch）
date: 2024-02-25 22:14:32
tags:
  - AI
categories:
  - 人工智能
---

#### 基于梯度的学习

简单来说，它是一种通过计算损失函数的梯度来优化模型参数的方法。这种方法可以帮助我们找到损失函数的最小值，从而使模型在训练数据上达到更好的性能。梯度下降是一种优化算法，用于寻找函数的最小值。机器学习中，我们通常使用梯度下降来训练模型，使其更好地适应数据。通过不断地迭代和调整模型的参数，使得损失函数逐渐减少，最终找到最优解。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 顶一个简单的神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        self.layer1 = nn.Linear(in_features=10, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=2)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
         
        return x

# 创建一个模型实例
model = SimpleModel()

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