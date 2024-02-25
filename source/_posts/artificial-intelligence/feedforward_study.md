---
title: 前馈神经网络（PyTorch）
date: 2024-02-25 23:14:32
tags:
  - AI
categories:
  - 人工智能
---

#### 前馈神经网络

简单来说，它是一种模仿人脑神经元结构的计算模型，可以用于解决各种复杂的问题，如图像识别、自然语言处理等。前馈神经网络的工作原理是搭建多级决策流程：输入层接收到数据后，通过隐藏层进行特征提取和转换，最终输出层生成预测结果。这个过程每一层都充满了无限潜能与可能。

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 顶一个简单的神经网络
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        self.layer1 = nn.Linear(in_features=10, out_features=5)
        self.relu = nn.ReLU()
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