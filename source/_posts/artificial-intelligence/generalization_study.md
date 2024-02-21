---
title: 模型的泛化能力（PyTorch）
date: 2024-02-21 16:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 模型的泛化能力

模型的泛化能力（`generalization capability`）是指一个机器学习和深度学习模型在未见过的新数据上的表现如何，即他能否有效地将从训练集学到的知识迁移到测试集或实际应用中去。如果一个模型对训练数据拟合的非常好，但在新的未知数据上表现糟糕，我们说这个模型过拟合了，它的泛化能力较弱；反之如果模型在保持训练数据良好拟合的同时，在新数据上也能保持较好的性能，则说明具有良好的泛化能力。下面的代码是通过P一个基于`PyTorch`框架的简单线性回归模型示例来直观展示模型泛化能力的概念。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设我们有一些模拟的数据点，训练集和验证集各一半
x_train = torch.randn(1000, 1)
y_train = 2 * x_train + 1 + torch.randn(1000, 1)  # 训练数据目标值，模拟线性关系并增加噪声

# 划分训练集和验证集
x_val = x_train[:500]
y_val = y_train[:500]
x_train = x_train[:500]
y_train = y_train[:500]

# 创建TensorDataset并将数据转化为DataLoader以便批量处理
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)

# 定义一个简单的线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 模型实例化
model = LinearModel()

# 使用均方误差损失函数和SGD优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 循环训练，这里简化未单个epoch
for epoch in range(10):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # 正向传递计算预测值
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            preds = model(inputs)
            val_loss = criterion(preds, targets).item() * len(inputs)
            
        val_loss /= len(val_data)
        
    print(f'Epoch {epoch + 1}, validation Loss: {val_loss:.4f}')
```
输出结果为：

```bash
Epoch 1, validation Loss: 0.2673
Epoch 2, validation Loss: 0.1716
Epoch 3, validation Loss: 0.1200
Epoch 4, validation Loss: 0.0916
Epoch 5, validation Loss: 0.0755
Epoch 6, validation Loss: 0.0661
Epoch 7, validation Loss: 0.0605
Epoch 8, validation Loss: 0.0570
Epoch 9, validation Loss: 0.0548
Epoch 10, validation Loss: 0.0534
```