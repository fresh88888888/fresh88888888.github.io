---
title: 随机森林（PyTorch）
date: 2024-02-27 09:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 随机森林

- 随机森林原理：随机森林，是一种基于决策树的集成学习算法。它通过构建多棵决策树，并将它们的预测结果进行投票或平均，从而提高预测的准确性和稳定性。每棵树都是在随机抽取的数据样本和特征上构建的，因此可以有效减少过拟合，提高模型的泛化能力。
- 应用场景：随机森林在许多领域都有广泛应用，如金融领域的信用评分、医疗领域的疾病预测、电商领域的推荐系统等它能够处理高维数据和缺失值，对异常值具有较强的鲁棒性，因此在实际问题中表现出色。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义随机森林模型
class RandomForest(nn.Module):
    def __init__(self):
        super(RandomForest, self).__init__()
        
        self.fc1 = nn.Linear(2, 2)  # 两个特征体重、颜色
        
    def forward(self, x):
        x = self.fc1(x)
        
        return x

# 训练数据
data = torch.tensor([[5.0,0], [4.0,1.0],[3.0,0], [6.0,1.0]], dtype=torch.float32)
labels = torch.tensor([0,1,0,1], dtype=torch.long)

# 模型、损失函数和优化器
model = RandomForest()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#训练过程
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 ==0:
        print(f'Epoch: {epoch + 1}/100, Loss: {loss.item():.4f}')
    
# 测试模型 
with torch.no_grad():
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted==labels).sum().item()
    
    print(f'Accuracy: {correct / len(labels * 100)}%')
```