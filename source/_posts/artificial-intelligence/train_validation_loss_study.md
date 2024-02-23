---
title: 训练/验证损失（PyTorch）
date: 2024-02-23 09:30:32
tags:
  - AI
categories:
  - 人工智能
---

#### 训练/验证损失

关于训练损失（`train Loss`）和验证损失（`validation Loss`），想象你在教一只狗做算术。你有一堆卡片，每张卡片都有一个数学问题，比如“`2 + 3 =？`”和答案。小狗的任务看着问题，然后找出正确答案。
- 训练损失：就像你教小狗时，它回答错误的那些问题的数量。你希望这个数字越小越好，因为这意味着小狗在训练时学的越好。
- 验证损失：就像你在训练一段时间后，拿出一些新的卡片来测试小狗。这些卡片上的问题小狗之前没见过。验证损失就是小狗在这些新问题上回答错误的数量。这个数字也很重要。证明小狗学到的知识是否能够应用到新的问上，也就是它泛化能力如何。

如果你发现小狗在训练时的错误很少（训练损失低）但是在新的问题上的错误很多（验证损失高），那就意味着小狗已经记住了你给它看过的哪些特定问题和答案，而没有真正学会怎么做算术。这就是模型过拟合了，支队训练数据学的很好，但对新的数据就不行了。你的目标是要让小狗在训练和验证时都尽量少犯错误，这样它才能学会做算术，也能应用到新的问题上。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optimxw
from torch.utils.data import DataLoader, TensorDataset

# 假设我们有一些随机数据来模拟训练过程，这里我们创建一些随机数据作为示例
features = torch.randn(100, 10) # 100个样本，每个样本10个特征
targets = torch.randint(0, 2, (100,))  # 100哥目标值，假设是二分类问题

# 将数据转换为PyTorch数据集
dataset = TensorDataset(features, targets)

# 创建数据加载器
train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(10, 2)  # 输入维度是10， 输出维度是2
        
    def forward(self, x):
        return self.layer(x)
    
# 实例化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#训练模型
num_epochs = 3  # 训练3个epoch
for epoch in range(num_epochs):
    # 训练阶段
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()       # 清除之前的梯度
        outputs = model(inputs)     # 前向传播
        loss = criterion(outputs, targets)   # 计算损失
        loss.backward()                      # 反向传播
        optimizer.step()                     # 更新权重
        train_loss += loss.item() * inputs.size(0) # 累加损失
    
    # 计算平均训练损失
    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')
    
    # 验证阶段（这里我们使用训练数据作为验证数据，实际中应该使用不同的数据集）
    model.eval()  # 设置模型为评估模式
    valid_loss = 0.0
    with torch.no_grad():  # 在验证阶段不计算梯度
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item() * inputs.size(0)
            
    # 计算平均验证损失
    valid_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch + 1} Validation Loss: {valid_loss:.4f}')
```