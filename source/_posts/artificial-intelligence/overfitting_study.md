---
title: 过度拟合（PyTorch）
date: 2024-02-26 09:14:32
tags:
  - AI
categories:
  - 人工智能
---

#### 过度拟合

过度拟合，简单来说，就是模型在训练数据上表现得很好，但是在新的、未见过的数据上表现不佳。这就好比我们在学校里学习，如果一个学生只是死记硬背，考试时能的高分，但是一旦遇到实际问题，就束手无策。这种情况就是过度拟合。那么，如何避免过度拟合？这就需要我们在训练模型时，不仅要关注模型在训练数据上的表现，还要关注模型在验证集上的表现。如果模型在验证集上的表现不佳，那就说明模型可能过度拟合了。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 生成模拟数据
x_train =np.random.rand(100, 1)
y_train = 2 * x_train + 3 + np.random.randn(100, 1) * 0.3
x_test = np.random.rand(20, 1)
y_test = 2 * x_test + 3 + np.random.randn(20, 1) * 0.3

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        
        return x

model = LinearRegression(1, 1)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    # 前向传播
    outputs = model(torch.from_numpy(x_train).float())
    loss = criterion(outputs, torch.from_numpy(y_train).float())
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/ 1000], Loss: {loss.item()}')
        
        
# 绘制结果
plt.scatter(x_train.flatten(), y_train.flatten(), c='orange')
plt.plot(x_train.flatten(), model(
    torch.from_numpy(x_train).float()).detach().numpy(), 'g-', lw=1)
plt.show()
```
结果输出为：
```bash
$ poetry run python overfiting._demo.py 

Epoch [1/ 1000], Loss: 16.579368591308594
Epoch [101/ 1000], Loss: 0.20334331691265106
Epoch [201/ 1000], Loss: 0.0935521200299263
Epoch [301/ 1000], Loss: 0.08888057619333267
Epoch [401/ 1000], Loss: 0.08583617955446243
Epoch [501/ 1000], Loss: 0.08352799713611603
Epoch [601/ 1000], Loss: 0.08177556097507477
Epoch [701/ 1000], Loss: 0.08044500648975372
Epoch [801/ 1000], Loss: 0.079434834420681
Epoch [901/ 1000], Loss: 0.07866783440113068
```
{% asset_img overfitting.png %}