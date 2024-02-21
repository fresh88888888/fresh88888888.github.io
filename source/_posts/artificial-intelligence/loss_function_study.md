---
title: 损失函数（PyTorch）
date: 2024-02-20 20:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 损失函数

损失函数（`Loss Function`）：通俗来讲，就像是一个衡量工具或者说‘打分老师’，在机器学习和深度学习中用来评价模型的预测结果有多接近或者说符合实际的真实答案。想象以下，你正在教一个小孩做数学题，每做完一道题，你会检查他的答案与正确答案之间的差距，并且基于这个差距给出反馈。如果完全答对了，那就给满分；如果答错了，错的越多得分就越低。

在模型训练的过程中，损失函数就是一个”打分机制“。比如模型预测房价、识别图像或翻译句子时，他会生成一个预测值，然后损失函数会对这个预测值与已知真实值，计算出一个误差值（也就是损失）。模型的目标就是通过不断调整内部参数，使得每次预测后的损失尽可能小，即预测结果越来越贴近真实结果。损失函数的主要作用：
- 评估模型性能：量化模型预测的好坏，损失越小表示预测越准确。
- 指导优化过程：在训练的过程中，通过梯度下降等优化算法，依据损失函数计算出模型参数应如何更新，从而改进模型预测能力。
- 模型选择与调参：不同的任务会选择不同的损失函数，合适的损失函数有助于提升模型在特定问题上的表现。
<!-- more -->

```python
import torch
import torch.nn as nn


# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)
    
# 初始化模型，优化器以及损失函数
model = SimpleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 假设我们有一些输入数据和对应的标签
inputs = torch.tensor([[1.0],[2.0], [3.0]], dtype=torch.float32)
targets = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32)

# 将数据送入模型并计算预测值
predictions = model(inputs)

# 计算损失
loss = loss_fn(predictions, targets)  #MSE损失是预测值和目标值之间的平方差的平均

# 打印当前损失
print('Current loss: {loss.item():.4f}', loss.item())

# 反向传播即参数更新
optimizer.zero_grad()  # 清零梯度缓冲区
loss.backward()        # 计算梯度
optimizer.step()       # 根据梯度更新模型参数
```
