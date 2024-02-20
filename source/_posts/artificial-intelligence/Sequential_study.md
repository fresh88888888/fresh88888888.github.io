---
title: Sequential容器（PyTorch）
date: 2024-02-20 13:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### Sequential容器

`Sequential`容器也被称为顺序容器，在标准库中，有三种常见的`Sequential`容器：`vector、list`和`deque`。虽然它们都存储元素，但访问方式和添加/删除元素的成本却大不相同。`vector`就像一个不断增长的数组，而`list`则像一个个串联的珠子。至于`deque`，它则兼具了`vector`和`list`的特点，既可以快速访问中间元素，又可以在两端高效地添加和删除元素。标准库还为我们提供了`Sequential`容器的适配器：`stack、queue`和`priority_queue`则允许我们快速访问或删除最大或最小的元素。
<!-- more -->

```python
import torch
import torch.nn as nn

# 定义Sequential
model = nn.Sequential(
    # 输入层和隐藏层的线性层
    nn.Linear(10, 20),
    # 激活函数
    nn.ReLU(),
    
    # 隐藏层和输出层的线性层
    nn.Linear(20, 10),
    nn.ReLU(),
)

# 输入数据
input_data = torch.randn(1, 10)

# 前向传播
output_data = model(input_data)

# 打印输出数据
print(output_data)
```
输出结果为：
```bash
tensor([[0.0000, 0.0000, 0.2003, 0.0982, 0.0294, 0.0486, 0.0000, 0.0000, 0.0000,
         0.0516]], grad_fn=<ReluBackward0>)
```