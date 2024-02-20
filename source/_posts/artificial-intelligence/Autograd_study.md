---
title: 自动求导（PyTorch）
date: 2024-02-20 11:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 自动求导

在神经网络的训练过程中，我们经常需要更新模型参数，而这个过程往往依赖于损失函数关于模型参数的梯度。自动求导技术能够自动计算这些梯度，极大地简化了开发过程。在PyTorch中，通过torch.Tensor类实现了自动求导。当我们创建一个Tensor时，通过设置requires_grad=True标记该Tensor需要进行梯度计算，即可开启自动求导功能。
<!-- more -->
```python
import torch

# 创建一个需要计算梯度的Tensor
x = torch.ones(2,2,requires_grad=True)
print('x: ', x)

# 对Tensor进行操作
y = x + 2
z = y * y * 3
out = z.mean()

print('Operations on x: y = x + 2, z = y * y * 3, out = z.mean()')
print('y: ', y)
print('z: ', z)
print('out (loss): ', out)

# 计算梯度， 由于out是标量， 所以不需要指定grad_variables
out.backward()

# 查看x的梯度，即out关于x的梯度
print("Gradient of out with respect to x: ")
print(x.grad)

# 使用梯度更新参数（实际中会用优化器如SGD更新）
learning_rate = 0.1
x.data -= learning_rate * x.grad

# 在更新参数后， 需要清零梯度，因为下一轮迭代的梯度是基于当前参数的新梯度
x.grad.zero_()
```
输出结果为：
```bash
x:  tensor([[1., 1.],
        [1., 1.]], requires_grad=True)
Operations on x: y = x + 2, z = y * y * 3, out = z.mean()
y:  tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
z:  tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>)
out (loss):  tensor(27., grad_fn=<MeanBackward0>)
Gradient of out with respect to x: 
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```