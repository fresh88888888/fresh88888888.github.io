---
title: 一个简单的图像识别示例（PyTorch）
date: 2024-02-20 10:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 揭秘”张量“

在深度学习和机器学习中，张量（`Tensor`）是一个充满魔法的多维数组。它能够包容各种数据类型，包括整数、浮点数、布尔值等，展现出强大的包容性和灵活性。在`Python`的深度学习库`PyTorch`中，我们能够轻松创建一维、二维和三维张量。值得注意的是，张量的维度也被称为秩（`rank`），一维张量就是我们口中常说的向量，二维张量即是矩阵高，而更高维度的张量对应于高阶数组。通过张量，我们能够开启深度学习的无限可能，探索机器智慧的奥秘。

```python
import torch

# 定义一个一维张量（向量）
tensor1 = torch.tensor([1,2,3,4])

# 定义一个二维张量（矩阵）
tensor2 = torch.tensor([[1,2],[3,4],[5,6]])

# 定义一个三维张量
tensor3 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

# 查看张量的形状
print(tensor1.shape)
print(tensor2.shape)
print(tensor3.shape)
```
输出结果为：
```bash
torch.Size([4])
torch.Size([3, 2])
torch.Size([2, 2, 3])
```
