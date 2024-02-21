---
title: CUDA让训练加速（PyTorch）
date: 2024-02-21 14:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### CUDA让训练加速

`CUDA`是英伟达推出的并行计算平台和编程模型，它能让开发者利用`GPU`的强大计算能力执行通用计算任务，而不仅仅处理图形数据。就像`CPU`是计算机的“大脑”，执行各种指令；`GPU`则是“超级助手”，尤其擅长同时处理大量相似数据。这在科学计算、机器学习和深度学习中的矩阵运算等计算密集型场景中非常高效。没有`CUDA`，用`CPU`运行深度学习模型可能会非常耗时。但有了`CUDA`和`cuDNN`这样的库，我们可以把计算密集型任务分发到`GPU`上并行处理，大大加速训练速度。
<!-- more -->

```python
import torch

# 创建一个在CPU上的张量
cpu_tensor = torch.randn(100, 100)  # 随机生成一个100 x 100 de 浮点数矩阵

# 如果系统安装CUDA并且支持的话，我们可以将CPU上的张量移动到GPU上
if torch.cuda.is_available():
    # 检查是否有可用的GPU
    device = torch.device('cuda')  # 定义设备为CUDA设备
    # 将CPU张量转移到GPU上
    gpu_tensor = cpu_tensor.to(device)
    
    print(f'CPU 张量：{cpu_tensor}')
    print(f'GPU 张量：{gpu_tensor}')

# 在GPU上执行矩阵乘法操作
result = torch.mm(gpu_tensor, gpu_tensor) # 矩阵乘法

# 当需要查看结果或进一步在CPU上处理时，可以将GPU上的结果再撤回到CPU
result_cpu = result.to('cpu')

# 注意：实际使用中，我们会直接在GPU上创建张量，并在那里完成所有计算。
```