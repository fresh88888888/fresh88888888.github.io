---
title: 保存和加载模型（PyTorch）
date: 2024-02-21 11:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 保存和加载模型

当我们在训练深度学习模型时，经常会遇到训练周期较长的情况，或者我们需要对模型进行反复调整和优化。为了节省时间，提高效率，我们可以将训练好的模型保存下来，然后在需要时加载模型进行推理和继续训练。在`PyTorch`中，保存和加载模型的过程非常简单，我们可以使用`torch.save()`和`torch.load()`函数来实现。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forwar(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型实例化
model = SimpleNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 模拟训练过程，此处省略
# ...

# 保存模型参数
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型参数(假设在另一个设备上运行)
model = SimpleNet()   # 创新创建模型结构，因为Pytorch的模型是不可序列化的，因此需要重新创建模型结构实例

m_obj = torch.load('model_weights.pth')
print(m_obj)
model.load_state_dict(m_obj)  # 加载模型参数到模型实例中
```
`model_weights.pth`文件内容为：
```bash
OrderedDict([('fc1.weight', tensor([[ 0.1588,  0.1576, -0.0710,  0.2026,  0.2325, -0.0658,  0.0469, -0.1095,
          0.0057, -0.0112],
        [ 0.0926, -0.1308,  0.3011,  0.0994, -0.1364,  0.2542,  0.2959, -0.1177,
          0.0156,  0.2856],
        [-0.0558, -0.2349,  0.2259, -0.0110, -0.2148, -0.0198, -0.0864,  0.0350,
         -0.2227,  0.0015],
        [ 0.3079, -0.0889, -0.2307, -0.2112, -0.1713,  0.2154,  0.1956,  0.2344,
         -0.0615,  0.0903],
        [-0.2575, -0.2548,  0.2570, -0.0239,  0.0828,  0.2584,  0.1157,  0.0566,
          0.2089,  0.0586]])), ('fc1.bias', tensor([-0.0067,  0.1385, -0.3088,  0.1175,  0.2169])), ('fc2.weight', tensor([[ 0.1616, -0.4355,  0.2955,  0.4270, -0.1871],
        [-0.4327, -0.1369,  0.3971,  0.0074, -0.1222]])), ('fc2.bias', tensor([0.1910, 0.3539]))])
        
```