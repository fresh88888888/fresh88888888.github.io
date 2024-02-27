---
title: 反向传播（PyTorch）
date: 2024-02-26 20:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 反向传播

想象一下，你正在玩一个猜数字的游戏，你需要猜一个数字，这个数字是正确答案。每次猜完后，都会有人告诉你猜的数字是偏大还是偏小了。根据这个反馈，你可以调整你的猜测，直到猜正确为止。这就是反向传播算法的基本思想。反向传播是一种用于训练神经网络的算法。它通过计算损失函数关于网络参数的梯度，从而对网络参数进行更新，以达到减小损失函数值的目的。这个过程中，算法会从输出层开始，逐层计算每一层的梯度，知道输入层。这个例子中，我们定义了一个简单线性模型，并使用随机梯度下降法进行训练。在每次迭代中，我们首先进行前向传播，计算输出和损失；然后进行反向传播，计算梯度；最后更新模型参数。通过这个例子，我们可以看到反向传播算法在神经网络训练中的重要作用。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,1)

    def forward(self, x):
        x = self.fc1(x)

        return x

# 创建一个模型实例
model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# 训练数据
x_train = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
y_train = torch.tensor([[2.0], [4.0], [6.0]])

# 训练过程
for epoch in range(10):
    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播和优化
    optimizer.zero_grad() # 清空过往梯度
    loss.backward()       # 反向传播，计算当前梯度
    optimizer.step()      # 更新参数
    
    print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

```
输出结果为：
```bash
$ poetry run python backpropagation_demo.py 

Epoch: 1, Loss: 10.4069
Epoch: 2, Loss: 210.8012
Epoch: 3, Loss: 4354.0044
Epoch: 4, Loss: 89996.3984
Epoch: 5, Loss: 1860259.0000
Epoch: 6, Loss: 38452288.0000
Epoch: 7, Loss: 794823872.0000
Epoch: 8, Loss: 16429318144.0000
Epoch: 9, Loss: 339600474112.0000
Epoch: 10, Loss: 7019675254784.0000
```