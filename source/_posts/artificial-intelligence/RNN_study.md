---
title: 双向/循环神经网络（PyTorch）
date: 2024-02-20 15:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 循环神经网络（RNN）

循环神经网络，不同于传统的神经网络，`RNN`在在处理序列数据时展现出惊人的记忆能力。它像一个经验丰富的讲述者，能够捕捉到数据中的长期依赖关系。这种能力使得它在自然语言处理和时间序列预测等领域大放异彩。双向循环神经网络（`Bi-RNN`）和长度记忆网络（`LSTM`）作为`RNN`的两种主要变体，更是将RNN的强大功能推向了新的高度。它们像是在时间序列中自由穿梭的舞者，既能回顾过去，又能展望未来。参数共享和图例完备性是`RNN`的两大特点，它们使得`RNN`在处理复杂问题时具有强大的表示能力。参数共享让`RNN`在处理不同任务时能够快速适应，而图灵完备性则意味着`RNN`几乎可以模拟任何计算过程。结合卷积神经网络结构构筑的循环神经网络不仅可以处理时间序列数据，还可以应对包含序列输入的计算机视觉问题。在深度学习的舞台上，循环神经网络无疑是最耀眼的明星之一。它不仅改变了我们对神经网络的认识，也引领这人工智能向前发展。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets


# 定义超参数
input_size = 10        # 输入图像的维度
hidden_size = 20       # 隐藏层神经元数量
num_layers = 1         # RNN层数
num_classes = 5        # 输出类别的数量（0~9）
num_epochs = 100       # 训练轮数
batch_sise = 10        # 批处理大小
learning_rate = 0.01   # 学习率

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 使用RNN层
        self.rnn = nn.RNN(input_size,  hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forwar(self, x):
        # 设置初始隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        
        # 通过RNN层，得到输出和最后隐藏状态
        out, _ = self.run(x, (h0,c0))
        # 去最后一个时间步的输出，通过全连接层得到最终输出
        out = self.fc(out[:, -1, :])
        return out


# 实例化模型、损失函数和优化器
model = RNN(input_size, hidden_size, num_layers, num_classes)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True)
test_dataset = datasets.MNIST(root='./data', train=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_sise, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_sise, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # 前向传播，得到预测输出
        outputs = model(inputs)
        # 计算损失值
        loss = criterion(outputs, labels)
        # 清空梯度缓存（因为PyTorch会累积梯度）
        optimizer.zero_grad()
        # 反向传播，计算梯度值
        loss.backward()
        # 根据梯度执行权重（执行优化步骤）
        optimizer.step()
        if (i + 1) % 100 == 0:
            # 每100个Batch打印一次损失值和准确率
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                  1, num_epochs, i + 1, len(train_loader), loss.item()))

print('Training finshied.')
```

#### 双向循环神经网络

双向循环神经网络（`Bi-RNN`）是标准RNN的升级版。它不仅让信息沿时间轴正向流动，还允许其反向流动。这意味着，对任何序列数据，每个时刻的隐藏状态不仅依赖之前的输入，还与之后的输入紧密相连。这种结构确保了我们能更全面地捕获序列的上下文信息。在`Bi-RNN`中，存在两个独立的隐藏层。一个负责正向传播，以负责反向传播，从序列的最后一个元素至首个。最后，这两个隐藏层在每个时间步的输出被合并形成最终的上下文表示，为后续的计算或预测提供有力支撑。简而言之，`Bi-RNN`是一个全能型的时间序列捕手，他让我们更能全面的了解数据背后的故事。如果你对时间序列的分析、自然语言处理或任何与序列数据相关的任务感兴趣，那么`Bi-RNN`绝对值得你深入了解。

```python
import torch
import torch.nn as nn
import torch.optim as optim


# 定义超参数
input_size = 10        # 输入图像的维度
output_size = 10
sequence_length = 10
hidden_size = 20       # 隐藏层神经元数量
num_layers = 1         # RNN层数
num_classes = 5        # 输出类别的数量（0~9）
num_epochs = 100       # 训练轮数
batch_sise = 10        # 批处理大小
learning_rate = 0.01   # 学习率

# 定义一个双向循环层，这里使用LSTM单元作为基础
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(BiRNN, self).__init__()

        # 正向和反向的LSTM层
        self.rnn = nn.LSTM(input_size,  hidden_size,
                          num_layers, bidirectional=True, dropout=dropout)
        # 全连接层, 假设我们做分类任务，类别数量为output_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forwar(self, x, hidden=None):
        batch_sise = x.size(0)
        seq_length = x.size(1)
        total_hidden_size = 2 * self.rnn.hidden_size  # 双向所以是两个隐藏层大小

        # LSTM前向传播
        outputs, (hidden, cell) = self.run(x, hidden)
        # 合并正向、反向的隐藏状态，得到每个时间步的完整上下文表示
        outputs = outputs.contiguous().view(-1, total_hidden_size)
        # 通过全连接进行分类
        predictions = self.fc(outputs)
        # 将预测的数据恢复为原始的时间序列形状
        predictions = predictions.view(batch_sise, seq_length, -1)
        return predictions, hidden

# 模型实例化
model = BiRNN(input_size, hidden_size, output_size)

# 假设x是准备好的输入数据
inputs = torch.randn((batch_sise, sequence_length, input_size))
outputs, _ = model(inputs)
```