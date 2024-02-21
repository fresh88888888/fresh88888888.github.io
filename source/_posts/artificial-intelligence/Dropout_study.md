---
title: Dropout正则化（PyTorch）
date: 2024-02-21 09:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### Dropout正则化

`Dropout`正则化是一种在训练深度神经网络时防止过拟合的技术。它的工作原理就像是每次训练时随机让一部分神经元“休息”，即暂时不参与计算，具体来说，每个神经元按照预设的概率`p`被临时从网络丢弃（其输出值被置`0`）。这样做的目的是避免模型对某些特征或特征组合过度依赖，从而提高模型的泛化性能。想想一个场景，如果一个团队过分依赖某几个核心成员，在这些关键人物不在场时，整个团队的表现可能大幅下滑。同样地，在神经网络结构中，通过`Dropout`技术，模型不会对一组局部特征过分敏感，这会促使网络学习更加稳定，多样化的特征表示。在实际代码实现上，`nn.Dropout(p)`是一个模块，当模型在训练模式下运行时，该模块会议给定的概率p随机丢弃输入信号的某些元素。值得注意的是，再适用`PyTorch`时。无需手动开关`Dropout`功能，因为框架会根据`.train()`和`.eval()`方法智能地控制`Dropout`在训练阶段和评估阶段的行为。
<!-- more -->

```python
import torch
import torch.nn as nn

# 假设我们构建一个多层感知机，包含两个隐藏层和一个输出层
class SimpleMLP(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 256, output_size = 10, dropout_prob = 0.5):
        super(SimpleMLP, self).__init__()
        
        # 隐藏层的定义
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        # 第一个隐藏层后的Dropout层
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 输入数据经过第一个隐藏层并激活
        x = torch.relu(self.hidden_layer1(x))
        # 应用Dropout，按概率dropout_prob丢弃一些神经元
        x = self.dropout1(x)
        
        # 继续通过第二隐藏层并激活
        x = torch.relu(self.hidden_layer2(x))
        # 应用Dropout，按概率dropout_prob丢弃一些神经元
        x = self.dropout2(x)
        
        # 最后通过输出层得到预测结果
        x = self.output_layer(x)

        return x

# 创建模型实例
model = SimpleMLP()

# 假设我们有一些输入数据
inputs = torch.randn(100, 784) # 100个样本，每个样本784个特征

# 前向传播过程，dropout会砸训练模式下生效
outputs = model(inputs)

# 在训练过程中，Dropout层会根据设定的概率进行丢弃
# 而在验证和测试阶段，通常会关闭Dropout以保持模型行为的一致性
model.train() # 设置模型为训练模式
with torch.set_grad_enabled(True):
    # 确保梯度计算开启
    outputs_with_dropout = model(inputs)  # 这里的Dropout会起作用
    print(outputs_with_dropout)

# 当需要验证和测试时，一般会禁用Dropout
model.eval() # 设置模型为评估模式
with torch.no_grad():  # 不进行梯度计算
    outputs_with_dropout = model(inputs)  # 在评估时Dropout不会生效
    print(outputs_with_dropout)

```