---
title: 循环神经网络 (RNN)(TensorFlow)
date: 2024-05-20 11:12:11
tags:
  - AI
categories:
  - 人工智能
mathjax:
  tex:
    tags: 'ams'
  svg:
    exFactor: 0.03
---

#### 门控循环单元

我们讨论了如何在循环神经网络中计算梯度，以及矩阵连续乘积可以导致梯度消失或梯度爆炸的问题。下面我们简单思考一下这种梯度异常在实践中的意义：
- 我们可能会遇到这样的情况：早期观测值对预测所有未来观测值具有非常重要的意义。考虑一个极端情况，其中第一个观测值包含一个校验和，目标是在序列的末尾辨别校验和是否正确。在这种情况下，第一个词元的影响至关重要。我们希望有某些机制能够在一个记忆元里存储重要的早期信息。如果没有这样的机制，我们将不得不给这个观测值指定一个非常大的梯度，因为它会影响所有后续的观测值。
- 我们可能会遇到这样的情况：一些词元没有相关的观测值。
- 我们可能会遇到这样的情况：序列的各个部分之间存在逻辑中断。

在学术界已经提出了许多方法来解决这类问题。其中最早的方法是“**长短期记忆**”(`long-short-term memory，LSTM`)，门控循环单元(`gated recurrent unit，GRU`)是一个稍微简化的变体，通常能够提供同等的效果，并且计算的速度明显更快。
<!-- more -->
##### 门控隐状态

**门控循环单元**与普通的循环神经网络之间的关键区别在于：前者支持隐状态的门控。 这意味着模型有专门的机制来确定应该何时更新隐状态，以及应该何时重置隐状态。这些机制是可学习的，并且能够解决了上面列出的问题。例如，如果第一个词元非常重要，模型将学会在第一次观测之后不更新隐状态。同样，模型也可以学会跳过不相关的临时观测。最后，模型还将学会在需要的时候重置隐状态。
###### 重置门和更新门

**重置门**(`reset gate`)和**更新门**(`update gate`)。我们把它们设计成{% mathjax %}(0,1){% endmathjax %}区间中的向量，这样我们就可以进行凸组合。重置门允许我们控制“可能还想记住”的过去状态的数量；更新门将允许我们控制新状态中有多少个是旧状态的副本。我们从构造这些门控开始。下图描述了门控循环单元中的重置门和更新门的输入，输入是由当前时间步的输入和前一时间步的隐状态给出。两个门的输出是由使用`sigmoid`激活函数的两个全连接层给出。
{% asset_img rnn_1.png "在门控循环单元模型中计算重置门和更新门" %}

我们来看一下门控循环单元的数学表达。对于给定的时间步{% mathjax %}t{% endmathjax %}，假设输入是一个小批量{% mathjax %}\mathbf{X}_t\in \mathbb{R}^{n\times d}{% endmathjax %}(样本个数{% mathjax %}n{% endmathjax %}，输入个数{% mathjax %}d{% endmathjax %})，上一个时间步的隐状态是{% mathjax %}\mathbf{H}_{t-1}\in \mathbb{R}^{n\times h}{% endmathjax %}（隐藏单元个数{% mathjax %}h{% endmathjax %}）。那么，重置门{% mathjax %}\mathbf{R}_t\in \mathbb{R}^{n\times h}{% endmathjax %}的计算如下所示：
\begin{align}
\mathbf{R}_t & = \sigma(\mathbf{X}_t\mathbf{W}_{xr} + \mathbf{H}_{t-1}\mathbf{W}_{hr} + \mathbf{b}_r) \\ 
\mathbf{Z}_t & = \sigma(\mathbf{X}_t\mathbf{W}_{xz} + \mathbf{H}_{t-1}\mathbf{W}_{hz} + \mathbf{b}_z) \\
\end{align}
{% endmathjax %}
其中{% mathjax %}\mathbf{W}_{xr}，{% mathjax %}\mathbf{W}_{xz}\in \mathbb{R}^{d\times h}{% endmathjax %}和{% mathjax %}\mathbf{W}_{hr}{% endmathjax %}、{% mathjax %}\mathbf{W}_{hz}\in \mathbb{R}^{h\times h}{% endmathjax %}是权重参数，{% mathjax %}\mathbf{b}_r{% endmathjax %}、{% mathjax %}\mathbf{b}_z\in \mathbb{R}^{1\times h}{% endmathjax %}是偏置参数。请注意，在求和过程中会触发广播机制。我们使用`sigmoid`函数将输入值转换到区间{% mathjax %}(0,1){% endmathjax %}。
###### 候选隐状态

