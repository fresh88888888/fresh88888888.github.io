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
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{R}_t & = \sigma(\mathbf{X}_t\mathbf{W}_{xr} + \mathbf{H}_{t-1}\mathbf{W}_{hr} + \mathbf{b}_r) \\ 
\mathbf{Z}_t & = \sigma(\mathbf{X}_t\mathbf{W}_{xz} + \mathbf{H}_{t-1}\mathbf{W}_{hz} + \mathbf{b}_z) \\
\end{align}
{% endmathjax %}
其中{% mathjax %}\mathbf{W}_{xr}{% endmathjax %}、{% mathjax %}\mathbf{W}_{xz}\in \mathbb{R}^{d\times h}{% endmathjax %}和{% mathjax %}\mathbf{W}_{hr}{% endmathjax %}、{% mathjax %}\mathbf{W}_{hz}\in \mathbb{R}^{h\times h}{% endmathjax %}是权重参数，{% mathjax %}\mathbf{b}_r{% endmathjax %}、{% mathjax %}\mathbf{b}_z\in \mathbb{R}^{1\times h}{% endmathjax %}是偏置参数。请注意，在求和过程中会触发广播机制。我们使用`sigmoid`函数将输入值转换到区间{% mathjax %}(0,1){% endmathjax %}。
###### 候选隐状态

接下来，让我们将重置门{% mathjax %}\mathbf{R}_t{% endmathjax %}与{% mathjax %}\mathbf{H}_t{% endmathjax %}中的常规隐状态更新机制集成，得到时间步{% mathjax %}t{% endmathjax %}的**候选隐状态**(`candidate hidden state`){% mathjax %}\tilde{\mathbf{H}}_t\in \mathbb{R}^{n\times h}{% endmathjax %}:
{% mathjax '{"conversion":{"em":14}}' %}
\tilde{\mathbf{H}}_t = \tanh(\mathbf{X}_t\mathbf{W}_{xh} + (\mathbf{R}_t\odot\mathbf{H}_{t-1})\mathbf{W}_{hh} + \mathbf{b}_h)
{% endmathjax %}
其中{% mathjax %}\mathbf{W}_{xh}\in\mathbb{R}^{d\times h}{% endmathjax %}和{% mathjax %}\mathbf{W}_{hh}\in\mathbb{R}^{h\times h}{% endmathjax %}是权重参数，{% mathjax %}\mathbf{b}_h\in\mathbb{R}^{1\times h}{% endmathjax %}是偏置项，符号{% mathjax %}\odot{% endmathjax %}是`Hadamard`积（按元素乘积）运算符。在这里，我们使用`tanh`非线性激活函数来确保候选隐状态中的值保持在区间{% mathjax %}(-1,1){% endmathjax %}中。上面公式中的{% mathjax %}\mathbf{R}_t{% endmathjax %}和{% mathjax %}\mathbf{H}_{t-1}{% endmathjax %}的元素相乘可以减少以往状态的影响。每当重置门{% mathjax %}\mathbf{R}_t{% endmathjax %}中的项接近1时，我们回复一个普通的循环神经网络。对于重置门{% mathjax %}\mathbf{R}_t{% endmathjax %}中所有接近0的项。候选隐状态是以{% mathjax %}\mathbf{X}_t{% endmathjax %}作为输入的多层感知机的结果。因此，任何预先存在的隐状态都会被重置为默认值。下图说明了应用重置门之后的计算流程：
{% asset_img rnn_2.png "在门控循环单元模型中计算候选隐状态" %}

###### 隐状态

上述的计算结果只是候选隐状态，我们仍然需要结合更新门{% mathjax %}\mathbf{Z}_t{% endmathjax %}的效果。这一步确定新的隐状态{% mathjax %}\mathbf{H}_t\in \mathbb{R}^{n\times h}{% endmathjax %}在多大程度上来自旧的状态{% mathjax %}\mathbf{H}_{t-1}{% endmathjax %}和新的候选状态{% mathjax %}\tilde{\mathbf{H}}_t{% endmathjax %}。更新门{% mathjax %}\mathbf{Z}_t{% endmathjax %}仅需要在{% mathjax %}\mathbf{H}_{t-1}{% endmathjax %}和{% mathjax %}\tilde{\mathbf{H}}_t{% endmathjax %}之间进行按元素的凸组合就可以实现这个目标。这就得出了门控神经单元的最终更新公式：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{H}_t = \mathbf{Z}_t\odot\mathbf{H}_{t-1} + (1 - \mathbf{Z}_t)\odot\tilde{\mathbf{H}}_t
{% endmathjax %}
每当更新门{% mathjax %}\mathbf{Z}_t{% endmathjax %}接近{% mathjax %}1{% endmathjax %}时，模型就倾向只保留旧状态。此时，来自{% mathjax %}\mathbf{X}_t{% endmathjax %}的信息基本上被忽略，从而有效地跳过了依赖链条中的时间步{% mathjax %}t{% endmathjax %}。相反，当{% mathjax %}\mathbf{Z}_t{% endmathjax %}接近{% mathjax %}0{% endmathjax %}时，新的隐状态{% mathjax %}\mathbf{H}_t{% endmathjax %}就会接近候选隐状态{% mathjax %}\tilde{\mathbf{H}}_t{% endmathjax %}。这些设计可以帮助我们处理循环神经网络中的梯度消失问题，并更好地捕获时间步距离很长的序列的依赖关系。例如，如果整个子序列的所有时间步的更新门都接近于`1`，则无论序列的长度如何，在序列起始时间步的旧隐状态都将很容易保留并传递到序列结束。下图说明了更新门起作用后的计算流。
{% asset_img rnn_3.png "计算门控循环单元模型中的隐状态" %}

总之，门控循环单元具有以下两个显著特征：
- 重置门有助于捕获序列中的短期依赖关系。
- 更新门有助于捕获序列中的长期依赖关系。

##### 总结

门控循环神经网络可以更好地捕获时间步距离很长的序列上的依赖关系。重置门有助于捕获序列中的短期依赖关系。更新门有助于捕获序列中的长期依赖关系。重置门打开时，门控循环单元包含基本循环神经网络；更新门打开时，门控循环单元可以跳过子序列。

#### 长短期记忆网络（LSTM）

长期以来，隐变量模型存在着长期信息保存和短期输入缺失的问题。解决这一问题的最早方法之一是**长短期存储器**(`long short-term memory，LSTM`)。它有许多与门控循环单元一样的属性。有趣的是，长短期记忆网络的设计比门控循环单元稍微复杂一些，却比门控循环单元早诞生了近`20`年。
##### 门控记忆元

可以说，长短期记忆网络的设计灵感来自于计算机的逻辑门。长短期记忆网络引入了记忆元(`memory cell`)，或简称为单元(`cell`)。有些文献认为记忆元是隐状态的一种特殊类型，它们与隐状态具有相同的形状，其设计目的是用于记录附加的信息。为了控制记忆元，我们需要许多门。其中一个门用来从单元中输出条目，我们将其称为输出门(`output gate`)。另外一个门用来决定何时将数据读入单元，我们将其称为输入门(`input gate`)。我们还需要一种机制来重置单元的内容，由遗忘门(`forget gate`)来管理，这种设计的动机与门控循环单元相同，能够通过专用机制决定什么时候记忆或忽略隐状态中的输入。让我们看看这在实践中是如何运作的。
###### 输入门、遗忘门和输出门

就如在门控循环单元中一样，当前时间步的输入和前一个时间步的隐状态作为数据送入长短期记忆网络的门中，如下图所示。它们由三个具有`sigmoid`激活函数的全连接层处理，以计算输入门、遗忘门和输出门的值。因此，这三个门的值都在{% mathjax %}(0,1){% endmathjax %}的范围内。
{% asset_img rnn_4.png "长短期记忆模型中的输入门、遗忘门和输出门" %}

我们来细化一下长短记忆网络的数学表达。假设{% mathjax %}h{% endmathjax %}个隐藏单元，批量大小为{% mathjax %}n{% endmathjax %}，输入数为{% mathjax %}d{% endmathjax %}。因此，输入为{% mathjax %}\mathbf{X}_t\in \mathbb{R}^{n\times d}{% endmathjax %}，前一时间步的隐状态为{% mathjax %}\mathbf{H}_{t-1}\in\mathbb{R}^{n\times h}{% endmathjax %}。相应地时间步{% mathjax %}t{% endmathjax %}的门被定义如下：输入门是{% mathjax %}\mathbf{I}_t\in\mathbb{R}^{n\times h}{% endmathjax %}，遗忘门是{% mathjax %}\mathbf{F}_t\in\mathbb{R}^{n\times h}{% endmathjax %}，输出门是{% mathjax %}\mathbf{O}_t\in\mathbb{R}^{n\times h}{% endmathjax %}。它们的计算方法如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{I}_t & = \sigma(\mathbf{X}_t\mathbf{W}_{xi} + \mathbf{H}_{t-1}\mathbf{W}_{hi} + \mathbf{b}_i) \\ 
\mathbf{F}_t & = \sigma(\mathbf{X}_t\mathbf{W}_{xf} + \mathbf{H}_{t-1}\mathbf{W}_{hf} + \mathbf{b}_f) \\
\mathbf{O}_t & = \sigma(\mathbf{X}_t\mathbf{W}_{xo} + \mathbf{H}_{t-1}\mathbf{W}_{ho} + \mathbf{b}_o) \\
\end{align}
{% endmathjax %}
其中{% mathjax %}\mathbf{X}_{xi}{% endmathjax %}，{% mathjax %}\mathbf{W}_{xf}{% endmathjax %}，{% mathjax %}\mathbf{W}_{xo}\in\mathbb{R}^{d\times h}{% endmathjax %}和{% mathjax %}\mathbf{W}_{hi}{% endmathjax %}，{% mathjax %}\mathbf{W}_{hf}{% endmathjax %}，{% mathjax %}\mathbf{W}_{ho}\in\mathbb{R}^{h\times h}{% endmathjax %}是权重参数{% mathjax %}\mathbf{b}_i{% endmathjax %}，{% mathjax %}\mathbf{b}_f{% endmathjax %}，{% mathjax %}\mathbf{b}_o\in\mathbb{R}^{1\times h}{% endmathjax %}是偏置参数。
###### 候选记忆元

由于还没有指定各种门的操作，所以先介绍**候选记忆元**(`candidate memory cell`){% mathjax %}\tilde{\mathbf{C}}_t\in\mathbb{R}_^{n\times h}{% endmathjax %}。它的计算与上面描述的三个门的计算类似，但是使用{% mathjax %}\tanh{% endmathjax %}函数作为激活函数，函数的值范围为{% mathjax %}(-1,1){% endmathjax %}。下面导出在时间步{% mathjax %}t{% endmathjax %}处的方程：
{% mathjax '{"conversion":{"em":14}}' %}
\tiled{\mathbf{C}}_t = \tanh(\mathbf{X}_t\mathbf{W}_{xc} + \mathbf{H}_{t-1}\mathbf{W}_{hc} + \mathbf{b}_c)
{% endmathjax %}
其中{% mathjax %}\mathbf{W}_{xc}\in\mathbb{R}^{d\times h}{% endmathjax %}和{% mathjax %}\mathbf{W}_{hc}\in\mathbb{R}^{h\times h}{% endmathjax %}是权重参数，{% mathjax %}\mathbf{b}_c\in\mathbb{R}^{1\times h}{% endmathjax %}是偏置参数。候选记忆元如下图所示：
{% asset_img rnn_5.png "长短期记忆模型中的候选记忆元" %}

###### 记忆元

在门控循环单元中，有一种机制来控制输入和遗忘（或跳过）。类似地，在长短期记忆网络中，也有两个门用于这样的目的：输入门{% mathjax %}\mathbf{I}_t{% endmathjax %}控制采用多少来自{% mathjax %}\tilde{\mathbf{C}}_t{% endmathjax %}的新数据，而遗忘门{% mathjax %}\mathbf{F}_t{% endmathjax %}控制保留多少过去的 记忆元{% mathjax %}\\mathbf{C}_{t-1}\in\mathbb{R}^{n\times h}{% endmathjax %}的内容。使用按元素乘法，得出：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{C}_t = \mathbf{F}_t\odot\mathbf{C}_{t-1} + \mathbf{I}_t\odot\tilde{\mathbf{C}}_t
{% endmathjax %}
如果遗忘门始终为`1`且输入门始终为`0`，则过去的记忆元{% mathjax %}\mathbf{C}_{t-1}{% endmathjax %}将随时间被保存并传递到当前时间步。引入这种设计是为了缓解梯度消失问题，并更好地捕获序列中的长距离依赖关系。这样我们就得到了计算记忆元的流程图：
{% asset_img rnn_6.png "在长短期记忆网络模型中计算记忆元" %}

###### 隐状态

最后，我们需要定义如何计算隐状态{% mathjax %}\mathbf{H}_t\in\mathbb{R}^{n\times h}{% endmathjax %}，这就是输出门发挥作用的地方。在长短期记忆网络中，它仅仅是记忆元的{% mathjax %}\tanh{% endmathjax %}的门控版本。这就确保了{% mathjax %}\mathbf{H}_t{% endmathjax %}的值始终在区间{% mathjax %}(-1,1){% endmathjax %}内：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{H}_t = \mathbf{O}_t\odot\tanh(\mathbf{C}_t)
{% endmathjax %}
只要输出门接近`1`，我们就能够有效地将所有记忆信息传递给预测部分，而对于输出门接近`0`，我们只保留记忆元内的所有信息，而不需要更新隐状态。
{% asset_img rnn_7.png "在长短期记忆模型中计算隐状态" %}

##### 总结

长短期记忆网络有三种类型的门：输入门、遗忘门和输出门。长短期记忆网络的隐藏层输出包括“隐状态”和“记忆元”。只有隐状态会传递到输出层，而记忆元完全属于内部信息。长短期记忆网络可以缓解梯度消失和梯度爆炸。

#### 深度循环神经网络

到目前为止，我们只讨论了具有一个单向隐藏层的循环神经网络。其中，隐变量和观测值与具体的函数形式的交互方式是相当随意的。只要交互类型建模具有足够的灵活性，这就不是一个大问题。然而，对一个单层来说，这可能具有相当的挑战性。之前在线性模型中，我们通过添加更多的层来解决这个问题。而在循环神经网络中，我们首先需要确定如何添加更多的层，以及在哪里添加额外的非线性，因此这个问题有点棘手。事实上，我们可以将多层循环神经网络堆叠在一起，通过对几个简单层的组合，产生了一个灵活的机制。特别是，数据可能与不同层的堆叠有关。例如，我们可能希望保持有关金融市场状况 （熊市或牛市）的宏观数据可用，而微观数据只记录较短期的时间动态。下图描述了一个具有{% mathjax %}L{% endmathjax %}个隐藏层的深度循环神经网络，每个隐状态都连续地传递到当前层的下一个时间步和下一层的当前时间步。
{% asset_img rnn_8.png "深度循环神经网络结构" %}

##### 函数依赖关系

