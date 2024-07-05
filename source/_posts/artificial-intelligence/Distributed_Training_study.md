---
title: 分布式训练—架构（深度学习&数据处理）
date: 2024-07-04 14:30:11
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

**分布式训练**是一种将模型训练工作负载分散到多个处理单元（如`GPU`或计算节点）上的技术，以加速训练过程并提高模型性能。分布式训练通过将训练任务分配给多个工作节点（`worker nodes`），这些节点并行工作，从而加速模型训练。分布式训练特别适用于深度学习模型，因为这些模型通常具有大量参数和计算需求。**分布式训练分类**：
<!-- more -->
- **数据并行**(`Data Parallelism`)：数据并行是最常见的分布式训练方法。它将训练数据分成多个部分，每个工作节点处理一个数据子集。每个节点都有完整的模型副本，并独立计算其数据子集的梯度。然后，这些梯度在所有节点之间同步，以更新模型参数。优点：实现简单，适用于大多数深度学习任务。缺点：每个节点需要足够的内存来存储整个模型。
- **模型并行**(`Model Parallelism`)：模型并行将模型本身分割成多个部分，每个节点处理模型的一部分。每个节点处理相同的数据，但只计算其负责的模型部分的梯度。优点：适用于超大模型，单个节点无法容纳整个模型。缺点：实现复杂，节点之间需要频繁通信。
- **流水线并行**(`Pipeline Parallelism`)：流水线并行将模型分割成多个阶段，每个阶段由不同的节点处理。数据在节点之间依次传递，每个节点处理其阶段的计算。优点：提高了计算资源的利用率。缺点：实现复杂，可能引入额外的延迟。

假设你想在一个非常大的数据集上训练一个语言模型，例如整个维基百科的内容。这个数据集非常大，因为它由数百万篇文章组成，每篇文章都有数千个`token`。在单个`GPU`上训练这个模型是可行的，但它带来了一些挑战：
- 模型可能不适合单个`GPU`：当模型具有许多参数时会发生这种情况。
- 您被迫使用较小的批量大小，因为较大的批量大小会导致`CUDA`出现内存不足错误。
- 由于数据集巨大，该模型可能需要数年时间才能训练完成。

如果你遇到了以上任何一种情况，那么你需要扩展你的训练设置。扩展可以垂直进行，也可以水平进行。让我们比较一下这两个选项。
{% asset_img dt_1.png %}

如果模型可以装入单个`GPU`，那么我们可以将训练分布在多台服务器上（每台服务器包含一个或多个`GPU`），每台`GPU`并行处理整个数据集的一个子集，并在反向传播期间同步梯度。此选项称为**数据并行**。
{% asset_img dt_2.png %}

如果模型无法容纳在单个`GPU`中，那么我们需要将模型“分解”为更少的层，并让每个`GPU`在梯度下降过程中处理前向/后向步骤的一部分。此选项称为**模型并行性**。
{% asset_img dt_3.png %}

#### 神经网络回顾

假设您想要训练一个神经网络来预测房屋的价格({% mathjax %}y_{\text{pred}}{% endmathjax %})，给定两个变量：房屋中的卧室数量({% mathjax %}x_1{% endmathjax %})和房屋中的浴室数量({% mathjax %}x_2{% endmathjax %})。我们认为输出和输入变量之间的关系是线性的。
{% mathjax '{"conversion":{"em":14}}' %}
y_{\text{pred}} = x_1w_1 + x_2w_2 + b
{% endmathjax %}
我们的目标是使用随机梯度下降来找到参数{% mathjax %}w_1{% endmathjax %}、{% mathjax %}w_2{% endmathjax %}和{% mathjax %}b{% endmathjax %}的值，使得实际房价({% mathjax %}y_{\text{target}}{% endmathjax %})和预测房价({% mathjax %}y_{\text{pred}}{% endmathjax %})之间的`MSE`损失最小化。
{% mathjax '{"conversion":{"em":14}}' %}
\underset{w_1,w_2,b}{\text{argmin}}(y_{\text{pred}} - y_{\text{target}})^2
{% endmathjax %}
##### 计算图

`PyTorch`会将我们的神经网络转换为计算图。让我们使用计算图一次可视化一个项目的训练过程：
|`Step`|`Description`|`Computational graph(without accumulation)`|
|:---|:---|:---|
|`1:forward`|使用输入{% mathjax %}x_1 = 6{% endmathjax %}、{% mathjax %}x_2 = 2{% endmathjax %}和{% mathjax %}y_{\text{target}} = 15{% endmathjax %}运行前向传播，并初始化权重。|{% asset_img dt_4.png %}<br>现在调用`loss.backward()`方法来计算损失函数相对于每个参数的梯度。|
|`1:loss.backward`||{% asset_img dt_5.png %}|
|`1:optimizer.step`|假设学习率为{% mathjax %}a = 10^{-3}{% endmathjax %}。每个参数更新如下：<br>{% mathjax %}\text{param}_{\text{new}}=\text{param}_{\text{old}} - a\times \text{grad}{% endmathjax %}<br>{% mathjax %}y_{\text{pred}} = x_1w_1 + x_2w_2 + b{% endmathjax %}|{% asset_img dt_6.png %}|
|`1:optimizer.zero`|将所有参数的梯度重置为`0`|{% asset_img dt_7.png %}|
|`2:forward`|使用输入{% mathjax %}x_1 = 5、x_2 = 2{% endmathjax %}和{% mathjax %}y_{\text{target}} = 12{% endmathjax %}运行前向传播|{% asset_img dt_8.png %}|
|`2:loss.backward`||{% asset_img dt_9.png %}|
|`2:optimizer.step`||{% asset_img dt_10.png %}|
|`2:optimizer.zero`||{% asset_img dt_11.png %}|

梯度下降（不带累积）：没有梯度积累，在每一步（每个数据项）更新模型的参数。
{% asset_img dt_12.png %}

|`Step`|`Description`|`Computational graph(with accumulation)`|
|:---|:---|:---|
|`1:forward`|使用输入{% mathjax %}x_1 = 6{% endmathjax %}、{% mathjax %}x_2 = 2{% endmathjax %}和{% mathjax %}y_{\text{target}} = 15{% endmathjax %}运行前向传播，并初始化权重。|{% asset_img dt_13.png %}|
|`1:loss.backward`||{% asset_img dt_14.png %}|
|`2:forward`|使用输入{% mathjax %}x_1 = 5、x_2 = 2{% endmathjax %}和{% mathjax %}y_{\text{target}} = 12{% endmathjax %}运行前向传播|{% asset_img dt_15.png %}|
|`2:loss.backward`|新的梯度与旧的梯度一起累积（求和）。现在已经达到了批量大小，可以运行`optimizer.step`方法|{% asset_img dt_16.png %}|
|`2:optimizer.step`||{% asset_img dt_17.png %}|
|`2:optimizer.zero`||{% asset_img dt_18.png %}|

梯度下降（带累积）：通过梯度积累，我们只需积累一批梯度后即可更新模型的参数。
{% asset_img dt_19.png %}

#### 分布式数据并行训练

假设您有一个在单台计算机/`GPU`上运行的训练脚本，但运行速度非常慢，因为：数据集很大，您不能使用大批量训练，因为这会导致`CUDA`出现内存不足错误。分布式数据并行是这种情况的解决方案。它适用于以下场景：
{% asset_img dt_20.png %}

从现在开始，我将交替使用“节点”和`“GPU”`这两个术语。如果一个集群由`2`台服务器组成，每台服务器有`2`个`GPU`，那么我们总共有`4`个节点。分布式数据并行的工作方式如下：
- 在训练开始时，模型的权重在一个节点上初始化，并发送到所有其他节点（广播）。
- 每个节点在数据集的子集上训练相同的模型（具有相同的初始权重）。
- 每隔几个批次，每个节点的梯度都会在一个节点上累积（总结），然后发送回所有其他节点（`All-Reduce`）。
- 每个节点使用自己的优化器用收到的梯度更新其本地模型的参数。
- 返回步骤`2`。

{% asset_img dt_21.png %}

