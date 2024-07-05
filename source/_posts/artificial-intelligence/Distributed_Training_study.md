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

#### 分布式—数据并行训练

假设您有一个在单台计算机/`GPU`上运行的训练脚本，但运行速度非常慢，因为：数据集很大，您不能使用大批量训练，因为这会导致`CUDA`出现内存不足错误。分布式—数据并行是这种情况的解决方案。它适用于以下场景：
{% asset_img dt_20.png %}

从现在开始，我将交替使用“节点”和`“GPU”`这两个术语。如果一个集群由`2`台服务器组成，每台服务器有`2`个`GPU`，那么我们总共有`4`个节点。分布式—数据并行的工作方式如下：
- 在训练开始时，模型的权重在一个节点上初始化，并发送到所有其他节点（广播）。
- 每个节点在数据集的子集上训练相同的模型（具有相同的初始权重）。
- 每隔几个批次，每个节点的梯度都会在一个节点上累积（总结），然后发送回所有其他节点（`All-Reduce`）。
- 每个节点使用自己的优化器用收到的梯度更新其本地模型的参数。
- 返回步骤`2`。

{% asset_img dt_21.png %}

|`Step`|`Description`|`Distributed Data Parallel`|
|:---|:---|:---|
|`step:1`|模型权重初始化（例如，随机）|{% asset_img dt_22.png %}|
|`step:1`|初始权重被发送到所有其他节点（广播）|{% asset_img dt_23.png %}|
|`step:2`|每个节点对一个或多个批次的数据进行前向和后向运算。这将产生局部梯度。局部梯度可能是一个或多个批次的累积。|{% asset_img dt_24.png %}|
|`step:3`|所有梯度的总和累积在一个节点上（`Reduce`）|{% asset_img dt_25.png %}|
|`step:3`|累积梯度被发送到所有其他节点(广播)。归约(`Reduce`)和广播(`Broadcast`)的序列被实现为单一操作(全归约,`All-Reduce`)。|{% asset_img dt_26.png %}|
|`step:4`|每个节点使用收到的梯度更新其本地模型的参数。更新后，梯度将重置为零，我们可以开始另一个循环。|{% asset_img dt_27.png %}|

在分布式计算环境中，一个节点可能需要与其他节点通信。如果通信模式类似于客户端和服务器，那么我们谈论的是点对点通信，因为一个客户端在请求-响应事件链中连接到一个服务器。但是，在某些情况下，一个节点需要同时与多个接收器通信：这是深度学习中数据并行训练的典型情况：一个节点需要将初始权重发送给所有其他节点。此外，所有其他节点都需要将其梯度发送到一个节点并接收累积梯度。**集体通信**允许对节点组之间的通信模式进行建模。让我们直观地看一下这两种通信模式之间的区别。
##### 点对点模式

假设您需要向`7`位好友发送一个文件。通过点对点通信，您可以逐一迭代地将文件发送给每位好友。假设网速为`1MB/s`，文件大小为`5MB`。
{% asset_img dt_28.png %}

{% asset_img dt_29.png %}

{% asset_img dt_30.png %}

{% asset_img dt_31.png %}

{% asset_img dt_32.png %}

{% asset_img dt_33.png %}

{% asset_img dt_34.png %}

如果我们同时将文件发送给所有`7`位朋友会怎么样？由于互联网通信速度为`1MB/s`，文件大小为`5MB`，您的连接将在7位朋友之间分配（每位朋友将以约`143 KB/s`的速度接收文件）。总时间仍为`35`秒。
{% asset_img dt_35.png %}

##### 广播模式(集体通信)

将数据发送到所有其他节点的操作称为广播操作。集体通信库（例如`NCCL`）为每个节点分配一个唯一`ID`，称为`RANK`。假设我们想以`1MB/s`的互联网速度发送`5MB`。
{% asset_img dt_36.png %}

{% asset_img dt_37.png %}

{% asset_img dt_38.png %}

这种方法被称为**分而治之法**。通过**集体通信**，我们利用节点之间的互连性来避免空闲时间并减少总通信时间。**广播**(`Broadcast`)操作用于在开始训练循环时将初始权重发送给所有其他节点。每个节点每处理几批数据，就需要将所有节点的梯度发送到一个节点并进行累加（求和）。此操作称为**规约**(`Reduce`)。让我们直观地了解一下它的工作原理。
##### 规约(集体通信)

