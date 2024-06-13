---
title: Transformer系列（机器学习）
date: 2024-06-08 10:50:11
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

`Transformer`（将被称为`“vanilla Transformer”`以区别于其他增强版本；`Vaswani`等人，`2017`年）模型具有**编码器-解码器架构**，这在许多`NMT`模型中很常见。后来简化的 `Transformer`在语言建模任务中表现出色，例如在仅编码器的`BERT`或仅解码器的`GPT`中。
<!-- more -->
#### 符号

|符号|含义|
|:--|:--|
|{% mathjax %}d{% endmathjax %}|模型大小/隐藏状态维度/位置编码大小。|
|{% mathjax %}h{% endmathjax %}|多头注意力层中的头的数量。|
|{% mathjax %}L{% endmathjax %}|输入序列的段长度。|
|{% mathjax %}N{% endmathjax %}|模型中注意力层的总数；不考虑`MoE`。|
|{% mathjax %}\mathbf{X}\in \mathbb{R}^{L\times d}{% endmathjax %}|	输入序列中的每个元素都被映射到形状为{% mathjax %}d{% endmathjax %}，与模型尺寸相同。|
|{% mathjax %}\mathbf{W}^k\in \mathbb{R}^{d\times d_k}{% endmathjax %}|键权重矩阵。|
|{% mathjax %}\mathbf{W}^q\in \mathbb{R}^{d\times d_k}{% endmathjax %}|查询权重矩阵。|
|{% mathjax %}\mathbf{W}^v\in \mathbb{R}^{d\times d_v}{% endmathjax %}|值权重矩阵。通常我们有{% mathjax %}d_k = d_v = d。{% endmathjax %}|
|{% mathjax %}\mathbf{W}_i^k,\mathbf{W}_i^q\in \mathbb{R}^{d\times d_k/h};\mathbf{W}_i^v\in \mathbb{R}^{d\times d_v/h}{% endmathjax %}|每个头的权重矩阵|
|{% mathjax %}\mathbf{W}^o\in \mathbb{R}^{d_v\times d}{% endmathjax %}|输出权重矩阵。|
|{% mathjax %}\mathbf{Q} = \mathbf{XW}^q\in \mathbb{R}^{L\times d_k}{% endmathjax %}|嵌入输入的查询。|
|{% mathjax %}\mathbf{K} = \mathbf{XW}^k\in \mathbb{R}^{L\times d_k}{% endmathjax %}|嵌入输入的键。|
|{% mathjax %}\mathbf{V} = \mathbf{XW}^v\in \mathbb{R}^{L\times d_v}{% endmathjax %}|嵌入输入的值。|
|{% mathjax %}\mathbf{q}_i,\mathbf{k}_i\in \mathbb{R}^{d_k},\mathbf{v}_i\in \mathbb{R}^{d_v}{% endmathjax %}|查询、键、值矩阵中的行向量，{% mathjax %}\mathbf{Q,K}{% endmathjax %}和{% mathjax %}\mathbf{V}{% endmathjax %}|
|{% mathjax %}\mathbf{S}_i{% endmathjax %}|第{% mathjax %}i{% endmathjax %}个查询{% mathjax %}\mathbf{q}_i{% endmathjax %}的键值对集合|
|{% mathjax %}\mathbf{A}\in \mathbb{R}^{L\times L}{% endmathjax %}|长度为{% mathjax %}L{% endmathjax %}的注意力，{% mathjax %}\mathbf{A} = \text{softmax}(\mathbf{QK}^{\mathsf{T}}/\sqrt{d_k}){% endmathjax %}。|
|{% mathjax %}a_{ij}\in \mathbf{A}{% endmathjax %}|查询{% mathjax %}i{% endmathjax %}和{% mathjax %}j{% endmathjax %}之间的标量注意得分{% mathjax %}\mathbf{q}_i{% endmathjax %}和键{% mathjax %}\mathbf{k}_j{% endmathjax %}。|
|{% mathjax %}P\in \mathbb{R}^{L\times d}{% endmathjax %}|位置编码矩阵，其中第{% mathjax %}i{% endmathjax %}行的{% mathjax %}\mathbf{p}_i{% endmathjax %}是输入的位置编码{% mathjax %}\mathbf{x}_i{% endmathjax %}。|
#### 注意力机制与自注意力机制

**注意力**是神经网络中的一种机制，模型可以通过选择性地关注给定的一组数据来学习做出预测。注意力的量由学习到的权重作为量化，因此输出通常为加权平均值。

**自注意力**是一种注意力机制，模型利用对同一样本的其他部分观察结果对数据样本的一部分进行预测。从概念上讲，它与**非局部均值**非常相似。注意，自注意力是置换不变的；它是对集合的操作。

注意力/自注意力有多种形式，`Transformer`依赖于缩放点积注意力：给定一个查询矩阵{% mathjax %}\mathbf{Q}{% endmathjax %}，键矩阵{% mathjax %}\mathbf{K}{% endmathjax %}和一个值矩阵{% mathjax %}\mathbf{V}{% endmathjax %}，输出是值向量的加权和，其中分配给每个值的权重由查询和对应的键的点积决定：
{% mathjax '{"conversion":{"em":14}}' %}
\text{attn}(\mathbf{Q,K,V}) = \text{softmax}(\frac{\mathbf{QK^{\mathsf{T}}}}{\sqrt{d_k}})\mathbf{V}
{% endmathjax %}
对于查询和键向量{% mathjax %}\mathbf{q}_i,\mathbf{k}_i\in \mathbb{R}^d{% endmathjax %}（查询和键矩阵中的行向量），我们有一个标量分数：
{% mathjax '{"conversion":{"em":14}}' %}
a_{ij} = \text{softmax}(\frac{\mathbf{q}_i {\mathbf{k}_j}^\top}{\sqrt{d_k}}) = \frac{\exp(\mathbf{q}_i {\mathbf{k}_j}^\top)}{\sqrt{d_k}\sum_{r\in S_i } \exp(\mathbf{q}_i {\mathbf{k}_r}^\top)}
{% endmathjax %}
#### 多头自注意力

**多头自注意力**模块是`Transformer`中的关键组件。多头机制不是只计算一次注意力，而是将输入拆分成更小的块，然后并行计算每个子块上的缩放点积注意力。单个注意力输出只是简单地连接起来并线性变换为预期的维度。
{% mathjax '{"conversion":{"em":14}}' %}
\text{MultiHeadAttn}(\mathbf{X}_q,\mathbf{X}_k,\mathbf{X}_v) = [\text{head}_1;\ldots;\text{head}_h]\mathbf{W}^o\;\;\text{where head}_i = \text{Attention}(\mathbf{X}_q \mathbf{W}_i^q, \mathbf{X}_k \mathbf{W}_i^k, \mathbf{X}_v \mathbf{W}_i^v)
{% endmathjax %}
在这里{% mathjax %}[\cdot;\cdot]{% endmathjax %}是一个连接运算。{% mathjax %}\mathbf{W}_i^q,\mathbf{W}_i^k\in \mathbb{R}^{d\times d_k/h}, \mathbf{W}_i^v\in \mathbb{R}^{d\times d_v/h}{% endmathjax %}是权重矩阵，用于映射大小为{% mathjax %}L\times d{% endmathjax %}的查询、键和值矩阵。并且{% mathjax %}\mathbf{W}^o\in \mathbb{R}^{d_v\times d}{% endmathjax %}是输出线性变换。所有权重都应在训练期间学习。
{% asset_img t_1.png "多头缩放点积注意机制示意图" %}

#### 编码器-解码器架构

**编码器**生成基于注意力机制的表示，能够从上下文中定位特定信息。它由`6`个模块的堆栈组成，每个模块包含两个子模块、一个多头自注意力层和一个逐点全连接前馈网络。逐点意味着它对序列中的每个元素应用相同的线性变换（具有相同的权重）。这也可以看作是过滤器大小为`1`的卷积层。每个子模块都有一个残差连接和层规范化。所有子模块都输出相同维度的数据{% mathjax %}d{% endmathjax %}。`Transformer`解码器的功能是从编码表示中检索信息。该架构与编码器非常相似，不同之处在于解码器包含两个多头注意子模块，而不是相同的重复模块中都有一个。第一个多头注意子模块被屏蔽，以防止位置关注未来。
{% asset_img t_2.png "原始Transformer模型的架构" %}

#### 位置编码

由于自注意力操作具有排列不变性，因此使用适当的位置编码为模型提供顺序信息非常重要。位置编码{% mathjax %}\mathbf{P}\in \mathbb{R}^{L\times d}{% endmathjax %}与输入嵌入具有相同的维度，因此可以直接添加到输入中。`vanilla Transformer`考虑了两种类型的编码：
##### 正弦位置编码

正弦位置编码定义如下，给定`token`位置{% mathjax %}i=1,\ldots,L{% endmathjax %}和维度{% mathjax %}\delta = 1,\ldots,d{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{PE}(i,\delta) = 
\begin{cases}
    \sin(\frac{i}{10000^{2\delta '/d}}) & \text{if }\delta = 2\delta '\\
    \cos(\frac{i}{10000^{2\delta '/d}}) & \text{if }\delta = 2\delta ' + 1\\
\end{cases}
{% endmathjax %}
这样，位置编码的每个维度都对应于不同维度中不同波长的正弦波，从{% mathjax %}2\pi{% endmathjax %}到{% mathjax %}10000\cdot 2\pi{% endmathjax %}。
{% asset_img t_3.png "正弦位置编码L=32和d=128。介于-1(黑色)~1(白色)之间，值-0为灰色" %}

##### 学习位置编码

学习到的位置编码为每个元素分配一个学习到的列向量，该向量对其绝对位置进行编码（`Gehring`等人，`2017`年），而且每层都可以通过不同的方式学习这种编码（`Al-Rfou`等人，`2018`年）。
##### 相对位置编码

`Shaw`等人（`2018`年）将相对位置信息纳入{% mathjax %}\mathbf{W}^k{% endmathjax %}和{% mathjax %}\mathbf{W}^v{% endmathjax %}最大相对位置被限制为最大绝对值{% mathjax %}k{% endmathjax %}这种裁剪操作使得模型能够推广到未知的序列长度。因此，在{% mathjax %}2k + 1{% endmathjax %}的标签范围内，将{% mathjax %}\mathbf{P}^k,\mathbf{P}^v\in \mathbb{R}^{2k+1}{% endmathjax %}作为可学习的相对位置表示。
{% mathjax '{"conversion":{"em":14}}' %}
A_{ij}^k = P_{\text{clip}(j-i,k)}^k\;A_{ij}^v = P_{\text{clip}(j-i,k)}^v\; \text{where clip}(x,k) = \text{clip}(x,-k,k)
{% endmathjax %}
`Transformer-XL`（`Dai`等人，`2019`年）提出了一种基于键和查询点积重参数化的相对位置编码。为了保持位置信息在各个段之间连贯流动，`Transformer-XL`对相对位置进行编码，因为知道位置偏移量就足以做出良好的预测，即{% mathjax %}i-j{% endmathjax %}，一个键向量{% mathjax %}\mathbf{k}_{\tau , j}{% endmathjax %}及其查询向量{% mathjax %}\mathbf{q}_{\tau , i}{% endmathjax %}。如果省略标量{% mathjax %}、\frac{1}{\sqrt{d_k}}{% endmathjax %}以及{% mathjax %}\text{softmax}{% endmathjax %}中的归一化项，包括位置编码，我们可以将位置处的查询注意力得分写为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
a_{ij} & = \mathbf{q}_i {\mathbf{k}_j}^\top = (\mathbf{x}_i + \mathbf{p}_i)\mathbf{W}^q ((\mathbf{x}_j + \mathbf{p}_j)\mathbf{W}^k)^\top \\
& = \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top \mathbf{x}_j^\top + \mathbf{x}_i\mathbf{W}^q {\mathbf{W}^k}^\top \mathbf{p}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top \mathbf{x}_j^\top + \mathbf{p}_i\mathbf{W}^q {\mathbf{W}^k}^\top \mathbf{p}_j^\top 
\end{align}
{% endmathjax %}
`Transformer-XL`对上述四个项重新参数化如下：
{% mathjax '{"conversion":{"em":14}}' %}
a_{ij}^{\text{rel}} = 
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_{\text{content-based addressing}} +
\underbrace{ \mathbf{x}_i\mathbf{W}^q \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{\mathbf{r}_{i-j}^\top} }_{\text{content-dependent positional bias}} +
\underbrace{ \color{red}{\mathbf{u}} \color{blue}{ {\mathbf{W}_E^k}^\top } \mathbf{x}_j^\top }_{\text{global content bias}} +
\underbrace{ \color{red}{\mathbf{v}} \color{blue}{ {\mathbf{W}_R^k}^\top } \color{green}{ \mathbf{x}_{i-j}^\top} }_{\text{global positional bias}}
{% endmathjax %}
- 代替{% mathjax %}\mathbf{p}_j{% endmathjax %}使用相对位置编码{% mathjax %}\mathbf{r}_{i-j}\in \mathbb{R}^d{% endmathjax %}。
- 代替{% mathjax %}\mathbf{p}_i\mathbf{W}^q{% endmathjax %}具有两个可训练参数{% mathjax %}\mathbf{u}{% endmathjax %}(内容)和{% mathjax %}\mathbf{v}{% endmathjax %}(表示位置)有两个不同的术语。
- 将{% mathjax %}\mathbf{W}^k{% endmathjax %}分裂成两个矩阵，{% mathjax %}\mathbf{W}_E^k{% endmathjax %}为内容信息，{% mathjax %}\mathbf{W}_R^k{% endmathjax %}为位置信息。
##### 旋转位置嵌入

旋转位置嵌入（`RoPE`；`Su`等人，`2021`）使用旋转矩阵对绝对位置进行编码，并将每个注意层的键和值矩阵与其相乘，以在每一层注入相对位置信息。当将相对位置信息编码到第{% mathjax %}i{% endmathjax %}键和第{% mathjax %}j{% endmathjax %}个查询，我们希望以这样一种方式来定义函数，即内积仅与相对位置{% mathjax %}i-j{% endmathjax %}有关。旋转位置嵌入（`RoPE`）利用欧几里得空间中的旋转操作，将相对位置嵌入构建为简单地将特征矩阵旋转与其位置索引成比例的角度。

给定一个向量{% mathjax %}\mathbf{z}{% endmathjax %}，如果我们想将其逆时针旋转{% mathjax %}\theta{% endmathjax %}，我们可以将其乘以旋转矩阵得到{% mathjax %}R\mathbf{z}{% endmathjax %}旋转矩阵{% mathjax %}R{% endmathjax %}定义为：
{% mathjax '{"conversion":{"em":14}}' %}
R = 
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
{% endmathjax %}
当推广到更高维空间时，`RoPE`将{% mathjax %}d{% endmathjax %}拓展到{% mathjax %}d/2{% endmathjax %}维空间，并构造旋转矩阵{% mathjax %}R{% endmathjax %}，大小为{% mathjax %}d\times d{% endmathjax %}，位置处的标记为{% mathjax %}i{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
R^d_{\Theta,i} = 
\begin{bmatrix}
\cos i\theta_1 & -\sin i\theta_1 & 0 & 0 & \ldots & 0 & 0 \\
\sin i\theta_1 & \cos i\theta_1 & 0 & 0 & \ldots & 0 & 0 \\
0 & 0 & \cos i\theta_2 & -\sin i\theta_2 & \ldots & 0 & 0 \\
0 & 0 & \sin i\theta_2 & \cos i\theta_2 & \ldots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \ldots & \cos i\theta_{d/2} & -\sin i\theta_{d/2} \\
0 & 0 & 0 & 0 & \ldots & \sin i\theta_{d/2} & \cos i\theta_{d/2} \\
\end{bmatrix}
{% endmathjax %}
在论文中{% mathjax %}\Theta = \theta_i = 10000^{-2(i-1)/d}, i\in [1,2,\ldots, d/2]{% endmathjax %}。请注意，这本质上等同于张娴位置编码，但以旋转矩阵的形式表示。然而键矩阵和查询矩阵都通过与此旋转矩阵相乘来整合位置信息。
{% mathjax '{"conversion":{"em":14}}' %}
{\mathbf{q}_i}^\top \mathbf{k}_j = (R^d_{\Theta,i}\mathbf{W}^q \mathbf{x}_i)^\top (R^d_{\Theta,j} \mathbf{W}^k \mathbf{x}_j) = {\mathbf{x}_i}^\top \mathbf{W}^q R^d_{\Theta, j - i} \mathbf{W}^k \mathbf{x}_j \; \text{where }R^d_{\Theta, j-i} = (R^d_{\Theta, i})^\top R^d_{\Theta, j}
{% endmathjax %}
{% asset_img t_4.png "旋转位置嵌入实现方式" %}
#### Transformer改进

`Transformer`模型在推理时的输入序列长度上限取决于训练的上下文长度。单纯增加上下文长度会导致时间和空间的大量消耗({% mathjax %}\mathcal{O}(L^2d){% endmathjax %})，并且由于硬件所限而不能支持。
##### 上下文内存

原始`Transformer`的注意力持续时间是固定且有限的。该模型在每个步骤更新只能关注同一段中的元素，并且没有信息可以跨越固定长度分段移动。这种上下文分段会导致以下几个问题：
- 该模型无法捕捉比较长的依赖关系。
- 在没有上下文或者上下文很稀少的情况下，很难预测每个片段中的前几个`token`。
- 评估的代价是昂贵的。每当片段向右移动一位时，新的片段都会从头开始重新处理，尽管有很多重叠的`token`。

`Transformer-XL`（`Dai`等人，`2019`年；`“XL”`表示“超长”）修改了架构，使用附加内存复用了段之间的隐藏状态。通过不断使用先前段的隐藏状态，将段之间的循环连接引入到模型中。
{% asset_img t_5.png "Transformer与Transformer-XL 的训练短语在段长度为4的比较" %}

让我们标记隐藏状态{% mathjax %}(\tau + 1){% endmathjax %}，第{% mathjax %}n{% endmathjax %}层模型中的第{% mathjax %}\mathbf{h}_{\tau + 1}^{(n)}\in \mathbb{R}^{L\times d}{% endmathjax %}段，除了最后一层隐藏状态为同一段{% mathjax %}\mathbf{h}_{\tau + 1}^{(n-1)}{% endmathjax %}，它还取决于前一段的同一层的隐藏状态{% mathjax %}\mathbf{h}_{\tau}^{(n)}{% endmathjax %}通过整合先前隐藏状态的信息，该模型可以将注意力跨度延长到过去的更长时间，涵盖了多个片段。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\color{red}{ \tilde{\mathbf{h}}_{\tau + 1}^{n - 1} } & = [\text{stop-gradient}(\mathbf{h}_{\tau}^{(n-1)})\circ \mathbf{h}_{\tau + 1}^{(n-1)}] \\
\mathbf{Q}_{\tau + 1}^{(n)} & = \mathbf{h}_{\tau + 1}^{(n-1)}\mathbf{W}^q \\
\mathbf{K}_{\tau + 1}^{(n)} & = \color{red}{ \tilde{\mathbf{h}}_{\tau + 1}^{n - 1} \mathbf{W}^k} \\
\mathbf{V}_{\tau + 1}^{(n)} & = \color{red}{ \tilde{\mathbf{h}}_{\tau + 1}^{n - 1} \mathbf{W}^v} \\
\mathbf{h}_{\tau + 1}^{(n)} & = \text{transformer-layer}(\mathbf{Q}_{\tau + 1}^{(n)},\mathbf{K}_{\tau + 1}^{(n)},\mathbf{V}_{\tau + 1}^{(n)})
\end{align}
{% endmathjax %}
请注意，键和值都依赖于扩展的隐藏状态，而查询仅使用当前步骤的隐藏状态。连接操作{% mathjax %}[\cdot\circ\cdot]{% endmathjax %}沿着序列长度维度。`Transformer-XL`需要使用相对位置编码，因为如果我们对绝对位置进行编码，则前一个段和当前段将被分配相同的编码，这是我们不希望看到的。`Compressive Transformer`（`Rae`等人，`2019`年）通过压缩过去的记忆来扩展`Transformer-XL`，以支持更长的序列。它明确添加了大小为{% mathjax %}m_m{% endmathjax %}每层存储该层的过去激活，以保留长上下文。当一些过去的激活变得足够旧时，它们会被压缩并保存在一个额外的压缩内存中，每层大小为{% mathjax %}m_{cm}{% endmathjax %}。
{% asset_img t_6.png "压缩Transformer维护两种类型的记忆槽，即记忆和压缩记忆，以支持长上下文" %}

内存和压缩内存都是`FIFO`队列。模型上下文长度为{% mathjax %}L{% endmathjax %}，压缩函数定义为：{% mathjax %}f_c:\mathbb{R}^{L\times d}\rightarrow \mathbb{[\frac{L}{c}]\times d}{% endmathjax %}、{% mathjax %}L{% endmathjax %}的映射激活压缩内存{% mathjax %}[\frac{L}{c}]{% endmathjax %}。压缩函数有多种选择：
- 内核的池化最大或平均和步幅大小{% mathjax %}c{% endmathjax %}。
- 具有内核和步幅大小的一维卷积{% mathjax %}c{% endmathjax %}（需要了解额外的参数）。
- 扩张卷积（需要学习其他参数）。
- 经常使用的内存。

`Compressive Transformer`还有两个额外的训练损失：
- **自动编码损失**（无损压缩目标）衡量我们压缩记忆-重建原始记忆的能力。
- **注意力重建损失**（有损目标）重建基于内容的注意力对记忆与压缩记忆的注意力，并最小化差异：

{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}_{ac} = \lVert \mathbf{\text{old_mem}}^{(i)} - g(\mathbf{\text{new_cm}}^{(i)})\rVert_2
{% endmathjax %}
翻转压缩函数{% mathjax %}f{% endmathjax %}为{% mathjax %}g:\mathbb{R}^{[\frac{L}{c}]\times d}\rightarrow \mathbb{R}^{L\times d}{% endmathjax %}
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}_{ar} = \lVert\text{attn}(\mathbf{h}^{(i)},\mathbf{\text{old_mem}}^{(i)}) - \text{attn}(\mathbf{h}^{(i)}, \mathbf{\text{new_cm}}^{(i)})\rVert_2
{% endmathjax %}
`Transformer-XL`的内存大小为{% mathjax %}m{% endmathjax %}，最大时间范围是{% mathjax %}m\times N{% endmathjax %}，模型的层数为{% mathjax %}N{% endmathjax %}。注意力的时间耗时为{% mathjax %}\mathcal{O}(L^2 + Lm){% endmathjax %}，压缩`Transformer`的时间为{% mathjax %}(m_m + c\cdot m_{cm})\times N{% endmathjax %}，且注意力的时间耗时为{% mathjax %}\mathcal{O}(L^2 + L(m_m + m_{cm})){% endmathjax %}。注意力权重权重存储在三个位置：压缩内存 → 内存 → 因果掩蔽序列。在实验中，他们观察到注意力权重从存储在常规内存中的最旧激活增加到存储在压缩内存中的激活，这意味着网络正在学习保存重要信息。
{% asset_img t_7.png "注意力权重与记忆位置的关系，以一个标准差作为误差线，从最旧（左）到最新（右）" %}

##### 外部存储器

对于`kNN-LM`（`Khandelwal`等人，`2020`年）通过单独的{% mathjax %}k{% endmathjax %}通过线性插值两个模型预测的下一个`token`概率，可以得到`kNN`模型。`kNN`模型建立在外部键值存储之上，该存储可以存储任何大型预训练数据集或`OOD`新数据集。此数据存储经过预处理以保存大量数据（上下文的`LM`嵌入表示、下一个`token`），并且最近邻检索发生在`LM`嵌入空间中。由于数据存储可能非常庞大，我们需要依靠库进行快速密集向量搜索，例如`FAISS`或`ScaNN`。索引过程仅发生一次，并且在推理时很容易实现并行性。在推理时，下一个`token`的概率是两个预测​​的加权和：
{% mathjax '{"conversion":{"em":14}}' %}
p(y|\mathbf{x})= \lambda p\text{kNN}(y|\mathbf{x}) + (1-\lambda)p\text{LM}(y|\mathbf{x})\;\;p\text{kNN}(y|\mathbf{x})\propto \sum_{(k_i,w_i)\in \mathcal{N}} \mathbb{1}[y = w_i]\exp(-d(k_i,f(x)))
{% endmathjax %}
{% mathjax %}\mathcal{N}{% endmathjax %}包含一组以最近邻数据{% mathjax %}k{% endmathjax %}的神经网络；{% mathjax %}d(\cdot,\cdot){% endmathjax %}是一个距离函数，例如`L2`距离。`SPALM`（自适应参数语言模型；`Yogatama`等人，`2021`年）结合了`Transformer-XL`风格的记忆，用于从外部上下文中获取隐藏状态作为短期记忆，`kNN-LM`风格的键值存储作为长记忆。
{% asset_img t_8.png "SPALM如何将过去隐藏状态的上下文记忆（短期记忆）与外部键值数据存储（长期记忆）相结合以支持更长的上下文" %}

`SPALM`运行`kNN`搜索获取`k`具有相关上下文的`token`。对于每个`token`，我们可以获得预训练`LM`提供的相同嵌入表示。表示为{% mathjax %}\{\mathbf{y}_i\}_{i=1}^k{% endmathjax %}。门控机制首先使用一个简单的注意力层聚合检索到的`token`嵌入{% mathjax %}\mathbf{h}_t^R{% endmathjax %}（token 的隐藏状态在层{% mathjax %}R{% endmathjax %}）作为查询，然后学习门控参数{% mathjax %}\mathbf{g}_t{% endmathjax %}平衡本地信息{% mathjax %}\mathbf{h}_t^R{% endmathjax %}和长期信息{% mathjax %}\mathbf{m}_t{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{m}_t & = \sum_{i=1}^k \frac{\exp(\mathbf{y}_i^\top \mathbf{h}_t^R)}{\sum_{j=1}^k \exp(\mathbf{y}_j^\top \mathbf{h}_t^R)}\cdot \mathbf{y}_i \\
\mathbf{g}_t & = \sigma (\mathbf{w}_g^\top \mathbf{h}_t^R) \\
\mathbf{z}_t & = (1 - \mathbf{g}_t) \odot \mathbf{m}_t + \mathbf{g}_t\odot \mathbf{h}_t^R \\
p(x_{t+1}| \mathbf{x}_{\leq t}) & = \text{softmax}(\mathbf{z}_t;\mathbf{W})
\end{align}
{% endmathjax %}
在这里{% mathjax %}\mathbf{w}_g{% endmathjax %}是需要学习的参数向量；{% mathjax %}\sigma(\cdot){% endmathjax %}是`S`形的；{% mathjax %}\mathbf{W}{% endmathjax %}是输入和输出`token`之间共享的词嵌入矩阵。不同于`kNN-LM`，没有发现最近距离对检索到的`token`的聚合有帮助。在训练期间，长期记忆中的关键表示保持不变，由预训练的`LM`产生，但值编码器（又称词嵌入矩阵）会进行更新。`Memorizing Transformer`（`Wu`等人，`2022`年）添加了一个`kNN`增强注意力层位于仅解码器的`Transformer`的顶部堆栈附近。这个特殊的层维护着过去键值对的`Transformer-XL`样式`FIFO`缓存。局部注意力和`kNN`机制。`kNN`查找返回顶部`k`（键，值）对用于输入序列中的每个查询，然后通过自注意力堆栈对其进行处理，以计算检索到的值的加权平均。两种类型的注意力与可学习的门控参数相结合。为了防止值幅度出现较大的分布偏移，缓存中的键和值都经过了规范化。`Memorizing Transformer`在实验中发现了以下现象：
- 一些实验观察，使用较小内存训练模型，然后使用较大内存进行微调比从头开始使用较大内存进行训练效果更好。
- 较小的`Memorizing Transformer`内存中只有`8k`的`token`，其困惑度可以与`vanilla Transformer`相媲美，且可训练参数相比高`5`倍。
- 增加外部存储器的大小可以获得一致的增益，最高可达`262K`。
- 非记忆`Tronsformer`在使用内存的情况下可以进行微调。

{% asset_img t_9.png "使用键值记忆对vanilla Transformer进行微调可实现与从头开始训练记忆Transformer达到类似的性能" %}

##### 距离增强注意力评分

`Distance Aware Transformer`（`DA-Transformer`；`Wu`等人，`2021`年）和具有线性偏差的注意力机制（`ALiBi；Press`等人，`2022`年）类似 — 为了鼓励模型在比模型训练的更长的上下文中进行推断，我们可以根据键`token`和查询`token`之间的距离，将位置信息明确地附加到每对注意力分数上。请注意，`vanilla Transformer`中默认的位置编码只会向输入序列添加位置信息，而后来改进的编码机制会改变每一层的注意力分数，例如旋转位置嵌入，它们的形式与距离增强注意力分数非常相似。`DA-Transformer`（`Wu`等人，`2021`年）将每一层的注意力得分乘以可学习的偏差，该偏差由键和查询之间的距离函数表示。不同的注意力头使用不同的参数来区分对短期和长期的不同偏好。给定两个位置，{% mathjax %}i,j{% endmathjax %}，`DA-Transformer`使用以下加权函数来改变自注意力分数：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{R}^{(i)} & = \alpha_i\mathbf{R}\;\;\text{where }R_{ij} = |i - j| \\
f(\mathbf{R}^{(i)};\beta_i) & = \frac{1 + \exp(\beta_i)}{1+ \exp(\beta_i - \mathbf{R}^{(i)})} \\
\text{attn}(\mathbf{Q}^{(i)},\mathbf{K}^{(i)},\mathbf{V}^{(i)}) = \text{row-softmax}(\frac{\text{ReLU}(\mathbf{Q}^{(i)}\mathbf{K}^{(i)\top})f(\mathbf{R}^{(i)})}{\sqrt{d}})\mathbf{V}^{(i)}
\end{align}
{% endmathjax %}
在这里{% mathjax %}\alpha_i{% endmathjax %}是一个可以学习的参数，用于对每个头部的相对距离进行不同的加权，其中头部用上标{% mathjax %}^{(i)}{% endmathjax %}表示；{% mathjax %}\beta_i{% endmathjax %}也是一个可以学习的参数，用于控制距离的上限和上升斜率第{% mathjax %}i{% endmathjax %}个注意力头。权重函数{% mathjax %}f(\cdot){% endmathjax %}定义如下：1.{% mathjax %}f(0) = 1{% endmathjax %}；2.{% mathjax %}f(\mathbf{R}^{(i)}) = 0\;\mathbf{R}^{(i)}\rightarrow -\infty{% endmathjax %}；3.{% mathjax %}f(\mathbf{R}^{(i)})\;\mathbf{R}^{(i)}\rightarrow +\infty{% endmathjax %}；4.规模可调；5.函数单调；{% mathjax %}f(\mathbf{R}^{(i)}){% endmathjax %}的时间复杂度为{% mathjax %}\mathcal{O}(2h){% endmathjax %}相对于自注意力的复杂度{% mathjax %}\mathcal{O}(L^2){% endmathjax %}来说小很多、内存消耗也很小。`ALiBi`（`Press`等人，`2022`年）不使用乘数，而是在查询关键字注意力得分上添加了一个常数偏差项，该偏差项与成对距离成比例。偏差引入了强烈的近因偏好，并惩罚距离太远的关键字。惩罚在不同的头中以不同的速率增加。
{% mathjax '{"conversion":{"em":14}}' %}
\text{softmax}(\mathbf{q}_i\mathbf{K}^\top + \alpha_i\cdot[0,-1,-2,\ldots, -(i-1)])
{% endmathjax %}
在这里{% mathjax %}\alpha_i{% endmathjax %}是头部特定的加权标量。与`DA-transformer`不同，{% mathjax %}\alpha_i{% endmathjax %}不是固定的序列，例如，对于8个头，{% mathjax %}\alpha_i = \frac{1}{2},\frac{1}{2^2},\ldots,\frac{1}{2^8}{% endmathjax %}。总体思路与相对位置编码所要解决的问题非常相似。
{% asset_img t_10.png "ALiBi如何通过位置偏差项提高注意力分数" %}

`ALiBi`在训练期间对上下文长度为`1024`的`1.3B`模型进行了训练，并在推理时推断为`2046`。
{% asset_img t_11.png "正弦位置编码、旋转位置编码、T5和ALiBi中的简化相对位置编码。所有模型都以较小的上下文长度进行训练，但推理运行的上下文长度要长得多" %}
##### 参数复用

