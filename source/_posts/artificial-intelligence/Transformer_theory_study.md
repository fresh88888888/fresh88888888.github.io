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
\text{attn}(\mathbf{Q}^{(i)},\mathbf{K}^{(i)},\mathbf{V}^{(i)}) & = \text{row-softmax}(\frac{\text{ReLU}(\mathbf{Q}^{(i)}\mathbf{K}^{(i)\top})f(\mathbf{R}^{(i)})}{\sqrt{d}})\mathbf{V}^{(i)}
\end{align}
{% endmathjax %}
在这里{% mathjax %}\alpha_i{% endmathjax %}是一个可以学习的参数，用于对每个头部的相对距离进行不同的加权，其中头部用上标{% mathjax %}^{(i)}{% endmathjax %}表示；{% mathjax %}\beta_i{% endmathjax %}也是一个可以学习的参数，用于控制距离的上限和上升斜率第{% mathjax %}i{% endmathjax %}个注意力头。权重函数{% mathjax %}f(\cdot){% endmathjax %}定义如下：1.{% mathjax %}f(0) = 1{% endmathjax %}；2.{% mathjax %}f(\mathbf{R}^{(i)}) = 0\;\mathbf{R}^{(i)}\rightarrow -\infty{% endmathjax %}；3.{% mathjax %}f(\mathbf{R}^{(i)})\;\mathbf{R}^{(i)}\rightarrow +\infty{% endmathjax %}；4.规模可调；5.函数单调；{% mathjax %}f(\mathbf{R}^{(i)}){% endmathjax %}的时间复杂度为{% mathjax %}\mathcal{O}(2h){% endmathjax %}相对于自注意力的复杂度{% mathjax %}\mathcal{O}(L^2){% endmathjax %}来说小很多、内存消耗也很小。`ALiBi`（`Press`等人，`2022`年）不使用乘数，而是在查询关键字注意力得分上添加了一个常数偏差项，该偏差项与成对距离成比例。偏差引入了强烈的近因偏好，并惩罚距离太远的关键字。惩罚在不同的头中以不同的速率增加。
{% mathjax '{"conversion":{"em":14}}' %}
\text{softmax}(\mathbf{q}_i\mathbf{K}^\top + \alpha_i\cdot[0,-1,-2,\ldots, -(i-1)])
{% endmathjax %}
在这里{% mathjax %}\alpha_i{% endmathjax %}是头部特定的加权标量。与`DA-transformer`不同，{% mathjax %}\alpha_i{% endmathjax %}不是固定的序列，例如，对于8个头，{% mathjax %}\alpha_i = \frac{1}{2},\frac{1}{2^2},\ldots,\frac{1}{2^8}{% endmathjax %}。总体思路与相对位置编码所要解决的问题非常相似。
{% asset_img t_10.png "ALiBi如何通过位置偏差项提高注意力分数" %}

`ALiBi`在训练期间对上下文长度为`1024`的`1.3B`模型进行了训练，并在推理时推测为`2046`。
{% asset_img t_11.png "正弦位置编码、旋转位置编码、T5和ALiBi中的简化相对位置编码。所有模型都以较小的上下文长度进行训练，但推理运行的上下文长度要长得多" %}
##### 参数复用

`Universal Transformer`（`Dehghani`等人，`2019`年）将`Transformer`中的自注意力机制与`RNN`中的循环机制相结合，旨在从`Transformer`捕获的长期依赖关系和`RNN`的学习归纳偏差中获益。`Universal Transformer`不是固定数量的层，而是使用自适应计算动态的调整步数。如果我们固定步数，`Universal Transformer`相当于一个跨层共享参数的多层`Transformer`。从高层次来看，通用`Transformer`可以看作是一个循环函数，用于学习每个`token`的隐藏状态表示。循环函数在`token`位置之间并行演化，位置之间的信息通过自注意力机制共享。
{% asset_img t_12.png "Universal Transformer如何并行地为每个位置反复细化一组隐藏状态表示" %}

给定长度的输入序列{% mathjax %}L{% endmathjax %}，Universal Transformer迭代更新表示为{% mathjax %}\mathbf{h}^t\in \mathbb{R}^{L\times d}{% endmathjax %}。在第{% mathjax %}0{% endmathjax %}步，{% mathjax %}\mathbf{h}^0{% endmathjax %}初始化与输入嵌入矩阵相同，所有位置在多头自注意力机制中并行处理，然后经过循环转换函数。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{A}^t & = \text{LayerNorm}(\mathbf{h}^{t-1}+\text{MultiHeadAttention}(\mathbf{h}^{t-1} + \mathbf{P}^t)) \\
\mathbf{h}^t & = \text{LayerNorm}(\mathbf{A}^{t-1} + \text{Transition}(\mathbf{A}^t))
\end{align}
{% endmathjax %}
在这里{% mathjax %}\text{Transition}(\cdot){% endmathjax %}是一个可分离卷积或全连接神经网络`affine transformation` + `ReLU`由两种位置(每一行的{% mathjax %}\mathbf{A}^t{% endmathjax %}都是独立的)组成。位置编码{% mathjax %}\mathbf{P}^t{% endmathjax %}使用正弦位置编码，但带有额外的时间维度：
{% mathjax '{"conversion":{"em":14}}' %}
\text{PE}(i, t, \delta) = 
\begin{cases}
\sin(\frac{i}{10000^{2\delta'/d}}) \oplus \sin(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' \\
\cos(\frac{i}{10000^{2\delta'/d}}) \oplus \cos(\frac{t}{10000^{2\delta'/d}}) & \text{if } \delta = 2\delta' + 1 \\
\end{cases}
{% endmathjax %}
{% asset_img t_13.png "通用Transformer的简化图。编码器和解码器共享相同的基本循环结构" %}

在自适应版本的`Universal Transformer`中，循环步骤的数量{% mathjax %}T{% endmathjax %}由`ACT`动态确定。每个位置都配备了动态`ACT`停止机制。一旦每个`token`循环块停止，它就会停止进行更多循环更新，而只是将当前值复制到下一步，直到所有块都停止或模型达到最大步数限制。

#### 自适应建模

自适应建模是指能够根据不同的输入调整计算量的机制。例如，有些`token`可能只需要局部信息，因此需要较短的注意力跨度；或者有些`token`相对比较容易预测，不需要经过整个注意力栈进行处理。
##### 自适应注意力

`Transformer`的一个关键优势是能够捕获长期依赖关系。根据上下文，模型可能更喜欢在某些时候关注更远的事物，或者一个注意力头可能具有与不同的注意力模式。如果注意力跨度可以灵活调整其长度，并且只在需要时关注更远的事物，那么这将有助于降低计算和内存成本，以支持模型中更长的最大上下文大小。这就是自适应注意力跨度的动机。`Sukhbaatar`等人（`2019`年）提出了一种寻求最佳注意力跨度的自注意力机制。假设不同的注意力头可能会在同一上下文窗口内分配不同的分数，因此最佳跨度将针对每个注意力头单独进行训练。
{% asset_img t_14.png "同一模型中的两个注意力头A和B在同一个上下文窗口内分配不同的注意力。注意力头A更多地关注最近的标记，而注意力头B则均匀地回顾过去" %}

鉴于第{% mathjax %}i{% endmathjax %}个`token`，我们需要计算这个`token`和其注意力范围内其他键之间的注意力权重{% mathjax %}s{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
e_{ij} & = \mathbf{q}_i {\mathbf{k}_j}^\top \\
a_{ij} & = \text{softmax}(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{r=i-s}^{i-1}\exp(e_{ir})} \\
\mathbf{y}_i & = \sum_{r=i-s}^{i-1} a_{ir}\mathbf{v}_r = \sum_{r=i-s}^{i-1} a_{ir}\mathbf{x}_r \mathbf{W}^v
\end{align}
{% endmathjax %}
{% mathjax %}m_z{% endmathjax %}被添加到控制中以获得有效的可调注意力跨度，将查询和键之间的距离映射到{% mathjax %}[0,1]{% endmathjax %}。{% mathjax %}m_z{% endmathjax %}参数化为{% mathjax %}z\in [0,s]{% endmathjax %}并且{% mathjax %}z{% endmathjax %}需要学习的是：
{% mathjax '{"conversion":{"em":14}}' %}
m_z(x) = \text{clip}(\frac{1}{R}(R + z - x),0,1)
{% endmathjax %}
在这里{% mathjax %}R{% endmathjax %}是一个超参数，它定义了{% mathjax %}m_z{% endmathjax %}：
{% asset_img t_15.png "自适应注意力跨度中使用的软遮罩函数" %}

软遮罩函数应用于注意力权重中的`softmax`元素：
{% mathjax '{"conversion":{"em":14}}' %}
a_{ij} = \frac{m_z(i-j)\exp(s_{ij})}{\sum_{r=i-s}^{i-1} m_z(i-r)\exp(s_{ir})}
{% endmathjax %}
在上面的等式中，{% mathjax %}z{% endmathjax %}是可微的，因此它与模型的其它部分联合训练。参数{% mathjax %}z^{(i)},i=1,\ldots,h{% endmathjax %}是每个头分别学习的，此外，损失函数有一个额外的惩罚({% mathjax %}\sum_{i=1}^h z^{(i)}{% endmathjax %})。使用自适应计算时间，该方法可以得到进一步增强，具有灵活的注意力跨度长度，可动态适应当前输入。注意力跨度参数{% mathjax %}z_t{% endmathjax %}是集中在时间{% mathjax %}t{% endmathjax %}上的{% mathjax %}S{% endmathjax %}函数，{% mathjax %}z_t = S\sigma(\mathbf{v}\cdot \mathbf{x}_t + b){% endmathjax %}，其中向量{% mathjax %}\mathbf{v}{% endmathjax %}和偏差标量{% mathjax %}b{% endmathjax %}跟其他参数共同学习。在具有自适应注意力跨度的`Transformer`实验中，`Sukhbaatar`等人（`2019`）发现一个普遍趋势，即较低层不需要非常长的注意力跨度，而较高层中的少数注意力头可能使用较长的跨度。自适应注意力跨度还有助于大大减少`FLOPS`数值，尤其是在具有许多注意力层和较大上下文长度的大型模型中。
##### 深度自适应Transformer

在推理时，我们自然会认为某些`token`更容易预测，因此不需要像其他`token`那样多的计算量。因此，我们可能只通过有限数量的层来处理其预测，以在速度和性能之间取得良好的平衡。深度自适应`Transformer`（`Elabyad`等人，`2020`年）和置信自适应语言模型（`CALM`；`Schuster`等人，`2022`年）都受到这一想法的启发，并学习预测不同输入`token`所需的最佳层数。深度自适应`Transformer`将输出分类器附加到每一层，以根据该层的激活产生退出预测。分类器权重矩阵可以每层不同，也可以跨层共享。在训练期间，模型会采样不同的退出序列，以便使用不同层的隐藏状态优化模型。学习目标结合了在不同层预测的似然概率，{% mathjax %}n=1,\ldots,N{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
LL_t^n = \log p(y_t|\mathbf{h}^n_{t-1})\;\;LL^n = \sum_{t=1}^{|y|} LL_t^n
{% endmathjax %}
自适应深度分类器输出参数分布为{% mathjax %}q_t{% endmathjax %}，使用交叉熵损失对`oracle`分布进行训练{% mathjax %}q_t^*{% endmathjax %}。在这里主要探讨了如何学习这种分类器的三种配置{% mathjax %}q_t{% endmathjax %}。
{% asset_img t_16.png "三种自适应深度分类器的图示" %}

- **序列特定的深度分类器**：同一序列的所有标记共享相同的出口。它取决于序列的编码器表示的平均值。给定一个输入序列{% mathjax %}\mathbf{x}{% endmathjax %}，长度为{% mathjax %}L{% endmathjax %}，分类器采用{% mathjax %}\bar{\mathbf{x}} = \frac{1}{L}\sum_{t=1}^L \mathbf{x}_t{% endmathjax %}作为输入和输出多项分布，{% mathjax %}N{% endmathjax %}为层数。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
q(n|\mathbf{x}) & = \text{softmax}(\mathbf{W}_n\bar{x} + b_n)\in \mathbb{R}^N \\
q^*_{\text{lik}}(\mathbf{x},\mathbf{y}) & = \delta(\arg\max_n LL^n - \lambda n) \\
\text{or }q^*_{\text{corr}}(\mathbf{x},\mathbf{y}) & = \delta(\arg\max_n C^n - \lambda n)\text{ where }C^n = |\{t|y_t = \arg\max_y p(y|\mathbf{h}^n_{t-1})\}|
\end{align}
{% endmathjax %}
在这里{% mathjax %}\delta{% endmathjax %}是狄拉克德尔塔（单位脉冲）函数，{% mathjax %}-\lambda n{% endmathjax %}是一个正则化项，用于鼓励较低层退出。
- **特定于token的深度分类器（多项式）**：每个`token`用不同的出口块解码，根据第一个解码器隐藏状态进行预测{% mathjax %}\mathbf{h}^1_t{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
q_t(n|\mathbf{x},\mathbf{y}_{<t}) = \text{softmax}(\mathbf{W}_n\mathbf{h}_t^1 + b_n)
{% endmathjax %}
- **特定于token的深度分类器（类似几何）**：每个`token`每层都有一个二进制出口预测分布{% mathjax %}\mathcal{X}_t^n{% endmathjax %}。`RBF`内核{% mathjax %}\mathcal{k}(t,t') = \exp(\frac{|t-t'|^2}{\sigma}){% endmathjax %}用于平滑预测，以纳入当前决策对未来时间步长的影响。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathcal{X}_t^n & = \text{sigmoid}(\mathbf{w}_n^\top \mathbf{h}_t^n + b_n)\;\forall n\in [1,ldots,N-1] \\
q_t(n|\mathbf{x},\mathbf{y}_{<t}) & = 
\begin{cases}
\mathcal{X}_t^n\prod_{n'<n} (1-\mathcal{X}_t^{n'}) & \text{ if }n<N \\
\prod_{n' < N} (1 - \mathcal{X}_t^{n'}) & \text{ otherwise } \\
\end{cases} \\
q^*_{lik}(\mathbf{x}, \mathbf{y}) & = \delta(\arg\max_n\tilde{LL}_t^n - \lambda n) \text{ where }\tilde{LL}_t^n = \sum_{t'=1}^{|y|} \mathcal{k}(t,t')LL_{t'}^n \\
\text{or }q^*_{\text{corr}}(\mathbf{x},\mathbf{y}) & = \delta(\arg\max_n\tilde{C}_t^n - \lambda n) \text{ where }C_t^n = \mathbb{1}[y_t = \arg\max_y p(y|\mathbf{h}_{t-1}^n)],\;\tilde{C}_t^n= \sum_{t' =1}^{|y|} \mathcal{k}(t,t')C_{t'}^n
\end{align}
{% endmathjax %}

在推理时，需要校准做出退出决策的置信度阈值。深度自适应`Transformer`通过网格搜索在验证集上找到这样的阈值。`CALM`（`Schuste`r等人，`2022`年）使用测试(`LTT`)框架（`Angelopoulos`等人，`2021`年）来识别有效阈值的子集，并选择最小值作为推理的阈值。除了训练每层退出分类器外，`CALM`还探索了其他自适应深度预测方法，包括`softmax`激活（即前两个`softmax`输出之间的差异）和隐藏状态饱和（即{% mathjax %}\cos(\mathbf{h}_t^n,\mathbf{h}_t^{n+1}){% endmathjax %}作为退出决策的置信度分数。他们发现`softmax`激活可带来最佳的推理加速。

#### 高效的注意力

普通`Transformer`的计算和内存成本随序列长度呈二次方增长，因此很难应用于非常长的序列。`Transformer`架构的许多效率改进都与自注意力模块有关-使其更便宜、更小或运行速度更快。
##### 稀疏注意力模式

###### 固定局部上下文

降低自我注意力成本的一个简单方法是将每个标记的注意力范围限制在局部上下文中，这样自我注意力就会随着序列长度线性增长。该想法由`Image Transformer`（`Parmer`等人，`2018`年）提出，它将图像生成表述为使用编码器-解码器`Transformer`架构的序列建模：
- 编码器生成源图像的上下文、每个像素通道表示。
- 解码器自回归生成输出图像，每个时间步、每个像素一个通道。

我们将要生成的当前像素的表示`token`为查询{% mathjax %}\mathbf{q}{% endmathjax %}。其他位置的表示将计算{% mathjax %}\mathbf{q}{% endmathjax %}的键向量{% mathjax %}\mathbf{k}_1,\mathbf{k}_2,\ldots{% endmathjax %}，它们共同组成记忆矩阵{% mathjax %}\mathbf{M}{% endmathjax %}。`Image Transformer`引入了两种类型的局部，如下图所示。
{% asset_img t_17.png "Image Transformer 中视觉输入的1D和2D注意力跨度说明。黑线标记查询块，青色勾勒出像素q的实际注意力跨度" %}

- 1D局部注意力：输入图像按光栅扫描顺序展平，即从左到右、从上到下。然后将线性的图像划分为不重叠的查询块。上下文窗口由与查询块相同的像素组成{% mathjax %}\mathbf{q}{% endmathjax %}以及在此查询块之前生成的固定数量的附加像素。
- 2D局部注意力：图像被划分为多个不重叠的矩形查询块。查询像素可以关注同一记忆块中的所有其他像素。为了确保左上角的像素也可以具有有效的上下文窗口，记忆块分别向上、向左和向右扩展固定量。
##### 步进上下文

`Sparse Transformer`（`Child`等人，`2019`年）通过稀疏矩阵分解引入了分解式自注意力，从而可以在高达`16,384`的序列长度上训练具有数百层的密集注意力网络，否则这在现代硬件上是不可行的。给定一组注意力连接模式{% mathjax %}\mathcal{S} = \{S_1,\ldots,S_n\}{% endmathjax %}，其中每个{% mathjax %}S_i{% endmathjax %}记录第{% mathjax %}i{% endmathjax %}个关键位置查询向量。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\text{Attend}(\mathbf{X},\mathcal{S}) & = (a(\mathbf{x}_i,S_i))_{i\in \{1,\ldots,L\}} \\
\text{ where }a(\mathbf{x}_i,S_i) & = \text{softmax}(\frac{(\mathbf{x}_i\mathbf{W}^q)(\mathbf{x}_j\mathbf{W}^k)_{j\in S_i}^\top}{\sqrt{d_k}})(\mathbf{x}_j\mathbf{W}^v)_{j\in S_i}
\end{align}
{% endmathjax %}
注意，尽管{% mathjax %}S_i{% endmathjax %}不固定，a(\mathbf{x}_i,S_i)总是大小为{% mathjax %}d_v{% endmathjax %}，因此{% mathjax %}\text{}Attend(\mathbf{X},\mathcal{S})\in \mathbb{R}^{L\times d}{% endmathjax %}。在自回归模型中，一个注意力跨度定义为{% mathjax %}S_i = \{j:j\leq i\}{% endmathjax %}，因为它允许每个`token`关注过去的所有位置。在分解自注意力机制中，集合{% mathjax %}S_i{% endmathjax %}分解成一个依赖关系树，这样对于每一对{% mathjax %}(i,j){% endmathjax %} j\leq i，有一条路径连接{% mathjax %}i{% endmathjax %}回到{% mathjax %}j{% endmathjax %}并且无论是直接还是间接{% mathjax %}i{% endmathjax %}是可以加入到{% mathjax %}j{% endmathjax %}。`Sparse Transformer`提出了两种类型的分形注意力机制。下图以二维图像输入为例，更容易理解这些概念。
{% asset_img t_18.png "顶行展示了(a)Transformer、(b)具有步进注意力的 Sparse Transformer和(c)具有固定注意力的 Sparse Transformer 中的注意力连接模式。底行包含相应的自注意力连接矩阵。请注意，顶行和底行的比例不同。" %}

- **带步长的步进注意力**{% mathjax %}\ell \sim \sqrt{n}{% endmathjax %}这适用于图像数据，因为结构与步幅对齐。在图像上面，每个像素都会关注所有先前的{% mathjax %}\ell{% endmathjax %}按光栅扫描顺序关注像素（自然覆盖图像的整个宽度），然后这些像素关注同一列中的其他像素（由另一个注意连接子集定义）。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
 A_i^{(1)} &= \{ t, t+1, \dots, i\} \text{, where } t = \max(0, i - \ell) \\
 A_i^{(2)} &= \{j: (i-j) \mod \ell = 0\}
\end{aligned}
{% endmathjax %}
- 固定注意力。一小部分`token`汇总了之前的位置，并将该信息传播到所有未来的位置。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
 A_i^{(1)} &= \{j: \lfloor \frac{j}{\ell} \rfloor = \lfloor \frac{i}{\ell} \rfloor \} \\
 A_i^{(2)} &= \{j: j \mod \ell \in \{\ell-c, \dots, \ell-1\} \}
 \end{aligned}
{% endmathjax %}

在这里{% mathjax %}c{% endmathjax %}是超参数。如果{% mathjax %}c = 1{% endmathjax %}它限制了表示，而许多表示依赖于少数位置。本文选择了{% mathjax %}c\in \{8,16,32\}{% endmathjax %}并且{% mathjax %}\ell\in \{128,256\}{% endmathjax %}。在`Transformer`架构中，有三种方法可以使用稀疏分解注意力模式：
- 每个残差块是一种注意力类型，然后将它们交错，{% mathjax %}\text{attn}(\mathbf{X}) = \text{Attend}(\mathbf{X}, A^{(n \mod p)}) \mathbf{W}^o{% endmathjax %}，在这里{% mathjax %}n{% endmathjax %}是当前残差块的索引。
- 设置一个单独的主管，负责所有分解后的主管负责的位置，{% mathjax %}\text{attn}(\mathbf{X}) = \text{Attend}(\mathbf{X}, \cup_{m=1}^p A^{(m)}) \mathbf{W}^o{% endmathjax %}。
- 使用多头注意力机制，但与原始 Transformer 不同，每个头可能采用上面提出的模式`1`或`2`，该选项通常效果最佳。

`Sparse Transformer`还提出了一系列变化，以便将`Transformer`训练到数百层，包括梯度检查点、在后向传递期间重新计算注意力和`FF`层、混合精度训练、高效的块稀疏实现等。分块注意力(`Qiu et al`. `2019`) 引入了一个稀疏块矩阵，只允许每个`token`关注一小部分其他`token`。每个注意力矩阵的大小为{% mathjax %}L\times L{% endmathjax %}被划分成了{% mathjax %}n\times n{% endmathjax %}更小的块{% mathjax %}frac{L}{n}\times\frac{L}{n}{% endmathjax %}和一个稀疏块矩阵{% mathjax %}\mathbf{M}\in \{0,1\}^{L\times L}{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\text{attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) &= \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} \odot \mathbf{M}\Big)\mathbf{V} \\
(\mathbf{A} \odot \mathbf{M})_{ij} &= \begin{cases}
A_{ij} & \text{if }M_{ij} = 1 \\
-\infty & \text{if }M_{ij} = 0 \\
\end{cases} \\
\text{where } M_{ij} &= \begin{cases}
1 & \text{if }\pi\big(\lfloor\frac{(i-1)n}{L} + 1\rfloor\big) = \lfloor\frac{(j-1)n}{L} + 1\rfloor \\
0 & \text{otherwise}
\end{cases}
\end{aligned}
{% endmathjax %}
`Blockwise Attention`的实际实现将`QKV`存储为块矩阵，每个矩阵的大小为{% mathjax %}n\times n{% endmathjax %}
{% mathjax '{"conversion":{"em":14}}' %}
\text{Blockwise-attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \begin{bmatrix}
\text{softmax}\big(\frac{\hat{\mathbf{q}}_1\hat{\mathbf{k}}_{\pi(1)}^\top}{\sqrt{d}} \Big)\hat{\mathbf{v}}_{\pi(1)} \\
\vdots \\
\text{softmax}\big(\frac{\hat{\mathbf{q}}_n\hat{\mathbf{k}}_{\pi(n)}^\top}{\sqrt{d}} \odot \Big)\hat{\mathbf{v}}_{\pi(n)} \\
\end{bmatrix}
{% endmathjax %}
在这里{% mathjax %}\hat{\mathbf{q}}_i,\hat{\mathbf{k}}_i{% endmathjax %}和{% mathjax %}\hat{\mathbf{v}}_i{% endmathjax %}分别为`QKV`块矩阵中的行。每个{% mathjax %}\mathbf{q}_i\mathbf{k}_{\pi(i)}^\top, \forall i = 1, \dots, n{% endmathjax %}大小为{% mathjax %}\frac{N}{n}\times\frac{N}{n}{% endmathjax %}因此`Blockwise Attention`能够将注意力矩阵的记忆复杂度从{% mathjax %}\mathcal{O}(L^2){% endmathjax %}到{% mathjax %}\mathcal{O}(\frac{L}{n}\times\frac{L}{n} \times n) = \mathcal{O}(L^2/n){% endmathjax %}。

`ETC`（扩展`Transformer`构造；`Ainslie`等人，`2019`年）、`Longformer`（`Beltagy`等人，`2020`年）和`Big Bird`（`Zaheer`等人，`2020`年）模型在构建注意力矩阵时结合了局部和全局组合。所有这些模型都可以从现有的预训练模型中初始化。ETC的全局-局部注意力机制（`Ainslie`等人，`2019`年）接受两个输入，（1）长输入{% mathjax %}\mathbf{x}^l{% endmathjax %}大小{% mathjax %}n_l{% endmathjax %}这是常规输入序列，（2）全局输入{% mathjax %}\mathbf{x}^g{% endmathjax %}大小{% mathjax %}n_g{% endmathjax %}包含少量的辅助`token`，{% mathjax %}n_g\ll n_l{% endmathjax %}。因此，注意力根据这两个输入的方向性注意力被分为四个部分：`g2g、g2l、l2g`和`l2l`。由于`l2l`注意力部分可能非常大，因此它被限制在固定大小的注意力范围半径内{% mathjax %}w{% endmathjax %}（即局部注意力广度）并且`l2l`矩阵可以重塑为{% mathjax %}n_l \times (2w + 1){% endmathjax %}。ETC 利用四个二进制矩阵来处理结构化输入，{% mathjax %}\mathbf{M}^{g2g},\mathbf{M}^{g2l},\mathbf{M}^{l2g}{% endmathjax %}和{% mathjax %}\mathbf{M}^{l2l}{% endmathjax %}。例如，每个元素{% mathjax %}z_i^g\in \mathbb{R}^d{% endmathjax %}在注意力输出中{% mathjax %}z^g = (z_1^g,\ldots,z^g_{n_g}){% endmathjax %}对于`g2g`注意力片段的格式如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
a^{g2g}_{ij} = \frac{1}{\sqrt{d}} x^g_i \mathbf{W}^Q (x^g_j \mathbf{W}^K + P^K_{ij})^\top - (1- M^{g2g}_{ij})C \\
A^{g2g}_{ij} = \frac{\exp(a^{g2g}_{ij})}{\sum_{k=1}^{n_g} \exp(a^{g2g}_{ik})} \quad
z^g_i = \sum^{n_g}_{j=1} A^{g2g}_{ij} x^g_j \mathbf{W}^V
\end{aligned}
{% endmathjax %}
在这里{% mathjax %}P_{ij}^K{% endmathjax %}是相对位置编码的可学习向量，{% mathjax %}C{% endmathjax %}是一个非常大的常数（{% mathjax %}C=10000{% endmathjax %}在论文中）来抵消摘下口罩时的注意力权重。
{% asset_img t_19.png "ETC、Longformer 和 Big Bird 的注意力模式" %}

`ETC`的另一个更新是整合了`CPC`（对比预测编码）任务，使用NCE 损失进入预训练阶段，除了`MLM`任务之外：当一句话被掩盖时，它的表征应该和它周围上下文的表征相似。全局输入{% mathjax %}\mathbf{x}^g{% endmathjax %}
`ETC`的构造如下：假设长输入（例如句子）中有一些片段，每个片段都附加一个辅助标记以学习全局输入。相对位置编码用于用标记位置标记全局片段标记。发现在一个方向上的硬掩码（即，前后标记的标记不同）可以在某些数据集中带来性能提升。

`Longformer`中的注意力模式包含三个部分：
- 局部注意力：与`ETC`类似，局部注意力由固定大小的滑动窗口控制{% mathjax %}w{% endmathjax %}。
- 预选`token`的全局注意力：`Longformer`为一些预选`token`（例如`[CLS]token`）分配了全局注意力跨度，也就是说，关注输入序列中的其他`token`。
- 扩张注意力机制：固定大小的扩张滑动窗口{% mathjax %}r{% endmathjax %}以及扩张尺寸的间隙{% mathjax %}d{% endmathjax %}，类似于`Sparse Transformer`。

`Big Bird`与`Longformer`非常相似，既配备了局部注意力机制，又配备了一些具有全局注意力范围的预选`token`，但`Big Bird`用一种新机制取代了扩张注意力机制，即所有`token`都关注一组随机`token`。这种设计的动机是，注意力模式可以看作是有向图，而随机图具有信息能够在任意一对节点之间快速流动的特性。`Longformer`在较低层使用较小的窗口大小，在较高层使用较大的窗口大小。消融研究表明，这种设置比反向或固定大小的配置效果更好。较低层没有扩大的滑动窗口，无法更好地学习使用直接的局部上下文。`Longformer`还有一个分阶段的训练程序，其中最初使用小窗口大小训练模型以从局部上下文中学习，然后在后续的训练阶段增加窗口大小并降低学习率。
##### 基于上下文注意力

`Reformer`（`Kitaev`等人，`2020`年）提出的改进旨在解决`vanilla Transformer`中的以下痛点：
- 自注意力模块内的二次方时间和内存复杂度。
- {% mathjax %}N{% endmathjax %}层比单层模型大{% mathjax %}N-{% endmathjax %}倍，因为我们需要存储反向传播的激活。
- 中间的`FF`层通常相当大。

提出了两项​​主要改进：
- 用局部敏感哈希(`LSH`)注意力机制代替点积注意力机制，从而降低复杂度{% mathjax %}\mathcal{O}(L^2){% endmathjax %}到{% mathjax %}\mathcal{O}(L\log L){% endmathjax %}。
- 用可逆残差层替换标准残差块，这样可以在训练期间仅存储一次激活，而不是{% mathjax %}N{% endmathjax %}倍（即与层数成正比）。

在{% mathjax %}\mathbf{QK}^\top{% endmathjax %}作为注意力公式的一部分，我们只对最大的元素感兴趣，因为只有大元素在`softmax`之后才会做出很大贡献。对于每个查询{% mathjax %}\mathbf{q}_i\in \mathbf{Q}{% endmathjax %}，我们正在寻找{% mathjax %}\mathbf{K}{% endmathjax %}最靠近{% mathjax %}\mathbf{q}_i{% endmathjax %}为了在高维空间中快速找到最近邻居，`Reformer`将局部敏感哈希(`LSH`)纳入其注意力机制中。哈希方案{% mathjax %}x\longmapsto h(x){% endmathjax %}如果它保留了数据点之间的距离信息，那么它就是局部敏感的，这样距离近的向量会获得相似的哈希值，而距离远的向量会获得非常不同的哈希值。`Reformer`采用这样的哈希方案，给定一个固定的随机矩阵{% mathjax %} \mathbf{R}\in \mathbb{R}^{d\times b/2}{% endmathjax %}（{% mathjax %}b{% endmathjax %}是超参数），哈希函数是{% mathjax %}h(x) = \arg\max([xR;-xR]){% endmathjax %}。
{% asset_img t_20.png "LSH注意力机制由4个步骤组成：分桶、排序、分块和注意力计算 & 局部敏感哈希(LSH)注意力机制示意图" %}

在`LSH`注意力机制中，查询只能关注同一哈希桶中的位置，{% mathjax %}S_i = \{j:h(\mathbf{q}_i) = \mathbf{h}(\mathbf{k}_j)\}{% endmathjax %}其具体过程如下，如左上图所示：
- 完全注意力的注意力矩阵通常很稀疏。
- 使用`LSH`，我们可以根据哈希桶对要对齐的键和查询进行排序。
- 设置{% mathjax %}\mathbf{Q} = \mathbf{K}{% endmathjax %}（{% mathjax %}\mathbf{k}_j = \mathbf{q}_j/|\mathbf{q}_j{% endmathjax %})，这样在一个`bucket`中就有相等数量的键和查询，更容易进行批处理。有趣的是，这种`“shared-QK”`配置不会影响`Transformer`的性能。
- 应用批处理，其中{% mathjax %}m{% endmathjax %}连续的查询被分组在一起。

`Reformer`的另一项改进是使用**可逆残差层**（`Gomez`等人，`2017`年）。可逆残差网络的动机是设计一种架构，使得任何给定层的激活都可以从下一层的激活中恢复，仅使用模型参数即可。因此，我们可以通过在反向传播期间重新计算激活来节省内存，而不是存储所有激活。给定一个层{% mathjax %}x\longmapsto y{% endmathjax %}，正常残差层{% mathjax %}y = x + F(x){% endmathjax %}但可逆层将输入和输出分成对{% mathjax %}(x_1,x_2)\longmapsto (y_1,y_2){% endmathjax %}然后执行以下操作：
{% mathjax '{"conversion":{"em":14}}' %}
y_1 = x_1 + F(_2), y_2 = x_2 + G(y_1)
{% endmathjax %}
Reformer 将同样的思想应用到 Transformer 中，结合了注意力机制（{% mathjax %}F{% endmathjax %}) 和前馈层 ({% mathjax %}G{% endmathjax %}）在可逆网络块内：
{% mathjax '{"conversion":{"em":14}}' %}
x_2 = y_2 - G(y_1),x_1 = y_1 - F(x_2)
{% endmathjax %}
通过对前馈计算进行分块，可以进一步减少内存：
{% mathjax '{"conversion":{"em":14}}' %}
Y_1 = X_1 + \text{Attention}(X_2), \; Y_2 = X_2 + \text{FeedForward}(Y_1)
{% endmathjax %}
通过对前馈计算进行分块，可以进一步减少内存：
{% mathjax '{"conversion":{"em":14}}' %}
Y_2 = [Y_2^{(1)}; \dots; Y_2^{(c)}] = [X_2^{(1)} + \text{FeedForward}(Y_1^{(1)}); \dots; X_2^{(c)} + \text{FeedForward}(Y_1^{(c)})]
{% endmathjax %}
由此产生的可逆`Transformer`不需要在每一层存储激活。`Routing Transformer`（`Roy`等人，`2021`年）也基于上下文的键和查询聚类。它不使用`LSH`之类的静态哈希函数，而是利用在线{% mathjax %}k{% endmathjax %}均值聚类，并将其与局部、时间稀疏注意力相结合，以降低注意力复杂度{% mathjax %}\mathcal{O}(L^2){% endmathjax %}到{% mathjax %}\mathcal{O}(L^{1.5}){% endmathjax %}。在路由注意力中，键和查询都聚类在一起{% mathjax %}k{% endmathjax %}均值聚类方法和同一组质心{% mathjax %}\mu= (\mu_1,\ldots,\mu_k)\in \mathbb{R}^{k\times d}{% endmathjax %}。查询被路由到分配到相同质心的键。总复杂度为{% mathjax %}\mathcal{O}(Lkd + L^2 d/k){% endmathjax %}，在这里{% mathjax %}\mathcal{O}(Lkd){% endmathjax %}用于运行聚类分配，{% mathjax %}\mathcal{O}(L^2 d/k){% endmathjax %}用于注意力计算。使用所有相关键和查询，通过`EMA`（指数移动平均）更新集群质心。在`Routing Transformer`的实验中，一些最佳配置仅在模型的最后两层和一半的注意力头中启用路由注意力，而另一半则使用局部注意力。他们还观察到局部注意力是一个非常强大的基线，更大的注意力窗口总是会带来更好的结果。
##### 低秩注意力

`Linformer`（`Wang`等人，`2020`年）用低秩矩阵近似整个注意力矩阵，将时间和空间复杂度降低到线性。`Linformer`不使用昂贵的`SVD`来识别低秩分解，而是添加了两个线性投影{% mathjax %} \mathbf{E}_i,\mathbf{F}_i\in \mathbb{R}^{L\times k}{% endmathjax %}对于键和值矩阵，分别将其维度从{% mathjax %}L\times d{% endmathjax %}到{% mathjax %}k\times d{% endmathjax %}。只要{% mathjax %}k \ll L{% endmathjax %}，注意力记忆就会大大降低。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\overline{\text{head}}_i 
&= \text{attn}(\mathbf{X}_q\mathbf{W}^q_i, \mathbf{E}_i\mathbf{X}_k\mathbf{W}^k_i, \mathbf{F}_i\mathbf{X}_v\mathbf{W}^v_i) \\
&= \underbrace{\text{softmax}\Big( \frac{\mathbf{X}_q\mathbf{W}^q_i (\mathbf{E}_i \mathbf{X}_k\mathbf{W}^k_i)^\top}{\sqrt{d}} \Big)}_{\text{low rank attention matrix }\bar{A} \in \mathbb{R}^{k \times d}} \mathbf{F}_i \mathbf{X}_v\mathbf{W}^v_i
\end{aligned}
{% endmathjax %}
可以采用其他技术来进一步提高`Linformer`的效率：
- 投影层之间的参数共享，例如头部、键值和层（跨所有层）共享。
- 使用不同的{% mathjax %}k{% endmathjax %}在不同的层上，因为较高层的头部往往具有更倾斜的分布（较低的等级），因此我们可以使用较小的{% mathjax %}k{% endmathjax %}在更高层。
- 使用不同类型的投影；例如均值/最大池化、带核和步幅的卷积层{% mathjax %}L/k{% endmathjax %}。

{% asset_img t_21.png "（左）Informer 为键和值添加了两个投影层。（右）推理时间与序列长度的关系图" %}

随机特征注意力（`RFA；Peng`等人，`2021`年）依赖于随机特征方法（`Rahimi & Recht，2007`) 用低秩特征图来近似自注意力中的`softmax`操作，以实现线性时间和空间复杂度。`Performers`(`Choromanski`等人，`2021`年) 还采用了随机特征注意，并改进了内核构造，以进一步降低内核近似误差。

