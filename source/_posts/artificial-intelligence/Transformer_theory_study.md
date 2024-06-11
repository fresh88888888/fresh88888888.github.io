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
a_{ij} = \text{softmax}(\frac{\mathbf{q}_i\mathbf{k}_j^{\mathsf{T}}}{\sqrt{d_k}}) = \frac{\exp(\mathbf{q}_i\mathbf{k}_j^{\mathsf{T}})}{\sqrt{d_k}\sum_{r\in S_i } \exp(\mathbf{q}_i \mathbf{k}_r^{\mathsf{T}})}
{% endmathjax %}
#### 多头自注意力