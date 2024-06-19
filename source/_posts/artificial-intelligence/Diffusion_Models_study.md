---
title: 什么是扩散模型（深度学习）
date: 2024-06-18 11:50:11
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

扩散模型的灵感来自非平衡热力学。它们定义了一个马尔可夫链扩散步骤，以缓慢地向数据添加随机噪声，然后学习逆转扩散过程以从噪声中构建所需的数据样本。与`VAE`或流模型不同，扩散模型是通过固定程序学习的，并且潜在变量具有高维度（与原始数据相同）。
<!-- more -->

{% asset_img dm_1.png "不同类型的生成模型" %}

#### 什么是扩散模型？

已经提出了几种基于扩散的生成模型，其基本思想类似，包括**扩散概率模型**([`Sohl-Dickstein`等，`2015`](https://arxiv.org/abs/1503.03585))、**噪声条件得分网络**([`NCSN；Yang & Ermon`，`2019`](https://arxiv.org/abs/1907.05600))和**去噪扩散概率模型**([`DDPM；Ho`等，`2020`](https://arxiv.org/abs/2006.11239))。
##### 正向扩散过程

给定从真实数据分布中采样的数据点{% mathjax %}\mathbf{x}_0\sim q(\mathbf{x}){% endmathjax %}，让我们来定义一个前向扩散过程，在这个扩散过程中，我们向样本中添加少量的高斯噪声{% mathjax %}T{% endmathjax %}，产生一系列嘈杂的样本{% mathjax %}\mathbf{x}_1,\ldots,\mathbf{x}_T{% endmathjax %}。步长由方差表来控制{% mathjax %}\{\beta_t \in (0, 1)\}_{t=1}^T{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})
{% endmathjax %}
数据样本{% mathjax %}\mathbf{x}_0{% endmathjax %}随着步伐的加快，其特征逐渐消失，{% mathjax %}t{% endmathjax %}变得更大。最终，当{% mathjax %}T\rightarrow \infty,\mathbf{x}_T{% endmathjax %}相当于各向同性的高斯分布。
{% asset_img dm_2.png "通过缓慢添加（去除）噪声生成样本的正向（反向）扩散过程的马尔可夫链" %}

上述过程的一个优点是我们可以采样{% mathjax %}\mathbf{x}_t{% endmathjax %}在任何时间步{% mathjax %}t{% endmathjax %}使用重新参数化以封闭形式呈现。