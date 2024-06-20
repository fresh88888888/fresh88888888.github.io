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

上述过程的一个优点是我们可以采样{% mathjax %}\mathbf{x}_t{% endmathjax %}在任何时间步{% mathjax %}t{% endmathjax %}使用重置参数化以封闭形式呈现。让{% mathjax %}\alpha_t = 1- \beta_t{% endmathjax %}和{% mathjax %}\bar{\alpha}_t= \prod^t_{i=1}\alpha_i{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\mathbf{x}_t 
&= \sqrt{\alpha_t}\mathbf{x}_{t-1} + \sqrt{1 - \alpha_t}\boldsymbol{\epsilon}_{t-1} & \text{ ;where } \boldsymbol{\epsilon}_{t-1}, \boldsymbol{\epsilon}_{t-2}, \dots \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \\
&= \sqrt{\alpha_t \alpha_{t-1}} \mathbf{x}_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \bar{\boldsymbol{\epsilon}}_{t-2} & \text{ ;where } \bar{\boldsymbol{\epsilon}}_{t-2} \text{ merges two Gaussians (*).} \\
&= \dots \\
&= \sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon} \\
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, (1 - \bar{\alpha}_t)\mathbf{I})
\end{aligned}
{% endmathjax %}
回想一下，当我们合并两个不同方差的高斯时，{% mathjax %}\mathcal{N}(\mathbf{0}, \sigma_1^2\mathbf{I}){% endmathjax %}和{% mathjax %}\mathcal{N}(\mathbf{0}, \sigma_2^2\mathbf{I}){% endmathjax %}，新的分布是{% mathjax %}\mathcal{N}(\mathbf{0}, (\sigma_1^2 + \sigma_2^2)\mathbf{I}){% endmathjax %}。此处合并的标准差为{% mathjax %}\sqrt{(1 - \alpha_t) + \alpha_t (1-\alpha_{t-1})} = \sqrt{1 - \alpha_t\alpha_{t-1}}{% endmathjax %}。通常，当样本噪声较大时，我们可以承受更大的更新步长，因此{% mathjax %}\beta_1< beta_2 < \ldots <\beta_T{% endmathjax %}，并且{% mathjax %}\bar{\alpha}_1 > \ldots > \bar{\alpha}_T{% endmathjax %}。

`Langevin dynamics`是物理学中的一个概念，用于对分子系统进行统计建模。结合了随机梯度下降，随机梯度`Langevin dynamics`(`SGLD`)([`Welling & Teh 2011`](https://www.stats.ox.ac.uk/~teh/research/compstats/WelTeh2011a.pdf))可以从概率密度中产生样本{% mathjax %}p(\mathbf{x}){% endmathjax %}在马尔科夫链中仅使用渐变{% mathjax %}\nabla_\mathbf{x} \log p(\mathbf{x}){% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{x}_t = \mathbf{x}_{t-1} + \frac{\delta}{2} \nabla_\mathbf{x} \log p(\mathbf{x}_{t-1}) + \sqrt{\delta} \boldsymbol{\epsilon}_t
,\quad\text{where }
\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
{% endmathjax %}
在这里{% mathjax %}\delta{% endmathjax %}是步长。当{% mathjax %}T \to \infty, \epsilon \to 0{% endmathjax %}，{% mathjax %}\mathbf{x}_T{% endmathjax %}等于真实概率密度{% mathjax %}p(\mathbf{x}){% endmathjax %}。与标准的SGD相比，随机梯度`Langevin dynamics`将高斯噪声注入参数更新中，以避免陷入局部最小值。
##### 逆向扩散过程

如果我们可以逆向上述过程并从{% mathjax %}q(\mathbf{x}_{t-1}|\mathbf{x}_t){% endmathjax %}，我们将能从高斯噪声输入中重建真实样本{% mathjax %}\mathbf{x}_T \sim \mathcal{N}(\mathbf{0}, \mathbf{I}){% endmathjax %}。注意，如果{% mathjax %}\beta_t{% endmathjax %}足够小，{% mathjax %}q(\mathbf{x}_{t-1}|\mathbf{x}_t){% endmathjax %}也将是高斯分布。不幸的是，我们无法估计{% mathjax %}q(\mathbf{x}_{t-1}|\mathbf{x}_t){% endmathjax %}。因为它需要使用整个数据集，所以我们需要学习一个模型{% mathjax %}p_{\theta}{% endmathjax %}近似这些条件概率以运行你想扩散过程。
{% mathjax '{"conversion":{"em":14}}' %}
p_\theta(\mathbf{x}_{0:T}) = p(\mathbf{x}_T) \prod^T_{t=1} p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) \quad
p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
{% endmathjax %}
{% asset_img dm_2.png "训练扩散模型以对2D瑞士卷数据进行建模的示例" %}

值得注意的是，当满足以下条件时，逆向条件概率是可处理的({% mathjax %}\mathbf{x}_0{% endmathjax %})：
{% mathjax '{"conversion":{"em":14}}' %}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \color{blue}{\tilde{\boldsymbol{\mu}}}(\mathbf{x}_t, \mathbf{x}_0), \color{red}{\tilde{\beta}_t} \mathbf{I})
{% endmathjax %}
利用贝叶斯规则，我们可以得到：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) 
&= q(\mathbf{x}_t \vert \mathbf{x}_{t-1}, \mathbf{x}_0) \frac{ q(\mathbf{x}_{t-1} \vert \mathbf{x}_0) }{ q(\mathbf{x}_t \vert \mathbf{x}_0) } \\
&\propto \exp \Big(-\frac{1}{2} \big(\frac{(\mathbf{x}_t - \sqrt{\alpha_t} \mathbf{x}_{t-1})^2}{\beta_t} + \frac{(\mathbf{x}_{t-1} - \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0)^2}{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp \Big(-\frac{1}{2} \big(\frac{\mathbf{x}_t^2 - 2\sqrt{\alpha_t} \mathbf{x}_t \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \alpha_t} \color{red}{\mathbf{x}_{t-1}^2} }{\beta_t} + \frac{ \color{red}{\mathbf{x}_{t-1}^2} \color{black}{- 2 \sqrt{\bar{\alpha}_{t-1}} \mathbf{x}_0} \color{blue}{\mathbf{x}_{t-1}} \color{black}{+ \bar{\alpha}_{t-1} \mathbf{x}_0^2}  }{1-\bar{\alpha}_{t-1}} - \frac{(\mathbf{x}_t - \sqrt{\bar{\alpha}_t} \mathbf{x}_0)^2}{1-\bar{\alpha}_t} \big) \Big) \\
&= \exp\Big( -\frac{1}{2} \big( \color{red}{(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}})} \mathbf{x}_{t-1}^2 - \color{blue}{(\frac{2\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{2\sqrt{\bar{\alpha}_{t-1}}}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)} \mathbf{x}_{t-1} \color{black}{ + C(\mathbf{x}_t, \mathbf{x}_0) \big) \Big)}
\end{aligned}
{% endmathjax %}
