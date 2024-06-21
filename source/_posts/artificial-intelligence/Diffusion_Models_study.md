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
{% asset_img dm_3.png "训练扩散模型以对2D瑞士卷数据进行建模的示例" %}

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
在这里{% mathjax %}C(\mathbf{x}_t,\mathbf{x}_0){% endmathjax %}先省略这部分的细节。按照标准的高斯密度函数，均值和方差可以参数化，如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\tilde{\beta}_t 
&= 1/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) 
= 1/(\frac{\alpha_t - \bar{\alpha}_t + \beta_t}{\beta_t(1 - \bar{\alpha}_{t-1})})
= \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
\tilde{\boldsymbol{\mu}}_t (\mathbf{x}_t, \mathbf{x}_0)
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0)/(\frac{\alpha_t}{\beta_t} + \frac{1}{1 - \bar{\alpha}_{t-1}}) \\
&= (\frac{\sqrt{\alpha_t}}{\beta_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1} }}{1 - \bar{\alpha}_{t-1}} \mathbf{x}_0) \color{green}{\frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t} \\
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \mathbf{x}_0\\
\end{aligned}
{% endmathjax %}
我们将{% mathjax %}\mathbf{x}_0 = frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}}){\boldsymbol{\epsilon}_t}{% endmathjax %}代入以上公式：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\tilde{\boldsymbol{\mu}}_t
&= \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} \mathbf{x}_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} \frac{1}{\sqrt{\bar{\alpha}_t}}(\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t) \\
&= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)}
\end{aligned}
{% endmathjax %}
这种设置与VAE非常相似，因此我们可以使用变分下界来优化负对数似然。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
- \log p_\theta(\mathbf{x}_0) 
&\leq - \log p_\theta(\mathbf{x}_0) + D_\text{KL}(q(\mathbf{x}_{1:T}\vert\mathbf{x}_0) \| p_\theta(\mathbf{x}_{1:T}\vert\mathbf{x}_0) ) \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_{\mathbf{x}_{1:T}\sim q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T}) / p_\theta(\mathbf{x}_0)} \Big] \\
&= -\log p_\theta(\mathbf{x}_0) + \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} + \log p_\theta(\mathbf{x}_0) \Big] \\
&= \mathbb{E}_q \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
\text{Let }L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log \frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \geq - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0)
\end{aligned}
{% endmathjax %}
使用 Jensen 不等式也可以直接得到相同的结果。假设我们想要最小化交叉熵作为学习目标：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
L_\text{CE}
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log p_\theta(\mathbf{x}_0) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int p_\theta(\mathbf{x}_{0:T}) d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \int q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} d\mathbf{x}_{1:T} \Big) \\
&= - \mathbb{E}_{q(\mathbf{x}_0)} \log \Big( \mathbb{E}_{q(\mathbf{x}_{1:T} \vert \mathbf{x}_0)} \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \Big) \\
&\leq - \mathbb{E}_{q(\mathbf{x}_{0:T})} \log \frac{p_\theta(\mathbf{x}_{0:T})}{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})} \\
&= \mathbb{E}_{q(\mathbf{x}_{0:T})}\Big[\log \frac{q(\mathbf{x}_{1:T} \vert \mathbf{x}_{0})}{p_\theta(\mathbf{x}_{0:T})} \Big] = L_\text{VLB}
\end{aligned}
{% endmathjax %}
为了将方程中的每个项转换为可分析计算的，可以进一步将目标重写为几个`KL`散度和熵项的组合：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
L_\text{VLB} 
&= \mathbb{E}_{q(\mathbf{x}_{0:T})} \Big[ \log\frac{q(\mathbf{x}_{1:T}\vert\mathbf{x}_0)}{p_\theta(\mathbf{x}_{0:T})} \Big] \\
&= \mathbb{E}_q \Big[ \log\frac{\prod_{t=1}^T q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{ p_\theta(\mathbf{x}_T) \prod_{t=1}^T p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t) } \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=1}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t\vert\mathbf{x}_{t-1})}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \Big( \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)}\cdot \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1}\vert\mathbf{x}_0)} \Big) + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_t \vert \mathbf{x}_0)}{q(\mathbf{x}_{t-1} \vert \mathbf{x}_0)} + \log\frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big] \\
&= \mathbb{E}_q \Big[ -\log p_\theta(\mathbf{x}_T) + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} + \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{q(\mathbf{x}_1 \vert \mathbf{x}_0)} + \log \frac{q(\mathbf{x}_1 \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)} \Big]\\
&= \mathbb{E}_q \Big[ \log\frac{q(\mathbf{x}_T \vert \mathbf{x}_0)}{p_\theta(\mathbf{x}_T)} + \sum_{t=2}^T \log \frac{q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)}{p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t)} - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1) \Big] \\
&= \mathbb{E}_q [\underbrace{D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T))}_{L_T} + \sum_{t=2}^T \underbrace{D_\text{KL}(q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_{t-1} \vert\mathbf{x}_t))}_{L_{t-1}} \underbrace{- \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)}_{L_0} ]
\end{aligned}
{% endmathjax %}
让我们分别标记变分下限损失中的每个元素：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
L_\text{VLB} &= L_T + L_{T-1} + \dots + L_0 \\
\text{where } L_T &= D_\text{KL}(q(\mathbf{x}_T \vert \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_T)) \\
L_t &= D_\text{KL}(q(\mathbf{x}_t \vert \mathbf{x}_{t+1}, \mathbf{x}_0) \parallel p_\theta(\mathbf{x}_t \vert\mathbf{x}_{t+1})) \text{ for }1 \leq t \leq T-1 \\
L_0 &= - \log p_\theta(\mathbf{x}_0 \vert \mathbf{x}_1)
\end{aligned}
{% endmathjax %}
{% mathjax %}L_T{% endmathjax %}是恒定的，在训练期间可以忽略，{% mathjax %}\mathbf{x}_T{% endmathjax %}是高斯噪声。{% mathjax %}L_0{% endmathjax %}使用单独的离散解码器{% mathjax %}\mathcal{N}(\mathbf{x}_0; \boldsymbol{\mu}_\theta(\mathbf{x}_1, 1), \boldsymbol{\Sigma}_\theta(\mathbf{x}_1, 1)){% endmathjax %}。
##### 参数化L_t训练损失

我们需要学习一个神经网络来近似逆向扩散过程中的条件概率分布，{% mathjax %}p_\theta(\mathbf{x}_{t-1} \vert \mathbf{x}_t) = \mathcal{N}(\mathbf{x}_{t-1}; \boldsymbol{\mu}_\theta(\mathbf{x}_t, t), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t)){% endmathjax %}，希望通过{% mathjax %}\boldsymbol{\mu}_{\theta}{% endmathjax %}预测{% mathjax %}\tilde{\boldsymbol{\mu}}_t = \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big){% endmathjax %}，因为{% mathjax %}\mathbf{x}_t{% endmathjax %}在训练时可以作为输入，我们可以重新参数化高斯噪声项，预测{% mathjax %}\boldsymbol{\epsilon}_t{% endmathjax %}从输入{% mathjax %}\mathbf{x}_t{% endmathjax %},在时间步{% mathjax %}t{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\boldsymbol{\mu}_\theta(\mathbf{x}_t, t) &= \color{cyan}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big)} \\
\text{Thus }\mathbf{x}_{t-1} &= \mathcal{N}(\mathbf{x}_{t-1}; \frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big), \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t))
\end{aligned}
{% endmathjax %}
损失项{% mathjax %}L_t{% endmathjax %}参数化，从而最小化{% mathjax %}\tilde{\boldsymbol{\mu}}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
L_t 
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2 \| \boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) \|^2_2} \| \color{blue}{\tilde{\boldsymbol{\mu}}_t(\mathbf{x}_t, \mathbf{x}_0)} - \color{green}{\boldsymbol{\mu}_\theta(\mathbf{x}_t, t)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{1}{2  \|\boldsymbol{\Sigma}_\theta \|^2_2} \| \color{blue}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_t \Big)} - \color{green}{\frac{1}{\sqrt{\alpha_t}} \Big( \mathbf{x}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) \Big)} \|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{\mathbf{x}_0, \boldsymbol{\epsilon}} \Big[\frac{ (1 - \alpha_t)^2 }{2 \alpha_t (1 - \bar{\alpha}_t) \| \boldsymbol{\Sigma}_\theta \|^2_2} \|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big] 
\end{aligned}
{% endmathjax %}
从实证研究来看，[`Ho`等人, `2020`年](https://arxiv.org/abs/2006.11239)发现，采用忽略加权项的简化方法来训练扩散模型效果会更好：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
L_t^\text{simple}
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\|^2 \Big] \\
&= \mathbb{E}_{t \sim [1, T], \mathbf{x}_0, \boldsymbol{\epsilon}_t} \Big[\|\boldsymbol{\epsilon}_t - \boldsymbol{\epsilon}_\theta(\sqrt{\bar{\alpha}_t}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_t}\boldsymbol{\epsilon}_t, t)\|^2 \Big]
\end{aligned}
{% endmathjax %}
最终的结果是：
{% mathjax '{"conversion":{"em":14}}' %}
L_\text{simple} = L_t^\text{simple} + C
{% endmathjax %}
在这里{% mathjax %}C{% endmathjax %}是常数，不依赖于{% mathjax %}\theta{% endmathjax %}。
{% asset_img dm_4.png "DDPM 中的训练和采样算法" %}

###### 噪声条件分数网络(NCSN)

[`Song & Ermon`,`2019`)](https://arxiv.org/abs/1907.05600)提出了一种基于分数的生成建模方法，其中样本通过随机梯度`Langevin dynamics`生成，使用通过分数匹配估计的数据分布梯度。每个样本的分数{% mathjax %}\mathbf{x}{% endmathjax %}的密度概率定义为其梯度{% mathjax %}\nabla_{\mathbf{x}} \log q(\mathbf{x}){% endmathjax %}。评分网络{% mathjax %}\mathbf{s}_\theta: \mathbb{R}^D \to \mathbb{R}^D{% endmathjax %}经过训练可以估算它，{% mathjax %}\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log q(\mathbf{x}){% endmathjax %}。为了使其能够在深度学习环境中扩展到高维数据，他们建议使用去噪分数匹配（[`Vincent，2011`](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)）或切片分数匹配([`Song`等人，`2019`](https://arxiv.org/abs/1905.07088))。去噪分数匹配会向数据添加预先指定的小噪声{% mathjax %}q(\tilde{\mathbf{x}}|\mathbf{x}){% endmathjax %}和{% mathjax %}q(\tilde{\mathbf{x}}){% endmathjax %}与分数匹配。回想一下，随机梯度`Langevin dynamics`可以仅使用分数从概率密度分布中采样数据点{% mathjax %}\nabla_{\mathbf{x}} \log q(\mathbf{x}){% endmathjax %}在一个迭代过程中。然而，根据流形假设，即使观察到的数据看起来只是任意高维的，大多数数据预计会集中在低维流形中。这会给分数估计带来负面影响，因为数据点无法覆盖整个空间。在数据密度较低的区域，分数估计的可靠性较低。在添加一个小的高斯噪声后，扰动的数据分布会覆盖整个空间{% mathjax %}\mathbb{R}^D{% endmathjax %}，得分估计网络的训练变得更加稳定。[`Song & Ermon,2019`](https://arxiv.org/abs/1907.05600)通过用不同级别的噪声扰动数据对其进行了改进，并训练了一个噪声条件下的得分网络来联合估计不同噪声水平下所有扰动数据的得分。噪声水平增加的时间表类似于正向扩散过程。如果我们使用扩散过程注释，得分近似于{% mathjax %}\mathbf{s}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t){% endmathjax %}。给定高斯分布{% mathjax %}\mathbf{x} \sim \mathcal{N}(\mathbf{\mu}, \sigma^2 \mathbf{I}){% endmathjax %}，我们可以将其密度函数对数的导数写为{% mathjax %}\nabla_{\mathbf{x}}\log p(\mathbf{x}) = \nabla_{\mathbf{x}} \Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) = - \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2} = - \frac{\boldsymbol{\epsilon}}{\sigma}{% endmathjax %}，在这里，{% mathjax %}\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I}){% endmathjax %}。则：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{s}_\theta(\mathbf{x}_t, t) 
\approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t)
= \mathbb{E}_{q(\mathbf{x}_0)} [\nabla_{\mathbf{x}_t} q(\mathbf{x}_t \vert \mathbf{x}_0)]
= \mathbb{E}_{q(\mathbf{x}_0)} \Big[ - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}} \Big]
= - \frac{\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)}{\sqrt{1 - \bar{\alpha}_t}}
{% endmathjax %}

##### 参数化\beta_t
在[`Ho,2020`](https://arxiv.org/abs/2006.11239)中，前向方差被设置为线性增加的常数序列，来自{% mathjax %}\beta_1 = 10^{-4}{% endmathjax %}到{% mathjax %}\beta_T = 0.02{% endmathjax %}。与标准化图像像素值相比，它们相对较小{% mathjax %}[-1,1]{% endmathjax %}。扩散模型在实验中表现出了高质量的样本，但仍然无法像其他生成模型那样实现有竞争力的模型对数似然。[`Nichol & Dhariwal,2021`](https://arxiv.org/abs/2102.09672)提出了几种改进技术，以帮助扩散模型获得更低的`NLL`。其中一项改进是使用基于余弦的方差调度。调度函数的选择可以是任意的，只要它在训练过程的中间提供近乎线性的下降，并在周围提供细微的变化即可{% mathjax %}t=0{% endmathjax %}和{% mathjax %}t=T{% endmathjax %}
{% mathjax '{"conversion":{"em":14}}' %}
\beta_t = \text{clip}(1-\frac{\bar{\alpha}_t}{\bar{\alpha}_{t-1}}, 0.999) \quad\bar{\alpha}_t = \frac{f(t)}{f(0)}\quad\text{where }f(t)=\cos\Big(\frac{t/T+s}{1+s}\cdot\frac{\pi}{2}\Big)^2
{% endmathjax %}
其中小偏移{% mathjax %}s{% endmathjax %}是为了防止{% mathjax %}\beta_t{% endmathjax %}靠近时太小，{% mathjax %}t=0{% endmathjax %}。
{% asset_img dm_5.png "线性和基于余弦的调度比较，\beta_t在训练过程中" %}

##### 逆向过程方差的参数化\sum_{\theta}

[`Ho`等人,`2020`年](https://arxiv.org/abs/2006.11239)选择修复{% mathjax %}\beta_t{% endmathjax %}作为常量，而不是让它们可学习和设置{% mathjax %}\boldsymbol{\sum}_{theta}(\mathbf{x}_t,t) = \sigma^2_t\mathbf{I}{% endmathjax %}，在这里{% mathjax %}\sigma_t{% endmathjax %}不是学习而是设置为{% mathjax %}\beta_t{% endmathjax %}或者{% mathjax %}\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t{% endmathjax %}因为发现学习对角方差{% mathjax %}\boldsymbol{\sum}_{\theta}{% endmathjax %}导致训练不稳定，样本质量较差。[`Nichol & Dhariwal,2021`](https://arxiv.org/abs/2102.09672)提出学习{% mathjax %}\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t){% endmathjax %}作为之间的插值{% mathjax %}\beta_t{% endmathjax %}和{% mathjax %}\tilde{\beta}_t{% endmathjax %}通过模型预测混合向量{% mathjax %}\mathbf{v}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\boldsymbol{\Sigma}_\theta(\mathbf{x}_t, t) = \exp(\mathbf{v} \log \beta_t + (1-\mathbf{v}) \log \tilde{\beta}_t)
{% endmathjax %}
然而，简单的目标{% mathjax %}L_\text{simple}{% endmathjax %}不依赖于{% mathjax %}\boldsymbol{\sum}_{\theta}{% endmathjax %}为了增加依赖性，他们构建了一个混合目标{% mathjax %}L_\text{hybrid} = L_\text{simple} + \lambda L_\text{VLB}{% endmathjax %}在这里{% mathjax %}\lambda= 0.001{% endmathjax %}很小，并且停止梯度{% mathjax %}\boldsymbol{\mu}_\theta{% endmathjax %}在里面{% mathjax %}L_\text{VLB}{% endmathjax %}术语{% mathjax %}L_\text{VLB}{% endmathjax %}仅指导学习{% mathjax %}\boldsymbol{\sum}_{\theta}{% endmathjax %}。他们通过实证研究观察到{% mathjax %}L_\text{VLB}{% endmathjax %}由于梯度噪声的存在，优化起来相当困难，因此他们建议使用时间平均平滑版本的{% mathjax %}L_\text{VLB}{% endmathjax %}具有重要性抽样。
{% asset_img dm_6.png "改进的DDPM与其他基于似然的生成模型的负对数似然比较。NLL以位/维为单位" %}

#### 条件生成

在使用条件信息的图像（例如`ImageNet`数据集）训练生成模型时，通常会生成以类标签或一段描述性文本为条件的样本。
##### 分类器引导扩散

为了将类别信息明确地纳入传播过程，[`Dhariwal & Nichol,2021`](https://arxiv.org/abs/2105.05233)训练了一个分类器{% mathjax %}f_\phi(y \vert \mathbf{x}_t, t){% endmathjax %}在嘈杂的图像上{% mathjax %}\mathbf{x}_t{% endmathjax %}并使用渐变{% mathjax %}\nabla_\mathbf{x} \log f_\phi(y \vert \mathbf{x}_t){% endmathjax %}引导扩散采样过程朝向调节信息{% mathjax %}y{% endmathjax %}（例如目标类别标签）通过改变噪声预测来实现。 回想一下{% mathjax %}\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) = - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t){% endmathjax %}我们可以写出联合分布的得分函数{% mathjax %}q(\mathbf{x}_t,y){% endmathjax %}如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t, y)
&= \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t) + \nabla_{\mathbf{x}_t} \log q(y \vert \mathbf{x}_t) \\
&\approx - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) + \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}} (\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t))
\end{aligned}
{% endmathjax %}
因此，一个新的分类器引导预测器{% mathjax %}\bar{\boldsymbol{\epsilon}}_\theta{% endmathjax %}将采用以下形式：
{% mathjax '{"conversion":{"em":14}}' %}
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)
{% endmathjax %}
为了控制分类器指导的强度，我们可以添加一个权重{% mathjax %}w{% endmathjax %}对于`delta`部分：
{% mathjax '{"conversion":{"em":14}}' %}
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(x_t, t) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log f_\phi(y \vert \mathbf{x}_t)
{% endmathjax %}
由此产生的消融扩散模型(`ADM`)和具有附加分类器指导的模型(`ADM-G`)能够取得比`SOTA`生成模型（例如`BigGAN`）更好的结果。
{% asset_img dm_7.png "算法使用分类器的指导，使用DDPM和DDIM运行条件生成" %}

此外，[`Dhariwal & Nichol,2021`](https://arxiv.org/abs/2105.05233)对`U-Net`架构进行了一些修改，其性能优于采用扩散模型的`GAN`。架构修改包括更大的模型深度/宽度、更多的注意力头、多分辨率注意力、用于上/下采样的`BigGAN`残差块、残差连接重新缩放{% mathjax %}1/\sqrt{2}{% endmathjax %}和自适应组规范化(`AdaGN`)。
##### 无分类器引导

没有独立的分类器{% mathjax %} {% endmathjax %}，仍然可以通过合并条件和非条件扩散模型的分数来运行条件扩散步骤([`Ho & Salimans, 2021`](https://openreview.net/forum?id=qw8AKxfYbI))。让无条件去噪扩散模型{% mathjax %}p_{\theta}(\mathbf{x}){% endmathjax %}通过分数估计器进行参数化{% mathjax %}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t){% endmathjax %}和条件模型{% mathjax %}p_{\theta}(\mathbf{x}|y){% endmathjax %}通过参数化{% mathjax %}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y){% endmathjax %}。这两个模型可以通过单个神经网络进行学习。确切地说，条件扩散模型{% mathjax %}p_{\theta}(\mathbf{x}|y){% endmathjax %}使用配对数据进行训练{% mathjax %}(\mathbf{x},y){% endmathjax %}，其中条件信息{% mathjax %}y{% endmathjax %}定期随机丢弃，以便模型知道如何无条件地生成图像，即{% mathjax %}\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) = \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y=\varnothing){% endmathjax %}。隐式分类器的梯度可以用条件和非条件分数估计器来表示。一旦插入分类器引导的修改分数，该分数就不依赖于单独的分类器。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t)
&= \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \vert y) - \nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t) \\
&= - \frac{1}{\sqrt{1 - \bar{\alpha}_t}}\Big( \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \Big) \\
\bar{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, y)
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \sqrt{1 - \bar{\alpha}_t} \; w \nabla_{\mathbf{x}_t} \log p(y \vert \mathbf{x}_t) \\
&= \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) + w \big(\boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t) \big) \\
&= (w+1) \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, y) - w \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)
\end{aligned}
{% endmathjax %}
他们的实验表明，无分类器指导可以在`FID`（区分合成图像和生成图像）和`IS`（质量和多样性）之间实现良好的平衡。**引导扩散模型GLIDE**（[`Nichol、Dhariwal 和 Ramesh`等人，`2022`年](https://arxiv.org/abs/2112.10741)）探索了两种引导策略，即`CLIP`引导和无分类器引导，并发现后者更受欢迎。他们假设这是因为`CLIP`引导利用对抗性示例对`CLIP`模型进行攻击，而不是优化更匹配的图像生成。

#### 加速扩散模型

通过遵循逆向扩散过程的马尔可夫链从`DDPM`生成样本非常慢，因为{% mathjax %}T{% endmathjax %}最多可以达到一到几千步。[`Song`等人,`2020`年](https://arxiv.org/abs/2010.02502)：“例如，从`DDPM`中采样`50k`张`32 × 32`大小的图像大约需要`20`个小时，但从`Nvidia 2080 Ti GPU`上的`GAN`中采样只需不到一分钟。”
##### 减少采样步骤和蒸馏

一种简单的方法是运行跨步采样计划([`Nichol & Dhariwal，2021`](https://arxiv.org/abs/2102.09672))，每隔一段时间进行一次采样更新{% mathjax %}\lceil T/S \rceil{% endmathjax %}减少流程的步骤{% mathjax %}T{% endmathjax %}到{% mathjax %}S{% endmathjax %}步骤。新的采样计划是{% mathjax %}\{\tau_1, \dots, \tau_S\}{% endmathjax %}在这里{% mathjax %}\tau_1 < \tau_2 < \dots <\tau_S \in [1, T]{% endmathjax %}和{% mathjax %}S < T{% endmathjax %}。对于另一种方法，让我们重写{% mathjax %}q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0){% endmathjax %}通过所需的标准偏差进行参数化{% mathjax %}\sigma_t{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\mathbf{x}_{t-1} 
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 +  \sqrt{1 - \bar{\alpha}_{t-1}}\boldsymbol{\epsilon}_{t-1} & \\
&= \sqrt{\bar{\alpha}_{t-1}}\mathbf{x}_0 + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \boldsymbol{\epsilon}_t + \sigma_t\boldsymbol{\epsilon} & \\
&= \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t) + \sigma_t\boldsymbol{\epsilon} \\
q_\sigma(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0)
&= \mathcal{N}(\mathbf{x}_{t-1}; \sqrt{\bar{\alpha}_{t-1}} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
\end{aligned}
{% endmathjax %}
回想一下，{% mathjax %}q(\mathbf{x}_{t-1} \vert \mathbf{x}_t, \mathbf{x}_0) = \mathcal{N}(\mathbf{x}_{t-1}; \tilde{\boldsymbol{\mu}}(\mathbf{x}_t, \mathbf{x}_0), \tilde{\beta}_t \mathbf{I}){% endmathjax %}，因此我们有：
{% mathjax '{"conversion":{"em":14}}' %}
\tilde{\beta}_t = \sigma_t^2 = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \cdot \beta_t
{% endmathjax %}
让{% mathjax %}\sigma_t^2 = \eta \cdot \tilde{\beta}_t{% endmathjax %}我们可以调整{% mathjax %}\eta\in \mathbb{R}^+{% endmathjax %}作为控制采样随机性的超参数。特殊情况是{% mathjax %}\eta = 0{% endmathjax %}使采样过程具有确定性。这种模型被称为去噪扩散隐式模([`DDIM；Song`等人，`2020`年](https://arxiv.org/abs/2010.02502))。`DDIM`具有相同的边际噪声分布，但确定性地将噪声映射回原始数据样本。在生成过程中，我们不必遵循整个链条{% mathjax %}t=1,\ldots,T{% endmathjax %}，而是步骤的子集。我们表示为{% mathjax %}s < t{% endmathjax %}在该加速轨迹中分为两个步骤。`DDIM`更新步骤为：
{% mathjax '{"conversion":{"em":14}}' %}
q_{\sigma, s < t}(\mathbf{x}_s \vert \mathbf{x}_t, \mathbf{x}_0)
= \mathcal{N}(\mathbf{x}_s; \sqrt{\bar{\alpha}_s} \Big( \frac{\mathbf{x}_t - \sqrt{1 - \bar{\alpha}_t} \epsilon^{(t)}_\theta(\mathbf{x}_t)}{\sqrt{\bar{\alpha}_t}} \Big) + \sqrt{1 - \bar{\alpha}_s - \sigma_t^2} \epsilon^{(t)}_\theta(\mathbf{x}_t), \sigma_t^2 \mathbf{I})
{% endmathjax %}
虽然所有模型都经过训练{% mathjax %}T=1000{% endmathjax %}在实验中的扩散步骤中，他们观察到`DDIM`（{% mathjax %}\eta = 0{% endmathjax %}）可以生产出最优质的样品时{% mathjax %}S{% endmathjax %}较小，而`DDPM`（{% mathjax %}\eta = 1{% endmathjax %}）在小规模上{% mathjax %}S{% endmathjax %}表现更差。当我们能够运行完整的逆向马尔可夫扩散步骤时，`DDPM`确实表现得更好({% mathjax %}S=T=1000{% endmathjax %})。使用`DDIM`，可以训练扩散模型进行任意数量的前向步骤，但只能从生成过程中的一部分步骤中进行采样。
{% asset_img dm_8.png "不同设置的扩散模型在CIFAR10和CelebA数据集上的FID分数" %}

与`DDPM`相比，`DDIM`能够：
- 使用更少的步骤生成更高质量的样本。
- 由于生成过程是确定性的，因此具有“一致性”属性，这意味着以相同潜在变量为条件的多个样本应该具有相似的高级特征。
- 由于一致性，`DDIM`可以在潜在变量中进行语义上有意义的插值。

{% asset_img dm_9.png "渐进式蒸馏可以在每次迭代中将扩散采样步骤减少一半" %}

渐进式蒸馏([`Salimans & Ho，2022`](https://arxiv.org/abs/2202.00512))是一种将训练好的确定性采样器蒸馏成采样步骤减半的新模型的方法。学生模型从教师模型初始化，并朝着一个学生`DDIM`步骤匹配`2`个步骤的目标进行去噪，而不是使用原始样本{% mathjax %}\mathbf{x}_0{% endmathjax %}作为去噪目标。在每次渐进蒸馏迭代中，我们可以将采样步骤减半。
{% asset_img dm_10.png "算法1（扩散模型训练）与算法2（渐进式蒸馏）的并排比较，其中渐进式蒸馏的相对变化以绿色突出显示" %}

一致性模型([`Song`等人，`2023`年](https://arxiv.org/abs/2303.01469))学习映射任何中间噪声数据点{% mathjax %}\mathbf{x}_t,t > 0{% endmathjax %}沿着扩散采样轨迹直接回到原点{% mathjax %}\mathbf{x}_0{% endmathjax %}。由于它具有自一致性，因此被称为**一致性模型**，因为同一轨迹上的任何数据点都映射到同一原点。
{% asset_img dm_11.png "一致性模型学习将轨迹上的任何数据点映射回其原点" %}

给定轨迹{% mathjax %}\{\mathbf{x}_t \vert t \in [\epsilon, T]\}{% endmathjax %}，一致性函数{% mathjax %}f{% endmathjax %}定义为{% mathjax %}f: (\mathbf{x}_t, t) \mapsto \mathbf{x}_\epsilon{% endmathjax %}和方程{% mathjax %}f(\mathbf{x}_t, t) = f(\mathbf{x}_{t’}, t’) = \mathbf{x}_\epsilon{% endmathjax %}适用于所有人{% mathjax %}t, t’ \in [\epsilon, T]{% endmathjax %}。什么时候{% mathjax %}t=\epsilon{% endmathjax %}，{% mathjax %}f{% endmathjax %}是一个识别函数。该模型可以参数化如下，其中{% mathjax %}c_\text{skip}(t){% endmathjax %}和{% mathjax %}c_\text{out}(t){% endmathjax %}函数的设计方式是{% mathjax %}c_\text{skip}(\epsilon) = 1, c_\text{out}(\epsilon) = 0{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
f_\theta(\mathbf{x}, t) = c_\text{skip}(t)\mathbf{x} + c_\text{out}(t) F_\theta(\mathbf{x}, t)
{% endmathjax %}
一致性模型可以一步生成样本，同时仍保持通过多步采样过程来交易计算以获得更好质量的灵活性。论文介绍了两种训练一致性模型的方法：
- 一致性蒸馏(`CD`)：通过最小化由相同轨迹生成的对之间的模型输出差异，将扩散模型蒸馏为一致性模型。这使得抽样评估的成本大大降低。一致性蒸馏损失为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
 \mathcal{L}^N_\text{CD} (\theta, \theta^-; \phi) &= \mathbb{E}
 [\lambda(t_n)d(f_\theta(\mathbf{x}_{t_{n+1}}, t_{n+1}), f_{\theta^-}(\hat{\mathbf{x}}^\phi_{t_n}, t_n)] \\
 \hat{\mathbf{x}}^\phi_{t_n} &= \mathbf{x}_{t_{n+1}} - (t_n - t_{n+1}) \Phi(\mathbf{x}_{t_{n+1}}, t_{n+1}; \phi)
 \end{aligned}
{% endmathjax %}
在这里：
  - {% mathjax %}\Phi(.;\phi){% endmathjax %}是单步`ODE`求解器的更新函数；
  - {% mathjax %}n \sim \mathcal{U}[1, N-1]{% endmathjax %}，具有均匀分布{% mathjax %}1, \dots, N-1{% endmathjax %}；
  - 网络参数{% mathjax %}\theta^{-}{% endmathjax %}是`EMA`版本的{% mathjax %}\theta{% endmathjax %}这极大地稳定了训练（就像在`DQN`或动量对比学习中一样）；
  - {% mathjax %}d(.,.){% endmathjax %}是一个正距离度量函数，满足{% mathjax %}\forall \mathbf{x}, \mathbf{y}: d(\mathbf{x}, \mathbf{y}) \leq 0{% endmathjax %}和{% mathjax %}d(\mathbf{x}, \mathbf{y}) = 0{% endmathjax %}当且仅当{% mathjax %}\mathbf{x} = \mathbf{y}{% endmathjax %}。例如{% mathjax %}\ell_2{% endmathjax %}，{% mathjax %}\ell_1{% endmathjax %}或`LPIPS`（学习感知图像块相似性）距离；
  - {% mathjax %}\lambda(.) \in \mathbb{R}^+{% endmathjax %}是一个正权重函数，论文中设置{% mathjax %}\lambda(t_n)=1{% endmathjax %}。

- 一致性训练(`CT`)：另一种选择是独立训练一致性模型。请注意，在`CD`中，预先训练的评分模型{% mathjax %}s_\phi(\mathbf{x}, t){% endmathjax %}用于近似真实得分{% mathjax %}\nabla\log p_t(\mathbf{x}){% endmathjax %}但在`CT`中，我们需要一种方法来估计这个得分函数，结果是{% mathjax %}\nabla\log p_t(\mathbf{x}){% endmathjax %}存在为{% mathjax %}-\frac{\mathbf{x}_t - \mathbf{x}}{t^2}{% endmathjax %}`CT`损失定义如下：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}^N_\text{CT} (\theta, \theta^-; \phi) = \mathbb{E}
[\lambda(t_n)d(f_\theta(\mathbf{x} + t_{n+1} \mathbf{z},\;t_{n+1}), f_{\theta^-}(\mathbf{x} + t_n \mathbf{z},\;t_n)]
\text{ where }\mathbf{z} \in \mathcal{N}(\mathbf{0}, \mathbf{I})
{% endmathjax %}
根据论文中的实验，他们发现:
- `Heun ODE`求解器比欧拉一阶求解器效果更好，因为高阶 ODE 求解器在同样的条件下估计误差更小{% mathjax %}N{% endmathjax %}。
- 在距离度量函数的不同选项中{% mathjax %}d(\cdot){% endmathjax %}，`LPIPS`指标比{% mathjax %}\ell_1{% endmathjax %}和{% mathjax %}\ell_2{% endmathjax %}距离。
- 较小{% mathjax %}N{% endmathjax %}导致更快的收敛但样本更差，而更大的{% mathjax %}N{% endmathjax %}导致收敛速度较慢，但​​收敛时样本更好。
{% asset_img dm_12.png "不同配置下一致性模型性能比较。CD的最佳配置是LPIPS距离度量、Heun ODE求解器和N = 18" %}

##### 潜变量空间

潜在扩散模型([`LDM；Rombach & Blattmann`等人，`2022`年](https://arxiv.org/abs/2112.10752))在潜变量空间而不是像素空间中运行扩散过程，从而降低训练成本并加快推理速度。它的动机是观察到图像的大多数位都对感知细节有贡献，并且语义和概念组成在经过大量压缩后仍然存在。`LDM`通过生成模型学习松散地分解感知压缩和语义压缩，首先使用自动编码器修剪像素级冗余，然后使用学习到的潜在扩散过程操纵/生成语义概念。
{% asset_img dm_13.png "压缩率与失真之间的权衡图，说明两阶段压缩-感知压缩和语义压缩" %}

感知压缩过程依赖于自动编码器模型。编码器{% mathjax %}\mathcal{E}{% endmathjax %}用于压缩输入图像{% mathjax %}\mathbf{x} \in \mathbb{R}^{H \times W \times 3}{% endmathjax %}转换为更小的二维潜在向量{% mathjax %}\mathbf{z} = \mathcal{E}(\mathbf{x}) \in \mathbb{R}^{h \times w \times c}{% endmathjax %}，其中下采样率{% mathjax %}f=H/h=W/w=2^m, m \in \mathbb{N}{% endmathjax %}。然后解码器{% mathjax %}\mathcal{D}{% endmathjax %}根据潜在向量重建图像，{% mathjax %}\tilde{\mathbf{x}} = \mathcal{D}(\mathbf{z}){% endmathjax %}。本文探讨了自动编码器训练中的两种正则化类型，以避免潜在空间中出现任意高的方差。
- `KL-reg`：对学习到的潜在标准正态分布的小`KL`惩罚，类似于`VAE`。
- `VQ-reg`：在解码器内使用矢量量化层，类似`VQVAE`，但量化层被解码器吸收。

扩散和去噪过程发生在潜在向量上{% mathjax %}\mathbf{z}{% endmathjax %}。去噪模型是一个时间条件化的`U-Net`，并增加了交叉注意机制，以处理用于图像生成的灵活条件信息（例如类标签、语义图、图像的模糊变体）。该设计相当于使用交叉注意机制将不同模态的表示融合到模型中。每种类型的条件信息都与特定于域的编码器配对{% mathjax %}\tau_{\theta}{% endmathjax %}投射条件输入{% mathjax %}y{% endmathjax %}可以映射到交叉注意组件的中间表示，{% mathjax %}\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
&\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\Big(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}}\Big) \cdot \mathbf{V} \\
&\text{where }\mathbf{Q} = \mathbf{W}^{(i)}_Q \cdot \varphi_i(\mathbf{z}_i),\;
\mathbf{K} = \mathbf{W}^{(i)}_K \cdot \tau_\theta(y),\;
\mathbf{V} = \mathbf{W}^{(i)}_V \cdot \tau_\theta(y) \\
&\text{and }
\mathbf{W}^{(i)}_Q \in \mathbb{R}^{d \times d^i_\epsilon},\;
\mathbf{W}^{(i)}_K, \mathbf{W}^{(i)}_V \in \mathbb{R}^{d \times d_\tau},\;
\varphi_i(\mathbf{z}_i) \in \mathbb{R}^{N \times d^i_\epsilon},\;
\tau_\theta(y) \in \mathbb{R}^{M \times d_\tau}
\end{aligned}
{% endmathjax %}
{% asset_img dm_14.png "潜在扩散模型(LDM)的架构" %}

#### 提高AI生成内容的分辨率和整体质量

为了生成高分辨率的高质量图像，[`Ho`等人,`2021`](https://arxiv.org/abs/2106.15282)提出使用多个扩散模型的管道来提高分辨率。管道模型之间的噪声条件增强对于最终的图像质量至关重要，即对条件输入应用强大的数据增强{% mathjax %}p_{\theta}(\mathbf{x}|\mathbf{z}){% endmathjax %}每个超分辨率模型{% mathjax %}\mathbf{z}{% endmathjax %}。调节噪声有助于减少管道设置中的复合误差。`U-net`是用于高分辨率图像生成的扩散建模中模型架构的常见选择。
{% asset_img dm_15.png "分辨率不断提高的多个扩散模型的级联管道" %}

他们发现最有效的噪声是在低分辨率下应用高斯噪声，在高分辨率下应用高斯模糊。此外，他们还探索了两种需要对训练过程进行小幅修改的条件增强形式。请注意，条件噪声仅适用于训练，而不适用于推理。
- 截断条件增强会在步骤早期停止扩散过程{% mathjax %}t > 0{% endmathjax %}低分辨率。
- 非截断条件增强会运行完整的低分辨率逆过程，直到步骤`0`，但随后会通过{% mathjax %}\mathbf{z}_t \sim q(\mathbf{x}_t \vert \mathbf{x}_0){% endmathjax %}然后喂养s损坏的{% mathjax %}\mathbf{z}_t{% endmathjax %}进入超分辨率模型。

两阶段扩散模型`unCLIP`([Ramesh等人，2022](https://arxiv.org/abs/2204.06125))年大量利用`CLIP`文本编码器来生成高质量的文本引导图像。给定一个预训练的`CLIP`模型{% mathjax %}\mathbf{c}{% endmathjax %}以及扩散模型的配对训练数据，{% mathjax %}(\mathbf{x},y){% endmathjax %}，在这里{% mathjax %}x{% endmathjax %}是一张图片，{% mathjax %}y{% endmathjax %}是相应的标题，我们可以计算`CLIP`文本和图像嵌入，{% mathjax %}\mathbf{c}^t(y){% endmathjax %}和{% mathjax %}\mathbf{c}^i(\mathbf{x}){% endmathjax %}`unCLIP`并行学习两个模型：
- 先前模型{% mathjax %}P(\mathbf{c}^i \vert y){% endmathjax %}：输出`CLIP`图像嵌入{% mathjax %}\mathbf{c}^i{% endmathjax %}鉴于文本{% mathjax %}y{% endmathjax %}。
- 解码器{% mathjax %}P(\mathbf{x} \vert \mathbf{c}^i, [y]){% endmathjax %}：生成图像{% mathjax %}\mathbf{x}{% endmathjax %}给定`CLIP`图像嵌入{% mathjax %}\mathbf{c}^i{% endmathjax %}以及可选的原始文本{% mathjax %}y{% endmathjax %}。

这两个模型可以实现条件生成，因为：
{% mathjax '{"conversion":{"em":14}}' %}
\underbrace{P(\mathbf{x} \vert y) = P(\mathbf{x}, \mathbf{c}^i \vert y)}_{\mathbf{c}^i\text{ is deterministic given }\mathbf{x}} = P(\mathbf{x} \vert \mathbf{c}^i, y)P(\mathbf{c}^i \vert y)
{% endmathjax %}
{% asset_img dm_16.png "unCLIP 的架构" %}

`unCLIP`遵循两阶段图像生成过程：
