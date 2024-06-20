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

[`Song & Ermon`,`2019`)](https://arxiv.org/abs/1907.05600)提出了一种基于分数的生成建模方法，其中样本通过随机梯度`Langevin dynamics`生成，使用通过分数匹配估计的数据分布梯度。每个样本的分数{% mathjax %}\mathbf{x}{% endmathjax %}的密度概率定义为其梯度{% mathjax %}\nabla_{\mathbf{x}} \log q(\mathbf{x}){% endmathjax %}。评分网络{% mathjax %}\mathbf{s}_\theta: \mathbb{R}^D \to \mathbb{R}^D{% endmathjax %}经过训练可以估算它，{% mathjax %}\mathbf{s}_\theta(\mathbf{x}) \approx \nabla_{\mathbf{x}} \log q(\mathbf{x}){% endmathjax %}。为了使其能够在深度学习环境中扩展到高维数据，他们建议使用去噪分数匹配（[`Vincent，2011`](http://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf)）或切片分数匹配([`Song`等人，`2019`](https://arxiv.org/abs/1905.07088))。去噪分数匹配会向数据添加预先指定的小噪声{% mathjax %}q(\tiled{\mathbf{x}}|\mathbf{x}){% endmathjax %}和{% mathjax %}q(\tilde{\mathbf{x}}){% endmathjax %}与分数匹配。回想一下，随机梯度`Langevin dynamics`可以仅使用分数从概率密度分布中采样数据点{% mathjax %}\nabla_{\mathbf{x}} \log q(\mathbf{x}){% endmathjax %}在一个迭代过程中。然而，根据流形假设，即使观察到的数据看起来只是任意高维的，大多数数据预计会集中在低维流形中。这会给分数估计带来负面影响，因为数据点无法覆盖整个空间。在数据密度较低的区域，分数估计的可靠性较低。在添加一个小的高斯噪声后，扰动的数据分布会覆盖整个空间{% mathjax %}\mathbb{R}^D{% endmathjax %}，得分估计网络的训练变得更加稳定。[`Song & Ermon,2019`](https://arxiv.org/abs/1907.05600)通过用不同级别的噪声扰动数据对其进行了改进，并训练了一个噪声条件下的得分网络来联合估计不同噪声水平下所有扰动数据的得分。噪声水平增加的时间表类似于正向扩散过程。如果我们使用扩散过程注释，得分近似于{% mathjax %}\mathbf{s}_\theta(\mathbf{x}_t, t) \approx \nabla_{\mathbf{x}_t} \log q(\mathbf{x}_t){% endmathjax %}。给定高斯分布{% mathjax %}\mathbf{x} \sim \mathcal{N}(\mathbf{\mu}, \sigma^2 \mathbf{I}){% endmathjax %}，我们可以将其密度函数对数的导数写为{% mathjax %}\nabla_{\mathbf{x}}\log p(\mathbf{x}) = \nabla_{\mathbf{x}} \Big(-\frac{1}{2\sigma^2}(\mathbf{x} - \boldsymbol{\mu})^2 \Big) = - \frac{\mathbf{x} - \boldsymbol{\mu}}{\sigma^2} = - \frac{\boldsymbol{\epsilon}}{\sigma}{% endmathjax %}，在这里，{% mathjax %}\boldsymbol{\epsilon} \sim \mathcal{N}(\boldsymbol{0}, \mathbf{I}){% endmathjax %}。则：
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

