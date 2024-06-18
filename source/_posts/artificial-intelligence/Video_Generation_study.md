---
title: 视频生成的扩散模型（深度学习）
date: 2024-06-18 14:50:11
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

过去几年，扩散模型在图像合成方面取得了显著成果。现在，研究界开始研究一项更艰巨的任务——将其用于视频生成。这项任务本身是图像情况的超集，因为图像是`1`帧的视频，而且它更具挑战性，因为：
- 它对时间上跨帧的时间一致性有额外的要求，这自然要求将更多的世界知识编码到模型中。
- 相比于文本或图像，收集大量高质量、高维的视频数据更加困难。
<!-- more -->

#### 视频生成建模

##### 参数化和采样

让{% mathjax %}\mathbf{x}\sim q_{\text{real}}{% endmathjax %}是从真实数据分布中采样的数据点。现在我们在时间上添加少量的高斯噪声，从而创建一系列噪声变化{% mathjax %}\mathbf{x}{% endmathjax %}，表示为{% mathjax %}\{\mathbf{z}_t|t=1,\ldots,T\}{% endmathjax %}，随着噪声量增加而t{% mathjax %}t{% endmathjax %}增加，最后{% mathjax %}q(\mathbf{z}_T)\sim \mathcal{N}(\mathbf{0},\mathbf{I}){% endmathjax %}。加噪前向过程为高斯过程。设{% mathjax %} {% endmathjax %}定义高斯过程的可微噪声过程。设{% mathjax %}\alpha_t,\sigma_t{% endmathjax %}定义高斯过程的可微噪声过程：
{% mathjax '{"conversion":{"em":14}}' %}
q(\mathbf{z}_t|\mathbf{x})= \mathcal{N}(\mathbf{z}_t;\alpha_t\mathbf{x},\sigma^2_t\mathbf{I})
{% endmathjax %}
代入{% mathjax %}q(\mathbf{z}_t|\mathbf{z}_s){% endmathjax %},满足{% mathjax %}0\leq s < t\leq T{% endmathjax %}，有：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\mathbf{z}_t &= \alpha_t \mathbf{x} + \sigma_t\boldsymbol{\epsilon}_t \\
\mathbf{z}_s &= \alpha_s \mathbf{x} + \sigma_s\boldsymbol{\epsilon}_s \\
\mathbf{z}_t &= \alpha_t \Big(\frac{\mathbf{z}_s - \sigma_s\boldsymbol{\epsilon}_s}{\alpha_s}\Big) + \sigma_t\boldsymbol{\epsilon}_t \\
\mathbf{z}_t &= \frac{\alpha_t}{\alpha_s}\mathbf{z}_s + \sigma_t\boldsymbol{\epsilon}_t - \frac{\alpha_t\sigma_s}{\alpha_s} \boldsymbol{\epsilon}_s \\
\text{Thus }q(\mathbf{z}_t \vert \mathbf{z}_s) &= \mathcal{N}\Big(\mathbf{z}_t; \frac{\alpha_t}{\alpha_s}\mathbf{z}_s, \big(1 - \frac{\alpha^2_t\sigma^2_s}{\sigma^2_t\alpha^2_s}\big)\sigma^2_t \mathbf{I}\Big)
\end{aligned}
{% endmathjax %}
假设对数信噪比为{% mathjax %}\lambda_t = \log[\alpha^2_t/\sigma^2_t]{% endmathjax %}，我们可以将`DDIM`[`Song 2020`](https://arxiv.org/abs/2010.02502)表示为：
{% mathjax '{"conversion":{"em":14}}' %}
q(\mathbf{z}_t \vert \mathbf{z}_s) = \mathcal{N}\Big(\mathbf{z}_t; \frac{\alpha_t}{\alpha_s}\mathbf{z}_s, \sigma^2_{t\vert s} \mathbf{I}\Big) \quad
\text{where }\sigma^2_{t\vert s} = (1 - e^{\lambda_t - \lambda_s})\sigma^2_t
{% endmathjax %}
有一个特殊的{% mathjax %}\mathbf{v}{% endmathjax %}`-prediction`({% mathjax %}\mathbf{v}=\alpha_t\epsilon - \sigma_t\mathbf{x}{% endmathjax %})参数化由[`Salimans & Ho (2022)`](https://arxiv.org/abs/2202.00512)提出，与其他方法({% mathjax %}\epsilon{% endmathjax %}`-parameterization`)相比，它已被证明有助于避免视频生成中的颜色偏移。这个{% mathjax %}\mathbf{v}{% endmathjax %}`-parameterization`是通过角坐标的一个技巧得出的。首先，我们定义{% mathjax %}\phi_t = \text{arctan}(\sigma_t/\alpha_t){% endmathjax %}，因此我们有{% mathjax %}\alpha_{\phi} = \cos(\phi),\sigma_t = \sin\phi,\mathbf{z}_{\phi} = \cos\phi\mathbf{x} + \sin\phi\epsilon{% endmathjax %}。{% mathjax %}\mathbf{z}_{\phi}{% endmathjax %}可以写成：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{v}_\phi = \nabla_\phi \mathbf{z}_\phi = \frac{d\cos\phi}{d\phi} \mathbf{x} + \frac{d\sin\phi}{d\phi}\boldsymbol{\epsilon} = \cos\phi\boldsymbol{\epsilon} -\sin\phi\mathbf{x}
{% endmathjax %}
然后我们可以推断出：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\sin\phi\mathbf{x} 
&= \cos\phi\boldsymbol{\epsilon}  - \mathbf{v}_\phi \\
&= \frac{\cos\phi}{\sin\phi}\big(\mathbf{z}_\phi - \cos\phi\mathbf{x}\big) - \mathbf{v}_\phi \\
\sin^2\phi\mathbf{x} 
&= \cos\phi\mathbf{z}_\phi - \cos^2\phi\mathbf{x} - \mathbf{v}_\phi \\
\mathbf{x} &= \cos\phi\mathbf{z}_\phi - \sin\phi\mathbf{v}_\phi \\
\text{Similarly }
\boldsymbol{\epsilon} &= \sin\phi\mathbf{z}_\phi + \cos\phi \mathbf{v}_\phi
\end{aligned}
{% endmathjax %}
`DDIM`被更新为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\mathbf{z}_{\phi_s} 
&= \cos\phi_s\hat{\mathbf{x}}_\theta(\mathbf{z}_{\phi_t}) + \sin\phi_s\hat{\epsilon}_\theta(\mathbf{z}_{\phi_t}) \quad\quad{\small \text{; }\hat{\mathbf{x}}_\theta(.), \hat{\epsilon}_\theta(.)\text{ are two models to predict }\mathbf{x}, \boldsymbol{\epsilon}\text{ based on }\mathbf{z}_{\phi_t}}\\
&= \cos\phi_s \big( \cos\phi_t \mathbf{z}_{\phi_t} - \sin\phi_t \hat{\mathbf{v}}_\theta(\mathbf{z}_{\phi_t} ) \big) +
\sin\phi_s \big( \sin\phi_t \mathbf{z}_{\phi_t} + \cos\phi_t \hat{\mathbf{v}}_\theta(\mathbf{z}_{\phi_t} ) \big) \\
&= {\color{red} \big( \cos\phi_s\cos\phi_t + \sin\phi_s\sin\phi_t \big)} \mathbf{z}_{\phi_t} + 
{\color{green} \big( \sin\phi_s \cos\phi_t - \cos\phi_s \sin\phi_t \big)} \hat{\mathbf{v}}_\theta(\mathbf{z}_{\phi_t} ) \\
&= {\color{red} cos(\phi_s - \phi_t)} \mathbf{z}_{\phi_t} +
{\color{green} \sin(\phi_s - \phi_t)} \hat{\mathbf{v}}_\theta(\mathbf{z}_{\phi_t}) \quad\quad{\small \text{; trigonometric identity functions.}}
\end{aligned}
{% endmathjax %}
{% asset_img vg_1.png "在角坐标系中可视化扩散更新步骤的工作原理" %}

在这里{% mathjax %}\mathbf{v}{% endmathjax %}`-parameterization`是为了预测{% mathjax %}\mathbf{v}_\phi = \cos\phi\boldsymbol{\epsilon} -\sin\phi\mathbf{x} = \alpha_t\boldsymbol{\epsilon} - \sigma_t\mathbf{x}{% endmathjax %}在视频生成的情况下，我们需要扩散模型运行多步上采样来延长视频长度或提高帧速率。这需要采样第二个视频的能力{% mathjax %}\mathbf{x}^b{% endmathjax %}以第一条件为条件{% mathjax %}\mathbf{x}^a{% endmathjax %}，{% mathjax %}\mathbf{x}^b\sim p_{\theta}(\mathbf{x}^b|\mathbf{x}^a){% endmathjax %}，在这里{% mathjax %}\mathbf{x}^b{% endmathjax %}可能是自回归扩展{% mathjax %}\mathbf{x}^a{% endmathjax %}或者是视频中间缺失的帧{% mathjax %}\mathbf{x}^a{% endmathjax %}以低帧率。采样{% mathjax %}\mathbf{x}_b{% endmathjax %}需要条件{% mathjax %}\mathbf{x}_a{% endmathjax %}除了其自身对应的噪声变量之外。**视频扩散模型**([`VDM；Ho & Salimans 2022`](https://arxiv.org/abs/2204.03458))提出了使用调整后的去噪模型的重建引导方法，使得{% mathjax %}\mathbf{x}^b{% endmathjax %}可以适当调节{% mathjax %}\mathbf{x}^a{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\mathbb{E}_q [\mathbf{x}_b \vert \mathbf{z}_t, \mathbf{x}^a] &= \mathbb{E}_q [\mathbf{x}^b \vert \mathbf{z}_t] + \frac{\sigma_t^2}{\alpha_t} \nabla_{\mathbf{z}^b_t} \log q(\mathbf{x}^a \vert \mathbf{z}_t) \\
q(\mathbf{x}^a \vert \mathbf{z}_t) &\approx \mathcal{N}\big[\hat{\mathbf{x}}^a_\theta (\mathbf{z}_t), \frac{\sigma_t^2}{\alpha_t^2}\mathbf{I}\big] & {\small \text{; the closed form is unknown.}}\\
\tilde{\mathbf{x}}^b_\theta (\mathbf{z}_t) &= \hat{\mathbf{x}}^b_\theta (\mathbf{z}_t) - \frac{w_r \alpha_t}{2} \nabla_{\mathbf{z}_t^b} \| \mathbf{x}^a - \hat{\mathbf{x}}^a_\theta (\mathbf{z}_t) \|^2_2 & {\small \text{; an adjusted denoising model for }\mathbf{x}^b}
\end{aligned}
{% endmathjax %}
在这里{% mathjax %}\hat{\mathbf{x}}^a_\theta (\mathbf{z}_t), \hat{\mathbf{x}}^b_\theta (\mathbf{z}_t){% endmathjax %}是重建{% mathjax %}\mathbf{x}^a,\mathbf{x}^b{% endmathjax %}由去噪模型提供。并且{% mathjax %}\omega_r{% endmathjax %}是一个权重因子，而且很大{% mathjax %}\omega_r > 1{% endmathjax %}可以提高样本质量。请注意，也可以同时对低分辨率视频进行条件处理，以使用相同的重建引导方法将样本扩展至高分辨率。

##### 模型架构：3D U-Net和DiT

与文本到图像的扩散模型类似，`U-net`和`Transformer`仍然是两种常见的架构选择。谷歌基于`U-net`架构发表了一系列扩散视频建模论文，而`OpenAI`最近提出的`Sora`模型则利用了`Transformer`架构。
