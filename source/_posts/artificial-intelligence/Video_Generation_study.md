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

`VDM`([`Ho & Salimans`等人，`2022`年](https://arxiv.org/abs/2204.03458))采用标准扩散模型设置，但采用适合视频建模的架构。它扩展了`2D U-net`以适用于`3D`数据([`Cicek`等人，`2016`年](https://arxiv.org/abs/1606.06650))，其中每个特征图为帧 `x` 高度 `x` 宽度 `x` 通道的`4D`张量。此`3D U-net`在空间和时间上进行分解，这意味着每一层仅在空间或时间维度上运行，但不能同时在两个维度上运行：
- 处理空间:1.每个旧的`2D U-net`中的`2D`卷积层都被扩展为仅空间的`3D`卷积；准确地说，`3x3`卷积变成了`1x3x3`卷积。2.每个空间注意力块仍然作为空间注意力，其中第一个轴(`frames`)被视为批次维度。
- 处理时间：每个空间注意力块后都会添加一个时间注意力块。它对第一个轴(`frames`)进行注意，并将空间轴视为批处理维度。相对位置嵌入用于跟踪帧的顺序。时间注意力块对于模型捕捉良好的时间连贯性非常重要。
{% asset_img vg_2.png "3D U-net架构" %}

`Imagen Video`([`Ho`等人，`2022`年](https://arxiv.org/abs/2210.02303))建立在一系列扩散模型之上，以提高视频生成质量，并升级到以`24 fps`的速度输出`1280x768`视频**Imagen Video**架构由以下组件组成，总共有`7`个扩散模型。
- 冻结的`T5`文本编码器提供文本嵌入作为条件输入。
- 一个基本的视频传播模型。
- 交错的空间和时间超分辨率扩散模型的级联，包括`3`个`TSR`（时间超分辨率）和`3`个`SSR`（空间超分辨率）组件。

{% asset_img vg_3.png "Imagen Video中的级联采样管道。实际上，文本嵌入被注入到所有组件中，而不仅仅是基础模型中" %}

基础去噪模型同时对所有具有共享参数的帧执行空间操作，然后时间层跨帧混合激活以更好地捕捉时间连贯性，这被发现比帧自回归方法效果更好。
{% asset_img vg_4.png "Imagen Video 扩散模型中一个时、空块可分离的架构" %}

`SSR`和`TSR`模型都是以通道连接噪声数据{% mathjax %}\mathbf{z}_t{% endmathjax %}的上采样输入为条件。`SSR`通过**可学习的双线性差值**进行上采样，而`TSR`通过重复帧或填充空白帧进行上采样。`Imagen Video`也应用了渐进式蒸馏来加速采样,每次蒸馏迭代可以将所需的采样步骤减半。他们的实验能够将所有`7`个视频扩散模型蒸馏到仅需`8`个采样步骤,而不会明显损失感知质量。为了实现更好的扩展效果，`Sora`([`Brooks`等人，`2024`年](https://openai.com/research/video-generation-models-as-world-simulators))利用`DiT`（扩散的`Tranformer`）架构，该架构对视频和图像潜在代码的时空块进行操作。视觉输入表示为时空块序列，这些时空块充当`Transformer`输入`token`。
{% asset_img vg_5.png "Sora是一个扩散Transformer的模型" %}

#### 调整图像模型来生成视频

**扩散视频建模**的另一种方法是通过插入时间层来“膨胀”预先训练的图像到文本扩散模型，然后我们可以选择仅在视频数据上微调新层，或者完全避免额外训练。新模型继承了文本-图像对的先验知识，因此它可以帮助减轻对文本-视频数据对的需求。
##### 视频数据微调

`Make-A-Video`([`Singer`等人，`2022`年](https://arxiv.org/abs/2209.14792))扩展了具有时间维度的预训练扩散图像模型，该模型由三个关键部分组成：
- 在文本-图像数据上训练的基本文本到图像模型。
- 时空卷积和注意力层扩展网络以覆盖时间维度。
- 用于高帧率生成的帧插值网络。

{% asset_img vg_6.png "Make-A-Video流程图" %}

最终的视频推理方案可以表述为：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{\mathbf{y}}_t = \text{SR}_h \circ \text{SR}^t_l \circ \uparrow_F \circ D^t \circ P \circ (\hat{\mathbf{x}}, \text{CLIP}_\text{text}(\mathbf{x}))
{% endmathjax %}
