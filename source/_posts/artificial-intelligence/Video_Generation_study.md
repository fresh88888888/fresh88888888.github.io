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
在这里：
- {% mathjax %}x{% endmathjax %}是文本。
- {% mathjax %}\hat{x}{% endmathjax %}是`BPE`编码的文本。
- {% mathjax %}\text{CLIP}_{\text{text}}(\cdot){% endmathjax %}是`CLIP`文本编码器，{% mathjax %}\mathbf{x}_e\text{CLIP}_{\text{text}}(\mathbf{x}){% endmathjax %}。
- {% mathjax %}P(\cdot){% endmathjax %}是先验，生成图像嵌入{% mathjax %}\mathbf{y}_e{% endmathjax %}、给定文本嵌入{% mathjax %}\mathbf{x}_e{% endmathjax %}和`BPE`编码的文本{% mathjax %}\hat{\mathbf{x}}:\mathbf{y}_e = P(\mathbf{x}_e,\hat{\mathbf{x}}){% endmathjax %}，这部分是在文本图像对上训练的，而不是在视频数据上微调的。
- {% mathjax %}D^t(\cdot){% endmathjax %}是时空解码器，可生成`16`帧，其中每帧都是低分辨率{% mathjax %}64\times 64{% endmathjax %}RGB图像{% mathjax %}\hat{\mathbf{y}}_l{% endmathjax %}。
- {% mathjax %}\uparrow_F (\cdot){% endmathjax %}是帧差值网络，通过在生成的帧之间进行差值来提高有效帧速率。这是一个针对预测视频采样蒙版帧任务的微调模型。
- {% mathjax %}SR_h (\cdot),SR_l^t(\cdot){% endmathjax %}是空间和时间超分辨率模型，分别将图像分辨率提高到{% mathjax %}256\times 256{% endmathjax %}和{% mathjax %}768\times 768{% endmathjax %}。
- {% mathjax %}\hat{\mathbf{y}}_t{% endmathjax %}即为最终生成的视频。

时空`SR`层包含伪`3D`卷积层和伪`3D`注意力层：
- **伪3D卷积层**：每个空间`2D`卷积层（从预训练图像模型初始化）后面都有一个时间`1D`层（初始化为恒等函数）。从概念上讲，卷积`2D`层首先生成多个帧，然后将这些帧重塑为视频片段。
- **伪3D注意力层**：在每个（预训练的）空间注意层之后，堆叠一个时间注意力层，并用于近似完整的时空注意力层。

{% asset_img vg_7.png "伪3D卷积（左）和注意力层（右）的工作原理" %}

它们可以表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\text{Conv}_\text{P3D} &= \text{Conv}_\text{1D}(\text{Conv}_\text{2D}(\mathbf{h}) \circ T) \circ T \\
\text{Attn}_\text{P3D} &= \text{flatten}^{-1}(\text{Attn}_\text{1D}(\text{Attn}_\text{2D}(\text{flatten}(\mathbf{h})) \circ T) \circ T)
\end{aligned}
{% endmathjax %}
输入张量{% mathjax %}\mathbf{h} \in \mathbb{R}^{B\times C \times F \times H \times W}{% endmathjax %}（对应批次大小、通道、框架、高度和重量）；以及{% mathjax %}\circ T{% endmathjax %}在时间和空间维度之间切换；{% mathjax %}\text{flatten}(.){% endmathjax %}是矩阵运算符来转换{% mathjax %}\mathbf{h}{% endmathjax %}成为{% mathjax %}\mathbf{h}’ \in \mathbb{R}^{B \times C \times F \times HW}{% endmathjax %}和{% mathjax %}\text{flatten}^{-1}(.){% endmathjax %}逆转该过程。在训练期间，`Make-A-Video`流程的不同组件会进行独立训练。
- 解码器{% mathjax %}D^t{% endmathjax %}，事先的{% mathjax %}P{% endmathjax %}以及两个超分辨率组件{% mathjax %}\text{SR}_t,\text{SR}_l^t{% endmathjax %}首先仅对图像进行训练，没有配对的文本。
- 接下来添加新的时间层，将其初始化为身份函数，然后对未标记的视频数据进行微调。

`Tune-A-Video`([`Wu`等人，`2023`年](https://openaccess.thecvf.com/content/ICCV2023/html/Wu_Tune-A-Video_One-Shot_Tuning_of_Image_Diffusion_Models_for_Text-to-Video_Generation_ICCV_2023_paper.html))扩展了一个预先训练的图像扩散模型，以实现一次性视频调整：给定一个包含{% mathjax %}m{% endmathjax %}框架，{% mathjax %}\mathcal{V}= \{v_i|i=1,\ldots,m\}{% endmathjax %}，并附有描述性提示{% mathjax %}\tau{% endmathjax %}，任务是生成一个新的视频{% mathjax %}\mathcal{V}^*{% endmathjax %}根据微调和相关的文本提示{% mathjax %}\tau^*{% endmathjax %}。例如，{% mathjax %}\tau{% endmathjax %}="A man is skiing"可以扩展为{% mathjax %}\tau^*{% endmathjax %}= "Spiderman is skiing on the beach"。Tune-A-Video 旨在用于对象编辑、背景更改和风格转换。除了扩大`2D`卷积层之外，`Tune-A-Video`的`U-Net`架构还结合了ST-Attention（时空注意）模块，通过查询前几帧中的相关位置来捕获时间一致性。给定帧的潜在特征{% mathjax %}v_i{% endmathjax %}、前几帧{% mathjax %}v_{i-1}{% endmathjax %}以及第一帧{% mathjax %}v_1{% endmathjax %}预计查询{% mathjax %}\mathbf{Q}{% endmathjax %}，键{% mathjax %}\mathbf{K}{% endmathjax %}和值{% mathjax %}\mathbf{V}{% endmathjax %}，`ST-attention`定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
&\mathbf{Q} = \mathbf{W}^Q \mathbf{z}_{v_i}, \quad \mathbf{K} = \mathbf{W}^K [\mathbf{z}_{v_1}, \mathbf{z}_{v_{i-1}}], \quad \mathbf{V} = \mathbf{W}^V [\mathbf{z}_{v_1}, \mathbf{z}_{v_{i-1}}] \\
&\mathbf{O} = \text{softmax}\Big(\frac{\mathbf{Q} \mathbf{K}^\top}{\sqrt{d}}\Big) \cdot \mathbf{V}
\end{aligned}
{% endmathjax %}
 
{% asset_img vg_8.png "Tune-A-Video架构概览。它首先在采样阶段之前对单个视频运行轻量级微调阶段。请注意，整个时间自注意力(T-Attn)层都会得到微调，因为它们是新添加的，但只有 ST-Attn和Cross-Attn中的查询投影会在微调期间更新，以保留先前的文本到图像知识。ST-Attn提高了时空一致性，Cross-Attn改进了文本-视频对齐" %}

`Runway`的`Gen-1`模型([`Esser`等人，`2023`年](https://arxiv.org/abs/2302.03011))旨在根据文本输入编辑给定视频。它分解了对视频结构和内容的考虑{% mathjax %}p(\mathbf{x},s,c){% endmathjax %}生成条件。然而，要对这两个方面进行清晰的分解并不容易。
- 内容{% mathjax %}c{% endmathjax %}指视频的外观和语义，从文本中采样以进行条件编辑。帧的`CLIP`嵌入是内容的良好表示，并且与结构特征基本保持正交。
- 结构{% mathjax %}s{% endmathjax %}描述几何和动态，包括形状、位置、物体的时间变化，以及{% mathjax %}s{% endmathjax %}是从输入视频中采样的。可以使用深度估计或其他特定于任务的辅助信息（例如，用于人体视频合成的人体姿势或面部特征点）。

`Gen-1`中的架构变化非常标准，即在其残差块中在每个`2D`空间卷积层之后添加`1D`时间卷积层，在其注意力块中在每个`2D`空间注意力块之后添加`1D`时间注意力块。在训练过程中，结构变量{% mathjax %}s{% endmathjax %}与扩散潜变量{% mathjax %}z{% endmathjax %}连接，其中内容变量{% mathjax %}c{% endmathjax %}在交叉注意力层中提供。在推理时，剪辑嵌入通过预先转换将`CLIP`文本嵌入转换为`CLIP`图像嵌入。
{% asset_img vg_9.png "Gen-1 模型训练流程" %}

视频`LDM`([`Blattmann`等人，`2023`年](https://arxiv.org/abs/2304.08818))首先训练`LDM`（潜在扩散模型）图像生成器。然后对模型进行微调，以生成添加了时间维度的视频。微调仅适用于编码图像序列上这些新添加的时间层。时间层{% mathjax %}\{l^i_\phi \mid i = \ 1, \dots, L\}{% endmathjax %}在视频`LDM`中与现有的空间层交错{% mathjax %}l^i_{\theta}{% endmathjax %}在微调过程中保持冻结状态。也就是说，我们只微调新参数{% mathjax %}\phi{% endmathjax %}但不是预先训练的图像骨干模型参数{% mathjax %}\theta{% endmathjax %}。`Video LDM`的流水线首先以低`fps`生成关键帧，然后通过`2`步潜在帧插值来提高`fps`。长度的输入序列{% mathjax %}T{% endmathjax %}被解释为一批图像（即{% mathjax %}B\cdot T{% endmathjax %}）为基础图像模型{% mathjax %}\theta{% endmathjax %}然后重新塑造成视频格式{% mathjax %}l^i_{\theta}{% endmathjax %}时间层。有一个跳跃连接导致时间层输出的组合{% mathjax %}\mathbf{x}'{% endmathjax %}和空间输出{% mathjax %}\mathbf{z}{% endmathjax %}通过学习合并参数{% mathjax %}\alpha{% endmathjax %}实践中实现了两种类型的时间混合层:(1)时间注意力和(2)基于`3D`卷积的残差块。
{% asset_img vg_10.png "用于图像合成的预训练 LDM被扩展为视频生成器。B、T、C、H、W分别是批量大小、序列长度、通道、高度和宽度。cs是一个可选的条件/上下文框架" %}

但是，`LDM`的预训练自动编码器仍然存在一个问题，它只能看到图像而看不到视频。如果天真地使用它来生成视频，可能会导致闪烁伪影，并且没有良好的时间连贯性。因此，视频`LDM`在解码器中添加了额外的时间层，并使用由`3D`卷积构建的逐块时间鉴别器对视频数据进行微调，而编码器保持不变，以便我们仍然可以重复使用预训练的`LDM`。在时间解码器微调期间，冻结的编码器会独立处理视频中的每一帧，并使用视频感知鉴别器强制跨帧进行时间连贯的重建。
{% asset_img vg_11.png "视频潜在扩散模型中自动编码器的训练流程。解码器经过微调，具有新的跨帧鉴别器的时间一致性，而编码器保持冻结状态" %}

与视频`LDM`类似，**稳定视频扩散**([`SVD；Blattmann`等人，`2023`年](https://arxiv.org/abs/2311.15127))的架构设计也是基于`LDM`，在每个空间卷积和注意力层之后插入时间层，但`SVD`会对整个模型进行微调。训练视频`LDM`分为三个阶段：
- 文本到图像的预训练很重要，有助于提高质量和快速跟进。
- 视频预训练有利于分离，理想情况下应该在更大规模、精选的数据集上进行。
- 高质量视频微调适用于较小、预先带有高视觉保真度字幕的视频。

`SVD`特别强调了数据集管理在模型性能中的关键作用。他们应用了剪切检测管道来获取每个视频的更多剪切，然后应用了三种不同的字幕模型：(1)`CoCa`用于中间帧，(2)`V-BLIP`用于视频字幕，(3)基于前两个字幕的`LLM`字幕。然后，他们能够继续改进视频数据集，方法是删除运动较少的剪辑（通过以`2fps`计算的低光流分数进行过滤）、过多的文本存在（应用光学字符识别来识别包含大量文本的视频）或通常美学价值较低的剪辑（使用`CLIP`嵌入注释每个剪辑的第一帧、中间帧和最后一帧并计算美学分数和文本-图像相似性）。实验表明，经过过滤的更高质量的数据集可以提高模型质量，即使这个数据集小很多。首先生成远距离关键帧，然后添加具有时间超分辨率的插值，其关键挑战在于如何保持高质量的时间一致性。`Lumiere`([`Bar-Tal`等人，`2024`年](https://arxiv.org/abs/2401.12945))采用了时空`U-Net`(`STUNet`)架构，该架构通过传递一次性生成视频的整个时间，从而消除了对`TSR`（时间超分辨率）组件的依赖。`STUNet`在时间和空间维度上对视频进行下采样，因此在紧凑的时空潜在空间中进行昂贵的计算。
{% asset_img vg_12.png "Lumiere移除了TSR（时间超分辨率）模型。由于内存限制，膨胀的SSR网络只能在视频的短片段上运行，因此SSR模型在一组较短但重叠的视频片段上运行" %}

`STUNet`扩展了预训练的文本到图像的`U-net`，使其能够在时间和空间维度上对视频进行上/下采样。基于卷积的块由预训练的文本到图像层组成，后跟分解的空间时间卷积。最粗略的`U-Net`级别的基于注意力的块包含预训练的文本到图像，后跟时间注意力。进一步的训练只发生在新添加的层上。
{% asset_img vg_13.png "(a)时空U-Net(STUNet)、(b)基于卷积的块和 (c)基于注意力的块的架构" %}
##### 无需训练的适应性

令人惊讶的是，可以采用预先训练的文本转图像模型来输出视频，而无需任何训练。如果我们天真地随机采样潜在代码序列，然后构建解码后的相应图像的视频，则无法保证对象和语义在时间上的一致性。`Text2Video-Zero`([`Khachatryan`等人，`2023`年](https://arxiv.org/abs/2303.13439)) 通过增强预训练的图像扩散模型，实现零样本、无需训练的视频生成，该模型具有两种实现时间一致性的关键机制：
- 对具有运动动态的潜在代码序列进行采样，以保持全局场景和背景时间的一致性。
- 重新编程帧级自注意力，在第一帧上使用每帧的新的跨帧注意力，以保留前景对象的上下文、外观和身份。

{% asset_img vg_14.png "Text2Video-Zero管道" %}

对一系列潜在变量进行采样的过程，{% mathjax %}\mathbf{x}^1_T, \dots, \mathbf{x}^m_T{% endmathjax %}，其运动信息描述如下：
- 确定方向{% mathjax %}\boldsymbol{\delta} = (\delta_x, \delta_y) \in \mathbb{R}^2{% endmathjax %}用于控制全局场景和相机运动；默认情况下，我们设置{% mathjax %}\boldsymbol{\delta} = (1, 1){% endmathjax %}。还定义一个超参数{% mathjax %}\lambda > 0{% endmathjax %}控制整体运动量。
- 首先随机采样第一帧的潜在代码，{% mathjax %}\mathbf{x}^1_T \sim \mathcal{N}(0, I){% endmathjax %}。
- 履行{% mathjax %}\Delta t \geq 0{% endmathjax %}`DDIM`使用预先训练的图像扩散模型（例如本文中的稳定扩散(`SD`)模型）进行后向更新步骤，并获取相应的潜在代码{% mathjax %}\mathbf{x}^1_{T'}{% endmathjax %}，在这里{% mathjax %}T’ = T - \Delta t{% endmathjax %}。
