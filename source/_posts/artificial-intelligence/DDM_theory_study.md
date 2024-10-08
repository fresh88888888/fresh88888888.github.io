---
title: 离散去噪扩散模型(DDMs) — 数据隐私探析（深度学习）
date: 2024-08-16 18:00:11
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

#### 介绍

**离散去噪扩散模型**(`Discrete Denoising Diffusion Models, DDMs`)是一种用于**生成合成数据**的**深度学习模型**，近年来因其在**隐私保护**方面的潜力而受到关注。随着对数据隐私的日益重视，研究人员开始探索这些模型在生成合成数据时的隐私保护能力。在生成合成数据的过程中，传统的隐私保护方法往往无法有效应对数据泄露的风险。**离散去噪扩散模型**通过逐步引入噪声并在后续步骤中去噪，生成与原始数据分布相似的合成数据。尽管已有实证研究评估了这些模型的性能，但对其隐私保护能力的数学表征仍存在较大缺口。
<!-- more -->

**离散去噪扩散模型**在隐私保护方面的研究为合成数据生成提供了新的视角。通过理论分析和实证验证，研究者们不仅揭示了这些模型的隐私泄露机制，还为未来的隐私保护技术提供了理论基础。这一研究方向将有助于在数据生成和使用中更好地平衡隐私保护与数据实用性之间的关系。

#### 模型

[`“On the Inherent Privacy Properties of Discrete Denoising Diffusion Models”`](https://openreview.net/pdf?id=UuU6C6CUoF)，这篇论文主要介绍了，隐私问题导致合成数据集创建的激增，而扩散模型则成为一种很有前途的技术手段。尽管先前的研究已经对这些模型进行了实际的评估，但在提供其隐私保护能力的数学表征方面仍存在差距。为了解决这个问题，作者提出了用于离散数据集生成的**离散去噪扩散模型**(`DDMs`)固有的隐私属性的开创性理论探索。作者的框架专注于每个实例的差异隐私(`pDP`)，阐明了给定训练数据集中每个数据点的潜在隐私泄露，并深入了解了每个点的隐私损失如何与数据集的分布相关联。结果表明，使用`s-sized`的数据点进行训练会导致(`DDMs`)从纯噪声阶段过渡到合成清洗数据阶段时从{% mathjax %}(\epsilon, \mathcal{O}(\frac{1}{s^2\epsilon}))\text{-pDP}{% endmathjax %}到{% mathjax %}(\epsilon, \mathcal{O}(\frac{1}{s\epsilon}))\text{-pDP}{% endmathjax %}的隐私泄漏激增，而扩散系数的更快衰减会增强隐私保证。最后，作者在合成数据集和真实数据集上进行了理论验证。

具有分类属性的离散表格或图形数据集在许多隐私敏感领域中很普遍，包括金融、电商和医学。例如，医学研究人员经常以离散表格形式收集患者数据，例如种族、性别和就医状况。然而，在这些领域使用和共享数据存在泄露个人信息的风险。为了解决这类问题，有人提出生成具有隐私保护的合成数据集，作为保护敏感信息和降低隐私泄露风险的一种方式。
{% asset_img d_1.png  "离散扩散模型(DDMs)的原理" %}

在论文中，作者分析了固定训练数据集的DDM隐私保护。利用了数据相关的隐私框架，称为**每个实例差异隐私**(`pDP`)，该框架是根据固定训练数据集中的实例定义的。`pDP`的分析允许对训练集中每个数据点的潜在隐私泄露进行细粒度的表征。这让数据管理员能够更好地了解训练数据的敏感性。作者的分析考虑了一个在`s`个样本上训练的`DDM`并生成`m`个样本，跟踪每个生成步骤中的隐私泄漏。实验证明，随着数据生成步骤从`t = T`（噪声状态）过渡到`t = 0`（无噪声状态），隐私泄漏从{% mathjax %}(\epsilon, \mathcal{O}(\frac{1}{s^2\epsilon(1-e^{}-\epsilon)}))\text{-pDP}{% endmathjax %}增加到{% mathjax %}(\epsilon, \mathcal{O}(\frac{1}{s\epsilon(1 - e^{-\epsilon})}))\text{-pDP}{% endmathjax %}，其中数据相关项隐藏在{% mathjax %}\text{big-}\mathcal{O}{% endmathjax %}符号中。因此，最后几个生成步骤主导了`DDM`中的主要隐私泄漏。此外，分析表明，当`m = 1`时，隐私边界{% mathjax %}\mathcal{O}(1/s){% endmathjax %}很紧，并强调了`DDM`固有的弱隐私保护。此外，**扩散系数衰减越快，隐私保护效果越好**。对于数据部分，作者开发了一种算法，根据`pDP`边界估计真实数据集中每个数据点的隐私泄漏。通过从数据集中删除最敏感的数据点（根据数据相关隐私参数）来训练`DDM`，然后评估基于`DDM`生成的合成数据集训练的`ML`模型，从而评估数据部分。有趣的是，作者观察到，在删除部分数据后获得的`ML`模型甚至超过没有删除此类数据的其他模型。作者将其归因于这样一个事实，即删除的数据点可能是异常值，这可能实际上不利于`ML`模型学习。

为了避免混淆，作者提供了几个重要的解释。最坏情况并与数据集无关的`DP`（`Wang，2019`）相比，针对训练集量身定制的`pDP`为数据管理员提供了对每个数据点潜在隐私泄漏的更准确、更细粒度的估计。然而，重要的是要理解`pDP`。直接为数据添加噪声是不允许的，因为添加的噪声可能会因其数据依赖性而泄露隐私信息。可以使用其他方法，例如**平滑灵敏度**（`Nissim`等人，`2007`）和**提议测试发布**（`Dwork & Lei，2009`）。作者的分析旨在深入了解 `DDM`所提供的固有隐私，并指导数据管理员评估与数据集不同部分的隐私泄露风险。这里并非以开发一种匹配特定隐私评估的算法为目标。鉴于此目的，`pDP`是比`DP`更合适。在实践中，`pDP`评估应该保密，并由数据管理员了解数据集并使用 `DDM`生成合成数据集时的潜在隐私泄露。

首先介绍一下用于分析的符号和概念。假设{% mathjax %}[n] = {1,2,\ldots,n}{% endmathjax %}，{% mathjax %}X^n{% endmathjax %}表示为一个`n`维的离散空间，每个维度有{% mathjax %}k{% endmathjax %}个类别，即{% mathjax %}X^n:= X^1\times\ldots\times X^n{% endmathjax %}，其中{% mathjax %}X^i = [k], i\in [n]{% endmathjax %}，假设训练数据集{% mathjax %}V{% endmathjax %}位于{% mathjax %}X^n{% endmathjax %}中，意味着样本是{% mathjax %}n{% endmathjax %}个元素的矢量值数据，每个元素属于{% mathjax %}k{% endmathjax %}个类别之一。假设每个列的类别一致，但分析可以使用最大类别计数来解释具有不同类别计数的数据集。

**基于实例的差分隐私**：`DP`是**量化隐私泄露**的事实标准。作者针对特定的相邻数据集调整了`DP`定义，引入了基于实例的`DP`：让{% mathjax %}V_0{% endmathjax %}作为一个训练数据集，{% mathjax %}\mathbf{v}^{\ast}\in V_0{% endmathjax %}为不动点并且{% mathjax %}M{% endmathjax %}为随机机制。定义相邻数据集{% mathjax %}V_1 = V_0\setminus\{\mathbf{v}^{\ast}\}{% endmathjax %}。如果对于所有测试集{% mathjax %}O\subset \text{range}(M),\{i,j\} = \{0,1\}{% endmathjax %}，则称{% mathjax %}M{% endmathjax %}满足关于{% mathjax %}(V_0,V^{\ast}){% endmathjax %}的{% mathjax %}(\epsilon,\delta)\text{-pDP}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
P(M(V_i)\in \mathcal{O}) \leq e^{\epsilon} P(M(V_j)\in \mathcal{O}) + \delta
{% endmathjax %}
需要强调的是，`pDP`是针对特定数据集-数据点对唯一的定义。

**离散扩散模型**(`DDMs`)：是可以生成分类数据的扩散模型。让{% mathjax %}v_t{% endmathjax %}表示时间{% mathjax %}t{% endmathjax %}时的数据随机变量。前向处理过程涉及使用噪声马尔可夫链逐{% mathjax %}q{% endmathjax %}渐破坏数据，记录{% mathjax %}q(\mathbf{v}_{1:T}|\mathbf{v}_0) = \prod^T_{t=1}q(\mathbf{v}_t|\mathbf{v}_{t-1}){% endmathjax %}，其中{% mathjax %}\mathbf{v}_{1:T} = \mathbf{v}_1,\mathbf{v}_2,\ldots,\mathbf{v}_T{% endmathjax %}。另一方面，反向处理过程，{% mathjax %}p_{\phi}(\mathbf{v}_{0:T}) = p(\mathbf{v}_T)\prod^T_{t=1}p_{\phi}(\mathbf{v}_{t-1}|\mathbf{v}_t){% endmathjax %}，从先前的数据集{% mathjax %}p(\mathbf{v}_T){% endmathjax %}开始重建新的数据集。去噪神经网络通过优化`ELBO`学习{% mathjax %}p_{\phi}(\mathbf{v}_{t-1}|\mathbf{v}_t){% endmathjax %}，其中包括了三个损失项：重建项({% mathjax %}L_r{% endmathjax %})、前项({% mathjax %}L_p{% endmathjax %})和去噪项({% mathjax %}L_t{% endmathjax %})，如下等式表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\underbrace{\mathbb{E}_{q(\mathbf{v}_1|\mathbf{v}_0)}[\log p_{\phi}(\mathbf{v}_0)|\mathbf{v}_1]}_{\text{Reconstruction Term }L_r} - \underbrace{D_{KL}(q(\mathbf{v}_T|\mathbf{v}_0)\parallel p_{\phi}(\mathbf{v}_T))}_{\text{Prior Term }L_p} - \sum^T_{t=2}\underbrace{\mathbb{E}_{q(\mathbf{v}_t|\mathbf{v}_0)[D_{KL}(q(\mathbf{v}_{t-1}| \mathbf{v}_t,\mathbf{v}_0)\parallel p_{\phi}(\mathbf{v}_{t-1}|\mathbf{v}_t))]}}_{\text{Denoising Term }L_t}
{% endmathjax %}
具体来说，前向处理过程由一批**转换核**来描述{% mathjax %}\{Q^i_t\}_{t\in [T],i\in [n]}{% endmathjax %}，其中任何元素{% mathjax %}\mathbf{v}^i,[Q^i_t]_{lh} = q(\mathbf{v}^i_t = h|\mathbf{v}^i_{t-1} = l){% endmathjax %}表示在时间{% mathjax %}t{% endmathjax %}时，第{% mathjax %}i{% endmathjax %}个元素从类别{% mathjax %}l{% endmathjax %}跳转到类别{% mathjax %}h{% endmathjax %}概率。对于每个实体{% mathjax %}i{% endmathjax %}，类别数目是相同的，我们可以在所有的维度上使用相同的转换核并且用{% mathjax %}Q_t{% endmathjax %}替代{% mathjax %}Q^i_t{% endmathjax %}，让{% mathjax %}\bar{Q}_t = Q_1Q_2\ldots Q_t{% endmathjax %}表示从时间1到时间{% mathjax %}t{% endmathjax %}的**累积转换矩阵**，我们用一个均匀先验分布{% mathjax %}p(\mathbf{v}_T){% endmathjax %}。双随机矩阵由一批扩散系数({% mathjax %}\{\alpha_t,t\in [T]|\alpha_t \in (0,1)\}{% endmathjax %})的参数所决定，这些参数控制从原始分布到均匀测度的转化率。具体而言，定义{% mathjax %} Q_t = \alpha_tI + (1 - \alpha_t)\frac{\mathbb{1}\mathbb{1}^T}{k}{% endmathjax %}，然后{% mathjax %}\bar{Q}_t = \bar{\alpha}_tI + (1 - \bar{\alpha}_t)\frac{\mathbb{1}\mathbb{1}^T}{k}{% endmathjax %}，其中{% mathjax %}\bar{\alpha}_t = \prod^t_{i=1}\alpha_t{% endmathjax %}。在反向处理过程中，利用去噪网络预测{% mathjax %}p_{\phi}(\mathbf{v}_{t-1}|\mathbf{v}_t){% endmathjax %}，期望逼近{% mathjax %}q()\mathbf{v}_{t-1}|\mathbf{v}_t,\mathbf{v}_0{% endmathjax %}，在实践中，去噪网络不是直接预测{% mathjax %}p_{\phi}(\mathbf{v}_{t-1}|\mathbf{v}_t){% endmathjax %}，而是通过噪声网络学习预测以噪声{% mathjax %}\mathbf{v}_t{% endmathjax %}作为输入在时间0时的干净数据{% mathjax %}\mathbf{v}_0{% endmathjax %}，为了训练去噪网络，需要从{% mathjax %}q(\mathbf{v}_t|\mathbf{v}_0){% endmathjax %}噪声点处采样并且将他们输入到去噪网络{% mathjax %}\phi_t{% endmathjax %}，得到{% mathjax %}p_{\phi}(\mathbf{v}_0|\mathbf{v}_t){% endmathjax %}，具体来说，我们采用：
{% mathjax '{"conversion":{"em":14}}' %}
L_{\text{train}} = D_{KL}(q(\mathbf{v}_0|\mathbf{v}_t)\parallel p_{\phi}(\mathbf{v}_0|\mathbf{v}_t)) = \frac{1}{|v|}\sum_{\mathbf{v}_0\in v} \mathbb{E}_{\mathbf{v}_t\sim q(\mathbf{v}_t|\mathbf{v}_0)}[\sum_{i=1}^n L_{CE}(\mathbf{v}_0^i,p_{\phi}(\mathbf{v}_0^i|\mathbf{v}_t))]
{% endmathjax %}
这个损失作为我们后边充分训练的一个基础，在训练的过程中，我需要桥接{% mathjax %}p_{\phi}(\mathbf{v}_{t-1}|\mathbf{v}_t){% endmathjax %}和{% mathjax %}p_{\phi}(\mathbf{v}_{0}|\mathbf{v}_t){% endmathjax %}的连接。实际取决于维度无关方面的条件。
{% mathjax '{"conversion":{"em":14}}' %}
p_{\phi}(\mathbf{v}_{t-1}|\mathbf{v}_t) = \prod_{i\in [n]} p_{\phi}(\mathbf{v}_{t-1}^i|\mathbf{v}_t) = \prod_{i\in [n]}\sum_{l\in x_i}q(\mathbf{v}_{t-1}^i|\mathbf{v}_t,\mathbf{v}_0^i = l)p_{\phi}(\mathbf{v}_{0}^i = l|\mathbf{v}_t)
{% endmathjax %}
其他符号，给定两个样本{% mathjax %}\mathbf{v}{% endmathjax %}和{% mathjax %}\tilde{\mathbf{v}}{% endmathjax %},让{% mathjax %}\bar{\omega}(\mathbf{v},\tilde{\mathbf{v}}){% endmathjax %}表示不同实体的计数。{% mathjax %}\bar{\omega}(\mathbf{v},\tilde{\mathbf{v}}) = \#\{i|\mathbf{v}^i \neq \tilde{\mathbf{v}}^i,i\in [n]\}{% endmathjax %}，{% mathjax %}\eta\in [n]{% endmathjax %}并且{% mathjax %}\mathbf{v}\in v_1{% endmathjax %}，定义{% mathjax %}N_{\eta}(\mathbf{v}) = |\{\mathbf{v}'\in v_1 : \bar{\omega}(\mathbf{v},\mathbf{v}')\leq \eta\}|{% endmathjax %}并且{% mathjax %}v_1^{i|l} = \{\mathbf{v}\in v_1|\mathbf{v}^1 = l\}{% endmathjax %}具有固定值匀速的数据点集。我们用{% mathjax %}\mathcal{D}_{KL}(\cdot\parallel\cdot){% endmathjax %}和{% mathjax %}\parallel\cdot\parallel_{TV}{% endmathjax %}作为`KL`散度和总变化。让{% mathjax %}\mu^{+}_t = \frac{1 + (k-1)\alpha_t}{k}{% endmathjax %}和{% mathjax %}\mu_t^{-}\frac{1-\alpha_t}{k}{% endmathjax %}表示分别在时间{% mathjax %}t{% endmathjax %}对于相同和不同状态下**的一步转换概率**，{% mathjax %}\bar{\mu}^{+}_t = \frac{1 + (k-1)\bar{\alpha}_t}{k}{% endmathjax %}和{% mathjax %}\bar{\mu}^{-}_t = \frac{1 -\bar{\alpha}_t}{k}{% endmathjax %}是累积的转换概率。转换概率比定义为{% mathjax %}R_t = \frac{\mu^{+}_t}{\mu^{-}_t}{% endmathjax %}和{% mathjax %}\bar{R}_t = \frac{\bar{\mu}^{+}_t}{\bar{\mu}^{-}_t}{% endmathjax %}，表示在扩散过程中，比率越大保持相同特征类别的可能性越高，定义{% mathjax %}(\cdot)_{+} = \max\{\cdot , 0\}{% endmathjax %}。

#### 主要结果

##### DDMs的固有隐私保护

首先，定义下面的分析机制。让{% mathjax %}\mathcal{M}_t(\mathcal{V};m){% endmathjax %}表示为机制，{% mathjax %}\mathcal{V}{% endmathjax %}作为一个输入数据集，它使用`DDM`的生成过程在时间{% mathjax %}t{% endmathjax %}时输出{% mathjax %}m{% endmathjax %}个样本。具体来说，在论文中{% mathjax %}\mathcal{M}_0(\mathcal{V};m){% endmathjax %}表示`DDM`最终生成的数据集。下面是一些假设的概述：
- 假设一：给定数据集{% mathjax %}\mathcal{V}{% endmathjax %}，让{% mathjax %}v_0{% endmathjax %}表示时间`0`处的预测随机变量，让{% mathjax %}\phi{% endmathjax %}表示在数据集{% mathjax %}\mathcal{V}{% endmathjax %}上训练的**去噪神经网络**(`NNs`)。如果存在小的常量{% mathjax %}\gamma t > 0{% endmathjax %}使得{% mathjax %}\forall{% endmathjax %}，则假设一是成立的。
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{D}_{KL}(q(\mathbf{v}_0^i|\mathbf{v}_t)\|p_{\phi}(\mathbf{v}_0^i|\mathbf{v}_t)) \leq \gamma_t, \forall i\in [n],\forall t\in [T]
{% endmathjax %}
- 假设二（前向和反向扩散路径之间的间隙）： 给定数据集{% mathjax %}\mathcal{V}{% endmathjax %}，让{% mathjax %}v_t{% endmathjax %}表示前向和反向处理过程在时间{% mathjax %}t{% endmathjax %}处的中间分布采样的随机变量。如果存在{% mathjax %}\tilde{\gamma}_t \ll 2{% endmathjax %}的正常数，则假设二成立。
{% mathjax '{"conversion":{"em":14}}' %}
\|q(\mathbf{v}_t) - p_{\phi}(\mathbf{v}_t)\|_{TV} \leq \tilde{\gamma}_t, \forall t \in [T]
{% endmathjax %}

假设一指出，当使用第一个公式中的损失函数训练去噪网络时，它可以有效地从中间噪声数据分布中推断出干净的数据。给定一个好的模型，估计{% mathjax %}\gamma_t{% endmathjax %}会很小。假设二的扩散和生成路径很接近，这是一个合理的假设。然而，不能使用第三个公式直接推导隐私界限，因为变化中的接近性并没有隐含`DP`。基于上述假设，作者研究了隐私泄露沿产生过程的流动情况。作者的分析主要围绕在特定训练下的固有隐私保护由`DDM`生成的样本，表示为{% mathjax %}T_{rl}{% endmathjax %}。
- 定理一（`DDMs`固有的`pDP`保护）：给定数据集{% mathjax %}\mathcal{V}_0{% endmathjax %},大小为{% mathjax %}|\mathcal{V}_0| = s + 1{% endmathjax %}和要保护的数据点{% mathjax %}v^{\ast}\in \mathcal{V}_0{% endmathjax %}，表示{% mathjax %}\mathcal{V}_1{% endmathjax %}，正如{% mathjax %}\mathcal{V}_1 = \mathcal{V}_0\setminus \{\mathbf{v}^{\ast}\}{% endmathjax %}。假设在{% mathjax %}\mathcal{V}_0{% endmathjax %}和{% mathjax %}\mathcal{V}_1{% endmathjax %}上训练的去噪网络满足于假设一和假设二。给定一个具体的时间步{% mathjax %}T_{rl}{% endmathjax %}，机制{% mathjax %}\mathcal{M}_{T_{rl}}(\cdot;m){% endmathjax %}相对于{% mathjax %}(\mathcal{V}_0,\mathbf{v}^{\ast}){% endmathjax %},给定{% mathjax %}\epsilon{% endmathjax %}满足{% mathjax %}(\epsilon,\delta)\text{-pDP}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\delta(\mathcal{V}_0,\mathbf{v}^{\ast}) \leq m[\underbrace{\sum_{t = T_{rl}}^T\min\big\{\frac{4N_{(1+c^{\ast}_t)\eta_t(\mathbf{v}^{\ast})}}{s},1 \big\}\cdot \frac{n}{s^{\psi_t}} + \frac{n(1 - \frac{1}{\bar{R}_{t-1}})}{s^2}}_{\text{Main Privacy Term}} + \underbrace{\mathcal{O}(\sqrt{\gamma t} + \tilde{\gamma}_t)}_{\text{Error Term}}]/(\epsilon(1 - e^{-\epsilon}))
{% endmathjax %}

其中{% mathjax %}\psi_t,\eta_t,c^{\ast}_t{% endmathjax %}是由{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}和{% mathjax %}\mathcal{V}_1{% endmathjax %}决定的**数据相关量**（通常用于统计学和数据分析中，指的是依赖于特定数据集的**量度**或**统计指标**）。定义**相似性度量**{% mathjax %}\text{Sim}(\mathbf{v}^{\ast}, \mathcal{V}) = \sum_{\mathbf{v}\in \mathcal{V}}\bar{R}_t^{-\bar{\omega}(\mathbf{v},\mathbf{v}^{\ast})}{% endmathjax %}。则{% mathjax %}\psi_t,\eta_t,c^{\ast}_t{% endmathjax %}如下：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{n}{s^{\psi_t}} = \frac{(\bar{\alpha}_{t-1} - \bar{\alpha}_t)/(k\bar{\mu}^+_t\bar{\mu}^-_t)}{1 + \text{Sim}(\mathbf{v}^{\ast},\mathcal{V}_1)}\cdot \sum_{i=1}^n\log\Big(1 + \frac{\bar{R}^2_{t-1} - 1}{\bar{R}^2_{t-1}\text{Sim}(\mathbf{v}^{\ast},\mathcal{V}_1^{i|\mathbf{v}^{\ast i}}) + \text{Sim}(\mathbf{v}^{\ast},\mathcal{V_1}) + 1}\Big)
{% endmathjax %}
并且{% mathjax %}c^{\ast}_t{% endmathjax %}满足{% mathjax %} c^{\ast}_t \in \{0,\frac{1}{\eta_t},\frac{2}{\eta_t},\ldots,\frac{n-\eta_t}{\eta_t}\}{% endmathjax %}，{% mathjax %}\eta_t \in \{1,2,\ldots,n\} {% endmathjax %}，则{% mathjax %}c^{\ast}_t{% endmathjax %}是最小的。其中{% mathjax %}\vartheta(\eta) = (s - N_{\eta}(\mathbf{v}^{\ast}))/N_{\eta}(\mathbf{v}^{\ast}){% endmathjax %}表示{% mathjax %}\eta{\text{-ball}}{% endmathjax %}内部和外部之间点数的比例。

定理一量化了训练集{% mathjax %}\mathcal{V}_0{% endmathjax %}中特定点{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}的隐私泄露。隐私界限包括一个主要隐私项，它代表`DDMs`固有的`pDP`保护，突出了界限的数据依赖性，以及一个去噪网络训练和路径差异的误差项。这些**数据相关量**很复杂，无法对**数据集-数据点对**进行严格的测量。接下来，将进一步解释这些量。首先，由于生成过程形成**马尔可夫链**，其中转移概率{% mathjax %}p_{\phi}(\mathbf{v}^{(t-1)}|\mathbf{v}^{(t)}){% endmathjax %}是从训练中学习的，因此每个生成步骤都会从训练数据集中泄露一些信息。可以证明，大多数情况泄漏如下：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbb{E}_{\mathbf{v}\sim p_{\phi}(\mathbf{v}_{t|0} = \mathbf{v})}d^{(t)}(\mathbf{v})
{% endmathjax %}
其中{% mathjax %}\mathbf{v}_{t|\lambda}{% endmathjax %}表示扩散模型在数据集{% mathjax %}\mathcal{V}_{\lambda}{% endmathjax %}训练时，生成过程在时间{% mathjax %}t{% endmathjax %}处产生数据的随机变量。{% mathjax %}\lambda \in \{0,1\}{% endmathjax %}和{% mathjax %}d^{(t)}(\mathbf{v}) = \sum_{\lambda\in \{0,1\}}\mathcal{D}_{KL}(p_{\phi}(\mathbf{v}_{t-1|\lambda}|\mathbf{v}_{t|\lambda} = \mathbf{v})\|p_{\phi}(\mathbf{v}_{t-1|\bar{\lambda}}|\mathbf{v}_{t|\bar{\lambda}} = \mathbf{v})){% endmathjax %}，其中{% mathjax %}\bar{\lambda} = 1 - \lambda{% endmathjax %}，它表征了通过学习扩散模型表征的两个条件分布之间的对称距离。本质上，这三个数据相关量{% mathjax %}\psi_t,\eta_t,c^{\ast}_t{% endmathjax %}约束了第十个公式。
- {% mathjax %}\psi_t{% endmathjax %}的数量：如下图所示，{% mathjax %}\frac{n}{s^{\psi t}}{% endmathjax %}量化了{% mathjax %}\max_{\mathbf{v}}d^{(t)}(\mathbf{v}){% endmathjax %}，其中最大值在被移除的点{% mathjax %}\mathbf{v} = \mathbf{v}^{\ast}{% endmathjax %}。通过仔细检查发现{% mathjax %}\psi_t{% endmathjax %}依赖于{% mathjax %}\text{Sim}(\mathbf{v}^{\ast},\mathcal{V}_1){% endmathjax %}和{% mathjax %}\text{Sim}(\mathbf{v}^{\ast},\mathcal{V}_1^{i|\mathbf{v}^{\ast i}}){% endmathjax %}。根据{% mathjax %}\bar{\omega}{% endmathjax %}的定义，让{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}和{% mathjax %}\mathcal{V}_1{% endmathjax %}的剩余点进行对齐。
- {% mathjax %}\psi_t{% endmathjax %}的变化：在生成阶段，{% mathjax %}t{% endmathjax %}从{% mathjax %}T{% endmathjax %}减少到{% mathjax %}1{% endmathjax %}，{% mathjax %}\frac{1}{s^{\psi_t}}{% endmathjax %}从{% mathjax %}\mathcal{O}_s(\frac{1}{s^2}){% endmathjax %}增长到{% mathjax %}\mathcal{O}_s(1){% endmathjax %}。随着数据生成过程从噪声演变成无噪声状态，潜在的隐私泄露风险会升级。

{% asset_img d_2.png "数据相关量图示" %}

****

{% asset_img d_3.png "数据集相似度与pDP泄露之间的相关性图示" %}

- {% mathjax %}\eta_t{% endmathjax %}和{% mathjax %}c_t^{\ast}{% endmathjax %}：很明显，生成的中间度量{% mathjax %}p_{\phi}(\mathbf{v}_{t|0}){% endmathjax %}偏离了最敏感点({% mathjax %}\delta_{\mathbf{v} = \mathbf{v}^{\ast}}{% endmathjax %})的狄拉克测度。因此，以{% mathjax %}d^{(t)}(\mathbf{v}){% endmathjax %}为特征的实际隐私泄露对度量值{% mathjax %}p_{\phi}(\mathbf{v}_t|0){% endmathjax %}取均值远小于其最大值。为了提供此类的严格表征，引入两个量{% mathjax %}\eta_t{% endmathjax %}和{% mathjax %}c^{\ast}_t{% endmathjax %}来定义以脆弱点{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}为中心的局部区域{% mathjax %}\mathcal{S} = \{\mathbf{v}'\in \mathcal{X}^n:\bar{\omega}(\mathbf{v},\mathbf{v}')\leq (1 + c^{\ast}_t)\eta_t\}{% endmathjax %}。其中，隐私泄露可以被限制在{% mathjax %}p_{\phi}(\mathbf{v}_{t|0}\in \mathcal{S})\max_{\mathbf{v}\in \mathcal{S}}d^{(t)}(\mathbf{v}){% endmathjax %}与{% mathjax %}p_{\phi}(\mathbf{v}_{t|0}\in \mathcal{S}){% endmathjax %}和{% mathjax %}p_{\phi}(\mathbf{v}_{t|0}\notin \mathcal{S})\max_{\mathbf{v}\notin \mathcal{S}}d^{(t)}(\mathbf{v}){% endmathjax %}与{% mathjax %}\max_{\mathbf{v}\notin \mathcal{S}}d^{(t)}(\mathbf{v}){% endmathjax %}之和的范围内。

##### DDM系数和数据集分布对隐私界限的影响

- **扩散系数的影响**：隐私项在很大程度上受{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}和{% mathjax %}\mathcal{V}_1{% endmathjax %}之间**邻近性**的影响。随着时间{% mathjax %}t{% endmathjax %}的推移，这种相似性由转换比率{% mathjax %}\bar{R}_t{% endmathjax %}决定，扩散系数趋于零的速度越快，该比率就越高，从而提高了隐私保护。
- **数据集分布的影响**：作者发现{% mathjax %}\psi_t{% endmathjax %}对隐私边界有很大影响。{% mathjax %}\psi_t{% endmathjax %}受附加点{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}与{% mathjax %}\mathcal{V}_0\setminus \{\mathbf{v}^{\ast}\}{% endmathjax %}中其余点的相似性影响，如果{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}变小（大），相应项{% mathjax %}s^{-\psi_t}{% endmathjax %}变大（小），这表明{% mathjax %}\mathbf{v}^{\ast}{% endmathjax %}的保护较弱。{% mathjax %}\text{Sim}(\mathcal{V}_0\setminus \{\mathbf{v}^{\ast}\},\mathbf{v}^{\ast},t){% endmathjax %}特别低的点可能是数据中的敏感点。

##### 在简单分布下表征数据依赖量

作者考虑从某些特定分布中抽样的训练数据集，进一步说明数据相关的量化。考虑一个分布，其中每一列以概率{% mathjax %}p(p\geq \frac{1}{k}){% endmathjax %}独立取值{% mathjax %}l\in [k]{% endmathjax %}，而任何其他{% mathjax %}k-1{% endmathjax %}个类别以概率{% mathjax %}\frac{1-p}{k-1}{% endmathjax %}在所有{% mathjax %}n{% endmathjax %}列上取非多数类别({% mathjax %}(\mathbf{v}^{\ast})^i\neq l{% endmathjax %})（称为非多数点，因此往往具有更高的隐私泄露），并且{% mathjax %}\mathcal{V}_0\setminus \{\mathbf{v}^{\ast}\}{% endmathjax %}中的其余点是从分布中采样的。我们有以下特征：

具有足够大的{% mathjax %}s{% endmathjax %}({% mathjax %}\frac{1}{s^{\psi_t}}{% endmathjax %})，则有更高的概率，{% mathjax %}\frac{1}{s^{\psi_t - 2}}\rightarrow \frac{(\bar{\alpha}_{t-1} - \bar{\alpha}_t)/(k\bar{\mu}^{+}_t)\\bar{\mu^{-}_t}}{\bar{R}^2_{t-1}\cdot \tau^{2n-1}_t\cdot \frac{1-p}{k-1}} + \tau^{2n}_t{% endmathjax %}，其中{% mathjax %}\tau_t:=\frac{1-p}{k-1} + \frac{\bar{\mu}^{-}_t}{\bar{\mu}^{+}_t}(1 - \frac{1-p}{k-1}){% endmathjax %}，在噪声区域，{% mathjax %}\tau_t\rightarrow 1,\frac{1}{s^{\psi_t}} = \mathcal{O}_s()\frac{1}{s^2}{% endmathjax %}。对于偏度较大的分布，即{% mathjax %}p{% endmathjax %}较大，{% mathjax %}\tau_t{% endmathjax %}较小，{% mathjax %}\frac{1}{s^{\psi_t}}{% endmathjax %}较大。下图与上述结论完全吻合。对于足够大的{% mathjax %}s{% endmathjax %}({% mathjax %}\eta_t,c^{\ast}_t{% endmathjax %})，{% mathjax %}\eta_t{% endmathjax %}和{% mathjax %}c^{\ast}_t{% endmathjax %}满足公式的充分条件。
{% mathjax '{"conversion":{"em":14}}' %}
\eta_t \geq n - \Big( \frac{n-\log(s\sqrt{\frac{\bar{\alpha}_{t-1} - \bar{\alpha_t}}{k\bar{\mu}_t^{+}\bar{\mu}_t^{-}}}/\log\frac{\bar{\mu}_t^{+}}{\bar{\mu}_t^{+}})}{2\log\frac{k-1}{1-p}/\log(\max\{\frac{1}{n\bar{\mu}_t^{-}},1 \}) + 1} \Big)_{+},\;\;\; c^{\ast} \geq \frac{\frac{n-\eta_t}{\eta_t}\log\frac{k-1}{1-p} - \log\frac{1}{2e}}{\log\frac{k-1}{1-p} + \log\frac{1}{e\bar{\mu}_t^{-}}}
{% endmathjax %}
在无噪声状态下{% mathjax %}(\alpha_t\rightarrow 1),\eta_t\rightarrow 0{% endmathjax %}，而在有噪声状态下{% mathjax %}(\alpha_t\rightarrow 0), \eta_t\rightarrow n{% endmathjax %}。从无噪声状态到有噪声状态，{% mathjax %}\bar{\mu}_t{% endmathjax %}增加，{% mathjax %}c^{\ast}_t\rightarrow \frac{n-\eta_t}{\eta_t}{% endmathjax %}此外，随着分布的偏度{% mathjax %}(p){% endmathjax %}的增加，`R.H.S`单调增加，导致{% mathjax %}\eta_t{% endmathjax %}和{% mathjax %}c_t^{\ast}{% endmathjax %}的值更大。下图（中间）与上述结论相符。
{% asset_img d_4.png %}

##### 在给定数据集上评估以下公式中隐私界限的算法

{% mathjax '{"conversion":{"em":14}}' %}
\delta(\mathcal{V}_0,\mathbf{v}^{\ast}) \leq m[\underbrace{\sum_{t = T_{rl}}^T\min\big\{\frac{4N_{(1+c^{\ast}_t)\eta_t(\mathbf{v}^{\ast})}}{s},1 \big\}\cdot \frac{n}{s^{\psi_t}} + \frac{n(1 - \frac{1}{\bar{R}_{t-1}})}{s^2}}_{\text{Main Privacy Term}} + \underbrace{\mathcal{O}(\sqrt{\gamma t} + \tilde{\gamma}_t)}_{\text{Error Term}}]/(\epsilon(1 - e^{-\epsilon}))
{% endmathjax %}

在实际情况下，当数据策展人发布合成数据时，评估在特定数据集上训练隐私保护措施至关重要。确保合成数据隐私和训练数据敏感信息的私密性。为此，我们引入了算法1（与算法2配对），以**计算隐私界限**，从而能够针对给定特定训练集的`DDM`生成的数据集计算每个实例的隐私泄漏。
{% asset_img d_5.png %}

****

{% asset_img d_6.png %}

#### 结论

作者分析了`DDM`生成的**合成数据集的数据相关隐私约束**，结果显示`DDM`的隐私保护能力较弱。为了满足实际需求，可能需要结合其他隐私保护技术，例如`DP-SGD`（`Abadi`等人，`2016`年）和`PATE`（`Papernot`等人，`2016`年）。作者对合成数据集和真实数据集的观察结果非常吻合。

