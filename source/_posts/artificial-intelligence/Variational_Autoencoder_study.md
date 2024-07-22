---
title: 变分自动编码器(VAE)—探析（深度学习）
date: 2024-07-22 11:00:11
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

`Autoencoder`是一种用于**无监督学习**的神经网络模型，旨在通过**压缩和重建数据**来学习数据的有效表示。`Autoencoder`由两个主要部分组成：**编码器**(`Encoder`)和**解码器**(`Decoder`)。
<!-- more -->
- **编码器**(`Encoder`)：将输入数据压缩到一个低维的潜在空间表示。编码器的输出维度通常比输入维度小，这个压缩过程可以去除输入数据中的噪声并保留关键特征。
- **解码器**(`Decoder`)：从潜在空间表示重建原始输入数据。解码器的结构通常是编码器的镜像，尝试尽可能准确地重建原始输入数据。

**工作机制**：
- 输入编码：编码器接收输入数据{% mathjax %}x{% endmathjax %}，并将其压缩到一个低维的潜在表示{% mathjax %}z{% endmathjax %}。
- 数据重建：解码器接收潜在表示{% mathjax %}z{% endmathjax %}，并尝试重建原始输入数据{% mathjax %}\hat{x}{% endmathjax %}。
- 损失函数：通过计算输入数据{% mathjax %}x{% endmathjax %}与重建数据{% mathjax %}\hat{x}{% endmathjax %}之间的**重建误差**来衡量模型的性能。常用的损失函数包括**均方误差**(`Mean Squared Error, MSE`)和**二元交叉熵**(`Binary Crossentropy`)。
{% asset_img va_1.png %}

`Autoencoder`的目标是**最小化重建误差**，其损失函数可以表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}(x,\hat{x}) = \|x-\hat{x}\|^2
{% endmathjax %}
其中{% mathjax %}x{% endmathjax %}是原始输入数据，{% mathjax %}\hat{x}{% endmathjax %}是重建数据。`Autoencoder`的种类：
- **基本自编码器**(`Autoencoder`)：最简单的自编码器结构，包含一个编码器和一个解码器。
- **变分自编码器**(`Variational Autoencoder, VAE`)：是一种特殊的`Autoencoder`，可以用于生成与训练数据相似的新的数据样本。
- **去噪自编码器**(`Denoising Autoencoder`)：通过在输入数据中添加噪声，并训练模型去除这些噪声，从而提高模型的鲁棒性。
- **稀疏自编码器**(`Sparse Autoencoder`)：通过在损失函数中添加稀疏性约束，鼓励潜在表示中的大部分节点保持为零，从而学习到更有意义的特征。
- **卷积自编码器**(`Convolutional Autoencoder`)：在编码器和解码器中使用卷积层，特别适用于图像数据的压缩和重建。

**应用场景**：
- **图像压缩与去噪**：`Autoencoder`可以用于图像压缩，减少图像的存储空间，同时尽量保留原始图像的信息。此外，去噪自编码器可以用于从噪声图像中恢复清晰图像。
{% asset_img va_2.png %}
- **异常检测**：由于`Autoencoder`可以学习数据的关键特征，它们可以用于检测异常数据。例如，在网络流量监控中，`Autoencoder`可以用于检测异常活动。
- **数据生成**：可以用于生成与训练数据相似的新的数据样本。

**优点**：
- **无监督学习**：无需标签数据，适用于大量未标注的数据集。
- **非线性特征提取**：相比于传统的降维方法（如主成分分析，`PCA`），`Autoencoder`可以通过非线性变换提取数据的复杂特征。

`Autoencoder`存在什么问题？该模型学习到的代码毫无意义。也就是说，该模型可以将任意向量分配给输入，而向量中的数字不代表任何模式。该模型不会捕获数据之间的任何语义关系。

综上所述，`Autoencoder`是一种强大的工具，用于**数据压缩、去噪、异常检测和数据生成**等任务。通过学习输入数据的有效表示，`Autoencoder`可以在许多实际应用中发挥重要作用。

#### 变分自动编码器(VAE)

变分自动编码器(`VAE`)是一种生成模型，旨在学习输入数据的概率潜在表示。与标准自编码器不同，`VAE`学习的是**潜在空间**上的分布，而不是固定的潜在表示。基本结构：VAE 的结构包括两个主要部分：**编码器**(`Encoder`)和**解码器**(`Decoder`)。
- **编码器**：将输入数据映射到**潜在空间**中的概率分布参数（均值和方差），而不是单个点。
- **解码器**：从**潜在空间**的分布中采样，然后使用这些样本重建原始输入数据。

**工作流程**：
- **输入编码**：编码器接收输入数据{% mathjax %}x{% endmathjax %}，并输出**潜在空间**的分布参数（均值{% mathjax %}\mu{% endmathjax %}和方差{% mathjax %}\sigma{% endmathjax %}）。
- **采样**：从编码器输出的分布中采样一个**潜在变量**{% mathjax %}z{% endmathjax %}。
- **解码**：将采样的潜在变量{% mathjax %}z{% endmathjax %}输入解码器，重建输入数据{% mathjax %}\hat{x}{% endmathjax %}。
- **损失函数**：VAE 的损失函数包括两个部分：1.重建误差：衡量重建数据{% mathjax %}\hat{x}{% endmathjax %}与原始数据{% mathjax %}x{% endmathjax %}之间的差异;2.KL 散度：衡量潜在分布{% mathjax %}q(z|x){% endmathjax %}与先验分布{% mathjax %}p(z){% endmathjax %}之间的差异，鼓励潜在分布接近标准正态分布。

`VAE`通过最大化证据下界(`Evidence Lower Bound, ELBO`)来训练，其目标是最大化数据的对数似然。`ELBO`的表达式为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}(\theta,\phi;x) = \mathbb{E}_{q_{\phi}(z|x)}[\log p_{\theta}(x|z)] - KL(q_{\phi}(z|x)\|p(z))
{% endmathjax %}
其中{% mathjax %}q_{\phi}(z|x){% endmathjax %}是近似后验分布，由编码器参数化。{% mathjax %}p_{\theta}(x|z){% endmathjax %}是生成分布，由解码器参数化。{% mathjax %}KL{% endmathjax %}是`Kullback-Leibler`散度，衡量两个分布之间的差异。

**优点**：
- **生成新数据**：由于`VAE`学习的是**潜在空间**的分布，可以通过采样生成与训练数据相似的新数据。
- **避免过拟合**：通过引入概率分布和正则化，`VAE`能够有效避免过拟合。

`VAE`广泛应用于图像生成、数据降维、异常检测等领域。例如，可以用`VAE`生成逼真的图像、进行复杂数据集的降维分析等。

就像你使用`Python`生成`[1,100]`之间的随机数一样，你是从`[1,100]`之间的均匀（伪）随机分布中进行采样。同样，我们可以从潜在空间中采样以生成随机向量，将其提供给解码器并生成新数据。
{% asset_img va_3.png %}

{% asset_img va_4.png %}

**库尔贝克-莱伯勒散度**：
{% asset_img va_5.png %}

以上散度不对称，值都大于等于`0`，当且仅当{% mathjax %}P = Q{% endmathjax %}时，它才等于{% mathjax %}0{% endmathjax %}。

鸡和蛋的问题：
{% asset_img va_6.png %}

{% asset_img va_7.png %}

**证据下界**(`Evidence Lower Bound, ELBO`)是**变分贝叶斯方法**中的一个关键概念，用于估计观测数据对数似然的下限。定义：在变分贝叶斯方法中，我们通常处理的是潜在变量模型。在这种模型中，假设观测数据{% mathjax %}X{% endmathjax %}和潜在变量{% mathjax %}Z{% endmathjax %}具有联合分布{% mathjax %}p(X,Z;\theta){% endmathjax %}，其中{% mathjax %}\theta{% endmathjax %}是模型参数。由于我们只观测到数据{% mathjax %}X{% endmathjax %}，而潜在变量{% mathjax %}Z{% endmathjax %}是未观测到的（即潜在的），我们需要估计{% mathjax %}\theta{% endmathjax %}并计算后验分布{% mathjax %}P(Z|X;\theta){% endmathjax %}。

{% asset_img va_8.png %}

{% asset_img va_9.png %}

最大化`ELBO`：**估算器**。
- 当我们想要最大化一个函数时，我们通常会采用梯度，并调整模型的权重，使它们沿着梯度方向移动。
- 当我们想要最小化一个函数时，我们通常会采用梯度，并调整模型的权重，它们沿着梯度的相反方向移动。

`Stochastic Gradient Descent`(`SGD`)是一种用于优化目标函数的迭代方法，广泛应用于机器学习和深度学习中。与传统的**批量梯度下降**(`Batch Gradient Descent`)相比，`SGD`在每次迭代中只使用一个或一小部分样本来计算梯度，从而显著降低计算成本。
{% mathjax '{"conversion":{"em":14}}' %}
w:= w - \eta \nabla Q(w) = w - \frac{\eta}{n}\sum_{i=1}^n \nabla Q_i(w)
{% endmathjax %}
其中{% mathjax %}w{% endmathjax %}是模型参数，{% mathjax %}\eta{% endmathjax %}是学习率，{% mathjax %}\nabla Q_i(w){% endmathjax %}是在第{% mathjax %}i{% endmathjax %}个样本上的梯度。

**算法步骤**：
- **初始化**：随机初始化模型参数{% mathjax %}w{% endmathjax %}和学习率{% mathjax %}\eta{% endmathjax %}。
- **迭代更新**：重复以下步骤直到收敛：1.随机打乱训练数据；2.对于每个训练样本{% mathjax %}i{% endmathjax %}执行以下操作：计算梯度{% mathjax %}\nabla Q_i(w){% endmathjax %}；更新参数{% mathjax %}w - \frac{\eta}{n}\sum_{i=1}^n \nabla Q_i(w){% endmathjax %}。

**变种**：
`Mini-batch SGD`：在每次迭代中使用一小批(`mini-batch`)样本来计算梯度，而不是单个样本。这种方法在计算效率和收敛速度之间取得了平衡。
`Online SGD`：每次迭代只使用一个样本来更新参数，适用于流数据或实时数据处理。

**优点**：
**计算效率高**：由于每次迭代只使用一个或少量样本，`SGD`的计算成本较低，特别适用于大规模数据集。
**快速收敛**：在许多实际应用中，`SGD`通常比批量梯度下降收敛更快，尽管路径更加噪声。

**缺点**：
**收敛路径噪声大**：由于每次迭代只使用部分数据，更新路径较为随机，可能导致收敛到局部最优解。
**需要调参**：学习率等超参数对`SGD`的性能影响较大，需要仔细调参。

```python
import tensorflow as tf

# 创建优化器对象
sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

# 定义变量和损失函数
var = tf.Variable(2.5)
cost = lambda: 2 + var ** 2

# 进行优化
for _ in range(100):
    sgd.minimize(cost, var_list=[var])

# 输出结果
print(var.numpy())  # 输出优化后的变量值
print(cost().numpy())  # 输出优化后的损失值
```
`SGD`是一种强大的优化算法，特别适用于大规模数据集和实时数据处理。尽管其收敛路径较为随机，但通过**合适的超参数调节和变种算法**（如`Mini-batch SGD`），可以在**计算效率和收敛速度**之间取得良好的平衡。

`SGD`是随机的，因为我们从数据集中随机选择小批量，然后对小批量的损失进行平均。
{% asset_img va_10.png %}

如何最大化`ELBO`？
{% asset_img va_11.png %}

这个**估计量**(`Estimator`)是**无偏**的，这意味着即使每一步它可能不等于真实期望，平均而言它会收敛到真实期望，但由于它是随机的，它也有方差，而且对于实际使用来说它恰好很高。另外，不能通过它进行**反向传播**。**估计量**(`Estimator`)是一种用于根据观测数据计算某个未知参数估计值的规则。**估计量**本质上是数据的函数，用于推断统计模型中的未知参数。
{% asset_img va_12.png %}

在重新参数化的模型上运行反向传播：
{% asset_img va_13.png %}

一种新的**估计量**(`Estimator`)：
{% asset_img va_14.png %}

{% asset_img va_15.png %}

如何推导出损失函数？
{% asset_img va_15.png %}

编码器分布为：{% mathjax %}q(z|x)=\mathcal{N}(z|\mu(x),\Sigma(x)){% endmathjax %}，其中，{% mathjax %}\Sigma=\text{diag}(\sigma_1^2,\ldots,\sigma^2_n){% endmathjax %}。潜在先验由下式给出{% mathjax %}p(z)=\mathcal{N}(0,I){% endmathjax %}。两者都是维度为{% mathjax %}n{% endmathjax %}的多元高斯，`KL`散度为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathfrak{D}_\text{KL}[p_1\mid\mid p_2] =
\frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - n + \text{tr} \{ \Sigma_2^{-1}\Sigma_1 \} + (\mu_2 - \mu_1)^T \Sigma_2^{-1}(\mu_2 - \mu_1)\right]
{% endmathjax %}。其中{% mathjax %}p_1 = \mathcal{N}(\mu_1,\Sigma_1){% endmathjax %}和{% mathjax %}p_2 = \mathcal{N}(\mu_2,\Sigma_2){% endmathjax %}。在 `VAE`中{% mathjax %}p_1 = q(z|x){% endmathjax %}和{% mathjax %}p_2=p(z){% endmathjax %}，所以，{% mathjax %}\mu_1=\mu, \Sigma_1 = \Sigma, \mu_2=\vec{0}, \Sigma_2=I{% endmathjax %}，因此，
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathfrak{D}_\text{KL}[q(z|x)\mid\mid p(z)] 
&=
\frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - n + \text{tr} \{ \Sigma_2^{-1}\Sigma_1 \} + (\mu_2 - \mu_1)^T \Sigma_2^{-1}(\mu_2 - \mu_1)\right]\\
&= \frac{1}{2}\left[\log\frac{|I|}{|\Sigma|} - n + \text{tr} \{ I^{-1}\Sigma \} + (\vec{0} - \mu)^T I^{-1}(\vec{0} - \mu)\right]\\
&= \frac{1}{2}\left[-\log{|\Sigma|} - n + \text{tr} \{ \Sigma \} + \mu^T \mu\right]\\
&= \frac{1}{2}\left[-\log\prod_i\sigma_i^2 - n + \sum_i\sigma_i^2 + \sum_i\mu^2_i\right]\\
&= \frac{1}{2}\left[-\sum_i\log\sigma_i^2 - n + \sum_i\sigma_i^2 + \sum_i\mu^2_i\right]\\
&= \frac{1}{2}\left[-\sum_i\left(\log\sigma_i^2 + 1\right) + \sum_i\sigma_i^2 + \sum_i\mu^2_i\right]\\
\end{align}
{% endmathjax %}
