---
title: 机器学习(ML)(十) — 探析
date: 2024-10-13 16:02:11
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

#### 因子分析(Factor Analysis)

**因子分析**(`Factor Analysis`)是一种统计方法，旨在通过识别潜在的**变量**（因子）来解释观测变量之间的相关性。它广泛应用于心理学、社会科学、市场研究和其他领域，以简化数据结构、减少维度和发现潜在的关系。**因子分析**(`Factor Analysis`)原理是将多个**观测变量**归结为少数几个潜在因子。这一过程通常包括以下步骤：1.**数据收集与准备**,收集相关的数据集，确保数据的质量和适用性；2.**相关矩阵计算**，计算观测变量之间的相关矩阵，以了解变量之间的关系；3.**因子提取**，使用统计方法（如**主成分分析**或**最大似然估计**）提取因子；**因子旋转**，为了使因子更易于解释，通常会对提取的因子进行旋转。旋转方法包括**正交旋转**（如`Varimax`）和**斜交旋转**（如`Promax`）；**因子解释**，根据因子的载荷（即每个观测变量与因子的关系）来解释每个因子的含义；**模型评估**，通过各种统计指标（如`KMO`检验和`Bartlett`球形检验）评估模型的适用性和有效性。
<!-- more -->

因子分析的优势：
- **数据简化**：通过减少变量数量，因子分析可以帮助研究者更好地理解数据结构，降低复杂性。
- **发现潜在结构**：能够识别出影响多个观测变量的潜在因素，从而揭示数据中的隐藏模式。
- **提高模型性能**：在构建预测模型时，使用因子分析提取的因子可以提高模型的准确性和可解释性。

在无监督学习中，有一个数据集{% mathjax %}X = \{x_1,x_2,\ldots,x_n\}{% endmathjax %}，这里如何使用数学方法描述这个数据集？一种方式是将数据集{% mathjax %}X{% endmathjax %}表示为：{% mathjax %}x_i = Wh_i + \mu + \epsilon{% endmathjax %}。其中{% mathjax %}h_i{% endmathjax %}称为潜在变量(因为它无法被观察到)，{% mathjax %}\epsilon{% endmathjax %}被看做均值为0，协方差为高斯分布的噪声项({% mathjax %}\epsilon \sim \mathcal{N}(0,\Psi){% endmathjax %})，{% mathjax %}\mu{% endmathjax %}是某个任意的迁移向量。如果这时的{% mathjax %}h_i{% endmathjax %}给定时，则上述方程的概率为：{% mathjax %}P(x_i|h_i) = \mathcal{N}(Wh_i + \mu, \Psi){% endmathjax %}，对于上述概率模型，我们还需要潜在变量的先验分布{% mathjax %}h{% endmathjax %}最直接的假设（基于**高斯分布**的良好特性）是{% mathjax %}h \sim \mathcal{N}(0,\mathbf{I}){% endmathjax %}。这是一个**高斯分布**，因为{% mathjax %}x{% endmathjax %}为：{% mathjax %}P(x) = \mathcal{N}(\mu, WW^T + \Psi){% endmathjax %}

下面的是一个带旋转的**因子分析**(`Factor Analysis`)示例，这里使用鸢尾花数据集，我们发现萼片长度、花瓣长度和宽度与高度相关。矩阵分解技术可以揭示这些潜在模式。对成分进行**旋转**并不会提高潜在空间的预测值，但可以在**因子分析**(`Factor Analysis`)中，通过旋转方法（如`Varimax`旋转），可以优化因子的解释性，以便更清晰地识别出哪些变量与特定因子相关联。在这种情况下，第二个成分的加载值仅在萼片宽度上为正值，从而简化模型并提高可解释性。
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis

# 加载鸢尾花数据
data = load_iris()
X = StandardScaler().fit_transform(data["data"])
feature_names = data["feature_names"]

# 绘制鸢尾花特征的协方差图
ax = plt.axes()
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(list(feature_names), rotation=90)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(list(feature_names))

im = ax.imshow(np.corrcoef(X.T), cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im).ax.set_ylabel("$r$", rotation=0)
ax.set_title("Iris feature correlation matrix")
plt.tight_layout()

# 使用Varimax旋转进行因子分析
n_comps = 2
methods = [
    ("PCA", PCA()),
    ("Unrotated FA", FactorAnalysis()),
    ("Varimax FA", FactorAnalysis(rotation="varimax")),
]
fig, axes = plt.subplots(ncols=len(methods), figsize=(10, 8), sharey=True)

for ax, (method, fa) in zip(axes, methods):
    fa.set_params(n_components=n_comps)
    fa.fit(X)

    components = fa.components_.T
    print("\n\n %s :\n" % method)
    print(components)

    vmax = np.abs(components).max()
    ax.imshow(components, cmap="RdBu_r", vmax=vmax, vmin=-vmax)
    ax.set_yticks(np.arange(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.set_title(str(method))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Comp. 1", "Comp. 2"])

fig.suptitle("Factors")
plt.tight_layout()
plt.show()
```
输出结果为：
```bash
 PCA :
[[ 0.52106591  0.37741762]
 [-0.26934744  0.92329566]
 [ 0.5804131   0.02449161]
 [ 0.56485654  0.06694199]]

 Unrotated FA :
[[ 0.88096009 -0.4472869 ]
 [-0.41691605 -0.55390036]
 [ 0.99918858  0.01915283]
 [ 0.96228895  0.05840206]]

 Varimax FA :
[[ 0.98633022 -0.05752333]
 [-0.16052385 -0.67443065]
 [ 0.90809432  0.41726413]
 [ 0.85857475  0.43847489]]

```
{% asset_img ml_1.png %}

**概率主成分分析**(`PPCA`)是一种基于**概率模型**的降维技术，旨在通过潜在变量模型来分析数据。与传统的**主成分分析**(`PCA`)不同，**概率主成分分析**(`PPCA`)不仅关注数据的线性结构，还考虑了数据中的**噪声**和**缺失值**。**概率主成分分析**(`PPCA`)的原理：其通过引入潜在变量来建模观测数据。其基本模型可以表示为：{% mathjax %}x_n = \mathbf{W}z_n+ \mu + \varepsilon_n{% endmathjax %}。其中{% mathjax %}x_n{% endmathjax %}是观测数据点，{% mathjax %}\mathbf{W}{% endmathjax %}是因子载荷矩阵，表示潜在因子与观测变量之间的关系。{% mathjax %}z_n{% endmathjax %}是潜在变量，通常假设其服从标准正态分布。{% mathjax %}\mu{% endmathjax %}是均值向量，{% mathjax %}\varepsilon_n{% endmathjax %}是噪声项，假设其服从正态分布。通过这种方式，**概率主成分分析**(`PPCA`)能够对每个数据点的分布进行建模：{% mathjax %}x_n \sim \mathcal{N}(\mu,WW^T + \sigma^2\mathbf{I}){% endmathjax %}。其中{% mathjax %}\sigma^2{% endmathjax %}表示**噪声**的方差。

**概率主成分分析**(`PPCA`)的优势：
- **处理缺失值**：**概率主成分分析**(`PPCA`)能够有效处理数据中的缺失值，而传统的**主成分分析**(`PCA`)无法做到这一点。这使得**概率主成分分析**(`PPCA`)在实际应用中更加灵活。
- **概率模型**：由于**概率主成分分析**(`PPCA`)基于**概率模型**，可以利用**最大似然估计**(`MLE`)来估计参数，并进行统计推断。
- **期望最大化算法**（`EM`算法）：**概率主成分分析**(`PPCA`)通常使用`EM`算法进行参数估计。`EM`算法通过迭代优化潜在变量和模型参数，提高了估计的准确性。

**概率主成分分析**(`PPCA`)提供了一种更为灵活和强大的降维方法，通过引入**概率模型**和潜在变量，它能够有效处理复杂的数据结构和缺失值问题。

#### 独立成分分析(ICA)

**独立成分分析**(`ICA`)是一种统计和计算技术，用于从多维信号中提取出相互独立的成分。**独立成分分析**(`ICA`)特别适用于处理混合信号，尤其是在信号处理、图像处理和生物医学工程等领域。**独立成分分析**(`ICA`)的原理是通过假设观测信号是若干个独立源信号的线性组合，来恢复这些源信号。其基本模型可以表示为：{% mathjax %}x = \mathbf{A}s{% endmathjax %}。其中{% mathjax %}x{% endmathjax %}是观测信号向量（例如多个传感器的输出），{% mathjax %}\mathbf{A}{% endmathjax %}是**混合矩阵**，表示源信号之间的线性组合。{% mathjax %}s{% endmathjax %}是独立源信号向量。**独立成分分析**(`ICA`)的目标是估计混合矩阵{% mathjax %}\mathbf{A}{% endmathjax %}和源信号{% mathjax %}s{% endmathjax %}，使得提取出的**成分**尽可能独立。常用的**独立成分分析**(`ICA`)算法包括：`FastICA`，一种迭代算法，通过**最大化非高斯性**来估计独立成分，速度较快且易于实现；`Infomax`，基于**信息论**的方法，通过最大化输出信息量来估计独立成分。**独立成分分析**(`ICA`)是一种强大的工具，通过假设观测信号是多个独立源的线性组合，能够有效地从复杂数据中提取出有意义的信息。**独立成分分析**(`ICA`)不是用于**降低维度**的，而是用于**分离叠加信号**。由于**独立成分分析**(`ICA`)模型不包含噪声项，因此为了使模型正确，必须应用白化。

下面有一个例子是利用**独立成分分析**(`ICA`)来估测噪声的来源。假设有`3`种乐器同时在演奏，`3`个麦克风录制混合信号。**独立成分分析**(`ICA`)则用于恢复混合信号的来源，即每种乐器演奏的内容。因为相关信号(乐器)反映了非高斯过程，所以**主成分分析**(`PCA`)无法恢复这些乐器信号。
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.decomposition import PCA, FastICA

np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # Add noise

S /= S.std(axis=0)  # Standardize data
# Mix data
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # Mixing matrix
X = np.dot(S, A.T)  # Generate observations

# Compute ICA
ica = FastICA(n_components=3, whiten="arbitrary-variance")
S_ = ica.fit_transform(X)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix

# We can `prove` that the ICA model applies by reverting the unmixing.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# For comparison, compute PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # Reconstruct signals based on orthogonal components

plt.figure()
models = [X, S, S_, H]
names = ["Observations (mixed signal)","True Sources","ICA recovered signals","PCA recovered signals",]
colors = ["red", "steelblue", "orange"]

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()
```
{% asset_img ml_2.png %}

#### 非负矩阵分解(NMF)

**非负矩阵分解**(`NMF`)是一种用于**降维**和**特征提取**的**矩阵分解技术**，广泛应用于信号处理、图像处理和文本挖掘等领域。**非负矩阵分解**(`NMF`)的主要特点是要求矩阵的所有元素均为非负值，这使得其在许多实际应用中更具可解释性。**非负矩阵分解**(`NMF`)的原理是将一个非负矩阵{% mathjax %}V{% endmathjax %}分解为两个非负矩阵{% mathjax %}W{% endmathjax %}和{% mathjax %}H{% endmathjax %}的乘积，使得：{% mathjax %}V\approx WH{% endmathjax %}。其中{% mathjax %}V{% endmathjax %}是原始数据矩阵，大小为{% mathjax %}m\times n{% endmathjax %}，{% mathjax %}W{% endmathjax %}是基矩阵，大小为{% mathjax %}m\times r{% endmathjax %}，表示基础特征。{% mathjax %}H{% endmathjax %}是系数矩阵，大小为{% mathjax %}r\times n{% endmathjax %}，表示每个样本在基础特征上的权重。

在**非负矩阵分解**(`NMF`)中，通常使用**弗罗贝纽斯范数**(`Frobenius norm`)来衡量**重构误差**。其目标函数可以表示为：{% mathjax %}\underset{W,H}{\min}\|V - WH\|_F^2{% endmathjax %}，其中{% mathjax %}\|\cdot\|_F{% endmathjax %}表示**弗罗贝纽斯范数**(`Frobenius norm`)，定义为：{% mathjax %}\|A\|_F = \sqrt{\sum\limits_{i=1}^m\sum\limits_{j = 1}^n a^2_{ij}}{% endmathjax %}，因此，**非负矩阵分解**(`NMF`)的优化目标可以简化为：{% mathjax %}\underset{W,H}{\min} \sum\limits_{i=1}^m\sum\limits_{j = 1}^n (v_{ij} - (WH)_{ij})^2{% endmathjax %}。为了求解上述优化问题，可以使用多种算法，其中最常见的是：**乘法更新法**，通过迭代更新{% mathjax %}W{% endmathjax %}和{% mathjax %}H{% endmathjax %}来**最小化重构误差**。更新规则如下：对于基矩阵{% mathjax %}W{% endmathjax %}的更新，则{% mathjax %}W \gets W\odot \frac{(VH^T)}{WHH^T + \epsilon}{% endmathjax %}；对于系数矩阵{% mathjax %}H{% endmathjax %}的更新，则{% mathjax %}H\gets H\odot \frac{W^T V}{W^T WH + \epsilon} {% endmathjax %}，其中{% mathjax %}\odot{% endmathjax %}表示元素逐个相乘，{% mathjax %}\epsilon{% endmathjax %}是一个小常数，用于避免除以`0`；**交替最小二乘法**，通过交替固定一个矩阵来优化另一个矩阵，以求解最优分解。使用**弗罗贝纽斯范数**(`Frobenius norm`)的**非负矩阵分解**(`NMF`)是一种有效的**降维**和**特征提取**方法，通过**最小化重构误差**，它能够提供高质量的结果。与`PCA`有所不同，向量的表示是通过叠加分量而不是减法方式获得的，而是以加法的方式获得。此类**加法模型**对于表示图像和文本非常有效。

**β散度**(`Beta-Divergence`)的**非负矩阵分解**(`NMF`)是一种扩展的**非负矩阵分解**(`NMF`)方法，旨在通过更灵活的**损失函数**来捕捉数据的特性。**β散度**提供了一种统一的框架，可以用于不同类型的数据分布，从而增强了**非负矩阵分解**(`NMF`)在各种应用中的适用性。**β散度**是一种衡量两个概率分布之间差异的度量，定义为：{% mathjax %}D_{\beta}(V\|WH) = \frac{1}{\beta(\beta - 1)} \sum\limits_{i=1}^m\sum\limits_{j = 1}^n\ (v_{ij}^{\beta} - (wh)_{ij}^{\beta} - \beta(v_{ij} - (wh)_{ij})){% endmathjax %}，其中{% mathjax %}V{% endmathjax %}是原始数据矩阵，{% mathjax %}W{% endmathjax %}和{% mathjax %}H{% endmathjax %}是非负矩阵分解的结果，{% mathjax %}\beta{% endmathjax %}是一个参数，控制损失函数的形式。{% mathjax %}\beta{% endmathjax %}参数的不同取值：
- 当{% mathjax %}\beta = 2{% endmathjax %}时，{% mathjax %}\beta{% endmathjax %}散度对应于最小化**弗罗贝纽斯范数**(`Frobenius norm`)，适用于**高斯分布**。
- 当{% mathjax %}\beta = 1{% endmathjax %}时，{% mathjax %}\beta{% endmathjax %}散度对应于最小化**绝对误差**，适用于**拉普拉斯分布**。
- 当{% mathjax %}\beta = 0{% endmathjax %}时，{% mathjax %}\beta{% endmathjax %}散度对应于**相对熵**（`Kullback-Leibler`散度），适用于**泊松分布**。

在**非负矩阵分解**(`NMF`)中，通过最小化{% mathjax %}\beta{% endmathjax %}散度来求解问题，可以表示为：{% mathjax %}\underset{W,H}{\min}D_{\beta}(V| WH){% endmathjax %}。

**小批量非负矩阵分解**(`Mini-batch NMF`)是一种改进的**非负矩阵分解**方法，旨在提高大规模数据集上**非负矩阵分解**(`NMF`)的效率和可扩展性。通过使用小批量数据进行更新，这种方法能够显著减少计算时间和内存消耗，同时保持分解的质量。**小批量非负矩阵分解**(`Mini-batch NMF`)的目标是将一个**非负矩阵**{% mathjax %}V{% endmathjax %}分解为两个**非负矩阵**{% mathjax %}W{% endmathjax %}和{% mathjax %}H{% endmathjax %}的乘积：{% mathjax %}V\approx WH{% endmathjax %}，在**小批量非负矩阵分解**(`Mini-batch NMF`)中，整个数据集被划分为多个小批量，每个小批量包含一定数量的数据样本。每次迭代仅使用一个小批量来更新**基矩阵**{% mathjax %}W{% endmathjax %}和**系数矩阵**{% mathjax %}H{% endmathjax %}。**小批量非负矩阵分解**(`Mini-batch NMF`)的执行步骤为：
- **数据分批**：将**原始数据矩阵**{% mathjax %}V{% endmathjax %}划分为多个小批量，每个小批量包含{% mathjax %}b{% endmathjax %}个样本。
- **初始化**：随机初始化**基矩阵**{% mathjax %}W{% endmathjax %}和**系数矩阵**{% mathjax %}H{% endmathjax %}，确保它们的元素均为非负值。
- **迭代更新**：对于每个小批量，进行以下步骤：1.计算当前小批量的**重构误差**；2.使用**乘法更新法**或其他优化算法更新**基矩阵**{% mathjax %}W{% endmathjax %}和**系数矩阵**{% mathjax %}H{% endmathjax %}。更新规则为：对于基矩阵{% mathjax %}W{% endmathjax %}的更新，则{% mathjax %}W \gets W\odot \frac{(VH^T)}{WHH^T + \epsilon}{% endmathjax %}；对于系数矩阵{% mathjax %}H{% endmathjax %}的更新，则{% mathjax %}H\gets H\odot \frac{W^T V}{W^T WH + \epsilon} {% endmathjax %}。
- **迭代直到收敛**：重复上述步骤，直到达到预定的迭代次数或满足收敛条件。

**小批量非负矩阵分解**(`Mini-batch NMF`)是一种高效且灵活的`NMF`方法，通过结合小批量处理技术，它不仅提高了训练速度，还增强了模型的**可扩展性**和**鲁棒性**。

#### 潜在狄利克雷分配(LDA)

**潜在狄利克雷分配**(`LDA`)是一种广泛使用的主题模型，旨在从文档集合中发现潜在主题。**潜在狄利克雷分配**(`LDA`)通过假设每个文档是由多个主题生成的，而每个主题又是由多个词构成的，从而实现对文本数据的建模和分析。**潜在狄利克雷分配**(`LDA`)广泛应用于多个领域，包括文本挖掘、推荐系统、社交网络分析等领域。**潜在狄利克雷分配**(`LDA`)的核心思想是将文档视为主题的混合，每个主题由一组词以不同的概率分布生成。其基本模型可以用以下步骤描述：**主题分布**，为每个文档生成一个主题分布，假设遵循**狄利克雷分布**。具体来说，对于文档{% mathjax %}d{% endmathjax %}，生成一个主题分布{% mathjax %}\theta_d{% endmathjax %}：{% mathjax %}\theta_d\sim \text{Dirichlet}(\alpha){% endmathjax %}，其中{% mathjax %}\alpha{% endmathjax %}是超参数，控制主题分布的**稀疏性**；**生成词**，对于文档中的每个词：从主题分布{% mathjax %}\theta_d{% endmathjax %}中抽取一个主题{% mathjax %}z_{d,n}{% endmathjax %}，则{% mathjax %}z_{d,n}\sim \text{Multinomial}(\theta_d){% endmathjax %}；从该主题的词分布中抽取一个词{% mathjax %}w_{d,n}{% endmathjax %}，则{% mathjax %}w_{d,n}\sim \text{Multinomial}(\phi_{z_{d,n}}){% endmathjax %}，其中{% mathjax %}\phi_k{% endmathjax %}是第{% mathjax %}k{% endmathjax %}个主题的词分布，通常假设遵循**狄利克雷分布**；**模型参数**，**潜在狄利克雷分配**(`LDA`)模型包括**超参数**{% mathjax %}\alpha{% endmathjax %}和每个主题的**词分布参数**{% mathjax %}\beta{% endmathjax %}。

**潜在狄利克雷分配**(`LDA`)的推断过程旨在估计给定文档集合下的隐含主题和相关参数。常用的方法包括：**变分推断**(`Variational Inference`)，通过优化变分下界来**近似后验分布**；**吉布斯采样**(`Gibbs Sampling`)，一种**马尔可夫链蒙特卡罗**(`MCMC`)方法，通过**迭代采样**来估计模型参数。**潜在狄利克雷分配**(`LDA`)的优势包括：
- **可解释性**：**潜在狄利克雷分配**(`LDA`)提供了清晰的主题表示，使得每个主题由一组相关词构成，便于理解和分析。
- **灵活性**：能够处理大规模文本数据，并适用于多种文本挖掘任务，如文档聚类、推荐系统和信息检索。
- **无监督学习**：**潜在狄利克雷分配**(`LDA`)是一种**无监督学习**方法，不需要预先标注的数据即可发现潜在结构。

**潜在狄利克雷分配**(`LDA`)是一种强大的主题建模工具，通过假设文档由多个潜在主题生成，它能够有效地从大规模文本数据中提取有意义的信息。随着自然语言处理和机器学习的发展，**潜在狄利克雷分配**(`LDA`)在许多实际应用中展现了其重要价值，尤其是在需要理解和分析文本内容时。
{% asset_img ml_3.png %}

#### 限制玻尔兹曼机(RBM)

**限制玻尔兹曼机**(`RBM`)是一种**无监督学习算法**，属于深度学习中的生成模型。**限制玻尔兹曼机**(`RBM`)由一组**可见单元**(`visible units`)和一组**隐藏单元**(`hidden units`)组成，二者之间通过权重连接，但同一层的单元之间没有连接。**限制玻尔兹曼机**(`RBM`)广泛应用于多个领域，包括：推荐系统、图像处理、自然语言处理。**限制玻尔兹曼机**(`RBM`)的结构可以用以下几个要素来描述：
**可见层**(`Visible Layer`)：表示输入数据的特征，通常与观测数据直接对应。每个**可见单元**表示一个特征或输入值。
**隐藏层**(`Hidden Layer`)：用于捕捉输入数据中的潜在特征。**隐藏单元**通过与**可见单元**的连接来学习数据的**隐含表示**。
**权重矩阵**(`Weight Matrix`)：连接**可见层**和**隐藏层**的**权重矩阵**{% mathjax %}W{% endmathjax %}，用于调整**可见单元**和**隐藏单元**之间的关系。
**偏置项**(`Bias Terms`)：每个**可见单元**和**隐藏单元**都有一个**偏置项**，用于调整**激活值**。

**限制玻尔兹曼机**(`RBM`)的工作原理是通过对输入数据进行概率建模来学习特征，其主要步骤如下：
- **前向传播**：给定可见单元的输入{% mathjax %}v{% endmathjax %}，计算**隐藏单元**的激活概率：{% mathjax %}P(h_j=1|v) = \sigma (b_j + \sum\limits_i v_i w_{ij}){% endmathjax %}。其中{% mathjax %}\sigma{% endmathjax %}是**激活函数**(`sigmoid`)，{% mathjax %}b_j{% endmathjax %}是**隐藏单元**的偏置，{% mathjax %}w_{ij}{% endmathjax %}是连接**可见单元**和**隐藏单元**的权重。
- **采样**：根据计算出的概率，从**隐藏单元**中采样得到激活状态{% mathjax %}h{% endmathjax %}。
- **重构**：使用**隐藏单元**的状态重构可见层：{% mathjax %}P(v_i|h) = \sigma (c_i + \sum\limits_j h_j w_{ij}){% endmathjax %}。其中{% mathjax %}c_i{% endmathjax %}是可见单元的偏置。
- **对比散度**(`Contrastive Divergence`)：通过**对比散度算法**更新**权重**和**偏置**，以最小化输入数据和重构数据之间的差异。这一过程通常涉及多个迭代步骤。

**限制玻尔兹曼机**(`RBM`)的优势：
- **无监督学习**：**限制玻尔兹曼机**(`RBM`)能够在没有标签的数据上进行训练，适用于大规模未标注数据集。
- **特征学习**：通过**隐含层**捕捉数据中的潜在结构，使得**限制玻尔兹曼机**(`RBM`)能够自动提取有用特征。
- **生成能力**：**限制玻尔兹曼机**(`RBM`)不仅可以用于特征提取，还可以生成新样本，具有良好的生成模型性能。

**限制玻尔兹曼机**(`RBM`)是一种强大的**无监督学习**工具，通过结合**可见层**和**隐藏层**，它能够有效地捕捉数据中的潜在结构。

下面有一个示例，利用**伯努利限制玻尔兹曼机模型**对灰度图像数据提取非线性特征，例如手写数字识别，为了从小数据集中学习到好的潜在表示，可以通过在每个方向上以`1`个像素的线性移位扰动训练数据来生成更多标记数据：
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
from sklearn.base import clone
from scipy.ndimage import convolve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

def nudge_dataset(X, Y):
    """
    This produces a dataset 5 times bigger than the original one,
    by moving the 8x8 images in X around by 1px to left, right, down, up
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    X = np.concatenate([X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors])
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, "float32")
X, Y = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# 利用伯努利限制玻尔兹曼机模型特征提取器和分类器构建分类管道
logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)
rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# 整个模型的超参数（学习率、隐藏层大小、正则化）通过网格搜索进行了优化，训练
# Hyper-parameters. These were set by cross-validation, using a GridSearchCV. 
# Here we are not performing cross-validation to save time.
rbm.learning_rate = 0.06
rbm.n_iter = 10
rbm.n_components = 100
logistic.C = 6000

# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(X_train, Y_train)

# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.0
raw_pixel_classifier.fit(X_train, Y_train)

# 评估
Y_pred = rbm_features_classifier.predict(X_test)
print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report(Y_test, Y_pred)))

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())

plt.suptitle("100 components extracted by RBM", fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
plt.show()
```
输出评估结果为：
```bash
Logistic regression using RBM features:
              precision    recall  f1-score   support

           0       0.10      1.00      0.18       174
           1       0.00      0.00      0.00       184
           2       0.00      0.00      0.00       166
           3       0.00      0.00      0.00       194
           4       0.00      0.00      0.00       186
           5       0.00      0.00      0.00       181
           6       0.00      0.00      0.00       207
           7       0.00      0.00      0.00       154
           8       0.00      0.00      0.00       182
           9       0.00      0.00      0.00       169

    accuracy                           0.10      1797
   macro avg       0.01      0.10      0.02      1797
weighted avg       0.01      0.10      0.02      1797
```
{% asset_img ml_4.png %}

##### 图模型与参数化

**限制玻尔兹曼机**(`RBM`)的图模型是全连通的二分图。
{% asset_img ml_5.png %}

节点是随机变量，其状态取决于它们所连接的其他节点的状态。因此，该模型由连接的权重以及每个**可见**和**隐藏单元**的一个**截距**（偏差）项参数化，为简单起见，图像中省略了该项。能量函数衡量联合分配的质量：{% mathjax %}E(v,h) = -\sum\limits_i\sum\limits_j w_{ij}v_i h_j - \sum\limits_i b_i v_i - \sum\limits_j c_j h_j{% endmathjax %}。在上面的公式中，{% mathjax %}b{% endmathjax %}和{% mathjax %}c{% endmathjax %}分别是**可见层**和**隐藏层**的**截距向量**。模型的**联合概率**根据能量定义：{% mathjax %}P(v,h) = \frac{e^{-E(v,h)}}{Z}{% endmathjax %}。受限一词指的是模型的**二分结构**，它禁止**隐藏单元**之间或**可见单元**之间的直接交互。意味着：{% mathjax %}h_i\perp h_j|v,\; v_i\perp v_j|h{% endmathjax %}，二分结构允许使用高效的**块吉布斯采样**(`Block Gibbs Sampling`)进行推理。

**能量函数**(`Energy Function`)是一个数学表示，用于**量化**系统某一状态所具有的能量。它在多个领域中具有重要应用，包括物理学、工程学和优化算法。**能量函数**(`Energy Function`)用于描述系统的状态，并评估该状态的可行性或优越性。在优化问题中，**能量函数**(`Energy Function`)通常用于指导搜索算法，如**模拟退火**和**禁忌搜索**，通过提供一个度量来判断解的接近程度，从而影响接受或拒绝新候选解的决策。在优化问题中，**能量函数**(`Energy Function`)用于评估不同解的优劣。例如，在机器学习和计算机视觉中，**能量函数**(`Energy Function`)可以帮助模型找到最优参数或配置。通过最小化能量函数，可以找到最优解或近似最优解。

**块吉布斯采样**(`Block Gibbs Sampling`)是一种**马尔可夫链蒙特卡罗**(`MCMC`)方法，用于从复杂的多维概率分布中抽样。与传统的**吉布斯采样**不同，**块吉布斯采样**(`Block Gibbs Sampling`)在每次迭代中同时更新多个变量，而不是逐个更新。这种方法在处理高维数据和具有复杂依赖关系的模型时特别有效。

##### 伯努利限制玻尔兹曼机

**伯努利限制玻尔兹曼机**(`BRBM`)是一种特殊类型的**限制玻尔兹曼机**(`RBM`)，其中**可见层**的单元使用**伯努利分布**来建模。这使得**伯努利限制玻尔兹曼机**(`BRBM`)特别适用于处理二元(`0-1`)数据，如图像的黑白像素或用户偏好（喜欢/不喜欢）等。其模型结构与工作原理都与**限制玻尔兹曼机**(`RBM`)相同。**伯努利限制玻尔兹曼机**(`BRBM`)优势：
- **适用于二元数据**：**伯努利限制玻尔兹曼机**(`BRBM`)特别适合处理二进制或稀疏数据，使其在许多实际应用中表现良好。
- **特征学习**：通过隐含层捕捉数据中的潜在结构，使得**伯努利限制玻尔兹曼机**(`BRBM`)能够自动提取有用特征。
- **生成能力**：**伯努利限制玻尔兹曼机**(`BRBM`)不仅可以用于特征提取，还可以生成新样本，具有良好的生成模型性能。

计算**隐藏单元**的激活概率：{% mathjax %}P(h_j=1|v) = \sigma (b_j + \sum\limits_i v_i w_{ij}){% endmathjax %}。其中{% mathjax %}\sigma{% endmathjax %}是**激活函数**(`sigmoid`)，{% mathjax %}b_j{% endmathjax %}是**隐藏单元**的偏置，{% mathjax %}w_{ij}{% endmathjax %}是连接**可见单元**和**隐藏单元**的权重。使用**隐藏单元**的状态重构可见层：{% mathjax %}P(v_i|h) = \sigma (c_i + \sum\limits_j h_j w_{ij}){% endmathjax %}。其中{% mathjax %}c_i{% endmathjax %}是可见单元的偏置，{% mathjax %}\sigma(x) = \frac{1}{1 + e^{-x}}{% endmathjax %}。

##### 随机最大似然学习

**随机最大似然学习**(`Stochastic Maximum Likelihood Learning`)是一种用于参数估计的优化方法，特别适用于处理大规模数据集和复杂模型。它结合了**最大似然估计**（`MLE`）和**随机梯度下降**(`Stochastic Gradient Descent, SGD`)等技术，以提高学习效率和收敛速度。**随机最大似然学习**的应用领域包括：机器学习、信号处理、生物信息学等。

**随机最大似然学习**(`Stochastic Maximum Likelihood Learning`)是一种统计方法，用于估计模型参数，使得在给定数据的情况下，观察到的数据具有最高的概率。对于给定的参数{% mathjax %}\theta{% endmathjax %}和数据集{% mathjax %}D{% endmathjax %}，**最大似然估计**的目标是**最大化似然函数**：{% mathjax %}L(\theta|D) = P(D|\theta){% endmathjax %}。

**随机梯度下降**是一种优化算法，通过使用小批量(`mini-batch`)或单个样本来更新模型参数，从而减少计算成本并加速收敛。其基本步骤如下：1.从数据集中随机选择一个样本或小批量；2.计算**损失函数**的**梯度**；3.更新参数：{% mathjax %}\theta \gets \theta - \eta\nabla L(\theta|D){% endmathjax %}。其中{% mathjax %}\eta{% endmathjax %}是学习率。**随机最大似然学习**的步骤：
- **初始化参数**：随机初始化模型参数。
- **选择小批量**：从训练数据中随机选择一个小批量样本。
- **计算梯度**：根据当前参数计算小批量样本的似然函数的梯度。
- **更新参数**：使用计算得到的梯度更新模型参数。
- **迭代**：重复上述步骤，直到达到预定的迭代次数或满足收敛条件。

**随机最大似然学习**的优势：**高效性**，通过使用小批量样本进行更新，**随机最大似然学习**能够显著减少每次迭代所需的计算时间，适合大规模数据集；**收敛速度快**，由于引入了随机性，能够避免陷入局部最优解，从而提高收敛速度；**适应性强**，可以灵活应用于各种模型，包括**线性回归**、**神经网络**和**隐马尔可夫模型**等。**随机最大似然学习**是一种高效的参数估计方法，通过结合**最大似然估计**和**随机梯度下降**，它能够在大规模数据集上实现快速收敛。

#### 协方差估计(Covariance Estimation)

**协方差估计**(`Covariance Estimation`)是统计学中用于估计随机变量之间的**协方差**的过程。**协方差**是衡量两个随机变量之间关系的一个重要指标，反映了它们如何共同变化。准确估计**协方差**对于许多应用至关重要，包括金融建模、机器学习和信号处理等。许多统计问题都需要估计总体的**协方差矩阵**，这可以看作是对数据集散点图形状的估计。大多数情况下，这种估计必须对样本进行，样本的属性（大小、结构、同质性）对估计的质量有很大影响。**协方差**定义：给定两个随机变量{% mathjax %}X{% endmathjax %}和{% mathjax %}Y{% endmathjax %}，它们的协方差可以定义为：{% mathjax %}\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])]{% endmathjax %}，其中{% mathjax %}E{% endmathjax %}表示**期望值**。**协方差矩阵**定义：对于多个随机变量，可以构建一个**协方差矩阵**{% mathjax %}\Sigma{% endmathjax %}，其元素为各对随机变量的协方差：
{% mathjax '{"conversion":{"em":14}}' %}
\Sigma = \;
\begin{bmatrix}
\text{Var}(X_1) & \text{Cov}(X_1,X_2) & \ldots & \text{Cov}(X_1,X_n) \\
\text{Cov}(X_2,X_1) & \text{Var}(X_2) & \ldots & \text{Cov}(X_2,X_n) \\
\vdots & \vdots & \ddots & \vdots \\
\text{Cov}(X_n,X_1) & \text{Cov}(X_n,X_2) & \ldots & \text{Var}(X_n) 
\end{bmatrix}
{% endmathjax %}

**协方差估计**(`Covariance Estimation`)方法有：
- **样本协方差矩阵**(`Sample Covariance Matrix`)：**样本协方差矩阵**是最常用的估计方法，适用于完整数据集。其计算公式为：{% mathjax %}S = \frac{1}{n - 1}\sum\limits_{i=1}^n (X_1 - \bar{X})(X_i - \bar{X}^T){% endmathjax %}。其中{% mathjax %}n{% endmathjax %}是样本的大小，{% mathjax %}X_i{% endmathjax %}是样本向量，{% mathjax %}\bar{X}{% endmathjax %}是样本均值。
- **正则化和收缩方法**(`Shrinkage Methods`)：在高维数据中，**样本协方差矩阵**可能会不稳定或非正定。收缩方法通过将样本**协方差矩阵**向某个目标矩阵（如单位矩阵）收缩，从而提高估计的稳定性和准确性。
- **图模型方法**(`Graphical Models`)：**图模型**（如**高斯图模型**）直接估计**精度矩阵**（**协方差矩阵**的逆），在某些情况下比直接估计**协方差矩阵**更有效。
- **自适应方法**(`Adaptive Methods`)：这些方法根据数据特性自适应地调整估计过程，以提高准确性。例如，使用**加权平均**或**局部回归技术**来估计**协方差**。

**协方差估计**(`Covariance Estimation`)是统计分析中的关键步骤，涉及多种方法和技术。选择合适的估计方法取决于数据的特性、维度以及应用场景。通过精确的**协方差估计**，可以更好地理解变量之间的关系，并做出更有效的决策。
##### 经验协方差

**经验协方差**(`Empirical Covariance`)是通过样本数据计算出的**协方差**，用于估计总体**协方差**。它是描述两个随机变量之间关系的重要统计量，能够反映这两个变量在样本中的共同变化程度。

##### 收缩协方差(Shrunk Covariance)

**收缩协方差**(`Shrunk Covariance`)是一种用于提高**协方差矩阵估计**稳定性和准确性的技术，特别是在高维数据分析中。传统的**样本协方差矩阵**在样本量相对较小或变量数量较多时，可能会变得不稳定或非正定（即其特征值中存在负值），这会影响后续的统计分析和模型构建。

##### 稀疏逆协方差(Sparse Inverse Covariance)

**稀疏逆协方差**(`Sparse Inverse Covariance`)是一种用于估计高维数据中的**协方差矩阵逆**（**精度矩阵**）的方法，尤其适用于变量数量远大于样本数量的情况。通过引入**稀疏性约束**，这种方法能够有效地识别变量之间的条件独立性，从而提供更具可解释性的模型。

##### 鲁棒方差估计(Robust Covariance Estimation)

**鲁棒方差估计**(`Robust Covariance Estimation`)是一种用于估计**协方差矩阵**的方法，旨在提高对异常值和噪声的抵抗能力。传统的**协方差估计**方法（如**样本协方差**）在数据中存在异常值时可能会产生偏差，导致不准确的结果。**鲁棒方差估计估计**通过引入特定的技术和算法，能够更好地处理这些问题。
