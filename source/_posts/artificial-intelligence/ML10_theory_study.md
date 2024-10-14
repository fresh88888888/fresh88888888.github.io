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

