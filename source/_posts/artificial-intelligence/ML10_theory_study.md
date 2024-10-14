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

