---
title: 机器学习(ML)(九) — 探析
date: 2024-10-09 16:24:11
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

#### 密度估计

**密度估计**(`Density Estimation`)是一种用于估计随机变量的**概率密度函数**(`PDF`)的非参数统计方法。它通过对样本数据进行分析，提供一个平滑的函数，以表示数据在不同值上的分布情况。**密度估计**(`Density Estimation`)在**数据分析**、**机器学习**、**信号处理**等多个领域中具有广泛应用。
<!-- more -->
**密度估计**(`Density Estimation`)的主要方法包括：**直方图**(`Histogram`)、**核密度估计**(`Kernel Density Estimation, KDE`)、**参数密度估计**：
- **直方图**(`Histogram`)：**直方图**是最基本的**密度估计**方法之一。通过将数据范围划分为若干个区间（称为“**箱**”），并计算每个区间内的数据点数量，**直方图**可以直观地展示数据分布。优点：简单易懂，易于实现；缺点：对箱宽和起始位置敏感，可能导致信息损失。不够平滑，无法很好地捕捉数据的细微结构。
- **核密度估计**(`Kernel Density Estimation, KDE`)：**核密度估计**是一种更为灵活和光滑的**密度估计**方法。它通过在每个数据点周围放置一个**核函数**（如高斯核）来构建平滑的**密度曲线**。`KDE`的公式为：{% mathjax %}\hat{f}(x) = \frac{1}{n}\sum\limits_{i=1}^n K(\frac{x - x_i}{h}){% endmathjax %}。其中{% mathjax %}\hat{f}(x){% endmathjax %}是数据点{% mathjax %}x{% endmathjax %}的估计密度，{% mathjax %}n{% endmathjax %}是样本数量，{% mathjax %}K{% endmathjax %}是核函数，{% mathjax %}h{% endmathjax %}是带宽参数，决定了核的宽度。优点：提供平滑的密度估计，能够更好地捕捉数据的分布特征，可以选择不同类型的核函数，以适应不同的数据分布。缺点：带宽选择对结果影响较大，选择不当可能导致**过拟合**或**欠拟合**，在高维空间中计算复杂度较高，可能导致“维度诅咒”。
- **参数密度估计**：**参数密度估计**假设数据遵循某种已知的分布（如**正态分布**、**指数分布**等），通过**最大似然估计**或**贝叶斯方法**来估计参数。这种方法通常适用于数据量较少且对分布有**先验知识**的情况。优点：计算效率高，模型简单。缺点：对模型假设敏感，如果真实分布与假设不符，可能导致偏差。

**密度估计**(`Density Estimation`)是一种重要的统计工具，通过不同的方法可以有效地描述和理解数据的分布特征。无论是使用简单的直方图还是更复杂的核密度估计，选择合适的方法和参数对于获取准确和有意义的结果至关重要。这里举一个例子，使用**核密度估计**(`KDE`)学习手写数字数据的生成模型并从该模型中提取新样本。
```python
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

# load the data
digits = load_digits()
# project the 64-dimensional data to a lower dimension
pca = PCA(n_components=15, whiten=False)
data = pca.fit_transform(digits.data)

# use grid search cross-validation to optimize the bandwidth
params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(), params)
grid.fit(data)

print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

# use the best estimator to compute the kernel density estimate
kde = grid.best_estimator_

# sample 44 new points from the data
new_data = kde.sample(44, random_state=0)
new_data = pca.inverse_transform(new_data)

# turn data into a 4 x 11 grid
new_data = new_data.reshape(4, 11, -1)
real_data = digits.data[:44].reshape(4, 11, -1)

# plot real digits and resample digits
fig, ax = plt.subplots(9, 11, subplot_kw=dict(xticks=[], yticks=[]))
for j in range(11):
    ax[4, j].set_visible(False)
    for i in range(4):
        im = ax[i, j].imshow(real_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest")
        im.set_clim(0, 16)
        im = ax[i + 5, j].imshow(real_data[i, j].reshape((8, 8)), cmap=plt.cm.binary, interpolation="nearest")
        im.set_clim(0, 16)

ax[0, 5].set_title("Selection from the input data")
ax[5, 5].set_title('"New" digits drawn from the kernel density model')
plt.show()
```
{% asset_img ml_1.png %}

#### 高斯混合模型(Gaussian Mixture Model)

**高斯混合模型**(`GMM`)是一种基于概率的统计模型，假设所有数据点是由多个**高斯分布**的混合生成的。它广泛应用于**聚类**、**密度估计**和**分类**等任务。**高斯混合模型**(`GMM`)的核心思想是将复杂的**数据分布**视为多个简单的**高斯分布**的组合，从而更好地捕捉数据的多样性和复杂性。
- **高斯分布**：**高斯分布**，也称为**正态分布**，是一种在自然界中常见的概率分布，其**概率密度函数**呈现出钟形曲线。**多维高斯分布**则通过**均值向量**和**协方差矩阵**来描述。
- **混合模型**：**高斯混合模型**(`GMM`)将数据视为多个**高斯分布**的加权组合。每个**高斯分布**对应一个“**成分**”，其权重表示该成分在整体模型中的重要性。
- **数学表示**：**高斯混合模型**(`GMM`)的**概率密度函数**可以表示为：{% mathjax %}p(x|\theta) = \sum\limits_{k=1}^K \pi_k N(x | \mu_k, \Sigma_k){% endmathjax %}，其中{% mathjax %}K{% endmathjax %}是成分数量，{% mathjax %}\pi_k{% endmathjax %}是第{% mathjax %}k{% endmathjax %}个成分的权重，{% mathjax %}N(x | \mu_k, \Sigma_k){% endmathjax %}是以均值{% mathjax %}\mu_k{% endmathjax %}和协方差{% mathjax %}\Sigma_k{% endmathjax %}为参数的高斯分布。

**高斯混合模型**(`GMM`)通常使用**期望最大化**(`Expectation-Maximization, EM`)算法进行参数估计。`EM`算法包含两个步骤：
- **期望步骤**（`E`步）：计算每个数据点属于各个成分的后验概率。
- **最大化步骤**（`M`步）：根据后验概率更新模型参数，包括均值、协方差和权重。

这个过程会反复进行，直到收敛于某个稳定状态。**高斯混合模型**(`GMM`)在多个领域有广泛应用，包括：
- 聚类分析：相比于`K-means`，**高斯混合模型**(`GMM`)可以处理形状各异、大小不同的簇，并允许数据点有一定的模糊分类（软分类）。
- 密度估计：通过**高斯混合模型**(`GMM`)，可以对复杂的数据分布进行建模，从而生成新的样本或进行异常检测。
- 图像处理与计算机视觉：在图像分割、目标检测等任务中，**高斯混合模型**(`GMM`)可以有效地提取特征。
- 语音识别：在语音信号处理领域，**高斯混合模型**(`GMM`)被用来建模声音特征，以提高识别精度。

**高斯混合模型**(`GMM`)是一种概率模型，它假设所有数据点都是由有限数量的**高斯分布**与未知参数的混合生成的。我们可以将**高斯混合模型**(`GMM`)视为扩展的`K-means`聚类，并纳入有关数据协方差结构以及潜在高斯中心的信息。这里举例说明，**高斯混合模型**(`GMM`)的**协方差类型**。我们使用鸢尾花数据集上的各种**高斯混合模型**(`GMM`)协方差类型在训练和测试数据上绘制预测标签。将**高斯混合模型**(`GMM`)与球面、对角、完整和绑定协方差矩阵进行比较，按性能的升序排列。虽然看起来**完整协方差**表现最佳，但它在小型数据集上容易**过度拟合**，并且不能很好地推广到测试数据集。
```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

colors = ["navy", "turquoise", "darkorange"]

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == "full":
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == "tied":
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == "diag":
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == "spherical":
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1], angle=180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

iris = datasets.load_iris()

# Break up the dataset into non-overlapping training (75%) and testing
# (25%) sets.
skf = StratifiedKFold(n_splits=4)
# Only take the first fold.
train_index, test_index = next(iter(skf.split(iris.data, iris.target)))

X_train = iris.data[train_index]
y_train = iris.target[train_index]
X_test = iris.data[test_index]
y_test = iris.target[test_index]

n_classes = len(np.unique(y_train))

# Try GMMs using different types of covariances.
estimators = {
    cov_type: GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=20, random_state=0)
    for cov_type in ["spherical", "diag", "tied", "full"]
}
n_estimators = len(estimators)

plt.figure(figsize=(3 * n_estimators // 2, 6))
plt.subplots_adjust(bottom=0.01, top=0.95, hspace=0.15, wspace=0.05, left=0.01, right=0.99)
for index, (name, estimator) in enumerate(estimators.items()):
    # Since we have class labels for the training data, we can
    # initialize the GMM parameters in a supervised manner.
    estimator.means_init = np.array([X_train[y_train == i].mean(axis=0) for i in range(n_classes)])

    # Train the other parameters using the EM algorithm.
    estimator.fit(X_train)

    h = plt.subplot(2, n_estimators // 2, index + 1)
    make_ellipses(estimator, h)

    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color, label=iris.target_names[n])
    # Plot the test data with crosses
    for n, color in enumerate(colors):
        data = X_test[y_test == n]
        plt.scatter(data[:, 0], data[:, 1], marker="x", color=color)

    y_train_pred = estimator.predict(X_train)
    train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
    plt.text(0.05, 0.9, "Train accuracy: %.1f" % train_accuracy, transform=h.transAxes)

    y_test_pred = estimator.predict(X_test)
    test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
    plt.text(0.05, 0.8, "Test accuracy: %.1f" % test_accuracy, transform=h.transAxes)

    plt.xticks(())
    plt.yticks(())
    plt.title(name)

plt.legend(scatterpoints=1, loc="lower right", prop=dict(size=12))
plt.show()
```
{% asset_img ml_2.png %}

在图中，训练数据以点显示，而测试数据以十字显示。