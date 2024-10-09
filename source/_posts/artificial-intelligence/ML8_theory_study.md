---
title: 机器学习(ML)(八) — 探析
date: 2024-10-04 11:24:11
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

**无监督机器学习**使用的是自学习算法，在学习时无需任何标签，也无需事先训练。相反，模型会获得不带标签的原始数据。自学习规则，并根据相似之处、差异和模式来建立信息结构，且无需向该模型提供关于如何处理各项数据的明确说明。**无监督机器学习**更适合处理复杂的任务。它能够很好的识别出数据中以前未检测到的模式，并且有助于识别用于数据分类的特征。假设有一个关于天气的大型数据集，无监督学习算法会分析数据并识别数据点中的模式。例如，它可能会按温度或类似的天气模式对数据进行分组。虽然算法本身无法根据之前提供的任何信息来理解这些模式，但可以查看数据分组情况，并根据对数据集的理解并对其进行分类。例如天气模式被划分为不同类型的天气，如雨、雨夹雪或雪。
<!-- more -->
{% asset_img ml_1.png %}

一般来说，有三种类型的**无监督学习**任务：**聚类**、**关联规则**和**降维**。
##### 聚类

**聚类**是一种探索未加标签的原始数据，并根据相似情况或差异将这些数据细分为多个组（或集群）的技术。该技术可用于各种应用，包括**客户细分**、**欺诈检测**和**图像分析**。聚类算法通过发现未分类数据中的相似结构或模式，将数据拆分为多个自然分组。**聚类**是最常用的**无监督机器学习**方法之一。用于**聚类**的**无监督式学习**算法有多种，其中包括独占、重叠、分层和概率学习算法。
- **独占聚类**：使用此算法分组数据时，单个数据点仅能存在于一个分组（集群）中。这也称为“硬”**聚类**。**独占聚类**的一个常见示例是`K-means`**聚类算法**，该算法将数据点划分为用户定义的`K`个**聚类**。
- **重叠聚类**：使用此算法分组数据时，单个数据点可存在于两个或多个关联紧密度不同的分组（集群）中。这也称为“软”**聚类**。
- **层次聚类**：数据根据相似性分成不同的**聚类**，然后根据它们的层次关系反复合并和组织。**层次聚类**主要有两种类型：**凝聚式聚类**和**分裂式层次聚类**。这种方法也称为`HAC`（**层次集群分析**）。
- **概率聚类**：根据每个数据点属于各个集群的概率，将数据分组到各分组（集群）中。这种方法与其他方法不同（根据数据点与集群中其他数据点的相似性对其进行分组）。

##### 关联(Association)

**关联**是一种基于**规则**的方法，揭示大型数据集中数据点之间的关系。**无监督学习算法**会搜索频繁的“`if-then`”**关联**（也称为规则），以发现数据中的相关性和同现情况，以及数据对象之间的不同联系。它最常用于分析零售购物车或交易数据集，以展示某些商品一起购买的频率。这些算法可以揭示客户购买模式以及之前未发现的产品关系，为**商品推荐引擎**或其他交叉销售机会提供信息。这些**规则**最常用的形式是线上零售商店中“经常一起购买”和“购买过该商品的人还买过”部分。**关联**通常也用于整理临床诊断的医疗数据集。使用**无监督式机器学习**和**关联**，可以帮助医生通过比较过往病例的症状之间的关系，确定特定诊断的可能性。一般来说，`Apriori`算法最常用于**关联学习**，以标识相关的内容集合。不过也会使用其他类型，例如`Eclat`算法和`FP-Growth`算法。

##### 降维

**降维**是一种**无监督机器学习**技术，用于减少数据集中的特征或维度数量。对机器学习而言，通常是数据越多越好，但是大量的数据也会增加直观呈现数据洞见的难度。**降维**可以从数据集中提取重要特征，从而减少其中不相关或随机特征的数量。这种方法使用**主成分分析**(`PCA`) 和**奇异值分解**(`SVD`)算法来减少数据输入的数量，同时不会破坏原始数据中属性的完整性。

##### 无监督学习示例

- **异常值检测**：**无监督式聚类**可以处理大型数据集，并发现数据集中非典型的数据点。
- **商品推荐引擎**：**无监督机器学习**可以利用**关联**（规则）来探索交易数据，从而发现模式或趋势，从而为线上零售商提供**个性化推荐**。
- **客户细分**：**无监督机器学习**也常用于通过对客户的共同特征或购买行为进行**聚类**来生成买家画像。然后，参考这些资料来制定营销和其他业务策略。
- **欺诈检测**：**无监督机器学习**对于**异常值检测**很有用，可以发现数据集中的异常数据点。这些数据分析有助于发现数据中偏离正常模式的事件或行为，从而揭露**欺诈性交易**或**机器人活动**等异常行为。
- **自然语言处理**(`NLP`)：**无监督机器学习**通常用于各种`NLP`应用，例如对新闻版面中的文章分类、文本翻译和分类，或对话界面中的语音识别。
- **基因研究**：**基因聚类**是另一个常见的**无监督机器学习**例子。**层次聚类算法**通常用于分析`DNA`模式和揭示进化关系。

#### 聚类

`1932`年，`HE Driver`和`ALKroeber`在论文[`Quantitative expression of cultural relationship`](https://digitalassets.lib.berkeley.edu/anthpubs/ucb/text/ucp031-005.pdf)中提出了**聚类**方法。自那时起，这项技术取得了长足进步，并被用于探索许多应用领域的未知领域。**聚类**是一种**无监督机器学习**，需要从未标记的数据集中提取参考(特征、模式)。通常，它用于捕获数据集中固有的有意义的结构、底层过程和分组。在**聚类任务**是将总体分成几组，使得同一组中的数据点比其他组中的数据点更相似。简而言之，它是基于相似性和不相似性的**对象集合**。通过**聚类**，数据科学家可以发现未标记数据中的内在分组。虽然没有特定的标准来衡量**聚类**，但这完全取决于用户，并如何使用它来满足特定的需求。它可用于查找数据中的异常数据点/异常值，或识别未知属性以在数据集中找到合适的**分组**。举个例子，假设您在沃尔玛商店担任经理，并希望更好地了解您的客户，以便使用新的和改进的**营销策略**来扩大业务。手动细分客户很困难。您有一些包含他们的年龄和购买历史的数据，在这里**聚类**可以根据客户的支出对数据进行**分组**。一旦完成客户细分，您就可以根据目标受众为每个组定义不同的**营销策略**。

##### K-means聚类

`K-means`**聚类**是一种**矢量量化方法**，最初来自**信号处理**，旨在将`n`个观测值划分为`k`个**聚类**，其中每个观测值属于最接近**均值**的**聚类**，作为该**聚类**的原型。`K-means`是一种基于**质心**的**聚类算法**，计算每个数据点与**质心**之间的距离，将其分配到一个**聚类**中。目标是识别数据集中的`K`个组。这是一个将每个数据点分配到组中的迭代过程，数据点会根据相似特征**聚类**。目标是最小化数据点与**聚类**中心之间的距离总和，以确定每个数据点应该属于的正确组。在这里，将数据空间划分为`K`个**簇**，并为每个**簇**分配一个平均值。数据点被放置在最接近该**簇**平均值的**簇**中。有几种距离度量可用于计算距离。 

`K-means`**的工作原理**：第一步，**选择聚类的数量**，就是定义簇数`K`，{% mathjax %}C = \{c_1,c_2,\ldots,c_K\}{% endmathjax %},其中每个{% mathjax %}c_K{% endmathjax %}都是{% mathjax %}K{% endmathjax %}维向量，表示第`k`个聚类的质心；第二步，**初始化聚类质心**，所谓**质心**就是聚类的中心，一般最初的时候聚类的中心都是未知的。所以我们选择随机的数据点作为每个聚类的**质心**；第三步，**将数据点分配给最近的簇**，质心已经初始化，接下来是将数据点{% mathjax %}x_i{% endmathjax %}分配给离{% mathjax %}x_i{% endmathjax %}最近的聚类质心{% mathjax %}c_k{% endmathjax %},在此步骤中，需要使用欧几里得距离来计算数据点{% mathjax %}x_i{% endmathjax %}和聚类质心{% mathjax %}c_k{% endmathjax %}之间的距离{% mathjax %}d(x_i,c_k) = \sqrt{\sum\limits_{i=1}^p (x_{i} -c_k)^2}{% endmathjax %},其中{% mathjax %}p{% endmathjax %}是数据的维度，然后选择数据点{% mathjax %}c_i{% endmathjax %}与聚类质心{% mathjax %}c_k{% endmathjax %}距离最小的聚类，即{% mathjax %}k = \arg \underset{k'}{\min} d(x_i,c_{k'}){% endmathjax %}；第四步，**更新聚类质心**，在所有数据点被分配到相应的簇后，更新每个簇的质心为该簇内所有数据点的均值：{% mathjax %}c_k = \frac{1}{S_k}\sum\limits_{x_i\in S_k} x_i{% endmathjax %}，其中{% mathjax %}S_k{% endmathjax %}是分配给簇{% mathjax %}k{% endmathjax %}的所有点的集合，{% mathjax %}|S_k|{% endmathjax %}是该集合中的数据点的数量；第五步，**迭代过程**，重复步骤`3`和`4`，直到质心不再发生显著变化，或达到设定的迭代次数，算法收敛时，聚类结果稳定；第六步，**目标函数**，`K-means`的目标是**最小化每个簇内的数据点与其质心之间的平方误差**，{% mathjax %}J = \sum\limits_{k=1}^K\sum\limits_{x_i\in S_k} d(x_i ,c_k)^2{% endmathjax %}，通过不断迭代，`K-means`算法旨在使目标函数{% mathjax %}J{% endmathjax %}达到最小值，从而优化聚类效果。

**质心初始化方法**的目标是**聚类的质心初始化后尽可能接近实际质心的最佳值**。常用的方法有，**随机数据点定义聚类的质心**，这是初始化质心的传统方法，其中选择{% mathjax %}K{% endmathjax %}个随机数据点并将其定义为聚类的质心；**简单分片**，分片质心初始化算法，主要依赖于数据集中特定实例或行的所有属性的综合值，其思想是计算综合值，然后对数据实例进行排序，然后将其水平划分为{% mathjax %}K{% endmathjax %}个分片，最后将各个分片的所有属性相加并计算平均值。分片属性平均值集合将被确定为初始化的聚类质心集合；`K-means++`，`K-means++`是`K-means`算法的智能质心初始化方法，目标是通过随机分配第一个质心，然后根据最大平方距离选择其余质心来分散初始质心，其想法将质心初始化为彼此远离，从而产生比随机初始化更好的结果。`K-means++`用作`K-means`的默认初始化，`scikit-learn`的`K-means++`初始化代码实现：
```python
import matplotlib.pyplot as plt

from sklearn.cluster import kmeans_plusplus
from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 4000
n_components = 4
X, y_true = make_blobs(n_samples=n_samples, centers=n_components, cluster_std=0.6, random_state=0)
X = X[:, ::-1]

# Calculate seeds from k-means++
centers_init, indices = kmeans_plusplus(X, n_clusters=4, random_state=0)

# Plot init seeds along side sample data
plt.figure(1)
colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
for k, col in enumerate(colors):
    cluster_data = y_true == k
    plt.scatter(X[cluster_data, 0], X[cluster_data, 1], c=col, marker=".", s=10)

plt.scatter(centers_init[:, 0], centers_init[:, 1], c='b', s=50)
plt.title("K - Means + + Initialization")
plt.xticks([])
plt.yticks([])
plt.show()
```
{% asset_img ml_2.png %}

`K-means`是如何处理真实数据的。这里有一个购物中心访客数据集（`2000`名）来创建客户细分，从而制定营销策略。先加载数据并检查是否有任何的缺失值：
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
customer_data = pd.read_csv("Mall_Customers.csv")

# read the data
print(customer_data.head())

# check for null or missing values
print(customer_data.isna().sum())
```
结果输出为：
```bash
   CustomerID   Genre  Age  Annual_Income_(k$)  Spending_Score
0           1    Male   19                  15              39
1           2    Male   21                  15              81
2           3  Female   20                  16               6
3           4  Female   23                  16              77
4           5  Female   31                  17              40


CustomerID            0
Genre                 0
Age                   0
Annual_Income_(k$)    0
Spending_Score        0
```
使用年度收入和支出分数来查找数据中的聚类。支出分数从`1~100`，根据客户行为和支出性质分配。先看一下数据并了解他是如何分布的。
```python
plt.scatter(customer_data['Annual_Income_(k$)'], customer_data['Spending_Score'])
plt.xlabel('Annual_Income_(k$)')
plt.ylabel('Spending_Score')
plt.show()
```
{% asset_img ml_3.png %}

从上面的散点图来看，很难判断数据集中是否存在任何模式。这时，**聚类**就会有所帮助。首先随机初始化**聚类质心**。
```python
centroids = customer_data.sample(n=3) # random init centroid
plt.scatter(customer_data['Annual_Income_(k$)'], customer_data['Spending_Score'])
plt.scatter(centroids['Annual_Income_(k$)'], centroids['Spending_Score'], c='black')
plt.xlabel('Annual_Income_(k$)')
plt.ylabel('Spending_Score')
plt.show()
```
{% asset_img ml_4.png %}

接下来遍历每个质心和数据点，计算它们之间的距离，找到{% mathjax %}K{% endmathjax %}个**聚类**并将数据点分配给一个最近的**聚类**。这个过程将持续到先前定义的质心和当前质心之间的差异接近为`0`：
```python
K = 3
centroids = customer_data.sample(n=K)
mask = customer_data['CustomerID'].isin(centroids.CustomerID.tolist())
X = customer_data[~mask]
diff = 1
j = 0
XD = X
while (diff != 0):
    i = 1
    for index1, row_c in centroids.iterrows():
        ED = []
        for index2, row_d in XD.iterrows():
            d1 = (row_c["Annual_Income_(k$)"] - row_d["Annual_Income_(k$)"])**2
            d2 = (row_c["Spending_Score"] - row_d["Spending_Score"])**2
            d = np.sqrt(d1 + d2)
            ED.append(d)
        X[i] = ED
        i = i + 1

    C = []
    for index, row in X.iterrows():
        min_dist = row[1]
        pos = 1
        for i in range(K):
            if row[i + 1] < min_dist:
                min_dist = row[i + 1]
                pos = i + 1
        C.append(pos)
    X["Cluster"] = C
    centroids_new = X.groupby(["Cluster"]).mean()[["Spending_Score", "Annual_Income_(k$)"]]
    if j == 0:
        diff = 1
        j = j + 1
    else:
        diff = (centroids_new['Spending_Score'] - centroids['Spending_Score']).sum() + (centroids_new['Annual_Income_(k$)'] - centroids['Annual_Income_(k$)']).sum()
    centroids = X.groupby(["Cluster"]).mean()[["Spending_Score", "Annual_Income_(k$)"]]

color = ['grey', 'blue', 'orange']
for k in range(K):
    data = X[X["Cluster"] == k + 1]
    plt.scatter(data["Annual_Income_(k$)"], data["Spending_Score"], c=color[k])
plt.scatter(centroids["Annual_Income_(k$)"], centroids["Spending_Score"], c='black')
plt.xlabel('Annual_Income_(k$)')
plt.ylabel('Spending_Score')
plt.show()
```
{% asset_img ml_5.png %}

`Scikit-Learn`实现`K-means`，首先，导入K-Means函数，然后通过传递**聚类**数量作为参数来调用该函数：
```python
import seaborn as sns
from sklearn.cluster import KMeans

km_sample = KMeans(n_clusters=3)
km_sample.fit(customer_data[['Annual_Income_(k$)','Spending_Score']])

labels_sample = km_sample.labels_
customer_data['label'] = labels_sample
sns.scatterplot(customer_data['Annual_Income_(k$)'],customer_data['Spending_Score'],hue=customer_data['label'],palette='Set1')
```
{% asset_img ml_6.png %}

我们使用`Scikit-Learn`几行代码创建客户数据的细分。最终的聚类数据在两种实现中是相同的。标签`0`：储蓄者，平均收入至高收入但明智消费；标签`1`：无忧无虑，收入低，但花钱大手大脚；标签`2`：消费者，平均收入至高收入。商场管理层可以相应地调整营销策略，例如，向标签`0`：储蓄者群体提供更多储蓄优惠，为标签`2`：大手笔消费者开设更多利润丰厚的商店。

`K`如何选择？一些因素会影响`K-means`**聚类**算法输出的有效性，其中之一就是确定聚类数({% mathjax %}K{% endmathjax %})。选择较少的聚类数会导致**欠拟合**，而指定较多的聚类数会导致**过拟合**。最佳**聚类**数取决于**相似性度量**和**用于聚类的参数**。因此，要找到数据中的聚类数，我们需要对执行`K-means`**聚类**一系列值进行比较。目前，可以使用一些技术来估计该值，包括**交叉验证**、**肘部法**(`Elbow Method`)、**信息准则**、**轮廓法**(`Silhouette`)和`G-means`算法。 
- **肘部法**(`Elbow Method`)：距离度量是比较不同{% mathjax %}K{% endmathjax %}值结果的常用度量之一。当簇数{% mathjax %}K{% endmathjax %}增加时，质心到数据点的距离将减小，并达到{% mathjax %}K{% endmathjax %}与数据点数相同的点。这就是我们一直使用到质心的距离平均值的原因。在**肘部法**(`Elbow Method`)中，绘制平均距离并寻找减少率发生变化的**肘点**。这个肘点可用于确定{% mathjax %}K{% endmathjax %}。 肘点在数学优化中用作终止点，决定在哪个点收益递减不再值得额外花费。在**聚类**中，当添加另一个聚类不会改善建模结果时，它用于选择一定数量的聚类。这是一个迭代过程，其中将对数据集进行`K-means`**聚类**，{% mathjax %}K{% endmathjax %}值的范围如下：使用所有{% mathjax %}K{% endmathjax %}值执行`K-means`**聚类**。对于每个{% mathjax %}K{% endmathjax %}值，计算所有数据点到质心的平均距离，绘制每个点并找到平均距离突然下降的点（肘部），代码实现如下：
```python
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)

# k means determine k
distortions = []
K = range(1,10)
for k in K:
   kmeanModel = KMeans(n_clusters=k).fit(X)
   kmeanModel.fit(X)
   distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
```
{% asset_img ml_7.png %}

这可能是确定最佳簇数的最常用方法。不过，找到拐点可能是一个挑战，因为在实践中可能没有尖锐的拐点。 
- **轮廓法**(`Silhouette`)：**轮廓法**是指一种解释和验证数据集群内一致性的方法。该技术以简洁的图形表示每个对象的分类情况。**轮廓系数**用于通过检查某个簇内的数据点与其他簇的相似程度来衡量簇的质量。**轮廓分析**可用于研究簇之间的距离。此离散测量值介于`-1`和`1`之间：`+1`：表示数据点距离相邻簇较远，因此位置最佳。`0`：表示它位于两个相邻集群之间的决策边界上。`-1`：表示数据点被分配到错误的簇。为了找到聚类数{% mathjax %}K{% endmathjax %}的最优值，使用轮廓图来显示一个聚类中每个点与相邻聚类中某个点的接近程度，从而提供一种直观评估聚类数等参数的方法。针对一系列值计算`K-means`**聚类**算法，对于{% mathjax %}K{% endmathjax %}的每个值，找到数据点的平均轮廓分数：
```python
from sklearn.metrics import silhouette_score

sil_avg = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for k in range_n_clusters:
 kmeans = KMeans(n_clusters = k).fit(X)
 labels = kmeans.labels_
 sil_avg.append(silhouette_score(X, labels, metric = 'euclidean'))

# 绘制每个K值的轮廓分数集合, 选择轮廓得分最大时的聚类数量：
plt.plot(range_n_clusters,sil_avg,'bx-')
plt.xlabel('Values of K')
plt.ylabel('Silhouette score')
plt.title('Silhouette analysis For Optimal k')
plt.show()
```
{% asset_img ml_8.png %}

利用**轮廓法**(`Silhouette`)分析，选择{% mathjax %}K = 3{% endmathjax %}时，平均轮廓(`Silhouette`)分数最高，表明数据点的位置处于最佳状态。

**聚类评估指标**：在聚类中，没有任何标记的数据，只有一组特征，目标是获得这些特征的高簇内相似性和低簇间相似性。评估任何**聚类**算法的性能并不像在监督学习中计算错误数量、找到**精度**或**召回率**那么容易，目前有多种可用的聚类评估指标。
- **兰德指数**(`Rand Index`)：是一种用于评估聚类结果与真实标签之间相似度的指标。它通过比较数据点在不同聚类中的分组情况，量化聚类的质量。**兰德指数**(`Rand Index`)的值介于`0`和`1`之间，值越高表示聚类结果与真实标签越一致。如果{% mathjax %}C{% endmathjax %}是真实类别分配，{% mathjax %}K{% endmathjax %}是聚类，定义{% mathjax %}a,b{% endmathjax %}:
  - `a`代表{% mathjax %}C{% endmathjax %}中属于同一集合{% mathjax %}K{% endmathjax %}中属于同一集合的元素对的数量。
  - `b`代表{% mathjax %}C{% endmathjax %}中不同集合中的元素对数和{% mathjax %}K{% endmathjax %}中不同集合的元素对数。

**兰德指数**(`Rand Index`)公式如下：{% mathjax %}RI = \frac{a + b}{C_2^{n_{\text{sample}}}}{% endmathjax %}，其中{% mathjax %}C_2^{n_{\text{sample}}}{% endmathjax %}是数据集中元素对数的总数，计算是在有序对还是无序对上执行并不重要，只要计算一致即可。然而，**兰德指数**(`Rand Index`)并不能保证随机标签分配会得到接近于零的值（特别是当聚类的数量与样本的数量在同一数量级时）。为了抵消这种影响，我们可以降低{% mathjax %}RI{% endmathjax %}的预期{% mathjax %}E[RI]{% endmathjax %}，调整如下所示：{% mathjax %}ARI = \frac{RI - E[RI]}{\max(RI) - E[RI]}{% endmathjax %}。
- **轮廓系数**(`Silhouette Coefficient`)：**轮廓系数**(`Silhouette Coefficient`)针对每个样本定义，**轮廓系数**(`Silhouette Coefficient`)介于`-1`（表示聚类不正确）和`+1`（表示聚类高度密集）之间。分数接近零表示聚类重叠。由两个分数组成：
  - `a`：样本与同一类别中所有其他点之间的平均距离。
  - `b`：一个样本与下一个最近簇中所有其他点之间的平均距离。
单个样本的**轮廓系数**(`Silhouette Coefficient`){% mathjax %}s{% endmathjax %}定义为：{% mathjax %}s = \frac{b-a}{\max(a,b)}{% endmathjax %}

`K-means`**聚类**的优点：
- 相对容易理解和实施。
- 可扩展至大型数据集。
- 更好的计算成本。
- 轻松热启动质心的分配和位置。

`K-means`**聚类**算法的缺点：
- 手动选择{% mathjax %}K{% endmathjax %}并依赖于初始值。
- 对于不同的{% mathjax %}K{% endmathjax %}值，缺乏一致的结果。
- 总是试图找到圆形簇。
- 由于数据集中的异常值，质心被拖拽。
- 维数灾难，当维数增加时，{% mathjax %}K{% endmathjax %}无效。

#### 异常值（Outliers）和新颖点（Novelty）

**异常值**(`Outliers`)和**新颖点**(`Novelty`)是数据分析中的两个重要概念，旨在识别与大多数数据点显著不同的实例。这些实例可能是**异常值**(`Outliers`)或**新颖点**(`Novelty`)，在许多应用中具有重要意义，如欺诈检测、故障检测和生物监测等。
- **异常值**(`Outliers`)：指在数据集中明显偏离其他数据点的实例。这些数据点通常是由于测量错误、数据录入错误或极端情况造成的。**异常值**(`Outliers`)可能会对分析结果产生显著影响，因此需要特别关注。
- **新颖点**(`Novelty`)：指在训练集未出现过的新模式或新实例。与异常值不同，**新颖点**(`Novelty`)并不一定是错误或噪声，而是代表了新的信息或趋势。在某些情况下**异常值**(`Outliers`)可能是值得进一步研究的潜在重要发现。

**异常值检测**和**新颖性检测**均用于**异常检测**，人们感兴趣的是**检测异常**或**不寻常的观察结果**。**异常值检测**也称为**无监督异常检测**，而**新颖性检测**则称为**半监督异常检测**。在**异常值检测**中，异常值/异常不能形成密集簇，因为可用的估计器假设异常值/异常位于低密度区域。相反，在**新颖性检测**中，只要新颖性/异常位于训练数据的低密度区域，它们就可以形成**密集簇**，在这种情况下被视为正常。
##### 局部异常因子(LOF)

**局部异常因子**(`LOF`)是应用最广泛的无监督的**局部异常检测算法**。它采用最近邻居的思想来确定**异常**或**离群值分数**。它计算数据点相对于其邻居的**局部密度偏差**。它将密度明显低于邻居的样本视为**离群值**。简单来说，**局部异常因子**(`LOF`)将一个数据点的**局部密度**与其{% mathjax %}k{% endmathjax %}个最近邻的**局部密度**进行比较，并给出一个分数作为最终输出。
{% asset_img ml_9.png %}

`k-distance`表示一个数据点到其第{% mathjax %}k{% endmathjax %}个邻居的距离，如下图所示，假设{% mathjax %}k = 3{% endmathjax %}计算`A`数据点的**局部异常因子**(`LOF`)，`k-distance`表示与`A`数据点排第三距离最近的的数据点，这里是数据点`E`。**可达距离**(`RD`)代表为相邻点的`k-distance`与两点之间距离最大值。从数据点A到数据点B，则{% mathjax %}RD = \max(\text{k-distance},\text{distance(A,B)}){% endmathjax %}。最后**局部可达密度**(`LRD`)的计算公式为：{% mathjax %}LRD_k(x) = 1/\big(\frac{\sum_{o\in N_k(x)}d_k(x,o)}{|N_k (x)|}\big){% endmathjax %}。最后计算`A`点的**局部异常因子**(`LOF`)，这里需要三个步骤：`1`.必须为每个数据点（假设为{% mathjax %}x{% endmathjax %}）找到{% mathjax %}k{% endmathjax %}个最近邻。`2`.使用{% mathjax %}k{% endmathjax %}个最近邻{% mathjax %}N_k{% endmathjax %}，通过计算**局部可达密度**(`LRD`)来估计数据点的**局部密度**。`3`.最后，通过将一条记录的`LRD`与其{% mathjax %}k{% endmathjax %}个邻居的`LRD`进行比较来计算`LOF`分数。**局部异常因子**(`LOF`)的计算公式为：{% mathjax %}\text{LOF}(x) = \frac{\sum\limits_{o\in N_k(x)}\frac{LRD_k(o)}{LRD_k(x)}}{|N_k(x)|}{% endmathjax %}。这里计算数据点`A`的**局部异常因子**(`LOF`){% mathjax %}RD_A = \max(3rd k-distance(B),distance(AB)) + \ldots + \max(3rd k-distance(B),distance(AE)){% endmathjax %}，并以B、C、E为参考点，计算{% mathjax %}RD_B,RD_C,RD_E{% endmathjax %}，计算A与其邻居每个数据点的`LRD`，则{% mathjax %}LOF_A =( (LRD_E +LRD_B +LRD_C)/ LRD_A) * 1/3{% endmathjax %}。密度与邻居密度一样大的点的结果得分约为{% mathjax %}1.0{% endmathjax %}。局部密度较低的异常将会产生更高的分数。
```python
from matplotlib.legend_handler import HandlerPathCollection
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

np.random.seed(42)
# Generate data with outliers
X_inliers = 0.3 * np.random.rand(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers] = -1

# 拟合异常值检测模型
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_

# 绘制图形
def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)
plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)})
plt.title("Local Outlier Factor (LOF)")
plt.show()
```
{% asset_img ml_10.png %}

##### 拟合椭圆包络(Fitting an elliptic envelope)

**拟合椭圆包络**(`Fitting an elliptic envelope`)是一种用于**异常检测**的机器学习算法，主要用于识别多维数据中的**异常值**。该算法假设数据点遵循**高斯分布**，并试图找到一个最小体积的椭球体，以包围正常数据。任何落在此估计椭球体之外的数据点都被视为**异常值**。**拟合椭圆包络**(`Fitting an elliptic envelope`)算法的原理是：**高斯分布假设**，数据点被认为是从**多维高斯分布**中生成的；**椭球体拟合**，算法通过计算数据的**均值**和**协方差矩阵**，拟合出一个椭球体，该椭球体能够覆盖大部分正常数据；**异常值判定**，任何位于此椭球体外的数据点被标记为**异常值**。**拟合椭圆包络**(`Fitting an elliptic envelope`)是一种有效的异常检测工具，适用于许多实际应用，如**金融欺诈检测**、**网络安全**等领域。通过合理设置参数和理解其假设，可以有效识别并处理数据中的**异常值**。

下面有一个实例，**鲁棒协方差估计与马哈拉诺比斯距离的相关性**。**鲁棒协方差**估计是一种用于**异常检测**的技术，特别适用于**高斯分布**的数据。**马哈拉诺比斯距离**是一种衡量一个点与分布之间距离的方法，它通过分布的**均值**和**协方差矩阵**来进行计算。其原理为：**高斯分布假设**，在**高斯分布**的数据中，**马哈拉诺比斯距离**可以用来计算**观测值**与**分布模式**之间的距离；**协方差矩阵的影响**，标准的**最大似然估计器**(`MLE`)对数据集中异常值非常敏感，因此计算出的**马哈拉诺比斯距离**也会受到影响。为了确保估计结果能够抵抗“错误”观测值的干扰，使用**鲁棒的协方差估计器**是更好的选择；**最小协方差行列式估计器**(`MCD`)：`MCD`是一种**鲁棒的协方差估计器**，具有高抗干扰能力。它通过寻找具有最小行列式的样本子集，从而计算出更准确的**协方差矩阵**。

这里的**马哈拉诺比斯距离**计算公式：{% mathjax %}d_{(\mu,\sum)}(x_i)^2 = (x_i - \mu)^T\sum^{-1}(x_i - \mu){% endmathjax %}，其中{% mathjax %}\mu{% endmathjax %}是高斯分布的位置，{% mathjax %}\sum{% endmathjax %}是高斯分布的协方差。最好使用**鲁棒的协方差估计器**来保证估计值能够抵抗数据集中的“错误”观测值，并且计算出的**马哈拉诺比斯距离**准确反映观测值的真实性。**最小协方差行列式估计器**(`MCD`)是一个鲁棒的、高抗干扰能力的协方差估计器。`MCD`是为了找到{% mathjax %}\frac{n_{\text{sample}} + n_{\text{\features}} + 1}{2}{% endmathjax %}最小行列式的样本子集，从而计算出更准确的协方差矩阵。此示例说明了**马哈拉诺比斯距离**如何受到异常数据的影响。当使用**最大似然估计**(`MLE`)的**马哈拉诺比斯距离**时，从污染分布中得出的观测值与来自真实**高斯分布**的观测值无法得到区分。使用基于**最小协方差行列式估计器**(`MCD`)的**马哈拉诺比斯距离**，这两个群体变得可以区分。
```python
import numpy as np

# for consistent results
np.random.seed(7)

n_samples = 125
n_outliers = 25
n_features = 2

# 生成一个包含125个样本和2个特征的数据集。两个特征都是高斯分布的，
# 均值为0，但特征1的标准差等于2，特征2 的标准差等于1
gen_cov = np.eye(n_features)
gen_cov[0, 0] = 2.0
X = np.dot(np.random.randn(n_samples, n_features), gen_cov)

# 将25个样本替换为高斯异常样本，其中特征1的标准差等于1，特征2的标准差等于7。
outliers_cov = np.eye(n_features)
outliers_cov[np.arange(1, n_features), np.arange(1, n_features)] = 7.0
X[-n_outliers:] = np.dot(np.random.randn(n_outliers, n_features), outliers_cov)

```
下面，将基于`MCD`和`MLE`的**协方差估计器**拟合到我们的数据中，并打印估计器的**协方差矩阵**。请注意，使用基于**最大似然估计器**(`MLE`)估计的特征`2`的方差比使用**最小协方差行列式估计器**(`MCD`)估计的特征`2`的方差高得多。这表明基于**最小协方差行列式估计器**(`MCD`)的鲁棒协方差估计对异常样本的抵抗力更强，这些样本在特征2中具有更大的方差。
```python
import matplotlib.pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# fit a MCD robust estimator to data
robust_cov = MinCovDet().fit(X)
# fit a MLE estimator to data
emp_cov = EmpiricalCovariance().fit(X)
print("Estimated covariance matrix:\nMCD (Robust):\n{}\nMLE:\n{}".format(robust_cov.covariance_, emp_cov.covariance_))
```
结果输出为：
```bash
MCD: [[ 3.26253567e+00 -3.06695631e-03] [-3.06695631e-03  1.22747343e+00]]
MLE: [[ 3.23773583 -0.24640578] [-0.24640578  7.51963999]]
```
为了更直观地展示差异，绘制这两种方法(`MCD,MLE`)计算出的**马哈拉诺比斯距离**的轮廓。请注意，基于鲁棒`MCD`的**马哈拉诺比斯距离**更适合内部黑点，而基于`MLE`的**马哈拉诺比斯距离**则受外部红点的影响更大。
```python
import matplotlib.lines as mlines

fig, ax = plt.subplots(figsize=(10, 5))
# Plot data set
inlier_plot = ax.scatter(X[:, 0], X[:, 1], color="black", label="inliers")
outlier_plot = ax.scatter(
    X[:, 0][-n_outliers:], X[:, 1][-n_outliers:], color="red", label="outliers"
)
ax.set_xlim(ax.get_xlim()[0], 10.0)
ax.set_title("Mahalanobis distances of a contaminated data set")

# Create meshgrid of feature 1 and feature 2 values
xx, yy = np.meshgrid(
    np.linspace(plt.xlim()[0], plt.xlim()[1], 100),
    np.linspace(plt.ylim()[0], plt.ylim()[1], 100),
)
zz = np.c_[xx.ravel(), yy.ravel()]
# Calculate the MLE based Mahalanobis distances of the meshgrid
mahal_emp_cov = emp_cov.mahalanobis(zz)
mahal_emp_cov = mahal_emp_cov.reshape(xx.shape)
emp_cov_contour = plt.contour(
    xx, yy, np.sqrt(mahal_emp_cov), cmap=plt.cm.PuBu_r, linestyles="dashed"
)
# Calculate the MCD based Mahalanobis distances
mahal_robust_cov = robust_cov.mahalanobis(zz)
mahal_robust_cov = mahal_robust_cov.reshape(xx.shape)
robust_contour = ax.contour(
    xx, yy, np.sqrt(mahal_robust_cov), cmap=plt.cm.YlOrBr_r, linestyles="dotted"
)

# Add legend
ax.legend(
    [
        mlines.Line2D([], [], color="tab:blue", linestyle="dashed"),
        mlines.Line2D([], [], color="tab:orange", linestyle="dotted"),
        inlier_plot,
        outlier_plot,
    ],
    ["MLE dist", "MCD dist", "inliers", "outliers"],
    loc="upper right",
    borderaxespad=0,
)

plt.show()
```
{% asset_img ml_11.png %}

如何基于**马哈拉诺比斯距离**来区分异常值？对马哈拉诺比斯距离取立方根，得到近似正态分布，然后用箱线图绘制正常和异常样本的值。从下图所示，可以看出基于鲁棒`MCD`的**马哈拉诺比斯距离**，异常值样本的分布比正常值样本的分布更加分离。
```python
fig, (ax1, ax2) = plt.subplots(1, 2)
plt.subplots_adjust(wspace=0.6)

# Calculate cubic root of MLE Mahalanobis distances for samples
emp_mahal = emp_cov.mahalanobis(X - np.mean(X, 0)) ** (0.33)
# Plot boxplots
ax1.boxplot([emp_mahal[:-n_outliers], emp_mahal[-n_outliers:]], widths=0.25)
# Plot individual samples
ax1.plot(
    np.full(n_samples - n_outliers, 1.26),
    emp_mahal[:-n_outliers],
    "+k",
    markeredgewidth=1,
)
ax1.plot(np.full(n_outliers, 2.26), emp_mahal[-n_outliers:], "+k", markeredgewidth=1)
ax1.axes.set_xticklabels(("inliers", "outliers"), size=15)
ax1.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
ax1.set_title("Using non-robust estimates\n(Maximum Likelihood)")

# Calculate cubic root of MCD Mahalanobis distances for samples
robust_mahal = robust_cov.mahalanobis(X - robust_cov.location_) ** (0.33)
# Plot boxplots
ax2.boxplot([robust_mahal[:-n_outliers], robust_mahal[-n_outliers:]], widths=0.25)
# Plot individual samples
ax2.plot(
    np.full(n_samples - n_outliers, 1.26),
    robust_mahal[:-n_outliers],
    "+k",
    markeredgewidth=1,
)
ax2.plot(np.full(n_outliers, 2.26), robust_mahal[-n_outliers:], "+k", markeredgewidth=1)
ax2.axes.set_xticklabels(("inliers", "outliers"), size=15)
ax2.set_ylabel(r"$\sqrt[3]{\rm{(Mahal. dist.)}}$", size=16)
ax2.set_title("Using robust estimates\n(Minimum Covariance Determinant)")

plt.show()
```
{% asset_img ml_12.png %}

##### 孤立森林(Isolation Forest)

**孤立森林**(`Isolation Forest`)是一种用于**异常检测**的**无监督学习算法**，最初由`Fei Tony Liu`等人在`2008`年提出。该算法通过构建多个二叉树来识别数据中的**异常点**，具有线性时间复杂度和低内存使用，特别适合处理高维数据。**孤立森林**(`Isolation Forest`)的核心原理是：**构建隔离树**，算法随机选择一个**特征**和一个**分割值**，将数据集分割成两个部分。这一过程会递归进行，直到每个数据点都被隔离到一个独立的叶节点；**路径长度**，每个数据点从根节点到达叶节点所经过的边数称为**路径长度**。对于异常点，由于其特性，通常需要更少的分割（即更短的路径）就能被隔离。因此，**路径长度**越短，表示该点越可能是异常；**异常分数**，算法通过计算每个数据点在所有隔离树中的**平均路径长度**来确定其**异常分数**。分数越低，表明该点越可能是异常。具体而言，**异常分数**可以通过以下公式计算：{% mathjax %}s(x) = 2^{-\frac{E[h(x)]}{c(n)}}{% endmathjax %}，其中，{% mathjax %}E[h(x)]{% endmathjax %}是通过所有树计算的**平均路径长度**，{% mathjax %}c(n){% endmathjax %}是二叉搜索树在{% mathjax %}n{% endmathjax %}个观察值下的**平均路径长度**。其主要应用于金融欺诈检测、网络安全、制造业等领域。**孤立森林**(`Isolation Forest`)是一种高效且灵活的**异常检测工具**，通过随机分割和路径长度来识别数据中的异常点。

**孤立森林**(`Isolation Forest`)是一组“**孤立树**”，通过**递归随机分割**来“**隔离**”观测值，可以用树结构表示。隔离样本所需的分割次数对于异常值较低，而对于正常值较高。现在有一个**孤立森林**(`Isolation Forest`)的例子，在玩具数据集上训练的**孤立森林**(`Isolation Forest`)的决策边界。
```python
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

n_samples, n_outliers = 120, 40
rng = np.random.RandomState(0)
covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # general
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # spherical
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

X = np.concatenate([cluster_1, cluster_2, outliers])
y = np.concatenate([np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)])
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# 可视化（绘图）
scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
handles, labels = scatter.legend_elements()
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.title("Gaussian inliers with \nuniformly distributed outliers")
plt.show()
```
{% asset_img ml_13.png %}

创建**孤立森林**(`Isolation Forest`)，代码实现为：
```python
from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)
```
结果输出为：
```bash
IsolationForest(max_samples=100, random_state=0)
```
**绘制离散决策边界**，背景颜色表示该给定区域中的样本是否被预测为异常值。散点图显示真实标签。
```python
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict",
    alpha=0.5,
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
disp.ax_.set_title("Binary decision boundary \nof IsolationForest")
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.show()
```
{% asset_img ml_14.png %}

**绘制路径长度的决策边界**，当一组随机树共同产生较短的**路径长度**并隔离某些特定样本时，该数据点很可能是异常值，**正态性度量**接近`0`。同样，较大的路径长度**正态性度量**接近`1`，该数据点很可能是正常值。
```python
disp = DecisionBoundaryDisplay.from_estimator(clf,X,response_method="decision_function",alpha=0.5,)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
disp.ax_.set_title("Path length decision boundary \nof IsolationForest")
plt.axis("square")
plt.legend(handles=handles, labels=["outliers", "inliers"], title="true class")
plt.colorbar(disp.ax_.collections[1])
plt.show()
```
{% asset_img ml_15.png %}
