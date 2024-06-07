---
title: 数据可视化（Python）
date: 2024-06-07 11:40:11
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

鸢尾花(`Iris`)数据集如何利用`pandas, matplotlib`和`seaborn`库进行可视化分析。
<!-- more -->
```python
# First, we'll import pandas, a data processing and CSV file I/O library
import pandas as pd
# We'll also import seaborn, a Python graphing library
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sns.set(style="white", color_codes=True)

# Next, we'll load the Iris flower dataset, which is in the "../input/" directory
iris = pd.read_csv("../input/Iris.csv") # the iris dataset is now a Pandas DataFrame

# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do
iris.head()
```
让我们看一下数据，结果输出为：
```bash
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species(物种)
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
```
每个物种有多少个实例：
```python
# Let's see how many examples we have of each species
iris["Species"].value_counts()
```
结果输出为：
```python
Iris-setosa        50
Iris-versicolor    50
Iris-virginica     50
Name: Species, dtype: int64
```
我使用`pandas`数据帧的`.plot()`绘图能力来制作鸢尾花(`Iris`)特征的散点图。
```python
# The first way we can plot things is using the .plot extension from Pandas dataframes
# We'll use this to make a scatterplot of the Iris features.
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
```
{% asset_img dv_1.png %}

我们还可以使用`seaborn`库来制作类似的散点图，`Seaborn`组合图在同一图中显示双变量散点图和单变量直方图：
```python
# We can also use the seaborn library to make a similar plot
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
```
{% asset_img dv_2.png %}

上图中缺少每种植物的物种信息。我们将使用`seaborn`的`FacetGrid`按物种为散点图着色：
```python
# One piece of information missing in the plots above is what species each plant is
# We'll use seaborn's FacetGrid to color the scatterplot by species
sns.FacetGrid(iris, hue="Species", size=5).map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
```
{% asset_img dv_3.png %}

我们可以通过箱线图查看`Seaborn`中的单个特征。
```python
# We can look at an individual feature in Seaborn through a boxplot
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
```
{% asset_img dv_4.png %}

