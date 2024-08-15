---
title: 数据科学 — 数学(四)（机器学习）
date: 2024-08-15 11:26:11
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

#### 主成分分析(PCA)

现在您已经了解了**投影**的概念，让我们看看`PCA`如何使用它来降低数据集的维度。如下图所示，每个点代表一个不同的观测值，由两个以{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}位置为图形的特征​​组成。降低此数据的维度意味着将图形为平面点的二维数据转变为图形为一条线的一维数据。该集合不是以原点`(0,0)`为中心，现在让我们看看如果投影到{% mathjax %}x{% endmathjax %}轴上会发生什么？
<!-- more -->
{% asset_img m_1.png  %}

您已经可以看到，此投影中的点分布较少，因为这些点彼此更接近。您可以将它们投影到任何一条线上，例如这条线，可以用方程：{% mathjax %}y +  x = 0{% endmathjax %}来表示，这相当于投影到向量`(1,-1)`上，因为这些点跨越了这条线。
{% asset_img m_2.png  %}

****

{% asset_img m_3.png  %}

或者你也可以考虑另一条线，它求解方程：{% mathjax %}2x -  y = 0{% endmathjax %}，它投影到向量`(1,2)`上。这条线与数据非常吻合，并且得到的投影点仍然相对分散。经过这些投影后，数据点可能会或多或少地分散，而这最终会变得非常重要。**原因是分散程度更高的数据点会保留原始数据集中的更多信息。换句话说，保留更多的分散度意味着保留更多信息**。现在，我将按点分散程度从大到小对投影进行排序。**顶部的投影点分散程度最高，因此它保留了原始数据集中最多的信息，底部的投影点分散程度最低，因此保留的信息最少**。因此，`PCA`的目标是**找到即使在降低数据集维度的情况下也能保留数据中最大可能分散度的投影**。
{% asset_img m_4.png  %}

****

{% asset_img m_5.png  %}

****

{% asset_img m_6.png  %}

再次强调，**降维**和`PCA`的好处如下。降维使数据集更容易管理，因为它们更小。`PCA`允许您在减少维度的同时最大限度地减少信息丢失。由于维度的减少，实现了不可能的方式分析可视化数据。

##### 方差(Variance)和协方差(Covariance)

考虑这个数据集，如下图所示，每个点都是由两个变量{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}组成的观测值。每个点位于{% mathjax %}(x_i,y_i){% endmathjax %}的位置。数据的**均值**是所有观测值的**平均值**，这些观测值将位于该点附近的某个位置。对于{% mathjax %}x{% endmathjax %}变量，将{% mathjax %}x{% endmathjax %}的所有`n`个值相加并除以`n`，{% mathjax %}y{% endmathjax %}的均值以相同的方式，您只需对每个特征的值取平均值即可。因此，这个中间点的坐标为{% mathjax %}x{% endmathjax %}的均值和{% mathjax %}y{% endmathjax %}的均值。
{% asset_img m_7.png  %}

{% mathjax '{"conversion":{"em":14}}' %}
\text{Mean}(x) = \frac{1}{n}\sum^n_{i=1} x_i
{% endmathjax %}
{% mathjax '{"conversion":{"em":14}}' %}
\text{Mean}(y) = \frac{1}{n}\sum^n_{i=1} y_i
{% endmathjax %}

接下来，您将了解**方差**的概念，**它描述了数据的分散程度**。如果你想描述这些点在图表中是如何出现的，你可能会说，如果我们沿着横轴看这些点，它们会更分散或分布得更大，而如果我们沿着纵轴看这些点，它们会更紧凑，分布得更小。在统计学中，**分布是用数据集的方差来衡量或描述的**。没有值分布的数据集的方差为`0`，而分布较大的数据集的方差较大。为了更清楚地看到这一点，我将使用二维图表，并将每个点移动到横轴上，如下图所示。
{% asset_img m_8.png  %}

不用担心方差是如何计算的，我们可以看到，水平{% mathjax %}x{% endmathjax %}轴上的方差相对较大。如果沿{% mathjax %}y{% endmathjax %}轴，值会分布在较小的范围内。{% mathjax %}y{% endmathjax %}方差相对较小，但仍大于零，因为值存在一些变化。
{% asset_img m_9.png  %}

请考虑一个非常简单的数据集，其中一列表示变量{% mathjax %}x{% endmathjax %}，五行编号为 `1-5`。每行都有一个观测值{% mathjax %}x_i{% endmathjax %}。首先，您需要将{% mathjax %}x_i{% endmathjax %}的值相加并除以`5`来计算{% mathjax %}x{% endmathjax %}的平均值。这样一来，总数为`45`，平均值为`9`。接下来，找出每个值和列{% mathjax %}x_i{% endmathjax %}与刚刚计算的平均值之间的差值，换句话说，只需从每行中减去`9`。现在求每个差值的平方，将结果放在一个新列中。求和将{% mathjax %}5^2{% endmathjax %}个值相加，得到总数为`64`，最后将这个总数除以{% mathjax %}n-1{% endmathjax %}，因为这里的`n`是`5`，我们除以`4`，得到方差`16`。
{% asset_img m_10.png  %}

方差通常缩写为{% mathjax %}\text{var}{% endmathjax %}，并使用希腊字母{% mathjax %}\mu{% endmathjax %}表示**平均值**。另一种思考方差的方式是将其视为与平均值的**平均平方距离**。但这里最重要的一点是，随着数据变得更加分散并且平均值离平均值越来越远，方差就会增加。回到之前的数据集，现在您将该平均点称为{% mathjax %}\mu_x{% endmathjax %}、{% mathjax %}\mu_y{% endmathjax %}。可以使用刚刚回顾的公式计算{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}的方差。{% mathjax %}x{% endmathjax %}方差大于{% mathjax %}y{% endmathjax %}方差，从该公式可以清楚地知道为什么会这样。沿{% mathjax %}x{% endmathjax %}轴构建，这些点的距离{% mathjax %}\mu_x{% endmathjax %}较远，因此平均平方距离也会较大。同时，沿{% mathjax %}y{% endmathjax %}轴，与{% mathjax %}\mu_Y{% endmathjax %}的平均平方距离较小。
{% asset_img m_11.png  %}

**方差**有助于我们量化数据的**分散程度**，但现在考虑单靠方差无济于事的情况。这两个数据集各有三个观测值。它们具有相同的{% mathjax %}y{% endmathjax %}方差和{% mathjax %}x{% endmathjax %}方差，但很明显，这些数据集的模式存在显著差异。解决方案是一个称为**协方差的度量**。**协方差**有助于衡量数据集的两个特征如何相互变化。请注意，在左侧数据集中，数据内的模式是向下和向右，因为{% mathjax %}x{% endmathjax %}值增加，{% mathjax %}y{% endmathjax %}值减少。在右侧数据集中，因为{% mathjax %}x{% endmathjax %}值变大，{% mathjax %}y{% endmathjax %}值也变大。协方差量化了这种关系，导致左侧数据集具有**负协方差**，而右侧数据集具有**正协方差**。
{% asset_img m_12.png  %}

现在您对它测量的内容有了很高的理解，让我们看看**协方差**是如何计算的。协方差方程如下所示。
{% asset_img m_13.png  %}

一开始可能有点复杂，所以我会把它分解成几个部分。但是，正如您所见，它看起来与**方差方程**非常相似，如果您在最后展开平方项，它们几乎相同。唯一的区别是，圆圈内的项现在取决于{% mathjax %}x{% endmathjax %}和 {% mathjax %}y{% endmathjax %}变量的值以及这些值的平均值{% mathjax %}\mu_x{% endmathjax %}和{% mathjax %}\mu_y{% endmathjax %}。让我们看看这三个示例数据集，以了解这个方程的工作原理。对于第一个数据集，我们期望计算出**负协方差**，因为数据呈下降趋势。在第二种情况下，**拟合观测值的趋势线似乎是平坦的**，因此我们期望协方差为零或非常小的值。在第三种情况下，{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}似乎一起呈上升趋势，这应该导致**正协方差**。您可以在每个数据集的顶部绘制均值点{% mathjax %}\mu_x{% endmathjax %}、{% mathjax %}\mu_y{% endmathjax %}。从每个{% mathjax %}x{% endmathjax %}中减去{% mathjax %}\mu_x{% endmathjax %}，从每个{% mathjax %}y{% endmathjax %}中减去{% mathjax %}\mu_y{% endmathjax %}，本质上会使数据围绕该点重新居中。您可以想象它将平面分成四个象限。数据集中的每个点都位于其中一个象限中，并对整体协方差产生正或负贡献。在第一个象限中，{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}都大于它们的均值，因此圈出项中的乘积为正。在第二象限中，{% mathjax %}x{% endmathjax %}小于其平均值，但{% mathjax %}y{% endmathjax %}仍然大于平均值。带圆圈的项现在是负数和正数的乘积，结果为负数。在第三象限中，{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}都小于其平均值。带圆圈的项现在是两个负数的乘积，因此结果为正数；对于最后一个象限中的点，{% mathjax %}x{% endmathjax %}大于其平均值，但{% mathjax %}y{% endmathjax %}较小。带圆圈的项是正数和负数的乘积，结果为负数。

您可以将**协方差**视为平均，正象限中的点更多还是负象限中的点更多?对于第一个象限，负象限中的点更多，因此协方差将为负数。在第二个象限中，点在正象限和负象限之间大致相等，这导致协方差接近于零。在第三个象限中，大多数点位于正象限，导致协方差为正。无论您是否觉得自己完全理解了协方差方程，此时最重要的是直观地了解它们测量的内容。您可以将**协方差视为测量两个变量之间关系的方向。负协方差表示负趋势，小协方差表示平缓趋势或无关系，正协方差表示正趋势**。

##### 协方差矩阵

