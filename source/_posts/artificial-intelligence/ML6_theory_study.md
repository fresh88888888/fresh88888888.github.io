---
title: 机器学习(ML)(六) — 探析
date: 2024-09-24 12:24:11
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

#### 偏差和方差

开发**机器学习系统**的典型工作流程是什么？当训练**机器学习模型**时。在给定数据集，如果想用直线去**拟合**它，可能做得并不好。我们说这个算法有很高的**偏差**，或者它对这个数据集的**拟合不足**，也可以称为**欠拟合**。如果要拟合一个四阶多项式，那么它有很高的**方差**，或者称为**过拟合**。如果拟合一个二次多项式，那么它看起来相当不错。如下图所示：
<!-- more -->
{% asset_img ml_1.png %}

但是如果有很多特征，就无法绘制图形来查看，不如诊断或查找算法在训练集和交叉验证集上是否具有**高偏差**、**高方差**来的更有效。让我们看一个例子。如果要计算{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}，从上图可看出这里的{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}很高，因为真实值和预测值之间存在相当大的误差。算法在它以前没有见过的例子上也表现不佳，所以{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}会很高。具有**高偏差**（**欠拟合**）的算法的一个特征是它在训练集上的表现不是很好。当{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}很高时，这是该算法具有**高偏差**的显性指标。如果要计算{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}，它在训练集上实际上表现很好。与训练数据非常吻合。这里的{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}值会很低。但是如果你在训练集中没有的样本上评估这个模型，你会发现**交叉验证误差**({% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %})很高。你的算法具有高方差的特征签名将会使{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}比{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}高得多。换句话说，它在见过的数据上的表现比没见过的数据上要好得多。这是该算法具有**高方差**的显性指标。再次强调，这样做的目的是计算{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}和{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}，看看{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}是否很高，或者{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}是否比{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}高很多。即使无法绘制函数{% mathjax %}f{% endmathjax %}，也能了解算法是具有**高偏差**还是**高方差**。最后，如果查看{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}的值非常低，因此在训练集上表现相当不错。如果查看**交叉验证集**的示例，您会发现{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}也非常低。{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}不太高表明它没有**高偏差**问题，而{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}不比{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}高很多，这表明它也没有**高方差**的问题。总结一下，当线性多项式的{% mathjax %}d=1{% endmathjax %}时，{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}很高，{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}也很高。当{% mathjax %}d = 4{% endmathjax %}时，{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}较低，但{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}较高。当{% mathjax %}d = 2{% endmathjax %}时，两者都相当低。

如下图所示，其中横轴是{% mathjax %}d{% endmathjax %}，是拟合数据的多项式的次数。在左边，将对应{% mathjax %}d{% endmathjax %}的一个小值，比如{% mathjax %}d = 1{% endmathjax %}，这对应于拟合直线。在右边，将对应{% mathjax %}d = 4{% endmathjax %}。会发现，当拟合次数越来越高的多项式时，这里我假设不使用正则化，训练误差会趋于下降，当有一个非常简单的线性函数时，它不能很好地拟合训练数据，当拟合二次函数或三阶多项式或四阶多项式时，拟合数据越来越好。随着多项式的次数增加，{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}通常会下降。我们看到，当{% mathjax %}d = 1{% endmathjax %}时，当多项式的次数非常低时，{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}很高，因为它**欠拟合**，所以它在**交叉验证集**上表现不佳。如果改变多项式的次数，你实际上会得到一条像这样的曲线，先下降然后又上升。如果多项式的次数太低，它就会**欠拟合**，因此无法进行**交叉验证集**；如果次数太高，它就会**过拟合**，在**交叉验证集**上的表现也不会很好。只有当它处于中间位置时，它才是恰到好处的，这就是为什么我们例子中的二阶多项式最终会得到较低的**交叉验证误差**，既不会出现**高偏差**也不会出现**高方差**的原因。
{% asset_img ml_2.png %}

总结一下，你如何诊断是否具有高偏差？如果你的学习算法有**高偏差**的数据，关键指标将是{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}是否很高。这对应于曲线的最左边部分，也就是{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}很高的地方。你如何诊断是否有**高方差**？而高方差的关键指标将是{% mathjax %}\mathbf{J}_{\text{cv}}{% endmathjax %}是否大于{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}的两倍。当我们将一个非常高阶多项式拟合到这个小数据集时，就会发生这种情况。在某些情况下，同时存在**高偏差**和**高方差**是可能的。对于**线性回归**，你不会看到这种情况发生，如果训练一个神经网络，有些场景下存在**高偏差**和**高方差**。如果{% mathjax %}\mathbf{J}_{\text{train}}{% endmathjax %}很高，那么在训练集上的表现就不是很好，**交叉验证误差**远大于训练集。**高偏差**意味着在训练集上的表现都不好，而**高方差**意味着在**交叉验证集**上的表现比训练集差得多。

