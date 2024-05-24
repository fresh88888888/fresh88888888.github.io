---
title: 优化算法 (机器学习)(TensorFlow)
date: 2024-05-24 10:24:11
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

优化算法对于深度学习非常重要。一方面，训练复杂的深度学习模型可能需要数小时、几天甚至数周。优化算法的性能直接影响模型的训练效率。另一方面，了解不同优化算法的原则及其超参数的作用将使我们能够以有针对性的方式调整超参数，以提高深度学习模型的性能。
<!-- more -->
#### 优化

对于深度学习问题，我们通常会先定义损失函数。一旦我们有了损失函数，我们就可以使用优化算法来尝试最小化损失。在优化中，损失函数通常被称为优化问题的目标函数。按照传统惯例，大多数优化算法都关注的是最小化。如果我们需要最大化目标，那么有一个简单的解决方案：在目标函数前加负号即可。
##### 优化目标

尽管优化提供了一种最大限度地减少深度学习损失函数的方法，但本质上，优化和深度学习的目标是根本不同的。前者主要关注的是最小化目标，后者则关注在给定有限数据量的情况下寻找合适的模型。由于优化算法的目标函数通常是基于训练数据集的损失函数，因此优化的目标是减少训练误差。但是，深度学习（或更广义地说，统计推断）的目标是减少泛化误差。为了实现后者，除了使用优化算法来减少训练误差之外，我们还需要注意过拟合。
##### 优化面临的挑战

深度学习优化存在许多挑战。其中最令人烦恼的是局部最小值、鞍点和梯度消失。
###### 局部最小值

对于任何目标函数{% mathjax %}f(x){% endmathjax %}，如果在{% mathjax %}x{% endmathjax %}处对应的{% mathjax %}f(x){% endmathjax %}值小于在{% mathjax %}x{% endmathjax %}附近任意其他点的{% mathjax %}f(x){% endmathjax %}值，那么{% mathjax %}f(x){% endmathjax %}可能是局部最小值。如果{% mathjax %}f(x){% endmathjax %}在{% mathjax %}x{% endmathjax %}处的值是整个域中目标函数的最小值，那么{% mathjax %}f(x){% endmathjax %}是全局最小值。例如，给定函数：
{% mathjax '{"conversion":{"em":14}}' %}
f(x) = x\cdot\cos(\pi x)\;\text{for}\; -1.0\leq x\leq 2.0
{% endmathjax %}
我们可以近似该函数的局部最小值和全局最小值。
{% asset_img oa_1.png %}

深度学习模型的目标函数通常有许多局部最优解。当优化问题的数值解接近局部最优值时，随着目标函数解的梯度接近或变为零，通过最终迭代获得的数值解可能仅使目标函数局部最优，而不是全局最优。只有一定程度的噪声可能会使参数跳出局部最小值。事实上，这是小批量随机梯度下降的有利特性之一。在这种情况下，小批量上梯度的自然变化能够将参数从局部极小值中跳出。
###### 鞍点

除了局部最小值外，鞍点是 梯度消失的另一个原因**鞍点**(`saddle point`)是指函数的所有梯度都消失但既不是全局最小值也不是局部最小值的任何位置。考虑这个函数{% mathjax %}f(x) = x^3{% endmathjax %}。它的一阶和二阶导数在{% mathjax %}x = 0{% endmathjax %}时消失。这时优化可能会停止，尽管它不是最小值。我们假设函数的输入是{% mathjax %}k{% endmathjax %}维向量，其输出是标量，因此其`Hessian`矩阵（也称黑塞矩阵）将有{% mathjax %}k{% endmathjax %}个特征值。函数的解可能是局部最小值、局部最大值或函数梯度为零位置处的鞍点：
- 当函数在零梯度位置处的`Hessian`矩阵的特征值全部为正值时，我们有该函数的局部最小值；
- 当函数在零梯度位置处的`Hessian`矩阵的特征值全部为负值时，我们有该函数的局部最大值；
- 当函数在零梯度位置处的`Hessian`矩阵的特征值为负值和正值时，我们有该函数的一个鞍点。

对于高维度问题，至少部分特征值为负的可能性相当高。这使得鞍点比局部最小值更有可能出现。简而言之，**凸函数**是`Hessian`函数的特征值永远不为负值的函数。不幸的是，大多数深度学习问题并不属于这一类。尽管如此，它还是研究优化算法的一个很好的工具。
###### 梯度消失

可能遇到的最隐蔽问题是梯度消失。回想一下常用的激活函数及其衍生函数。例如，假设我们想最小化函数{% mathjax %}f(x) = \tanh(x){% endmathjax %}，然后我们恰好从{% mathjax %}x=4{% endmathjax %}开始。正如我们所看到的那样，{% mathjax %}f{% endmathjax %}的梯度接近零。更具体地说，{% mathjax %}f'(x) = 1 - \tanh_2 (x){% endmathjax %}，因此是{% mathjax %}f'(4) = 0.0013{% endmathjax %}。因此，在我们取得进展之前，优化将会停滞很长一段时间。事实证明，这是在引入`ReLU`激活函数之前训练深度学习模型相当棘手的原因之一。
{% asset_img oa_2.png %}

正如我们所看到的那样，深度学习的优化充满挑战。幸运的是，有一系列强大的算法表现良好，即使对于初学者也很容易使用。此外，没有必要找到最优解。局部最优解或其近似解仍然非常有用。
##### 总结

最小化训练误差并不能保证我们找到最佳的参数集来最小化泛化误差。优化问题可能有许多局部最小值。一个问题可能有很多的鞍点，因为问题通常不是凸的。梯度消失可能会导致优化停滞，重参数化通常会有所帮助。对参数进行良好的初始化也可能是有益的。

#### 凸性

**凸性**(`convexity`)在优化算法的设计中起到至关重要的作用，这主要是由于在这种情况下对算法进行分析和测试要容易。换言之，如果算法在凸性条件设定下的效果很差，那通常我们很难在其他条件下看到好的结果。此外，即使深度学习中的优化问题通常是非凸的，它们也经常在局部极小值附近表现出一些凸性。这可能会产生一些像这样比较有意思的新优化变体。
##### 定义

在进行凸分析之前，我们需要定义**凸集**(`convex sets`)和**凸函数**(`convex functions`)。

###### 凸集

**凸集**(`convex set`)是凸性的基础。简单地说，如果对于任何{% mathjax %}a,b\in \mathcal{X}{% endmathjax %}，连接{% mathjax %}a{% endmathjax %}和{% mathjax %}b{% endmathjax %}的线段也位于{% mathjax %}\mathcal{X}{% endmathjax %}中，则向量空间中的一个集合{% mathjax %}\mathcal{X}{% endmathjax %}是**凸**(`convex`)的。在数学术语上，这意味着对于所有{% mathjax %}\lambda\in [0,1]{% endmathjax %}，我们得到：
{% mathjax '{"conversion":{"em":14}}' %}
\lambda a + (1-\lambda)b\in \mathcal{X}\;a,b\in \mathcal{X}
{% endmathjax %}
这听起来有点抽象，那我们来看下图里的例子。第一组存在不包含在集合内部的线段，所以该集合是非凸的，而另外两组则没有这样的问题。
{% asset_img oa_3.png "第一组是非凸的，另外两组是凸的" %}

接下来看一下交集，如下图所示。假设{% mathjax %}\mathcal{X}{% endmathjax %}和{% mathjax %}\mathcal{Y}{% endmathjax %}是凸集，那么{% mathjax %}\mathcal{X}\cap \mathcal{Y}{% endmathjax %}也是凸集的。现在考虑任意的{% mathjax %}a,b\in \mathcal{X}\cap \mathcal{Y}{% endmathjax %}，因为{% mathjax %}\mathcal{X}{% endmathjax %}和{% mathjax %}\mathcal{Y}{% endmathjax %}是凸集，所以连接{% mathjax %}a{% endmathjax %}和{% mathjax %}b{% endmathjax %}的线段包含在{% mathjax %}\mathcal{X}{% endmathjax %}和{% mathjax %}\mathcal{Y}{% endmathjax %}当中。鉴于此，它们也需要包含在{% mathjax %}\mathcal{X}\cap \mathcal{Y}{% endmathjax %}中，从而证明我们的定理。
{% asset_img oa_4.png "两个凸集的交集是凸的" %}

我们可以毫不费力进一步得到这样的结果：给定凸集{% mathjax %}\mathcal{X}_i{% endmathjax %}，它们的交集{% mathjax %}\cap_i\mathcal{X}_i{% endmathjax %}是凸的，但是反向是不正确的，考虑两个不想交的集合{% mathjax %}\mathcal{X}\cap \mathcal{Y} = \emptyset{% endmathjax %}，取{% mathjax %}a\in \mathcal{X}{% endmathjax %}和{% mathjax %}b\in \mathcal{Y}{% endmathjax %}。因为我们假设{% mathjax %}\mathcal{X}\cap \mathcal{Y} = \emptyset{% endmathjax %}，在下图中连接{% mathjax %}a{% endmathjax %}和{% mathjax %}b{% endmathjax %}的线段需要包含一部分既不在{% mathjax %}\mathcal{X}{% endmathjax %}，也不在{% mathjax %}\mathcal{Y}{% endmathjax %}中。因此线段也不在{% mathjax %}\mathcal{X}\cup \mathcal{Y}{% endmathjax %}中，因此证明了凸集的并集不一定是凸的，即**非凸**(`nonconvex`)的。
{% asset_img oa_5.png "两个凸集的并集不一定是凸的" %}

通常，深度学习的问题通常是在凸集上定义的。例如，{% mathjax %}\mathbb{R}^d{% endmathjax %}，即实数的{% mathjax %}d{% endmathjax %}维向量的集合是凸集（毕竟{% mathjax %}\mathbb{R}^d{% endmathjax %}中任意两点之间的线存在{% mathjax %}\mathbb{R}^d{% endmathjax %}）中。在某些情况下，我们使用有界长度的变量，例如球的半径定义为{% mathjax %}\{\mathbf{x}|mathbf{x}\in \mathbb{R}^d\;且\;\lVert \mathbf{x} \rVert \leq r\}{% endmathjax %}。
###### 凸函数

现在我们有了凸集，我们可以引入**凸函数**(`convex function`){% mathjax %}f{% endmathjax %}。给定一个凸集{% mathjax %}\mathcal{X}{% endmathjax %}，如果对于所有{% mathjax %}x,x'\in \mathcal{X}{% endmathjax %}和所有{% mathjax %}\lambda\in [0,1]{% endmathjax %}，函数{% mathjax %}f:\mathcal{X}\rightarrow \mathbb{R}{% endmathjax %}是凸的，我们可以得到：
{% mathjax '{"conversion":{"em":14}}' %}
\lambda f(x) + (1 - \lambda)f(x')\geq f(\lambda x + (1 - \lambda)x')
{% endmathjax %}
```python
f = lambda x: 0.5 * x**2  # 凸函数
g = lambda x: np.cos(np.pi * x)  # 非凸函数
h = lambda x: np.exp(0.5 * x)  # 凸函数
```
{% asset_img oa_6.png %}

不出所料，余弦函数为非凸的，而抛物线函数和指数函数为凸的。请注意，为使该条件有意义，{% mathjax %}\mathcal{X}{% endmathjax %}是凸集的要求是必要的。否则可能无法很好地界定{% mathjax %}f(\lambda x + (1 - \lambda)x'){% endmathjax %}的结果。
###### 詹森不等式

给定一个凸函数{% mathjax %}f{% endmathjax %}，最有用的数学工具之一就是**詹森不等式**(`Jensen’s inequality`)。它是凸性定义的一种推广：
{% mathjax '{"conversion":{"em":14}}' %}
\sum_i \alpha_i f(x_i)\geq f(\sum_i \alpha_ix_i)\;\text{and}\;E_X[f(X)]\geq f(E_X[X])
{% endmathjax %}
其中{% mathjax %}\alpha_i{% endmathjax %}是满足{% mathjax %}\sum_i \alpha_i = 1{% endmathjax %}的非负实数，{% mathjax %}X{% endmathjax %}是随机变量。换句话说凸函数的期望不小于期望的凸函数，其中后者是一个更简单的表达式，为了证明第一个不等式，我们多次将凸性的定义应用于一次求和的一项。詹森不等式的一个常见应用：用一个较简单的表达式约束一个较复杂的表达式。例如，它可以应用于部分观察到的随机变量的对数似然。具体地说，由于{% mathjax %}\int P(Y)P(X|Y)dY = P(X){% endmathjax %}，所以：
{% mathjax '{"conversion":{"em":14}}' %}
E_{Y\sim P(Y)}[-\log P(X|Y)]\geq -\log P(X)
{% endmathjax %}
这里，{% mathjax %}Y{% endmathjax %}是典型的未观察到的随机变量，{% mathjax %}P(Y){% endmathjax %}是它可能如何分布的最佳猜测，{% mathjax %}P(X){% endmathjax %}是将{% mathjax %}Y{% endmathjax %}积分后的分布。例如，再聚类中{% mathjax %}Y{% endmathjax %}可能是簇标签，而在应用簇标签时，{% mathjax %}P(X|Y){% endmathjax %}是生成模型。
##### 性质

###### 局部极小值是全局极小值

首先凸函数的局部极小值也是全局极小值。下面我们用反证法给出证明。