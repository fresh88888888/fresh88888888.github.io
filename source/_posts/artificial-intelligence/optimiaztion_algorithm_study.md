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

通常，深度学习的问题通常是在凸集上定义的。例如，{% mathjax %}\mathbb{R}^d{% endmathjax %}，即实数的{% mathjax %}d{% endmathjax %}维向量的集合是凸集（毕竟{% mathjax %}\mathbb{R}^d{% endmathjax %}中任意两点之间的线存在{% mathjax %}\mathbb{R}^d{% endmathjax %}）中。在某些情况下，我们使用有界长度的变量，例如球的半径定义为{% mathjax %}\{\mathbf{x}|\mathbf{x}\in \mathbb{R}^d\;且\;\lVert \mathbf{x} \rVert \leq r\}{% endmathjax %}。
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

首先凸函数的局部极小值也是全局极小值。下面我们用反证法给出证明。假设{% mathjax %}x^{\ast}\in \mathcal{X}{% endmathjax %}是一个局部最小值，则存在一个很小的正值{% mathjax %}p{% endmathjax %}，使得当{% mathjax %}x\in \mathcal{X}{% endmathjax %}满足{% mathjax %}0 < |x - x^{\ast}| \leq p{% endmathjax %}时，有f{% mathjax %}f(x^{\ast}) < f(x){% endmathjax %}。现在假设局部极小值{% mathjax %}x^{\ast}{% endmathjax %}不是{% mathjax %}f{% endmathjax %}的全局极小值：存在{% mathjax %}x'\in \mathcal{X}{% endmathjax %}使得{% mathjax %}f(x') < f(x^{\ast}){% endmathjax %}。则存在{% mathjax %}\lambda \in [0,1){% endmathjax %}，比如{% mathjax %}\lambda = 1 - \frac{p}{|x^{\ast} - x'|}{% endmathjax %}，使得{% mathjax %}0 < |\lambda x^{\ast} + (1 - \lambda)x' - x^{\ast}| \leq p{% endmathjax %}。然而根据凸性的性质有：
{% mathjax '{"conversion":{"em":14}}' %}
f(\lambda x^{\ast} + (1 - \lambda)x') \leq \lambda f(x^{\ast}) + (1 - \lambda)f(x') < \lambda f(x^{\ast}) + (1-\lambda)f(x^{\ast}) = f(x^{\ast})
{% endmathjax %}
这与{% mathjax %}x^{\ast}{% endmathjax %}是局部最小值相矛盾。因此，不存在{% mathjax %}x'\in \mathcal{X}{% endmathjax %}满足{% mathjax %}f(x') < f(x^{\ast}){% endmathjax %}。综上所述，局部最小值{% mathjax %}x^{\ast}{% endmathjax %}也是全局最小值。例如，对于凸函数{% mathjax %}f(x) = (x - 1)^2{% endmathjax %}，有一个局部最小值{% mathjax %}x = 1{% endmathjax %}，它也是全局最小值。
{% asset_img oa_7.png %}

凸函数的局部最小值也是全局最小值这一性质很方便。这意味着如果我们最小化函数，我们就不会“卡住”。但是请注意，这并不能意味着不能有多个全局最小值，或者可能不存在一个全局最小值。例如，函数{% mathjax %}f(x) = \max(|x| - 1,0){% endmathjax %}在{% mathjax %}[-1,1]{% endmathjax %}区间上的都是最小值。相反，函数{% mathjax %}f(x) = \exp(x){% endmathjax %}在{% mathjax %}\mathbb{R}{% endmathjax %}上没有取得最小值。对于{% mathjax %}x\rightarrow -\infty{% endmathjax %}，它趋近于0，但是没有{% mathjax %}f(x) = 0{% endmathjax %}的{% mathjax %}x{% endmathjax %}。
###### 凸函数的下水平集是凸的

我们可以方便地通过凸函数的**下水平集**(`below sets`)定义凸集。具体来说，给定一个定义在凸集{% mathjax %}\mathcal{X}{% endmathjax %}上的凸函数{% mathjax %}f{% endmathjax %}，其任意一个下水平集：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{S}_b := \{x|x\in \mathcal{X}\;\text{and}\;f(x)\leq b\}
{% endmathjax %}
是凸的。让我们快速证明一下。对于任何{% mathjax %}x,x'\in \mathcal{S}_b{% endmathjax %}，我们需要证明：当{% mathjax %}\lambda\in[0,1]{% endmathjax %}时，{% mathjax %}\lambda x + (1 - \lambda)x'\in \mathcal{S}_b{% endmathjax %}。因为{% mathjax %}f(x) \leq b{% endmathjax %}且{% mathjax %}f(x') \leq b{% endmathjax %}，所以：
{% mathjax '{"conversion":{"em":14}}' %}
f(\lambda x + (1 - \lambda)x') \leq \lambda f(x) + (1 - \lambda)f(x') \leq b
{% endmathjax %}
###### 凸性和二阶导数

当一个函数的二阶导数{% mathjax %}f:\mathbb{R}^n\rightarrow \mathbb{R}{% endmathjax %}存在时，我们很容易检查这个函数的凸性。我们需要做的就是检查{% mathjax %}\nabla^2 f\succeq 0{% endmathjax %}，即对于所有{% mathjax %}\mathbf{x}\in \mathbb{R}^n, \mathbf{x}^{\mathsf{T}}\mathbf{Hx}\geq 0{% endmathjax %}例如，函数{% mathjax %}f(x) \frac{1}{2}\lVert\mathbf{x}\rVert^2{% endmathjax %}是凸的，因为{% mathjax %}\succeq^2 f = 1{% endmathjax %}，即其导数是单位矩阵。更正式地讲，{% mathjax %}f{% endmathjax %}为凸函数，当且仅当任意二次可微一维函数{% mathjax %}f:\mathcal{R}^n\rightarrow\mathcal{R}{% endmathjax %}，它是凸的当且仅当它的{% mathjax %}\text{Hession}\nabla^2 f\succeq 0{% endmathjax %}。首先我们来证明一下一维情况。为了证明凸函数的{% mathjax %}f''(x)\geq 0{% endmathjax %}，我们使用：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{1}{2}f(x + \epsilon) + \frac{1}{2}f(x - \epsilon) \geq f(\frac{x + \epsilon}{2}, frac{x - \epsilon}{2}) = f(x)
{% endmathjax %}
因为二阶导数是由有限差分的极限给出的，所以遵循：
{% mathjax '{"conversion":{"em":14}}' %}
f''(x) = \lim_{\epsilon\rightarrow 0} \frac{f(x + \epsilon) + f(x - \epsilon) - 2f(x)}{\epsilon^2} \geq 0
{% endmathjax %}
为了证明{% mathjax %}f''\geq 0{% endmathjax %}可以推导{% mathjax %}f{% endmathjax %}是凸的，我们使用这样一个事{% mathjax %}f''\geq 0{% endmathjax %}意味着{% mathjax %}f'{% endmathjax %}是一个单调的非递减函数。假设{% mathjax %}a < x < b{% endmathjax %}是{% mathjax %}\mathbb{R}{% endmathjax %}中的三个点，其中，{% mathjax %}x = (1 - \lambda)a + \lambda b{% endmathjax %}且{% mathjax %}\lambda\in (0,1){% endmathjax %}，根据中值定理，存在{% mathjax %}\alpha\in [a,x]，\beta\in [x,b]{% endmathjax %}，使得
{% mathjax '{"conversion":{"em":14}}' %}
f'(\alpha) = \frac{f(x) - f(a)}{x - a}\;且\;f'(\beta) \frac{f(b) - f(x)}{b - x}
{% endmathjax %}
通过单调性{% mathjax %}f'(\beta)\geq f'(\alpha){% endmathjax %}，因此
{% mathjax '{"conversion":{"em":14}}' %}
\frac{x - a}{b - a}f(b) + frac{b - x}{b - a}f(a)\geq f(x)
{% endmathjax %}
由于{% mathjax %}x = (1 - \lambda)a + \lambda b{% endmathjax %}，所以
{% mathjax '{"conversion":{"em":14}}' %}
\lambda f(b) + (1 - \lambda)f(a) \geq f((1 - \lambda)a + \lambda b)
{% endmathjax %}
从而证明了凸性。第二，我们需要一个引理证明多维情况：{% mathjax %}f:\mathbb{R}^n\rightarrow \mathbb{R}{% endmathjax %}是凸的当且仅当对于所有{% mathjax %}\mathbf{x},\mathbf{y}\in \mathbb{R}^n{% endmathjax %}
{% mathjax '{"conversion":{"em":14}}' %}
g(z) = \underset{=}{\text{def}}f(z\mathbf{x} + (1 - z)\mathbf{y})\;\text{where}\;z\in [0,1]
{% endmathjax %}
是凸的，为了证明{% mathjax %}f{% endmathjax %}的凸性意味着{% mathjax %}g{% endmathjax %}是凸的，我们可以证明，对于所有的{% mathjax %}a,b,\lambda \in [0,1]{% endmathjax %}（这样有{% mathjax %}0\leq \lambda a + (1 - \lambda)b \leq 1{% endmathjax %}）：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
&g(\lambda a + (1 - \lambda)b)  \\ 
&= f((\lambda a + (1 - \lambda)b)\mathbf{x} + (1 - \lambda a - (1 - \lambda)b)\mathbf{y}) \\ 
&= f(\lambda(a\mathbf{x} + (1 - a)\mathbf{y}) + (1 - \lambda)(b\mathbf{x} + (1 - b)\mathbf{y})) \\
&\leq \lambda f(a\mathbf{x} + (1 - a)\mathbf{y}) + (1 - \lambda)f(b\mathbf{x} + (1 - b)\mathbf{y}) \\
&= \lambda g(a) + (1 -\lambda)g(b)
\end{align}
{% endmathjax %}
为了证明这一点，我们可以证明对{% mathjax %}[0,1]{% endmathjax %}中的所有{% mathjax %}\lambda{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& f(\lambda\mathbf{x} + (1 - \lambda)\mathbf{y}) \\ 
&= g(\lambda\cdot 1 + (1 - \lambda)\cdot 0) \\ 
&\leq \lambda g(1) + (1 - \lambda)g(0) \\
&= \lambda f(\mathbf{x}) + (1 - \lambda)f(\mathbf{y}) \\
\end{align}
{% endmathjax %}
最后，利用上面的引理和一维情况的结果，我们可以证明多维情况：多维函数{% mathjax %}f:\mathbb{R}^n\rightarrow \mathbb{R}{% endmathjax %}是凸函数，当且仅当{% mathjax %}g(z)\underset{=}{\text{def}} = f(x\mathbf{x} + (1 - z)\mathbf{y}){% endmathjax %}是凸的，这里{% mathjax %}z\in[0,1]，\mathbf{x},\mathbf{y}\in \mathbb{R}^n{% endmathjax %}。根据一维情况，此条成立的条件为，当且仅当对于所有{% mathjax %}\mathbf{x},\mathbf{y}\in \mathbb{R}^n, g'' = (\mathbf{x} - \mathbf{y})^{\mathsf{T}}\mathbf{H}(\mathbf{x} - \mathbf{y})\geq 0 \;(\mathbf{H}\underset{=}{\text{def}} = \nabla^2 f){% endmathjax %}。这相当于根据半正定矩阵的定义，{% mathjax %}\mathbf{H}\succeq 0{% endmathjax %}。
##### 约束

凸优化的一个很好的特性是能够让我们有效地处理**约束**(`constraints`)。即它使我们能够解决以下形式的**约束优化**(`constrained optimization`)问题：
{% mathjax '{"conversion":{"em":14}}' %}
\underset{x}{\text{minimize}}f(x)\;\text{subject to}\;c_i(\mathbf{x})\leq 0\;\text{for all}\; i\in \{,\ldots, N\}
{% endmathjax %}
这里{% mathjax %}f{% endmathjax %}是目标函数，{% mathjax %}c_i{% endmathjax %}是约束函数。例如第一个约束{% mathjax %}c_1(\mathbf{x}) = \lVert\mathbf{x}\rVert_2 - 1{% endmathjax %}，则参数{% mathjax %}\mathbf{x}{% endmathjax %}被限制为单位球。如果第二个约束{% mathjax %}c_2(\mathbf{x}) = \mathbf{v}^{\mathsf{T}}\mathbf{x} + b{% endmathjax %}，那么这对应于半空间上所有的{% mathjax %}\mathbf{x}{% endmathjax %}。同时满足这两个约束等于选择一个球的切片作为约束集。
###### 拉格朗日函数

通常，求解一个有约束的优化问题是困难的，解决这个问题的一种方法来自物理中相当简单的直觉。想象一个球在一个盒子里，球会滚到最低的地方，重力将与盒子两侧对球施加的力平衡。简而言之，目标函数（即重力）的梯度将被约束函数的梯度所抵消（由于墙壁的“推回”作用，需要保持在盒子内）。请注意，任何不起作用的约束（即球不接触壁）都将无法对球施加任何力。这里我们简略拉格朗日函数{% mathjax %}L{% endmathjax %}的推导，上述推理可以通过以下鞍点优化问题来表示：
{% mathjax '{"conversion":{"em":14}}' %}
L(\mathbf{x},\alpha_1,\ldots,\alpha_n) = f(\mathbf{x}) + \sum_{i=1}^n \alpha_ic_i(\mathbf{x})\;\text{where}\;\alpha_i \geq 0
{% endmathjax %}
这里的变量{% mathjax %}\alpha_i\;(i = 1,\ldots,n){% endmathjax %}是所谓的拉格朗日乘数(`Lagrange multipliers`)，它确保约束被正确的执行。选择它们的大小足以确保所有{% mathjax %}i{% endmathjax %}的{% mathjax %}c_i(\mathbf{x})\leq 0{% endmathjax %}。例如，对于{% mathjax %}c_i \leq 0{% endmathjax %}中任意{% mathjax %}\mathbf{x}{% endmathjax %}，我们最终会选择{% mathjax %}\alpha_i = 0{% endmathjax %}。此外，这是一个**鞍点**(`saddlepoint`)优化问题。在这个问题中我们想要使{% mathjax %}L{% endmathjax %}相对于{% mathjax %}\alpha_i{% endmathjax %}最大化(`maximize`)，同时使它相对于{% mathjax %}\mathbf{x}{% endmathjax %}最小化。有大量的文献解释如何得出函数{% mathjax %}L(\mathbf{x},\alpha_1,\ldots,\alpha_n){% endmathjax %}，我这里只需要知道{% mathjax %}L{% endmathjax %}的鞍点是原始约束优化问题的最优解就足够了。
###### 惩罚

一种至少近似地满足约束优化问题的方法是采用拉格朗日函数{% mathjax %}L{% endmathjax %}。除了满足{% mathjax %}c_i(\mathbf{x})\leq 0{% endmathjax %}之外，我们只需将{% mathjax %}\alpha_ic_i(\mathbf{x}){% endmathjax %}添加到目标函数{% mathjax %}f(x){% endmathjax %}。这样可以确保不会严重违反约束。事实上，我们一直在使用这个技巧。在目标函数中加入{% mathjax %}\frac{\lambda}{2}|\mathbf{w}|^2{% endmathjax %}，以确保{% mathjax %}\mathbf{w}{% endmathjax %}不会增长太大。使用约束优化的观点，我们可以看到，对于若干半径{% mathjax %}r{% endmathjax %}，这将确保{% mathjax %}|\mathbf{w}|^2 - r^2 \leq 0{% endmathjax %}。通过调整{% mathjax %}\lambda{% endmathjax %}的值，我们可以改变{% mathjax %}\mathbf{w}{% endmathjax %}的大小。通常，添加惩罚是确保近似满足约束的一种好方法。在实践中，这被证明比精确的满意度更可靠。此外，对于非凸问题，许多使精确方法在凸情况下的性质（例如，可求最优解）不再成立。
###### 投影

满足约束条件的另一种策略是**投影**(`projections`)。同样，我们之前也遇到过，我们通过：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{g}\leftarrow \mathbf{g}\cdot\min (1,\theta/\lVert\mathbf{g}\rVert)
{% endmathjax %}
确保梯度的长度以{% mathjax %}\theta{% endmathjax %}为界限。这就是在半径为{% mathjax %}\theta{% endmathjax %}的球上的投影(`projection`)。更泛化地说，在凸集{% mathjax %}\mathcal{X}{% endmathjax %}上的投影被定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\text{Proj}_{\mathcal{X}}(\mathbf{x}) = \underset{x'\in \mathcal{X}}{\text{argmin}}\lVert \mathbf{x} - \mathbf{x}'\rVert
{% endmathjax %}
{% asset_img oa_8.png "Convex Projections" %}

图中有两个凸集，一个圆和一个菱形。两个集合内的点（黄色）在投影期间保持不变。两个集合（黑色）之外的点投影到集合中接近原始点（黑色）的点（红色）。虽然对{% mathjax %}L_2{% endmathjax %}的球面来说，方向保持不变，但一般情况下不需要这样。凸投影的一个用途是计算稀疏权重向量。在本例中，我们将权重向量投影到一个{% mathjax %}L_1{% endmathjax %}的球上，这是上图中菱形例子的一个广义版本。
##### 总结

在深度学习的背景下，凸函数的主要目的是帮助我们详细了解优化算法。我们由此得出梯度下降法和随机梯度下降法是如何相应推导出来的。凸集的交点是凸的，并集不是。根据詹森不等式，“一个多变量凸函数的总期望值”大于或等于“用每个变量的期望值计算这个函数的总值“。一个二次可微函数是凸函数，当且仅当其`Hessian`（二阶导数矩阵）是半正定的。凸约束可以通过拉格朗日函数来添加。在实践中，只需在目标函数中加上一个惩罚就可以了。投影映射到凸集中最接近原始点的点。

#### 梯度下降

尽管**梯度下降**(`gradient descent`)很少直接用于深度学习，但了解它是理解随机梯度下降算法的关键。例如，由于学习率过大，优化问题可能会发散，这种现象早已在梯度下降中出现。同样地，**预处理**(`preconditioning`)是梯度下降中的一种常用技术，还被沿用到更高级的算法中。让我们从简单的一维梯度下降开始。
##### 一维梯度下降

为什么梯度下降算法可以优化目标函数？一维中的梯度下降给我们很好的启发。考虑一类连续可微实值函数{% mathjax %}f:\mathbb{R}\rightarrow \mathbb{R}{% endmathjax %}，利用泰勒展开，我们可以得到：
{% mathjax '{"conversion":{"em":14}}' %}
f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2)
{% endmathjax %}
即在一阶近似中，{% mathjax %}f(x + \epsilon){% endmathjax %}可通过{% mathjax %}x{% endmathjax %}处的函数值{% mathjax %}f(x){% endmathjax %}和一阶导数{% mathjax %}f'(x){% endmathjax %}得出。我们可以假设在负梯度方向上移动的{% mathjax %}\epsilon{% endmathjax %}会减少{% mathjax %}f{% endmathjax %}。为了简单起见，我们选择固定步长{% mathjax %}\eta > 0{% endmathjax %}，然后取{% mathjax %}\epsilon = -\eta f'(x){% endmathjax %}。将其代入泰勒展开式我们可以得到：
{% mathjax '{"conversion":{"em":14}}' %}
f(x - \eta f'(x)) = f(x) -\eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x))
{% endmathjax %}
如果其导数{% mathjax %}f'(x)\neq 0{% endmathjax %}没有消失，我们就能继续展开，这是因为{% mathjax %}\eta f'^2(x) > 0{% endmathjax %}。此外，我们总是可以令{% mathjax %}\eta{% endmathjax %}小到足以使高阶项变得不相关。因此：
{% mathjax '{"conversion":{"em":14}}' %}
f(x - \eta f'(x)) \lessapprox f(x)
{% endmathjax %}
这意味着，如果我们使用：
{% mathjax '{"conversion":{"em":14}}' %}
x \leftarrow x - \eta f'(x)
{% endmathjax %}
来迭代{% mathjax %}x{% endmathjax %}，函数{% mathjax %}f(x){% endmathjax %}的值可能会下降。因此，在梯度下降中，我们首先选择初始值{% mathjax %}x{% endmathjax %}和常数{% mathjax %}\eta > 0{% endmathjax %}，然后使用它们连续迭代{% mathjax %}x{% endmathjax %}，直到停止条件达成，例如，当梯度{% mathjax %}|f'(x)|{% endmathjax %}的幅度足够小或迭代次数达到某个值时。
下面我们来展示如何实现梯度下降。为了简单起见，我们选用目标函数{% mathjax %}f(x) = x^2{% endmathjax %}。尽管我们知道{% mathjax %}x = 0{% endmathjax %}时{% mathjax %}f(x){% endmathjax %}能取得最小值，但我们仍然使用这个简单的函数来观察{% mathjax %}x{% endmathjax %}的变化。
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def f(x):  # 目标函数
    return x ** 2

def f_grad(x):  # 目标函数的梯度(导数)
    return 2 * x

def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    return results

# 我们使用x = 10, 并假设eta = 0.2, 使用梯度下降法迭代x共10次，x的值最终将接近最优解。
results = gd(0.2, f_grad)

def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = tf.range(-n, n, 0.01)
    plt.figure(figsize=(10, 6))
    plt.plot([f_line, results], [[f(x) for x in f_line], [f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```
{% asset_img oa_9.png %}

###### 学习率

**学习率**(`learning rate`)决定目标函数能否收敛到局部最小值，以及何时收敛到最小值。学习率{% mathjax %}\eta{% endmathjax %}可由算法设计者设置。请注意，如果我们使用的学习率太小，将导致{% mathjax %}x{% endmathjax %}的更新非常缓慢，需要更多的迭代。例如，考虑同一优化问题中{% mathjax %}\eta = 0.05{% endmathjax %}的进度。如下所示，尽管经过了`10`个步骤，我们仍然离最优解很远。
```python
show_trace(gd(0.05, f_grad), f)
```
{% asset_img oa_10.png %}

相反，如果我们使用过高的学习率，{% mathjax %}|\eta f'(x)|{% endmathjax %}对于一级泰勒展开式可能太大，也就是说{% mathjax %}\mathcal{O}(\eta^2f'^2(x)){% endmathjax %}可能变得显著了，在这种情况下{% mathjax %}x{% endmathjax %}的迭代不能保证降低{% mathjax %}f(x){% endmathjax %}的值。例如，当学习率{% mathjax %}\eta= 1.1{% endmathjax %}，{% mathjax %}x{% endmathjax %}超出了最优解{% mathjax %}x = 0{% endmathjax %}并逐渐发散。
```python
show_trace(gd(1.1, f_grad), f)
```
{% asset_img oa_11.png %}

###### 局部最小值

为了演示非凸函数的梯度下降，考虑函数{% mathjax %}f(x) = x\cdot \cos(cx){% endmathjax %}，其中{% mathjax %}c{% endmathjax %}为某常数。这个函数有无穷多个局部最小值。根据我们选择的学习率，我们最终可能只会得到许多解的一个。下面的例子说明了（不切实际的）高学习率如何导致较差的局部最小值。
```python
c = tf.constant(0.15 * np.pi)

def f(x):  # 目标函数
    return x * tf.cos(c * x)

def f_grad(x):  # 目标函数的梯度
    return tf.cos(c * x) - c * x * tf.sin(c * x)

show_trace(gd(2, f_grad), f)
```
{% asset_img oa_12.png %}

##### 多元梯度下降

现在我们对单变量的情况有了更好的理解，让我们考虑一下{% mathjax %}\mathbf{x} = [x_1,x_2,\ldots,x_d]^{\mathsf{T}}{% endmathjax %}的情况，即目标函数{% mathjax %}f:\mathbb{R}\rightarrow \mathbb{R}{% endmathjax %}将向量映射成标量。相应地，它的梯度也是多元的，它是一个由{% mathjax %}d{% endmathjax %}个偏导数组成的向量：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla f(\mathbf{x}) = [\frac{\partial f(\mathbf{x})}{\partial x_1},\frac{\partial f(\mathbf{x})}{\partial x_2},\ldots,\frac{\partial f(\mathbf{x})}{\partial x_d}]^{\mathsf{T}}
{% endmathjax %}
梯度中的每个偏导数元素{% mathjax %}\partial f(\mathbf{x})/\partial x_i{% endmathjax %}代表了当输入{% mathjax %}x_i{% endmathjax %}时{% mathjax %}f{% endmathjax %}在{% mathjax %}\mathbf{x}{% endmathjax %}处的变化率。和先前单变量的情况一样，我们可以对多变量函数使用相应的泰勒近似来思考。具体来说，
{% mathjax '{"conversion":{"em":14}}' %}
f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^{\mathsf{T}}\nabla f(\mathbf{x}) + \mathcal{O}(\lVert\mathbf{\epsilon}\rVert^2)
{% endmathjax %}
换句话说，在{% mathjax %}\mathbf{\epsilon}{% endmathjax %}在二阶项中，最陡下降的方向由负梯度{% mathjax %}-\nabla f(\mathbf{x}){% endmathjax %}得出。选择合适的学习率{% mathjax %}\eta > 0{% endmathjax %}来生成典型的梯度下降算法：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{x}\leftarrow \mathbf{x} - \eta\nabla f(\mathbf{x})
{% endmathjax %}
这个算法在实践中的表现如何呢？我们构造一个目标函数{% mathjax %}f(\mathbf{x}) = x_1^2 + 2x_2^2{% endmathjax %}，并由二维向量{% mathjax %}\mathbf{x} = [x_1,x_2]^{\mathsf{T}}{% endmathjax %}作为输入，标量作为输出。梯度由{% mathjax %}\nabla f(\mathbf{x}) = [2x_1,4x_2]^{\mathsf{T}}{% endmathjax %}给出。我们将从初始位置{% mathjax %}[-5,-2]{% endmathjax %}通过梯度下降观察{% mathjax %}x{% endmathjax %}的轨迹。接下来，我们观察学习率{% mathjax %}\eta= 0.1{% endmathjax %}时优化变量{% mathjax %}x{% endmathjax %}的轨迹。可以看到，经过`20`步之后，{% mathjax %}x{% endmathjax %}的值接近其位于[0,0]的最小值。虽然进展相当顺利，但相当缓慢。
{% asset_img oa_13.png %}

##### 自适应方法

选择“恰到好处”的学习率{% mathjax %}\eta{% endmathjax %}是很棘手的。如果我们把它选得太小，就没有什么进展；如果太大，得到的解就会振荡，甚至可能发散。如果我们可以自动确定{% mathjax %}\eta{% endmathjax %}，或者完全不必选择学习率，会怎么样？除了考虑目标函数的值和梯度、还考虑它的曲率的二阶方法可以帮我们解决这个问题。虽然由于计算代价的原因，这些方法不能直接应用于深度学习，但它们为如何设计高级优化算法提供了有用的思维直觉，这些算法可以模拟下面概述的算法的许多理想特性。
###### 牛顿法

回顾一下函数{% mathjax %}f:\mathbb{R}^d\rightarrow \mathbb{R}{% endmathjax %}的泰勒展开式，事实上我们可以把它写成：
{% mathjax '{"conversion":{"em":14}}' %}
f(\mathbf{x} + \mathbf{\epsilon}) = f(\mathbf{x}) + \mathbf{\epsilon}^{\mathsf{T}}\nabla f(\mathbf{x}) + \frac{1}{2}\mathbf{\epsilon}^{\mathsf{T}}\nabla^2 f(\mathbf{x})\epsilon + \mathcal{O}(\lVert\epsilon\rVert^3)
{% endmathjax %}
为了避免繁琐的符号，我们将{% mathjax %}\mathbf{H} = \underset{=}{\text{def}} = \nabla^2 f(\mathbf{x}){% endmathjax %}定义为{% mathjax %}f{% endmathjax %}的`Hessian`，是{% mathjax %}d\times d{% endmathjax %}矩阵。当{% mathjax %}d{% endmathjax %}的值很小且问题很简单时，{% mathjax %}\mathbf{H}{% endmathjax %}很容易计算。但是对于深度神经网络而言，考虑到{% mathjax %}\mathbf{H}{% endmathjax %}可能非常大，{% mathjax %}\mathcal{O}(d^2){% endmathjax %}哥条目的存储代价非常大，此外通过反向传播计算可能雪上加霜。然而，我们姑且先忽略这些考量，看看会得到什么算法。毕竟，{% mathjax %}f{% endmathjax %}的最小值满足{% mathjax %}\nabla f = 0{% endmathjax %}，忽略不重要的高阶项，我们便得到：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla f(x) + \mathbf{H}\mathbf{\epsilon} = 0\;\text{and hence }\mathbf{\epsilon} = -\mathbf{H}^{-1}\nabla f(\mathbf{x})
{% endmathjax %}
也就是说，作为优化问题的一部分，我们需要将`Hessian`矩阵{% mathjax %}\mathbf{H}{% endmathjax %}求逆。
###### 收敛性分析

在此，我们以部分目标凸函数{% mathjax %}f{% endmathjax %}为例，分析它们的牛顿法收敛速度。这些目标凸函数三次可微，而且二阶导数不为零，即{% mathjax %}f'' > 0{% endmathjax %}。由于多变量情况下的证明是对以下一维参数情况证明的直接拓展，对我们理解这个问题不能提供更多帮助，因此我们省略了多变量情况的证明。用{% mathjax %}x^{(k)}{% endmathjax %}表示{% mathjax %}x{% endmathjax %}在第{% mathjax %}k^{\text{th}}{% endmathjax %}次迭代的值，令{% mathjax %}e^{(k)}\underset{=}{\text{def}} x^{(k)} - x_{\ast}{% endmathjax %}表示{% mathjax %}k^{\text{th}}{% endmathjax %}迭代时与最优性的距离。通过泰勒展开，我们得到条件{% mathjax %}f'(x^{\ast}) = 0{% endmathjax %}可以写成：
{% mathjax '{"conversion":{"em":14}}' %}
0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)}f''(x^{(k)}) \frac{1}{2}(e^{(k)})^2 f'''(\xi^{(k)})
{% endmathjax %}
这对某些{% mathjax %}\xi^{(k)}\in [x^{(k)} - e^{(k)}, x^{(k)}]{% endmathjax %}成立。将上述展开除以{% mathjax %}f''(x^{(k)}){% endmathjax %}得到
{% mathjax '{"conversion":{"em":14}}' %}
e^{(k)} - \frac{'(x^{(k)})}{f''(x^{}(k))} = \frac{1}{2}(e^{(k)})^2\frac{f'''(\xi^{(k)})}{f''(x^{(k)})}
{% endmathjax %}
回想之前的方程{% mathjax %}x^{(k+1)} = x^{(k)} - f'(x^{(k)})/f''(x^{(k)}){% endmathjax %}。带入这个更新方程，去两边的绝对值，我们得到：
{% mathjax '{"conversion":{"em":14}}' %}
|e^{(k + 1)}| \frac{1}{2}(e^{(k)})^2\frac{|f'''(\xi^{(k)})|}{f''(x^{(k)})}
{% endmathjax %}
因此，每当我们处于有界区域{% mathjax %}|f'''(\xi^{(k)})|/(2f''(x^{(k)})) \leq c{% endmathjax %}，我们就有一个二次递减误差：
{% mathjax '{"conversion":{"em":14}}' %}
|e^{(k+1)}| \leq c(e^{(k)})^2
{% endmathjax %}
另一方面，优化研究人员称之为“线性”收敛，而将{% mathjax %}|e^{(k+1)}| \leq \alpha|e^{(k)}|{% endmathjax %}这样的条件称为“恒定”收敛速度。请注意，我们无法估计整体收敛的速度，但是一旦我们接近极小值，收敛将变得非常快。另外，这种分析要求{% mathjax %}f{% endmathjax %}在高阶导数上表现良好，即确保{% mathjax %}f{% endmathjax %}在如何变化它的值方面没有任何“超常”的特性。
######  预处理

计算和存储完整的`Hessian`非常昂贵，而改善这个问题的一种方法是“预处理”。 它回避了计算整个`Hessian`，而只计算“对角线”项，即如下的算法更新：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{x} \leftarrow \mathbf{x} - \eta\text{diag}(\mathbf{H})^{-1}\nabla f(\mathbf{x})
{% endmathjax %}
虽然这不如完整的牛顿法精确，但它仍然比不使用要好得多。为什么预处理有效呢？假设一个变量以毫米表示高度，另一个变量以公里表示高度的情况。假设这两种自然尺度都以米为单位，那么我们的参数化就出现了严重的不匹配。幸运的是，使用预处理可以消除这种情况。梯度下降的有效预处理相当于为每个变量选择不同的学习率（矢量{% mathjax %}\mathbf{x}{% endmathjax %}的坐标）。
###### 梯度下降和线搜索

梯度下降的一个关键问题是我们可能会超过目标或进展不足，解决这一问题的简单方法是结合使用线搜索和梯度下降。也就是说，我们使用{% mathjax %}\nabla f(\mathbf{x}){% endmathjax %}给出的方向，然后进行二分搜索，以确定哪个学习率{% mathjax %}\eta{% endmathjax %}使{% mathjax %}f(\mathbf{x} - \eta\nabla f(\mathbf{x})){% endmathjax %}取最小值。有关分析和证明，此算法收敛迅速。然而，对深度学习而言，这不太可行。因为线搜索的每一步都需要评估整个数据集上的目标函数，实现它的方式太昂贵了。
##### 总结

学习率的大小很重要：学习率太大会使模型发散，学习率太小会没有进展。梯度下降会可能陷入局部极小值，而得不到全局最小值。在高维模型中，调整学习率是很复杂的。预处理有助于调节比例。牛顿法在凸问题中一旦开始正常工作，速度就会快得多。对于非凸问题，不要不作任何调整就使用牛顿法。

#### 随机梯度下降

##### 随机梯度更新

在深度学习中，目标函数通常是训练数据集中每个样本的损失函数的平均值。给定{% mathjax %}n{% endmathjax %}个样本的训练数据集，我们假设{% mathjax %}f_i(\mathbf{x}){% endmathjax %}是关于索引{% mathjax %}i{% endmathjax %}的训练样本的损失函数，其中{% mathjax %}\mathbf{x}{% endmathjax %}是参数向量。然后我们得到目标函数。
{% mathjax '{"conversion":{"em":14}}' %}
f(\mathbf{x}) = \frac{1}{n}\sum_{i=1}^n f_i(\mathbf{x})
{% endmathjax %}
{% mathjax %}\mathbf{x}{% endmathjax %}的目标函数的梯度计算为：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla f(\mathbf{x}) \frac{1}{n}\sum_{i=1}^n \nabla f_i(\mathbf{x})
{% endmathjax %}
如果使用梯度下降法，则每个自变量迭代的计算代价为{% mathjax %}\mathcal{O}(n){% endmathjax %}，它随{% mathjax %}n{% endmathjax %}线性增长。因此，当训练数据集较大时，每次迭代的梯度下降计算代价将较高。**随机梯度下降**(`SGD`)可降低每次迭代时的计算代价。在随机梯度下降的每次迭代中，我们对数据样本随机均匀采样一个索引{% mathjax %}i{% endmathjax %}，其中{% mathjax %}i\in \{1,\ldots,n\}{% endmathjax %}，并计算梯度{% mathjax %}\nabla f_i(\mathbf{x}){% endmathjax %}以更新{% mathjax %}\mathbf{x}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{x}\leftarrow \mathbf{x} - \eta\nabla f_i(\mathbf{x})
{% endmathjax %}
其中{% mathjax %}\eta{% endmathjax %}是学习率。我们可以看到，每次迭代的计算代价从梯度下降的{% mathjax %}\mathcal{O}(n){% endmathjax %}将至常数{% mathjax %}\mathcal{O}(1){% endmathjax %}。此外，我们要强调，随机梯度{% mathjax %}\nabla f_i(\mathbf{x}){% endmathjax %}是对完整梯度{% mathjax %}\nabla f(\mathbf{x}){% endmathjax %}的无偏估计，因为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbb{E}_i\nabla f_i(\mathbf{x}) = \frac{1}{n}\sum_{i = 1}^n \nabla f_i(\mathbf{x}) = \nabla f(x)
{% endmathjax %}
这意味着，平均而言，随机梯度是对梯度的良好估计。现在，我们将把它与梯度下降进行比较，方法是向梯度添加均值为0、方差为1的随机噪声，以模拟随机梯度下降。
```python
def f(x1, x2):  # 目标函数
    return x1 ** 2 + 2 * x2 ** 2

def f_grad(x1, x2):  # 目标函数的梯度
    return 2 * x1, 4 * x2

def sgd(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    # 模拟有噪声的梯度
    g1 += tf.random.normal([1], 0.0, 1)
    g2 += tf.random.normal([1], 0.0, 1)
    eta_t = eta * lr()
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0)

def constant_lr():
    return 1

eta = 0.1
lr = constant_lr  # 常数学习速度
# ...
# epoch 50, x1: -0.051145, x2: -0.028135
```
{% asset_img oa_14.png %}

正如我们所看到的，随机梯度下降中变量的轨迹比我们在之前观察到的梯度下降中观察到的轨迹嘈杂得多。这是由于梯度的随机性质。也就是说，即使我们接近最小值，我们仍然受到通过{% mathjax %}\eta\nabla f_i(\mathbf{x}){% endmathjax %}的瞬间梯度所注入的不确定性的影响。即使经过`50`次迭代，质量仍然不那么好。更糟糕的是，经过额外的步骤，它不会得到改善。这给我们留下了唯一的选择：改变学习率{% mathjax %}\eta{% endmathjax %}。但是，如果我们选择的学习率太小，我们一开始就不会取得任何有意义的进展。另一方面，如果我们选择的学习率太大，我们将无法获得一个好的解决方案，如上所示。解决这些相互冲突的目标的唯一方法是在优化过程中动态降低学习率。这也是在`sgd`步长函数中添加学习率函数`lr`的原因。
##### 动态学习率

用与时间相关的学习率{% mathjax %}\eta(t){% endmathjax %}取代{% mathjax %}\eta{% endmathjax %}增加了控制优化算法收敛的复杂性。特别是，我们需要弄清{% mathjax %}\eta{% endmathjax %}的衰减速度。如果太快，我们将过早停止优化。如果减少的太慢，我们会在优化上浪费太多时间。以下是随着时间推移调整{% mathjax %}\eta{% endmathjax %}时使用的一些基本策略：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \eta(t) = \eta_i\;\text{if}\; t_i\leq t\leq t_{i+1} &\;\;\text{分段常数}\\ 
& \eta(t) = \eta_0\cdot e^{-\lambda t} &\;\;\text{指数衰减}  \\
& \eta(t) = \eta_0\cdot (\beta t + 1)^{-\alpha} &\;\;\text{多项式衰减}  \\
\end{align}
{% endmathjax %}
在第一个**分段常数**(`piecewise constant`)场景中，我们会降低学习率，例如，每当优化进度停顿时。这是训练深度网络的常见策略。或者，我们可以通过**指数衰减**(`exponential decay`)来更积极地减低它。不幸的是，这往往会导致算法收敛之前过早停止。一个受欢迎的选择是{% mathjax %} {% endmathjax %}的**多项式衰减**(`polynomial decay`)。在凸优化的情况下，有许多证据表明这种速率表现良好。让我们看看指数衰减在实践中是什么样子。
```python
def exponential_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return math.exp(-0.1 * t)

def polynomial_lr():
    # 在函数外部定义，而在内部更新的全局变量
    global t
    t += 1
    return (1 + 0.1 * t) ** (-0.5)

t = 1
lr = exponential_lr
# lr = polynomial_lr

# epoch 1000, x1: -0.749494, x2: -0.058892
# epoch 50, x1: -0.061881, x2: -0.026958
```
{% asset_img oa_15.png %}

正如预期的那样，参数的方差大大减少。但是，这是以未能收敛到最优解{% mathjax %}\mathbf{x} = (0,0){% endmathjax %}为代价的。即使经过`1000`个迭代步骤，我们仍然离最优解很远。事实上，该算法根本无法收敛。另一方面，如果我们使用多项式衰减，其中学习率随迭代次数的平方根倒数衰减，那么仅在`50`次迭代之后，收敛就会更好。
{% asset_img oa_16.png %}

关于如何设置学习率，还有更多的选择。例如，我们可以从较小的学习率开始，然后使其迅速上涨，再让它降低，尽管这会更慢。我们甚至可以在较小和较大的学习率之间切换。现在，让我们专注于可以进行全面理论分析的学习率计划，即凸环境下的学习率。对一般的非凸问题，很难获得有意义的收敛保证，因为总的来说，最大限度地减少非线性非凸问题是`NP`困难的。
##### 凸目标的收敛性分析

##### 随机梯度和有限样本

我们假设从分布{% mathjax %}p(x,y){% endmathjax %}中采样得到样本{% mathjax %}x_i{% endmathjax %}（通常带有标签{% mathjax %}y_i{% endmathjax %}），并且用它来以某种方式更新模型参数。特别是，对于有限的样本数量，我们仅仅讨论了由某些允许我们在其上执行随机梯度下降的函数{% mathjax %}\delta_{x_i}{% endmathjax %}和{% mathjax %}\delta_{y_i}{% endmathjax %}组成的离散分布{% mathjax %}p(x,y) = \frac{1}{n}\sum_{i=1}^n \delta_{x_i}(x)\delta_{y_i}(y){% endmathjax %}。

但是，这不是我们真正做的。在本节的简单示例中，我们只是将噪声添加到其他非随机梯度上，也就是说，我们假装有成对的{% mathjax %}(x_i,y_i){% endmathjax %}。事实证明，这种做法在这里是合理的。更麻烦的是，在以前的所有讨论中，我们显然没有这样做。相反，我们遍历了所有实例恰好一次。要了解为什么这更可取，可以反向虑一下，即我们有替换地从离散分布中采样{% mathjax %}n {% endmathjax %}个观测值。随机选择一个元素{% mathjax %}i{% endmathjax %}的概率是{% mathjax %}1/n{% endmathjax %}。因此选择它至少一次就是：
{% mathjax '{"conversion":{"em":14}}' %}
P(\text{choose }i) =  - P(\text{omit }i) = 1 - (1/n)^n \approx e^{-1} \approx 0.37
{% endmathjax %}
类似的推理表明，挑选一些样本（即训练示例）恰好一次的概率是：
{% mathjax '{"conversion":{"em":14}}' %}
\left(\begin{array}{ccc}
n \\
1
\end{array}\right)
\frac{1}{n}(1 - \frac{1}{n})^{n - 1} = \frac{n}{n - 1}(1 - \frac{1}{n})^n \approx 0.37
{% endmathjax %}
这导致与无替换采样相比，方差增加并且数据效率降低。因此，在实践中我们执行后者。最后一点注意，重复采用训练数据集的时候，会以不同的随机顺序遍历它。
##### 总结

对于凸问题，我们可以证明，对于广泛的学习率选择，随机梯度下降将收敛到最优解。对于深度学习而言，情况通常并非如此。但是，对凸问题的分析使我们能够深入了解如何进行优化，即逐步降低学习率，尽管不是太快。如果学习率太小或太大，就会出现问题。实际上，通常只有经过多次实验后才能找到合适的学习率。当训练数据集中有更多样本时，计算梯度下降的每次迭代的代价更高，因此在这些情况下，首选**随机梯度下降**。随机梯度下降的最优性保证在非凸情况下一般不可用，因为需要检查的局部最小值的数量可能是指数级的。

#### 小批量随机梯度下降

我们在基于梯度的学习方法中遇到了两个极端情况：**使用完整数据集来计算梯度并更新参数，一次处理一个训练样本来取得进展**。二者各有利弊：每当数据非常相似时，梯度下降并不是“数据高效”。而由于`CPU`和`GPU`无法充分利用向量化，随机梯度下降并不“计算高效”。这暗示了两者之间可能有折中方案，这便涉及到**小批量随机梯度下降**(`minibatch gradient descent`)。
##### 向量化和缓存

使用小批量的决策的核心是计算效率。 当考虑与多个`GPU`和多台服务器并行处理时，这一点最容易被理解。在这种情况下，我们需要向每个`GPU`发送至少一张图像。有了每台服务器`8`个`GPU`和`16`台服务器，我们就能得到大小为`128`的小批量。当涉及到单个`GPU`甚至`CPU`时，事情会更微妙一些：这些设备有多种类型的内存、通常情况下多种类型的计算单元以及在它们之间不同的带宽限制。例如，一个`CPU`有少量寄存器（`register`），`L1`和`L2`缓存，以及`L3`缓存（在不同的处理器内核之间共享）。随着缓存的大小的增加，它们的延迟也在增加，同时带宽在减少。可以说，处理器能够执行的操作远比主内存接口所能提供的多得多。

减轻这些限制的方法是使用足够快的`CPU`缓存层次结构来为处理器提供数据。这是深度学习中批量处理背后的推动力。举一个简单的例子：矩阵-矩阵乘法。比如{% mathjax %}\mathbf{A} = \mathbf{BC}{% endmathjax %}，我们有很多方法来计算{% mathjax %}\mathbf{A}{% endmathjax %}。例如，我们可以尝试以下方法：
- 我们可以计算{% mathjax %}\mathbf{A}_{ij} = \mathbf{B}_{i,:}\mathbf{C}_{:,j}^{\mathsf{T}}{% endmathjax %}，也就是说，我们可以通过点积进行逐元素计算。
- 我们可以计算{% mathjax %}\mathbf{A}_{ij} = \mathbf{B}_{i,:}\mathbf{C}_{:,j}^{\mathsf{T}}{% endmathjax %}，也就是说，我们可以一次计算一列。同样，我们可以一次计算{% mathjax %}\mathbf{A}{% endmathjax %}一行{% mathjax %}\mathbf{A}_{i,:}{% endmathjax %}。
- 我们可以简单地计算{% mathjax %}\mathbf{A} = \mathbf{BC}{% endmathjax %}。
- 我们可以将{% mathjax %}\mathbf{B}{% endmathjax %}和{% mathjax %}\mathbf{C}{% endmathjax %}分成较小的区块矩阵，然后一次计算{% mathjax %}\mathbf{A}{% endmathjax %}的一个区块。

如果我们使用第一个选择，每次我们计算一个元素{% mathjax %}\mathbf{A}_{ij}{% endmathjax %}时，都需要将一行和一列向量复制到`CPU`中。更糟糕的是，由于矩阵元素是按顺序对齐的，因此当从内存中读取它们时，我们需要访问两个向量中许多不相交的位置。第二种选择相对更有利：我们能够在遍历{% mathjax %}\mathbf{B}{% endmathjax %}的同时，将列向量{% mathjax %}\mathbf{C}_{:,j}{% endmathjax %}保留在`CPU`缓存中。它将内存带宽需求减半，相应地提高了访问速度。第三种选择表面上是最可取的，然而大多数矩阵可能不能完全放入缓存中。第四种选择提供了一个实践上很有用的方案：我们可以将矩阵的区块移到缓存中然后在本地将它们相乘。 
##### 小批量

之前我们会理所当然地读取数据的小批量，而不是观测单个数据来更新参数，现在简要解释一下原因。处理单个观测值需要我们执行许多单一矩阵-矢量（甚至矢量-矢量）乘法，这耗费相当大，而且对应深度学习框架也要巨大的开销。这既适用于计算梯度以更新参数时，也适用于用神经网络预测。也就是说，每当我们执行{% mathjax %}\mathbf{w}\leftarrow \mathbf{w} - \eta_t\mathbf{g}_t{% endmathjax %}时，消耗巨大。其中：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{g}_t = \partial_w f(\mathbf{x}_t,\mathbf{w})
{% endmathjax %}
我们可以通过将其应用于一个小批量观测值来提高此操作的计算效率。也就是说，我们将梯度{% mathjax %}\mathbf{g}_t{% endmathjax %}替换为一个小批量而不是单个观测值。
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{g}_t= \partial_w \frac{1}{|\mathcal{B}_t|}\sum_{i\in\mathcal{B}_t} f(\mathbf{x}_i,\mathbf{w})
{% endmathjax %}
让我们看看这对{% mathjax %}\mathbf{g}_t{% endmathjax %}的统计属性有什么影响：由于{% mathjax %}\mathbf{x}_t{% endmathjax %}和小批量{% mathjax %}\mathcal{B}_t{% endmathjax %}的所有元素都是从训练集随机抽出的，因此梯度的期望保持不变。另一方面，方差显著降低。由于小批量梯度由正在被平均计算的{% mathjax %}b:=|\mathcal{B}_t|{% endmathjax %}个独立梯度组成，其标准差降低了{% mathjax %}b^{-frac{1}{2}}{% endmathjax %}。这本身就是一件好事，因为这意味着更新与完整的梯度更接近了。直观来说，这表明选择大型的小批量{% mathjax %}\mathcal{B}_t{% endmathjax %}将是可行的。 然而，经过一段时间后，与计算代价的线性增长相比，标准差的额外减少是微乎其微的。在实践中我们选择一个足够大的小批量，它可以提供良好的计算效率同时仍适合`GPU`的内存。
##### 总结

由于减少了深度学习框架的额外开销，使用更好的内存定位以及`CPU`和`GPU`上的缓存，向量化使代码更加高效。随机梯度下降的“统计效率”与大批量一次处理数据的“计算效率”之间存在权衡。小批量随机梯度下降提供了两全其美的答案：计算和统计效率。在小批量随机梯度下降中，我们处理通过训练数据的随机排列获得的批量数据（即每个观测值只处理一次，但按随机顺序）。在训练期间降低学习率有助于训练。一般来说，小批量随机梯度下降比随机梯度下降和梯度下降的速度快，收敛风险较小。

#### 动量法

我们在选择学习率需要格外谨慎。如果衰减速度太快，收敛就会停滞。相反，如果太宽松，我们可能无法收敛到最优解。
##### 泄漏平均值

我们讨论了小批量随机梯度下降作为加速计算的手段。它也有很好的副作用，即平均梯度减小了方差。小批量随机梯度下降可以通过以下方式计算：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{g}_{t,t-1} = \partial_w\frac{1}{|\mathcal{B}_t|}\sum_{i\in\mathcal{B}_t} f(\mathbf{x}_i,\mathbf{w}_{t-1}) = \frac{1}{|\mathcal{B}_t|}\sum_{i\in\mathcal{B}_t} \mathbf{h}_{i,t-1}
{% endmathjax %}
为了保持记法简单，在这里我们使用{% mathjax %}\mathbf{h}_{i,t-1}{% endmathjax %}作为样本{% mathjax %}i{% endmathjax %}的随机梯度下降，使用时间{% mathjax %}t-1{% endmathjax %}时更新的权重{% mathjax %}t-1{% endmathjax %}。如果我们能够从方差的减少的影响中受益，甚至超过小批量上的梯度平均值，那很不错。完成这项任务的一种选择是用**泄漏平均值**(`leaky average`)取代梯度计算：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \mathbf{g}_{t,t-1}
{% endmathjax %}
其中{% mathjax %}\beta\in (0,1){% endmathjax %}。这有效地将瞬时梯度有效地替换为多个“过去”梯度的平均值。{% mathjax %}v{% endmathjax %}被称为动量(`momentum`)，它累加了过去的梯度。为了更详细地解释，让我们递归地将{% mathjax %}\mathbf{v}_t{% endmathjax %}扩展到：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{v}_t = \beta^2\mathbf{v}_{t-2} + \beta\mathbf{g}_{t-1,t-2} + \mathbf{g}_{t,t-1} = \ldots, = \sum_{\tau=0}^{t-1} \beta^{\tau}\mathbf{g}_{t-\tau,t-\tau-1}
{% endmathjax %}
其中，较大的{% mathjax %}\beta{% endmathjax %}相当于长期平均值，而较小的{% mathjax %}\beta{% endmathjax %}相对于梯度法只是略有修正。新的梯度替换不再指向特定实例下降最陡的方向，而是指向过去梯度的加权平均值的方向。这使我们能够实现对单批量计算平均值的大部分好处，而不产生实际计算其梯度的代价。上述推理构成了“加速”梯度方法的基础，例如具有动量的梯度。在优化问题条件不佳的情况下（例如，有些方向的进展比其他方向慢得多，类似狭窄的峡谷），“加速”梯度还额外享受更有效的好处。此外，它们允许我们对随后的梯度计算平均值，以获得更稳定的下降方向。诚然，即使是对于无噪声凸问题，加速度这方面也是动量如此起效的关键原因之一。
##### 动量法

**动量法**(`momentum`)使我们能够解决上面描述的梯度下降问题。观察上面的优化轨迹，我们可能会直觉到计算过去的平均梯度效果会很好。毕竟，在{% mathjax %}x_1{% endmathjax %}方向上，这将聚合非常对齐的梯度，从而增加我们在每一步中覆盖的距离。相反，在梯度振荡的{% mathjax %}x_2{% endmathjax %}方向，由于相互抵消了对方的振荡，聚合梯度将减小步长大小。使用{% mathjax %}\mathbf{v}_t{% endmathjax %}而不是梯度{% mathjax %}\mathbf{g}_t{% endmathjax %}可以生成以下更新等式：
