---
title: 多层感知器(MLP) vs 科尔莫戈罗夫-阿诺德网络(KAN)（机器学习）
date: 2024-06-25 15:20:11
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

#### 多层感知器(MLP)

多层感知器(`MLP`)是如何工作的？多层感知器(`MLP`)是由多个神经元层组成的神经网路，每个神经元层以前馈方式组织，这意味着一层的输出作为下一层的输入，通常在每一层，我们会放置一些非线性激活函数，例如`RelU`，在这种情况下，会形成一个非常简单的网络，如下图所示：
{% asset_img km_1.png %}
<!-- more -->

上图中有包含三个特征的输入向量，接下来有`5`个神经元组成的第一层，`5`个输出特征的网络，这里的网络采用`5`个特征作为输入，并产生`5`个输出特征，它们具有相同的结构，通常我们将激活函数放在各层中。让我们看看它在`pytorch`中是如何工作的：
{% asset_img km_2.png %}

在文档中你将看到线性层执行这个非常简单的操作，它接受输入，你可以将其视为由特征组成的向量，或者你可以将其视为由许多项目组成的矩阵，每个项目都具有输入特征，让后我们将它乘以权重矩阵，我们它称之为{% mathjax %}A{% endmathjax %}，让我们分析一下它的结构。假设我们有`10`个输入，每个输入向量有`3`个特征组成，在这里称之为{% mathjax %}f_1、f_2{% endmathjax %}和{% mathjax %}f_3{% endmathjax %}，这三个输入特征时线性层。将执行以下操作，即输入乘以某个权重矩阵，该矩阵由权重组成，可以通过网络学习获得，加上偏置权重矩阵。在这里我们讨论的是线性层，`3`个特征作为输入并产生`5`个输出特征，因此，权重矩阵转置后的权重矩阵将是{% mathjax %}(3,5){% endmathjax %},你可以将每个神经元视为一列权重。如果{% mathjax %}n{% endmathjax %}是输入特征的数量，则每个神经元将具有{% mathjax %}n{% endmathjax %}个权重，每个输入特征都有一个权重，并且它将产生一些输出，因此当我们将{% mathjax %}X\times W^\top{% endmathjax %}执行这个矩阵乘法，将产生以下结果。你可以认为这个矩阵是一批`10`个项目，因为在输入中我们有`10`个项目(`items`)，每个项目有`5`个特征，因为线性层从`3`个特征变为`5`个特征输出。第`1`个特征是由第一个项目与第`1`个神经元的点积生成的。这里的值是第`2`个输出第一项的特征是第一项的特征值乘以第`2`个神经元，因此第`2`个神经元的权重在执行此乘法之后负责该线性层中的一个输出特征，我们添加一个偏差项，在本例中是一个向量，每个神经元都有一个值，我们将它广播到这个矩阵。每个项目的每个输出特征将有一个附加项，即与该特定神经元相关的偏差，因此第一个神经元将添加第一个特征的值加上{% mathjax %}b_1{% endmathjax %}，第二个特征将具有第二个特征的值加上{% mathjax %}b_2{% endmathjax %}等等。这将产生线性层的输出，我们得到`10`个项目，每个项目有`5`个特征。线性层是从`3`变换到`5`。所以基本上它是每个项目的输入特征乘以相应的权重加上偏差。
{% asset_img km_3.png %}

##### 为什么线性层需要激活函数？

主要有以下几个原因：
- 引入非线性：非线性激活函数能够为神经网络引入非线性特性，使网络能够学习和表示复杂的非线性关系。如果没有非线性激活函数，多层神经网络就等同于单层线性网络，无法建模复杂的函数。
- 增强表达能力：非线性激活函数使神经网络能够近似任意复杂的函数。根据通用近似定理，具有非线性激活函数的前馈网络可以近似任何连续函数。
- 解决梯度消失问题：某些非线性激活函数（如`ReLU`）可以帮助缓解深度网络中的梯度消失问题，使得深层网络更容易训练。
- 实现决策边界：非线性激活函数使得神经网络能够学习非线性决策边界，这对于解决复杂的分类问题至关重要。
- 映射到特定范围：某些激活函数（如`Sigmoid`和`Tanh`）可以将输入映射到特定的范围，这对于某些任务（如概率预测）非常有用。
- 促进反向传播：非线性激活函数的可微性质使得反向传播算法能够有效地调整网络权重。
- 增加网络深度能力：使用非线性激活函数使得增加网络深度变得有意义，因为每一层都可以学习更复杂的特征表示。

总之，非线性激活函数是使神经网络能够学习复杂模式和关系的关键组成部分，它们使得神经网络成为强大的通用函数逼近器，能够解决各种复杂的机器学习任务。
##### 数据拟合

想象以下我们正在创建一个`2D`游戏，其中我们有一个角色，想要通过由点组成来制作动画。当然，制作角色动画的方法是：用这些点并在它们之间画直线，这样就可以为角色设置动画。但看起来不太好，这个动作不太平滑，你可以看到这个角色在刚性运动。所以有一种更好的办法是制作一条弯曲的多项式曲线。它穿过这些点并生成更平滑的路径，这看起来更好。创建一条多项式，穿过这些点的线是弯曲的，如何做到这一点？当你有两个点时，你只能在它们之间画一条线，当你有三个点时，则意味着它们不再是直线，你可以画一条二次曲线，想抛物线一样；如果你有四个点，你需要一个3次方程来画一条穿过它们的多项式线。如何计算这条线的方程而正好穿过所有这些点？我们想象有`4`个点{% mathjax %}(x,y)\in {(0,5),(1,1),(2,3),(5,2)}{% endmathjax %}，我们最终要写出{% mathjax %}n-1{% endmathjax %}次的多项式曲线方程：
{% mathjax '{"conversion":{"em":14}}' %}
y = ax^3+ bx^2 + cx + d
{% endmathjax %}
我们创一个方程组，在该方程组中我们加一个条件，即该曲线必须从所有这些点上通过，一次我们通过代入来写出该曲线方程：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
5 & = a(0)^3 + b(0)^2 + c(0) + d \\
1 & = a(1)^3 + b(1)^2 + c(1) + d \\
3 & = a(2)^3 + b(2)^2 + c(2) + d \\
2 & = a(5)^3 + b(5)^2 + c(5) + d 
\end{aligned}
{% endmathjax %}
{% asset_img km_4.png %}

如果现在我们有数百个点并且我们想生成一条穿过它们的平滑曲线，我们可以这样做。但是会存在两个问题：1.我们需要求解非常复杂的方程组，随着点数的增加，这条多项式曲线会以越来越奇怪的方式呈现，我们需要一种方法来控制这条曲线的平滑度，并且不让它在极端情况下变得如此疯狂，在这里需要研究`B`曲线，看看它是如何工作的。
###### B样条曲线

`B`样条曲线是一条参数化的曲线，因为该差值曲线上的点坐标取决于一个称为{% mathjax %}T{% endmathjax %}的自变量，在这种情况下，你可以将其视为从{% mathjax %}[0,1]{% endmathjax %}的时间，例如，我们只有两个点，两个点只能画一条直线，假设你想在{% mathjax %}p_0{% endmathjax %}和{% mathjax %}p_1{% endmathjax %}之间画一条线。我们{% mathjax %}p_0{% endmathjax %}开始，走向{% mathjax %}p_1{% endmathjax %}，这是我们的差值线，正如你所看到的，随着时间的移动，该点越来越接近{% mathjax %}p_1{% endmathjax %}而远离{% mathjax %}p_0{% endmathjax %}，因此，你可以将变量{% mathjax %}t{% endmathjax %}视为时间变量，方程如下：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{B}(t) = \mathbf{P}_0 + t(\mathbf{p}_1 - \mathbf{P}_0) = (1 - t) + t\mathbf{P}_1
{% endmathjax %}
现在这条曲线是参数化，这也是你所看到的{% mathjax %}B{% endmathjax %}是粗体，而{% mathjax %}t{% endmathjax %}不是粗体的原因，因为它可以是一个向量，意味着这个点位于{% mathjax %}X{% endmathjax %}和{% mathjax %}Y{% endmathjax %}坐标中，所有这些坐标都取决于{% mathjax %}t{% endmathjax %}变量，因此在每个时间步，它都会告诉我们相对于时间的位置。当然我们可以将其扩展到3个点，我们可以绘制一条平滑的曲线，对它们进行差值，在B曲线的情况下，它仅从第一个点和最后一个点穿过，并在中间点之间进行差值。因此它不会接触中间点，而是接近中间点。如何计算这条差值曲线的方程呢？（即红色曲线），在这种情况下我们要做的就是进行递归计算。首先在{% mathjax %}p_0{% endmathjax %}和{% mathjax %}p_1{% endmathjax %}之间进行线性差值，你可以使用这个方程{% mathjax %}(1 - t) + t\mathbf{P}_1{% endmathjax %}，{% mathjax %}p_0{% endmathjax %}和{% mathjax %}p_1{% endmathjax %}差值计算的点为{% mathjax %}q_0{% endmathjax %},随着时间的移动，我们从{% mathjax %}p_0{% endmathjax %}移动到{% mathjax %}p_1{% endmathjax %}；然后我们在{% mathjax %}p_1{% endmathjax %}和{% mathjax %}p_2{% endmathjax %}之间进行线性差值，从而产生一个新的点{% mathjax %}q_1{% endmathjax %}，随着时间的推移，该点将从{% mathjax %}p_1{% endmathjax %}移动到{% mathjax %}p_2{% endmathjax %}，接下来我们在{% mathjax %}q_0{% endmathjax %}和{% mathjax %}q_1{% endmathjax %}之间创建另一个线性插值。这将为我们提供差值3个点的曲线的差值点的坐标和最终曲线方程。公式如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\mathbf{Q}_0(t) & = (1-t)\mathbf{P}_0 + t\mathbf{P}_1 \\
\mathbf{Q}_1(t) & = (1-t)\mathbf{P}_1 + t\mathbf{P}_2 \\
\mathbf{B}(t) & = (1-t)\mathbf{Q}_0 + t\mathbf{Q}_1 \\
& = (1-t)[(1-t)\mathbf{P}_0 + t\mathbf{P}_1] + t[(1-t)\mathbf{P}_1 + t\mathbf{P}_2] \\
& = (1 - t)^2\mathbf{P}_0 + 2(1-t)t\mathbf{P}_1 + t^2\mathbf{P}_2 
\end{aligned}
{% endmathjax %}

我们有一个公式来计算B样条曲线的方程，而无需进行递归计算，我们一系列的点{% mathjax %}p_0,p1,\ldots,p_n{% endmathjax %},我们可以用这个公式来计算这些点之间的差值。
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{B}(t) = \sum_{i=0}^n 
\left(
    \begin{array}{ccc}
        n \\
        i 
    \end{array} 
\right)
(1-t)^{n-i}t^i\mathbf{P}_i = \sum_{i=0}^n b_{i,n} (t)\mathbf{P}_i
{% endmathjax %}
{% asset_img km_5.png %}

`B`样条曲线是通过控制点{% mathjax %}p_i{% endmathjax %}和B样条曲线基函数{% mathjax %}N_{i,k}(t){% endmathjax %}的线性组合定义的，如下所示：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{r}(t) = \sum_{i=0}^n \mathbf{p}_i N_{i,k}(t),\;\;n\geq k - 1,\;\; t\in [t_{k-1},t_{n+1}]
{% endmathjax %}
在这，控制点被称为`de Boor`点，基函数{% mathjax %}N_{i,k}(t){% endmathjax %}被定义在节点向量上。
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{T} = (t_0,t_1,\ldots,t_{k-1},t_{k},t_{k+1},\ldots, t_{n-1},t_n,t_{n+1},\ldots,t_{n+k})
{% endmathjax %}
其中有{% mathjax %}n+k+1{% endmathjax %}个元素，即控制点的数量({% mathjax %}n+1{% endmathjax %})加上曲线的阶数{% mathjax %}k{% endmathjax %}，每个节点跨度被映射到两个连续关节点{% mathjax %}t_i\leq t\leq t_{i+1}{% endmathjax %}之间的多项式曲线{% mathjax %}r(t_i){% endmathjax %}和{% mathjax %}r(t_{i+1}){% endmathjax %}上。对节点向量进行归一化，使其在区间{% mathjax %}[0,1]{% endmathjax %}之间，这样有助于提高浮点数计算的数值精度，因为该区间{% mathjax %}[133,300]{% endmathjax %}浮点数密度较高。给定一个节点向量{% mathjax %}\mathbf{T}{% endmathjax %}，B样条基函数{% mathjax %}N_{i,k}(t){% endmathjax %}定义为：
{% mathjax '{"conversion":{"em":14}}' %}
N_{i,1}(t) = 
\begin{cases}
    1 \text{ for }t_i\leq t < t_{i+1} \\
    0 \text{ otherwise }
\end{cases}
{% endmathjax %}
即{% mathjax %}k = 1{% endmathjax %}，并且：
{% mathjax '{"conversion":{"em":14}}' %}
N_{i,k}(t) = \frac{t - t_i}{t_{i+k-1} - t_i} N_{i,k-1}(t) + \frac{t_{i+k} - t}{t_{i+k} - t_{i+1}} N_{i+1,k-1}(t)
{% endmathjax %}
即{% mathjax %}k>1{% endmathjax %}。

#### 柯尔莫哥洛夫-阿诺德网络(KAN)

##### 柯尔莫哥洛夫-阿诺德表示定理
在实分析和近似理论中，**柯尔莫哥洛夫-阿诺德表示定理（或叠加定理）**指出，每个多元连续函数{% mathjax %}{\displaystyle f\colon [0,1]^{n}\to \mathbb {R}}{% endmathjax %}可以表示为一个变量的连续函数的二元加法的叠加。它解决了希尔伯特第十三问题的一个更受约束的形式，因此原始的希尔伯特第十三问题是一个推论。如果{% mathjax %}\displaystyle f{% endmathjax %}是多元连续函数，则{% mathjax %}\displaystyle f{% endmathjax %}可以写成由一个单变量连续函数和二元加法运算组成的有限复合函数。更具体地说
{% mathjax '{"conversion":{"em":14}}' %}
{\displaystyle f(\mathbf {x} )=f(x_{1},\ldots ,x_{n})=\sum _{q=0}^{2n}\Phi _{q}\!\left(\sum _{p=1}^{n}\phi _{q,p}(x_{p})\right)}。
{% endmathjax %}
在这里，{% mathjax %}{\displaystyle \phi _{q,p}\colon [0,1]\to \mathbb {R} }{% endmathjax %}和{% mathjax %}{\displaystyle \Phi _{q}\colon \mathbb {R} \to \mathbb {R} }{% endmathjax %}。

假设有一个多元连续函数{% mathjax %}y=f(x_1,x_2){% endmathjax %}，它可以表达为一个有着`2`个输入（{% mathjax %}x_1{% endmathjax %}和{% mathjax %}x_2{% endmathjax %}）、一个输出({% mathjax %}y{% endmathjax %})、以及`5`个隐藏层神经元的`Kolmogorov Network`。隐藏层神经元数量为{% mathjax %}2n+1=5{% endmathjax %}，这里的{% mathjax %}n{% endmathjax %}指的是输入变量的个数。
{% asset_img km_6.png %}

对于第一个神经元，它接收到两个`branch`的信号，分别是{% mathjax %}\Phi_{1,1}(x_1){% endmathjax %}和{% mathjax %}\Phi_{1,2}(x_2){% endmathjax %}，这里的{% mathjax %}\Phi(x_i){% endmathjax %}是{% mathjax %}x_i{% endmathjax %}的一元函数。把{% mathjax %}\Phi_{1,1}(x_1){% endmathjax %}和{% mathjax %}\Phi_{1,2}(x_2){% endmathjax %}简单相加，就得到第一个神经元的取值。以此类推，第`2-5`个神经元也是如此，这是第一层神经元取值的计算方法。为了计算第二层神经元的结果，我们需要对第一层中的每个神经元构造一元函数（{% mathjax %}\Phi_1{% endmathjax %}到{% mathjax %}\Phi_5{% endmathjax %}），然后相加。这里无论是第一层的函数还是第二层的函数，都是一元函数，所以用曲线将其可视化的表达出来。
##### MLP vs KAN

{% asset_img km_7.png %}

##### 多层KAN实现

{% asset_img km_8.png "左：流经网络的激活。右：激活函数被参数化为B样条，它允许在粗粒度和细粒度网格之间切换" %}

{% asset_img km_9.png "KAN网络实现细节" %}

##### KAN参数数量
假设深度为{% mathjax %}L{% endmathjax %}，这些层的宽度为{% mathjax %}n_0 = n_1 = \ldots = n_L= N{% endmathjax %}，在{% mathjax %}G{% endmathjax %}的节点向量区间上每个样条的阶数为{% mathjax %}k{% endmathjax %}（通常{% mathjax %}k=3{% endmathjax %}）。则参数时间复杂度{% mathjax %}\mathcal{O}(N^2 L(G+k))\sim\mathcal{O}(N^2 LG){% endmathjax %}，相比之下，深度为{% mathjax %}L{% endmathjax %}且宽度为{% mathjax %}N{% endmathjax %}的参数时间复杂度为{% mathjax %}\mathcal{O}(N^2 L){% endmathjax %}，这似乎比`KAN`更有效，幸运的是，`KAN`通常比`MLP`需要更小的{% mathjax %}N{% endmathjax %}不仅仅节省了参数，而且还实现了更好的泛化并提高了可解释性。我们注意到，对于一维问题，我们可以取{% mathjax %}N = L = 1{% endmathjax %}，而我们实现中的`KAN`网络只不过是一个样条近似。

##### KAN可解释性

上面我们提出了多种`KAN`简化技术。我们可以将这些简化选项视为可以点击的按钮。与这些按钮交互的用户可以决定接下来最有可能点击哪个按钮以使`KAN`更具可解释性。我们使用下面的示例来展示用户如何与`KAN`交互以获得最大程度可解释性的结果。
{% asset_img km_11.png "如何使用KAN进行符号回归的示例" %}

#### 总结

`KAN`的优势：
- 效率：`KAN`通常比`MLP`具有更高的参数效率。它们可以用更少的参数实现相当甚至更好的准确率。这在涉及回归和偏微分方程(`PDE`)的任务中尤为明显，在这些任务中`KAN`的表现明显优于`MLP`。
- 可解释性：由于使用了样条函数，`KAN`的可解释性更好。与`MLP`中基于节点的固定激活相比，边缘上的激活函数可以更直观地可视化和理解。这使得`KAN`在科学应用中特别有用，因为理解模型的行为至关重要。
- 避免灾难性遗忘：`KAN`已展现出避免灾难性遗忘的能力，这是神经网络中常见的问题，学习新信息可能会导致模型忘记之前学习的信息。这是由于基于样条函数的激活函数的局部可塑性，它只影响附近的样条函数系数，而不会影响远处的系数。

`KAN`的局限性：
- 训练速度慢： `KAN`的训练速度通常比具有相同数量参数的`MLP`慢`10`倍左右。这种低效率被认为是一个工程问题，而不是根本限制，这表明未来还有优化的空间。
- 维数灾难：虽然 `KAN`中使用的样条函数对于低维函数来说是精确的，但由于维数灾难，它们对于高维函数来说却很困难。这是因为样条函数无法有效地利用组合结构，与`MLP`相比，这是一个重大限制。
- 泛化和现实设置：虽然 `KAN`在避免灾难性遗忘和可解释性方面已显示出良好的效果，但它们是否可以推广到更现实和更复杂的设置仍不清楚。

{% asset_img km_10.png "我应该使用KAN还是MLP？" %}

柯尔莫哥洛夫-阿诺德网络(`KAN`)是一种新型神经网络架构，有望突破传统多层感知器(`MLP`)的局限性。其效率、可解释性和对灾难性遗忘的适应能力使其成为特定应用中的有前途的替代方案。然而，在它们完全取代传统神经网络之前，还有许多挑战需要解决。
