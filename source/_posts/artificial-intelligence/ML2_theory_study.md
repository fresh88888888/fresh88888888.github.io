---
title: 机器学习(ML)(二) — 探析
date: 2024-08-25 17:45:11
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

#### 梯度下降

我们看到了**成本函数**{% mathjax %}\mathbf{J}{% endmathjax %}的可视化，以及如何尝试选择不同的参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}。如果我们有一种更系统的方法来找到{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}的值，从而得到{% mathjax %}w、b{% endmathjax %}的最小成本{% mathjax %}\mathbf{J}{% endmathjax %}。事实证明，有一种称为**梯度下降**的算法可实现这一点。**梯度下降**在机器学习中随处可见，不仅用于**线性回归**，还用于训练一些最先进的神经网络模型，也称为**深度学习模型**。
<!-- more -->

**梯度下降**奠定机器学习中最重要的基石之一。这里有最小化的{% mathjax %}w,b{% endmathjax %}的成本函数{% mathjax %}\mathbf{J}{% endmathjax %}。目前看到的例子中，这是**线性回归**的**成本函数**，事实证明，**梯度下降**是一种可用于尝试最小化任何函数的算法，而不仅仅是**线性回归**的**成本函数**。**梯度下降**适用于更一般的函数，包括两个以上参数的模型的**成本函数**。例如，如果你有一个**成本函数**{% mathjax %}\mathbf{J}{% endmathjax %}，它是{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}和{% mathjax %}b{% endmathjax %}的函数，你的目标是最小化参数{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}和{% mathjax %}b{% endmathjax %}上的{% mathjax %}\mathbf{J}{% endmathjax %}。换句话说，你想为{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}和{% mathjax %}b{% endmathjax %}选择值，从而给出{% mathjax %}\mathbf{J}{% endmathjax %}的最小可能值。事实证明，**梯度下降**是一种可用于尝试最小化成本函数{% mathjax %}\mathbf{J}{% endmathjax %}的算法。你要做的只是从{% mathjax %}w{% endmathjax %}和 {% mathjax %}b{% endmathjax %}的一些初始猜测开始。在**线性回归**中，初始值是什么并不重要，因此常见的选择是将它们都设置为`0`。例如，您可以将{% mathjax %}w{% endmathjax %}设置为`0`，将{% mathjax %}b{% endmathjax %}设置为`0`作为初始猜测。使用**梯度下降算法**，您要做的就是，每次都稍微改变参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}，以尝试降低{% mathjax %}w,b{% endmathjax %}的成本{% mathjax %}\mathbf{J}{% endmathjax %}，直到{% mathjax %}\mathbf{J}{% endmathjax %}稳定在最小值或接近最小值。我应该注意的一件事是，对于某些可能不是弓形或吊床形的函数{% mathjax %}\mathbf{J}{% endmathjax %}，可能存在多个可能的最小值。让我们看一个更复杂的曲面图 {% mathjax %}\mathbf{J}{% endmathjax %}的示例，看看梯度在做什么。

此函数不是**平方误差成本函数**。对于具有平方误差成本函数的**线性回归**，您总是会得到弓形或吊床形。但如果您训练神经网络模型，您可能会得到这种类型的成本函数。注意轴，即底部轴上的 {% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}。对于不同的{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}值，你会得到这个曲面上的不同点，{% mathjax %}w,b{% endmathjax %}的{% mathjax %}\mathbf{J}{% endmathjax %}，其中曲面在某个点的高度是**成本函数**的值。现在，让我们想象一下，这个曲面图实际上是一个略微丘陵的户外公园，其中高点是山丘，低点是山谷，就像这样。想象一下，你现在正站在山上的这个点上。如果这能帮助你放松，想象一下有很多非常漂亮的绿草、蝴蝶和鲜花，这是一座非常漂亮的山。你的目标是从这里出发，尽可能高效地到达其中一个山谷的底部。**梯度下降算法**的作用是，你要旋转`360`度，环顾四周，问自己，如果我要朝一个方向迈出一小步，我想尽快下坡到其中一个山谷。我会选择朝哪个方向迈出这一小步？如果你想尽可能高效地走下这座山，那么如果你站在山上的这个点上环顾四周，你会发现，你下一步下山的最佳方向大致就是那个方向。从数学上讲，这是下降速度最快的方向。意味着，当你迈出一小步时，这会比你朝其他方向迈出一小步的速度更快。迈出第一步后，你现在就站在山上的这个点上。现在让我们重复这个过程。站在这个新的点上，你将再次旋转`360`度，问自己，下一步我要朝哪个方向迈出一小步才能下山？如果你这样做，再迈出一步，你最终会朝那个方向移动一点，然后你就可以继续走下去了。从这个新的点开始，你可以再次环顾四周，决定哪个方向可以让你最快地下山。再走一步，再走一步，等等，直到你发现自己到了山谷的底部，到了这个局部最小值，就在这里。你刚才做的是经过**多步梯度下降**。**梯度下降**有一个有趣的特性。你可以通过选择参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}的起始值来选择表面的起点。刚才执行**梯度下降**时，你是从这里的这个点开始的。现在，想象一下，如果你再次尝试梯度下降，但这次你选择一个不同的起点，通过选择将你的起点放在这里右边几步的参数。如果你重复梯度下降过程，这意味着你环顾四周，朝着最陡峭的上升方向迈出一小步，你就会到达这里。然后你再次环顾四周，再迈出一步，依此类推。如果你第二次运行**梯度下降**，从我们第一次执行的位置的右​​边几步开始，那么你最终会到达一个完全不同的山谷。右边这个不同的最小值。第一个和第二个山谷的底部都称为**局部最小值**。因为如果你开始沿着第一个山谷向下走，梯度下降不会带你到第二个山谷，同样，如果你开始沿着第二个山谷向下走，你会停留在第二个最小值，而不会找到进入第一个局部最小值的路。
{% asset_img ml_1.png %}

让我们看看如何实现**梯度下降算法**。在每一步中，参数{% mathjax %}w{% endmathjax %}都会更新为{% mathjax %}w = w - \alpha\frac{\partial}{\partial w}J(w,b){% endmathjax %}。这个表达式的意思是，取 {% mathjax %}w{% endmathjax %}的当前值并对其进行少量调整，也就是右边的这个表达式，减去{% mathjax %}\alpha{% endmathjax %}乘以这里的这个项。

具体来说，在这个上下文中，如果你写的代码是{% mathjax %}a = c{% endmathjax %}，这意味着取值{% mathjax %}c{% endmathjax %}并将其存储在你的计算机中，在变量{% mathjax %}a{% endmathjax %}中。或者你写 {% mathjax %}a = a + 1{% endmathjax %}，意味着将{% mathjax %}a{% endmathjax %}的值设置为{% mathjax %}a+1{% endmathjax %}。在`Python`和其他编程语言中，真值断言有时写成等于，所以如果你在测试{% mathjax %}a{% endmathjax %}是否等于{% mathjax %}c{% endmathjax %}。在这个等式中，{% mathjax %}\alpha{% endmathjax %}也称为**学习率**。**学习率**通常是`0~1`之间的一个正数，比如`0.01`。{% mathjax %}\alpha{% endmathjax %}的作用是，它基本上控制你下坡的步数。如果{% mathjax %}\alpha{% endmathjax %}非常大，那么这相当于一个非常激进的**梯度下降**过程。如果{% mathjax %}\alpha{% endmathjax %}非常小，那么你就会小步走下坡路。最后，这里的这个项是**成本函数**{% mathjax %}\mathbf{J}{% endmathjax %} 的导数项。结合学习率{% mathjax %}\alpha{% endmathjax %}，它还决定了你想要走下坡路的步数。记住模型有两个参数，不仅仅是{% mathjax %}w{% endmathjax %}，还有{% mathjax %}b{% endmathjax %}。你还有一个赋值操作来更新看起来非常相似的参数{% mathjax %}b{% endmathjax %}。{% mathjax %}b = b - \alpha\frac{\partial}{\partial b}J(w,b){% endmathjax %}。记住，在曲面图的图形中，你一步步地走，直到到达值的底部，对于**梯度下降算法**，你将重复这两个更新步骤，直到算法收敛。所谓**收敛**，意思是你到达局部最小值，此时参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}不再随着你采取的每个额外步骤而发生很大变化。现在，关于如何正确地进行**语义梯度下降**，还有一个更微妙的细节，你将更新两个参数 {% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}。一个重要的细节是，对于**梯度下降**，你想同时更新{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}，在这个表达式中，你将把{% mathjax %}w{% endmathjax %}从旧的{% mathjax %}w{% endmathjax %}更新为新的{% mathjax %}w{% endmathjax %}，你还将把{% mathjax %}b{% endmathjax %}从最旧值更新为新的{% mathjax %}b{% endmathjax %}值。实现的方法是计算右边，计算{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}的值，同时将{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}更新为新值。这是实现**梯度下降**的正确方法，它会同时进行更新。当你听到有人谈论**梯度下降**时，是指执行参数同步更新的**梯度下降**。

这个希腊符号{% mathjax %}\alpha{% endmathjax %}，是**学习率**。**学习率**控制更新模型参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}时采取的步长。这个{% mathjax %}\frac{\partial}{\partial w}{% endmathjax %}，是一个导数项。让我们使用一个稍微简单的例子，最小化一个参数。假设你只有一个参{% mathjax %}w{% endmathjax %}的**成本函数**{% mathjax %}\mathbf{j}{% endmathjax %}，其中{% mathjax %}w{% endmathjax %}是一个数字。**梯度下降**现在看起来像这样，{% mathjax %}w = w - \alpha\frac{\partial}{\partial w}J(w,b){% endmathjax %}。你试图通过调整参数{% mathjax %}w{% endmathjax %}来最小化成本。现在，我们来初始化梯度下降，并为{% mathjax %}w{% endmathjax %}设定一个起始值。在这个位置初始化它。你从函数{% mathjax %}\mathbf{J}{% endmathjax %}的这个点开始，梯度下降会将{% mathjax %}w{% endmathjax %}更新为{% mathjax %}w = w - \alpha\frac{\partial}{\partial w}J(w,b){% endmathjax %}。让我们看看这里的导数项是什么意思。考虑直线上这一点的导数的一种方法是画一条切线，它是一条在该点与曲线相切的直线。这条线的斜率是函数{% mathjax %}\mathbf{J}{% endmathjax %}在该点的导数。要得到斜率，你可以画一个像这样的小三角形。如果你计算这个三角形的高度除以宽度，那就是**斜率**。例如，这个斜率可能是`2/1`，当切线指向右上方时，斜率为正，这意味着这个导数是一个正数，所以大于`0`。更新后的{% mathjax %}w{% endmathjax %}将是{% mathjax %}w{% endmathjax %}减去学习率乘以某个正数。**学习率**始终是正数。如果用{% mathjax %}w{% endmathjax %}减去一个正数，最终会得到一个较小的{% mathjax %}w{% endmathjax %}新值。在图表上，向左移动，{% mathjax %}w{% endmathjax %}的值就会减小。您可能会注意到，如果您的目标是降低成本{% mathjax %}\mathbf{J}{% endmathjax %}，那么这样做是正确的，因为当我们沿着这条曲线向左移动时，成本{% mathjax %}\mathbf{J}{% endmathjax %}会减小，并且您会越来越接近{% mathjax %}\mathbf{J}{% endmathjax %}的最小值。到目前为止，梯度下降似乎做得对。现在，让我们看另一个例子。让我们采用与上面相同的{% mathjax %}w{% endmathjax %}函数{% mathjax %}\mathbf{J}{% endmathjax %}，现在假设您在不同的位置初始化梯度下降。比如说，通过选择{% mathjax %}w{% endmathjax %}的起始值，它就在左边。这就是函数{% mathjax %}\mathbf{J}{% endmathjax %}的点。导数项是{% mathjax %}\frac{\mathbf{J}(w)\partial}{\partial w}{% endmathjax %}，当我们查看此处的切线时，这条线的斜率就是{% mathjax %}\mathbf{J}{% endmathjax %}在此点的导数。但是这条切线是向右下方倾斜的。这条向右下方倾斜的线具有负斜率。换句话说，{% mathjax %}\mathbf{J}{% endmathjax %}在这一点的导数是负数。例如，如果你画一个三角形，那么像这样的高度是`-2`，宽度是`1`，斜率就是`-2/1`，也就是`-2`，这是一个负数。当你更新{% mathjax %}w{% endmathjax %}时，你会得到{% mathjax %}w{% endmathjax %}减去学习率乘以一个负数。这意味着你从{% mathjax %}w{% endmathjax %}中减去一个负数。但减去一个负数意味着增加一个正数，所以你最终会增加{% mathjax %}w{% endmathjax %}。梯度下降的这一步会导致{% mathjax %}w{% endmathjax %}增加，意味着你正在向图的右侧移动。**梯度下降算法**中的另一个关键量是**学习率**{% mathjax %}\alpha{% endmathjax %}。

##### 学习率

**学习率**{% mathjax %}\alpha{% endmathjax %}的选择将对**梯度下降**的效率产生巨大影响。{% mathjax %}w = w - \alpha\frac{\partial}{\partial w}J(w,b){% endmathjax %}。要了解有关学习率{% mathjax %}\alpha{% endmathjax %}的更多信息。让我们看看学习率{% mathjax %}\alpha{% endmathjax %}太小或太大会发生什么。对于学习率太小的情况。这是一张图表，其中横轴是{% mathjax %}w{% endmathjax %}，纵轴是成本{% mathjax %}\mathbf{J}{% endmathjax %}。从此处开始分级下降，如果学习率太小。这个数字非常小，比如`0.0000001`。所以你最终会迈出一小步。然后从这一点开始，你将迈出另一个微小的婴儿步。但由于学习率非常小，第二步也微不足道。这个过程的结果是，你最终会降低成本{% mathjax %}\mathbf{J}{% endmathjax %}，但速度非常慢。但你可能会注意到，你需要很多步才能达到最小值。总结一下，如果学习率太小，那么**梯度下降**会起作用，但速度会很慢。这将需要很长时间，因为它需要采取这些微小的婴儿步。而且它需要很多步才能接近最小值。如果学习率太大会发生什么？这是成本函数的另一张图。假设我们从这里的{% mathjax %}w{% endmathjax %}值开始缓慢下降。实际上，它已经非常接近最小值了。但是，如果学习率太大，那么你就会以非常大的步长更新{% mathjax %}w{% endmathjax %}。现在，成本变得更糟了。如果学习率太大，那么你又会加速迈出一大步，再次超过最小值。所以现在你到了右边的这个点，再一次进行更新。您可能会注意到，您实际上离最小值越来越远。如果学习率太大，那么创建的感觉可能会超调，并且可能永远无法达到最小值。换句话说，大交叉点可能无法收敛，甚至可能**发散**。假设有成本函数{% mathjax %}\mathbf{J}{% endmathjax %}。您在这里看到的不是平方误差成本函数，并且该成本函数有两个局部最小值，对应于您在这里看到的两个谷值。现在假设经过一些梯度下降步骤后，您的参数{% mathjax %}w{% endmathjax %}就在这里，比如等于`5`。这是{% mathjax %}w{% endmathjax %}的当前值。这意味着你现在处于成本函数{% mathjax %}\mathbf{J}{% endmathjax %}的这个点。如果你注意这个点的函数，就会发现这恰好是一个局部最小值。这条线的斜率为零。如果你已经处于局部最小值，梯度下降将使{% mathjax %}w{% endmathjax %}保持不变。因为它只是将{% mathjax %}w{% endmathjax %}的新值更新为与{% mathjax %}w{% endmathjax %}的旧值完全相同。具体来说，假设{% mathjax %}w{% endmathjax %}的当前值为`5`。一次迭代后，它仍然等于`5`。因此，如果你的参数已经将你带到了局部最小值，然后进一步的梯度下降会趋于零。它不会改变参数，这正是你想要的，因为它将解决方案保持在局部最小值。这也解释了为什么梯度下降可以达到局部最小值，即使学习率{% mathjax %}\alpha{% endmathjax %}固定。这是我们想要最小化的{% mathjax %}w{% endmathjax %}的成本函数{% mathjax %}\mathbf{J}{% endmathjax %}。让我们在这一点初始化梯度下降。如果我们采取一个更新步骤，也许它会带我们到那个点。而且因为这个导数相当大，所以梯度下降会采取一个相对较大的步骤。现在，我们处于第二个点，我们再迈出一步。你可能会注意到斜率不像第一个点那么陡峭。所以导数没有那么大。所以下一个更新步骤不会像第一步那么大。这里的第三点，导数比上一步要小。当我们接近最小值时，它会采取更小的步骤。越来越接近零。因此，当我们运行梯度下降时，最终我们会采取非常小的步长，直到最终达到局部最小值。当我们接近局部最小值时，**梯度下降**将自动采取较小的步长。这是因为当我们接近局部最小值时，**导数**会自动变小。这意味着更新步骤也会自动变小。学习率{% mathjax %} \alpha{% endmathjax %}保持在某个固定值。这就是**梯度下降算法**，你可以用它来尝试最小化任何成本函数{% mathjax %}\mathbf{J}{% endmathjax %}。不仅仅是**均方误差成本函数**。
{% asset_img ml_2.png %}

##### 线性回归的梯度下降

如下图所示，这是**线性回归模型**。右边是**平方误差成本函数**。下面是**梯度下降算法**。如果您计算这些导数，您将得到这些项。关于{% mathjax %}w{% endmathjax %}的导数是{% mathjax %}1/m{% endmathjax %}，{% mathjax %}i{% endmathjax %}的范围是{% mathjax %}1,\ldots,m{% endmathjax %}`。然后是**误差项**，即预测值与实际值之间的差乘以输入特征{% mathjax %}x^{(i)}{% endmathjax %}。关于{% mathjax %} b{% endmathjax %}的导数是这里的公式，它看起来与上面的公式相同，只是最后没有那个{% mathjax %}x^{(i)}{% endmathjax %}项。如果您使用这些公式来计算这两个导数并以这种方式实现**梯度下降**，它就会起作用。现在，你可能想知道，我从哪里得到这些公式？它们是使用**微积分**推导出来的。如何计算**导数项**。让我们从第一项开始。成本函数{% mathjax %}\mathbf{J}{% endmathjax %}对{% mathjax %}w{% endmathjax %}的导数。我们首先代入成本函数{% mathjax %}\mathbf{J}{% endmathjax %}的定义{% mathjax %}\mathbf{J}(w,b) = \frac{1}{2m}\sum_{i=1}^m(f_{w,b}(x^{(i)}) - y^{(i)})^2{% endmathjax %}。我们想要做的是计算这个方程的导数，也称为{% mathjax %}w{% endmathjax %}的偏导数。带入公式推导出：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\frac{\partial}{\partial w}\mathbf{J}(w,b) & = \frac{\partial}{\partial w}\frac{1}{2m}\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2 \\
& = \frac{\partial}{\partial w}\frac{1}{2m}\sum_{i=1}^m (wx^{(i)} + b - y^{(i)})^2 \\
& = \frac{1}{2m}\sum_{i=1}^m (wx^{(i)} + b - y^{(i)})2x^{(i)} \\
& = \frac{1}{m}\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)}
\end{aligned}
{% endmathjax %}

{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\frac{\partial}{\partial b}\mathbf{J}(w,b) & = \frac{\partial}{\partial b}\frac{1}{2m}\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})^2 \\
& = \frac{\partial}{\partial b}\frac{1}{2m}\sum_{i=1}^m (wx^{(i)} + b - y^{(i)})^2 \\
& = \frac{1}{2m}\sum_{i=1}^m (wx^{(i)} + b - y^{(i)})2 \\
& = \frac{1}{m}\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})
\end{aligned}
{% endmathjax %}

{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
w & = w - \alpha\frac{\partial}{\partial w}\mathbf{J}(w,b) = w - \alpha\frac{1}{m}\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \\
b & = b - \alpha\frac{\partial}{\partial b}\mathbf{J}(w,b) = b - \alpha\frac{1}{m}\sum_{i=1}^m (f_{w,b}(x^{(i)}) - y^{(i)})
\end{aligned}
{% endmathjax %}
根据您初始化参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}的位置，您可能会得到不同的局部最小值。当你在**线性回归**中使用**平方误差成本函数**时，成本函数永远不会有多个局部最小值。它只有一个**全局最小值**。这个成本函数是一个凸函数。通俗地说，**凸函数**是碗状函数，除了单个全局最小值外，它不能有任何局部最小值。当你在**凸函数**上实现**梯度下降**时，只要你选择适当的**学习率**，它就会一直收敛到**全局最小值**。
{% asset_img ml_3.png %}

****

{% asset_img ml_4.png %}

让我们看看**在线性回归**中运行**梯度下降**时会发生什么？如下图所示，左上角是模型和数据的图，右上角是**成本函数**的轮廓图，底部是同一成本函数的表面图。通常{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}都会初始化为`0`，但为了演示，我们初始化{% mathjax %}w = -0.1{% endmathjax %}和{% mathjax %}b=900{% endmathjax %}。因此这对应于{% mathjax %}f(x) = -0.1x + 900{% endmathjax %}。如果我们使用**梯度下降**迈出一步，最终会从成本函数的这一点向右下方移动到下一点，注意到直线拟合也发生了一点变化。**成本函数**现在已移至第三个点，并且函数{% mathjax %}f(x){% endmathjax %}也发生了一些变化。随着更多步骤，每次更新的成本都会下降。所以参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}遵循这个轨迹。看左边，你会看到相应的直线拟合越来越好，直到我们达到全局最小值。全局最小值对应于这个直线拟合，它与数据的拟合相对较好。这就是**梯度下降**，我们将用它来拟合模型以适配数据。
{% asset_img ml_5.png %}