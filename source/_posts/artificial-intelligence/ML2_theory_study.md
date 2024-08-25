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

**梯度下降**奠定机器学习中最重要的基石之一。以下是使用**梯度下降**的概述。这里有最小化的{% mathjax %}w,b{% endmathjax %}的成本函数{% mathjax %}\mathbf{J}{% endmathjax %}。目前看到的例子中，这是**线性回归**的**成本函数**，事实证明，**梯度下降**是一种可用于尝试最小化任何函数的算法，而不仅仅是**线性回归**的**成本函数**。**梯度下降**适用于更一般的函数，包括两个以上参数的模型的**成本函数**。例如，如果你有一个成本函数{% mathjax %}\mathbf{J}{% endmathjax %}，它是{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}和{% mathjax %}b{% endmathjax %}的函数，你的目标是最小化参数{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}和{% mathjax %}b{% endmathjax %}上的{% mathjax %}\mathbf{J}{% endmathjax %}。换句话说，你想为{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}和{% mathjax %}和{% mathjax %}b{% endmathjax %}选择值，从而给出{% mathjax %}\mathbf{J}{% endmathjax %}的最小可能值。事实证明，**梯度下降**是一种可用于尝试最小化成本函数{% mathjax %}\mathbf{J}{% endmathjax %}的算法。你要做的只是从{% mathjax %}w{% endmathjax %}和 {% mathjax %}b{% endmathjax %}的一些初始猜测开始。在**线性回归**中，初始值是什么并不重要，因此常见的选择是将它们都设置为`0`。例如，您可以将{% mathjax %}w{% endmathjax %}设置为`0`，将{% mathjax %}b{% endmathjax %}设置为`0`作为初始猜测。使用**梯度下降算法**，您要做的就是，每次都稍微改变参数{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}，以尝试降低{% mathjax %}w,b{% endmathjax %}的成本{% mathjax %}\mathbf{J}{% endmathjax %}，直到{% mathjax %}\mathbf{J}{% endmathjax %}稳定在最小值或接近最小值。我应该注意的一件事是，对于某些可能不是弓形或吊床形的函数{% mathjax %}\mathbf{J}{% endmathjax %}，可能存在多个可能的最小值。让我们看一个更复杂的曲面图 {% mathjax %}\mathbf{J}{% endmathjax %}的示例，看看梯度在做什么。
