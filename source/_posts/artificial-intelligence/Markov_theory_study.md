---
title: 序列模型 (循环神经网络)(TensorFlow)
date: 2024-05-16 17:38:11
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

简言之，如果说卷积神经网络可以有效地处理空间信息，**循环神经网络**(`recurrent neural network，RNN`)则可以更好地处理序列信息。循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出。
<!-- more -->
想象一下有人正在看电影。一名忠实的用户会对每一部电影都给出评价，毕竟一部好电影需要更多的支持和认可。然而事实证明，事情并不那么简单。随着时间的推移，人们对电影的看法会发生很大的变化。简而言之，电影评分决不是固定不变的。因此，使用时间动力学可以得到更准确的电影推荐。当然，序列数据不仅仅是关于电影评分的。处理序列数据需要统计工具和新的深度神经网络架构。为了简单起见，我们以下图所示的股票价格（富时`100`指数）为例。
{% asset_img mar_1.png "近30年的富时100指数" %}

其中，用{% mathjax %}x_t{% endmathjax %}表示价格，即在时间步(`time step`){% mathjax %}t\in \mathbb{Z}^{+}{% endmathjax %}时，观察到的价格{% mathjax %}x_t{% endmathjax %}。请注意，{% mathjax %}t{% endmathjax %}对于本文中的序列通常是离散的，并在证书或其子集上变化。假设一个交易员想在{% mathjax %}t{% endmathjax %}日的股市中表现良好，于是通过以下途径预测{% mathjax %}x_t{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
x_t \sim P(x_t|x_{t-1},\ldots,x_1)
{% endmathjax %}
#### 自回归模型

为了实现这个预测，交易员可以使用回归模型，输入数据的数量这个数字将会随着我们遇到的数据量的增加而增加，因此需要一个近似方法来使这个计算变得容易处理。简单地说，它归结为以下两种策略。
- 第一种策略，假设在现实情况下相当长的序列{% mathjax %}x_{t-1},\dots,x_1{% endmathjax %}可能是不必要的， 因此我们只需要满足某个长度为{% mathjax %}\tau{% endmathjax %}的时间跨度，即使用观测序列{% mathjax %}x_{t-1},\ldots,x_{t-\tau}{% endmathjax %}。当下获得的最直接的好处就是参数的数量总是不变的，至少在{% mathjax %}t > \tau{% endmathjax %}时如此，这就使我们能够训练一个上面提及的深度网络。 这种模型被称为**自回归模型**(`autoregressive models`)，因为它们是对自己执行回归。
- 第二种策略，如下图所示，是保留一些对过去观测的总结{% mathjax %}h_t{% endmathjax %}，并且同时更新预测{% mathjax %}\hat{x}_t{% endmathjax %}和总结{% mathjax %}h_t{% endmathjax %}。这就产生了基于{% mathjax %}\hat{x}_t= P(x_t|h_t){% endmathjax %}估计{% mathjax %}x_t{% endmathjax %}，以及公式{% mathjax %}h_t = g(h_{t-1},x_{t-1}){% endmathjax %}更新的模型。由于{% mathjax %}h_t{% endmathjax %}从未被观测到，这类模型也被称为**隐变量自回归模型**(`latent autoregressive models`)。
{% asset_img mar_2.png "隐变量自回归模型" %}

这两种情况都有一个显而易见的问题：如何生成训练数据？一个经典方法是使用历史观测来预测下一个未来观测。显然，我们并不指望时间会停滞不前。然而，一个常见的假设是虽然特定值{% mathjax %}x_t{% endmathjax %}可能会改变，但是序列本身的动力学不会改变。这样的假设是合理的，因为新的动力学一定受新的数据影响，而我们不可能用目前所掌握的数据来预测新的动力学。统计学家称不变的动力学为静止的(`stationary`)。因此，整个序列的估计值都将通过以下的方式获得：