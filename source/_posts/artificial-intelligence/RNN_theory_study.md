---
title: 循环神经网络模型 (RNN)(TensorFlow)
date: 2024-05-17 14:38:11
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

我们学习了{% mathjax %}n{% endmathjax %}元语法模型，其中单词{% mathjax %}x_t{% endmathjax %}在时间步{% mathjax %}t{% endmathjax %}的条件概率仅取决于前面{% mathjax %}n-1{% endmathjax %}个单词。对于时间步{% mathjax %}t - (n - 1){% endmathjax %}之前的单词，如果我们想将其可能产生的影响合并到{% mathjax %}x_t{% endmathjax %}上，需要增加{% mathjax %}n{% endmathjax %}，然而模型参数的数量也会随之呈指数增长，因为词表{% mathjax %}\mathcal{V}{% endmathjax %}需要存储{% mathjax %}|\mathcal{V}^n|{% endmathjax %}个数字，因此与其将{% mathjax %}P(x_t|x_{t-1},\ldots,x_{t-n+1}){% endmathjax %}模型化，不如使用隐变量模型：
{% mathjax '{"conversion":{"em":14}}' %}
P(x_t|x_{t-1},\ldots,x_1) \approx P(x_t|h_{t-1})
{% endmathjax %}
<!-- more -->
其中{% mathjax %}h_{t-1}{% endmathjax %}是**隐状态**(`hidden state`)，也称为**隐藏变量**(`hidden variable`)，它存储了到时间步{% mathjax %}t-1{% endmathjax %}的序列信息。通常，我们可以基于当前输入{% mathjax %}x_t{% endmathjax %}和先前隐状态{% mathjax %}h_{t-1}{% endmathjax %}来计算时间步{% mathjax %}t{% endmathjax %}处的任何时间的隐状态：
{% mathjax '{"conversion":{"em":14}}' %}
h(t) = f(x_t,h_{t-1})
{% endmathjax %}
对于函数{% mathjax %}f{% endmathjax %}，隐变量模型不是近似值。毕竟{% mathjax %}h_t{% endmathjax %}是可以仅仅存储到目前为止观察到的所有数据，然而这样的操作可能会使计算和存储的代价都变得昂贵。回想一下，我们讨论过的具有隐藏单元的隐藏层。值得注意的是，隐藏层和隐状态指的是两个截然不同的概念。如上所述，隐藏层是在从输入到输出的路径上（以观测角度来理解）的隐藏的层，而隐状态则是在给定步骤所做的任何事情（以技术角度来定义）的输入，并且这些状态只能通过先前时间步的数据来计算。**循环神经网络**(`recurrent neural networks，RNNs`)是具有隐状态的神经网络。