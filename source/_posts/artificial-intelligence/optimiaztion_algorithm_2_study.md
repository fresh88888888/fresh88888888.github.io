---
title: 优化算法 (机器学习)(TensorFlow)
date: 2024-05-29 12:24:11
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

#### RMSProp算法

`RMSProp`算法作为将速率调度与坐标自适应学习率分离的简单修复方法。问题在于，`Adagrad`算法将梯度{% mathjax %}\mathbf{g}_t{% endmathjax %}的平方累加成状态矢量{% mathjax %}\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2{% endmathjax %}。因此，由于缺乏规范化，没有约束力，{% mathjax %}\mathbf{s}_t{% endmathjax %}持续增长，几乎是在算法收敛时呈线性递增。解决此问题的一种方法是使用{% mathjax %}\mathbf{s}_t/t{% endmathjax %}。对{% mathjax %}\mathbf{g}_t{% endmathjax %}的合理分布来说，它将收敛。遗憾的是，限制行为生效可能需要很长时间，因为该流程记住了值的完整轨迹。另一种方法是按动量法中的方式使用泄漏平均值，即{% mathjax %}\mathbf{s}_t \leftarrow\gamma \mathbf{s}_{t-1} + (1-\gamma)\mathbf{g}_t^2{% endmathjax %}，其中参数{% mathjax %}\gamma > 0{% endmathjax %}。保持所有其它部分不变就产生了`RMSProp`算法。
<!-- more -->
##### 算法

让我们详细写出这些方程式。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \mathbf{s}_t \leftarrow\gamma \mathbf{s}_{t-1} + (1-\gamma)\mathbf{g}_t^2 \\ 
& \mathbf{x}_t \leftarrow\mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \odot\mathbf{g}_t 
\end{align}
{% endmathjax %}
常数{% mathjax %}\epsilon > 0{% endmathjax %}通常设置为{% mathjax %}10^{-6}{% endmathjax %}，以确保我们不会因除以零或步长过大而受到影响。鉴于这种扩展，我们现在可以自由控制学习率{% mathjax %}\eta{% endmathjax %}，而不考虑基于每个坐标应用的缩放。就泄漏平均值而言，我们可以采用与之前在动量法中适用的相同推理。扩展{% mathjax %}\mathbf{s}_t{% endmathjax %}定义可获得：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{s}_t & = (1-\gamma)\mathbf{g}_t^2 + \gamma\mathbf{g}_t^2 \\ 
& = (1-\gamma)(\mathbf{g}_t^2 + \gamma\mathbf{g}_{t-1}^2 + \gamma^2\mathbf{g}_{t-1} + \ldots,)
\end{align}
{% endmathjax %}
##### 总结 

`RMSProp`算法与`Adagrad`算法非常相似，因为两者都使用梯度的平方来缩放系数。`RMSProp`算法与动量法都使用泄漏平均值。但是，`RMSProp`算法使用该技术来调整按系数顺序的预处理器。在实验中，学习率需要由实验者调度。系数{% mathjax %}\gamma{% endmathjax %}决定了在调整每坐标比例时历史记录的时长。