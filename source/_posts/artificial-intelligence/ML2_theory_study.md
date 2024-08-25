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



