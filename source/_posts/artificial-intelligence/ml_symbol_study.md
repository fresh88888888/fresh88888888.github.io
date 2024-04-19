---
title: 数学符号—说明（机器学习）
date: 2024-04-19 17:32:11
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

#### 数字

|符号|描述|
|:--|:--|
|{% mathjax %} x{% endmathjax %}|标量|
|{% mathjax %} \mathrm {x}{% endmathjax %}|向量|
|{% mathjax %} \mathbf {X}{% endmathjax %}|矩阵|
|{% mathjax %} \mathsf {X}{% endmathjax %}|张量|
|{% mathjax %} \mathbf {I}{% endmathjax %}|单位矩阵|
|{% mathjax %} x_i, \left [ \mathrm {x}_i \right ]{% endmathjax %}|向量{% mathjax %} x{% endmathjax %}第{% mathjax %} i{% endmathjax %}个元素|
|{% mathjax %} x_{ij}, \left [ \mathbf {X}_{ij} \right ]{% endmathjax %}|矩阵{% mathjax %} \mathbf {X}{% endmathjax %}第{% mathjax %} i{% endmathjax %}行第{% mathjax %} j{% endmathjax %}列的元素|
<!-- more -->
#### 集合论

|符号|描述|
|:--|:--|
|{% mathjax %} \chi{% endmathjax %}|集合|
|{% mathjax %} \mathbb{Z}{% endmathjax %}|整数集合|
|{% mathjax %} \mathbb{R}{% endmathjax %}|实数集合|
|{% mathjax %} \mathbb{R}^n{% endmathjax %}|{% mathjax %} n{% endmathjax %}维实数向量集合|
|{% mathjax %} \mathbb{R}^{a\times b}{% endmathjax %}|包含{% mathjax %} a{% endmathjax %}行和{% mathjax %} b{% endmathjax %}列的实数矩阵集合|
|{% mathjax %} A\cup B{% endmathjax %}|集合{% mathjax %} A{% endmathjax %}和{% mathjax %} B{% endmathjax %}的并集|
|{% mathjax %} A\cap B{% endmathjax %}|集合{% mathjax %} A{% endmathjax %}和{% mathjax %} B{% endmathjax %}的交集|
|{% mathjax %} A\\ B{% endmathjax %}|集合{% mathjax %} A{% endmathjax %}与集合{% mathjax %} B{% endmathjax %}相减，{% mathjax %} B{% endmathjax %}关于{% mathjax %} A{% endmathjax %}的相对补集|
#### 函数与运算符

|符号|描述|
|:--|:--|
|{% mathjax %} f(\cdot){% endmathjax %}|函数|
|{% mathjax %} log(\cdot){% endmathjax %}|自然对数|
|{% mathjax %} exp(\cdot){% endmathjax %}|指数函数|
|{% mathjax %} 1_x{% endmathjax %}|指示函数|
|{% mathjax %} (\cdot)^T{% endmathjax %}|向量或矩阵的转置|
|{% mathjax %} \mathbf {X^{-1}}{% endmathjax %}|矩阵的逆|
|{% mathjax %} [\cdot ,\cdot]{% endmathjax %}|连结|
|{% mathjax %} \mid \chi\mid{% endmathjax %}|集合的基数|
|{% mathjax %} \parallel \cdot\parallel_p{% endmathjax %}|{% mathjax %} L_p{% endmathjax %}正则|
|{% mathjax %} \parallel \cdot\parallel{% endmathjax %}|{% mathjax %} L_2{% endmathjax %}正则|
|{% mathjax %} \langle x, y\rangle{% endmathjax %}|向量{% mathjax %} x(\cdot){% endmathjax %}和{% mathjax %} y{% endmathjax %}的点积|
|{% mathjax %} \sum{% endmathjax %}|连加|
|{% mathjax %} \prod{% endmathjax %}|连乘|
|{% mathjax %} \defeq{% endmathjax %}|定义|
#### 微积分

|符号|描述|
|:--|:--|
|{% mathjax %} \frac {dy}{dx}{% endmathjax %}|{% mathjax %} y{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的导数|
|{% mathjax %} \frac {\partial y}{\partial x}{% endmathjax %}|{% mathjax %} y{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的偏导数|
|{% mathjax %} \nabla_xy{% endmathjax %}|{% mathjax %} y{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的梯度|
|{% mathjax %} \int\nolimits_{a}^{b}f(x)dx{% endmathjax %}|{% mathjax %} f{% endmathjax %}在{% mathjax %} a{% endmathjax %}到{% mathjax %} b{% endmathjax %}区间上关于{% mathjax %} x{% endmathjax %}的定积分|
|{% mathjax %} \int f(x)dx{% endmathjax %}|{% mathjax %} f{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的不定积分|

#### 概率与信息论

|符号|描述|
|:--|:--|
|{% mathjax %} P(\cdot){% endmathjax %}|概率分布|
|{% mathjax %} z\sim P{% endmathjax %}|随机变量{% mathjax %} z{% endmathjax %}具有概率分布{% mathjax %} P{% endmathjax %}|
|{% mathjax %} P(X\mid Y){% endmathjax %}|{% mathjax %} X\mid Y{% endmathjax %}的条件概率|
|{% mathjax %} p(x){% endmathjax %}|概率的密度函数|
|{% mathjax %} E_x[f(x)]{% endmathjax %}|函数{% mathjax %} f{% endmathjax %}对{% mathjax %} x{% endmathjax %}的数学期望|
|{% mathjax %} X\angle Y{% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}是独立的|
|{% mathjax %} X\angle Y\mid Z{% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}在给定随机变量{% mathjax %} Z{% endmathjax %}的条件下是独立的|
|{% mathjax %} Var(X){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}的方差|
|{% mathjax %} \sigma x{% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}的标准差|
|{% mathjax %} Cov(X,Y){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}的协方差|
|{% mathjax %} \rho (X,Y){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}的相关性|
|{% mathjax %} H(X){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}的熵|
|{% mathjax %} D_{KL}(P\parallel Q){% endmathjax %}|{% mathjax %} P{% endmathjax %}和{% mathjax %} Q{% endmathjax %}的KL散度|