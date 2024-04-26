---
title: 线性回归（线性神经网络）(PyTorch)
date: 2024-04-25 16:29:11
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

#### 线性回归

**回归**(`regression`)是能为一个或多个自变量与因变量之间关系建模的一类方法。在自然科学和社会科学领域，回归经常用来表示输入和输出之间的关系。在机器学习领域中的大多数任务通常都与预测(`prediction`)有关。当我们向预测一个数值时，就会涉及到回归问题。常见的例子包括：预测价格（房屋、股票等）、预测住院时间（针对住院病人等）、预测需求（零售销量等）。但不是所有的预测都是回归问题。分类问题的目标是预测数据属于一组类别中的哪一个。
<!-- more -->
##### 线性回归的基本元素

线性回归(`linear regression`)可以追溯到19世纪初，它在回归的各种工具中最简单且最流行。线性回归基于几个简单的假设：首先，自变量{% mathjax %}x{% endmathjax %}与因变量{% mathjax %}y{% endmathjax %}之间的关系是线性的，即{% mathjax %}y{% endmathjax %}可以表示为{% mathjax %}x{% endmathjax %}中元素的加权和，这里通常允许包含观测值的一些噪声；其次，我们假设任何噪声都比较正常，如噪声遵循正态分布。

为了解释线性回归，我们举一个实际的例子：我们希望根据房屋的面积（平方英尺）和房龄（年）来估算房屋的价格（美元）。为了开发一个能够预测房价的模型，我们需要收集一些真实的数据集。这个数据集包括了房屋的销售价格、面积和房龄。在机器学习术语中，该数据集称为训练数据集和训练集。每行数据（比如一次房屋交易相对应的数据）称为养样本，也可以称为数据点或数据样本，我们把试图预测的目标（比如预测房屋价格）称为标签(`label`)或目标(`target`)。预测所依据的自变量（面积和房龄）称为特征(`feature`)或协变量(`covariate`)。通常，我们用{% mathjax %}n{% endmathjax %}来表示数据集中的样本数。对索引为{% mathjax %}i{% endmathjax %}的样本其输入表示为{% mathjax %}x^{(i)}=[x_1^{(i)},x_2^{(i)}]^{\mathsf{T}}{% endmathjax %}。

##### 线性模型

线性假设是指目标（房屋价格）可以表示为特征（面积和房龄）的加权和，如下面的式子：
{% mathjax '{"conversion":{"em":14}}' %}
price=w_{area}\cdot area + w_{age}\cdot age + b
{% endmathjax %}

对于{% mathjax %}w_{area}{% endmathjax %}和{% mathjax %}w_{age}{% endmathjax %}称为**权重**`(weight)`，权重决定了每个特征对我们预测值的影响，{% mathjax %}b{% endmathjax %}称为偏置`(bias)`、偏移量`(offset)`或截距`(intercept)`。偏置是指当所有特征取值为`0`时，预测值应该为多少。既使现实生活中不会有房屋的面积为`0`或房龄正好为`0`年，我们仍然需要偏置项。如果没有偏置项，我们模型的表达能力将会受到限制。严格来说，它是输入特征的一个**仿射变换**`(affine transformation)`。仿射变换的特点是通过加权和对特征进行线性变换`(linear transformation)`，并通过偏置项来进行平移`(translation)`。给定一个数据集。我们的目标是寻找模型的权重{% mathjax %}w{% endmathjax %}和偏置{% mathjax %}b{% endmathjax %}，使得根据模型做出的预测大体符合数据里的真实价格。输出的预测值由输入特征通过线性模型的仿射变换决定。仿射变换由所选权重和偏置确定。而在机器学习领域，我们通常使用的是高维数据集，建模时采用线性代数表示法会比较方便。当我们的输入包含{% mathjax %}d{% endmathjax %}个特征时，我们将预测结果{% mathjax %}\hat{y}{% endmathjax %}(通常用“尖角”符号表示{% mathjax %}y{% endmathjax %}的估计值)表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{y}=w_1x_1+\dots +w_dx_d + b
{% endmathjax %}
将所有特征放到向量{% mathjax %}x\in \mathbb{R}^d{% endmathjax %}中，我们可以用点积形式来简洁地表达模型：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{y}=w^{\mathsf{T}}x + b
{% endmathjax %}
向量{% mathjax %}x{% endmathjax %}对应于单个数据样本的特征。用符号表示的矩阵{% mathjax %}X\in \mathbb{R}^{n\times d}{% endmathjax %}可以很方便的引用我们整个数据集的{% mathjax %}n{% endmathjax %}个样本。其中，{% mathjax %}X{% endmathjax %}的每一行是一个样本，每一列是一种特征。对于特征集合{% mathjax %}X{% endmathjax %}，预测值{% mathjax %}\hat{y}\in \mathbb{R}^n{% endmathjax %}可以通过矩阵向量乘法表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{y}=Xw+b
{% endmathjax %}
这个过程中的求和将使用广播机制。给定训练数据特征{% mathjax %}X{% endmathjax %}和对应的已知标签{% mathjax %}y{% endmathjax %}，线性回归的目标是找到一组权重向量{% mathjax %}w{% endmathjax %}和偏置{% mathjax %}b{% endmathjax %}：当给定从{% mathjax %}X{% endmathjax %}的同分布中取样的新样本特征时，这组权重向量和偏置能够使得新样本预测标签的误差尽可能小。虽然我们相信给定{% mathjax %}x{% endmathjax %}预测{% mathjax %}y{% endmathjax %}的最佳模型会是线性的，但我们很难找到一个有{% mathjax %}n{% endmathjax %}个样本的真实数据集，其中对于所有的{% mathjax %}1\leq i \leq n,y^{(i)}{% endmathjax %}完全等于{% mathjax %}\mathbf{w}^{\mathsf{T}}\mathbf{x}^{(i)}+b{% endmathjax %}。无论我们是用什么手段来观察特征{% mathjax %}X{% endmathjax %}和标签{% mathjax %}y{% endmathjax %}，都可能会出现少量的观测误差。因此，即使确信特征与标签的潜在关系是线性的，我们也会加入一个噪声项来考虑观测误差带来的影响。再开始寻找最好的模型参数(`model parameters`){% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}之前，我们还需要两个东西：(1)一种模型质量的度量方式；(2)一种能够更新模型以提高模型预测质量的方法。
##### 损失函数

在我们开始考虑如何用模型拟合(`fit`)数据之前，我们需要确定一个拟合程度的度量。损失函数(`loss function`)能够量化目标的实际值与预测值之间的差距。通常我们会选择非负数作为损失，且数值越小表示损失越小，完美预测时的损失为`0`。回归问题中最常用的损失函数是平方误差函数。当样本{% mathjax %}i{% endmathjax %}的预测值为{% mathjax %}\hat{y}^{(i)}{% endmathjax %}，其相应的真实标签为{% mathjax %}y^{(i)}{% endmathjax %}时，平方误差可以定义为以下公式：
{% mathjax '{"conversion":{"em":14}}' %}
l^{(i)}(\mathbf{w},b)=\frac{1}{2}(\hat{y}^{(i)}-y^{(i)})^2
{% endmathjax %}
常数{% mathjax %}\frac{1}{2}{% endmathjax %}不会带来本质的差别，但这样在形式上稍微简单一些（因为当我们对损失函数求导后常数系数为1）。由于训练数据集并不受我们控制，所以经验误差只是关于模型参数的函数。为了进一步说明，来看下面的例子。我们为一维情况下的回归问题绘制图像，如下图所示：
{% asset_img r_1.png %}

由于平方误差函数中的二次方项，估计值{% mathjax %}\hat{y}^{(i)}{% endmathjax %}和观测值{% mathjax %}y^{(i)}{% endmathjax %}之间较大的差异将导致更大的损失。为了度量模型在整个数据集上的质量，我们需计算训练集{% mathjax %}n{% endmathjax %}个样本上的误差均值（也等价于求和）。
{% mathjax '{"conversion":{"em":14}}' %}
L(\mathbf{w},b) = \frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(\mathbf{w}^{\mathsf{T}}x^{(i)} + b - y^{(i)})^2
{% endmathjax %}
在训练模型时，我们希望寻找一组参数{% mathjax %}(\mathbf{w}^{\ast},b^{\ast}){% endmathjax %}，这组参数能最小化在所有训练样本上的总损失。如下：
{% mathjax '{"conversion":{"em":14}}' %}
\DeclareMathOperator*{\argmin}{argmin}
\mathbf{w}^{\ast},b^{\ast} = \argmin_{\mathbf{w},b} L(\mathbf{w},b)
{% endmathjax %}
##### 解析解

线性回归刚好是一个很简单的优化问题。线性回归的解可以用一个公式简单地表达出来，这类解叫做**解析解**(`analytical solution`)。首先我们将偏置`b`合并到参数{% mathjax %}\mathbf{w}{% endmathjax %}中，合并方法是在包含所有参数的矩阵中附加一列。我们的预测问题是最小化{% mathjax %}\lVert \mathbf{y}-\mathbf{Xw} \rVert^2{% endmathjax %}。这在损失平面上只有一个临界点，这个临界点对应整个平面的损失极小点。将损失关于{% mathjax %}w{% endmathjax %}的导数设为0，得到解析解：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{w}^{\ast} = (\mathbf{X}^{\mathsf{T}}\mathbf{X})^{-1} \mathbf{X}^{\mathsf{T}}\mathbf{y}
{% endmathjax %}
像线性回归这样的简单问题存在解析解，但并不是所有的问题都存在解析解。解析解可以很好的进行数学分析，但解析解对问题的限制很严格，导致它无法广泛的应用在深度学习当中。
##### 随机梯度下降

