---
title: 数学符号&名词解释（机器学习）
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

#### 引言

为了解决各种各样的机器学习问题，深度学习提供了强大的工具。虽然许多深度学习方法都是最近才有重大突破，但使用数据和神经网络编程的核心思想已经研究了几个世纪。事实上，人类长期以来就有分析数据和预测未来结果的愿望，而自然科学大部分都植根于此。例如，伯努利分布是以雅各布•伯努利（`1654-1705`）命名的。而高斯分布是由卡尔•弗里德里希•高斯（`1777-1855`）发现的，他发明了最小均方算法，至今仍用于解决从保险计算到医疗诊断的许多问题。这些工具算法催生了自然科学中的一种实验方法—例如，电阻中电流和电压的欧姆定律可以用线性模型完美地描述。
<!-- more -->
机器学习的关键组件：
- 可以用来学习的数据（`data`）。
- 如何转换数据的模型（`model`），我觉得也可以称为结构体。
- 一个目标函数（`objective function`）或者叫评估指标，用来量化模型的有效性。
- 调整模型参数以优化目标函数的算法（`algorithm`）。

#### 数学符号

##### 数字

|符号|描述|
|:--|:--|
|{% mathjax %} x{% endmathjax %}|标量|
|{% mathjax %} \mathrm {x}{% endmathjax %}|向量|
|{% mathjax %} \mathbf {X}{% endmathjax %}|矩阵|
|{% mathjax %} \mathsf {X}{% endmathjax %}|张量|
|{% mathjax %} \mathbf {I}{% endmathjax %}|单位矩阵|
|{% mathjax %} x_i, \left [ \mathrm {x}_i \right ]{% endmathjax %}|向量{% mathjax %} x{% endmathjax %}第{% mathjax %} i{% endmathjax %}个元素|
|{% mathjax %} x_{ij}, \left [ \mathbf {X}_{ij} \right ]{% endmathjax %}|矩阵{% mathjax %} \mathbf {X}{% endmathjax %}第{% mathjax %} i{% endmathjax %}行第{% mathjax %} j{% endmathjax %}列的元素|

##### 集合论

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
##### 函数与运算符

|符号|描述|
|:--|:--|
|{% mathjax %} f(\cdot){% endmathjax %}|函数|
|{% mathjax %} log(\cdot){% endmathjax %}|自然对数|
|{% mathjax %} exp(\cdot){% endmathjax %}|指数函数|
|{% mathjax %} 1_x{% endmathjax %}|指示函数|
|{% mathjax %} (\cdot)^T{% endmathjax %}|向量或矩阵的转置|
|{% mathjax %} \mathbf {X^{-1}}{% endmathjax %}|矩阵的逆|
|{% mathjax %} \odot{% endmathjax %}|按元素相乘|
|{% mathjax %} [\cdot ,\cdot]{% endmathjax %}|连结|
|{% mathjax %} \mid \chi\mid{% endmathjax %}|集合的基数|
|{% mathjax %} \parallel \cdot\parallel_p{% endmathjax %}|{% mathjax %} L_p{% endmathjax %}正则|
|{% mathjax %} \parallel \cdot\parallel{% endmathjax %}|{% mathjax %} L_2{% endmathjax %}正则|
|{% mathjax %} \langle x, y\rangle{% endmathjax %}|向量{% mathjax %} x(\cdot){% endmathjax %}和{% mathjax %} y{% endmathjax %}的点积|
|{% mathjax %} \sum{% endmathjax %}|连加|
|{% mathjax %} \prod{% endmathjax %}|连乘|
|{% mathjax %} \stackrel{def}{=}{% endmathjax %}|定义|
##### 微积分

|符号|描述|
|:--|:--|
|{% mathjax %} \frac {dy}{dx}{% endmathjax %}|{% mathjax %} y{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的导数|
|{% mathjax %} \frac {\partial y}{\partial x}{% endmathjax %}|{% mathjax %} y{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的偏导数|
|{% mathjax %} \nabla_xy{% endmathjax %}|{% mathjax %} y{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的梯度|
|{% mathjax %} \int\nolimits_{a}^{b}f(x)dx{% endmathjax %}|{% mathjax %} f{% endmathjax %}在{% mathjax %} a{% endmathjax %}到{% mathjax %} b{% endmathjax %}区间上关于{% mathjax %} x{% endmathjax %}的定积分|
|{% mathjax %} \int f(x)dx{% endmathjax %}|{% mathjax %} f{% endmathjax %}关于{% mathjax %} x{% endmathjax %}的不定积分|

##### 概率与信息论

|符号|描述|
|:--|:--|
|{% mathjax %} P(\cdot){% endmathjax %}|概率分布|
|{% mathjax %} z\sim P{% endmathjax %}|随机变量{% mathjax %} z{% endmathjax %}具有概率分布{% mathjax %} P{% endmathjax %}|
|{% mathjax %} P(X\mid Y){% endmathjax %}|{% mathjax %} X\mid Y{% endmathjax %}的条件概率|
|{% mathjax %} p(x){% endmathjax %}|概率的密度函数|
|{% mathjax %} E_x[f(x)]{% endmathjax %}|函数{% mathjax %} f{% endmathjax %}对{% mathjax %} x{% endmathjax %}的数学期望|
|{% mathjax %} X\bot Y{% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}是独立的|
|{% mathjax %} X\bot Y\mid Z{% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}在给定随机变量{% mathjax %} Z{% endmathjax %}的条件下是独立的|
|{% mathjax %} Var(X){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}的方差|
|{% mathjax %} \sigma x{% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}的标准差|
|{% mathjax %} Co\mathrm{v} (X,Y){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}的协方差|
|{% mathjax %} \rho (X,Y){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}和{% mathjax %} Y{% endmathjax %}的相关性|
|{% mathjax %} H(X){% endmathjax %}|随机变量{% mathjax %} X{% endmathjax %}的熵|
|{% mathjax %} D_{KL}(P\parallel Q){% endmathjax %}|{% mathjax %} P{% endmathjax %}和{% mathjax %} Q{% endmathjax %}的`KL`散度|

##### 复杂度

|符号|描述|
|:--|:--|
|{% mathjax %} O{% endmathjax %}|复杂度标记|

#### 名词解释

##### 数据

毋庸置疑，如果没有数据，那么数据科学好无用武之地。每个数据集由一个个样本组成，大多时候，它们遵循独立和相同分布。样本有时也叫数据点(`data point`)或者叫做数据实例(`data instance`)，通常每个样本由一组称为特征(`feature`，或协变量(`covariates`))的属性组成。机器学习模型会根据这些属性进行预测。要预测的是一个特殊的属性，它被称为标签(`label`，或目标(`target`))。当处理图像数据时，每一张单独的照片即为一个样本，它的特征由每个像素数值的有序列表表示。 比如，{% mathjax %}200\times 200{% endmathjax %}彩色照片由{% mathjax %}200\times 200\times 3{% endmathjax %}个数值组成，其中的“`3`”对应于每个空间位置的红、绿、蓝通道的强度。 再比如，对于一组医疗数据，给定一组标准的特征（如年龄、生命体征和诊断），此数据可以用来尝试预测患者是否会存活。当每个样本的特征类别数量都相同时，其特征向量的长度是固定的，这个长度被称为数据的维数(`dimensionality`)。

然而并不是所有的数据都可以用“固定长度”的向量来表示，以图像数据为例，如果他们全部都来自标准显微镜设备，那么固定长度是可取的，但如果图像数据来自互联网，他们很难具有相同的分辨率和形状。这时将图像裁剪成标准尺寸是一种方法，但这种方法很局限，有丢失信息的风险，此外文本数据跟不符合固定长度的要求。比如，对于亚马逊等电子商务网站上的客户评论，有些文本数据很简短（比如“好极了”），有些则长篇大论。 与传统机器学习方法相比，深度学习的一个主要优势是可以处理不同长度的数据。一般来说，拥有越多数据的时候，工作就越容易。更多的数据可以被用来训练出更强大的模型，从而减少对预先设想假设的依赖。数据集的由小变大为现代深度学习的成功奠定基础。在没有大数据集的情况下许多令人兴奋的深度学习模型黯然失色。就算一些深度学习模型在小数据集上能够工作，但其效能并不比传统方法高。

请注意，仅仅拥有海量数据是不够的，我们还需要正确的数据。如果数据中充满了错误，或者如果数据的特征不能预测任务目标，那么模型很可能无效。有一句古话说得好：“输入的是垃圾，输出的也是垃圾。”，此外，糟糕的预测性能甚至会加倍放大事态的严重性。在一些敏感应用中，如预测性监管、简历筛选和用于信贷的分险模型，我们必须特别警惕垃圾数据带来的后果。一种常见的问题来自不均衡的数据集，比如在一个有关医疗的的数据集中，某些人群没有样本表示。想象一下，我们要训练一个皮肤癌识别模型，但它（在训练数据集中）从未见过黑色皮肤的人群，这个模型就会顿时束手无策。再比如，如果“用过去的招聘决策数据”来训练一个筛选简历的模型，那么机器学习模型就会捕捉到历史残留的不公正，并将其自动化。然而，这一切都可能在不知情的情况下发生。因此，当数据不具有充分代表性，甚至包含了一些社会偏见性时，模型就很有可能有偏见。

##### 模型

大多数机器学习会涉及到数据的转换。比如一个“摄取照片并预测笑脸”的系统，再比如通过摄取到的一组传感器读数预测读数的正常与异常程度。虽然简单的模型能够解决如上简单的问题，但本文中关注到的问题超出了经典方法的极限。深度学习与经典方法的区别主要在于：前者关注的是功能强大的模型，这些模型由神经网络错综复杂的交织在一起，包含层层数据转换，因此被称为深度学习（`deep learning`）。

##### 目标函数

前面的内容将机器学习介绍为“从经验中学习”。这里所说的“学习”，是指**自主提高模型完成某些任务的效能**。但是，什么才算真正的提高呢？在机器学习中，我们需要定义模型的优劣程度的度量，这个度量在大多数情况是“可优化”的，这被称之为**目标函数**（`objective function`）。我们通常定义一个目标函数，并希望优化它到最低点。因为越低越好，所以这些函数有时被称为**损失函数**（`loss function`，或`cost function`）。但这只是一个惯例，我们也可以取一个新的函数，优化到它的最高点。这两个函数本质上是相同的，只是翻转一下符号。当任务在视图预测数值时，最常见的损失函数是**平方误差**(`squared error`)，即预测值与实际值之差的平方。当试图解决分类问题时最常见的损失函数是“**最小化错误率**”，即预测与实际情况不符的样本比例。有些损失函数（平方误差）很容易被优化；有些损失函数（如错误率）由于不可微性或其他复杂性难以直接优化。在这些情况下，通常会优化替代目标。通常损失函数是根据模型参数定义的，并取决于数据集。在一个数据集上，我们可以通过最小化总损失来学习模型参数的最佳值，该数据集由一些为训练而收集的样本组成，称为训练数据集(`training dataset`)。然而，在训练数据上表现良好的模型，并不一定在新数据集上有同样的性能，这里的“新数据集”通常称为**测试数据集**(`test dataset`)。

综上所述，可用数据集通常分成两部分：**训练数据集用于拟合模型参数，测试数据集用于评估拟合的模型**。然后我们观察模型在这两部分数据集上的性能。“一个模型在训练数据集上的性能”可以被想象成“一个学生在模拟考试中的分数”。这个分数用来为一些真正的期末考试做参考，即使成绩令人鼓舞，也不能保证期末考试成功。换言之，测试性能可能会偏离训练性能。当一个模型在训练集上表现良好，但不能推广到测试集时，这个模型被称为过拟合(`overfitting`)的。就像在现实生活中，尽管模拟考试考得很好，真正的考试不一定百发百中。

##### 优化算法

当我们获得了一些数据及其表示、一个模型和一个合适的损失函数，接下来就需要一种算法，它能够搜索出最佳参数，以最小化损失函数。深度学习中，大多数流行的优化算法通常基于一个基本方法—**梯度下降**(`gradient descent`)。简而言之，在每个步骤中，梯度下降法都会检测每个参数，看看如果仅对该参数进行少量变动，训练集损失会朝着哪个方向移动。然后，它在可以减少损失的方向上优化参数。

##### 监督学习

##### 无监督学习

##### 与环境互动

##### 强化学习

##### 深度学习