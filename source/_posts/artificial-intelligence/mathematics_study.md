---
title: 数据科学 — 数学（机器学习）
date: 2024-08-06 15:35:11
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

#### 介绍

**数据科学**是一门跨学科的领域，结合了**统计学**、**计算机科学**和领域知识，以从数据中提取有价值的信息。**数学**在数据科学中起着至关重要的作用，以下是数据科学中一些关键的数学基础。
- **线性代数**：**矩阵和向量**—线性代数是数据科学的基础，特别是在机器学习和数据分析中。矩阵和向量用于表示和操作数据集；**矩阵分解**—如**特征值分解**和**奇异值分解**(`SVD`)，这些技术在**降维**和**数据压缩**中非常重要。
- **微积分**：**导数和积分**—**微积分**用于优化算法，尤其是**梯度下降法**，这是训练机器学习模型的核心技术；**偏导数和多变量微积分**—在复杂模型中，涉及多个变量的优化问题需要用到这些概念。
- **概率与统计**：**基本概率**—包括**概率分布**、**期望值**和**方差**，这些是理解随机过程和不确定性的重要工具；**统计推断**—如**假设检验**、**置信区间**和**贝叶斯统计**，用于从样本数据中推断总体特征。
- **最优化**：**线性规划和非线性规划**—用于解决**资源分配**和**决策**问题；**凸优化**—许多机器学习算法的基础，通过优化**目标函数**来找到最佳参数。
<!-- more -->

##### 数学—符号注释

|符号|解释|
|:---|:---|
|{% mathjax %}A,B,C{% endmathjax %}|大写字母代表矩阵(`Matrix`)|
|{% mathjax %}u,v,w{% endmathjax %}|小写字母代表矢量|
|{% mathjax %}A, m\times n{% endmathjax %}|矩阵(`Matrix`){% mathjax %}A{% endmathjax %}有{% mathjax %}m{% endmathjax %}行和{% mathjax %}n{% endmathjax %}列|
|{% mathjax %}A^{\mathsf{T}}{% endmathjax %}|矩阵(`Matrix`){% mathjax %}A{% endmathjax %}的转置|
|{% mathjax %}v^{\mathsf{T}}{% endmathjax %}|向量{% mathjax %}v{% endmathjax %}的转置|
|{% mathjax %}A^{-1}{% endmathjax %}|矩阵(`Matrix`){% mathjax %}A{% endmathjax %}的逆矩阵|
|{% mathjax %}\text{det}(A){% endmathjax %}|矩阵(`Matrix`){% mathjax %}A{% endmathjax %}的行列式|
|{% mathjax %}AB{% endmathjax %}|矩阵(`Matrix`){% mathjax %}A{% endmathjax %}和矩阵{% mathjax %}B{% endmathjax %}的矩阵乘法|
|{% mathjax %}u\cdot v{% endmathjax %}|向量{% mathjax %}u{% endmathjax %}和向量{% mathjax %}v{% endmathjax %}的点积|
|{% mathjax %}\mathbb{R}{% endmathjax %}|实数集，例如`0,−0.642,2,3.456`|
|{% mathjax %}\mathbb{R}^2{% endmathjax %}|二维向量集|
|{% mathjax %}\mathbb{R}^n{% endmathjax %}|`n`维向量集|
|{% mathjax %}v\in \mathbb{R}^2{% endmathjax %}|向量{% mathjax %}v{% endmathjax %}是{% mathjax %}\mathbb{R}^2{% endmathjax %}中的一个元素|
|{% mathjax %}|v|_1{% endmathjax %}|向量的`L1-norm`|
|{% mathjax %}|v|_2{% endmathjax %}|向量的`L2-norm`|
|{% mathjax %}T:\mathbb{R}^2 \rightarrow \mathbb{R}^3; T(v) = w{% endmathjax %}|将向量{% mathjax %}v\in \mathbb{R}^2{% endmathjax %}转化为向量{% mathjax %}w \in \mathbb{R}^3{% endmathjax %}|

#### 方程组

##### 线性代数

一种常用的系统建模机器学习方法称为：**“线性回归”**。**线性回归**是一种监督式的机器学习方法，假如你已经收集了许多输入、输出数据，你的目标是发现它们之间的关系。例如，你想预测风力涡轮机的电力输出。如果你只有一个特征，`X`轴上显示的是风速，`Y`轴上显示的是输出功率，这里的数据点代表风速和功率输出的实际测量值。显然，线性回归的目标是找到最接近这些数据点的线。对于这样的模型，假设这种关系是线性的，它可以用一条线建模。例如，风速以每秒`5`米的速度吹来，那么我预测风力涡轮机的功率输出降为`1500`千瓦。现在这个模型并不完美，你可以看到实际数据分散在模型的线条上，但它做的不错，这里的模型是线性方程：{% mathjax %}y = mx + b{% endmathjax %}，其中{% mathjax %}y{% endmathjax %}是功率输出，{% mathjax %}x{% endmathjax %}是风速。你的目标是找出适合数据的{% mathjax %}m{% endmathjax %}和{% mathjax %}b{% endmathjax %}的最佳值，在机器学习中你经常会看到该模型的方程会写成：{% mathjax %}y = wx + b{% endmathjax %}，因为数字{% mathjax %}w{% endmathjax %}乘以{% mathjax %}x{% endmathjax %}中的{% mathjax %}w{% endmathjax %}被称为权重，{% mathjax %}b{% endmathjax %}称为偏差，只有一个特征的线性回归很容易可视化。
{% asset_img m_1.png  %}

但在机器学习问题中，你需要考虑更多的特征。在预测涡轮机的功率输出时，包括风速、温度，为了包括新的输入，需要更改方程：{% mathjax %}y = w_1x_1 + w_2x_2 + b{% endmathjax %}，为第二个变量添加了新的权重，这个方程将不是一条直线，而是三维空间中的平面。但是如果你想考虑更多的特征，比如压力、湿度等，这时候该怎么办？这时你只需要为每个特征添加一个新的权重。
{% asset_img m_2.png  %}

以此类推，你有{% mathjax %}w_nx_n{% endmathjax %}任意数量的特征。然后你添加{% mathjax %}b{% endmathjax %}，并将其全部设置为{% mathjax %}y{% endmathjax %}，如果你将方程想象为数据集中的一行，那你已经知道了{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}的值，你的目标是找到{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}的值来适合这个方程，由于数据集有很多条数据记录，所以你可以写下更多个方程。写下第一个方程中的所有{% mathjax %}x_n{% endmathjax %}和{% mathjax %}y{% endmathjax %}的上面添加一个带括号的标号`1`，则依此类推，包含{% mathjax %}m{% endmathjax %}条记录的数据集，最后一个方程的标号为{% mathjax %}m{% endmathjax %}。
{% asset_img m_3.png  %}

同时求解所有这些方程的权重{% mathjax %}w_n{% endmathjax %}和偏差值{% mathjax %}b{% endmathjax %}，而这些方程被统称为：**“线性方程组”**。数据集包含了一系列特征，例如风速、温度、大气压力、湿度等。我们用{% mathjax %}x_1,x_2,x_3,x_4,\ldots,x_n{% endmathjax %}来表示。然后向数据集添加一个上标来表示这一组特征是属于哪一行数据。用模型权重乘可以表示为{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}，最后添加了一个偏差项{% mathjax %}b{% endmathjax %}。
{% asset_img m_4.png  %}

这里有一个权重向量{% mathjax %}w{% endmathjax %}，它由{% mathjax %}w_1,w_2,\ldots,w_n{% endmathjax %}组成。还有一个特征矩阵用{% mathjax %}X{% endmathjax %}表示。
{% asset_img m_5.png  %}

##### 句子系统

 **组合句子**为你提供信息的方式与**组合方程**为你提供信息的方式非常相似。换句话说，**句子系统**的行为很像方程组，让我们从一些句子系统的例子开始。假设你只有一只狗和一只猫，而且 它们都只有一种颜色。你会得到一些信息，你的目标是尝试弄清楚每种动物的颜色。因此，这是带有句子的系统`1`：“狗是黑色的，猫是橙色的”。带有句子的系统`2`：“狗是黑色的，狗是黑色的”。最后，带有句子的系统`3`：“狗是黑色的， 狗是白色的”。每个句子都有一个信息。因此，诸如狗是黑色的，狗是白色或不允许的句子，它们分别包含两条信息。系统的目标是用这些简单的句子尽可能多地传达信息。请注意这些系统有很大的不同。特别是，第一个句子系统包含两个句子和两条信息。这意味着该系统包含的信息与句子一样多，这就是所谓的**完整系统**。 第二个句子系统的信息量要少一些，因为它有两个句子，但它们完全一样。 因此，尽管该系统包含两个句子，但它只携带一条信息，这些句子 它是重复的，因此该系统被称为**冗余**。最后一个句子系统，句子相互矛盾的。这是因为狗不可能同时是黑白的，记住我们有一条狗，它只能有一种颜色。因此，该系统被称为**矛盾系统**。 系统携带的信息越多，对您就越有用。当系统冗余或相互矛盾时，它被称为**单一系统**。当一个系统完整时，它被称为非单一系统。简而言之，非单一系统是一个承载与句子一样多的信息的系统。因此，它是信息量最大的系统，而**单一系统**的信息量不如非单一系统。句子系统可以包含两个以上的句子。实际上，它们可以随心所欲地携带。以下是一些包含三个句子的系统的示例。 在这个新示例中，您有三只动物，并且再次尝试确定它们的颜色。第一个系统有句子：“狗是黑色的，胡萝卜是橙色的，鸟是红色的”。第二个系统句子：“狗是黑色的，狗是黑色的，鸟是红色的”。第三个系统句子：“说狗是黑色的，狗是黑色的，狗是黑色的”。“狗是黑色的，狗是白色的，鸟是红色的”。因此，第一个句子系统是完整的，因为它使用三个句子传递三条不同的信息，因此它是**完整且非单一**的。 请注意，第三个系统比第二个系统更冗余。是否可以衡量系统的冗余程度？答案是肯定的，它叫做**等级**。

 ##### 线性方程组

 例如，如何将方程{% mathjax %}a + b = 10{% endmathjax %}可视化为一条直线？首先，让我们绘制一个网格，其中横轴代表{% mathjax %}a{% endmathjax %}，即苹果的价格，纵轴代表{% mathjax %}b{% endmathjax %}，也就是香蕉的价格。现在让我们来看看这个方程{% mathjax %}a + b =10{% endmathjax %}的解。两个显而易见的解是点{% mathjax %}(10,0){% endmathjax %}，因此{% mathjax %}a{% endmathjax %}坐标，苹果的价格为`10`。 而且{% mathjax %}b{% endmathjax %}坐标，即香蕉的价格为`0`，因为{% mathjax %}10 + 0{% endmathjax %}等于 `10`。另一个解是{% mathjax %}a{% endmathjax %}为`0`且{% mathjax %}b{% endmathjax %}等于`10`的点{% mathjax %}(0,10){% endmathjax %}。其他解{% mathjax %}(4,6){% endmathjax %}，因为{% mathjax %}4+6 = 10{% endmathjax %}。这时{% mathjax %}a{% endmathjax %}等于`4`且{% mathjax %}b{% endmathjax %}等于`6`；{% mathjax %}(8,2){% endmathjax %}其中{% mathjax %}a =8{% endmathjax %}且{% mathjax %}b =2{% endmathjax %}。请注意，您也可以使用负解，例如{% mathjax %}(-4,14){% endmathjax %}。现在注意所有这些点形成一条线。实际上，这条线中的每一个点都是方程的解。因此，您可以将方程{% mathjax %}a + b =10{% endmathjax %}与该直线相关联。现在让我们再做一个方程式。假设方程{% mathjax %}a + 2b =12{% endmathjax %}，水平坐标加上两倍垂直坐标的点加起来等于`12`。因此，这个方程的解包含点{% mathjax %}(0,6){% endmathjax %}。再一次，这些点形成一条线，线上的每个点都是这个方程的解。因此，这条直线与方程{% mathjax %}a + 2b =12{% endmathjax %}相关联。
 {% asset_img m_6.png  %}

 请注意，直线方程{% mathjax %}a + b =10{% endmathjax %}的穿过点{% mathjax %}(10,0), (0,10){% endmathjax %}和{% mathjax %}2a + 2b = 24{% endmathjax %}的穿过点{% mathjax %}(12,0),(0,12){% endmathjax %}。这两个方程非常相似，他们只不过是偏移了`2`个单位。这两个方程组与坐标平面平行的两条直线相关联，平行线永远不会相遇，所以这两个方程没有解。

 ##### 奇异/非奇异矩阵

 线性代数中最重要和最基本的对象之一—“矩阵”。矩阵有许多非常重要的属性。由下图所示，如果你取{% mathjax %}a,b{% endmathjax %}的系数，把它们放在一个{% mathjax %}2\times 2{% endmathjax %}的方框内，这个方框被称为矩阵，这就是系统对应的矩阵。与第一个系统对应的矩阵是{% mathjax %}\begin{bmatrix} 1 & 1&\\ 1 & 2& \end{bmatrix}{% endmathjax %}。在矩阵中，每行对应每个方程，每列对应每个变量的系数，所以{% mathjax %}a{% endmathjax %}代表第一列，{% mathjax %}b{% endmathjax %}代表第二列。同理，第二个系统的矩阵也一样。矩阵就像线性方程组一样，也可以是奇异或非奇异的。第一个系统是非奇异的，因为他有一个独特的解；第二个系统是奇异的，因为它有无限多个解，所以它对应的矩阵是奇异矩阵。
