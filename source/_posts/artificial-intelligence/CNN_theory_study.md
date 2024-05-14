---
title: 卷积神经网络 (CNN) (TensorFlow)
date: 2024-05-14 11:00:11
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

卷积神经网络(`convolutional neural network，CNN`)是一类强大的、为处理图像数据而设计的神经网络。基于卷积神经网络架构的模型在计算机视觉领域中已经占主导地位，当今几乎所有的图像识别、目标检测或语义分割相关的学术竞赛和商业应用都以这种方法为基础。
<!-- more -->
现代卷积神经网络的设计得益于生物学、群论和一系列的补充实验。卷积神经网络需要的参数少于全连接架构的网络，而且卷积也很容易用`GPU`并行计算。因此卷积神经网络除了能够高效地采样从而获得精确的模型，还能够高效地计算。久而久之，从业人员越来越多地使用卷积神经网络。即使在通常使用循环神经网络的一维序列结构任务上（例如音频、文本和时间序列分析），卷积神经网络也越来越受欢迎。通过对卷积神经网络一些巧妙的调整，也使它们在图结构数据和推荐系统中发挥作用。
#### 从全连接层到卷积

##### 不变性

- 平移不变性(`translation invariance`)：不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应，即为“**平移不变性**”。
- 局部性(`locality`)：神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系，这就是“局部性”原则。最终，可以聚合这些局部特征，以在整个图像级别进行预测。

##### 多层感知机的限制

首先，多层感知机的输入是二维图像{% mathjax %}\mathbf{X}{% endmathjax %}，其隐藏表示{% mathjax %}\mathbf{H}{% endmathjax %}在数学上是一个矩阵，在代码中表示二维张量。其中{% mathjax %}X{% endmathjax %}和{% mathjax %}H{% endmathjax %}具有相同的形状，为了方便理解，我们可以认为，无论是输入还是隐藏表示都拥有空间结构。使用{% mathjax %}[\mathbf{X}_{i,j}]{% endmathjax %}和{% mathjax %}[\mathbf{X}_{i,j}]{% endmathjax %}分别表示输入图像和隐藏表示中位置{% mathjax %}(i,j){% endmathjax %}处的像素。为了使每个隐藏神经元都能接受到每个数像素的信息，我们将参数从权重矩阵（如同我们先前在多层感知机中所做的那样）替换为四阶权重张量{% mathjax %}\mathsf{W}{% endmathjax %}。假设{% mathjax %}\mathbf{U}{% endmathjax %}包含偏置参数，我们可以将全连接层形式化的表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
[\mathbf{H}]_{i,j} & = [\mathbf{U}]_{i,j} + \sum_k \sum_l [\mathsf{W}]_{i,j,k,l}[\mathbf{X}]_{k,l}
& = [\mathbf{U}]_{i,j} + \sum_a \sum_b [\mathsf{V}]_{i,j,a,b}[\mathbf{X}]_{i + a,j + b}
\end{align}
{% endmathjax %}
其中从{% mathjax %}\mathsf{W}{% endmathjax %}到{% mathjax %}\mathsf{V}{% endmathjax %}的转换只是形式上的转换，因为在这两个四节张量的元素之间存在一一对应的关系，我们只需从新索引下标{% mathjax %}(k,l){% endmathjax %}，使{% mathjax %} k = i + a{% endmathjax %}、{% mathjax %}l = j + b{% endmathjax %}，由此可得{% mathjax %}[\mathsf{V}]_{i,j,a,b} = [\mathsf{W}]_{i,j,i+a,j+b}{% endmathjax %}。索引{% mathjax %}a{% endmathjax %}和{% mathjax %}b{% endmathjax %}通过在正偏移和负偏移之间移动覆盖了整个图像。对于隐藏表示中任意给定位置{% mathjax %}(i,j){% endmathjax %}处的像素值{% mathjax %}[\mathbf{H}]_{i,j}{% endmathjax %}，可以通过在{% mathjax %}x{% endmathjax %}中以{% mathjax %}(i,j){% endmathjax %}为中心对像素进行加权求和得到，加权使用的权重为{% mathjax %}[\mathsf{V}]_{i,j,a,b}{% endmathjax %}。
###### 平移不变性

现在引用上述的第一个原则：平移不变性。这意味着检测对象的输入{% mathjax %}\mathsf{X}{% endmathjax %}中的平移，应该仅导致隐藏表示{% mathjax %}\mathbf{H}{% endmathjax %}中的平移。也就是说，{% mathjax %}\mathsf{V}{% endmathjax %}和{% mathjax %}\mathsf{U}{% endmathjax %}实际上不依赖于{% mathjax %}(i,j){% endmathjax %}的值，即{% mathjax %}[\mathsf{V}]_{i,j,a,b} = [\mathsf{V}]_{a,b}{% endmathjax %}。并且{% mathjax %}\mathsf{U}{% endmathjax %}是一个常数，比如{% mathjax %}u{% endmathjax %}。因此，我们可以简化{% mathjax %}\mathbf{H}{% endmathjax %}定义为：
{% mathjax '{"conversion":{"em":14}}' %}
[H]_{i,j} = u + \sum_a \sum_b [\mathbf{V}]_{a,b}[\mathbf{X}]_{i+a,j+b}
{% endmathjax %}
这就是**卷积**(`convolution`)，我们使用系数{% mathjax %}[\mathbf{V}]_{a,b}{% endmathjax %}对位置{% mathjax %}(i,j){% endmathjax %}附近的像素{% mathjax %}(i+a,j+b){% endmathjax %}进行加权得到{% mathjax %}[\mathbf{H}]_{i,j}{% endmathjax %}。注意{% mathjax %}[\mathbf{V}]_{a,b}{% endmathjax %}的系数比{% mathjax %}[\mathnf{V}]_{i,j,a,b}{% endmathjax %}少很多，因为前者不再依赖于图像中的位置。这就是显著的进步。
###### 局部性

现在引用上述的第二个原则：局部性。如上所述，为了收集用来训练参数{% mathjax %}[\mathbf{H}]_{i,j}{% endmathjax %}的相关信息。我们不应偏离到距{% mathjax %}(i,j){% endmathjax %}很远的地方。这意味着{% mathjax %}|a| > \Delta{% endmathjax %}或{% mathjax %}|b| > \Delta{% endmathjax %}的范围之外，我们可以设置{% mathjax %}[\mathbf{V}]_{a,b} = 0{% endmathjax %}。因此我们可以将{% mathjax %}[\mathbf{H}]_{i,j}{% endmathjax %}重写为：
{% mathjax '{"conversion":{"em":14}}' %}
[H]_{i,j} = u + \sum_{a=-\Delta}^{\Delta} \sum_{b=-\Delta}^{\Delta} [\mathbf{V}]_{a,b}[\mathbf{X}]_{i+a,j+b}
{% endmathjax %}
简而言之，这是一个**卷积层**(`convolutional layer`)，而卷积神经网络是包含卷积层的一类特殊的神经网络。在深度学习研究社区中，{% mathjax %}\mathbf{V}{% endmathjax %}被称为卷积核(`convolution kernel`)或者滤波器(`filter`)，亦或简单地称之为该**卷积层的权重**，通常该权重是可学习的参数。当图像处理的局部区域很小时，卷积神经网络与多层感知机的训练差异可能是巨大的：以前，多层感知机可能需要数十亿个参数来表示网络中的一层，而现在卷积神经网络通常只需要几百个参数，而且不需要改变输入或隐藏表示的维数。参数大幅减少的代价是，我们的特征现在是平移不变的，并且当确定每个隐藏活性值时，每一层只包含局部的信息。以上所有的权重学习都将依赖于归纳偏置。当这种偏置与现实相符时，我们就能得到样本有效的模型，并且这些模型能很好地泛化到未知数据中。但如果偏置与现实不符时，比如当图像不满足平移不变时，我们的模型可能难以拟合我们的训练数据。
##### 卷积

我们先简要回顾一下为什么上面的操作被称为卷积。在数学中，两个函数（比如{% mathjax %}f,g: \mathbb{R}^d \rightarrow \mathbb{R}{% endmathjax %}）之间的“卷积”被定义为：
{% mathjax '{"conversion":{"em":14}}' %}
(f\ast g)(\mathbf{x}) = \int \;f(\mathbf{z})g(\mathbf{x} - \mathbf{z})d_{\mathbf{z}}
{% endmathjax %}
也就是说，卷积是当把一个函数“翻转”并移位{% mathjax %}\mathbf{x}{% endmathjax %}时，测量{% mathjax %}f{% endmathjax %}和{% mathjax %}g{% endmathjax %}之间的重叠。当为离散对象时，积分就变成求和。例如，对于由索引为{% mathjax %}\mathbb{Z}{% endmathjax %}的、平方可和的、无限维向量集合中抽取的向量，我们得到以下定义：
{% mathjax '{"conversion":{"em":14}}' %}
(f\ast g)(i) = \sum_a f()g(i - a)
{% endmathjax %}
对于二维张量，则为{% mathjax %}f{% endmathjax %}的索引{% mathjax %}(a,b){% endmathjax %}和{% mathjax %}g{% endmathjax %}的索引{% mathjax %}(i-a,j-b){% endmathjax %}上的对应加和：
{% mathjax '{"conversion":{"em":14}}' %}
(f\ast g)(i,j) = \sum_a\sum_b f(a,b)g(i-a,j-b)
{% endmathjax %}
##### 总结

图像的平移不变性使我们以相同的方式处理局部图像，而不在乎它的位置。局部性意味着计算相应的隐藏表示只需一小部分局部图像像素。在图像处理中，卷积层通常比全连接层需要更少的参数，但依旧获得高效用的模型。卷积神经网络(`CNN`)是一类特殊的神经网络，它可以包含多个卷积层。多个输入和输出通道使模型在每个空间位置可以获取图像的多方面特征。

#### 图像卷积

我们解析了卷积层的原理，现在我们看看它的实际应用。
##### 互相关运算

严格来说，卷积层是个错误的叫法，因为它所表达的运算其实是**互相关运算**(`cross-correlation`)，而不是卷积运算。首先，我们暂时忽略通道（第三维）这一情况，看看如何处理二维图像数据和隐藏表示。输入是高度为`3`、宽度为`3`的二维张量。卷积核的高度和宽度都是`2`，而卷积核窗口（或卷积窗口）的形状由内核的高度和宽度决定。
{% asset_img cnn_1.png "二维互相关运算。阴影部分是第一个输出元素，以及用于计算输出的输入张量元素和核张量元素" %}

在二维互相关运算中，卷积窗口从输入张量的左上角开始，从左到右、从上到下滑动。当卷积窗口滑动到新一个位置时，包含在该窗口中的部分张量与卷积核张量进行按元素相乘，得到的张量再求和得到一个单一的标量值，由此我们得出了这一位置的输出张量值。在如上例子中，输出张量的四个元素由二维互相关运算得到，这个输出高度为`2`、宽度为`2`，如下所示：
{% asset_img cnn_2.png %}

