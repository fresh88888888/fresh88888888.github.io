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
这就是**卷积**(`convolution`)，我们使用系数{% mathjax %}[\mathbf{V}]_{a,b}{% endmathjax %}对位置{% mathjax %}(i,j){% endmathjax %}附近的像素{% mathjax %}(i+a,j+b){% endmathjax %}进行加权得到{% mathjax %}[\mathbf{H}]_{i,j}{% endmathjax %}。注意{% mathjax %}[\mathbf{V}]_{a,b}{% endmathjax %}的系数比{% mathjax %}[\mathbf{V}]_{i,j,a,b}{% endmathjax %}少很多，因为前者不再依赖于图像中的位置。这就是显著的进步。
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

注意，输出大小略小于输入大小。这是因为卷积核的宽度和高度大于1，而卷积核只与图像中每个大小完全适合的位置进行互相关运算。所以输出大小等于输入大小{% mathjax %}n_h\times n_w{% endmathjax %}减去卷积核大小{% mathjax %}k_h\times k_w{% endmathjax %}，即：
{% mathjax '{"conversion":{"em":14}}' %}
(n_h-k_h + 1)\times (n_w - k_w + 1)
{% endmathjax %}
这是因为我们需要足够的空间在图像上“移动”卷积核。我们在`corr2d`函数中实现如上过程，该函数接受输入张量`X`和卷积核张量`K`，并返回输出张量`Y`。
```python
import tensorflow as tf

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(X[i: i + h, j: j + w] * K))
    return Y

# 通过输入张量X和卷积核张量K，我们来验证上述二维互相关运算的输出。
X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = tf.constant([[0.0, 1.0], [2.0, 3.0]])
corr2d(X, K)

# <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=array([[19., 25.],[37., 43.]], dtype=float32)>
```
##### 卷积层

卷积层对输入和卷积核权重进行互相关运算，并在添加标量偏置之后产生输出。所以，卷积层中的两个被训练的参数是卷积核权重和标量偏置。就像我们之前随机初始化全连接层一样，在训练基于卷积层的模型时，我们也随机初始化卷积核权重。基于上面定义的`corr2d`函数实现二维卷积层。在`__init__`构造函数中，将`weight`和`bias`声明为两个模型参数。前向传播函数调用`corr2d`函数并添加偏置。
```python
class Conv2D(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, kernel_size):
        initializer = tf.random_normal_initializer()
        self.weight = self.add_weight(name='w', shape=kernel_size, initializer=initializer)
        self.bias = self.add_weight(name='b', shape=(1, ), initializer=initializer)

    def call(self, inputs):
        return corr2d(inputs, self.weight) + self.bias
```
高度和宽度分别为{% mathjax %}h{% endmathjax %}和{% mathjax %}w{% endmathjax %}的卷积核可以被称为{% mathjax %}h\times w{% endmathjax %}卷积或{% mathjax %}h\times w{% endmathjax %}卷积核。我们也将带有{% mathjax %}h\times w{% endmathjax %}卷积核的卷积层称为{% mathjax %}h\times w{% endmathjax %}卷积层。
#####  图像中目标的边缘检测

如下是卷积层的一个简单应用：通过找到像素变化的位置，来检测图像中不同颜色的边缘。首先，我们构造一个{% mathjax %}6\times 8{% endmathjax %}像素的黑白图像。中间四列为黑色(`0`)，其余像素为白色(`1`)。
```python
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X

# <tf.Variable 'Variable:0' shape=(6, 8) dtype=float32, numpy=
# array([[1., 1., 0., 0., 0., 0., 1., 1.],
#        [1., 1., 0., 0., 0., 0., 1., 1.],
#        [1., 1., 0., 0., 0., 0., 1., 1.],
#        [1., 1., 0., 0., 0., 0., 1., 1.],
#        [1., 1., 0., 0., 0., 0., 1., 1.],
#        [1., 1., 0., 0., 0., 0., 1., 1.]], dtype=float32)>

K = tf.constant([[1.0, -1.0]])

# 接下来，我们构造一个高度为1、宽度为2的卷积核K。当进行互相关运算时，如果水平相邻的两元素相同，则输出为零，否则输出为非零。

# 现在，我们对参数X（输入）和K（卷积核）执行互相关运算。
# 如下所示，输出Y中的1代表从白色到黑色的边缘，-1代表从黑色到白色的边缘，其他情况的输出为0。
Y = corr2d(X, K)
Y

# <tf.Variable 'Variable:0' shape=(6, 7) dtype=float32, numpy=
# array([[ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#        [ 0.,  1.,  0.,  0.,  0., -1.,  0.],
#        [ 0.,  1.,  0.,  0.,  0., -1.,  0.]], dtype=float32)>

# 现在我们将输入的二维图像转置，再进行如上的互相关运算。其输出如下，之前检测到的垂直边缘消失了。
# 不出所料，这个卷积核K只可以检测垂直边缘，无法检测水平边缘。
corr2d(tf.transpose(X), K)

# <tf.Variable 'Variable:0' shape=(8, 5) dtype=float32, numpy=
# array([[0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.],
#        [0., 0., 0., 0., 0.]], dtype=float32)>
```
##### 卷积核

如果我们只需寻找黑白边缘，那么以上`[1, -1]`的边缘检测器足以。然而，当有了更复杂数值的卷积核，或者连续的卷积层时，我们不可能手动设计滤波器。那么我们是否可以学习由`X`生成`Y`的卷积核呢？现在让我们看看是否可以通过仅查看“输入-输出”对来学习由`X`生成`Y`的卷积核。我们先构造一个卷积层，并将其卷积核初始化为随机张量。接下来，在每次迭代中，我们比较`Y`与卷积层输出的平方误差，然后计算梯度来更新卷积核。为了简单起见，我们在此使用内置的二维卷积层，并忽略偏置。
```python
# 构造一个二维卷积层，它具有1个输出通道和形状为（1，2）的卷积核
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# 这个二维卷积层使用四维输入和输出格式（批量大小、高度、宽度、通道），
# 其中批量大小和通道数都为1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))
lr = 3e-2  # 学习率

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # 迭代卷积核
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        if (i + 1) % 2 == 0:
            print(f'epoch {i+1}, loss {tf.reduce_sum(l):.3f}')

# epoch 2, loss 4.427
# epoch 4, loss 0.759
# epoch 6, loss 0.134
# epoch 8, loss 0.025
# epoch 10, loss 0.005

# 在10次迭代之后，误差已经降到足够低。现在我们来看看我们所学的卷积核的权重张量。
tf.reshape(conv2d.get_weights()[0], (1, 2))

# <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[ 0.99359465, -0.98468703]], dtype=float32)>
```
你会发现，我们学习到的卷积核权重非常接近我们之前定义的卷积核`K`。
##### 互相关和卷积

为了得到正式的卷积运算输出，我们需要执行之前定义的严格卷积运算，而不是互相关运算。幸运的是，它们差别不大，我们只需水平和垂直翻转二维卷积核张量，然后对输入张量执行互相关运算。
##### 总结

二维卷积层的核心计算是二维互相关运算。最简单的形式是，对二维输入数据和卷积核执行互相关操作，然后添加一个偏置。我们可以设计一个卷积核来检测图像的边缘。我们可以从数据中学习卷积核的参数。学习卷积核时，无论用严格卷积运算或互相关运算，卷积层的输出不会受太大影响。当需要检测输入特征中更广区域时，我们可以构建一个更深的卷积网络。

#### 填充和步幅

卷积的输出形状取决于输入形状和卷积核的形状。还有什么因素会影响输出的大小呢？假设以下情景：有时，在应用了连续的卷积之后，我们最终得到的输出远小于输入大小。这是由于卷积核的宽度和高度通常大于`1`所导致的。比如，一个{% mathjax %}240\times 240{% endmathjax %}像素的图像，经过{% mathjax %}10{% endmathjax %}层{% mathjax %}5\times 5{% endmathjax %}的卷积后，将减少到{% mathjax %}200\times 200{% endmathjax %}像素。如此一来，原始图像的边界丢失了许多有用信息。而填充是解决此问题最有效的方法；有时，我们可能希望大幅降低图像的宽度和高度。例如，如果我们发现原始的输入分辨率十分冗余。步幅则可以在这类情况下提供帮助。
##### 填充

如上所述，在应用多层卷积时，我们常常丢失边缘像素。由于我们通常使用小卷积核，因此对于任何单个卷积，我们可能只会丢失几个像素。但随着我们应用许多连续卷积层，累积丢失的像素数就多了。解决这个问题的简单方法即为填充(`padding`)：在输入图像的边界填充元素（通常填充元素是0）。例如，在下图中，我们将{% mathjax %}3\times 3{% endmathjax %}输入填充到{% mathjax %}5\times 5{% endmathjax %}，那么它的输出就增加为{% mathjax %}4\times 4{% endmathjax %}。阴影部分是第一个输出元素以及用于输出计算的输入和核张量元素：{% mathjax %}0\times 0 + 0\times 1 + 0\times 2 + 0\times 3 = 0{% endmathjax %}。
{% asset_img cnn_3.png "带填充的二维互相关" %}

通常，我们添加{% mathjax %}p_h{% endmathjax %}行填充（大约一半在顶部，一半在底部）和{% mathjax %}p_w{% endmathjax %}列填充（左侧大约一半，右侧一半），则输出形状将为
{% mathjax '{"conversion":{"em":14}}' %}
(n_h - k_h + p_h + 1)\times (n_w - k_w + p_w + 1)
{% endmathjax %}
这意味着输出的高度和宽度将分别增加{% mathjax %}p_h{% endmathjax %}和{% mathjax %}p_w{% endmathjax %}。在许多情况下，我们需要设置{% mathjax %}p_h = k_h -1{% endmathjax %}和{% mathjax %}p_w = k_w - 1{% endmathjax %}，是输入和输出具有相同的高度和宽度。这样可以在构建网络时更容易地预测每个图层的输出形状。假设{% mathjax %}k_h{% endmathjax %}是奇数，我们将在高度的两侧填充{% mathjax %}\lceil p_h/2\rceil{% endmathjax %}行，在底部填充{% mathjax %}\lfloor p_h/2\rfloor{% endmathjax %}。我们填充宽度的两侧。卷积神经网络中卷积核的高度和宽度通常为奇数，例如`1、3、5`或`7`。选择奇数的好处是，保持空间维度的同时，我们可以在顶部和底部填充相同数量的行，在左侧和右侧填充相同数量的列。此外，使用奇数的核大小和填充大小也提供了书写上的便利。对于任何二维张量`X`，当满足：`1`.卷积核的大小是奇数；`2`.所有边的填充行数和列数相同；`3`.输出与输入具有相同高度和宽度则可以得出：输出`Y[i, j]`是通过以输入`X[i, j]`为中心，与卷积核进行互相关计算得到的。

比如，在下面的例子中，我们创建一个高度和宽度为`3`的二维卷积层，并在所有侧边填充`1`个像素。给定高度和宽度为`8`的输入，则输出的高度和宽度也是`8`。
```python
import tensorflow as tf

# 为了方便起见，我们定义了一个计算卷积层的函数。此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    return tf.reshape(Y, Y.shape[1:3])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape

# TensorShape([8, 8])

# 当卷积核的高度和宽度不同时，我们可以填充不同的高度和宽度，使输出和输入具有相同的高度和宽度。
# 在如下示例中，我们使用高度为5，宽度为3的卷积核，高度和宽度两边的填充分别为2和1。
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
comp_conv2d(conv2d, X).shape

# TensorShape([8, 8])
```
##### 步幅

在计算互相关时，卷积窗口从输入张量的左上角开始，向下、向右滑动。在前面的例子中，我们默认每次滑动一个元素。但是，有时候为了高效计算或是缩减采样次数，卷积窗口可以跳过中间位置，每次滑动多个元素。我们将每次滑动元素的数量称为**步幅**(`stride`)。到目前为止，我们只使用过高度或宽度为{% mathjax %}1{% endmathjax %}的步幅，那么如何使用较大的步幅呢？下图是垂直步幅为{% mathjax %}3{% endmathjax %}，水平步幅为{% mathjax %}2{% endmathjax %}的二维互相关运算。着色部分是输出元素以及用于输出计算的输入和内核张量元素：{% mathjax %}0\times 0 + 0\times 1 + 1\times 2 + 2\times 3 = 8{% endmathjax %}、{% mathjax %}0\times 0 + 6\times 1 + 0\times 2 + 0\times 3 = 6{% endmathjax %}。
{% asset_img cnn_4.png "垂直步幅为3，水平步幅为2的二维互相关运算" %}
通常，当垂直步幅为{% mathjax %}s_h{% endmathjax %}、水平步幅为{% mathjax %}s_w{% endmathjax %}时，输出形状为：
{% mathjax '{"conversion":{"em":14}}' %}
\lfloor (n_h - k_h + p_h + s_h)/s_h\rfloor\times \lfloor (n_w - k_w + p_w + s_w)/s_w\rfloor
{% endmathjax %}
如果我们设置了{% mathjax %}p_h = k_h - 1{% endmathjax %}和{% mathjax %}p_w = k_w - 1{% endmathjax %}，则输出形状将简化为{% mathjax %}\lfloor (n_h + s_h - 1)/s_h\rfloor\times \lfloor (n_w + s_w - 1)/s_w\rfloor{% endmathjax %}。更进一步，如果输入的高度和宽度可以被垂直和水平步幅整除，则输出形状将为{% mathjax %}(n_h/s_h)\times (n_w/s_w){% endmathjax %}。下面，我们将高度和宽度的步幅设置为`2`，从而将输入的高度和宽度减半。
```python
conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
comp_conv2d(conv2d, X).shape

# TensorShape([4, 4])

conv2d = tf.keras.layers.Conv2D(1, kernel_size=(3,5), padding='valid', strides=(3, 4))
comp_conv2d(conv2d, X).shape

# TensorShape([2, 1])
```
为了简洁起见当输入高度和宽度两侧的填充数量分别为{% mathjax %}p_h{% endmathjax %}和{% mathjax %}p_w{% endmathjax %}时。当{% mathjax %}p_h = p_w = p{% endmathjax %}时，填充是{% mathjax %}p{% endmathjax %}。同理，当高度和宽度的步幅分别为{% mathjax %}s_h{% endmathjax %}和{% mathjax %}s_w{% endmathjax %}时，我们称之为步幅{% mathjax %}(s_h,s_w){% endmathjax %}，{% mathjax %}s_h = s_w = s{% endmathjax %}时，我们称步幅为{% mathjax %}s{% endmathjax %}。默认情况下填充为0，步幅为1.在实践中，我们很少使用不一致的步幅或填充，也就是说，我们通常有{% mathjax %}p_h = p_w{% endmathjax %}和{% mathjax %}s_h = s_w{% endmathjax %}。
##### 总结

填充可以增加输出的高度和宽度。这常用来使输出与输入具有相同的高和宽。步幅可以减小输出的高和宽，例如输出的高和宽仅为输入的高和宽的{% mathjax %}1/n{% endmathjax %}（{% mathjax %}n{% endmathjax %}是一个大于的整数）。填充和步幅可用于有效地调整数据的维度。