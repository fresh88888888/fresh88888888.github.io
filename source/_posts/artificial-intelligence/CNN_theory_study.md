---
title: 卷积神经网络 (CNN)(TensorFlow)
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
[\mathbf{H}]_{i,j} & = [\mathbf{U}]_{i,j} + \sum_k \sum_l [\mathsf{W}]_{i,j,k,l}[\mathbf{X}]_{k,l}\\
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

#### 多输入多输出通道

##### 多输入通道

当输入包含多个通道时，需要构造一个与输入数据具有相同输入通道数的卷积核，以便与输入数据进行互相关运算。假设输入的通道数为{% mathjax %}c_i{% endmathjax %}，那么卷积核的输入通道数也需要为{% mathjax %}c_i{% endmathjax %}。如果卷积核的窗口形状是{% mathjax %}k_h\times k_w{% endmathjax %}，那么当{% mathjax %}c_i = 1{% endmathjax %}时，我们可以把卷积核看作形状为{% mathjax %}k_h\times k_w{% endmathjax %}的二维张量。然而，当{% mathjax %}c_i > 1{% endmathjax %}时，我们卷积核的每个输入通道将包含形状为{% mathjax %}k_h\times k_w{% endmathjax %}的张量。将这些张量{% mathjax %}c_i{% endmathjax %}连结在一起可以得到形状为{% mathjax %}c_i\times k_h\times k_w{% endmathjax %}的卷积核。由于输入和卷积核都有{% mathjax %}c_i{% endmathjax %}个通道，我们可以对每个通道输入的二维张量和卷积核的二维张量进行互相关运算，再对通道求和（将{% mathjax %}c_i{% endmathjax %}的结果相加）得到二维张量。这是多通道输入和多输入通道卷积核之间进行二维互相关运算的结果。在下图中，我们演示了一个具有两个输入通道的二维互相关运算的示例。阴影部分是第一个输出元素以及用于计算这个输出的输入和核张量元素：{% mathjax %}(1\times 1+ 2\times 2 + 4\times 3 + 5\times 4) + (0\times 0 + 1\times 1 + 3\times 2 + 4\times 3) = 56{% endmathjax %}。
{% asset_img cnn_5.png "两个输入通道的互相关计算" %}

为了加深理解，我们实现一下多输入通道互相关运算。简而言之，我们所做的就是对每个通道执行互相关操作，然后将结果相加。
```python
import tensorflow as tf

def corr2d_multi_in(X, K):
    # 先遍历“X”和“K”的第0个维度（通道维度），再把它们加在一起
    return tf.reduce_sum([d2l.corr2d(x, k) for x, k in zip(X, K)], axis=0)

# 我们可以构造与相对应的输入张量X和核张量K，以验证互相关运算的输出。
X = tf.constant([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = tf.constant([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
corr2d_multi_in(X, K)

# <tf.Tensor: shape=(2, 2), dtype=float32, numpy=array([[ 56.,  72.],[104., 120.]], dtype=float32)>
```
##### 多输出通道

到目前为止，不论有多少输入通道，我们还只有一个输出通道。在最流行的神经网络架构中，随着神经网络层数的加深，我们常会增加输出通道的维数，通过减少空间分辨率以获得更大的通道深度。直观地说，我们可以将每个通道看作对不同特征的响应。而现实可能更为复杂一些，因为每个通道不是独立学习的，而是为了共同使用而优化的。因此，多输出通道并不仅是学习多个单通道的检测器。用{% mathjax %}c_i{% endmathjax %}和{% mathjax %}c_o{% endmathjax %}分别表示输入和输出通道的数目，并让{% mathjax %}k_h{% endmathjax %}和{% mathjax %}k_w{% endmathjax %}为卷积核的高度和宽度。为了获得多个通道的输出，我们可以为每个输出通道创建一个形状为{% mathjax %}c_i\times k_h\times k_w{% endmathjax %}的卷积核张量，这样卷积核的形状是{% mathjax %}c_o\times c_i\times k_h\times k_w{% endmathjax %}。在互相关运算中，每个输出通道先获取所有输入通道，再以对应该输出通道的卷积核计算出结果。如下所示，我们实现一个计算多个通道的输出的互相关函数。
```python
def corr2d_multi_in_out(X, K):
    # 迭代“K”的第0个维度，每次都对输入“X”执行互相关运算。
    # 最后将所有结果都叠加在一起
    return tf.stack([corr2d_multi_in(X, k) for k in K], 0)

# 通过将核张量K与K+1（K中每个元素加1）和K+2连接起来，构造了一个具有2个输出通道的卷积核。
K = tf.stack((K, K + 1, K + 2), 0)
K.shape

# TensorShape([3, 2, 2, 2])

# 下面，我们对输入张量X与卷积核张量K执行互相关运算。现在的输出包含3个通道，第一个通道的结果与先前输入张量X和多输入单输出通道的结果一致。
corr2d_multi_in_out(X, K)

# <tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
# array([[[ 56.,  72.],[104., 120.]],
#        [[ 76., 100.],[148., 172.]],
#        [[ 96., 128.],[192., 224.]]], dtype=float32)>
```
#####  {% mathjax %}1\times 1{% endmathjax %}卷积层

{% mathjax %}1\times 1{% endmathjax %}卷积，即{% mathjax %}k_h = k_w = 1{% endmathjax %}，看起来似乎没有多大意义。毕竟，卷积的本质是有效提取相邻像素间的相关特征而，{% mathjax %}1\times 1{% endmathjax %}卷积显然没有此作用，尽管如此，{% mathjax %}1\times 1{% endmathjax %}仍然流行，经常包含在复杂深层网络的设计中。因为使用了最小窗口，{% mathjax %}1\times 1{% endmathjax %}卷积失去了卷积层的特有能力—在高度和宽度维度上，识别相邻元素间相互作用的能力。其实{% mathjax %}1\times 1{% endmathjax %}卷积的唯一计算发生在通道上。

下图展示了使用{% mathjax %}1\times 1{% endmathjax %}卷积核与`3`个输入通道和`2`个输出通道的互相关计算。这里输入和输出具有相同的高度和宽度，输出中的每个元素都是从输入图像中同一位置的元素的线性组合。我们可以将{% mathjax %}1\times 1{% endmathjax %}卷积层看作在每个像素位置应用的全连接层，以{% mathjax %}c_i{% endmathjax %}个输入值转换为{% mathjax %}c_o{% endmathjax %}个输出值。 因为这仍然是一个卷积层，所以跨像素的权重是一致的。同时，{% mathjax %}1\times 1{% endmathjax %}卷积层需要的权重维度为{% mathjax %}c_o\times c_i{% endmathjax %}，再额外加上一个偏置。
{% asset_img cnn_6.png "互相关计算使用了具有3个输入通道和2个输出通道的1 x 1卷积核。其中，输入和输出具有相同的高度和宽度" %}

下面，我们使用全连接层实现{% mathjax %}1\times 1{% endmathjax %}卷积。请注意，我们需要对输入和输出的数据形状进行调整。
```python
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X, (c_i, h * w))
    K = tf.reshape(K, (c_o, c_i))
    # 全连接层中的矩阵乘法
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (c_o, h, w))

# 当执行卷积运算时，上述函数相当于先前实现的互相关函数corr2d_multi_in_out。让我们用一些样本数据来验证这一点。
X = tf.random.normal((3, 3, 3), 0, 1)
K = tf.random.normal((2, 3, 1, 1), 0, 1)

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(tf.reduce_sum(tf.abs(Y1 - Y2))) < 1e-6
```
##### 总结

多输入多输出通道可以用来扩展卷积层的模型。当以每像素为基础应用时，{% mathjax %}1\times 1{% endmathjax %}卷积层相当于全连接层。{% mathjax %}1\times 1{% endmathjax %}卷积层通常用于调整网络层的通道数量和控制模型复杂性。

#### 汇聚层

汇聚层(`pooling`)，它具有双重目的：降低卷积层对位置的敏感性，同时降低对空间降采样表示的敏感性。
##### 最大汇聚层和平均汇聚层

与卷积层类似，汇聚层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输入的所有区域上滑动，为固定形状窗口（有时称为汇聚窗口）遍历的每个位置计算一个输出。然而，不同于卷积层中的输入与卷积核之间的互相关计算，汇聚层不包含参数。相反，池运算是确定性的，我们通常计算汇聚窗口中所有元素的最大值或平均值。这些操作分别称为最大汇聚层(`maximum pooling`)和平均汇聚层(`average pooling`)。在这两种情况下，与互相关运算符一样，汇聚窗口从输入张量的左上角开始，从左往右、从上往下的在输入张量内滑动。在汇聚窗口到达的每个位置，它计算该窗口中输入子张量的最大值或平均值。计算最大值或平均值是取决于使用了最大汇聚层还是平均汇聚层。
{% asset_img cnn_7.png "汇聚窗口形状为2 x 2的最大汇聚层。着色部分是第一个输出元素，以及用于计算这个输出的输入元素:max(0,1,3,4) = 4" %}

上图中中的输出张量的高度为{% mathjax %}p\times q{% endmathjax %}，宽度为{% mathjax %}p\times q{% endmathjax %}。这四个元素为每个汇聚窗口中的最大值：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \text{max}(0,1,3,4) = 4\\
& \text{max}(1,2,4,5) = 5\\
& \text{max}(3,4,6,7) = 7\\
& \text{max}(4,5,7,8) = 8
\end{align}
{% endmathjax %}
汇聚窗口形状为{% mathjax %}p\times q{% endmathjax %}的汇聚层称为{% mathjax %}p\times q{% endmathjax %}汇聚层，汇聚层操作称为{% mathjax %}p\times q{% endmathjax %}操作。回到本节开头提到的对象边缘检测示例，现在我们将使用卷积层的输出作为{% mathjax %}2\times 2{% endmathjax %}最大汇聚的输入。设置卷积层输入为`X`，汇聚层输出为`Y`。无论`X[i, j]`和`X[i, j + 1]`的值相同与否，或`X[i, j + 1]`和`X[i, j + 2]`的值相同与否，汇聚层始终输出`Y[i, j] = 1`。也就是说，使用{% mathjax %}2\times 2{% endmathjax %}最大汇聚层，即使在高度或宽度上移动一个元素，卷积层仍然可以识别到模式。
```python
import tensorflow as tf

def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w +1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode =='avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y

# 我们可以构建上图中的输入张量X，验证二维最大汇聚层的输出。
X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
pool2d(X, (2, 2))

# <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=array([[4., 5.], [7., 8.]], dtype=float32)>

# 此外，我们还可以验证平均汇聚层
pool2d(X, (2, 2), 'avg')

# <tf.Variable 'Variable:0' shape=(2, 2) dtype=float32, numpy=array([[2., 3.],[5., 6.]], dtype=float32)>
```
##### 填充和步幅

与卷积层一样，汇聚层也可以改变输出形状。和以前一样，我们可以通过填充和步幅以获得所需的输出形状。下面，我们用深度学习框架中内置的二维最大汇聚层，来演示汇聚层中填充和步幅的使用。我们首先构造了一个输入张量`X`，它有四个维度，其中样本数和通道数都是`1`。请注意，`Tensorflow`采用“通道最后”（`channels-last`）语法，对其进行优化，（即`Tensorflow`中输入的最后维度是通道）。
```python
X = tf.reshape(tf.range(16, dtype=tf.float32), (1, 4, 4, 1))
X

# <tf.Tensor: shape=(1, 4, 4, 1), dtype=float32, numpy=
# array([[[[ 0.],[ 1.],[ 2.],[ 3.]],
#         [[ 4.],[ 5.],[ 6.],[ 7.]],
#         [[ 8.],[ 9.],[10.],[11.]],
#         [[12.],[13.],[14.],[15.]]]], dtype=float32)>
```
默认情况下，深度学习框架中的步幅与汇聚窗口的大小相同。因此，如果我们使用形状为`(3, 3)`的汇聚窗口，那么默认情况下，我们得到的步幅形状为`(3, 3)`。
```python
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
pool2d(X)

# <tf.Tensor: shape=(1, 1, 1, 1), dtype=float32, numpy=array([[[[10.]]]], dtype=float32)>

# 填充和步幅可以手动设定。
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid', strides=2)
pool2d(X_padded)

# <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=array([[[[ 5.],[ 7.]],[[13.],[15.]]]], dtype=float32)>

# 当然，我们可以设定一个任意大小的矩形汇聚窗口，并分别设定填充和步幅的高度和宽度。
paddings = tf.constant([[0, 0], [0, 0], [1, 1], [0, 0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[2, 3], padding='valid',strides=(2, 3))
pool2d(X_padded)

# <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=array([[[[ 5.],[ 7.]],[[13.],[15.]]]], dtype=float32)>
```
##### 多个通道

在处理多通道输入数据时，汇聚层在每个输入通道上单独运算，而不是像卷积层一样在通道上对输入进行汇总。这意味着汇聚层的输出通道数与输入通道数相同。下面，我们将在通道维度上连结张量`1`和`X+1`，以构建具有`2`个通道的输入。
```python
X = tf.concat([X, X + 1], 3)

# 汇聚后输出通道的数量仍然是2。
paddings = tf.constant([[0, 0], [1,0], [1,0], [0,0]])
X_padded = tf.pad(X, paddings, "CONSTANT")
pool2d = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='valid',strides=2)
pool2d(X_padded)

# <tf.Tensor: shape=(1, 2, 2, 2), dtype=float32, numpy=
# array([[[[ 5.,  6.],[ 7.,  8.]],
#         [[13., 14.],[15., 16.]]]], dtype=float32)>
```
##### 总结

对于给定输入元素，最大汇聚层会输出该窗口内的最大值，平均汇聚层会输出该窗口内的平均值。汇聚层的主要优点之一是减轻卷积层对位置的过度敏感。我们可以指定汇聚层的填充和步幅。使用最大汇聚层以及大于`1`的步幅，可减少空间维度（如高度和宽度）。汇聚层的输出通道数与输入通道数相同。

#### 卷积神经网络（LeNet）

卷积神经网络(`LeNet`)，它是最早发布的卷积神经网络之一，因其在计算机视觉任务中的高效性能而受到广泛关注。这个模型是由`AT&T`贝尔实验室的研究员`Yann LeCun`在`1989`年提出的（并以其命名），目的是识别图像中的手写数字。当时，`Yann LeCun`发表了第一篇通过反向传播成功训练卷积神经网络的研究，这项工作代表了十多年来神经网络研究开发的成果。当时，`LeNet`取得了与支持向量机(`support vector machines`)性能相媲美的成果，成为监督学习的主流方法。`LeNet`被广泛用于自动取款机(`ATM`)机中，帮助识别处理支票的数字。时至今日，一些自动取款机仍在运行`Yann LeCun`和他的同事`Leon Bottou`在上世纪`90`年代写的代码呢。

##### LeNet

总体来看，`LeNet（LeNet-5）`由两个部分组成：
- 卷积编码器：由两个卷积层组成;
- 全连接层密集块：由三个全连接层组成。

该架构如下图所示：
{% asset_img cnn_8.png " LeNet中的数据流。输入是手写数字，输出为10种可能结果的概率" %}

每个卷积块中的基本单元是一个卷积层、一个`sigmoid`激活函数和平均汇聚层。请注意，虽然`ReLU`和最大汇聚层更有效，但它们在`20`世纪`90`年代还没有出现。每个卷积层使用{% mathjax %}5\times 5{% endmathjax %}卷积核和一个sigmoid激活函数。这些层将输入映射到多个二维特征输出，通常同时增加通道的数量。第一卷积层有`6`个输出通道，而第二个卷积层有`16`个输出通道。每个{% mathjax %}2\times 2{% endmathjax %}池操作（步幅`2`）通过空间下采样将维数减少`4`倍。卷积的输出形状由批量大小、通道数、高度、宽度决定。为了将卷积块的输出传递给稠密块，我们必须在小批量中展平每个样本。换言之，我们将这个四维输入转换成全连接层所期望的二维输入。这里的二维表示的第一个维度索引小批量中的样本，第二个维度给出每个样本的平面向量表示。LeNet的稠密块有三个全连接层，分别有`120、84`和`10`个输出。因为我们在执行分类任务，所以输出层的10维对应于最后输出结果的数量。通过下面的`LeNet`代码，可以看出用深度学习框架实现此类模型非常简单。我们只需要实例化一个`Sequential`块并将需要的层连接在一起。
```python
import tensorflow as tf

def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])
```
我们对原始模型做了一点小改动，去掉了最后一层的高斯激活。除此之外，这个网络与最初的`LeNet-5`一致。下面，我们将一个大小为{% mathjax %}28\times 28{% endmathjax %}的单通道（黑白）图像通过`LeNet`。通过在每一层打印输出的形状，我们可以检查模型，以确保其操作与我们期望的下图一致。
{% asset_img cnn_9.png " LeNet 的简化版" %}
```python
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)

# Conv2D output shape:         (1, 28, 28, 6)
# AveragePooling2D output shape:       (1, 14, 14, 6)
# Conv2D output shape:         (1, 10, 10, 16)
# AveragePooling2D output shape:       (1, 5, 5, 16)
# Flatten output shape:        (1, 400)
# Dense output shape:          (1, 120)
# Dense output shape:          (1, 84)
# Dense output shape:          (1, 10)
```
请注意，在整个卷积块中，与上一层相比，每一层特征的高度和宽度都减小了。 第一个卷积层使用2个像素的填充，来补偿{% mathjax %}5\times 5{% endmathjax %}卷积核导致的特征减少。相反，第二个卷积层没有填充，因此高度和宽度都减少了`4`个像素。随着层叠的上升，通道的数量从输入时的`1`个，增加到第一个卷积层之后的`6`个，再到第二个卷积层之后的`16`个。同时，每个汇聚层的高度和宽度都减半。最后，每个全连接层减少维数，最终输出一个维数与结果分类数相匹配的输出。
##### 模型训练

现在我们已经实现了`LeNet`，我们训练和评估`LeNet-5`模型。
```python
lr, num_epochs = 0.9, 10
train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```
{% asset_img cnn_10.png %}

##### 总结

卷积神经网络(`CNN`)是一类使用卷积层的网络。在卷积神经网络中，我们组合使用卷积层、非线性激活函数和汇聚层。为了构造高性能的卷积神经网络，我们通常对卷积层进行排列，逐渐降低其表示的空间分辨率，同时增加通道数。在传统的卷积神经网络中，卷积块编码得到的表征在输出之前需由一个或多个全连接层进行处理。`LeNet`是最早发布的卷积神经网络之一。

