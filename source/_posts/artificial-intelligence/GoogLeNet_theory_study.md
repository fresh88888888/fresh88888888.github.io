---
title: 含并行连结的网络 (GoogLeNet)(TensorFlow)
date: 2024-05-16 11:38:11
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

在`2014`年的`ImageNet`图像识别挑战赛中，一个名叫`GoogLeNet`的网络架构大放异彩。`GoogLeNet`吸收了`NiN`中串联网络的思想，并在此基础上做了改进。这篇论文的一个重点是解决了什么样大小的卷积核最合适的问题。毕竟，以前流行的网络使用小到{% mathjax %}1\times 1{% endmathjax %}，大到{% mathjax %}11\times 11{% endmathjax %}的卷积核。
<!-- more -->
#### Inception块

在`GoogLeNet`中，基本的卷积块被称为`Inception`块(`Inception block`)。这很可能得名于电影《盗梦空间》(`Inception`)，因为电影中的一句话“我们需要走得更深”(`We need to go deeper`)。
{% asset_img gln_1.png "Inception块的架构" %}

如上图所示，`Inception`块由四条并行路径组成。前三条路径使用窗口大小为{% mathjax %}1\times 1{% endmathjax %}、{% mathjax %}3\times 3{% endmathjax %}和{% mathjax %}5\times 5{% endmathjax %}的卷积层，从不同空间大小中提取信息。中间的两条路径在输入上执行{% mathjax %}1\times 1{% endmathjax %}卷积，以减少通道数，从而降低模型的复杂性。第四条路径使用{% mathjax %}3\times 3{% endmathjax %}最大汇聚层，然后使用{% mathjax %}1\times 1{% endmathjax %}卷积层来改变通道数。这四条路径都使用合适的填充来使输入与输出的高和宽一致，最后我们将每条线路的输出在通道维度上连结，并构成`Inception`块的输出。在`Inception`块中，通常调整的超参数是每层输出通道数。那么为什么`GoogLeNet`这个网络如此有效呢？首先我们考虑一下滤波器(`filter`)的组合，它们可以用各种滤波器尺寸探索图像，这意味着不同大小的滤波器可以有效地识别不同范围的图像细节。同时，我们可以为不同的滤波器分配不同数量的参数。
#### GoogLeNet模型

如下图所示，`GoogLeNet`一共使用`9`个`Inception`块和全局平均汇聚层的堆叠来生成其估计值。`Inception`块之间的最大汇聚层可降低维度。第一个模块类似于`AlexNet`和`LeNet`，`Inception`块的组合从`VGG`继承，全局平均汇聚层避免了在最后使用全连接层。
{% asset_img gln_2.png "GoogLeNet架构" %}

现在，我们逐一实现`GoogLeNet`的每个模块。第一个模块使用`64`个通道、{% mathjax %}7\times 7{% endmathjax %}卷积层。
```python
import tensorflow as tf

class Inception(tf.keras.Model):
    # c1--c4是每条路径的输出通道数
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 线路1，单1x1卷积层
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')
        # 线路2，1x1卷积层后接3x3卷积层
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same',activation='relu')
        # 线路3，1x1卷积层后接5x5卷积层
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same',activation='relu')
        # 线路4，3x3最大汇聚层后接1x1卷积层
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度上连结输出
        return tf.keras.layers.Concatenate()([p1, p2, p3, p4])

def b1():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same',activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 第二个模块使用两个卷积层：第一个卷积层是64个通道、1 x 1卷积层；第二个卷积层使用将通道数量增加三倍的3 x 3卷积层。 这对应于Inception块中的第二条路径。
def b2():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

def b3():
    return tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 第四模块更加复杂， 它串联了5个Inception块，其输出通道数分别是：512、512、512、528、832。
# 这些路径的通道数分配和第三模块中的类似，首先是含3 x 3卷积层的第二条路径输出最多通道，其次是仅含1 x 1。
# 卷积层的第一条路径，之后是含5 x 5，卷积层的第三条路径和含3 x 3。最大汇聚层的第四条路径。
# 其中第二、第三条路径都会先按比例减小通道数。这些比例在各个Inception块中都略有不同。
def b4():
    return tf.keras.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 第五模块包含输出通道数为832和1024的两个Inception块。
# 其中每条路径通道数的分配思路和第三、第四模块中的一致，只是在具体数值上有所不同。 
# 需要注意的是，第五模块的后面紧跟输出层，该模块同NiN一样使用全局平均汇聚层，将每个通道的高和宽变成1。 
# 最后我们将输出变成二维数组，再接上一个输出个数为标签类别数的全连接层。
def b5():
    return tf.keras.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Flatten()
    ])

def net():
    return tf.keras.Sequential([b1(), b2(), b3(), b4(), b5(),tf.keras.layers.Dense(10)])

# GoogLeNet模型的计算复杂，而且不如VGG那样便于修改通道数。
# 为了使Fashion-MNIST上的训练短小精悍，我们将输入的高和宽从224降到96，这简化了计算。
X = tf.random.uniform(shape=(1, 96, 96, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape:\t', X.shape)
```
结果输出为：
```bash
Sequential output shape:     (1, 24, 24, 64)
Sequential output shape:     (1, 12, 12, 192)
Sequential output shape:     (1, 6, 6, 480)
Sequential output shape:     (1, 3, 3, 832)
Sequential output shape:     (1, 1024)
Dense output shape:  (1, 10)
```
#### 训练模型

和以前一样，我们使用`Fashion-MNIST`数据集来训练我们的模型。在训练之前，我们将图片转换为{% mathjax %} 96\times 96{% endmathjax %}分辨率。
{% asset_img gln_3.png %}

#### 总结

`Inception`块相当于一个有`4`条路径的子网络。它通过不同窗口形状的卷积层和最大汇聚层来并行抽取信息，并使用{% mathjax %}1\times 1{% endmathjax %}卷积层减少每像素级别上的通道维数从而降低模型复杂度。`GoogLeNet`将多个设计精细的`Inception`块与其他层（卷积层、全连接层）串联起来。其中`Inception`块的通道数分配之比是在`ImageNet`数据集上通过大量的实验得来的`GoogLeNet`和它的后继者们一度是`ImageNet`上最有效的模型之一：它以较低的计算复杂度提供了类似的测试精度。