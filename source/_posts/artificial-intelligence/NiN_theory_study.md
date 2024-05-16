---
title: 网络中的网络 (NiN)(TensorFlow)
date: 2024-05-16 11:20:11
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

`LeNet、AlexNet`和`VGG`都有一个共同的设计模式：通过一系列的卷积层与汇聚层来提取空间结构特征；然后通过全连接层对特征的表征进行处理。`AlexNet`和`VGG`对`LeNet`的改进主要在于如何扩大和加深这两个模块。或者，可以想象在这个过程的早期使用全连接层。然而，如果使用了全连接层，可能会完全放弃表征的空间结构。**网络中的网络**(`NiN`)提供了一个非常简单的解决方案：在每个像素的通道上分别使用多层感知机。
<!-- more -->
#### NiN块

回想一下，卷积层的输入和输出由四维张量组成，张量的每个轴分别对应样本、通道、高度和宽度。另外，全连接层的输入和输出通常是分别对应于样本和特征的二维张量。`NiN`的想法是在每个像素位置（针对每个高度和宽度）应用一个全连接层。如果我们将权重连接到每个空间位置，我们可以将其视为{% mathjax %}1\times 1{% endmathjax %}卷积层，或作为在每个像素位置上独立作用的全连接层。从另一个角度看，即将空间维度中的每个像素视为单个样本，将通道维度视为不同特征(`feature`)。

下图说明了`VGG`和`NiN`及它们的块之间主要架构差异。`NiN`块以一个普通卷积层开始，后面是两个{% mathjax %}1\times 1{% endmathjax %}的卷积层。这两个{% mathjax %}1\times 1{% endmathjax %}卷积层充当带有`ReLU`激活函数的逐像素全连接层。第一层的卷积窗口形状通常由用户设置。随后的卷积窗口形状固定为{% mathjax %}1\times 1{% endmathjax %}。
{% asset_img nin_1.png "对比 VGG 和 NiN 及它们的块之间主要架构差异" %}

#### NiN模型

最初的`NiN`网络是在`AlexNet`后不久提出的，显然从中得到了一些启示。`NiN`使用窗口形状为{% mathjax %}11\times 11{% endmathjax %}、{% mathjax %}5\times 5{% endmathjax %}和{% mathjax %}3\times 3{% endmathjax %}的卷积层，输出通道数量与`AlexNet`中的相同。每个`NiN`块后有一个最大汇聚层，汇聚窗口形状为{% mathjax %} {% endmathjax %}，步幅为`2`。`NiN`和`AlexNet`之间的一个显著区别是`NiN`完全取消了全连接层。相反，`NiN`使用一个`NiN`块，其输出通道数等于标签类别的数量。最后放一个全局平均汇聚层(`global average pooling layer`)，生成一个对数几率(`logits`)。`NiN`设计的一个优点是，它显著减少了模型所需参数的数量。然而，在实践中，这种设计有时会增加训练模型的时间。
```python
import tensorflow as tf

def nin_block(num_channels, kernel_size, strides, padding):
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(num_channels, kernel_size, strides=strides, padding=padding, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=1, activation='relu')])

def net():
    return tf.keras.models.Sequential([
        nin_block(96, kernel_size=11, strides=4, padding='valid'), 
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(256, kernel_size=5, strides=1, padding='same'), 
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),
        tf.keras.layers.Dropout(0.5),
        # 标签类别数是10
        nin_block(10, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Reshape((1, 1, 10)),
        # 将四维的输出转成二维的输出，其形状为(批量大小,10)
        tf.keras.layers.Flatten(),
    ])

# 我们创建一个数据样本来查看每个块的输出形状。
X = tf.random.uniform((1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```
结果输出为：
```bash
Sequential output shape:     (1, 54, 54, 96)
MaxPooling2D output shape:   (1, 26, 26, 96)
Sequential output shape:     (1, 26, 26, 256)
MaxPooling2D output shape:   (1, 12, 12, 256)
Sequential output shape:     (1, 12, 12, 384)
MaxPooling2D output shape:   (1, 5, 5, 384)
Dropout output shape:        (1, 5, 5, 384)
Sequential output shape:     (1, 5, 5, 10)
GlobalAveragePooling2D output shape:         (1, 10)
Reshape output shape:        (1, 1, 1, 10)
Flatten output shape:        (1, 10)
```
#### 训练模型

和以前一样，我们使用`Fashion-MNIST`来训练模型。训练`NiN`与训练`AlexNet、VGG`时相似。
{% asset_img nin_2.png %}

#### 总结

`NiN`块使用由一个卷积层和多个{% mathjax %}1\times 1{% endmathjax %}卷积层组成的块。该块可以在卷积神经网络中使用，以允许更多的每像素非线性。`NiN`去除了容易造成过拟合的全连接层，将它们替换为全局平均汇聚层（即在所有位置上进行求和）。该汇聚层通道数量为所需的输出数量（例如，`Fashion-MNIST`的输出为`10`）。移除全连接层可减少过拟合，同时显著减少`NiN`的参数。`NiN`的设计影响了许多后续卷积神经网络的设计。
