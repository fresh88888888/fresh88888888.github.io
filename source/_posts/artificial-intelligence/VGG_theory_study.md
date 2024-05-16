---
title: 使用块的网络 (VGG)(TensorFlow)
date: 2024-05-16 10:56:11
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

虽然`AlexNet`证明深层神经网络卓有成效，但它没有提供一个通用的模板来指导后续的研究人员设计新的网络。与芯片设计中工程师从放置晶体管到逻辑元件再到逻辑块的过程类似，神经网络架构的设计也逐渐变得更加抽象。研究人员开始从单个神经元的角度思考问题，发展到整个层，现在又转向块，重复层的模式。使用块的想法首先出现在牛津大学的视觉几何组(`visual geometry group`)的`VGG`网络中。通过使用循环和子程序，可以很容易地在任何现代深度学习框架的代码中实现这些重复的架构。
<!-- more -->
#### VGG块

经典卷积神经网络的基本组成部分是下面的这个序列：
- 带填充以保持分辨率的卷积层。
- 非线性激活函数，如`ReLU`。
- 汇聚层，如最大汇聚层。

而一个`VGG`块与之类似，由一系列卷积层组成，后面再加上用于空间下采样的最大汇聚层。在最初的`VGG`论文中(`Simonyan and Zisserman, 2014`)，作者使用了带有{% mathjax %}3\times 3{% endmathjax %}卷积核、填充为1（保持高度和宽度）的卷积层，和带有{% mathjax %}2\times 2{% endmathjax %}汇聚窗口、步幅为`2`（每个块后的分辨率减半）的最大汇聚层。在下面的代码中，我们定义了一个名为`vgg_block`的函数来实现一个`VGG`块。
```python
import tensorflow as tf

def vgg_block(num_convs, num_channels):
    blk = tf.keras.models.Sequential()
    for _ in range(num_convs):
        blk.add(tf.keras.layers.Conv2D(num_channels,kernel_size=3,padding='same',activation='relu'))
    blk.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    return blk
```
#### VGG网络

与`AlexNet、LeNet`一样，`VGG`网络可以分为两部分：第一部分主要由卷积层和汇聚层组成，第二部分由全连接层组成。如下图中所示。
{% asset_img vgg_1.png "从AlexNet到VGG，它们本质上都是块设计" %}

`VGG`神经网络连接上图的几个`VGG`块（在`vgg_block`函数中定义）。其中有超参数变量`conv_arch`。该变量指定了每个`VGG`块里卷积层个数和输出通道数。全连接模块则与`AlexNet`中的相同。原始`VGG`网络有`5`个卷积块，其中前两个块各有一个卷积层，后三个块各包含两个卷积层。第一个模块有`64`个输出通道，每个后续模块将输出通道数量翻倍，直到该数字达到`512`。由于该网络使用`8`个卷积层和`3`个全连接层，因此它通常被称为`VGG-11`。
```python
conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg(conv_arch):
    net = tf.keras.models.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层部分
    net.add(tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10)]))
    return net

net = vgg(conv_arch)

# 接下来，我们将构建一个高度和宽度为224的单通道数据样本，以观察每个层输出的形状。
X = tf.random.uniform((1, 224, 224, 1))
for blk in net.layers:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t', X.shape)

# Sequential output shape:     (1, 112, 112, 64)
# Sequential output shape:     (1, 56, 56, 128)
# Sequential output shape:     (1, 28, 28, 256)
# Sequential output shape:     (1, 14, 14, 512)
# Sequential output shape:     (1, 7, 7, 512)
# Sequential output shape:     (1, 10)
```
正如从代码中所看到的，我们在每个块的高度和宽度减半，最终高度和宽度都为`7`。最后再展平表示，送入全连接层处理。
#### 训练模型

由于`VGG-11`比`AlexNet`计算量更大，因此我们构建了一个通道数较少的网络，足够用于训练`Fashion-MNIST`数据集。
{% asset_img vgg_2.png %}
#### 总结

`VGG-11`使用可复用的卷积块构造网络。不同的`VGG`模型可通过每个块中卷积层数量和输出通道数量的差异来定义。块的使用导致网络定义的非常简洁。使用块可以有效地设计复杂的网络。在`VGG`论文中，`Simonyan`和`Ziserman`尝试了各种架构。特别是他们发现深层且窄的卷积（即{% mathjax %}3\times 3{% endmathjax %}）比较浅层且宽的卷积更有效。

