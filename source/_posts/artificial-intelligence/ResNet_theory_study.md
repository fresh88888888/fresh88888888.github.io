---
title: 残差网络 (ResNet)(TensorFlow)
date: 2024-05-16 14:38:11
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

随着我们设计越来越深的网络，深刻理解“新添加的层如何提升神经网络的性能”变得至关重要。更重要的是设计网络的能力，在这种网络中，添加层会使网络更具表现力，为了取得质的突破，我们需要一些数学基础知识。
<!-- more -->
#### 函数类

首先，假设有一类特定的神经网络架构{% mathjax %}\mathcal{F}{% endmathjax %}，它包括学习速率和其他超参数设置。对于所有{% mathjax %}f\in \mathcal{F}{% endmathjax %}，存在一些参数集（例如权重和偏置），这些参数可以通过在合适的数据集上进行训练而获得。现在假设{% mathjax %}f^{\ast}{% endmathjax %}是我们真正想要找到的函数，如果{% mathjax %}f^{\ast}\in \mathcal{F}{% endmathjax %}，那我们可以轻而易举的得到它，但通常我们不会幸运。相反，我们将尝试找到一个函数{% mathjax %}f_{\mathcal{F}}^{\ast}{% endmathjax %}，这是我们在{% mathjax %}\mathcal{F}{% endmathjax %}中的最佳选择。例如，给定一个具有{% mathjax %}\mathcal{X}{% endmathjax %}特性和{% mathjax %}\mathcal{y}{% endmathjax %}标签的数据集，我们可以尝试通过以下优化问题来找到它：
{% mathjax '{"conversion":{"em":14}}' %}
f_{\mathcal{F}}^{\ast} := \underset{f}{\text{argmin}}L(\mathbf{X},\mathbf{y},f)\;\text{subject to}\;f\in \mathcal{F}
{% endmathjax %}
那么怎样得到更近似真正{% mathjax %}f^{\ast}{% endmathjax %}的函数呢？唯一合理的可能性是，我们需要设计一个更强大的架构{% mathjax %}\mathcal{F}'{% endmathjax %}。换句话说，我们预计{% mathjax %}f_{\mathcal{F}'}^{\ast}{% endmathjax %}比{% mathjax %}f_{\mathcal{F}}^{\ast}{% endmathjax %}更近似。然而，如果{% mathjax %}\mathcal{F}\not\subseteq \mathcal{F}'{% endmathjax %}，则无法保证新的体系“更近似”。事实上{% mathjax %} f_{\mathcal{F}'}^{\ast}{% endmathjax %}更糟：如下图所示，对于非嵌套函数(`non-nested function`)类，较复杂的函数类并不总是向“真”函数{% mathjax %}f^{\ast}{% endmathjax %}靠拢。在下图的左边，虽然{% mathjax %}\mathcal{F}_3{% endmathjax %}比{% mathjax %}\mathcal{F}_1{% endmathjax %}更接近{% mathjax %}f^{\ast}{% endmathjax %}。，但{% mathjax %}\mathcal{F}_6{% endmathjax %}却离得更远了。相反对于下图右侧的嵌套函数(`nested function`)类{% mathjax %}\mathcal{F}_1\subseteq \dots \subseteq \mathcal{F}_6{% endmathjax %}，我们可以避免上述问题。
{% asset_img rn_1.png "对于非嵌套函数类，较复杂（由较大区域表示）的函数类不能保证更接近真函数）。这种现象在嵌套函数类中不会发生。" %}

因此，只有当较复杂的函数类包含较小的函数类时，我们才能确保提高它们的性能。对于深度神经网络，如果我们能将新添加的层训练成**恒等映射**(`identity function`){% mathjax %}f(\mathbf{x}) = \mathbf{x}{% endmathjax %}，新模型和原模型将同样有效。同时，由于新模型可能得出更优的解来拟合训练数据集，因此添加层似乎更容易降低训练误差。针对这一问题，何恺明等人提出了**残差网络**(`ResNet`)。它在`2015`年的`ImageNet`图像识别挑战赛夺魁，并深刻影响了后来的深度神经网络的设计。残差网络的核心思想是：**每个附加层都应该更容易地包含原始函数作为其元素之一**。于是，**残差块**(`residual blocks`)便诞生了，这个设计对如何建立深层神经网络产生了深远的影响。凭借它，`ResNet`赢得了`2015`年`ImageNet`大规模视觉识别挑战赛。
#### 残差块

让我们聚焦于神经网络局部：如下图所示，假设我们的原始输入为{% mathjax %}x{% endmathjax %}，而希望学出的理想映射为{% mathjax %} {% endmathjax %}。下左图虚线框中的部分需要直接拟合出该映射{% mathjax %}f(\mathbf{x}){% endmathjax %}，而右图虚线框中的部分则需要拟合出残差映射{% mathjax %}f(\mathbf{x}){% endmathjax %}。残差映射在现实中往往更容易优化。开头提到的恒等映射作为我们希望学出的理想映射{% mathjax %}f(\mathbf{x}) - \mathbf{x}{% endmathjax %}，我们只需将下右图虚线框内上方的加权运算（如仿射）的权重和偏置参数设成`0`，那么{% mathjax %}f(\mathbf{x}){% endmathjax %}即为恒等映射。实际中，当理想映射{% mathjax %}f(\mathbf{x}){% endmathjax %}极接近于恒等映射时，残差映射也易于捕捉恒等映射的细微波动。下右图是`ResNet`的基础架构–残差块(`residual block`)。在残差块中，输入可通过跨层数据线路更快地向前传播。
{% asset_img rn_2.png "一个正常块（左图）和一个残差块（右图）" %}

`ResNet`沿用了`VGG`完整的{% mathjax %}3\times 3{% endmathjax %}卷积层设计。残差块里首先有`2`个有相同输出通道数的{% mathjax %}3\times 3{% endmathjax %}卷积层。每个卷积层后接一个批量规范化层和`ReLU`激活函数。然后我们通过跨层数据通路，跳过这`2`个卷积运算，将输入直接加在最后的`ReLU`激活函数前。这样的设计要求`2`个卷积层的输出与输入形状一样，从而使它们可以相加。如果想改变通道数，就需要引入一个额外的{% mathjax %}1\times 1{% endmathjax %}卷积层来将输入变换成需要的形状后再做相加运算。如下图所示，此代码生成两种类型的网络：一种是当`use_1x1conv=False`时，应用`ReLU`非线性函数之前，将输入添加到输出。另一种是当`use_1x1conv=True`时，添加通过{% mathjax %}1\times 1{% endmathjax %}卷积调整通道和分辨率。
{% asset_img rn_3.png "包含以及不包含1 x 1卷积层的残差块" %}

#### ResNet模型

`ResNet`的前两层跟之前介绍的`GoogLeNet`中的一样：在输出通道数为`64`、步幅为`2`的{% mathjax %}7\times 7{% endmathjax %}卷积层后，接步幅为`2`的{% mathjax %}3\times 3{% endmathjax %}的最大汇聚层。不同之处在于`ResNet`每个卷积层后增加了批量规范化层。`GoogLeNet`在后面接了`4`个由`Inception`块组成的模块。`ResNet`则使用`4`个由残差块组成的模块，每个模块使用若干个同样输出通道数的残差块。第一个模块的通道数同输入通道数一致。由于之前已经使用了步幅为`2`的最大汇聚层，所以无须减小高和宽。之后的每个模块在第一个残差块里将上一个模块的通道数翻倍，并将高和宽减半。
```python
import tensorflow as tf

class Residual(tf.keras.Model):  
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(num_channels, padding='same', kernel_size=3, strides=strides)
        self.conv2 = tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(num_channels, kernel_size=1, strides=strides)

        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()

    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

class ResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X

b1 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')])

# 接着在ResNet加入所有残差块，这里每个模块使用2个残差块。
b2 = ResnetBlock(64, 2, first_block=True)
b3 = ResnetBlock(128, 2)
b4 = ResnetBlock(256, 2)
b5 = ResnetBlock(512, 2)

# 最后，与GoogLeNet一样，在ResNet中加入全局平均汇聚层，以及全连接层输出。

# 回想之前我们定义一个函数，以便用它在tf.distribute.MirroredStrategy的范围，
# 来利用各种计算资源，例如gpu。另外，尽管我们已经创建了b1、b2、b3、b4、b5，
# 但是我们将在这个函数的作用域内重新创建它们
def net():
    return tf.keras.Sequential([
        # Thefollowinglayersarethesameasb1thatwecreatedearlier
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),
        # Thefollowinglayersarethesameasb2,b3,b4,andb5thatwe,createdearlier
        ResnetBlock(64, 2, first_block=True),
        ResnetBlock(128, 2),
        ResnetBlock(256, 2),
        ResnetBlock(512, 2),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=10)])

X = tf.random.uniform(shape=(1, 224, 224, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__,'output shape:\t', X.shape)
```
结果输出为：
```bash
Conv2D output shape:         (1, 112, 112, 64)
BatchNormalization output shape:     (1, 112, 112, 64)
Activation output shape:     (1, 112, 112, 64)
MaxPooling2D output shape:   (1, 56, 56, 64)
ResnetBlock output shape:    (1, 56, 56, 64)
ResnetBlock output shape:    (1, 28, 28, 128)
ResnetBlock output shape:    (1, 14, 14, 256)
ResnetBlock output shape:    (1, 7, 7, 512)
GlobalAveragePooling2D output shape:         (1, 512)
Dense output shape:  (1, 10)
```
每个模块有`4`个卷积层（不包括恒等映射的{% mathjax %}1\times 1{% endmathjax %}卷积层）。加上第一个{% mathjax %}7\times 7{% endmathjax %}卷积层和最后一个全连接层，共有`18`层。因此，这种模型通常被称为`ResNet-18`。通过配置不同的通道数和模块里的残差块数可以得到不同的`ResNet`模型，例如更深的含`152`层的`ResNet-152`。虽然`ResNet`的主体架构跟`GoogLeNet`类似，但`ResNet`架构更简单，修改也更方便。这些因素都导致了`ResNet`迅速被广泛使用。下图描述了完整的`ResNet-18`。
{% asset_img rn_4.png "ResNet-18 架构" %}

在训练`ResNet`之前，让我们观察一下`ResNet`中不同模块的输入形状是如何变化的。在之前所有架构中，分辨率降低，通道数量增加，直到全局平均汇聚层聚集所有特征。
#### 训练模型

同之前一样，我们在`Fashion-MNIST`数据集上训练`ResNet`。
{% asset_img rn_5.png %}
#### 总结

学习嵌套函数(`nested function`)是训练神经网络的理想情况。在深层神经网络中，学习另一层作为恒等映射(`identity function`)较容易（尽管这是一个极端情况）。残差映射可以更容易地学习同一函数，例如将权重层中的参数近似为零。利用残差块(`residual blocks`)可以训练出一个有效的深层神经网络：输入可以通过层间的残余连接更快地向前传播。残差网络(`ResNet`)对随后的深层神经网络设计产生了深远影响。
