---
title: 计算架构 (机器学习) (TensorFlow)
date: 2024-05-13 09:00:11
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

#### 层和块

介绍神经网络时，我们关注的是具有单一输出的线性模型。在这里，整个模型只有一个输出。注意，单个神经网络（1）接受一些输入；（2）生成相应的标量输出；（3）具有一组相关参数(`parameters`)，更新这些参数可以优化某目标函数。然后，当考虑具有多个输出的网络时， 我们利用矢量化算法来描述整层神经元。像单个神经元一样，层（1）接受一组输入，（2）生成相应的输出，（3）由一组可调整参数描述。当我们使用`softmax`回归时，一个单层本身就是模型。然而，即使我们随后引入了多层感知机，我们仍然可以认为该模型保留了上面所说的基本架构。对于多层感知机而言，整个模型及其组成层都是这种架构。整个模型接受原始输入（特征），生成输出（预测），并包含一些参数（所有组成层的参数集合）。同样，每个单独的层接收输入（由前一层提供），生成输出（到下一层的输入），并且具有一组可调参数，这些参数根据从下一层反向传播的信号进行更新。
<!-- more -->
为了实现这些复杂的网络，我们引入了神经网络块的概念。**块**(`block`)可以描述单个层、由多个层组成的组件或整个模型本身。使用块进行抽象的一个好处是可以将一些块组合成更大的组件，这一过程通常是递归的，如下图所示。通过定义代码来按需生成任意复杂度的块，我们可以通过简洁的代码实现复杂的神经网络。
{% asset_img dp_1.png "多个层被组合成块，形成更大的模型" %}

从编程的角度来看，块由**类**(`class`)表示。 它的任何子类都必须定义一个将其输入转换为输出的前向传播函数，并且必须存储任何必需的参数。注意，有些块不需要任何参数。最后，为了计算梯度，块必须具有反向传播函数。在定义我们自己的块时，由于自动微分提供了一些后端实现，我们只需要考虑前向传播函数和必需的参数。在构造自定义块之前，我们先回顾一下多层感知机的代码。下面的代码生成一个网络，其中包含一个具有`256`个单元和`ReLU`激活函数的全连接隐藏层，然后是一个具有`10`个隐藏单元且不带激活函数的全连接输出层。
```python
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

X = tf.random.uniform((2, 20))
net(X)

# <tf.Tensor: shape=(2, 10), dtype=float32, numpy=
# array([[-0.1256689 ,  0.03234727, -0.41341984,  0.05109158, -0.11376685,
#          0.1656029 ,  0.13811918, -0.0332518 , -0.28478232, -0.04640551],
#        [ 0.00945267,  0.01088307, -0.3047434 ,  0.05576317,  0.08904827,
#          0.11957583,  0.10018335,  0.07535183, -0.17810427, -0.03584548]],
#       dtype=float32)>
```
在这个例子中，我们通过实例化`keras.models.Sequential`来构建我们的模型，层的执行顺序是作为参数传递的。简而言之，`Sequential`定义了一种特殊的`keras.Model`，即在`Keras`中表示一个块的类。它维护了一个由`Model`组成的有序列表，注意两个全连接层都是`Model`类的实例，这个类本身就是`Model`的子类。前向传播(`call`)函数也非常简单：它将列表中的每个块连接在一起，将每个块的输出作为下一个块的输入。注意，到目前为止，我们一直在通过`net(X)`调用我们的模型来获得模型的输出。这实际上是`net.call(X)`的简写，这是通过`Block`类的`__call__`函数实现的一个`Python`技巧。
##### 自定义块

要想直观地了解块是如何工作的，最简单的方法就是自己实现一个。在实现我们自定义块之前，我们简要总结一下每个块必须提供的基本功能。
- 将输入数据作为其前向传播函数的参数。
- 通过前向传播函数来生成输出。请注意，输出的形状可能与输入的形状不同。例如，我们上面模型中的第一个全连接的层接收任意维的输入，但是返回一个维度`256`的输出。
- 计算其输出关于输入的梯度，可通过其反向传播函数进行访问。通常这是自动发生的。
- 存储和访问前向传播计算所需的参数。
- 根据需要初始化模型参数。

在下面的代码片段中，我们从零开始编写一个块。它包含一个多层感知机，其具有`256`个隐藏单元的隐藏层和一个`10`维输出层。注意，下面的`MLP`类继承了表示块的类。我们的实现只需要提供我们自己的构造函数（`Python`中的`__init__`函数）和前向传播函数。
```python
class MLP(tf.keras.Model):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self):
        # 调用MLP的父类Model的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params
        super().__init__()
        # Hiddenlayer
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)  # Outputlayer

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def call(self, X):
        return self.out(self.hidden((X)))

net = MLP()
net(X)

# <tf.Tensor: shape=(2, 10), dtype=float32, numpy=
# array([[ 0.14015174,  0.17783523,  0.03422496,  0.23184124,  0.310251  ,
#          0.14864878, -0.5013749 , -0.0734642 , -0.03820562, -0.12923583],
#        [ 0.47990555,  0.42501003,  0.10588682,  0.03492985,  0.2023867 ,
#          0.25548872, -0.50454026, -0.39808106, -0.00930042, -0.17727089]],
#       dtype=float32)>
```
我们首先看一下前向传播函数，它以`X`作为输入，计算带有激活函数的隐藏表示，并输出其未规范化的输出值。在这个`ML`P实现中，两个层都是实例变量。要了解这为什么是合理的，可以想象实例化两个多层感知机（`net1`和`net2`），并根据不同的数据对它们进行训练。当然，我们希望它们学到两种不同的模型。接着我们实例化多层感知机的层，然后在每次调用前向传播函数时调用这些层。注意一些关键细节：首先，我们定制的`__init__`函数通过`super().__init__()`调用父类的`__init__`函数，省去了重复编写模版代码的痛苦。然后，我们实例化两个全连接层，分别为`self.hidden`和`self.out`。注意，除非我们实现一个新的运算符，否则我们不必担心反向传播函数或参数初始化，系统将自动生成这些。块的一个主要优点是它的多功能性。我们可以子类化块以创建层（如全连接层的类）、整个模型（如上面的`MLP`类）或具有中等复杂度的各种组件。
##### 顺序块

现在我们可以更仔细地看看`Sequential`类是如何工作的，回想一下`Sequential`的设计是为了把其他模块串起来。为了构建我们自己的简化的`MySequential`，我们只需要定义两个关键函数：
- 一种将块逐个追加到列表中的函数；
- 一种前向传播函数，用于将输入按追加块的顺序传递给块组成的“链条”。

下面的`MySequential`类提供了与默认`Sequential`类相同的功能。
```python
class MySequential(tf.keras.Model):
    def __init__(self, *args):
        super().__init__()
        self.modules = []
        for block in args:
            # 这里，block是tf.keras.layers.Layer子类的一个实例
            self.modules.append(block)

    def call(self, X):
        for module in self.modules:
            X = module(X)
        return X

net = MySequential(tf.keras.layers.Dense(units=256, activation=tf.nn.relu),tf.keras.layers.Dense(10))
net(X)

# <tf.Tensor: shape=(2, 10), dtype=float32, numpy=
# array([[ 0.4245665 ,  0.1554529 , -0.06504549,  0.0987289 , -0.08489662,
#          0.16747624,  0.20746413,  0.05763938, -0.16166216,  0.382744  ],
#        [ 0.47101185, -0.0233981 ,  0.21728408,  0.14111494, -0.18493696,
#          0.08736669,  0.13651624,  0.3103686 , -0.09440522,  0.35759482]],
#       dtype=float32)>
```
当`MySequential`的前向传播函数被调用时，每个添加的块都按照它们被添加的顺序执行。现在可以使用我们的`MySequential`类重新实现多层感知机。
##### 在前向传播函数中执行代码

`Sequential`类使模型构造变得简单，允许我们组合新的架构，而不必定义自己的类。然而，并不是所有的架构都是简单的顺序架构。当需要更强的灵活性时，我们需要定义自己的块。例如，我们可能希望在前向传播函数中执行`Python`的控制流。此外，我们可能希望执行任意的数学运算，而不是简单地依赖预定义的神经网络层。到目前为止，我们网络中的所有操作都对网络的激活值及网络的参数起作用。然而，有时我们可能希望合并既不是上一层的结果也不是可更新参数的项，我们称之为**常数参数**(`constant parameter`)。例如，我们需要一个计算函数{% mathjax %}f(\mathbf{x},\mathbf{w}) = c \cdot \mathbf{w}^{\mathsf{T}}\mathbf{x}{% endmathjax %}的层，其中{% mathjax %}\mathbf{x}{% endmathjax %}是输入，{% mathjax %}\mathbf{w}{% endmathjax %}是参数，{% mathjax %}c{% endmathjax %}是某个在优化过程中没有更新的指定常量。因此我们实现了一个`FixedHiddenMLP`类，如下所示：
```python
class FixedHiddenMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        # 使用tf.constant函数创建的随机权重参数在训练期间不会更新（即为常量参数）
        self.rand_weight = tf.constant(tf.random.uniform((20, 20)))
        self.dense = tf.keras.layers.Dense(20, activation=tf.nn.relu)

    def call(self, inputs):
        X = self.flatten(inputs)
        # 使用创建的常量参数以及relu和matmul函数
        X = tf.nn.relu(tf.matmul(X, self.rand_weight) + 1)
        # 复用全连接层。这相当于两个全连接层共享参数。
        X = self.dense(X)
        # 控制流
        while tf.reduce_sum(tf.math.abs(X)) > 1:
            X /= 2
        return tf.reduce_sum(X)

net = FixedHiddenMLP()
net(X)

# <tf.Tensor: shape=(), dtype=float32, numpy=0.7753998>
```
在这个`FixedHiddenMLP`模型中，我们实现了一个隐藏层，其权重（`self.rand_weight`）在实例化时被随机初始化，之后为常量。这个权重不是一个模型参数，因此它永远不会被反向传播更新。然后，神经网络将这个固定层的输出通过一个全连接层。注意，在返回输出之前，模型做了一些不寻常的事情：它运行了一个`while`循环，在{% mathjax %}L_1{% endmathjax %}范数大于`1`的条件下，将输出向量除以`2`，直到它满足条件为止。最后，模型返回了`X`中所有项的和。注意，此操作可能不会常用于在任何实际任务中，我们只展示如何将任意代码集成到神经网络计算的流程中。我们可以混合搭配各种组合块的方法。在下面的例子中，我们以一些想到的方法嵌套块。
```python
class NestMLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.net = tf.keras.Sequential()
        self.net.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        self.net.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
        self.dense = tf.keras.layers.Dense(16, activation=tf.nn.relu)

    def call(self, inputs):
        return self.dense(self.net(inputs))

chimera = tf.keras.Sequential()
chimera.add(NestMLP())
chimera.add(tf.keras.layers.Dense(20))
chimera.add(FixedHiddenMLP())
chimera(X)

# <tf.Tensor: shape=(), dtype=float32, numpy=0.7313081>
```
你可能会担心操作效率的问题。毕竟，我们在一个高性能的深度学习库中进行了大量的字典查找、代码执行和许多其他的`Python`代码。`Python`的问题全局解释器锁 是众所周知的。在深度学习环境中，我们担心速度极快的`GPU`可能要等到`CPU`运行`Python`代码后才能运行另一个作业。

##### 总结

一个块可以由许多层组成；一个块可以由许多块组成。块可以包含代码。块负责大量的内部处理，包括参数初始化和反向传播。层和块的顺序连接由`Sequential`块处理。

#### 参数管理

在选择了架构并设置了超参数后，我们就进入了训练阶段。此时，我们的目标是找到使损失函数最小化的模型参数值。经过训练后，我们将需要使用这些参数来做出未来的预测。此外，有时我们希望提取参数，以便在其他环境中复用它们，将模型保存下来，以便它可以在其他软件中执行。我们首先看一下具有单隐藏层的多层感知机。
```python
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
net(X)

# <tf.Tensor: shape=(2, 1), dtype=float32, numpy=array([[-0.44218335],[ 0.57875514]], dtype=float32)>
```
