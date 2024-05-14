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

# 通过索引可以访问模型的任意层
print(net.layers[2].weights)

# <tf.Tensor: shape=(2, 1), dtype=float32, numpy=array([[-0.44218335],[ 0.57875514]], dtype=float32)>

# [<tf.Variable 'dense_1/kernel:0' shape=(4, 1) dtype=float32, numpy=
# array([[-1.0469127 ],
#        [ 0.31355536],
#        [ 0.5405549 ],
#        [ 0.7610214 ]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
```
我们从已有的模型访问参数。当通过`Sequential`类定义模型时，我们可以通过索引访问模型的任意层。就像模型是一个列表一样，每层的参数都在其属性中。输出的结果告诉我们一些重要的：首先，这个全连接层包含两个参数，分别是该层的权重和偏置，两者都存储为单精度浮点数，注意，参数名称允许唯一标识每个参数，即使包含数百层的网络中也是如此。
##### 目标参数

注意，每个参数都表示参数类的一个实例。要对参数执行任何操作，首先我们需要访问底层的数值。有几种方法可以做到这一点。有些比较简单，而另一些则比较通用。下面的代码从第二个全连接层（即第三个神经网络层）提取偏置，提取后返回的是一个参数类实例，并进一步访问该参数的值。
```python
print(type(net.layers[2].weights[1]))
print(net.layers[2].weights[1])
print(tf.convert_to_tensor(net.layers[2].weights[1]))

# <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
# <tf.Variable 'dense_1/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>
# tf.Tensor([0.], shape=(1,), dtype=float32)
```
##### 一次性访问所有参数

当我们对所有参数执行操作时，逐个访问它们会很麻烦。当我们处理更复杂的块（例如，嵌套块）时，情况可能会变的特别复杂，因为我们需要递归整个树来提取每个子块的参数，下面，我们将通过演示来比较访问第一个全连接层的参数和访问所有层。
```python
print(net.layers[1].weights)
print(net.get_weights())

# [<tf.Variable 'dense/kernel:0' shape=(4, 4) dtype=float32, numpy=
# array([[ 0.07396382, -0.6543436 , -0.7244056 , -0.6157465 ],
#        [ 0.40404958,  0.7228444 ,  0.4572547 ,  0.7116396 ],
#        [ 0.5283937 ,  0.25636894, -0.49113625,  0.6337872 ],
#        [ 0.5183577 ,  0.19943613,  0.5296057 ,  0.6009421 ]],
#       dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>]
# [array([[ 0.07396382, -0.6543436 , -0.7244056 , -0.6157465 ],
#        [ 0.40404958,  0.7228444 ,  0.4572547 ,  0.7116396 ],
#        [ 0.5283937 ,  0.25636894, -0.49113625,  0.6337872 ],
#        [ 0.5183577 ,  0.19943613,  0.5296057 ,  0.6009421 ]],
#       dtype=float32), array([0., 0., 0., 0.], dtype=float32), array([[-1.0469127 ],
#        [ 0.31355536],
#        [ 0.5405549 ],
#        [ 0.7610214 ]], dtype=float32), array([0.], dtype=float32)]
```
这为我们提供了另一种访问网络参数的方式，如下所示
```python
net.get_weights()[1]

# array([0., 0., 0., 0.], dtype=float32)
```
##### 从嵌套块收集参数

让我们看看，如果我们将多个块相互嵌套，参数命名约定是如何工作的。我们首先定义一个生成块的函数（可以说是块工厂）然后将这些块组合到更大的块中。
```python
def block1(name):
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.relu)],
        name=name)

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add(block1(name=f'block-{i}'))
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(1))
rgnet(X)

# <tf.Tensor: shape=(2, 1), dtype=float32, numpy=array([[0.04106742],[0.08519742]], dtype=float32)>
```
设计了网络之后，可以看看它是如何工作的。
```python
print(rgnet.summary())
```
结果输出为：
```bash
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 sequential_2 (Sequential)   (2, 4)                    80

 dense_6 (Dense)             (2, 1)                    5

=================================================================
Total params: 85
Trainable params: 85
Non-trainable params: 0
_________________________________________________________________
None
```
因为层是分层嵌套的，所以我们也可以像通过嵌套列表索引一样访问它们。下面我们访问第一个主要的块中、第二子块的第一层偏置项。
```python
rgnet.layers[0].layers[1].layers[1].weights[1]

# <tf.Variable 'dense_3/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>
```
##### 参数初始化

知道了如何访问参数后，现在我们看看如何正确地初始化参数。默认情况下，`Keras`会根据一个范围均匀地初始化权重矩阵，这个范围是根据输入和输出维度计算出的。偏置参数设置为`0`。`TensorFlow`在根模块和`keras.initializers`模块中提供了各种初始化方法。
###### 内置初始化

让我们首先调用内置的初始化器。下面的代码将所有权重参数初始化为标准差为`0.01`的高斯随机变量，且将偏置参数设置为`0`。
```python
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)])

net(X)
net.weights[0], net.weights[1]

# (<tf.Variable 'dense_7/kernel:0' shape=(4, 4) dtype=float32, numpy=
#  array([[-0.0038965 ,  0.00505942,  0.02313872,  0.01330682],
#         [-0.00415377, -0.00385469,  0.01013125, -0.00638383],
#         [-0.00714976,  0.00160496,  0.01625365,  0.00301881],
#         [-0.01674125, -0.04097489,  0.0116432 ,  0.01404491]],
#        dtype=float32)>,<tf.Variable 'dense_7/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>)
```
我们还可以将所有参数初始化为给定的常数，比如初始化为`1`。
```python
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu,kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
net.weights[0], net.weights[1]

# (<tf.Variable 'dense_9/kernel:0' shape=(4, 4) dtype=float32, numpy=
#  array([[1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.],
#         [1., 1., 1., 1.]], dtype=float32)>,
#  <tf.Variable 'dense_9/bias:0' shape=(4,) dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>)
```
我们还可以对某些块应用不同的初始化方法。例如，下面我们使用`Xavier`初始化方法初始化第一个神经网络层，然后将第三个神经网络层初始化为常量值`1`。
```python
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu,kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.Constant(1)),
])

net(X)
print(net.layers[1].weights[0])
print(net.layers[2].weights[0])

# <tf.Variable 'dense_11/kernel:0' shape=(4, 4) dtype=float32, numpy=
# array([[ 0.46413535, -0.41529804, -0.7629936 ,  0.6337715 ],
#        [-0.03254855, -0.7786831 ,  0.5948091 , -0.34829807],
#        [ 0.8473349 ,  0.54338247, -0.27631932,  0.76972014],
#        [ 0.3425359 ,  0.84501284, -0.62855405, -0.02751094]],
#       dtype=float32)>
# <tf.Variable 'dense_12/kernel:0' shape=(4, 1) dtype=float32, numpy=array([[1.],[1.],[1.],[1.]], dtype=float32)>
```
###### 自定义初始化

有时，深度学习框架没有提供我们需要的初始化方法。在下面的例子中，我们使用以下的分布为任意权重参数{% mathjax %}w{% endmathjax %}定义初始化方法：
{% mathjax '{"conversion":{"em":14}}' %}
w \sim 
\begin{cases}
    U(5,10) & \text{可能性}\;\frac{1}{4}\\
    0 & \text{可能性}\;\frac{1}{2}\\
    U(-10,-5) & \text{可能性}\;\frac{1}{4}
\end{cases}
{% endmathjax %}
在这里，我们定义了一个`Initializer`的子类，并实现了`__call__`函数。该函数返回给定形状和数据类型的所需张量。
```python
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data=tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor=(tf.abs(data) >= 5)
        factor=tf.cast(factor, tf.float32)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4,activation=tf.nn.relu,kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print(net.layers[1].weights[0])

# <tf.Variable 'dense_13/kernel:0' shape=(4, 4) dtype=float32, numpy=
# array([[ 0.       , -0.       ,  0.       , -0.       ],
#        [-0.       ,  6.2459354, -0.       ,  0.       ],
#        [ 9.651949 , -0.       , -6.009481 , -0.       ],
#        [ 7.8956127, -6.5848374, -0.       , -8.0049305]], dtype=float32)>
```
注意，我们始终可以直接设置参数。
```python
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
net.layers[1].weights[0]

# <tf.Variable 'dense_13/kernel:0' shape=(4, 4) dtype=float32, numpy=
# array([[42.       ,  1.       ,  1.       ,  1.       ],
#        [ 1.       ,  7.2459354,  1.       ,  1.       ],
#        [10.651949 ,  1.       , -5.009481 ,  1.       ],
#        [ 8.895613 , -5.5848374,  1.       , -7.0049305]], dtype=float32)>
```
##### 参数绑定

有时我们希望多个层之间共享参数：我们可以定义一个稠密层然后使用它的参数来设置另一个层的参数。
```python
# tf.keras的表现有点不同。它会自动删除重复层
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([tf.keras.layers.Flatten(),shared,shared,tf.keras.layers.Dense(1),])

net(X)
# 检查参数是否不同
print(len(net.layers) == 3)
```
##### 总结

我们有几种方法可以访问、初始化和绑定模型参数。可以使用自定义初始化方法。

#### 延后初始化

到目前为止，我们忽略了建立网络时需要做的以下这些事情：
- 我们定义了网络架构，但没有指定输入维度。
- 我们添加层时没有指定前一层的输出维度。
- 我们在初始化参数时，甚至没有足够的信息来确定模型应该包含多少参数。

深度学习框架无法判断网络的输入维度是什么。这里的诀窍是框架的延后初始化(`defers initialization`)，即直到数据第一次通过模型传递时，框架才会动态地推断出每个层的大小。
##### 实例化网络

首先，让我们实例化一个多层感知机。
```python
import tensorflow as tf

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

[net.layers[i].get_weights() for i in range(len(net.layers))]

# [[], []]
```
此时，因为输入维数是未知的，所以网络不可能知道输入层权重的维数。因此，框架尚未初始化任何参数，我们通过尝试访问以下参数进行确认。请注意，每个层对象都存在，但权重为空。使用`net.get_weights()`将抛出一个错误，因为权重尚未初始化。接下来让我们将数据通过网络，最终使框架初始化参数。
```python
X = tf.random.uniform((2, 20))
net(X)
[w.shape for w in net.get_weights()]

# [(20, 256), (256,), (256, 10), (10,)]
```
一旦我们知道输入维数是`20`，框架可以通过代入值`20`来识别第一层权重矩阵的形状。识别出第一层的形状后，框架处理第二层，依此类推，直到所有形状都已知为止。注意，在这种情况下，只有第一层需要延迟初始化，但是框架仍是按顺序初始化的。等到知道了所有的参数形状，框架就可以初始化参数。
##### 总结

延后初始化使框架能够自动推断参数形状，使修改模型架构变得容易，避免了一些常见的错误。我们可以通过模型传递数据，使框架最终初始化参数。

#### 自定义层

深度学习成功背后的一个因素是神经网络的灵活性：我们可以用创造性的方式组合不同的层，从而设计出适用于各种任务的架构。例如，研究人员发明了专门用于处理图像、文本、序列数据和执行动态规划的层。有时我们会遇到或要自己发明一个现在在深度学习框架中还不存在的层。在这些情况下，必须构建自定义层。
##### 不带参数的层

首先，我们构造一个没有任何参数的自定义层。下面的`CenteredLayer`类要从其输入中减去均值。要构建它，我们只需继承基础层类并实现前向传播功能。
```python
import tensorflow as tf

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)

# 让我们向该层提供一些数据，验证它是否能按预期工作。
layer = CenteredLayer()
layer(tf.constant([1, 2, 3, 4, 5]))

# <tf.Tensor: shape=(5,), dtype=int32, numpy=array([-2, -1,  0,  1,  2], dtype=int32)>

# 现在，我们可以将层作为组件合并到更复杂的模型中。
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])

# 作为额外的健全性检查，我们可以在向该网络发送随机数据后，检查均值是否为0。 
# 由于我们处理的是浮点数，因为存储精度的原因，我们仍然可能会看到一个非常小的非零数。
Y = net(tf.random.uniform((4, 8)))
tf.reduce_mean(Y)

# <tf.Tensor: shape=(), dtype=float32, numpy=-9.313226e-10>
```
##### 带参数的层

以上我们知道了如何定义简单的层，下面我们继续定义具有参数的层，这些参数可以通过训练进行调整。我们可以使用内置函数来创建参数，这些函数提供一些基本的管理功能。比如管理访问、初始化、共享、保存和加载模型参数。这样做的好处之一是：我们不需要为每个自定义层编写自定义的序列化程序。现在，让我们实现自定义版本的全连接层。回想一下，该层需要两个参数，一个用于表示权重，另一个用于表示偏置项。在此实现中，我们使用修正线性单元作为激活函数。该层需要输入参数：`in_units`和`units`，分别表示输入数和输出数。
```python
class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)

# 接下来，我们实例化MyDense类并访问其模型参数。
dense = MyDense(3)
dense(tf.random.uniform((2, 5)))
dense.get_weights()

# [array([[-0.013614  , -0.01669732,  0.02921283],
#         [ 0.03179312,  0.0889833 , -0.02140525],
#         [ 0.05018818,  0.02113006,  0.07468227],
#         [ 0.03596197, -0.0285063 ,  0.04013855],
#         [-0.0061096 , -0.00112533,  0.01261374]], dtype=float32),
#  array([0., 0., 0.], dtype=float32)]

# 我们可以使用自定义层直接执行前向传播计算。
dense(tf.random.uniform((2, 5)))

# <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
# array([[0.04618492, 0.01554962, 0.10059999],[0.0296516 , 0.01305952, 0.08538137]], dtype=float32)>

# 我们还可以使用自定义层构建模型，就像使用内置的全连接层一样使用自定义层。
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
net(tf.random.uniform((2, 64)))

# <tf.Tensor: shape=(2, 1), dtype=float32, numpy=array([[0.00289017],[0.00536015]], dtype=float32)>
```
##### 总结

我们可以通过基本层类设计自定义层。这允许我们定义灵活的新层，其行为与深度学习框架中的任何现有层不同。在自定义层定义完成后，我们就可以在任意环境和网络架构中调用该自定义层。层可以有局部参数，这些参数可以通过内置函数创建。

#### 读写文件

到目前为止，我们讨论了如何处理数据，以及如何构建、训练和测试深度学习模型。然而，有时我们希望保存训练的模型，以备将来在各种环境中使用（比如在部署中进行预测）。此外，当运行一个耗时较长的训练过程时，最佳的做法是定期保存中间结果，以确保在服务器电源被不小心断掉时，我们不会损失几天的计算结果。因此，现在是时候学习如何加载和存储权重向量和整个模型了。
##### 加载和保存张量

对于单个张量，我们可以直接调用`load`和`save`函数分别读写它们。这两个函数都要求我们提供一个名称，`save`要求将要保存的变量作为输入。
```python
import numpy as np
import tensorflow as tf

x = tf.range(4)
np.save('x-file.npy', x)

# 我们现在可以将存储在文件中的数据读回内存。
x2 = np.load('x-file.npy', allow_pickle=True)
x2

# array([0, 1, 2, 3], dtype=int32)

# 我们可以存储一个张量列表，然后把它们读回内存。
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
(x2, y2)

# (array([0., 1., 2., 3.]), array([0., 0., 0., 0.]))

# 我们甚至可以写入或读取从字符串映射到张量的字典。当我们要读取或写入模型中的所有权重时，这很方便。
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
mydict2 = np.load('mydict.npy', allow_pickle=True)
mydict2

# array({'x': <tf.Tensor: shape=(4,), dtype=int32, numpy=array([0, 1, 2, 3], dtype=int32)>, 
#        'y': <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0., 0., 0., 0.], dtype=float32)>},dtype=object)
```
##### 加载和保存模型参数

保存单个权重向量（或其他张量）确实有用，但是如果我们想保存整个模型，并在以后加载它们，单独保存每个向量则会变得很麻烦。毕竟，我们可能有数百个参数散布在各处。因此，深度学习框架提供了内置函数来保存和加载整个网络。需要注意的一个重要细节是，这将保存模型的参数而不是保存整个模型。例如，如果我们有一个`3`层多层感知机，我们需要单独指定架构。因为模型本身可以包含任意代码，所以模型本身难以序列化。因此，为了恢复模型，我们需要用代码生成架构，然后从磁盘加载参数。让我们从熟悉的多层感知机开始尝试一下。
```python
class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)

# 接下来，我们将模型的参数存储在一个叫做“mlp.params”的文件中。
net.save_weights('mlp.params')

# 为了恢复模型，我们实例化了原始多层感知机模型的一个备份。这里我们不需要随机初始化模型参数，而是直接读取文件中存储的参数。
clone = MLP()
clone.load_weights('mlp.params')

# <tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7fda848f2a30>

# 由于两个实例具有相同的模型参数，在输入相同的X时，两个实例的计算结果应该相同。让我们来验证一下。
Y_clone = clone(X)
Y_clone == Y

# <tf.Tensor: shape=(2, 10), dtype=bool, numpy=
# array([[ True,  True,  True,  True,  True,  True,  True,  True,  True,True],
#        [ True,  True,  True,  True,  True,  True,  True,  True,  True,True]])>
```
##### 总结

`save`和`load`函数可用于张量对象的文件读写。我们可以通过参数字典保存和加载网络的全部参数。保存架构必须在代码中完成，而不是在参数中完成。

#### GPU

我们将讨论如何利用这种计算性能进行研究。首先是如何使用单个`GPU`，然后是如何使用多个`GPU`和多个服务器（具有多个`GPU`）。我们先看看如何使用单个`NVIDIA GPU`进行计算。首先，确保至少安装了一个`NVIDIA GPU`。然后，下载`NVIDIA`驱动和`CUDA`并按照提示设置适当的路径。当这些准备工作完成，就可以使用`nvidia-smi`命令来查看显卡信息。
```bash
!nvidia-smi

Fri Aug 18 06:58:40 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.161.03   Driver Version: 470.161.03   CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:1B.0 Off |                    0 |
| N/A   41C    P0    54W / 300W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  Tesla V100-SXM2...  Off  | 00000000:00:1C.0 Off |                    0 |
| N/A   42C    P0    53W / 300W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   2  Tesla V100-SXM2...  Off  | 00000000:00:1D.0 Off |                    0 |
| N/A   41C    P0    57W / 300W |      0MiB / 16160MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   3  Tesla V100-SXM2...  Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   43C    P0    59W / 300W |      0MiB / 16160MiB |      2%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
要运行此部分中的程序，至少需要两个`GPU`。
##### 计算设备

我们可以指定用于存储和计算的设备，如`CPU`和`GPU`。默认情况下，张量是在内存中创建的，然后使用`CPU`计算它。
```python
import tensorflow as tf

tf.device('/CPU:0'), tf.device('/GPU:0'), tf.device('/GPU:1')

# (<tensorflow.python.eager.context._EagerDeviceContext at 0x7fee482b4f00>,
#  <tensorflow.python.eager.context._EagerDeviceContext at 0x7fee482b49c0>,
#  <tensorflow.python.eager.context._EagerDeviceContext at 0x7fee482b48c0>)

# 我们可以查询可用gpu的数量。
len(tf.config.experimental.list_physical_devices('GPU'))

# 2

# 现在我们定义了两个方便的函数，这两个函数允许我们在不存在所需所有GPU的情况下运行代码。
def try_gpu(i=0):  
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if len(tf.config.experimental.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus(): 
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    num_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

try_gpu(), try_gpu(10), try_all_gpus()

# (<tensorflow.python.eager.context._EagerDeviceContext at 0x7fecce154d00>,
#  <tensorflow.python.eager.context._EagerDeviceContext at 0x7fecce14e5c0>,
#  [<tensorflow.python.eager.context._EagerDeviceContext at 0x7fecce152080>,
#   <tensorflow.python.eager.context._EagerDeviceContext at 0x7fecce152100>])
```
##### 张量与GPU

我们可以查询张量所在的设备。默认情况下，张量是在CPU上创建的。
```python
x = tf.constant([1, 2, 3])
x.device

# '/job:localhost/replica:0/task:0/device:GPU:0'

# 需要注意的是，无论何时我们要对多个项进行操作，它们都必须在同一个设备上。 
# 例如，如果我们对两个张量求和，我们需要确保两个张量都位于同一个设备上，否则框架将不知道在哪里存储结果，甚至不知道在哪里执行计算。
```
###### 存储在GPU上

有几种方法可以在`GPU上`存储张量。例如，我们可以在创建张量时指定存储设备。接下来，我们在第一个`gpu`上创建张量变量`X`。在`GPU`上创建的张量只消耗这个`GPU`的显存。我们可以使用`nvidia-smi`命令查看显存使用情况。一般来说，我们需要确保不创建超过`GPU`显存限制的数据。
```python
with try_gpu():
    X = tf.ones((2, 3))
X

# <tf.Tensor: shape=(2, 3), dtype=float32, numpy=array([[1., 1., 1.],[1., 1., 1.]], dtype=float32)>

# 假设我们至少有两个GPU，下面的代码将在第二个GPU上创建一个随机张量。
with try_gpu(1):
    Y = tf.random.uniform((2, 3))
Y

# <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
# array([[0.7405896 , 0.82039356, 0.7482799 ],[0.4989698 , 0.53144634, 0.8621385 ]], dtype=float32)>
```
###### 复制

如果我们要计算`X + Y`，我们需要决定在哪里执行这个操作。例如，如下图所示，我们可以将`X`传输到第二个`GPU`并在那里执行操作。不要简单地`X`加上`Y`，因为这会导致异常，运行时引擎不知道该怎么做：它在同一设备上找不到数据会导致失败。由于`Y`位于第二个`GPU`上，所以我们需要将`X`移到那里，然后才能执行相加运算。
{% asset_img dp_2.png "复制数据以在同一设备上执行操作" %}

```python
with try_gpu(1):
    Z = X
print(X)
print(Z)

# tf.Tensor([[1. 1. 1.][1. 1. 1.]], shape=(2, 3), dtype=float32)
# tf.Tensor([[1. 1. 1.][1. 1. 1.]], shape=(2, 3), dtype=float32)

# 现在数据在同一个GPU上（Z和Y都在），我们可以将它们相加。
Y + Z

# <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
# array([[1.7405896, 1.8203936, 1.7482799],[1.4989698, 1.5314463, 1.8621385]], dtype=float32)>

# 假设变量Z已经存在于第二个GPU上。如果我们仍然在同一个设备作用域下调用Z2=Z会发生什么？它将返回Z，而不会复制并分配新内存。
with try_gpu(1):
    Z2 = Z
Z2 is Z

# True
```
###### 旁注

人们使用`GPU`来进行机器学习，因为单个`GPU`相对运行速度快。但是在设备（`CPU、GPU`和其他机器）之间传输数据比计算慢得多。这也使得并行化变得更加困难，因为我们必须等待数据被发送（或者接收），然后才能继续进行更多的操作。这就是为什么拷贝操作要格外小心。根据经验，多个小操作比一个大操作糟糕得多。此外，一次执行几个操作比代码中散布的许多单个操作要好得多。如果一个设备必须等待另一个设备才能执行其他操作，那么这样的操作可能会阻塞。这有点像排队订购咖啡，而不像通过电话预先订购：当客人到店的时候，咖啡已经准备好了。最后，当我们打印张量或将张量转换为`NumPy`格式时，如果数据不在内存中，框架会首先将其复制到内存中，这会导致额外的传输开销。更糟糕的是，它现在受制于全局解释器锁，使得一切都得等待`Python`完成。
##### 神经网络与GPU

类似地，神经网络模型可以指定设备。下面的代码将模型参数放在`GPU`上。
```python
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    net = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])

# 当输入为GPU上的张量时，模型将在同一GPU上计算结果。
net(X)

# <tf.Tensor: shape=(2, 1), dtype=float32, numpy=array([[-1.1429136],[-1.1429136]], dtype=float32)>

让我们确认模型参数存储在同一个GPU上。
net.layers[0].weights[0].device, net.layers[0].weights[1].device

# ('/job:localhost/replica:0/task:0/device:GPU:0','/job:localhost/replica:0/task:0/device:GPU:0')
```
总之，只要所有的数据和参数都在同一个设备上，我们就可以有效地学习模型。
##### 总结

我们可以指定用于存储和计算的设备，例如`CPU`或`GPU`。默认情况下，数据在主内存中创建，然后使用`CPU`进行计算。深度学习框架要求计算的所有输入数据都在同一设备上，无论是`CPU`还是`GPU`。不经意地移动数据可能会显著降低性能。一个典型的错误如下：计算`GPU`上每个小批量的损失，并在命令行中将其报告给用户（或将其记录在`NumPy ndarray`中）时，将触发全局解释器锁，从而使所有`GPU`阻塞。最好是为`GPU`内部的日志分配内存，并且只移动较大的日志。