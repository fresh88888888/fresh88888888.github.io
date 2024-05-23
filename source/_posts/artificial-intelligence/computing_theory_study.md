---
title: 计算性能 (机器学习)(TensorFlow)
date: 2024-05-23 09:00:11
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

#### 编译器和解释器

**命令式编程**(`imperative programming`)。命令式编程使用诸如`print、“+”`和`if`之类的语句来更改程序的状态。考虑下面这段简单的命令式程序：
<!-- more -->
```python
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

print(fancy_func(1, 2, 3, 4))

# 10
```
`Python`是一种解释型语言(interpreted language)。因此，当对上面的`fancy_func`函数求值时，它按顺序执行函数体的操作。也就是说，它将通过对`e = add(a, b)`求值，并将结果存储为变量`e`，从而更改程序的状态。接下来的两个语句`f = add(c, d)`和`g = add(e, f)`也将执行类似地操作，即执行加法计算并将结果存储为变量。下图说明了数据流。
{% asset_img cp_1.png "命令式编程中的数据流" %}

尽管命令式编程很方便，但可能效率不高。一方面原因，`Python`会单独执行这三个函数的调用，而没有考虑`add`函数在`fancy_func`中被重复调用。如果在一个`GPU`（甚至多个`GPU`）上执行这些命令，那么`Python`解释器产生的开销可能会非常大。此外，它需要保存`e`和`f`的变量值，直到`fancy_func`中的所有语句都执行完毕。这是因为程序不知道在执行语句`e = add(a, b)`和`f = add(c, d)`之后，其他部分是否会使用变量`e`和`f`。
##### 符号式编程

考虑另一种选择**符号式编程**(`symbolic programming`)，即代码通常只在完全定义了过程之后才执行计算。这个策略被多个深度学习框架使用，包括`Theano`和`TensorFlow`（后者已经获得了命令式编程的扩展）。一般包括以下步骤：
- 定义计算流程。
- 将流程编译成可执行的程序。
- 给定输入，调用编译好的程序执行。

这将允许进行大量的优化。首先，在大多数情况下，我们可以跳过`Python`解释器。从而消除因为多个更快的`GPU`与单个`CPU`上的单个`Python`线程搭配使用时产生的性能瓶颈。其次，编译器可以将上述代码优化和重写为`print((1 + 2) + (3 + 4))`甚至`print(10)`。因为编译器在将其转换为机器指令之前可以看到完整的代码，所以这种优化是可以实现的。例如，只要某个变量不再需要，编译器就可以释放内存（或者从不分配内存），或者将代码转换为一个完全等价的片段。下面，我们将通过模拟命令式编程来进一步了解符号式编程的概念。

命令式（解释型）编程和符号式编程的区别如下：
- 命令式编程更容易使用。在`Python`中，命令式编程的大部分代码都是简单易懂的。命令式编程也更容易调试，这是因为无论是获取和打印所有的中间变量值，或者使用`Python`的内置调试工具都更加简单；
- 符号式编程运行效率更高，更易于移植。符号式编程更容易在编译期间优化代码，同时还能够将程序移植到与`Python`无关的格式中，从而允许程序在非`Python`环境中运行，避免了任何潜在的与`Python`解释器相关的性能问题。

##### 混合式编程

历史上，大部分深度学习框架都在命令式编程与符号式编程之间进行选择。例如，`Theano、TensorFlow`（灵感来自前者）、`Keras`和`CNTK`采用了符号式编程。相反地，`Chainer`和`PyTorch`采取了命令式编程。在后来的版本更新中，`TensorFlow2.0`和`Keras`增加了命令式编程。

要了解混合式编程的工作原理，最简单的方法是考虑具有多层的深层网络。按照惯例，`Python`解释器需要执行所有层的代码来生成一条指令，然后将该指令转发到`CPU`或`GPU`。对于单个的（快速的）计算设备，这不会导致任何重大问题。另一方面，如果我们使用先进的`8-GP`U服务器，`Python`将很难让所有的GPU都保持忙碌。在这里，瓶颈是单线程的`Python`解释器。让我们看看如何通过将`Sequential`替换为`HybridSequential`来解决代码中这个瓶颈。首先，我们定义一个简单的多层感知机。
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense

# 生产网络的工厂模式
def get_net():
    net = tf.keras.Sequential()
    net.add(Dense(256, input_shape = (512,), activation = "relu"))
    net.add(Dense(128, activation = "relu"))
    net.add(Dense(2, activation = "linear"))
    return net

x = tf.random.normal([1,512])
net = get_net()
net(x)

# <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[ 0.9541333 , -0.74289465]], dtype=float32)>

net = tf.function(net)
net(x)

# <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[ 0.9541333 , -0.74289465]], dtype=float32)>
```
一开始，`TensorFlow`中构建的所有函数都是作为计算图构建的，因此默认情况下是`JIT`编译的。但是，随着`TensorFlow2.X`和`EargeTensor`的发布，计算图就不再是默认行为。我们可以使用`tf.function`重新启用这个功能。`tf.function`更常被用作函数装饰器，它也可以直接将其作为普通的`Python`函数调用。模型的计算结果保持不变。

我们编写与之前相同的代码，再使用`tf.function`简单地转换模型，当完成这些任务后，网络将以`TensorFlow`的`MLIR`中间表示形式构建为一个计算图，并在编译器级别进行大量优化以满足快速执行的需要（我们将在下面对性能进行基准测试）。通过将`jit_compile = True`标志添加到`tf.function()`的函数调用中可以显式地启用`TensorFlow`中的`XLA`（线性代数加速）功能。在某些情况下，`XLA`可以进一步优化JIT的编译代码。如果没有这种显式定义，图形模式将会被启用，但是`XLA`可以使某些大规模的线性代数的运算速度更快（与我们在深度学习程序中看到的操作类似），特别是在`GPU`环境中。
```python
import time

class Benchmark:
    """用于测量运行时间"""
    def __init__(self, description='Done'):
        self.description = description

    def __enter__(self):
        self.t1 = time.clock()
        return self

    def __exit__(self, *args):
        self.t2 = time.clock()
        print(f'{self.description}: {(self.t2 - self.t1):.4f} sec')

# 现在我们可以调用网络三次，一次使用eager模式，一次是使用图模式。
net = get_net()
with Benchmark('Eager模式'):
    for i in range(1000): net(x)

net = tf.function(net)
with Benchmark('Graph模式'):
    for i in range(1000): net(x)

# Eager模式: 1.2769 sec
# Graph模式: 0.5811 sec
```
如以上结果所示，在`tf.keras.Sequential`的实例被函数`tf.function`脚本化后，通过使用`TensorFlow`中的图模式执行方式实现的符号式编程提高了计算性能。

##### 序列化

编译模型的好处之一是我们可以将模型及其参数序列化（保存）到磁盘。这允许这些训练好的模型部署到其他设备上，并且还能方便地使用其他前端编程语言。同时，通常编译模型的代码执行速度也比命令式编程更快。在`TensorFlow`中保存模型的底层`API`是`tf.saved_model`，让我们来看看`saved_model`的运行情况。
```bash
net = get_net()
tf.saved_model.save(net, 'my_mlp')
!ls -lh my_mlp*
```
输出结果为：
```bash
INFO:tensorflow:Assets written to: my_mlp/assets
total 64K
drwxr-xr-x 2 ci ci   6 Aug 18 07:38 assets
-rw-r--r-- 1 ci ci 64K Aug 18 07:38 saved_model.pb
drwxr-xr-x 2 ci ci  66 Aug 18 07:38 variables
```
##### 总结

命令式编程使得新模型的设计变得容易，因为可以依据控制流编写代码，并拥有相对成熟的`Python`软件生态。符号式编程要求我们先定义并且编译程序，然后再执行程序，其好处是提高了计算性能。

#### 异步计算

今天的计算机是高度并行的系统，由多个`CPU`核、多个`GPU`、多个处理单元组成。通常每个`CPU`核有多个线程，每个设备通常有多个`GPU`，每个`GPU`有多个处理单元。总之，我们可以同时处理许多不同的事情，并且通常是在不同的设备上。不幸的是，`Python`并不善于编写并行和异步代码，至少在没有额外帮助的情况下不是好选择。归根结底，`Python`是单线程的，将来也是不太可能改变的。因此在诸多的深度学习框架中，`TensorFlow`则采用了一种**异步编程**(`asynchronous programming`)模型来提高性能，而`PyTorch`则使用了`Python`自己的调度器来实现不同的性能权衡。对`PyTorch`来说`GPU`操作在默认情况下是异步的。当调用一个使用`GPU`的函数时，操作会排队到特定的设备上，但不一定要等到以后才执行。这允许我们并行执行更多的计算，包括在`CPU`或其他`GPU`上的操作。