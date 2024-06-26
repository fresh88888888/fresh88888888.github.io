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

##### 命令式编程

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
`Python`是一种解释型语言(`interpreted language`)。因此，当对上面的`fancy_func`函数求值时，它按顺序执行函数体的操作。也就是说，它将通过对`e = add(a, b)`求值，并将结果存储为变量`e`，从而更改程序的状态。接下来的两个语句`f = add(c, d)`和`g = add(e, f)`也将执行类似地操作，即执行加法计算并将结果存储为变量。下图说明了数据流。
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

要了解混合式编程的工作原理，最简单的方法是考虑具有多层的深层网络。按照惯例，`Python`解释器需要执行所有层的代码来生成一条指令，然后将该指令转发到`CPU`或`GPU`。对于单个的（快速的）计算设备，这不会导致任何重大问题。另一方面，如果我们使用先进的`8-GPU`服务器，`Python`将很难让所有的`GPU`都保持忙碌。在这里，瓶颈是单线程的`Python`解释器。让我们看看如何通过将`Sequential`替换为`HybridSequential`来解决代码中这个瓶颈。首先，我们定义一个简单的多层感知机。
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
##### 总结

深度学习框架可以将`Python`前端的控制与后端的执行解耦，使得命令可以快速地异步插入后端、并行执行。异步产生了一个相当灵活的前端，但请注意：过度填充任务队列可能会导致内存消耗过多。建议对每个小批量进行同步，以保持前端和后端大致同步。芯片供应商提供了复杂的性能分析工具，以获得对深度学习效率更精确的洞察。

#### 自动并行

深度学习框架会在后端自动构建计算图。利用计算图，系统可以了解所有依赖关系，并且可以选择性地并行执行多个不相互依赖的任务以提高速度。通常情况下单个操作符将使用所有`CPU`或单个`GPU`上的所有计算资源。例如，即使在一台机器上有多个`CPU`处理器，`dot`操作符也将使用所有`CPU`上的所有核心（和线程）。这样的行为同样适用于单个`GPU`。因此，并行化对单设备计算机来说并不是很有用，而并行化对于多个设备就很重要了。虽然并行化通常应用在多个`GPU`之间，但增加本地`CPU`以后还将提高少许性能。例如，则把结合`GPU`和`CPU`的训练应用到计算机视觉模型中。借助自动并行化框架的便利性，我们可以依靠几行`Python`代码实现相同的目标。
{% asset_img cp_2.png "在一个CPU和两个GPU上的两层的多层感知机的计算图及其依赖关系" %}

##### 总结 

现代系统拥有多种设备，如多个`GPU`和多个`CPU`，还可以并行地、异步地使用它们。现代系统还拥有各种通信资源，如`PCI Express`、存储（通常是固态硬盘或网络存储）和网络带宽，为了达到最高效率可以并行使用它们。后端可以通过自动化地并行计算和通信来提高性能。

#### 硬件

很好地理解算法和模型才可以捕获统计方面的问题，构建出具有出色性能的系统。同时，至少对底层硬件有一定的了解也是必不可少的。一个好的设计可以很容易地在性能上造就数量级的差异，这也是后续产生的能够训练网络（例如，训练时间为`1`周）和无法训练网络（训练时间为`3`个月，导致错过截止期）之间的差异。我们先从计算机的研究开始。然后深入查看`CPU`和`GPU`。最后，再查看数据中心或云中的多台计算机的连接方式。
{% asset_img cp_3.png "每种计算的延迟时间" %}

##### 计算机

大多数深度学习研究者和实践者都可以使用一台具有相当数量的内存、计算资源、某种形式的加速器（如一个或者多个`GPU`）的计算机。计算机由以下关键部件组成：
- 一个处理器（也被称为`CPU`），它除了能够运行操作系统和许多其他功能之外，还能够执行给定的程序。它通常由`8`个或更多个核心组成；
- 内存（随机访问存储，`RAM`）用于存储和检索计算结果，如权重向量和激活参数，以及训练数据；
- 一个或多个以太网连接，速度从`1GB/s`到`100GB/s`不等。在高端服务器上可能用到更高级的互连；
- 高速扩展总线（`PCIe`）用于系统连接一个或多个`GPU`。服务器最多有8个加速卡，通常以更高级的拓扑方式连接，而桌面系统则有`1`个或`2`个加速卡，具体取决于用户的预算和电源负载的大小；
- 持久性存储设备，如磁盘驱动器、固态驱动器，在许多情况下使用高速扩展总线连接。它为系统需要的训练数据和中间检查点需要的存储提供了足够的传输速度。

{% asset_img cp_4.png "计算机组件的连接" %}

如上图所示，高速扩展总线由直接连接到`CPU`的多个通道组成，将`CPU`与大多数组件（网络、`GPU`和存储）连接在一起。例如，`AMD`的`Threadripper3`有`64`个`PCIe4.0`通道，每个通道都能够双向传输`16Gbit/s`的数据。内存直接连接到`CPU`，总带宽高达`100GB/s`。当我们在计算机上运行代码时，需要将数据转移到处理器上（`CPU`或`GPU`）执行计算，然后将结果从处理器移回到随机访问存储和持久存储器中。因此，为了获得良好的性能，需要确保每一步工作都能无缝链接，而不希望系统中的任何一部分成为主要的瓶颈。例如，如果不能快速加载图像，那么处理器就无事可做。同样地，如果不能快速移动矩阵到`CPU`（或`GPU`）上，那么`CPU`（或`GPU`）就会无法全速运行。最后，如果希望在网络上同步多台计算机，那么网络就不应该拖累计算速度。一种选择是通信和计算交错进行。接下来将详细地介绍各个组件。
##### 内存

最基本的内存主要用于存储需要随时访问的数据。目前，`CPU`的内存通常为`DDR4`类型，每个模块提供`20-25Gb/s`的带宽。每个模块都有一条`64`位宽的总线。通常使用成对的内存模块来允许多个通道。`CPU`有`2`到`4`个内存通道，也就是说，它们内存带宽的峰值在`40GB/s`到`100GB/s`之间。一般每个通道有两个物理存储体(`bank`)。例如`AMD`的`Zen 3 Threadripper`有`8`个插槽。虽然这些数字令人印象深刻，但实际上它们只能说明了一部分故事。当我们想要从内存中读取一部分内容时，需要先告诉内存模块在哪里可以找到信息。也就是说，我们需要先将地址(`address`)发送到`RAM`。然后我们可以选择只读取一条`64`位记录还是一长串记录。后者称为突发读取(`burst read`)。概括地说，向内存发送地址并设置传输大约需要`100ns`（细节取决于所用内存芯片的特定定时系数），每个后续传输只需要`0.2ns`。总之，第一次读取的成本是后续读取的`500`倍！请注意，每秒最多可以执行一千万次随机读取。这说明应该尽可能地避免随机内存访问，而是使用突发模式读取和写入。

当考虑到拥有多个物理存储体时，事情就更加复杂了。每个存储体大部分时候都可以独立地读取内存。这意味着两件事。一方面，如果随机读操作均匀分布在内存中，那么有效的随机读操作次数将高达`4`倍。这也意味着执行随机读取仍然不是一个好主意，因为突发读取的速度也快了`4`倍。另一方面，由于内存对齐是`64`位边界，因此最好将任何数据结构与相同的边界对齐。当设置了适当的标志时，编译器基本上就是自动化地执行对齐操作。`GPU`内存的带宽要求甚至更高，因为它们的处理单元比`CPU`多得多。总的来说，解决这些问题有两种选择。首先是使内存总线变得更宽。例如，`NVIDIA`的`RTX 2080Ti`有一条`352`位宽的总线。这样就可以同时传输更多的信息。其次，`GPU`使用特定的高性能内存。消费级设备，如`NVIDIA`的`RTX`和`Titan`系列，通常使用`GDDR6`模块。它们使用截然不同的接口，直接与专用硅片上的`GPU`连接。这使得它们非常昂贵，通常仅限于高端服务器芯片，如`NVIDIA Volta V100`系列加速卡。毫不意外的是`GPU`的内存通常比`CPU`的内存小得多，因为前者的成本更高。就目的而言，它们的性能与特征大体上是相似的，只是`GPU`的速度更快。
##### 存储器

随机访问存储的一些关键特性是带宽(`bandwidth`)和延迟(`latency`)。存储设备也是如此，只是不同设备之间的特性差异可能更大。
###### 硬盘驱动器

硬盘驱动器(`hard disk drive，HDD`)已经使用了半个多世纪。简单的说，它们包含许多旋转的盘片，这些盘片的磁头可以放置在任何给定的磁道上进行读写。高端磁盘在`9`个盘片上可容纳高达`16TB`的容量。硬盘的主要优点之一是相对便宜，而它们的众多缺点之一是典型的灾难性故障模式和相对较高的读取延迟。要理解后者，请了解一个事实即硬盘驱动器的转速大约为`7200RPM`（每分钟转数）。它们如果转速再快些，就会由于施加在碟片上的离心力而破碎。在访问磁盘上的特定扇区时，还有一个关键问题：需要等待碟片旋转到位（可以移动磁头，但是无法对磁盘加速）。因此，可能需要`8`毫秒才能使用请求的数据。一种常见的描述方式是，硬盘驱动器可以以大约`100IOPs`（每秒输入/输出操作）的速度工作，并且在过去二十年中这个数字基本上没变。同样糟糕的是，带宽（大约为`100-200MB/s`）也很难增加。毕竟，每个磁头读取一个磁道的比特，因此比特率只随信息密度的平方根缩放。因此，对于非常大的数据集，`HDD`正迅速降级为归档存储和低级存储。
###### 固态驱动器

固态驱动器(`solid state drives，SSD`)使用闪存持久地存储信息。这允许更快地访问存储的记录。现代的固态驱动器的`IOPs`可以达到`10`万到`50`万，比硬盘驱动器快`3`个数量级。而且，它们的带宽可以达到`1-3GB/s`，比硬盘驱动器快一个数量级。这些改进听起来好的难以置信，而事实上受固态驱动器的设计方式，它仍然存在下面的附加条件。
- 固态驱动器以块的方式（`256KB`或更大）存储信息。块只能作为一个整体来写入，因此需要耗费大量的时间，导致固态驱动器在按位随机写入时性能非常差。而且通常数据写入需要大量的时间还因为块必须被读取、擦除，然后再重新写入新的信息。如今固态驱动器的控制器和固件已经开发出了缓解这种情况的算法。尽管有了算法，写入速度仍然会比读取慢得多，特别是对于`QLC`（四层单元）固态驱动器。提高性能的关键是维护操作的“队列”，在队列中尽可能地优先读取和写入大的块。
- 固态驱动器中的存储单元磨损得比较快（通常在几千次写入之后就已经老化了）。磨损程度保护算法能够将退化平摊到许多单元。也就是说，不建议将固态驱动器用于交换分区文件或大型日志文件。
- 最后，带宽的大幅增加迫使计算机设计者将固态驱动器与`PCIe`总线相连接，这种驱动器称为`NVMe`（非易失性内存增强），其最多可以使用`4`个`PCIe`通道。在`PCIe4.0`上最高可达`8GB/s`。

###### 云存储

云存储提供了一系列可配置的性能。也就是说，虚拟机的存储在数量和速度上都能根据用户需要进行动态分配。建议用户在延迟太高时（例如，在训练期间存在许多小记录时）增加`IOPs`的配置数。
##### CPU

中央处理器(`central processing unit，CPU`)是任何计算机的核心。它们由许多关键组件组成：处理器核心(`processor cores`)用于执行机器代码的；总线(`bus`)用于连接不同组件（注意，总线会因为处理器型号、各代产品和供应商之间的特定拓扑结构有明显不同）；缓存(`cach`)相比主内存实现更高的读取带宽和更低的延迟内存访问。最后，因为高性能线性代数和卷积运算常见于媒体处理和机器学习中，所以几乎所有的现代`CPU`都包含向量处理单元(`vector processing unit`)为这些计算提供辅助。
{% asset_img cp_5.png "Intel Skylake消费级四核CPU" %}

上图描述了`Intel Skylake`消费级四核`CPU`。它包含一个集成`GPU`、缓存和一个连接四个核心的环总线。例如，以太网、WiFi、蓝牙、`SSD`控制器和`USB`这些外围设备要么是芯片组的一部分，要么通过`PCIe`直接连接到`CPU`。
###### 微体系结构

每个处理器核心都由一组相当复杂的组件组成。虽然不同时代的产品和供应商的细节有所不同，但基本功能都是标准的。前端加载指令并尝试预测将采用哪条路径（例如，为了控制流），然后将指令从汇编代码解码为微指令。汇编代码通常不是处理器执行的最低级别代码，而复杂的微指令却可以被解码成一组更低级的操作，然后由实际的执行核心处理。通常执行核心能够同时执行许多操作，例如，下图的`ARM Cortex A77`核心可以同时执行多达`8`个操作。
{% asset_img cp_6.png "ARM Cortex A77微体系结构" %}

这意味着高效的程序可以在每个时钟周期内执行多条指令，前提是这些指令可以独立执行。不是所有的处理单元都是平等的。一些专用于处理整数指令，而另一些则针对浮点性能进行了优化。为了提高吞吐量，处理器还可以在分支指令中同时执行多条代码路径，然后丢弃未选择分支的结果。这就是为什么前端的分支预测单元很重要，因为只有最有希望的路径才会被继续执行。
###### 矢量化

深度学习的计算量非常大。因此，为了满足机器学习的需要，`CPU`需要在一个时钟周期内执行许多操作。这种执行方式是通过向量处理单元实现的。这些处理单元有不同的名称:在`ARM`上叫做`NEON`,在`x86`上被称为`AVX2`。一个常见的功能是它们能够执行**单指令多数据**(`single instruction multiple data，SIMD`)操作。下图显示了如何在ARM上的一个时钟周期中完成`8`个整数加法。
{% asset_img cp_7.png "128位NEON矢量化" %}

根据体系结构的选择，此类寄存器最长可达512位，最多可组合64对数字。例如，我们可能会将两个数字相乘，然后与第三个数字相加，这也称为**乘加融合**(`fused multiply-add`)。`Intel`的`OpenVino`就是使用这些处理器来获得可观的吞吐量，以便在服务器级`CPU`上进行深度学习。不过请注意，这个数字与`GPU`的能力相比则相形见绌。例如，`NVIDIA`的`RTX 2080Ti`拥有`4352个CUDA`核心，每个核心都能够在任何时候处理这样的操作。
###### 缓存

考虑以下情况：我们有一个中等规模的`4`核心的`CPU`，运行在`2GHz`频率。此外，假设向量处理单元启用了`256`位带宽的`AVX2`，其`IPC`（指令/时钟）计数为`1`。进一步假设从内存中获取用于`AVX2`操作的指令至少需要一个寄存器。这意味着`CPU`每个时钟周期需要消耗`4 x 256 bit = 128 bytes`的数据。除非我们能够每秒向处理器传输{% mathjax %}2\times 10^9\times 128 = 256\times 10^9{% endmathjax %}字节，否则用于处理的数据将会不足。不幸的是，这种芯片的存储器接口仅支持`20-40Gb/s`的数据传输，即少了一个数量级。解决方法是尽可能避免从内存中加载新数据，而是将数据放在`CPU`的缓存上。这就是使用缓存的地方。通常使用以下名称或概念。
- 寄存器，严格来说不是缓存的一部分，用于帮助组织指令。也就是说，寄存器是CPU可以以时钟速度访问而没有延迟的存储位置。`CPU`有几十个寄存器，因此有效地使用寄存器取决于编译器（或程序员）。例如，`C`语言有一个`register`关键字。
- 一级缓存是应对高内存带宽要求的第一道防线。一级缓存很小（常见的大小可能是`32-64KB`），内容通常分为数据和指令。当数据在一级缓存中被找到时，其访问速度非常快，如果没有在那里找到，搜索将沿着缓存层次结构向下寻找。
- 二级缓存是下一站。根据架构设计和处理器大小的不同，它们可能是独占的也可能是共享的。即它们可能只能由给定的核心访问，或者在多个核心之间共享。二级缓存比一级缓存大（通常每个核心`256-512KB`），而速度也更慢。此外，我们首先需要检查以确定数据不在一级缓存中，才会访问二级缓存中的内容，这会增加少量的额外延迟。
- 三级缓存在多个核之间共享，并且可以非常大。`AMD`的`EPYC 3`服务器的`CPU`在多个芯片上拥有高达`256MB`的高速缓存。更常见的数字在`4-8MB`范围内。

预测下一步需要哪个存储设备是优化芯片设计的关键参数之一。例如，建议以向前的方向遍历内存，因为大多数缓存算法将试图**向前读取**(`read forward`)而不是向后读取。同样，将内存访问模式保持在本地也是提高性能的一个好方法。

添加缓存是一把双刃剑。一方面，它能确保处理器核心不缺乏数据。但同时，它也增加了芯片尺寸，消耗了原本可以用来提高处理能力的面积。此外，缓存未命中的代价可能会很昂贵。考虑最坏的情况，如下图所示的**错误共享**(`false sharing`)。当处理器`1`上的线程请求数据时，内存位置缓存在处理器`0`上。为了满足获取需要，处理器`0`需要停止它正在做的事情，将信息写回主内存，然后让处理器`1`从内存中读取它。在此操作期间，两个处理器都需要等待。与高效的单处理器实现相比，这种代码在多个处理器上运行的速度可能要慢得多。这就是为什么缓存大小（除了物理大小之外）有实际限制的另一个原因。
{% asset_img cp_8.png "错误共享" %}
##### GPU和其他加速卡

毫不夸张地说，如果没有`GPU`，深度学习就不会成功。基于同样的原因，有理由认为`GPU`制造商的财富由于深度学习而显著增加。这种硬件和算法的协同进化导致了这样一种情况：无论好坏，深度学习都是更可取的统计建模范式。因此，了解`GPU`和其他加速卡（如`TPU`）的具体好处是值得的。值得注意的是，在实践中经常会有这样一个判别：加速卡是为训练还是推断而优化的。对于后者，我们只需要计算网络中的前向传播。而反向传播不需要存储中间数据。还有，我们可能不需要非常精确的计算（`FP16`或`INT8`通常就足够了）。对于前者，即训练过程中需要存储所有的中间结果用来计算梯度。而且，累积梯度也需要更高的精度，以避免数值下溢（或溢出）。这意味着最低要求也是`FP16`（或`FP16`与`FP32`的混合精度）。所有这些都需要更快、更大的内存（`HBM2`或者`GDDR6`）和更高的处理能力。例如，`NVIDIA`优化了`Turing T4 GPU`用于推断和`V100 GPU`用于训练。
{% asset_img cp_9.png "NVIDIA Turing处理块（图片由英伟达提供）" %}

回想一下矢量化。处理器核心中添加向量处理单元可以显著提高吞吐量。例如，在矢量化的例子中，我们能够同时执行`16`个操作。首先，如果我们添加的运算不仅优化了向量运算，而且优化了矩阵运算，会有什么好处？稍后我们将讨论基于这个策略引入的张量核(`tensor cores`)。第二，如果我们增加更多的核心呢？简而言之，以上就是`GPU`设计决策中的两种策略。下图给出了基本处理块的概述。它包含`16`个整数单位和`16`个浮点单位。除此之外，两个张量核加速了与深度学习相关的附加操作的狭窄的子集。每个流式多处理器都由这样的四个块组成。
{% asset_img cp_10.png "NVIDIA Turing架构（图片由英伟达提供）" %}

接下来，将`12`个流式多处理器分组为图形处理集群，这些集群构成了高端`TU102`处理器。充足的内存通道和二级缓存完善了配置。下图有相关的细节。设计这种设备的原因之一是可以根据需要独立地添加或删除模块，从而满足设计更紧凑的芯片和处理良品率问题（故障模块可能无法激活）的需要。幸运的是，在`CUDA`和框架代码层之下，这类设备的编程对深度学习的临时研究员隐藏得很好。特别是，只要有可用的资源`GPU`上就可以同时执行多个程序。尽管如此，了解设备的局限性是值得的，以避免对应的设备内存的型号不合适。
{% asset_img cp_11.jpeg "NVIDIA Turing架构中的张量核心（图片由英伟达提供）" %}

最后值得一提的是张量核(`tensor core`)。它们是最近增加更多优化电路趋势的一个例子，这些优化电路对深度学习特别有效。例如，`TPU`添加了用于快速矩阵乘法的脉动阵列，这种设计是为了支持非常小数量（第一代`TPU`支持数量为`1`）的大型操作。而张量核是另一个极端。它们针对{% mathjax %}4\times 4{% endmathjax %}和{% mathjax %}16\times 16{% endmathjax %}矩阵之间的小型运算进行了优化，具体取决于它们的数值精度。下图给出了优化的概述。

显然，我们最终会在优化计算时做出某些妥协。其中之一是`GPU`不太擅长处理稀疏数据和中断。尽管有一些明显的例外，如`Gunrock`，但`GPU`**擅长的高带宽突发读取操作并不适合稀疏的矩阵和向量的访问模式**。访问稀疏数据和处理中断这两个目标是一个积极研究的领域。例如：`DGL`，一个专为图深度学习而设计的库。
##### 网络和总线

每当单个设备不足以进行优化时，我们就需要来回传输数据以实现同步处理，于是网络和总线就派上了用场。我们有许多设计参数：带宽、成本、距离和灵活性。应用的末端有`WiFi`，它有非常好的使用范围，非常容易使用（毕竟没有线缆），而且还便宜，但它提供的带宽和延迟相对一般。头脑正常的机器学习研究人员都不会用它来构建服务器集群。
- `PCIe`，一种专用总线，用于每个通道点到点连接的高带宽需求（在`16`通道插槽中的`PCIe4.0`上高达`32GB/s`），延迟时间为个位数的微秒（`5μs`）。`PCIe`链接非常宝贵。处理器拥有的数量：`AMD`的`EPYC 3`有`128`个通道，`Intel`的`Xeon`每个芯片有`48`个通道；在桌面级`CPU`上，数字分别是`20（Ryzen9）`和`17（Core i9）`。由于`GPU`通常有`16`个通道，这就限制了以全带宽与`CPU`连接的`GPU`数量。毕竟，它们还需要与其他高带宽外围设备（如存储和以太网）共享链路。与`RAM`访问一样，由于减少了数据包的开销，因此更适合大批量数据传输。
- 以太网，连接计算机最常用的方式。虽然它比`PCIe`慢得多，但它的安装成本非常低，而且具有很强的弹性，覆盖的距离也要长得多。低级服务器的典型带宽为`1GBit/s`。高端设备（如云中的`C5`实例。这进一步增加了开销。与`PCIe`类似，以太网旨在连接两个设备，例如计算机和交换机。
- 交换机，一种连接多个设备的方式，该连接方式下的任何一对设备都可以同时执行（通常是全带宽）点对点连接。例如，以太网交换机可能以高带宽连接`40`台服务器。请注意，交换机并不是传统计算机网络所独有的。甚至`PCIe`通道也可以是可交换的，例如：`P2`实例就是将大量`GPU`连接到主机处理器。
- `NVLink`，是`PCIe`的替代品，适用于非常高带宽的互连。它为每条链路提供高达`300Gbit/s`的数据传输速率。服务器`GPU（Volta V100）`有六个链路。而消费级`GPU（RTX 2080Ti）`只有一个链路，运行速度也降低到`100Gbit/s`。建议使用`NCCL`来实现`GPU`之间的高速数据传输。
##### 总结

设备有运行开销。因此，数据传输要争取量大次少而不是量少次多。这适用于`RAM`、固态驱动器、网络和`GPU`。矢量化是性能的关键。确保充分了解加速器的特定功能。例如，一些`Intel Xeon CPU`特别适用于`INT8`操作，`NVIDIA Volta GPU`擅长`FP16`矩阵操作，`NVIDIA Turing`擅长`FP16、INT8`和`INT4`操作。在训练过程中数据类型过小导致的数值溢出可能是个问题（在推断过程中则影响不大）。数据混叠现象会导致严重的性能退化。`64`位`CPU`应该按照`64`位边界进行内存对齐。在`GPU`上建议保持卷积大小对齐，例如：与张量核对齐。将算法与硬件相匹配（例如，内存占用和带宽）。将命中参数装入缓存后，可以实现很大数量级的加速比。在验证实验结果之前，建议先在纸上勾勒出新算法的性能。关注的原因是数量级及以上的差异。使用调试器跟踪调试寻找性能的瓶颈。训练硬件和推断硬件在性能和价格方面有不同的优点。

#### 多GPU训练

##### 问题拆分

我们从一个简单的计算机视觉问题和一个稍稍过时的网络开始。这个网络有多个卷积层和汇聚层，最后可能有几个全连接的层，看起来非常类似于`LeNet`或`AlexNet`。假设我们有多个`GPU`。我们希望以一种方式对训练进行拆分，为实现良好的加速比，还能同时受益于简单且可重复的设计选择。毕竟，多个`GPU`同时增加了内存和计算能力。简而言之，对于需要分类的小批量训练数据，我们有以下选择。

第一种方法，在多个`GPU`之间拆分网络。也就是说，每个`GPU`将流入特定层的数据作为输入，跨多个后续层对数据进行处理，然后将数据发送到下一个`GPU`。与单个`GPU`所能处理的数据相比，我们可以用更大的网络处理数据。此外，每个`GPU`占用的**显存**(`memory footprint`)可以得到很好的控制，虽然它只是整个网络显存的一小部分。然而，`GPU`的接口之间需要的密集同步可能是很难办的，特别是层之间计算的工作负载不能正确匹配的时候，还有层之间的接口需要大量的数据传输的时候（例如：激活值和梯度，数据量可能会超出`GPU`总线的带宽）。此外，计算密集型操作的顺序对拆分来说也是非常重要的，其本质仍然是一个困难的问题，目前还不清楚研究是否能在特定问题上实现良好的线性缩放。综上所述，除非存框架或操作系统本身支持将多个`GPU`连接在一起，否则不建议这种方法。

第二种方法，拆分层内的工作。例如，将问题分散到`4`个`GPU`，每个`GPU`生成`16`个通道的数据，而不是在单个`GPU`上计算`64`个通道。对于全连接的层，同样可以拆分输出单元的数量。下图描述了这种设计，其策略用于处理显存非常小（当时为`2GB`）的`GPU`。当通道或单元的数量不太小时，使计算性能有良好的提升。此外，由于可用的显存呈线性扩展，多个`GPU`能够处理不断变大的网络。
{% asset_img cp_12.png "由于GPU显存有限，原有AlexNet设计中的模型并行" %}

然而，我们需要大量的同步或**屏障操作**(`barrier operation`)，因为每一层都依赖于所有其他层的结果。此外，需要传输的数据量也可能比跨`GPU`拆分层时还要大。因此，基于带宽的成本和复杂性，我们同样不推荐这种方法。

最后一种方法，跨多个`GPU`对数据进行拆分。 这种方式下，所有`GPU`尽管有不同的观测结果，但是执行着相同类型的工作。在完成每个小批量数据的训练之后，梯度在`GPU`上聚合。这种方法最简单，并可以应用于任何情况，同步只需要在每个小批量数据处理之后进行。也就是说，当其他梯度参数仍在计算时，完成计算的梯度参数就可以开始交换。而且，`GPU`的数量越多，小批量包含的数据量就越大，从而就能提高训练效率。但是，添加更多的`GPU`并不能让我们训练更大的模型。
{% asset_img cp_13.png "在多个GPU上并行化。从左到右：原始问题、网络并行、分层并行、数据并行" %}

上图中比较了多个`GPU`上不同的并行方式。总体而言，只要`GPU`的显存足够大，数据并行是最方便的。在深度学习的早期，`GPU`的显存曾经是一个棘手的问题，然而如今除了非常特殊的情况，这个问题已经解决。下面我们将重点讨论数据并行性。
##### 数据并行性

假设一台机器有{% mathjax %}k{% endmathjax %}个`GPU`。给定需要训练的模型，虽然每个GPU上的参数值都是相同且同步的，但是每个`GPU`都将独立地维护一组完整的模型参数。例如，下图演示了在{% mathjax %}k=2{% endmathjax %}时基于数据并行方法训练模型。
{% asset_img cp_14.png "利用两个GPU上的数据，并行计算小批量随机梯度下降" %}

一般来说，{% mathjax %}k{% endmathjax %}个`GPU`并行训练过程如下：
- 在任何一次训练迭代中，给定的随机的小批量样本都将被分成{% mathjax %}k{% endmathjax %}个部分，并均匀地分配到`GPU`上；
- 每个`GPU`根据分配给它的小批量子集，计算模型参数的损失和梯度；
- 将{% mathjax %}k{% endmathjax %}个`GPU`中的局部梯度聚合，以获得当前小批量的随机梯度；
- 聚合梯度被重新分发到每个`GPU`中；
- 每个`GPU`使用这个小批量随机梯度，来更新它所维护的完整的模型参数集。

在实践中请注意，当在{% mathjax %}k{% endmathjax %}个`GPU`上训练时，需要扩大小批量的大小为{% mathjax %}k{% endmathjax %}的倍数，这样每个`GPU`都有相同的工作量，就像只在单个`GPU`上训练一样。因此，在`16-GPU`服务器上可以显著地增加小批量数据量的大小，同时可能还需要相应地提高学习率。
##### 总结

有多种方法可以在多个`GPU`上拆分深度网络的训练。拆分可以在层之间、跨层或跨数据上实现。前两者需要对数据传输过程进行严格编排，而最后一种则是最简单的策略。数据并行训练本身是不复杂的，它通过增加有效的小批量数据量的大小提高了训练效率。在数据并行中，数据需要跨多个`GPU`拆分，其中每个`GPU`执行自己的前向传播和反向传播，随后所有的梯度被聚合为一，之后聚合结果向所有的`GPU`广播。小批量数据量更大时，学习率也需要稍微提高一些。

#### 参数服务器

当我们从一个`GPU`迁移到多个`GPU`时，以及再迁移到包含多个`GPU`的多个服务器时（可能所有服务器的分布跨越了多个机架和多个网络交换机），分布式并行训练算法也需要变得更加复杂。通过细节可以知道，一方面是不同的互连方式的带宽存在极大的区别（例如，`NVLink`可以通过设置实现跨`6`条链路的高达`100GB/s`的带宽，`16`通道的`PCIe4.0`提供`32GB/s`的带宽，而即使是高速`100GbE`以太网也只能提供大约`10GB/s`的带宽）；另一方面是期望开发者既能完成统计学习建模还精通系统和网络也是不切实际的。参数服务器的核心思想首先是在分布式隐变量模型的背景下引入的。
##### 数据并行训练

让我们回顾一下在分布式架构中数据并行的训练方法。由于当今的`GPU`拥有大量的显存，因此在实际场景中（不包括图深度学习）只有数据并行这种并行训练策略值得推荐。下图中描述了实现的数据并行的变体。其中的关键是梯度的聚合需要在单个`GPU`(`GPU 0`)上完成，然后再将更新后的参数广播给所有`GPU`。
{% asset_img cp_15.png "左图是单GPU训练；右图是多GPU训练的一个变体" %}

选择`GPU 0`进行聚合似乎是个很随便的决定，当然也可以选择`CPU`上聚合，事实上只要优化算法支持，在实际操作中甚至可以在某个`GPU`上聚合其中一些参数，而在另一个`GPU`上聚合另一些参数。例如，如果有四个与参数向量相关的梯度{% mathjax %}\mathbf{g}_1,\ldots,\mathbf{g}_4{% endmathjax %}，还可以一个`GPU`对一个{% mathjax %}\mathbf{g}_i(i=1,\ldots,4){% endmathjax %}）地进行梯度聚合。
{% asset_img cp_16.png "一个4路GPU服务器" %}

为了便于讨论，我们假设所有梯度共需`160MB`。在这种情况下，将其中`3`个`GPU`的梯度发送到第`4`个`GPU`上需要`30`毫秒（每次传输需要`10`毫秒`=160MB/16GB/s`）。再加上`30`毫秒将权重向量传输回来，得到的结果是总共需要`60`毫秒。如果将所有的数据发送到`CPU`，总共需要`80`毫秒，其中将有`40`毫秒的惩罚，因为`4`个`GPU`每个都需要将数据发送到`CPU`。最后，假设能够将梯度分为`4`个部分，每个部分为`40MB`，现在可以在不同的`GPU`上同时聚合每个部分。因为`PCIe`交换机在所有链路之间提供全带宽操作，所以传输需要`2.5 x 3 = 7.5`毫秒，而不是`30`毫秒，因此同步操作总共需要`15`毫秒。简而言之，一样的参数同步操作基于不同的策略时间可能在`15`毫秒到`80`毫秒之间。下图描述了交换参数的不同策略。
{% asset_img cp_17.png "参数同步策略" %}

请注意，我们还可以使用另一个工具来改善性能：在深度网络中，从顶部到底部计算所有梯度需要一些时间，因此即使还在忙着为某些参数计算梯度时，就可以开始为准备好的参数同步梯度了。

##### 环同步（Ring Synchronization）

当谈及现代深度学习硬件的同步问题时，我们经常会遇到大量的定制的网络连接。每个`GPU`通过`PCIe`链路连接到主机`CPU`，该链路最多只能以`16GB/s`的速度运行。此外，每个`GPU`还具有`6`个`NVLink`连接，每个`NVLink`连接都能够以`300Gbit/s`进行双向传输。这相当于每个链路每个方向约{% mathjax %}300\div 8\div 2\approx 18\mathbf{GB}/s{% endmathjax %}。简言之，聚合的`NVLink`带宽明显高于`PCIe`带宽，问题是如何有效地使用它。
{% asset_img cp_18.png "在8台V100 GPU服务器上连接NVLink（图片由英伟达提供）" %}

的研究结果表明最优的同步策略是将网络分解成两个环，并基于两个环直接同步数据。下图描述了网络可以分解为一个具有双`NVLink`带宽的环（`1-2-3-4-5-6-7-8-1`）和一个具有常规带宽的环（`1-4-6-3-5-8-2-7-1`）。在这种情况下，设计一个高效的同步协议是非常重要的。
{% asset_img cp_19.png "将NVLink网络分解为两个环" %}

考虑下面的思维试验：给定由{% mathjax %}n{% endmathjax %}个计算节点（或`GPU`）组成的一个环，梯度可以从第一个节点发送到第二个节点，在第二个结点将本地的梯度与传送的梯度相加并发送到第三个节点，依此类推。在{% mathjax %}n-1{% endmathjax %}步之后，可以在最后访问的节点中找到聚合梯度。也就是说，聚合梯度的时间随节点数线性增长。但如果照此操作，算法是相当低效的。归根结底，在任何时候都只有一个节点在通信。如果我们将梯度分为{% mathjax %}n{% endmathjax %}个块，并从节点{% mathjax %}i{% endmathjax %}开始同步块{% mathjax %}i{% endmathjax %}，会怎么样？因为每个块的大小是{% mathjax %}1/n{% endmathjax %}，所以总时间现在是{% mathjax %}(n-1)/n\approx 1{% endmathjax %}。换句话说，当我们增大环的大小时，聚合梯度所花费的时间不会增加。这是一个相当惊人的结果。下图说明了{% mathjax %}n=4{% endmathjax %}个节点上的步骤顺序。
{% asset_img cp_20.png "将NVLink网络分解为两个环" %}

如果我们使用相同的例子，跨`8`个`V100 GPU`同步`160MB`，我们得到的结果大约是{% mathjax %}2\times 160MB\div (3\times 18\mathbf{GB}/s)\approx 6ms{% endmathjax %}。这比使用`PCIe`总线要好，即使我们现在使用的是`8`个`GPU`，注意到有一种常见的误解认为环同步与其他同步算法在本质上是不同的，实际上与简单的树算法相比其唯一的区别是同步路径稍微精细一些。
##### 多机训练

新的挑战出现在多台机器上进行分布式训练：我们需要服务器之间相互通信，而这些服务器又只通过相对较低的带宽结构连接，在某些情况下这种连接的速度可能会慢一个数量级，因此跨设备同步是个棘手的问题。毕竟，在不同机器上运行训练代码的速度会有细微的差别，因此如果想使用分布式优化的同步算法就需要**同步**(`synchronize`)这些机器。下图说明了分布式并行训练是如何发生的。
- 在每台机器上读取一组（不同的）批量数据，在多个`GPU`之间分割数据并传输到`GPU`的显存中。基于每个`GPU`上的批量数据分别计算预测和梯度。
- 来自一台机器上的所有的本地`GPU`的梯度聚合在一个`GPU`上（或者在不同的`GPU`上聚合梯度的某些部分）。
- 每台机器的梯度被发送到其本地`CPU`中。
- 所有的`CPU`将梯度发送到中央参数服务器中，由该服务器聚合所有梯度。
- 然后使用聚合后的梯度来更新参数，并将更新后的参数广播回各个`CPU`中。
- 更新后的参数信息发送到本地一个（或多个）`GPU`中。
- 所有`GPU`上的参数更新完成。

{% asset_img cp_21.png "多机多GPU分布式并行训练" %}

以上这些操作似乎都相当简单，而且事实上它们可以在一台机器内高效地执行，但是当我们考虑多台机器时，就会发现中央的参数服务器成为了瓶颈。毕竟，每个服务器的带宽是有限的，因此对{% mathjax %}m{% endmathjax %}个工作节点来说，将所有梯度发送到服务器所需的时间是{% mathjax %}\mathcal{O}(m){% endmathjax %}。我们也可以通过将参数服务器数量增加到{% mathjax %}n{% endmathjax %}来突破这一障碍。此时，每个服务器只需要存储{% mathjax %}\mathcal{O}(1/n){% endmathjax %}个参数，因此更新和优化的总时间变为{% mathjax %}\mathcal{O}(m/n){% endmathjax %}。这两个数字的匹配会产生稳定的伸缩性，而不用在乎我们需要处理多少工作节点。在实际应用中，我们使用同一台机器既作为工作节点还作为服务器。设计说明请参考下图。特别是，确保多台机器只在没有不合理延迟的情况下工作是相当困难的。
{% asset_img cp_22.png "上图：单参数服务器是一个瓶颈，因为它的带宽是有限的；下图：多参数服务器使用聚合带宽存储部分参数" %}

##### 键值存储

在实践中，实现分布式多`GPU`训练所需要的步骤绝非易事。这就是公共抽象值得使用的原因，**公共抽象**即重新定义具有更新语义的**键－值存储**(`key-value store`)的抽象。在许多工作节点和许多`GPU`中，梯度{% mathjax %}i{% endmathjax %}的计算可以定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{g}_i = \sum_{k\in \text{workers}}\sum_{j\in \text{GPUs}} \mathbf{g}_{ijk}
{% endmathjax %}
其中{% mathjax %}\mathbf{g}_{ijk}{% endmathjax %}是在工作节点{% mathjax %}k{% endmathjax %}的GPU{% mathjax %}j{% endmathjax %}上拆分的梯度{% mathjax %}i{% endmathjax %}的一部分。这个运算的关键在于它是一个**交换归约**(`commutative reduction`)，也就是说，它把许多向量变换成一个向量，而运算顺序在完成向量变换时并不重要。这对实现我们的目标来说是非常好的，因为不需要为何时接收哪个梯度进行细粒度的控制。此外，请注意，这个操作在不同的{% mathjax %}i{% endmathjax %}之间是独立的。这就允许我们定义下面两个操作：`push`（用于累积梯度）和`pull`（用于取得聚合梯度）。因为我们有很多层，也就有很多不同的梯度集合，因此需要用一个键{% mathjax %} {% endmathjax %}来对梯度建索引。这个与`Dynamo`中引入的键－值存储之间存在相似性并非巧合。它们两个定义都拥有许多相似的性质，特别是在多个服务器之间分发参数时。

键－值存储的`push`与`pull`操作描述如下：
- `push(key，value)`将特定的梯度值从工作节点发送到公共存储，在那里通过某种方式（例如，相加）来聚合值；
- `pull(key，value)`从公共存储中取得某种方式（例如，组合来自所有工作节点的梯度）的聚合值。

通过将同步的所有复杂性隐藏在一个简单的`push`和`pull`操作背后，我们可以将统计建模人员（他们希望能够用简单的术语表达优化）和系统工程师（他们需要处理分布式同步中固有的复杂性）的关注点解耦。
##### 总结

同步需要高度适应特定的网络基础设施和服务器内的连接，这种适应会严重影响同步所需的时间。环同步对于`p3`和`DGX-2`服务器是最佳的，而对于其他服务器则未必。当添加多个参数服务器以增加带宽时，分层同步策略可以工作的很好。