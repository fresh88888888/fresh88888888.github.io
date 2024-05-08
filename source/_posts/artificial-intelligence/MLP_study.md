---
title: 多层感知机 (非线性神经网络) (PyTorch)
date: 2024-05-08 12:00:11
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

#### 隐藏层

回想一下`softmax`回归的模型架构。该模型通过单个仿射变换将我们的输入直接映射到输出，然后进行`softmax`操作。如果我们的标签通过仿射变换后确实与我们的输入数据相关，那么这种方法确实足够了。但是，仿射变换中的线性是一个很强的假设。例如，线性意味着单调假设：任何特征的增大都会导致模型输出的增大（如果对应的权重为正），或者导致模型输出的减小（如果对应的权重为负）。例如，如果我们试图预测一个人是否会偿还贷款。我们可以认为，在其它条件不变的情况下，收入较高的申请人比收入较低的申请人更有可能偿还贷款。但是，虽然收入与还款概率存在单调性，但是它们不是线性相关的。收入从0增加到5万，可能比从100万增加到105万带来更大的还款可能性。处理这一问题的一种方法是对我们的数据进行预处理，使线性变得更合理，如使用收入的对数作为我们的特征。然而我们可以很容易找出违反单调性的例子。
<!-- more -->
例如，我们想要根据体温预测死亡率。对于体温高于37摄氏度的人来说，温度越高风险越大。然而对于体温低于37摄氏度的人来说，温度越高风险就越低。在这种情况下，我们也可以通过一些巧妙的预处理来解决问题。例如，我们可以使用与37摄氏度的距离作为特征。但是，如何对猫和狗的图像进行分类呢？增加位置`(13,17)`处像素的强度是否总是增加（或降低）图像描绘狗的似然？对线性模型的依赖对应于一个隐含的假设，即区分猫和狗的唯一要求是评估单个像素的强度。在一个倒置图像后依然保留类别的世界里，这种方法注定会失败。与我们前面的例子相比，这里的线性很荒谬，而且我们难以通过简单的预处理来解决这个问题。这是因为任何像素的重要性都以复杂的方式取决于该像素的上下文（周围像素的值）。我们的数据可能会有一种表示，这种表示会考虑到我们在特征之间的相关交互作用。在此表示的基础上建立一个线性模型可能会是合适的，但我们不知道如何手动计算这么一种表示。对于深度神经网络，我们使用观测数据来联合学习隐藏层表示和应用于该表示的线性预测器。
##### 网络架构

我们可以通过在网络中加入一个或多个隐藏层来克服线性模型的限制，使其能够处理更普遍的函数关系类型。要做到这一点，最简单的方法是将许多全连接层堆叠在一起，每一层都输出到上面的层，知道生成最后的输出。我们可以把钱{% mathjax %}L-1{% endmathjax %}层看做表示，把最后一层看作是线性预测器。这种架构通常称为**多层感知机**(`multilayer perceptron`)，通常缩写为`MLP`。下面，我们以图的方式描述了多层感知机：
{% asset_img mlp_1.png "一个单隐藏层的多层感知机，具有5个隐藏单元" %}

这个多层感知机有4个输入，3个输出，其隐藏层包含5个隐藏单元。输入层不涉及任何计算，因此使用此网络产生输出只需要实现隐藏层和输出层的计算。因此，这个多层感知机中的层数为2.注意，这两个层都是全连接的。每个输入都会影响隐藏层中的每个神经元，而隐藏层中的每个神经元又会影响输出层中的每个神经元。具有全连接层的多层感知机的参数开销可能会高的令人望而却步。即使不改变输入或输出大小的情况下，可能在参数节约和模型有效性之间进行平衡。

同之前一样，我们通过矩阵{% mathjax %}\mathbf{X}\in \mathbb{R}^{n\times d}{% endmathjax %}来表示{% mathjax %}n{% endmathjax %}个样本的小批量，其中每个样本具有{% mathjax %}d{% endmathjax %}个输入特征。对于具有{% mathjax %}h{% endmathjax %}个隐藏单元的但隐藏层多层感知机，用{% mathjax %}\mathbf{H}\in \mathbb{R}^{n\times h}{% endmathjax %}表示隐藏层的输出，称为隐藏表示(`hidden representations`)。在数学或代码中，{% mathjax %}\mathbf{H}{% endmathjax %}也被称为隐藏层变量(`hidden-layer variable`)因为隐藏层和输出层都是全连接的，所以我们有隐藏层权重{% mathjax %}\mathbf{W}^{(1)}\in \mathbb{R}^{d\times h}{% endmathjax %}和隐藏层偏置{% mathjax %}\mathbf{b}^{(1)}\in \mathbb{R}^{1\times h}{% endmathjax %}以及输出层权重{% mathjax %}\mathbf{W}^{(2)}\in \mathbb{R}^{1\times q}{% endmathjax %}和输出层偏置{% mathjax %}\mathbf{b}^{(2)}\in \mathbb{R}^{1\times q}{% endmathjax %}。在形式上，我们按如下方式计算但隐藏层多层感知机的输出{% mathjax %}\mathbf{O}\in \mathbb{R}^{n\times q}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \mathbf{H} = \mathbf{XW}^{(1)} + \mathbf{b}^{(1)} \\
& \mathbf{O} = \mathbf{HW}^{(2)} + \mathbf{b}^{(2)} \\
\end{align}
{% endmathjax %}
注意在添加隐藏层之后，模型现在需要跟踪和更新额外的参数。可我们能从中得到什么好处呢？在上面定义的模型里，我们没有好处！原因很简单：上面的隐藏单元由输入的仿射函数给出，而输出（`softmax`操作前）只是隐藏单元的仿射函数。仿射函数的仿射函数本身就是仿射函数，但是我们之前的线性模型已经能够表示任何仿射函数。我们可以证明这一等价性，即对于任意权重值，我们只需合并隐藏层，便可产生具有参数{% mathjax %}\mathbf{W} = \mathbf{W}^{(1)}\mathbf{W}^{(2)}{% endmathjax %}和{% mathjax %}\mathbf{b}=\mathbf{b}^{(1)} \mathbf{W}^{(2)} + \mathbf{b}^{(2)}{% endmathjax %}的等价单层模型。
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{O} = (\mathbf{XW}^{(1)}+b^{(1)})\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{XW}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)} = \mathbf{XW} + \mathbf{b}
{% endmathjax %}
为了f发挥多层架构的潜力，我们还需要一个额外的关键要素：在仿射变换之后对每个隐藏单元应用非线性激活函数(`activation function`){% mathjax %}\sigma{% endmathjax %}。激活函数的输出（例如，{% mathjax %}\sigma(\cdot){% endmathjax %}）被称为活性值(`activations`)。一般来说，有了激活函数，就不可能再将我们的多层感知机退化成线性模型：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \mathbf{H} = \sigma(\mathbf{XW}^{(1)} + \mathbf{b}^{(1)}) \\
& \mathbf{O} = \mathbf{HW}^{(2)} + \mathbf{b}^{(2)} \\
\end{align}
{% endmathjax %}
由于{% mathjax %}\mathbf{X}{% endmathjax %}中的每一行对应于小批量中的每一个样本，出于记号习惯的考量，我们定义非线性函数{% mathjax %}\sigma{% endmathjax %}也以按行的方式作用于其输入，记一次计算一个样本。以相同的方式使用`softmax`符号来表示按行操作。但是应用于隐藏层的激活函数通常不仅按行操作，也按元素操作。这意味着，在计算每一层的线性部分之后，我们可以计算每个活性值，而不需要查看其他隐藏单元所取的值。对于大多数激活函数都是这样。为了构建更通用的多层感知机，我们可以继续堆叠这样的隐藏层，例如：{% mathjax %}\mathbf{H}^{(1)} = \sigma_1(\mathbf{XW}^{(1)} + \mathbf{b}^{(1)}){% endmathjax %}和{% mathjax %}\mathbf{H}^{(2)} = \sigma_2(\mathbf{H}^{(1)}\mathbf{W}^{(2)} + \mathbf{b}^{(2)}){% endmathjax %}一层叠一层，从而产生更有表达能力的模型。

多层感知机可以通过隐藏神经元，捕捉到输入之间复杂的相互作用，这些神经元依赖于每个输入的值。我们可以很容易地设计隐藏节点来执行任意计算。例如，在一对输入上进行基本逻辑操作，多层感知机是通用近似器。即使是网络只有一个隐藏层，给定足够的神经元和正确的权重，我们可以对任意函数建模，尽管实际中学习该函数是很困难的。神经网络有点像`C`语言。`C`语言和任何其他现代编程语言一样，能够表达任何可计算的程序。但实际上，想出一个符合规范的程序才是最困难的部分。而且，虽然一个单隐层网络能学习任何函数，但并不意味着我们应该尝试使用单隐藏层网络来解决所有问题。事实上，通过使用更深（而不是更广）的网络，我们可以更容易地逼近许多函数。

##### 激活函数

激活函数(`activation function`)通过计算加权和并加上偏置来确定神经元是否应该被激活，它们将输入信号转换为输出的可微运算。大多数激活函数都是非线性的。由于激活函数是深度学习的基础，下面简要介绍一些常见的激活函数。

###### ReLU函数

最受欢迎的激活函数是修正线性单元（Rectified linear unit，ReLU），因为它实现简单，同时在各种预测任务中表现良好。ReLU提供了一种非常简单的非线性变换。给定元素{% mathjax %}x{% endmathjax %}，ReLU函数被定义为该元素与{% mathjax %}0{% endmathjax %}的最大值：
{% mathjax '{"conversion":{"em":14}}' %}
ReLU(x) = \text{max}(x,0)
{% endmathjax %}
通俗地说，`ReLU`函数通过将相应的活性值设为`0`，仅保留正元素并丢弃所有负元素。为了直观感受一下，我们可以画出函数的曲线图。正如从图中所看到，激活函数是分段线性的。
```python
x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.relu(x)
plt.plot(x.detach(), y.detach(), 'x', 'relu(x)', figsize=(5, 2.5))
```
{% asset_img mlp_2.png %}

当输入为负时，`ReLU`函数的导数为`0`，而当输入为正时，`ReLU`函数的导数为`1`。 注意，当输入值精确等于`0`时，`ReLU`函数不可导。在此时，我们默认使用左侧的导数，即当输入为`0`时导数为`0`。我们可以忽略这种情况，因为输入可能永远都不会是`0`。这里引用一句古老的谚语，“如果微妙的边界条件很重要，我们很可能是在研究数学而非工程”，这个观点正好适用于这里。下面我们绘制`ReLU`函数的导数。
```python
y.backward(torch.ones_like(x), retain_graph=True)
plt.plot(x.detach(), x.grad, 'x', 'grad of relu', figsize=(5, 2.5))
```
{% asset_img mlp_3.png %}

使用`ReLU`的原因是，它求导表现得特别好：要么让参数消失，要么让参数通过。这使得优化表现得更好，并且`ReLU`减轻了困扰以往神经网络的梯度消失问题。注意，`ReLU`函数有许多变体，包括参数化`ReLU（Parameterized ReLU，pReLU）`函数，该变体为ReLU添加了一个线性项，因此即使参数是负的，某些信息仍然可以通过：
{% mathjax '{"conversion":{"em":14}}' %}
p\mathbf{ReLU}(x) = \text{max}(x,0) + \alpha\text{min}(0,x)
{% endmathjax %}
###### sigmoid函数

对于一个定义域在{% mathjax %}\mathbb{R}{% endmathjax %}中的输入，`sigmoid`函数将输入变换为区间(0, 1)上的输出。因此，sigmoid通常称为挤压函数（`squashing function`）：它将范围(`-inf, inf`)中的任意输入压缩到区间(`0, 1`)中的某个值：
{% mathjax '{"conversion":{"em":14}}' %}
\text{sigmoid}(x) = \frac{1}{1+\text{exp}(-x)}
{% endmathjax %}
在最早的神经网络中，科学家们感兴趣的是对“激发”或“不激发”的生物神经元进行建模。因此，这一领域的先驱可以一直追溯到人工神经元的发明者麦卡洛克和皮茨，他们专注于阈值单元。阈值单元在其输入低于某个阈值时取值`0`，当输入超过阈值时取值`1`。当人们逐渐关注到到基于梯度的学习时，`sigmoid`函数是一个自然的选择，因为它是一个平滑的、可微的阈值单元近似。当我们想要将输出视作二元分类问题的概率时，`sigmoid`仍然被广泛用作输出单元上的激活函数（`sigmoid`可以视为`softmax`的特例）。然而，`sigmoid`在隐藏层中已经较少使用，它在大部分时候被更简单、更容易训练的`ReLU`所取代。下面，我们绘制`sigmoid`函数。注意，当输入接近0时，`sigmoid`函数接近线性变换。
```python
y = torch.sigmoid(x)
plt.plot(x.detach(), y.detach(), 'x', 'sigmoid(x)', figsize=(5, 2.5))
```
{% asset_img mlp_4.png %}

`sigmoid`函数的导数为下面的公式：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{d}{dx}\text{sigmoid}(x) = \frac{\text{exp}(-x)}{(1 + \text{exp}(-x))^2} = \text{sigmoid}(x)(1-\text{sigmoid}(x))
{% endmathjax %}
`sigmoid`函数的导数图像如下所示。注意，当输入为0时，`sigmoid`函数的导数达到最大值`0.25`；而输入在任一方向上越远离`0`点时，导数越接近`0`。
```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(), x.grad, 'x', 'grad of sigmoid', figsize=(5, 2.5))
```
{% asset_img mlp_5.png %}

###### tanh函数

与`sigmoid`函数类似，`tanh`(双曲正切)函数也能将其输入压缩转换到区间(`-1, 1`)上。`tanh`函数的公式如下：
{% mathjax '{"conversion":{"em":14}}' %}
\text{tanh}(x) = \frac{1-\text{exp}(-2x)}{1+\text{exp}(-2x)}
{% endmathjax %}
下面我们绘制`tanh`函数。注意，当输入在`0`附近时，`tanh`函数接近线性变换。函数的形状类似于`sigmoid`函数，不同的是`tanh`函数关于坐标系原点中心对称。
```python
y = torch.tanh(x)
plt.plot(x.detach(), y.detach(), 'x', 'tanh(x)', figsize=(5, 2.5))
```
{% asset_img mlp_6.png %}

`tanh`函数的导数是：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{d}{dx}\text{tanh}(x) = 1-\text{tanh}^2(x)
{% endmathjax %}
`tanh`函数的导数图像如下所示。当输入接近`0`时，`tanh`函数的导数接近最大值`1`。与我们在`sigmoid`函数图像中看到的类似，输入在任一方向上越远离`0`点，导数越接近`0`。
```python
# 清除以前的梯度
x.grad.data.zero_()
y.backward(torch.ones_like(x),retain_graph=True)
plt.plot(x.detach(), x.grad, 'x', 'grad of tanh', figsize=(5, 2.5))
```
{% asset_img mlp_7.png %}

#### 总结

多层感知机在输出层和输入层之间增加一个或多个全连接隐藏层，并通过激活函数转换隐藏层的输出。常用的激活函数包括`ReLU`函数、`sigmoid`函数和`tanh`函数。