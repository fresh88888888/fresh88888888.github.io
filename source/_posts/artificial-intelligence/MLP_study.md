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

最受欢迎的激活函数是修正线性单元（`Rectified linear unit，ReLU`），因为它实现简单，同时在各种预测任务中表现良好。`ReLU`提供了一种非常简单的非线性变换。给定元素{% mathjax %}x{% endmathjax %}，`ReLU`函数被定义为该元素与{% mathjax %}0{% endmathjax %}的最大值：
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

##### 总结

多层感知机在输出层和输入层之间增加一个或多个全连接隐藏层，并通过激活函数转换隐藏层的输出。常用的激活函数包括`ReLU`函数、`sigmoid`函数和`tanh`函数。

#### 模型选择、欠拟合和过拟合

我们的目标是发现**模式**(`pattern`)。但是，我们如何才能确定模型是真正发现了一种泛化模式，而不是简单的记住了数据呢？例如，我们想要在患者的基因数据与痴呆状态之间寻找模式，其中标签是从集合{痴呆,轻度认知障碍,健康}中提取的。因为基因可以唯一确定每个个体（不考虑双胞胎），所以在这个任务中是有可能记住这个数据集。我们不想让模型只会做这样的事情：“那是鲍勃！我记得他！他有痴呆症！”。原因很简单：当我们将来部署该模型时，模型需要判断从未见过的患者。只有当模型真正发现了一种泛化模式时，才会作出有效的预测。

更正式地说，我们的目标是发现某些模式，这些模式捕捉到了我们训练集潜在总体的规律。如果成功做到了这点，即使是对以前从未遇到过的个体，模型也可以成功地评估风险。如何发现可以泛化的模式是机器学习的根本问题。困难在于，当我们训练模型时，我们只能访问数据中的小部分样本。最大的公开图像数据集包含大约一百万张图像。而在大部分时候，我们只能从数千或数万个数据样本中学习。在大型医院系统中，我们可能会访问数十万份医疗记录。当我们使用有限的样本时，可能会遇到这样的问题：当收集到更多的数据时，会发现之前找到的明显关系并不成立。将模型在训练数据上拟合的比在潜在分布中更接近的现象称为过拟合(`overfitting`)，用于对抗过拟合的技术称为正则化(`regularization`)。

##### 训练误差和泛化误差

为了进一步讨论这一现象，我们需要了解**训练误差**和**泛化误差**。**训练误差**(`training error`)是指，模型在训练数据集上计算得到的误差。**泛化误差**(`generalization error`)是指，模型应用在同样从原始样本的分布中抽取的无限多数据样本时，模型误差的期望。问题是，我们永远不能准确地计算出泛化误差。这是因为无限多的数据样本是一个虚构的对象。在实际中，我们只能通过将模型应用于一个独立的测试集来估计泛化误差，该测试集由随机选取的、未曾在训练集中出现的数据样本构成。

下面的三个思维实验将有助于更好地说明这种情况。假设一个大学生正在努力准备期末考试。一个勤奋的学生会努力做好练习，并利用往年的考试题目来测试自己的能力。尽管如此，在过去的考试题目上取得好成绩并不能保证他会在真正考试时发挥出色。例如，学生可能试图通过死记硬背考题的答案来做准备。他甚至可以完全记住过去考试的答案。另一名学生可能会通过试图理解给出某些答案的原因来做准备。在大多数情况下，后者会考得更好。

类似地，考虑一个简单地使用查表法来回答问题的模型。如果允许的输入集合是离散的并且相当小，那么也许在查看许多训练样本后，该方法将执行得很好。但当这个模型面对从未见过的例子时，它表现的可能比随机猜测好不到哪去。这是因为输入空间太大了，远远不可能记住每一个可能的输入所对应的答案。例如，考虑{% mathjax %}28\times 28{% endmathjax %}的灰度图像。如果每个像素可以取{% mathjax %}256{% endmathjax %}个灰度值中的一个，则有{% mathjax %}256^{784}{% endmathjax %}个可能的图像。这意味着指甲大小的低分辨率灰度图像的数量比宇宙中的原子要多得多。即使我们可能遇到这样的数据，我们也不可能存储整个查找表。

最后，考虑对掷硬币的结果（类别0：正面，类别1：反面）进行分类的问题。假设硬币是公平的，无论我们想出什么算法，泛化误差始终是{% mathjax %}\frac{1}{2}{% endmathjax %}。然而，对于大多数算法，我们应该期望训练误差会更低（取决于运气）。考虑数据集{`0，1，1，1，0，1`}。我们的算法不需要额外的特征，将倾向于总是预测多数类，从我们有限的样本来看，它似乎是1占主流。在这种情况下，总是预测类1的模型将产生{% mathjax %}\frac{1}{3}{% endmathjax %}的误差，这比我们的泛化误差要好得多。当我们逐渐增加数据量，正面比例明显偏离{% mathjax %}\frac{1}{2}{% endmathjax %}的可能性将会降低，我们的训练误差将与泛化误差相匹配。

由于泛化是机器学习中的基本问题，许多数学家和理论家毕生致力于研究描述这一现象的形式理论。在同名定理(`eponymous theorem`)中，格里文科和坎特利推导出了训练误差收敛到泛化误差的速率。在一系列开创性的论文中，`Vapnik`和`Chervonenkis`将这一理论扩展到更一般种类的函数。这项工作为统计学习理论奠定了基础。在我们目前已探讨、并将在之后继续探讨的监督学习情景中，我们假设训练数据和测试数据都是从相同的分布中独立提取的。这通常被称为**独立同分布假设**，这意味着对数据进行采样的过程没有进行“记忆”。换句话说，抽取的第`2`个样本和第`3`个样本的相关性，并不比抽取的第`2`个样本和第`200`万个样本的相关性更强。

有时候我们即使轻微违背独立同分布假设，模型仍将继续运行得非常好。比如，我们有许多有用的工具已经应用于现实，如人脸识别、语音识别和语言翻译。毕竟，几乎所有现实的应用都至少涉及到一些违背独立同分布假设的情况。有些违背独立同分布假设的行为肯定会带来麻烦。比如，我们试图只用来自大学生的人脸数据来训练一个人脸识别系统，然后想要用它来监测疗养院中的老人。这不太可能有效，因为大学生看起来往往与老年人有很大的不同。目前，即使认为独立同分布假设是理所当然的，理解泛化性也是一个困难的问题。此外，能够解释深层神经网络泛化性能的理论基础，也仍在继续困扰着学习理论领域最伟大的学者们。当我们训练模型时，我们试图找到一个能够尽可能拟合训练数据的函数。但是如果它执行地“太好了”，而不能对看不见的数据做到很好泛化，就会导致过拟合。这种情况正是我们想要避免或控制的。深度学习中有许多启发式的技术旨在防止过拟合。

当我们有简单的模型和大量的数据时，我们期望**泛化误差与训练误差相近**。当我们有更复杂的模型和更少的样本时，我们预计训练误差会下降，但泛化误差会增大。模型复杂性由什么构成是一个复杂的问题。一个模型是否能很好地泛化取决于很多因素。例如，具有更多参数的模型可能被认为更复杂，参数有更大取值范围的模型可能更为复杂。通常对于神经网络，我们认为需要更多训练迭代的模型比较复杂，而需要**早停**(`early stopping`)的模型（即较少训练迭代周期）就不那么复杂。

我们很难比较本质上不同大类的模型之间（例如，决策树与神经网络）的复杂性。就目前而言，一条简单的经验法则相当有用：统计学家认为，能够轻松解释任意事实的模型是复杂的，而表达能力有限但仍能很好地解释数据的模型可能更有现实用途。在哲学上，这与波普尔的科学理论的可证伪性标准密切相关：如果一个理论能拟合数据，且有具体的测试可以用来证明它是错误的，那么它就是好的。这一点很重要，因为所有的统计估计都是事后归纳。也就是说，我们在观察事实之后进行估计，因此容易受到相关谬误的影响。目前，我们将把哲学放在一边，坚持更切实的问题。

影响模型泛化的因素：
- 可调整参数的数量。当可调整参数的数量（有时称为自由度）很大时，模型往往更容易过拟合。
- 参数采用的值。当权重的取值范围较大时，模型可能更容易过拟合。
- 训练样本的数量。即使模型很简单，也很容易过拟合只包含一两个样本的数据集。而过拟合一个有数百万个样本的数据集则需要一个极其灵活的模型。

##### 模型选择

在机器学习中，我们通常在评估几个候选模型后选择最终的模型。这个过程叫做模型选择。有时，需要进行比较的模型在本质上是完全不同的（比如，决策树与线性模型）。又有时，我们需要比较不同的超参数设置下的同一类模型。例如，训练多层感知机模型时，我们可能希望比较具有 不同数量的隐藏层、不同数量的隐藏单元以及不同的激活函数组合的模型。为了确定候选模型中的最佳模型，我们通常会使用验证集。

原则上，在我们确定所有的超参数之前，我们不希望用到测试集。如果我们在模型选择过程中使用测试数据，可能会有过拟合测试数据的风险，那就麻烦大了。如果我们过拟合了训练数据，还可以在测试数据上的评估来判断过拟合。但是如果我们过拟合了测试数据，我们又该怎么知道呢？因此，我们决不能依靠测试数据进行模型选择。然而，我们也不能仅仅依靠训练数据来选择模型，因为我们无法估计训练数据的泛化误差。在实际应用中，情况变得更加复杂。虽然理想情况下我们只会使用测试数据一次，以评估最好的模型或比较一些模型效果，但现实是测试数据很少在使用一次后被丢弃。我们很少能有充足的数据来对每一轮实验采用全新测试集。解决此问题的常见做法是将我们的数据分成三份，除了训练和测试数据集之外，还增加一个验证数据集(`validation dataset`)，也叫验证集(`validation set`)。

当训练数据稀缺时，我们甚至可能无法提供足够的数据来构成一个合适的验证集。这个问题的一个流行的解决方案是采用{% mathjax %}K{% endmathjax %}折交叉验证。这里，原始训练数据被分成{% mathjax %}K{% endmathjax %}个不重叠的子集。然后执行{% mathjax %}K{% endmathjax %}次模型训练和验证，每次在{% mathjax %}K-1{% endmathjax %}个子集上进行训练，并在剩余的一个子集（在该轮中没有用于训练的子集）上进行验证。最后，通过对{% mathjax %}K{% endmathjax %}次实验的结果取平均来估计训练和验证误差。

##### 欠拟合还是过拟合?

当我们比较训练和验证误差时，我们要注意两种常见的情况。首先，我们要注意这样的情况：训练误差和验证误差都很严重，但它们之间仅有一点差距。如果模型不能降低训练误差，这可能意味着模型过于简单（即表达能力不足），无法捕获试图学习的模式。此外，由于我们的训练和验证误差之间的泛化误差很小，我们有理由相信可以用一个更复杂的模型降低训练误差。这种现象被称为欠拟合(`underfitting`)。另一方面，当我们的训练误差明显低于验证误差时要小心，这表明严重的过拟合(`overfitting`)。注意，过拟合并不总是一件坏事。特别是在深度学习领域，众所周知，最好的预测模型在训练数据上的表现往往比在保留（验证）数据上好得多。最终，我们通常更关心验证误差，而不是训练误差和验证误差之间的差距。

为了说明一些关于过拟合和模型复杂性的经典直觉，我们给出一个多项式的例子。给定由单个特征{% mathjax %}x{% endmathjax %}和对应实数标签{% mathjax %}y{% endmathjax %}组成的训练数据，我们试图找到下面的{% mathjax %}d{% endmathjax %}阶多项式来估计标签{% mathjax %}y{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\hat{y} = \sum_{i=0}^{d} x^i w_i
{% endmathjax %}
这只是一个线性回归问题，我们的特征是{% mathjax %}x{% endmathjax %}的幂给出的，模型的权重是{% mathjax %}w_i{% endmathjax %}给出的，偏置是{% mathjax %}w_0{% endmathjax %}给出的（因为对于所有的{% mathjax %}x{% endmathjax %}都有{% mathjax %}x^0 = 1{% endmathjax %}）。由于这只是一个线性回归问题，我们可以使用平方误差作为我们的损失函数。

高阶多项式函数比低阶多项式函数复杂得多。高阶多项式的参数较多，模型函数的选择范围较广。因此在固定训练数据集的情况下，高阶多项式函数相对于低阶多项式的训练误差应该始终更低（最坏也是相等）。事实上，当数据样本包含了{% mathjax %}x{% endmathjax %}的不同值时，函数阶数等于数据样本数量的多项式函数可以完美拟合训练集。在下图中直观地描述了多项式的阶数和欠拟合与过拟合之间的关系。
{% asset_img mlp_8.png "模型复杂度对欠拟合和过拟合的影响" %}

另一个重要因素是数据集的大小。训练数据集中的样本越少，我们就越有可能（且更严重地）过拟合。随着训练数据量的增加，泛化误差通常会减小。此外，一般来说，更多的数据不会有什么坏处。对于固定的任务和数据分布，模型复杂性和数据集大小之间通常存在关系。给出更多的数据，我们可能会尝试拟合一个更复杂的模型。能够拟合更复杂的模型可能是有益的。如果没有足够的数据，简单的模型可能更有用。对于许多任务，深度学习只有在有数千个训练样本时才优于线性模型。从一定程度上来说，深度学习目前的生机要归功于廉价存储、互联设备以及数字化经济带来的海量数据集。
##### 多项式回归

###### 生成数据集

给定{% mathjax %}x{% endmathjax %}，我们将使用以下三阶多项式来生成训练和测试数据的标签：
{% mathjax '{"conversion":{"em":14}}' %}
y = 5 + 1.2x- 3.4\frac{x^2}{2!} + 5.6\frac{x^3}{3!} + \epsilon\;\text{where}\;\epsilon \sim\mathcal{N}(0,0.1^2)
{% endmathjax %}
噪声项{% mathjax %}\epsilon{% endmathjax %}服从均值为`0`且标准差为`0.1`正态分布。在优化的过程中，我们通常希望避免非常大的梯度值或损失值。这就是我们将特征从{% mathjax %}x^i{% endmathjax %}调整为{% mathjax %}\frac{x^i}{i!}{% endmathjax %}的原因，这样可以避免很大的{% mathjax %}i{% endmathjax %}带来的特别大的指数值。我们降为训练集和测试集各生成`100`个样本。
```python
import math
import numpy as np
import torch
import pandas pd
from torch import nn

max_degree = 20  # 多项式的最大阶数
n_train, n_test = 100, 100  # 训练和测试数据集大小
true_w = np.zeros(max_degree)  # 分配大量的空间
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

features = np.random.normal(size=(n_train + n_test, 1))
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1)  # gamma(n)=(n-1)!

# labels的维度:(n_train+n_test,)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape)

# NumPy ndarray转换为tensor
true_w, features, poly_features, labels 
        = [torch.tensor(x, dtype = torch.float32) for x in [true_w, features, poly_features, labels]]
features[:2], poly_features[:2, :], labels[:2]

# (tensor([[ 1.6580],[-1.6392]]),
#  tensor([[ 1.0000e+00,  1.6580e+00,  1.3745e+00,  7.5967e-01,  3.1489e-01,
#            1.0442e-01,  2.8855e-02,  6.8346e-03,  1.4165e-03,  2.6096e-04,
#            4.3267e-05,  6.5217e-06,  9.0110e-07,  1.1493e-07,  1.3611e-08,
#            1.5045e-09,  1.5590e-10,  1.5206e-11,  1.4006e-12,  1.2223e-13],
#          [ 1.0000e+00, -1.6392e+00,  1.3435e+00, -7.3408e-01,  3.0082e-01,
#           -9.8622e-02,  2.6944e-02, -6.3094e-03,  1.2928e-03, -2.3546e-04,
#            3.8597e-05, -5.7516e-06,  7.8567e-07, -9.9066e-08,  1.1599e-08,
#           -1.2676e-09,  1.2986e-10, -1.2522e-11,  1.1403e-12, -9.8378e-14]]),
#  tensor([ 6.6262, -5.4505]))
```
###### 对模型进行训练和测试

首先让我们实现一个函数来评估模型在给定数据集上的损失。
```python
def evaluate_loss(net, data_iter, loss):  #@save
    """评估给定数据集上模型的损失"""
    metric = pd.Accumulator(2)  # 损失的总和,样本数量
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def train(train_features, test_features, train_labels, test_labels, num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1]
    # 不设置偏置，因为我们已经在多项式中实现了它
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = pd.load_array((train_features, train_labels.reshape(-1,1)), batch_size)
    test_iter = pd.load_array((test_features, test_labels.reshape(-1,1)), batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    animator = pd.Animator(xlabel='epoch', ylabel='loss', yscale='log',
                            xlim=[1, num_epochs], ylim=[1e-3, 1e2],
                            legend=['train', 'test'])
    for epoch in range(num_epochs):
        pd.train_epoch_ch3(net, train_iter, loss, trainer)
        if epoch == 0 or (epoch + 1) % 20 == 0:
            animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss),
                                     evaluate_loss(net, test_iter, loss)))
    print('weight:', net[0].weight.data.numpy())
```
###### 三阶多项式函数拟合(正常)

我们将首先使用三阶多项式函数，它与数据生成函数的阶数相同。结果表明，该模型能有效降低训练损失和测试损失。学习到的模型参数也接近真实值{% mathjax %}w=[5,1.2,-3.4,5.6]{% endmathjax %}。
```python
# 从多项式特征中选择前4个维度，即1,x,x^2/2!,x^3/3!
train(poly_features[:n_train, :4], poly_features[n_train:, :4], labels[:n_train], labels[n_train:])

# weight: [[ 5.010476   1.2354498 -3.4229028  5.503297 ]]
```
{% asset_img mlp_9.png %}

###### 线性函数拟合(欠拟合)

让我们再看看线性函数拟合，减少该模型的训练损失相对困难。在最后一个迭代周期完成后，训练损失仍然很高。当用来拟合非线性模式（如这里的三阶多项式函数）时，线性模型容易欠拟合。
```python
# 从多项式特征中选择前2个维度，即1和x
train(poly_features[:n_train, :2], poly_features[n_train:, :2], labels[:n_train], labels[n_train:])

# weight: [[3.4049764 3.9939284]]
```
{% asset_img mlp_10.png %}

###### 高阶多项式函数拟合(过拟合)

现在，让我们尝试使用一个阶数过高的多项式来训练模型。在这种情况下，没有足够的数据用于学到高阶系数应该具有接近于零的值。因此，这个过于复杂的模型会轻易受到训练数据中噪声的影响。虽然训练损失可以有效地降低，但测试损失仍然很高。结果表明，复杂模型对数据造成了过拟合。
```python
# 从多项式特征中选取所有维度
train(poly_features[:n_train, :], poly_features[n_train:, :], labels[:n_train], labels[n_train:], num_epochs=1500)

# weight: [[ 4.9849787   1.2896876  -3.2996354   5.145749   -0.34205326  1.2237961
#         0.20393135  0.3027379  -0.20079008 -0.16337848  0.11026663  0.21135856
#         -0.00940325  0.11873583 -0.15114897 -0.05347819  0.17096086  0.1863975
#         -0.09107699 -0.02123026]]
```
{% asset_img mlp_11.png %}

##### 总结

欠拟合是指模型无法继续减少训练误差。过拟合是指训练误差远小于验证误差。由于不能基于训练误差来估计泛化误差，因此简单地最小化训练误差并不一定意味着泛化误差的减小。机器学习模型需要注意防止过拟合，即防止泛化误差过大。验证集可以用于模型选择，但不能过于随意地使用它。我们应该选择一个复杂度适当的模型，避免使用数量不足的训练样本。

#### 权重衰减

回想一下，在多项式回归的例子中，我们可以通过调整拟合多项式的阶数来限制模型的容量。实际上，限制特征的数量是缓解过拟合的一种常用技术。然而，简单的丢弃特征对这项工作来说可能过于生硬。我们继续思考多项式回归的例子，考虑高维输入可能发生的情况。多项式对多变量数据的自然扩展称**单项式**(`monomials`)。也可以说变量幂的乘积，单项式的阶数是幂的和。例如，{% mathjax %}x_1^2x_2 {% endmathjax %}和{% mathjax %}x_3x_5^2{% endmathjax %}都是三次单项式。注意，随着阶数{% mathjax %}d{% endmathjax %}的增长，带有阶数{% mathjax %}d{% endmathjax %}的项数迅速增加。给定{% mathjax %}k{% endmathjax %}个变量，阶数为{% mathjax %}d{% endmathjax %}的项的个数为{% mathjax %}\binom{k-1+d}{k-1}{% endmathjax %}，即{% mathjax %}C_{k-1+d}^{k-1} = \frac{(k-1+d)!}{(d)!(k-1)!}{% endmathjax %}，因此即使是阶数上的微小变化，比如从2到3，也会显著增加我们模型的复杂性。仅仅通过简单的限制特征数量（在多项式回归中体现为限制阶数），可能仍然使模型在过简单和过复杂中徘徊，我们需要一个更细粒度的工具来调整函数的复杂性，使其达到一个合适的平衡位置。在训练参数化机器学习模型时，**权重衰减**(`weight decay`)是最广泛使用的正则化技术之一，它通常也被称为{% mathjax %}L_2{% endmathjax %}正则化。这项技术通过函数与零的距离来衡量函数的复杂度，因为在所有函数{% mathjax %}f{% endmathjax %}中，函数{% mathjax %}f=0{% endmathjax %}(所有输入都得到值为0)在某种意义上是最简单的。但是我们应该如何精确地测量一个函数和零之间的距离呢？没有一个正确的答案。事实上，函数分析和巴拿赫空间理论的研究，都在致力于回答这个问题。

一种简单的方法是通过线性函数{% mathjax %}f(\mathbf{x})=\mathbf{w}^{\mathsf{T}}\mathbf{x}{% endmathjax %}中的权重向量的某个范数来度量其复杂性，例如{% mathjax %}\|\mathbf{w}\|^2{% endmathjax %}。要保证权重向量比较小，最常用的方法是将其范数作为惩罚项加到最小化损失的问题中。将原来的训练目标最小化训练标签上的预测损失， 调整为最小化预测损失和惩罚项之和。 现在，如果我们的权重向量增长的太大， 我们的学习算法可能会更集中于最小化权重范数{% mathjax %}\|\mathbf{w}\|^2{% endmathjax %}，这正是我们想要的。我们的损失由下式给出：
{% mathjax '{"conversion":{"em":14}}' %}
L(\mathbf{w},b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}(\mathbf{w}^{\mathsf{T}}x^{(i)} + b - y^{(i)})^2
{% endmathjax %}
回想一下，{% mathjax %}\mathbf{x}^{(i)}{% endmathjax %}是样本{% mathjax %}i{% endmathjax %}的特征，{% mathjax %}(\mathbf{w},b){% endmathjax %}是权重和偏置参数。为了惩罚权重向量的大小，我们必须以某种方式在损失函数中添加{% mathjax %}\|\mathbf{w}\|^2{% endmathjax %}但是模型应该如何平衡这个新的额外惩罚的损失？ 实际上，我们通过正则化常数{% mathjax %}\lambda{% endmathjax %}来描述这种权衡，这是一个非负超参数，我们使用验证数据拟合：
{% mathjax '{"conversion":{"em":14}}' %}
L(\mathbf{w},b) + \frac{\lambda}{2} \|\mathbf{w}\|^2
{% endmathjax %}
对于{% mathjax %}\lambda = 0{% endmathjax %}，我们恢复了原来的损失函数。对于{% mathjax %}\lambda > 0{% endmathjax %}，我们限制{% mathjax %}\|\mathbf{w}\|{% endmathjax %}的大小。这里我们仍然除以2，当我们去一个二次函数的导数时{% mathjax %}2{% endmathjax %}和{% mathjax %}\frac{1}{2}{% endmathjax %}会抵消。以确保更新表达式看起来既漂亮又简单。为什么在这里我们使用平方范数而不是标准范数（即欧几里得距离）？我们这样做是为了便于计算。通过平方{% mathjax %}L_2{% endmathjax %}范数。我们去掉平方根，留下权重向量每个分量的平方和。这使得惩罚的导数很容易计算：导数的和等于和的导数。

此外，为什么我们首先使用{% mathjax %}L_2{% endmathjax %}范数而不是{% mathjax %}L_1{% endmathjax %}范数。事实上，这个选择在整个统计领域中都是有效的，{% mathjax %}L_2{% endmathjax %}正则化线性模型构成经典的**岭回归算法**(`ridge regression`)，{% mathjax %}L_1{% endmathjax %}正则化线性回归是统计学中类似的基本模型，通常被称为**套索回归**(`lasso regression`)。使用{% mathjax %}L_2{% endmathjax %}范数的一个原因是它对权重向量的大分量施加了巨大的惩罚。这使得我们的学习算法偏向于在大量特征上均匀分布权重的模型。在实践中，这可能使它们对单个变量中的观测误差更为稳定，相比之下，{% mathjax %}L_1{% endmathjax %}惩罚会导致模型降权重集中在一小部分特征上，而将其他权重清除为0。这称为**特征选择**(`feature selection`)，这可能是其他场景下需要的。{% mathjax %}L_2{% endmathjax %}正则化回归的小批量随机梯度下降更新如下式：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{w} \leftarrow (1 - \eta\lambda)\mathbf{w} - \frac{\eta}{|\mathcal{B}|}\sum_{i\in \mathcal{B}} \mathbf{x}^{(i)}(\mathbf{w}^{\mathsf{T}}\mathbf{x}^{(i)} + b - y^{(i)})
{% endmathjax %}
我们根据估计值与观测值之间的差异来更新{% mathjax %}\mathbf{w}{% endmathjax %}。然而，我们同时也在试图将{% mathjax %}\mathbf{w}{% endmathjax %}的大小缩小到0。这就是为什么这种方法被称为**权重衰减**，我们仅考虑惩罚项优化算法在训练的每一步衰减权重。与特征选择相比，权重衰减为我们提供了一种连续的机制来调整函数的复杂度。较小的{% mathjax %}\lambda{% endmathjax %}值对应较少约束的{% mathjax %}\mathbf{w}{% endmathjax %}，而较大的{% mathjax %}\lambda{% endmathjax %}的值对{% mathjax %}\mathbf{w}{% endmathjax %}的约束更大。是否对相应的偏置{% mathjax %}b^2{% endmathjax %}进行惩罚在不同的实践中会有所不同，在神经网络的不同层中也会有所不同。通常，网络输出层的偏置项不会被正则化。

#### 暂退法（Dropout）

我们介绍了通过惩罚权重的{% mathjax %}L_2{% endmathjax %}范数来正则化统计模型的经典方法。在概率角度看，我们可以通过以下论证来证明这一技术的合理性：我们已经假设了一个先验，即权重的值取自均值为0的高斯分布。更直观的是，我们希望模型深度挖掘特征，即将其权重分散到许多特征中，而不是过于依赖少数潜在的虚假关联。

##### 重新审视过拟合

当面对更多的特征而样本不足时，线性模型往往会过拟合。相反，当给出更多样本而不是特征，通常线性模型不会过拟合。不幸的是，线性模型泛化的可靠性是有代价的。简单地说，线性模型没有考虑到特征之间的交互作用。对于每个特征，线性模型必须指定正的或负的权重，而忽略其他特征。泛化性和灵活性之间的这种基本权衡被描述为偏差-方差权衡(`bias-variance tradeoff`)。线性模型有很高的偏差：它们只能表示一小类函数。然而，这些模型的方差很低：它们在不同的随机数据样本上可以得出相似的结果。深度神经网络位于偏差-方差谱的另一端。与线性模型不同，神经网络并不局限于单独查看每个特征，而是学习特征之间的交互。即使我们有比特征多得多的样本，深度神经网络也有可能过拟合。`2017`年，一组研究人员通过在随机标记的图像上训练深度网络。这展示了神经网络的极大灵活性，因为人类很难将输入和随机标记的输出联系起来，但通过随机梯度下降优化的神经网络可以完美地标记训练集中的每一幅图像。想一想这意味着什么？假设标签是随机均匀分配的，并且有`10`个类别，那么分类器在测试数据上很难取得高于`10%`的精度，那么这里的泛化差距就高达`90%`，如此严重的过拟合。

##### 扰动的稳健性

在探究泛化性之前，我们先来定义一下什么是一个“好”的预测模型？ 我们期待“好”的预测模型能在未知的数据上有很好的表现：经典泛化理论认为，为了缩小训练和测试性能之间的差距，应该以简单的模型为目标。简单性以较小维度的形式展现，简单性的另一个角度是平滑性，即函数不应该对其输入的微小变化敏感。例如，当我们对图像进行分类时，我们预计向像素添加一些随机噪声应该是基本无影响的。 `1995`年，克里斯托弗·毕晓普证明了 具有输入噪声的训练等价于`Tikhonov`正则化 (`Bishop, 1995`)。这项工作用数学证实了“要求函数光滑”和“要求函数对输入的随机噪声具有适应性”之间的联系。然后在`2014`年，斯里瓦斯塔瓦等人 (`Srivastava et al., 2014`) 就如何将毕晓普的想法应用于网络的内部层提出了一个想法：在训练过程中，他们建议在计算后续层之前向网络的每一层注入噪声。因为当训练一个有多层的深层网络时，注入噪声只会在输入-输出映射上增强平滑性。

这个想法被称为暂退法(`dropout`)。暂退法在前向传播过程中，计算每一内部层的同时注入噪声，这已经成为训练神经网络的常用技术。这种方法之所以被称为暂退法，因为我们从表面上看是在训练过程中丢弃(`drop out`)一些神经元。在整个训练过程的每一次迭代中，标准暂退法包括在计算下一层之前将当前层中的一些节点置零。需要说明的是，暂退法的原始论文提到了一个关于有性繁殖的类比：神经网络过拟合与每一层都依赖于前一层激活值相关，称这种情况为“共适应性”。作者认为，暂退法会破坏共适应性，就像有性生殖会破坏共适应的基因一样。那么关键的挑战就是如何注入这种噪声。一种想法是以一种无偏向(`unbiased`)的方式注入噪声。这样在固定住其他层时，每一层的期望值等于没有噪音时的值。在毕晓普的工作中，他将高斯噪声添加到线性模型的输入中。在每次训练迭代中，他将从均值为零的分布{% mathjax %}\epsilon \sim \mathcal{N}(0, \sigma^2){% endmathjax %}采样噪声添加到输入{% mathjax %}\mathbf{x}{% endmathjax %}从而产生扰动点{% mathjax %}x' = x + \epsilon{% endmathjax %}，预期是{% mathjax %}E[\mathbf{x}'] = \mathbf{x}{% endmathjax %}。在标准暂退法正则化中，通过按保留（未丢弃）的节点的分数进行规范化来消除每一层的偏差。换言之，每个中间活性值{% mathjax %}h{% endmathjax %}以暂退概率{% mathjax %}p{% endmathjax %}由随机变量{% mathjax %}h'{% endmathjax %}替换如下所示：
{% mathjax '{"conversion":{"em":14}}' %}
h' = 
 \begin{cases}
      0 & \text{概率为p}\\
      \frac{h}{1-p} & \text{其它情况}\\
 \end{cases}
{% endmathjax %}
根据此模型的设计，其期望值保持不变，即{% mathjax %}E[h'] = h{% endmathjax %}。回想一下带有`1`个隐藏层和`5`个隐藏单元的多层感知机。当我们将暂退法应用到隐藏层，以{% mathjax %}p{% endmathjax %}的概率将隐藏单元置为零时，结果可以看作一个只包含原始神经元子集的网络。比如删除了{% mathjax %}h_2{% endmathjax %}和{% mathjax %}h_5{% endmathjax %}并且它们各自的梯度在执行反向传播时也会消失。这样，输出层的计算不能过度依赖于{% mathjax %}h_1,\ldots,h_5{% endmathjax %}的任何一个元素。
{% asset_img mlp_12.png %}

通常，我们在测试时不用暂退法。给定一个训练好的模型和一个新的样本，我们不会丢弃任何节点，因此不需要标准化。然而也有一些例外：一些研究人员在测试时使用暂退法，用于估计神经网络预测的“不确定性”：如果通过许多不同的暂退法遮盖后得到的预测结果都是一致的，那么我们可以说网络发挥更稳定。

#### 前向传播、反向传播和计算图

我们已经学习了如何用小批量随机梯度下降训练模型。然而当实现该算法时，我们只考虑了通过前向传播(`forward propagation`)所涉及的计算。在计算梯度时，我们只调用了深度学习框架提供的反向传播函数，而不知其所以然。梯度的自动计算（自动微分）大大简化了深度学习算法的实现。在自动微分之前，即使是对复杂模型的微小调整也需要手工重新计算复杂的导数，学术论文也不得不分配大量页面来推导更新规则。

##### 前向传播

前向传播(`forward propagation`)指的是：按顺序（从输入层到输出层）计算和存储神经网络中每层的结果。我们将一步步研究单隐藏层神经网络的机制，为了简单起见，我们假设输入样本是{% mathjax %}\mathbf{x}\in \mathbb{R}^d{% endmathjax %}，并且我们的隐藏层不包括偏置项，这里的中间变量是：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{z} = \mathbf{W}^{(1)}\mathbf{x}
{% endmathjax %}
其中{% mathjax %}\mathbf{W}^{(1)}\in \mathbb{R}^{h\times d}{% endmathjax %}是隐藏层的权重参数。将中间变量{% mathjax %}\mathbf{z}\in \mathbb{R}^h{% endmathjax %}通过激活函数{% mathjax %}\phi{% endmathjax %}后，我们得到长度为{% mathjax %}h{% endmathjax %}的隐藏激活向量：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{h} = \phi (\mathbf{z})
{% endmathjax %}
隐藏变量{% mathjax %}h{% endmathjax %}也是一个中间变量。假设输出层的参数只有权重{% mathjax %}\mathbf{W}^{{2}}\in \mathbb{R}^{q\times h}{% endmathjax %}，我们可以得到输出层变量，它是一个长度为{% mathjax %}q{% endmathjax %}的向量：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{o} = \mathbf{W}^{(2)}\mathbf{h}
{% endmathjax %}
假设损失函数为{% mathjax %}l{% endmathjax %}，样本标签为{% mathjax %}y{% endmathjax %}，我们可以计算单个样本的损失项：
{% mathjax '{"conversion":{"em":14}}' %}
L = l(\mathbf{o},y)
{% endmathjax %}
根据{% mathjax %}L_2{% endmathjax %}正则化的定义，给定超参数，正则化项为
{% mathjax '{"conversion":{"em":14}}' %}
s = \frac{\lambda}{2}(\|\mathbf{W}^{(1)}\|_F^2 + \|\mathbf{W}^{(2)}\|_F^2)
{% endmathjax %}
其中矩阵的`Frobenius`范数是将矩阵展平为向量后应用的{% mathjax %}L_2{% endmathjax %}范数。最后，模型在给定样本上的正则化损失为：
{% mathjax '{"conversion":{"em":14}}' %}
J = L + s
{% endmathjax %}
下面，我们将{% mathjax %}J{% endmathjax %}称为**目标函数**(`objective function`)。
##### 前向传播计算图

绘制计算图有助于我们可视化计算中操作符和变量的依赖关系。下图是上述简单网络相对应的计算图，其中正方形表示变量，圆圈表示操作符。左下角表示输入，右上角表示输出。注意显示数据流的箭头方向主要是向右和向上的。
{% asset_img mlp_13.png "前向传播的计算图" %}

##### 反向传播

反向传播(`backward propagation`)指的是计算神经网络参数梯度的方法。简言之，该方法根据微积分中的链式规则，按相反的顺序从输出层到输入层遍历网络。该算法存储了计算某些参数梯度时所需的任何中间变量（偏导数）。假设我们有函数{% mathjax %}\mathsf{Y} = f(\mathsf{X}){% endmathjax %}和{% mathjax %}\mathsf{Z} = g(\mathsf{Y}){% endmathjax %}，其中输入和输出{% mathjax %}\mathsf{X,Y,Z}{% endmathjax %}是任意形状的张量。利用链式法则，我们可以计算{% mathjax %}\mathsf{Z}{% endmathjax %}关于{% mathjax %}\mathsf{X}{% endmathjax %}的导数：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial\mathsf{Z}}{\partial\mathsf{X}} = \text{prod}(\frac{\partial\mathsf{Z}}{\partial\mathsf{Y}},\frac{\partial\mathsf{Y}}{\partial\mathsf{X}})
{% endmathjax %}
在这里，我们使用{% mathjax %}\text{prod}{% endmathjax %}运算符在执行必要的操作（如换位和交换输入位置）后将其参数相乘。对于向量，这很简单，它只是矩阵-矩阵乘法。对于高维张量，我们使用适当的对应项。运算符{% mathjax %}\text{prod}{% endmathjax %}指代了所有的这些符号。回想一下，前向传播计算图中的但隐藏层简单网络的参数是{% mathjax %}\mathbf{W}^{(1)}{% endmathjax %}和{% mathjax %}\mathbf{W}^{(2)}{% endmathjax %}。**反向传播的目的是计算梯度**，{% mathjax %}\partial J/\partial\mathbf{W}^{(1)}{% endmathjax %}和{% mathjax %}\partial J/\partial\mathbf{W}^{(2)}{% endmathjax %}。为此，我们应用链式法则，以此计算每个中间变量和参数的梯度。计算的顺序与前向传播中执行的顺序相反，因此我们需要从计算图的结果开始，并朝着参数的方向努力。第一步是计算目标函数{% mathjax %}J = L + s{% endmathjax %}相对于损失项{% mathjax %}L{% endmathjax %}和正则项{% mathjax %}s{% endmathjax %}的梯度。
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial J}{\partial L} = 1\;\text{and}\;\frac{\partial J}{\partial s} = 1
{% endmathjax %}
接下来，我们根据链式法则计算目标函数关于输出层变量{% mathjax %}\mathbf{o}{% endmathjax %}的梯度：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial J}{\partial\mathbf{o}} = \text{prod}(\frac{\partial J}{\partial L},\frac{\partial L}{\partial\mathbf{o}}) = \frac{\partial L}{\partial\mathbf{o}}\in \mathsf{R}^q
{% endmathjax %}
接下来，我们计算正则化项相对于两个参数的梯度：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial s}{\partial\mathbf{W}^{(1)}} = \lambda\mathbf{W}^{(1)}\;\text{and}\;\frac{\partial s}{\partial\mathbf{W}^{(2)}} = \lambda\mathbf{W}^{(2)}
{% endmathjax %}
现在我们可以计算最接近输出层的模型参数的梯度{% mathjax %}\partial J/\partial\mathbf{W}^{(2)}\in \mathsf{R}^{q\times h}{% endmathjax %}。使用链式法则得出：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial J}{\partial\mathbf{W}^{(2)}} = \text{prod}(\frac{\partial J}{\partial\mathbf{o}},\frac{\partial\mathbf{o}}{\partial\mathbf{W}^{(2)}}) + \text{prod}(\frac{\partial J}{\partial s},\frac{\partial s}{\partial\mathbf{W}^{(2)}}) = \frac{\partial J}{\partial\mathbf{o}}\mathbf{h}^{\mathsf{T}} + \lambda\mathbf{W}^{(2)}
{% endmathjax %}
为了获得关于{% mathjax %}\mathbf{W}^{(1)}{% endmathjax %}的梯度，我们需要继续沿着输出层到隐藏层反向传播。关于隐藏层输出的梯度{% mathjax %}\partial J/\partial\mathbf{h}\in \mathbb{R}^h{% endmathjax %}由下式给出：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial J}{\partial\mathbf{h}}= \text{prod}(\frac{\partial J}{\partial\mathbf{o}},\frac{\partial\mathbf{o}}{\partial\mathbf{h}}) = \mathbf{W}^{(2)^{\mathsf{T}}}\frac{\partial J}{\partial\mathbf{o}}
{% endmathjax %}
由于激活函数{% mathjax %}\phi{% endmathjax %}是按元素计算的，计算中间变量{% mathjax %}\mathbf{z}{% endmathjax %}的梯度{% mathjax %}\partial J/\partial\mathbf{z}\in \mathbb{R}^h{% endmathjax %}需要使用按元素乘法运算符，我们用{% mathjax %}\odot{% endmathjax %}表示：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial J}{\partial\mathbf{z}}= \text{prod}(\frac{\partial J}{\partial\mathbf{h}},\frac{\partial\mathbf{h}}{\partial\mathbf{z}}) = \frac{\partial J}{\partial\mathbf{h}}\odot\phi'(\mathbf{z})
{% endmathjax %}
最后，我们可以得到最接近输入层的模型参数的梯度{% mathjax %}\partial J/\partial\mathbf{W}^{(1)}\in \mathbb{R}^{h\times d}{% endmathjax %}。根据链式法则，我们得到：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial J}{\partial\mathbf{W}^{(1)}} = \text{prod}(\frac{\partial J}{\partial\mathbf{z}},\frac{\partial\mathbf{z}}{\partial\mathbf{W}^{(1)}}) + \text{prod}(\frac{\partial J}{\partial s},\frac{\partial s}{\partial\mathbf{W}^{(1)}}) = \frac{\partial J}{\partial\mathbf{z}}\mathbf{x}^{\mathsf{T}} + \lambda\mathbf{W}^{(1)}
{% endmathjax %}
##### 训练神经网络

在训练神经网络时，前向传播和反向传播相互依赖。对于前向传播，我们沿着依赖的方向遍历计算图并计算其路径上的所有变量。然后将这些用于反向传播，其中计算顺序与计算图的相反。在训练神经网络时，在初始化模型参数后，我们交替使用前向传播和反向传播，利用反向传播给出的梯度来更新模型参数。注意，反向传播重复利用前向传播中存储的中间值，以避免重复计算。带来的影响之一是我们需要保留中间值，直到反向传播完成。这也是训练比单纯的预测需要更多的内存（显存）的原因之一。此外，这些中间值的大小与网络层的数量和批量的大小大致成正比。因此，使用更大的批量来训练更深层次的网络更容易导致内存不足(`out of memory`)错误。

##### 总结

前向传播在神经网络定义的计算图中按顺序计算和存储中间变量，它的顺序是从输入层到输出层。反向传播按相反的顺序（从输出层到输入层）计算和存储神经网络的中间变量和参数的梯度。在训练深度学习模型时，前向传播和反向传播是相互依赖的。训练比预测需要更多的内存。

#### 数值稳定性和模型初始化

我们实现的每个模型都是根据某个预先指定的分布来初始化模型的参数。有人会认为初始化方案是理所当然的，忽略了如何做出这些选择的细节。甚至有人可能会觉得，初始化方案的选择并不是特别重要。相反，初始化方案的选择在神经网络学习中起着举足轻重的作用，它对保持数值稳定性至关重要。此外，这些初始化方案的选择可以与非线性激活函数的选择有效的结合在一起。我们选择哪个函数以及如何初始化参数可以决定优化算法收敛的速度有多快。糟糕的选择可能会导致我们在训练时遇到梯度爆炸或梯度消失。

##### 梯度消失和梯度爆炸

考虑一个具有{% mathjax %}L{% endmathjax %}层、输入{% mathjax %}\mathbf{x}{% endmathjax %}和输出{% mathjax %}\mathbf{o}{% endmathjax %}的深层网络。每一层{% mathjax %}l{% endmathjax %}由变换{% mathjax %}f_l{% endmathjax %}定义，该变换的参数为权重{% mathjax %}\mathbf{W}^{(l)}{% endmathjax %}，其隐藏变量是{% mathjax %}\mathbf{h}^{(l)}{% endmathjax %}(令{% mathjax %}\mathbf{h}^{(0)}=\mathbf{x}{% endmathjax %})。我们的网络可以表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{h}^{(l)} = f_l(\mathbf{h}^{(l-1)})\;\text{因此}\;\mathbf{o} = f_L\circ\dots\circ f_1(\mathbf{x})
{% endmathjax %}
如果所有隐藏变量和输入都是向量，我们可以将{% mathjax %}\mathbf{o}{% endmathjax %}关于任何一组参数{% mathjax %}\mathbf{W^{(l)}}{% endmathjax %}的梯度写为下式：
{% mathjax '{"conversion":{"em":14}}' %}
\partial_{\mathbf{W}^{(l)}}\mathbf{o} = \underbrace{\partial_{h^{(L-1)}}\mathbf{h}^{(L)}}_{M^{(L)}\stackrel{def}{=}} \cdot \ldots \cdot \underbrace{\partial_{h^{(l)}}\mathbf{h}^{(l+1)}}_{M^{(l+1)}\stackrel{def}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}}\mathbf{h}^{(l)}}_{v^{(l)}\stackrel{def}{=}}
{% endmathjax %}
换言之，该梯度是{% mathjax %}L - l{% endmathjax %}个矩阵{% mathjax %}\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}{% endmathjax %}与梯度向量{% mathjax %}\mathbf{v}^{(l)}{% endmathjax %}的乘积。因此，我们容易受到数值下溢问题的影响。当太多的概率乘在一起时，这些问题经常会出现。在处理概率时，一个常见的技巧是切换到对数空间将数值表示的压力从尾数转移到指数。不幸的是，上面的问题更为严重。最初{% mathjax %}\mathbf{M^{(l)}}{% endmathjax %}可能具有各种各样的特征值。他们可能很小，也可能很大； 他们的乘积可能非常大，也可能非常小。

不稳定梯度带来的风险不止在于数值表示；不稳定梯度也威胁到我们优化算法的稳定性。我们可能面临一些问题。要么是**梯度爆炸**(`gradient exploding`)问题：参数更新过大，破坏了模型的稳定收敛；要么是**梯度消失**(`gradient vanishing`)问题：参数更新过小，在每次更新时几乎不会移动，导致模型无法学习。
###### 梯度消失

曾经`sigmoid`函数{% mathjax %}1/(1 + \text{exp}(-x)){% endmathjax %}很流行，因为它类似于阈值函数。由于早期的人工神经网络受到生物神经网络的启发，神经元要么完全激活要么完全不激活（就像生物神经元）的想法很有吸引力。然而，它却是导致梯度消失问题的一个常见的原因，让我们仔细看看`sigmoid`函数为什么会导致梯度消失。
```python
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

plt.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))

```
{% asset_img mlp_14.png %}

正如上图，当`sigmoid`函数的输入很大或是很小时，它的梯度都会消失。此外，当反向传播通过许多层时，除非我们在刚刚好的地方，这些地方`sigmoid`函数的输入接近于零，否则整个乘积的梯度可能会消失。当我们的网络有很多层时，除非我们很小心，否则在某一层可能会切断梯度。事实上，这个问题曾经困扰着深度网络的训练。因此，更稳定的`ReLU`系列函数已经成为从业者的默认选择（虽然在神经科学的角度看起来不太合理）。
###### 梯度爆炸

相反，梯度爆炸可能同样令人烦恼。为了更好地说明这一点，我们生成100个高斯随机矩阵，并将它们与某个初始矩阵相乘。对于我们选择的尺度（方差{% mathjax %}\sigma^2{% endmathjax %}），矩阵乘积发生爆炸。当这种情况是由于深度网络的初始化所导致时，我们没有机会让梯度下降优化器收敛。
```python
M = torch.normal(0, 1, size=(4,4))
print('一个矩阵 \n',M)
for i in range(100):
    M = torch.mm(M,torch.normal(0, 1, size=(4, 4)))

print('乘以100个矩阵后\n', M)

# 一个矩阵
#  tensor([[-0.7872,  2.7090,  0.5996, -1.3191],
#         [-1.8260, -0.7130, -0.5521,  0.1051],
#         [ 1.1213,  1.0472, -0.3991, -0.3802],
#         [ 0.5552,  0.4517, -0.3218,  0.5214]])
# 乘以100个矩阵后
#  tensor([[-2.1897e+26,  8.8308e+26,  1.9813e+26,  1.7019e+26],
#         [ 1.3110e+26, -5.2870e+26, -1.1862e+26, -1.0189e+26],
#         [-1.6008e+26,  6.4559e+26,  1.4485e+26,  1.2442e+26],
#         [ 3.0943e+25, -1.2479e+26, -2.7998e+25, -2.4050e+25]])
```
###### 打破对称性

神经网络设计中的另一个问题是其参数化所固有的对称性。假设我们有一个简单的多层感知机，它有一个隐藏层和两个隐藏单元。在这种情况下，我们可以对第一层的权重{% mathjax %}\mathbf{W^{(1)}}{% endmathjax %}进行重排列，并且同样对输出层的权重进行重排列，可以获得相同的函数。第一个隐藏单元与第二个隐藏单元没有什么特别的区别。换句话说，我们在每一层的隐藏单元之间具有排列对称性。假设输出层将上述两个隐藏单元的多层感知机转换为仅一个输出单元。想象一下，如果我们将隐藏层的所有参数初始化为{% mathjax %}\mathbf{W^{(1)}} = c{% endmathjax %}
，{% mathjax %}c{% endmathjax %} 为常量，会发生什么？在这种情况下，在前向传播期间，两个隐藏单元采用相同的输入和参数，产生相同的激活，该激活被送到输出单元。在反向传播期间，根据参数{% mathjax %}\mathbf{W^{(1)}}{% endmathjax %}对输出单元进行微分，得到一个梯度，其元素都取相同的值。因此，在基于梯度的迭代（例如，小批量随机梯度下降）之后，{% mathjax %}\mathbf{W^{(1)}}{% endmathjax %}的所有元素仍然采用相同的值。这样的迭代永远不会打破对称性，我们可能永远也无法实现网络的表达能力。隐藏层的行为就好像只有一个单元。请注意，虽然小批量随机梯度下降不会打破这种对称性，但暂退法正则化可以。
##### 参数初始化

解决（或至少减轻）上述问题的一种方法是进行**参数初始化**，优化期间的注意和适当的正则化也可以进一步提高稳定性。我们使用正态分布来初始化权重值。如果我们不指定初始化方法，框架将使用默认的随机初始化方法，对于中等难度的问题，这种方法通常很有效。让我们看看某些没有非线性的全连接层输出（例如，隐藏变量）{% mathjax %}o_i{% endmathjax %}的尺度分布。对于该层{% mathjax %}n_{in}{% endmathjax %}输入{% mathjax %}x_j{% endmathjax %}及其相关权重{% mathjax %}w_{ij}{% endmathjax %}，输出由下式给出：
{% mathjax '{"conversion":{"em":14}}' %}
o_i = \sum_{j=1}^{n_{in}} w_{ij}x_j
{% endmathjax %}