---
title: softmax回归（线性神经网络）(PyTorch)
date: 2024-04-28 11:00:11
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

#### softmax 回归

通常，机器学习实践者用分类这个词来描述两个有微妙差别的问题：1.我们只对样本的“硬性”类别感兴趣，即属于哪个类别；2.我们希望得到“软性”类别，即得到属于每个类别的概率。这两者的界限往往很模糊。其中的一个原因是：即使我们只关心硬类别，我们仍然使用软类别的模型。我们从一个图像分类问题。假设每次输入是一个{% mathjax %}2\times 2{% endmathjax %}的灰度图像，我们可以用一个标量来表示每个像素值，每个图相对应四个特征{% mathjax %}x_1,x_2,x_3,x_4{% endmathjax %}。假设每个图像属于类别“猫”“鸡”和“狗”中的一个。接下来，我们要选择如何表示标签。我们有两个明显的选择：最直接的想法是选择{% mathjax %}y\in {1,2,3}{% endmathjax %},其中整数分别代表{狗，猫，鸡}。这是在计算机上存储此类信息的有效方法。如果累别间有一些自然顺序，比如我们视图预测{婴儿，儿童，青少年，青年人，中年人，老年人}，那么将这个问题转换为回归问题，并且保留这些格式是有意义的。但是一般的分类问题并不与类别之间的自然顺序有关。幸运的是，统计学家很早以前就发明了一种表示分类数据的简单方法：**读热编码**(`one-hot encoding`)。读热编码是一个向量，它的分量和类别一样多。类别对应的分量设置为1，其它所有分量设置为0，在我们的例子中，标签{% mathjax %}y{% endmathjax %}将是一个三维向量，其中`(1,0,0)`对应于猫，`(0,1,0)`对应于鸡，(0,0,1)对应于狗：
{% mathjax '{"conversion":{"em":14}}' %}
y \in \{(1,0,0),(0,1,0),(0,0,1)\}
{% endmathjax %}
<!-- more -->

##### 网络架构

为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。为了解决线性模型的分类问题，我们需要和输出一样多的仿射函数（`affine function`）。每个输出对应于它自己的仿射函数。在我们的例子中，由于我们有`4`个特征和`3`个可能的输出类别，我们将需要`12`个标量来表示权重（带下标的{% mathjax %}w{% endmathjax %}），`3`个标量来表示偏置（带下标的{% mathjax %}b{% endmathjax %}）。下面我们为每个输入计算三个未规范化的预测。下面我们为每个输入计算未规范化的预测(`logit`)：{% mathjax %}o_1,o_2,o_3{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& o_1=x_1w_11 + x_2w_12 + x_3w_13 + x_4w_14 + b_1,\\
& o_2=x_1w_21 + x_2w_22 + x_3w_23 + x_4w_24 + b_2,\\
& o_3=x_1w_31 + x_2w_32 + x_3w_33 + x_4w_34 + b_3,\\
\end{align}
{% endmathjax %}
我们可以用神经网络图来描述这个计算过程。与线性回归一样，`softmax`回归也是一个单层神经网络。由于计算每个输出{% mathjax %}o_1、o_2、o_3{% endmathjax %}取决于所有输入{% mathjax %}x_1、x_2、x_3、x_4{% endmathjax %}，所以`softmax`回归的输出层也是全连接层。
{% asset_img s_1.png "softmax回归是一种单层神经网络" %}

为了更简洁地表达模型，我们仍然使用线性代数符号。通过向量形式表达为{% mathjax %}\mathbf{o} = \mathbf{Wx} + b{% endmathjax %}，这是一种更适合数学和编写代码的形式，由此，我们已经将所有权重放到一个{% mathjax %}3\times 4{% endmathjax %}的矩阵中。对于给定数据样本的特征{% mathjax %}x{% endmathjax %}，我们的输出是由权重和输入特征进行矩阵-向量乘法再加上偏置{% mathjax %}b{% endmathjax %}得到的。
##### 全连接层的参数开销

在深度学习中，全连接层无处不在。然而，全连接层是完全连接的，可能有很多可学习的参数。具体来说，对于任何具有{% mathjax %}d{% endmathjax %}个输入和{% mathjax %}q{% endmathjax %}个输出的全连接层，参数开销为{% mathjax %}\mathcal{O}(dq){% endmathjax %}，这个数在在实践中可能高得令人望而却步。幸运的是，将{% mathjax %}d{% endmathjax %}个输入转化为{% mathjax %}q{% endmathjax %}个输出成本可以减少到{% mathjax %}\mathcal{O}(\frac{dq}{n}){% endmathjax %}，其中参数{% mathjax %}n{% endmathjax %}可以由我们灵活指定，以在实际应用中平衡参数节约和模型有效性。
##### softmax运算

现在我们将优化参数以最大化观测数据的概率。为了得到预测结果，我们将设置一个阈值，如选择具有最大概率的标签。我们希望模型的输出{% mathjax %}\hat{y}_j{% endmathjax %}可以视为属于类{% mathjax %}j{% endmathjax %}的概率，然后选择具有最大输出值的类别{% mathjax %}\text{argmax}_jy_j{% endmathjax %}作为我们的预测。例如，如果{% mathjax %}\hat{y}_1、\hat{y}_2{% endmathjax %}和{% mathjax %}\hat{y}_3{% endmathjax %}分别为`0.1`、`0.8`和`0.1`，那么我们的预测类别是`2`，在我们的例子中代表鸡。然而我们能否将未规范化的预测{% mathjax %}o{% endmathjax %}直接视作我们感兴趣的输出呢？答案是否定的。因为将线性层的输出直接视为概率时存在一些问题：一方面，我们没有限制这些输出数字的总和为`1`。另一方面，根据输入的不同，它们可以为负值。要将输出视为概率，我们必须保证在任何数据上的输出都是非负的且总和为1。此外，我们需要一个训练的目标函数，来激励模型精准地估计概率。例如，在分类器输出`0.5`的所有样本中，我们希望这些样本是刚好有一半实际上属于预测的类别。这个属性叫做校准(`calibration`)。

社会科学家邓肯·卢斯于`1959`年在选择模型(`choice model`)的理论基础上发明的`softmax`函数正是这样做的：`softmax`函数能够将未规范化的预测变换为非负数并且总和为`1`，同时让模型保持 可导的性质。为了完成这一目标，我们首先对每个未规范化的预测求幂，这样可以确保输出非负。为了确保最终输出的概率值总和为`1`，我们再让每个求幂后的结果除以它们的总和。如下式：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{\mathbf{y}} = \text{softmax}(\mathbf{o}) \quad \text{其中} \quad \hat{y}_j= \frac{\text{exp}(o_j)}{\sum_k \text{exp}(o_k)}
{% endmathjax %}
这里，对于所有的{% mathjax %}j{% endmathjax %}总有{% mathjax %}0\leq \hat{y}_j \leq 1{% endmathjax %}。因此，{% mathjax %}\hat{\mathbf{y}}{% endmathjax %}可以视为一个正确的概率分布。`softmax`运算不会改变为规范化的预测{% mathjax %}\mathbf{o}{% endmathjax %}之间的大小次序，只会确定分配给每个类别的概率。因此，在预测过程中，我们仍然可以用下面的公式来选择最有可能的类别。
{% mathjax '{"conversion":{"em":14}}' %}
\underset{j}{\text{argmax}}\hat{y}_j = \underset{j}{\text{argmax}}o_j
{% endmathjax %}
尽管`softmax`是一个非线性函数，但softmax回归的输出仍然由输入特征的仿射变换决定。因此，`softmax`回归是一个线性模型(`linear model`)。
##### 小批量样本的矢量化

为了提高计算效率并且充分利用GPU，我们通常会对小批量样本的数据执行矢量计算。假设我们读取了一个批量的样本{% mathjax %}\mathbf{X}{% endmathjax %}，其中特征维度（输入数量）为{% mathjax %}d{% endmathjax %}，批量大小为{% mathjax %}n{% endmathjax %}。此外，假设我们在输出中有{% mathjax %}q{% endmathjax %}个类别。那么小批量样本的特征为{% mathjax %}\mathbf{X}\in \mathbb{R}^{d\times q}{% endmathjax %}，偏置为{% mathjax %}b\in \mathbb{R}^{1\times q}{% endmathjax %}。`softmax`回归的矢量计算表达式为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \mathbf{O} = \mathbf{XW} + \mathbf{b} \\
& \hat{\mathbf{Y}} = \text{softmax}(\mathbf{O}) \\
\end{align}
{% endmathjax %}
相对于一次处理一个样本，小批量样本的矢量化加快了{% mathjax %}\mathbf{X}{% endmathjax %}和{% mathjax %}\mathbf{W}{% endmathjax %}的矩阵-向量乘法，由于{% mathjax %}\mathbf{X}{% endmathjax %}中的每一行，代表一个数据样本，那么`softmax`运算可以按行(`rowwise`)执行；对于{% mathjax %}\mathbf{O}{% endmathjax %}的每一行，我们先对所有项进行幂运算，然后通过求和对他们进行标准化。{% mathjax %}\mathbf{XW} + \mathbf{b}{% endmathjax %}的求和会使用广播机制，小批量的未规范化预测{% mathjax %}\mathbf{O}{% endmathjax %}和输出概率{% mathjax %}\hat{\mathbf{Y}}{% endmathjax %}都是形状为{% mathjax %}n\times q{% endmathjax %}的矩阵。
##### 损失函数

接下来，我们需要一个损失函数来度量预测的效果。我们将使用最大似然估计，这与在线性回归中的方法相同。`softmax`函数给出了一个向量{% mathjax %}\hat{\mathbf{y}}{% endmathjax %}，我们可以将其视为“对给定任意输入{% mathjax %}\mathbf{x}{% endmathjax %}的每个类的条件概率”。例如，{% mathjax %}\hat{y}_1 = P(y=猫|\mathbf{x}){% endmathjax %}，假设整个数据集{% mathjax %}{\mathbf{X,Y}}{% endmathjax %}具有{% mathjax %}n{% endmathjax %}个样本，其中索引{% mathjax %}i{% endmathjax %}的样本由特征向量{% mathjax %}\mathbf{x}^{(i)}{% endmathjax %}和标签向量{% mathjax %}\mathbf{y}^{(i)}{% endmathjax %}组成。我们可以将估计值和实际值进行比较：
{% mathjax '{"conversion":{"em":14}}' %}
P(\mathbf{Y|X}) = \prod_{i=1}^n P(\mathbf{y}^{(i)}|\mathbf{x}^{(i)})
{% endmathjax %}
根据最大似然估计，我们最大化{% mathjax %}P(\mathbf{Y}|\mathbf{X}){% endmathjax %}，相当于最小化负对数似然：
{% mathjax '{"conversion":{"em":14}}' %}
-\log P(\mathbf{Y}|\mathbf{X}) = \sum_{i=1}^n - \log P(\mathbf{y}^{(i)}|\mathbf{x}^{(i)}) = \sum_{i=1}^n l(\mathbf{y}^{(i)},\hat{\mathbf{y}}^{(i)})
{% endmathjax %}
其中，对于任何标签{% mathjax %}\mathbf{y}{% endmathjax %}和模型预测{% mathjax %}\hat{\mathbf{y}}{% endmathjax %}，损失函数为：
{% mathjax '{"conversion":{"em":14}}' %}
l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{j=1}^q y_{j}\log\hat{y}_j
{% endmathjax %}
以上的损失函数，通常被称为交叉熵损失(`cross-entropy loss`)。由于{% mathjax %}\mathbf{y}{% endmathjax %}是一个长度为{% mathjax %}q{% endmathjax %}的编码向量，所以除了一个之外的所有项{% mathjax %}j{% endmathjax %}都消失了。由于所有{% mathjax %}\hat{y}_j{% endmathjax %}都是预测的概率，所以它们的对数永远不会大于`0`，因此，如果正确地预测实际标签{% mathjax %}P(\mathbf{y}|\mathbf{x})=1{% endmathjax %}，则损失函数不能进一步最小化。注意，这往往是不可能的。例如，数据集中可能存在标签噪声（比如某些样本可能被误标），或输入特征没有足够的信息来完美地对每一个样本分类。由于`softmax`和相关的损失函数很常见，因此我们需要更好地理解它的计算方式。利用`softmax`的定义，我们得到：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
l(\mathbf{y}, \hat{\mathbf{y}}) & = -\sum_{j=1}^q y_j\log \frac{\text{exp}(o_j)}{\sum_{k=1}^q \text{exp}(o_k)} \\
& = \sum_{j=1}^{q}y_i\log\sum_{k=1}^q \text{exp}(o_k) - \sum_{j=1}^qy_jo_j \\
& = \log\sum_{k=1}^q\text{exp}(o_k) - \sum_{j=1}^qy_jo_j \\
\end{align}
{% endmathjax %}
考虑相对于任何未规范化的预测{% mathjax %}o_j{% endmathjax %}的导数，我们得到：
{% mathjax '{"conversion":{"em":14}}' %}
\partial_{o_j}l(\mathbf{y},\hat{\mathbf{y}}) = \frac{\text{exp}(o_j)}{\sum_{k=1}^q\text{exp}(o_k)} - y_i = \text{softmax}(\mtahbf{o})_j - y_j
{% endmathjax %}
换句话说，导数是我们`softmax`模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。从这个意义上讲，这与我们在回归中看到的非常相似，其中梯度是观测值{% mathjax %}y{% endmathjax %}和估计值{% mathjax %}\hat{y}{% endmathjax %}之间的差异。这不是巧合，在任何指数族分布模型中，对数似然的梯度正是由此得出的。这使梯度计算在实践中变得容易很多。现在让我们考虑整个结果分布的情况，即观察到的不仅仅是一个结果。对于标签{% mathjax %}\mathbf{y}{% endmathjax %}，我们可以使用与以前相同的表示形式。唯一的区别是，我们现在用一个概率向量表示如`(0.1,0.2,0.7)`，而不是仅包含二元项的向量`(0,0,1)`。我们定义损失{% mathjax %}l{% endmathjax %}，它是所有标签分布的预期损失值。此损失称为**交叉熵损失**(`cross-entropy loss`)，它是分类问题最常用的损失之一。
##### 信息论

信息论(`information theory`)涉及编码、解码、发送以及尽可能简洁地处理信息或数据。信息论的核心思想是量化数据中的信息内容。在信息论中，该数值被称为分布{% mathjax %}P{% endmathjax %}的熵(`entropy`)。可以通过以下方程得到：
{% mathjax '{"conversion":{"em":14}}' %}
H[P] = \sum_j- P(j)\log P(j)
{% endmathjax %}
信息论的基本定理之一指出，为了对从分布{% mathjax %}p{% endmathjax %}随机抽取的数据进行编码，我们至少需要{% mathjax %}\mathbf{H}[\mathbf{P}]{% endmathjax %}纳特(`nat`)对其进行编码。“纳特”相当于比特(`bit`)，但是对数底为{% mathjax %}e{% endmathjax %}而不是2。因此一个纳特是{% mathjax %}\frac{1}{\log(2)}\approx 1.44{% endmathjax %}比特。

压缩与预测有什么关系呢？想象一下，我们有一个要压缩的数据流。如果我们很容易预测下一个数据，那么这个数据就很容易压缩。为什么呢？，举一个极端的例子，加入数据流中的每一个数据都相同，这会是一个无聊的数据流，由于它们总是相同的，我们总是知道下一个数据是什么。所以，为了传递数据流的内容，我们不必传输任何信息。也就是说，“下一个数据是`xx`”这个事件毫无信息量。但是，如果我们不能完全预测每一个事件，那么我们有时可能会感到“惊异”。克劳德·香农决定用信息量{% mathjax %}\log\frac{1}{P(j)} = -\log P(j){% endmathjax %}来量化这种惊异程度，在观察一个事件{% mathjax %}j{% endmathjax %}，并赋予它概率{% mathjax %}P(j){% endmathjax %}。当我们赋予一个事件较低的概率时，我们的惊异会更大，该事件的信息量也就更大。如果把熵{% mathjax %}H(P){% endmathjax %}想象为“知道真实概率的人所经历的惊异程度”，那么什么是交叉熵？交叉熵从{% mathjax %}P{% endmathjax %}到{% mathjax %}Q{% endmathjax %}记为{% mathjax %}H(P,Q){% endmathjax %}。我们可以把交叉熵想象为“主观概率为{% mathjax %}Q{% endmathjax %}观察者在看到根据概率{% mathjax %}P{% endmathjax %}生成的数据时的预期惊异”。当{% mathjax %}P=Q{% endmathjax %}时交叉熵达到最低。在这种情况下，从{% mathjax %}P{% endmathjax %}到{% mathjax %}Q{% endmathjax %}的交叉熵是{% mathjax %}H(P,P) = H(P){% endmathjax %}。简而言之，我们可以从两方面来考虑交叉熵的分类目标：(i)最大化观测数据的似然；(ii)最小化传达标签所需的惊异。

在训练`softmax`回归模型后，给出任何样本特征，我们可以预测每个输出类别的概率。通常我们使用预测概率最高的类别作为输出类别。如果预测与实际类别（标签）一致，则预测是正确的。在接下来的实验中，我们将使用精度(`accuracy`)来评估模型的性能。精度等于正确预测数与预测总数之间的比率。

#### 总结

`softmax`运算获取一个向量并将其映射为概率。`softmax`回归适用于分类问题，它使用了`softmax`运算中输出类别的概率分布。交叉熵是一个衡量两个概率分布之间差异的很好的度量，它测量给定模型编码数据所需的比特数。