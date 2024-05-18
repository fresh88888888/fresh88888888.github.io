---
title: 循环神经网络模型 (RNN)(TensorFlow)
date: 2024-05-17 14:38:11
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

我们学习了{% mathjax %}n{% endmathjax %}元语法模型，其中单词{% mathjax %}x_t{% endmathjax %}在时间步{% mathjax %}t{% endmathjax %}的条件概率仅取决于前面{% mathjax %}n-1{% endmathjax %}个单词。对于时间步{% mathjax %}t - (n - 1){% endmathjax %}之前的单词，如果我们想将其可能产生的影响合并到{% mathjax %}x_t{% endmathjax %}上，需要增加{% mathjax %}n{% endmathjax %}，然而模型参数的数量也会随之呈指数增长，因为词表{% mathjax %}\mathcal{V}{% endmathjax %}需要存储{% mathjax %}|\mathcal{V}^n|{% endmathjax %}个数字，因此与其将{% mathjax %}P(x_t|x_{t-1},\ldots,x_{t-n+1}){% endmathjax %}模型化，不如使用隐变量模型：
{% mathjax '{"conversion":{"em":14}}' %}
P(x_t|x_{t-1},\ldots,x_1) \approx P(x_t|h_{t-1})
{% endmathjax %}
<!-- more -->
其中{% mathjax %}h_{t-1}{% endmathjax %}是**隐状态**(`hidden state`)，也称为**隐藏变量**(`hidden variable`)，它存储了到时间步{% mathjax %}t-1{% endmathjax %}的序列信息。通常，我们可以基于当前输入{% mathjax %}x_t{% endmathjax %}和先前隐状态{% mathjax %}h_{t-1}{% endmathjax %}来计算时间步{% mathjax %}t{% endmathjax %}处的任何时间的隐状态：
{% mathjax '{"conversion":{"em":14}}' %}
h(t) = f(x_t,h_{t-1})
{% endmathjax %}
对于函数{% mathjax %}f{% endmathjax %}，隐变量模型不是近似值。毕竟{% mathjax %}h_t{% endmathjax %}是可以仅仅存储到目前为止观察到的所有数据，然而这样的操作可能会使计算和存储的代价都变得昂贵。回想一下，我们讨论过的具有隐藏单元的隐藏层。值得注意的是，隐藏层和隐状态指的是两个截然不同的概念。如上所述，隐藏层是在从输入到输出的路径上（以观测角度来理解）的隐藏的层，而隐状态则是在给定步骤所做的任何事情（以技术角度来定义）的输入，并且这些状态只能通过先前时间步的数据来计算。**循环神经网络**(`recurrent neural networks，RNNs`)是具有隐状态的神经网络。
#### 无隐状态的神经网络

让我们来看一看只有但隐藏层的多层感知机。设隐藏层的激活函数为{% mathjax %}\phi{% endmathjax %}，给定一个小批量样本{% mathjax %}\mathbf{X}\in \mathbb{R}^{n\times d}{% endmathjax %}，其中批量大小为{% mathjax %}n{% endmathjax %}，输入维度为{% mathjax %}d{% endmathjax %}，则隐藏层的输出为{% mathjax %}\mathbf{H}\in \mathbb{R}^{n\times h}{% endmathjax %}通过下式计算：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{H} = \phi (\mathbf{XW}_{xh} + \mathbf{b}_h)
{% endmathjax %}
在上面公式中，我们拥有的隐藏层权重参数为{% mathjax %}\mathbf{W}_{xh}\in \mathbb{R}^{d\times h}{% endmathjax %}，偏置参数{% mathjax %}b_h\in \mathbb{R}^{1\times h}{% endmathjax %}，以及隐藏单元的数目为{% mathjax %}h{% endmathjax %}。因此求和时可以应用广播机制。接下来，将隐藏变量{% mathjax %}\mathbf{H}{% endmathjax %}用作输出层的输入。输出层由下式给出：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{O} = \mathbf{HW}_{hq} + b_q
{% endmathjax %}
其中，{% mathjax %}\mathbf{O}\in \mathbb{R}^{n\times q}{% endmathjax %}是输出变量，{% mathjax %}\mathbf{W}_{hq}\in \mathbb{R}^{h\times q}{% endmathjax %}是权重参数，{% mathjax %}\mathbf{b}_q\in \mathbb{R}^{1\times q}{% endmathjax %}输出层的偏置参数。如果是分类问题，我们可以用{% mathjax %}\text{softmax}(\mathbf{O}){% endmathjax %}来计算输出类别的概率分布。只要可以随机选择“特征-标签”对，并且通过自动微分和随机梯度下降能够学习网络参数就可以了。
#### 有隐状态的循环神经网络

有了隐状态后，情况就完全不同了。假设我们在时间步{% mathjax %}t{% endmathjax %}有小批量输入{% mathjax %}\mathbf{X}_t\in \mathbb{R}^{n\times d}{% endmathjax %}。换言之，对于{% mathjax %}n{% endmathjax %}个序列样本的小批量，{% mathjax %}\mathbf{X}_t{% endmathjax %}的每一行对应于来自该序列的时间步{% mathjax %}t{% endmathjax %}处的一个样本。接下来，用{% mathjax %}\mathbf{H}_t\in \mathbb{R}^{n\times h}{% endmathjax %}表示时间步{% mathjax %}t{% endmathjax %}的隐藏变量。与多层感知机不同的是，我们在这里保存了前一个时间步的隐藏变量{% mathjax %}\mathbf{H}_{t-1}{% endmathjax %}，并引入了一个新的权重参数{% mathjax %}\mathbf{W}_{hh}\in \mathbb{R}^{h\times h}{% endmathjax %}，来描述如何在当前时间步中使用前一个时间步的隐藏变量。具体地说，当前时间步隐藏变量由当前时间步的输入与前一个时间步的隐藏变量一起计算得出：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{H}_t= \phi (\mathbf{X}_t\mathbf{W}_{xh} + \mathbf{H}_{t-1}\mathbf{W}_{hh} + \mathbf{b}_h)
{% endmathjax %}
多添加了一项{% mathjax %}\mathbf{H}_{t-1}\mathbf{W}_{hh}{% endmathjax %}，从而实例化了。从相邻时间步的隐藏变量{% mathjax %}\mathbf{H}_t{% endmathjax %}和{% mathjax %}\mathbf{H}_{t-1}{% endmathjax %}之间的关系可知，这些变量捕获并保留了序列直到其当前时间步的历史信息，就如当前时间步下神经网络的状态或记忆，因此这样的隐藏变量被称为**隐状态**(`hidden state`)。由于在当前时间步中，隐状态使用的定义与前一个时间步中使用的定义相同，因此是循环的(`recurrent`)。于是基于循环计算的隐状态神经网络被命名为 循环神经网络(`recurrent neural network`)。在循环神经网络中执行计算的层称为循环层(`recurrent layer`)。有许多不同的方法可以构建循环神经网络，输出层的输出类似于多层感知机中的计算：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{O}_t = \mathbf{H}_t\mathbf{W}_{hq} + \mathbf{b}_q
{% endmathjax %}
循环神经网络的参数包括隐藏层的权重{% mathjax %}\mathbf{W}_{xh}\in \mathbb{R}^{d\times h}, \mathbf{W}_{hh}\in \mathbb{R}^{h\times h}{% endmathjax %}和偏置{% mathjax %}\mathbf{b}_h\in \mathbb{R}^{1\times h}{% endmathjax %}，以及输出层的权重{% mathjax %}\mathbf{W}_{hq}\in \mathbb{R}^{h\times q}{% endmathjax %}和偏置{% mathjax %}\mathbf{b}_q\in \mathbb{R}^{1\times q}{% endmathjax %}。值得一提的是，即使在不同的时间步，循环神经网络也总是使用这些模型参数。因此，循环神经网络的参数开销不会随着时间步的增加而增加。

下图展示了循环神经网络在三个相邻时间步的计算逻辑。在任意时间步{% mathjax %}t{% endmathjax %}，隐状态的计算可以被视为：
- 拼接当前时间步{% mathjax %}t{% endmathjax %}的输入{% mathjax %}\mathbf{X}_t{% endmathjax %}和前一时间步{% mathjax %}t-1{% endmathjax %}的隐状态{% mathjax %}\mathbf{H}_{t-1}{% endmathjax %}；
- 将拼接的结果送入带有激活函数{% mathjax %}\phi{% endmathjax %}的全连接层。全连接层的输出是当前时间步{% mathjax %}t{% endmathjax %}的隐状态{% mathjax %}\mathbf{H}_t{% endmathjax %}。
在本例中，模型参数是{% mathjax %}\mathbf{W}_{xh}{% endmathjax %}和{% mathjax %}\mathbf{W}_{hh}{% endmathjax %}的拼接，以及{% mathjax %}b_h{% endmathjax %}的偏置。当前时间步隐状态{% mathjax %}\mathbf{H}_t{% endmathjax %}将参与计算下一时间步{% mathjax %}t+1{% endmathjax %}的隐状态{% mathjax %}\mathbf{H}_{t+1}{% endmathjax %}。而且{% mathjax %}\mathbf{H}_t{% endmathjax %}还将送入全连接输出层，用于计算当前时间步{% mathjax %}t{% endmathjax %}的输出{% mathjax %}\mathbf{O}_t{% endmathjax %}。
{% asset_img rnn_1.png "具有隐状态的循环神经网络" %}
#### 基于循环神经网络的字符级语言模型

回想一下语言模型，我们的目标是根据过去的和当前的词元预测下一个词元，因此我们将原始序列移位一个词元作为标签。`Bengio`等人首先提出使用神经网络进行语言建模。接下来，我们看一下如何使用循环神经网络来构建语言模型。设小批量大小为`1`，批量中的文本序列为`“machine”`。为了简化后续部分的训练，我们考虑使用字符级语言模型(`character-level language model`)，将文本词元化为字符而不是单词。下图演示了如何通过基于字符级语言建模的循环神经网络，使用当前的和先前的字符预测下一个字符。
{% asset_img rnn_2.png "基于循环神经网络的字符级语言模型" %}

在训练过程中，我们对每个时间步的输出层的输出进行`softmax`操作，然后利用交叉熵损失计算模型输出和标签之间的误差。由于隐藏层中隐状态的循环计算，上图中的第3个时间步的输出{% mathjax %}\mathbf{O}_3{% endmathjax %}由文本序列`“m”“a”`和`“c”`确定。由于训练数据中这个文本序列的下一个字符是`“h”`，因此第`3`个时间步的损失将取决于下一个字符的概率分布，而下一个字符是基于特征序列`“m”“a”“c”`和这个时间步的标签`“h”`生成的。在实践中，我们使用的批量大小为{% mathjax %}n > 1{% endmathjax %}，每个词元都由一个{% mathjax %}d{% endmathjax %}维向量表示。因此，在时间步{% mathjax %}t{% endmathjax %}输入{% mathjax %}\mathbf{X}_t{% endmathjax %}将是一个{% mathjax %}n\times d{% endmathjax %}矩阵。
#### 困惑度（Perplexity）

最后，让我们讨论如何度量语言模型的质量。我们在引入·回归时定义了熵、惊异和交叉熵，并在信息论中讨论了更多的信息论知识。如果想要压缩文本，我们可以根据当前词元集预测的下一个词元。一个更好的语言模型应该能让我们更准确地预测下一个词元。因此，它应该允许我们在压缩序列时花费更少的比特。所以我们可以通过一个序列中所有的{% mathjax %}n{% endmathjax %}个词元的交叉熵损失的平均值来衡量：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{1}{n}\sum_{t=1}^n -\log P(x_t|x_{t-1},\ldots,x_1)
{% endmathjax %}
其中{% mathjax %}P{% endmathjax %}由语言模型给出，{% mathjax %}x_t{% endmathjax %}是在时间步{% mathjax %}t{% endmathjax %}从该序列中观察到的实际词元。这使得不同长度的文档的性能具有了可比性。由于历史原因，自然语言处理的科学家更喜欢使用一个叫做**困惑度**(`perplexity`)的量。
{% mathjax '{"conversion":{"em":14}}' %}
\exp (-\frac{1}{n}\sum_{t=1}^n \log P(x_t|x_{t-1},\ldots,x_1))
{% endmathjax %}
困惑度的最好的理解是“下一个词元的实际选择数的调和平均数”。
- 在最好的情况下，模型总是完美地估计标签词元的概率为`1`。在这种情况下，模型的困惑度为`1`。
- 在最坏的情况下，模型总是预测标签词元的概率为`0`。在这种情况下，困惑度是正无穷大。
- 在基线上，该模型的预测是词表的所有可用词元上的均匀分布。在这种情况下，困惑度等于词表中唯一词元的数量。事实上，如果我们在没有任何压缩的情况下存储序列，这将是我们能做的最好的编码方式。因此，这种方式提供了一个重要的上限，而任何实际模型都必须超越这个上限。

#### 通过时间反向传播

到目前为止，我们已经反复提到像梯度爆炸或梯度消失，以及需要对循环神经网络分离梯度。循环神经网络中的前向传播相对简单。通过时间反向传播(`backpropagation through time，BPTT`)实际上是循环神经网络中反向传播技术的一个特定应用。它要求我们将循环神经网络的计算图一次展开一个时间步，以获得模型变量和参数之间的依赖关系。然后，基于链式法则，应用反向传播来计算和存储梯度。由于序列可能相当长，因此依赖关系也可能相当长。例如，某个`1000`个字符的序列，其第一个词元可能会对最后位置的词元产生重大影响。这在计算上是不可行的（它需要的时间和内存都太多了），并且还需要超过`1000`个矩阵的乘积才能得到非常难以捉摸的梯度。这个过程充满了计算与统计的不确定性。
##### 循环神经网络的梯度分析

我们从一个描述循环神经网络工作原理的简化模型开始，此模型忽略了隐状态的特性及其更新方式的细节。这里的数学表示没有像过去那样明确地区分标量、向量和矩阵，因为这些细节对于分析并不重要， 反而只会使本小节中的符号变得混乱。在简化模型中我们将时间步{% mathjax %}t{% endmathjax %}的隐状态表示为{% mathjax %}h_t{% endmathjax %}，输入表示为{% mathjax %}x_t{% endmathjax %}，输出表示为{% mathjax %}o_t{% endmathjax %}。输入与隐状态可以拼接后与隐藏层中的一个权重变量相乘。因此，我们分别使用{% mathjax %}w_h{% endmathjax %}和{% mathjax %}w_o{% endmathjax %}来表示隐藏层与输出层的权重，每个时间步的隐状态和输出可以写为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
h_t & = f(x_t,h_{t-1},w_h) \\ 
o_t & = g(h_t, w_o) \\
\end{align}
{% endmathjax %}
其中{% mathjax %}f{% endmathjax %}和{% mathjax %}g{% endmathjax %}分别是隐藏层和输出层的变换，因此，我们有一个链{% mathjax %}{\ldots,(x_{t-1},h_{t-1},o_{t-1}),(x_t,h_t,o_t),\ldots}{% endmathjax %}，它们通过循环计算彼此依赖。前向传播相当简单，一次一个时间步的遍历三元组{% mathjax %}(x_t,h_t,o_t){% endmathjax %}，然后通过一个目标函数在所有{% mathjax %}T{% endmathjax %}个时间步内评估输出{% mathjax %}o_t{% endmathjax %}和对应的标签{% mathjax %}y_t{% endmathjax %}之间的差异：
{% mathjax '{"conversion":{"em":14}}' %}
L(x_1,\ldots,x_{T},y_1,\ldots,y_{T},w_h,w_o) = \frac{1}{T}\sum_{t=1}^{T} l(y_t,o_t)
{% endmathjax %}
对于反向传播，问题则有点棘手，特别是当我们计算目标函数{% mathjax %}L{% endmathjax %}关于参数{% mathjax %}w_h{% endmathjax %}的梯度时，具体来说，按照链式法则：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\frac{\partial L}{\partial w_h} & = \frac{1}{T}\sum_{t=1}^T\frac{\partial l(y_t,o_t)}{\partial w_h} \\ 
& = \frac{1}{T}\sum_{t=1}^T\frac{\partial l(y_t,o_t)}{\partial o_t}\frac{\partial g(h_t,w_o)}{\partial h_t}\frac{\partial h_t}{\partial w_h} \\
\end{align}
{% endmathjax %}
在上图中乘积的第一项和第二项很容易计算，而第三项{% mathjax %}\frac{\partial h_t}{\partial w_h}{% endmathjax %}变得比较棘手，因为我们需要循环地计算参数{% mathjax %}w_h{% endmathjax %}对{% mathjax %}h_t{% endmathjax %}的影响。根据递归计算，{% mathjax %}h_t{% endmathjax %}既依赖于{% mathjax %}h_{t-1}{% endmathjax %}又依赖于{% mathjax %}w_h{% endmathjax %}，其中{% mathjax %}h_{t-1}{% endmathjax %}的计算也依赖于{% mathjax %}w_h{% endmathjax %}。因此，使用链式法则则产生：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial h_t}{\partial w_h} = \frac{\partial f(x_t,h_{t-1},w_h)}{\partial w_h} + \frac{\partial f(x_t,h_{t-1},w_h)}{\partial h_{t-1}}\frac{partial h_{t-1}}{\partial w_h}
{% endmathjax %}
为了导出上述梯度，假设我们有三个序列{% mathjax %}\{a_t\},\{b_t\},\{c_t\}{% endmathjax %}，当{% mathjax %}t=1,2,\ldots{% endmathjax %}时，序列满足{% mathjax %}a_0=0{% endmathjax %}且{% mathjax %}a_t = b_t + c_ta_{t-1}{% endmathjax %}。对于{% mathjax %}t\geq 1{% endmathjax %}，就很容易得出：
{% mathjax '{"conversion":{"em":14}}' %}
a_t = b_t + \sum_{i=1}^{t-1}(\prod_{j=i+1}^t c_j)b_i
{% endmathjax %}
基于下列公式替换{% mathjax %}a_t,b_t{% endmathjax %}和{% mathjax %}c_t{% endmathjax %}：
\begin{align}
a_t & = \frac{\partial h_t}{\partial w_h} \\ 
b_t & = \frac{\partial f(x_t,h_{t-1},w_h)}{\partial w_h} \\
c_t & = \frac{\partial f(x_t,h_{t-1},w_h)}{\partial h_{t-1}} \\
\end{align}
以上公式中梯度计算，满足{% mathjax %}a_t = b_t + c_ta_{t-1}{% endmathjax %}。因此，我们可以使用下面的公式移除循环计算：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial h_t}{\partial w_h} = \frac{\partial x_t,h_{t-1},w_h}{\partial w_h} + \sum_{i=1}^{t-1}(\prod_{j=i+1}^t \frac{\partial f(x_j,h_{j-1},w_h)}{\partial h_{j-1}})\frac{\partial f(x_i,h_{i-1},w_h)}{\partial w_h}
{% endmathjax %}
虽然我们可以使用链式法则递归地计算{% mathjax %}\partial h_t/\partial w_h{% endmathjax %}但当{% mathjax %}t{% endmathjax %}很大时这个链就会变得很长。我们需要想想办法来处理这一问题。