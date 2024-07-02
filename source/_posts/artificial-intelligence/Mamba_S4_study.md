---
title: 序列化建模：Mamba / S4（深度学习）
date: 2024-06-29 10:00:11
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

序列模型的目标是将输入序列映射到输出序列。我们可以将连续输入序列{% mathjax %}x(t){% endmathjax %}映射到输出序列{% mathjax %}y(t){% endmathjax %}，或者将离散输入序列映射到离散输出序列。
{% asset_img ms_1.png %}
<!-- more -->

例如连续信号可以是音频，离散信号可以是文本，实际上大多数时候我们都使用离散信号，即使在音频的情况下也是如此。当语言建模时，我们考虑的是离散输入，因为只有有限数量的`token`，将其映射到`token`的输出序列，我们可以选择许多模型的来进行序列建模。

第一个模型是**循环神经网络**(`RNN`)。在该网络中，我们有一个隐藏状态，按如下方式计算输出，例如我们输入的序列由{% mathjax %}x_1,x_2{% endmathjax %}和{% mathjax %}x_3{% endmathjax %}组成。初始化隐藏状态的隐藏序列使用`0`来进行初始化。我们将其插入到网络的第一个隐藏状态，因此与第一个输入组合为`0`，然后将产生第一个输出。上一步的输出将产生一个新的隐藏状态和新输入`token`，这将是{% mathjax %}y_2{% endmathjax %}，输出数为`2`。之前生成的隐藏状态与新的输入`token`，用于生成新的输出`token`，以及下一个新隐藏状态的`token`。输出的生成顺序不可以并行化，因为要生成`N`个`token`，我们需要`n-1`个节点。但是每个token的推理时间是不变的，从计算和内存的角度来看，输出中生成的`token`的是相同的。无论它是第一个`token`，还是第`100`个`token`。在这里操作和计算的数量级是一样的，理论上这个输出token拥有无限的上下文长度。但实际情况是不能的，因为它存在梯度消失和爆炸的问题。

另一个模型是**卷积神经网络**(`CNN`)。主要用于计算机视觉任务，它有一个有限的上下文窗口，并需要构建一个可运行的内核，通过输入来产生输出特征的内核，它可以并行化计算，因为每个输出都使用相同的内核。

最后一个模型是`Transformer`,在训练时可以进行并行化计算。拥有计算点积的自注意力机制，由于它是一个矩阵乘法，我们可以并行化该操作。该操作由序列、输入序列、注意力掩码定义的有限上下文窗口。对于这种模型的推理，每个`token`并不是恒定的。在`Transformer`模型中，我们生成第一个`token`输出。我使用`1`个点积来生成；如果我们生成第`10`个输出`token`，我们将需要10个点积来生成它；如果你要生成第`100`个输出`token`，我们将需要100个点积。因此生成第一个`token`的耗时与生成第`10`个`token`的耗时不同。这样不好，这会影响扩展。如果我们将序列长度加倍，则需要花费`4`倍的计算时间来训练该模型。但它可以并行化，我们像`Transformer`一样并行化训练。这样可以很好地利用`GPU`。并像`RNN`一样线性扩展到长序列。

#### 微分方程

让我们用一个非常简单的例子来讨论微分方程。假设你有一些兔子，兔子的数量以恒定速度增长，这意味着每只兔子都会生下{% mathjax %}\lambda{% endmathjax %}只小兔子。因此，我们可以说兔子数量的变化率如下：（特定时间步长{% mathjax %}t{% endmathjax %}出生的小兔子数量）= {% mathjax %}\lambda{% endmathjax %} ×（同一时间步长{% mathjax %}t{% endmathjax %}的兔子数量），则（相对于时间的变化率）= {% mathjax %}\lambda{% endmathjax %} ×（时间{% mathjax %}t{% endmathjax %}时的兔子数量）：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{db}{dt} = \lambda b(t)
{% endmathjax %}
{% mathjax %}\frac{db}{dt}{% endmathjax %}表示兔子数量随时间的变化率。{% mathjax %}b(t){% endmathjax %}表示在时间{% mathjax %}t{% endmathjax %}时我们有多少只兔子。已知{% mathjax %}t = 0{% endmathjax %}时兔子的数量为`5`只，那么如何求出`t = 100`时的兔子数量？我们需要找到描述兔子数量随时间变化的{% mathjax %}b(t){% endmathjax %}。求解微分方程：就是找到一个函数{% mathjax %}b(t){% endmathjax %}，使上述公式对所有{% mathjax %}t{% endmathjax %}值都成立。我们可以验证解为{% mathjax %}b(t) = ke^{\lambda t}{% endmathjax %}，其中{% mathjax %}k = b(0) = 5{% endmathjax %}，即兔子的初始数量。通常，我们通过省略变量{% mathjax %}t{% endmathjax %}来简化微分方程，如下所示：{% mathjax %}\dot{b} = \lambda b{% endmathjax %}。我们通常使用微分方程来模拟系统随时间的变化状态，目的是找到一个函数，给定时间为`0`时系统的初始状态，给出任意时间步的系统状态。

#### 状态空间模型

**状态空间模型**(`State Space Model, SSM`)是一种用于描述动态系统的数学模型,广泛应用于控制理论和信号处理等领域。**状态空间模型**通过一组一阶微分方程来描述系统的动态行为。它由以下两个基本方程组成。
- 状态方程: {% mathjax %}h(t) = \mathbf{A}(t)h(t) + \mathbf{B}(t)x(t){% endmathjax %}。
- 输出方程: {% mathjax %}y(t) = \mathbf{C}(t)h(t) + \mathbf{D}(t)x(t){% endmathjax %}。

其中{% mathjax %}h(t){% endmathjax %}是状态向量，{% mathjax %}x(t){% endmathjax %}是输入向量，{% mathjax %}y(t){% endmathjax %}是输出向量，{% mathjax %}\mathbf{A}(t){% endmathjax %}是状态矩阵，{% mathjax %}\mathbf{B}(t){% endmathjax %}是输入矩阵，{% mathjax %}\mathbf{C}(t){% endmathjax %}是输出矩阵，{% mathjax %}\mathbf{D}(t){% endmathjax %}是直接传递矩阵。该状态空间模型是线性的和时间不变的。线性是因为上述表达式中的关系是线性的，时间不变是因为参数矩阵`A、B、C、D`不依赖于时间（它们是固定的）。为了找到时间{% mathjax %}t{% endmathjax %}时的输出信号{% mathjax %}y(t){% endmathjax %}，我们首先需要找到一个函数{% mathjax %}h(t){% endmathjax %}，该函数描述系统在所有时间步骤的状态。但这很难通过分析解决。通常我们从不使用连续信号，而总是使用离散信号（因为我们对其进行采样），那么我们如何为离散信号产生输出{% mathjax %}y(t){% endmathjax %}呢？我们首先需要将系统离散化！
在实际应用中,通常需要将连续时间状态空间模型离散化,以便于数字计算机处理。离散化是状态空间模型中非常重要的一步,它使得模型可以从连续视角转换为递归视角和卷积视角。

#### 离散化

**离散化**是一种将连续数据或模型转换为离散形式的过程，目的是简化计算和分析,同时尽可能保持准确性。要解微分方程，我们需要找到使方程两边相等的函数{% mathjax %}h(t){% endmathjax %}，但大多数时候很难找到微分方程的解析解，这就是为什么我们可以近似解微分方程。找到微分方程的近似解意味着找到{% mathjax %}h(0)、h(1)、h(2)、h(3){% endmathjax %}等序列，描述我们的系统随时间的变化。因此，我们不是要找到{% mathjax %}h(t){% endmathjax %}，而是要找到{% mathjax %}h(t_k) = h(k\Delta){% endmathjax %}，其中{% mathjax %}\Delta{% endmathjax %}是我们的步长。还记得兔子问题吗？让我们尝试使用**欧拉方法**找到近似解。
- 首先让我们重写我们的兔子种群模型：{% mathjax %}b'(t) = \lambda b(t){% endmathjax %}。
- 函数的导数是函数的变化率，即，{% mathjax %}\lim_{\Delta\rightarrow 0}\frac{b(t + \Delta) -b(t)}{\Delta} = b'(t){% endmathjax %}，一次通过选择较小的步长{% mathjax %}\Delta{% endmathjax %}，我们可以摆脱这个极限：{% mathjax %}\frac{b(t + \Delta) -b(t)}{\Delta} \cong b'(t){% endmathjax %}通过与{% mathjax %}\Delta{% endmathjax %}相乘并且移动项，我们可以写出：{% mathjax %}b(t + \Delta) \cong b'(t)\Delta + b(t){% endmathjax %}。
- 然后，我们可以将兔子种群模型带入到上面的公式中，得到：{% mathjax %}b(t + \Delta) \cong \lambda b(t)\Delta + b(t){% endmathjax %}。
- 最后得到了一个循环公式。

让我们使用递归公式来近似兔子种群随时间的状态变化：{% mathjax %}b(t + \Delta) \cong \lambda b(t)\Delta + b(t){% endmathjax %}，条件假设{% mathjax %}\lambda = 2, \Delta = 1{% endmathjax %}。例如，如果我们在时间{% mathjax %}t = 0{% endmathjax %}时从{% mathjax %}5{% endmathjax %}只兔子开始，我们可以按如下方式计算种群的演变：
- 知道时间{% mathjax %}t = 0{% endmathjax %}时的数量，我们可以计算时间{% mathjax %}t = 1{% endmathjax %}时的数量：{% mathjax %}b(1) = \Delta\lambda b(0) + b(0) = 1\times 2\times 5 + 5 = 15{% endmathjax %}。
- 知道时间{% mathjax %}t = 1{% endmathjax %}时的数量，我们可以计算时间{% mathjax %}t = 2{% endmathjax %}时的数量：{% mathjax %}b(2) = \Delta\lambda b(1) + b(1) = 1\times 2\times 15 + 15 = 45{% endmathjax %}。
- 知道时间{% mathjax %}t = 2{% endmathjax %}时的数量，我们可以计算时间{% mathjax %}t = 3{% endmathjax %}时的数量：{% mathjax %}b(3) = \Delta\lambda b(2) + b(2) = 1\times 2\times 45 + 45 = 135{% endmathjax %}。

步长{% mathjax %}\Delta{% endmathjax %}的值越小，相对于解析解{% mathjax %}b(t) = 5e^{\lambda t}{% endmathjax %}的近似值越好。通过使用与兔子场景类似的推理，我们也可以离散化我们的状态空间模型，以便我们可以使用递归公式计算状态。
- 利用导数的定义，知道：{% mathjax %}h(t + \Delta) \cong \Delta h'(t) + h(t){% endmathjax %}。
- 这是（连续）状态空间模型：{% mathjax %}h'(t) = \mathbf{A}h(t) + \mathbf{B}x(t){% endmathjax %}。
- 我们可以将状态空间模型代入第一个表达式，得到以下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
h(t + \Delta) & \cong \Delta(\mathbf{A}h(t) \mathbf{B}x(t)) + h(t) \\
& = \Delta\mathbf{A}h(t) + \Delta\mathbf{B}x(t) + h(t) \\
& = (1 + \Delta\mathbf{A})h(t) + \Delta\mathbf{B}x(t) \\
& = \bar{\mathbf{A}}h(t) + \bar{\mathbf{B}}x(t)
\end{aligned}
{% endmathjax %}
有了一个递归公式，在知道前一个时间步的状态的情况下一步一步地迭代计算系统的状态。矩阵{% mathjax %}\mathbf{A}{% endmathjax %}和{% mathjax %}\mathbf{B}{% endmathjax %}是模型的离散参数。这还允许我们计算离散时间步长的输出{% mathjax %}y(t){% endmathjax %}。
{% asset_img ms_2.png %}

在论文中，他们没有使用欧拉方法，而是使用`Zero-Order Hold`(`ZOH`)规则来离散系统的。
{% asset_img ms_3.png %}

注意：在实践中，我们不选择对{% mathjax %}\Delta{% endmathjax %}离散化，而是将其作为模型的参数，以便可以通过梯度下降进行学习。

#### 递归计算

现在我们有了递归公式，那么如何使用它来计算系统在不同时间步长的输出呢？为简单起见，假设系统的初始状态为`0`。
{% mathjax '{"conversion":{"em":14}}' %}
h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}x_t,\;\; y_t = \mathbf{C}h_t
{% endmathjax %}
{% asset_img ms_4.png %}

递归计算存在的问题：递归公式非常适合推理，因为我们可以在恒定的内存/计算下一个`token`。这使得它适合在大型语言模型中进行推理，在这种模型中，我们希望根据提示词和之前生成的`token`生成下一个`token`。但是，递归公式并不适合训练，因为在训练期间我们已经拥有了输入和目标的所有`token`，所以我们希望使用并行化计算，就像`Transformer`一样。值得庆幸的是，**状态空间模型**也提供了卷积模式。

#### 卷积计算

让我们扩展之前为每个时间步构建的输出。
{% mathjax '{"conversion":{"em":14}}' %}
h_t = \bar{\mathbf{A}}h_{t-1} + \bar{\mathbf{B}}x_t,\;\; y_t = \mathbf{C}h_t
{% endmathjax %}
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
h_0 & = \bar{\mathbf{B}}x_0 \\
y_0 & = \mathbf{C}h_0 = \mathbf{C}\bar{\mathbf{B}}x_0 \\
\\
h_1 & = \bar{\mathbf{A}}h_0 + \bar{\mathbf{B}}x_1 = \bar{\mathbf{A}}\bar{\mathbf{B}}x_0 + \bar{\mathbf{B}}x_1 \\
y_1 & = \mathbf{C}h_1 = \mathbf{C}(\bar{\mathbf{A}}\bar{\mathbf{B}}x_0 + \bar{\mathbf{B}}x_1) = \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x_0 + \mathbf{C}\bar{\mathbf{B}}x_1 \\
\\
h_2 & = \bar{\mathbf{A}}h_1 + \bar{\mathbf{B}}x_2 = \bar{\mathbf{A}}(\bar{\mathbf{A}}\bar{\mathbf{B}}x_0 + \bar{\mathbf{B}}x_1 ) + \bar{\mathbf{B}}x_2 = \bar{\mathbf{A}}^2\bar{\mathbf{B}}x_0 + \bar{\mathbf{A}}\bar{\mathbf{B}}x_1 + \bar{\mathbf{B}}x_2 \\
y_2 & = \mathbf{C}h_2 = \mathbf{C}(\bar{\mathbf{A}}^2\bar{\mathbf{B}}x_0 + \bar{\mathbf{A}}\bar{\mathbf{B}}x_1 + \bar{\mathbf{B}}x_2) = \mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}}x_0 + \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x_1 + \mathbf{C}\bar{\mathbf{B}}x_2 \\
\\
y_k & = \mathbf{C}\bar{\mathbf{A}}^k\bar{\mathbf{B}}x_0 + \mathbf{C}\bar{\mathbf{A}}^{k-1}\bar{\mathbf{B}}x_1 + \ldots + \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x_{k-1} + \mathbf{C}\bar{\mathbf{B}}x_k
\end{aligned}
{% endmathjax %}
通过使用我们推导出的公式，我们注意到一些有趣的事情：系统的输出可以通过内核{% mathjax %}\bar{K}{% endmathjax %}与输入{% mathjax %}x(t){% endmathjax %}的卷积来计算。
{% asset_img ms_5.png %}

#### 卷积公式

第一步：{% mathjax %}\mathbf{C}\bar{\mathbf{B}}x_0{% endmathjax %}
{% asset_img ms_6.png %}

第二步：{% mathjax %}\mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x_0 + \mathbf{C}\bar{\mathbf{B}}x_1{% endmathjax %}
{% asset_img ms_7.png %}

第三步：{% mathjax %}\mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}}x_0 + \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x_1 + \mathbf{C}\bar{\mathbf{B}}x_2{% endmathjax %}
{% asset_img ms_8.png %}

第四步：{% mathjax %}\mathbf{C}\bar{\mathbf{A}}^3\bar{\mathbf{B}}x_0 + \mathbf{C}\bar{\mathbf{A}}^2\bar{\mathbf{B}}x_1 + \mathbf{C}\bar{\mathbf{A}}\bar{\mathbf{B}}x_2 + \mathbf{C}\bar{\mathbf{B}}x_3{% endmathjax %}
{% asset_img ms_9.png %}

卷积计算的最大优点是它可以并行化，因为输出{% mathjax %}y_k{% endmathjax %}不依赖于{% mathjax %}y_{k-1}{% endmathjax %}，而只依赖于内核和输入。但是，从内存的角度来看，实现（构建）内核的计算成本可能很高。
- 我们可以使用卷积计算进行训练，因为我们已经拥有所有输入的`token`序列，并且可以轻松并行化。
- 我们可以使用循环公式进行推理，一次一个`token`，每个`token`使用恒定量的计算和内存。

到目前为止，我们研究的状态空间模型只为每个输入`token`（由一个数字表示）计算一个输出。当输入/输出信号是矢量时，我们该如何工作？**状态空间模型**：每个维度都由独立的状态空间模型管理！
{% asset_img ms_10.png %}

**Transformer**：每组维度分别由多头注意力机制的不同头进行管理！
{% asset_img ms_11.png %}

当然，我们可以通过处理批量输入来并行化所有这些操作。这样，参数`A、B、C、D`和输入{% mathjax %}x(t){% endmathjax %}和{% mathjax %}y(t){% endmathjax %}就变成了向量和矩阵。这样，所有维度的计算都将并行完成。**状态空间模型**中的`A`矩阵可以直观地看作是从前一个状态“捕获”信息以构建新状态的矩阵。它还决定了这些信息如何随时间向前复制。这意味着我们需要小心`A`矩阵的结构，否则它可能无法很好地捕获看到的所有输入的历史记录，而这是产生下一个输出必要条件。这对于语言模型非常重要：模型生成的下一个`token`应该取决于前一个`token`。为了让`A`矩阵表现良好，作者选择使用`HIPPO`理论。让我们看看它是如何工作的。傅里叶变换使我们能够将信号分解为正弦函数，使得这些函数的总和可以（很好地）近似于原始信号。利用`HIPPO`理论，我们可以做类似的事情，但我们使用的不是正弦函数，而是`Legendre`多项式。

#### HIPPO

利用`HIPPO`理论，我们以这样一种方式构建`A`矩阵，即它将看到的所有输入信号近似为一个系数向量（`Legendre`多项式）。与傅里叶变换的不同之处在于，我们不是完美地构建所有原始信号，而是非常精确地构建较新的信号，而较旧的信号则呈指数衰减（如`EMA`）。因此，状态{% mathjax %}h(t){% endmathjax %}捕获的有关最近看到的`token`的信息比更早的`token`多。
{% asset_img ms_12.png %}

#### Mamba

`SSM`或者`S4`（结构化状态空间模型）表现不佳。一次重写一个`token`的输入，但进行时间移位。这可以通过`SSM`来执行，并且可以通过卷积来学习时间延迟。给定`Twitter`上的一条评论，通过删除所有脏话（白色标记）来重写该评论。这不能由`SSM`来执行，因为它需要内容感知推理，而`SSM`无法做到这一点，因为它们是时间不变的（这意味着参数`A、B、C、D`对于它生成的每个`token`都是相同的）。
{% asset_img ms_13.png %}

例如，通过“少量”提示词，我们可以“教”`LLM`新任务如何执行。使用基于`Transformer`的模型，可以“轻松”完成此任务，因为基于`Transformer`的模型可以在生成当前`token`时关注先前的`token`，因此它们可以“回忆以前的历史”。时间不变的`SSM`无法执行此任务，因为它们无法“选择”从其历史中回忆先前的`token`。
##### 选择性SSM

{% asset_img ms_14.png %}

`Mamba`无法使用卷积进行评估，因为模型的参数对于每个输入`token`都不同，即使我们想运行卷积，我们也需要构建`L`（序列长度）个不同的卷积核，这从内存/计算的角度来看太复杂。您是否注意到作者在评估递归时谈到了“扫描”操作？`B`：批次大小、`L`：序列长度、`D`：输入向量的大小（相当于`Transformer`中的`d_model`）、`N`：隐藏状态`h`的大小。
##### 扫描操作

如果你曾经参加过竞技编程，那么你一定对前缀和数组很熟悉，这是一个按顺序计算的数组，每个位置的值表示所有前面值的总和。我们可以用线性时间的`for`循环轻松计算它。扫描操作是指计算类似`Prefix-Sum`的数组，其中每个值都可以通过之前计算的值和当前输入来计算。`SSM`模型的递归公式也可以看作是一种扫描操作，其中每个状态都是前一个状态与当前输入的和。为了生成输出，我们只需将每个{% mathjax %}h_k{% endmathjax %}与矩阵`C`相乘即可生成输出标记{% mathjax %}y_k{% endmathjax %}。
##### 并行扫描

如果我告诉您扫描操作可以并行化，您会怎么想？您可能不相信，但可以！只要我们执行的操作具有结合性（即操作受益于结合性）。结合性简单地说就是{% mathjax %}A * B * C = (A * B) * C = A * (B * C){% endmathjax %}，因此我们执行操作的顺序无关紧要。
{% asset_img ms_15.png %}

我们可以生成多个线程来并行执行二元运算，并且每个步骤进行同步。时间复杂度不再是{% mathjax %}\mathcal{O}(N){% endmathjax %}，而是降低为{% mathjax %}\mathcal{O}(N/T){% endmathjax %}，其中{% mathjax %}T{% endmathjax %}是并行线程的数量。
##### 选择性扫描

由于`Mamba`无法使用卷积进行评估（因为它是时变的），因此无法并行化。我们唯一的计算方法是使用递归公式，但由于并行扫描算法，它可以并行化。
##### GPU的内存结构

由于`GPU`在移动张量时速度不是很快，但在计算操作时速度却非常快，因此有时，我们算法中的问题可能不是我们进行的计算次数，而是我们在不同的内存层次结构中移动了多少张量。在这种情况下，我们称该操作是`IO`密集型的。
{% asset_img ms_16.png %}

{% asset_img ms_17.png %}

##### 内核融合

当我们执行张量运算时，我们的深度学习框架（例如`PyTorch`）将张量加载到`GPU`的快速内存(`SRAM`)中，执行运算（例如矩阵乘法），然后将结果保存回`GPU`的高带宽内存中。如果我们对同一个张量执行多个运算（例如`3`个运算）会怎样？然后深度学习框架将执行以下序列：
- 将输入从`HBM`加载到`SRAM`，计算第一个操作（第一个操作对应的`CUDA`内核），然后将结果保存回`HBM`
- 将上一次结果从`HBM`加载到`SRAM`，计算第二个操作（第二个操作对应的`CUDA`内核），然后将结果保存回`HBM`
- 将上一次结果从`HBM`加载到`SRAM`，计算第三个操作（第三个操作对应的`CUDA`内核），然后将结果保存回`SRAM`。

如您所见，总时间都被我们正在执行的复制操作所占用，因为我们知道`GPU`在复制数据方面比计算操作速度相对较慢。为了使一系列操作更快，我们可以融合`CUDA`内核来生成一个自定义`CUDA`内核，该内核依次执行三个操作，而不是将中间结果复制到`HBM`，而只复制最终结果。
##### 重新计算

当我们训练深度学习模型时，它会转换成计算图。当我们执行反向传播时，为了计算每个节点的梯度，我们需要缓存前向步骤的输出值，如下所示：
{% asset_img ms_18.png %}

由于缓存激活然后在反向传播期间重新使用它们，意味着我们需要将它们保存到`HBM`，然后在反向传播期间将它们从`HBM`复制回来，因此在反向传播期间重新计算它们可能会更快！
##### Mamba块

`Mamba`是通过堆叠`Mamba`块的多个层来构建的，如下所示。这与`Transformer`模型的堆叠层非常相似。`Mamba`架构源自`Hungry Hungry Hippo (H3)`架构。
{% asset_img ms_19.png %}

##### Mamba模型架构

{% asset_img ms_20.png %}

##### Mamba性能

{% asset_img ms_21.png %}

{% asset_img ms_22.png %}

