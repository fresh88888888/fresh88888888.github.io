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

