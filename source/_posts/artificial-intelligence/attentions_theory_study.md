---
title: 理解自注意力&多头注意力&交叉注意力&因果注意力（深度学习）
date: 2024-06-17 15:50:11
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

#### 自注意力机制

自注意力等相关机制是LLM的核心组成部分。深度学习中的“注意力”概念源于改进循环神经网络(`RNN`)以处理较长的序列或句子所做的努力。例如，考虑将一个句子从一种语言翻译成另一种语言。逐字翻译一个句子通常不是一种选择，因为它忽略了每种语言独有的复杂语法结构和惯用表达，导致翻译不准确或无意义。
{% asset_img a_1.png  %}
<!-- more -->

为了解决这个问题，我们引入了注意力机制，以便在每个时间步骤中访问所有序列元素。关键是要有选择性，并确定哪些词在特定上下文中是最重要的。`2017`年，`Transformer`架构引入了独立的自注意力机制，完全消除了对`RNN`的需要。我们可以将自注意力机制视为一种通过包含有关输入上下文的信息来增强输入嵌入信息内容的机制。换句话说，自注意力机制使模型能够衡量输入序列中不同元素的重要性，并动态调整它们对输出的影响。这对于语言处理任务尤为重要，因为单词的含义可以根据句子或文档中的上下文而改变。

{% note warning %}
**注意**，自注意力机制有很多变体。其中特别关注的是提高自注意力机制的效率。然而，大多数论文仍然采用**缩放点积注意力机制**，因为它通常能带来更高的准确率，而且对于大多数训练大规模 `Transformer`的公司来说，自注意力机制很少成为计算瓶颈。
{% endnote %}
我们重点介绍原始的**缩放点积注意力机制**（称为自注意力机制），它仍然是实践中最流行、应用最广泛的注意力机制。

#### 嵌入输入句子

在开始之前，我们先考虑一个输入句子“`Life is short, eat dessert first`”，我们想将其放入自注意力机制中。与其他类型的文本处理建模方法（例如，使用循环神经网络或卷积神经网络）类似，我们首先创建一个句子嵌入。为简单起见，此处我们的词典`dc`仅限于输入句子中出现的单词。在实际应用中，我们会考虑训练数据集中的所有单词（典型词汇量在`30k~50k`之间）。
输入：
```python
sentence = 'Life is short, eat dessert first'
dc = {s:i for i,s in enumerate(sorted(sentence.replace(',', '').split()))}
print(dc)
```
输出结果为：
```bash
{'Life': 0, 'dessert': 1, 'eat': 2, 'first': 3, 'is': 4, 'short': 5}
```
接下来，我们使用这本词典为每个单词分配一个整数索引：
```python
import torch

sentence_int = torch.tensor([dc[s] for s in sentence.replace(',', '').split()])
print(sentence_int)
```
输出结果为：
```bash
tensor([0, 4, 5, 2, 1, 3])
```
现在，使用输入句子的整数向量表示，我们可以使用嵌入层将输入编码为实向量嵌入。在这里，我们将使用`16`维嵌入，这样每个输入单词都由一个`16`维向量表示。由于该句子由`6`个单词组成，因此这将导致{% mathjax %}6\times 16{% endmathjax %}维嵌入。
```python
import torch

torch.manual_seed(123)
embed = torch.nn.Embedding(6, 16)
embedded_sentence = embed(sentence_int).detach()

print(embedded_sentence)
print(embedded_sentence.shape)
```
结果输出为：
```bash
tensor([[ 0.3374, -0.1778, -0.3035, -0.5880,  0.3486,  0.6603, -0.2196, -0.3792,
          0.7671, -1.1925,  0.6984, -1.4097,  0.1794,  1.8951,  0.4954,  0.2692],
        [ 0.5146,  0.9938, -0.2587, -1.0826, -0.0444,  1.6236, -2.3229,  1.0878,
          0.6716,  0.6933, -0.9487, -0.0765, -0.1526,  0.1167,  0.4403, -1.4465],
        [ 0.2553, -0.5496,  1.0042,  0.8272, -0.3948,  0.4892, -0.2168, -1.7472,
         -1.6025, -1.0764,  0.9031, -0.7218, -0.5951, -0.7112,  0.6230, -1.3729],
        [-1.3250,  0.1784, -2.1338,  1.0524, -0.3885, -0.9343, -0.4991, -1.0867,
          0.8805,  1.5542,  0.6266, -0.1755,  0.0983, -0.0935,  0.2662, -0.5850],
        [-0.0770, -1.0205, -0.1690,  0.9178,  1.5810,  1.3010,  1.2753, -0.2010,
          0.4965, -1.5723,  0.9666, -1.1481, -1.1589,  0.3255, -0.6315, -2.8400],
        [ 0.8768,  1.6221, -1.4779,  1.1331, -1.2203,  1.3139,  1.0533,  0.1388,
          2.2473, -0.8036, -0.2808,  0.7697, -0.6596, -0.7979,  0.1838,  0.2293]])
torch.Size([6, 16])
```
#### 定义权重矩阵

让我们讨论一下被广泛使用的自注意力机制，即缩放点积注意力机制，它被集成到`Transformer`架构中。自注意力机制使用三个权重矩阵{% mathjax %}\mathbf{W}_q,\mathbf{W}_k{% endmathjax %}和{% mathjax %}\mathbf{W}_v{% endmathjax %}，它们在训练期间作为模型参数进行调整。这些矩阵分别用于将输入投影到序列的查询、键和值组件中。通过权重矩阵{% mathjax %}\mathbf{W}{% endmathjax %}之间的矩阵乘法获得各自的查询、键和值序列以及嵌入的输入{% mathjax %}\mathbf{x}{% endmathjax %}。
- 查询序列：{% mathjax %}\mathbf{q}^{(i)} = \mathbf{W}_q\mathbf{x}^{(i)}\text{ for }i\in [1,T]{% endmathjax %}。
- 键序列：{% mathjax %}\mathbf{k}^{(i)} = \mathbf{W}_k\mathbf{x}{(i)}\text{ for }i\in [1,T]{% endmathjax %}。
- 值序列：{% mathjax %}\mathbf{v}^{(i)} = \mathbf{W}_v\mathbf{x}^{(i)}\text{ for }i\in [1,T]{% endmathjax %}。

索引{% mathjax %}i{% endmathjax %}指索引序列`token`的索引位置。其长度为{% mathjax %}T{% endmathjax %}
{% asset_img a_2.png  %}

这里的{% mathjax %}\mathbf{q}^{(i)},\mathbf{k}^{(i)}{% endmathjax %}是维度{% mathjax %}d_k{% endmathjax %}的两个向量。投影矩阵{% mathjax %}\mathbf{W}_q{% endmathjax %}和{% mathjax %}\mathbf{W}_k{% endmathjax %}的形状为{% mathjax %}d_k\times d{% endmathjax %}，而{% mathjax %}\mathbf{W}_v{% endmathjax %}的形状为{% mathjax %}d_v\times d{% endmathjax %}（{% mathjax %}d{% endmathjax %}表示每个词向量的大小）。由于我们正在计算查询和键向量之间的点积，因此这两个向量必须包含相同数量的元素({% mathjax %}d_q = d_k{% endmathjax %})。当然，值向量{% mathjax %}\mathbf{v}_{(i)}{% endmathjax %}中的元素数量决定了得到的上下文向量的大小。我们这里假设{% mathjax %}d_q= d_k = 24{% endmathjax %}，并且{% mathjax %}d_v = 28{% endmathjax %}，初始化投影矩阵如下：
```python
torch.manual_seed(123)

d_q, d_k, d_v = 24, 24, 28
d = embedded_sentence.shape[1]
W_query = torch.nn.Parameter(torch.rand(d_q, d))
W_key = torch.nn.Parameter(torch.rand(d_k, d))
W_value = torch.nn.Parameter(torch.rand(d_v, d))
```
#### 计算非规范化注意力权重

现在，假设我们想计算第二个输入元素的注意向量—第二个输入元素在这里充当查询：
{% asset_img a_3.png  %}

在代码中，它看起来如下所示：
```python
x_2 = embedded_sentence[1]
query_2 = W_query.matmul(x_2)
key_2 = W_key.matmul(x_2)
value_2 = W_value.matmul(x_2)

print(query_2.shape)
print(key_2.shape)
print(value_2.shape)

# torch.Size([24])
# torch.Size([24])
# torch.Size([28])
```
然后，我们可以将其推广到计算所有输入的剩余键和值元素，因为在下一步计算非规范化注意力权重时，我们将需要{% mathjax %}\omega{% endmathjax %}。
```python
keys = W_key.matmul(embedded_sentence.T).T
values = W_value.matmul(embedded_sentence.T).T

print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

# keys.shape: torch.Size([6, 24])
# values.shape: torch.Size([6, 28])
```
现在我们有了所有的键和值，我们可以继续下一步，计算非规范化的注意力权重{% mathjax %}\omega{% endmathjax %}。如下图所示：
{% asset_img a_4.png  %}

如上图所示，我们计算的{% mathjax %}\omega_{i,j}{% endmathjax %}是查询和键序列的点积，{% mathjax %}\omega_{i,j} = \mathbf{q}^{(i)^\top}\mathbf{k}^{(j)}{% endmathjax %}。例如，我们可以计算查询和第`5`个输入元素（对应索引位置`4`）的非规范化注意力权重，如下所示：
```python
omega_24 = query_2.dot(keys[4])
print(omega_24)

# tensor(11.1466)
```
由于我们稍后需要它们来计算注意力分数，因此我们来计算{% mathjax %}\omega{% endmathjax %}所有输入`token`的值如上图所示：
```python
omega_2 = query_2.matmul(keys.T)
print(omega_2)

# tensor([ 8.5808, -7.6597,  3.2558,  1.0395, 11.1466, -0.4800])
```
#### 计算注意力分数

自注意力的后续步骤是对未规范化的注意力权重进行规范化，{% mathjax %}\omega{% endmathjax %}获得规范化注意力权重，{% mathjax %}\alpha{% endmathjax %}应`用softmax`函数。此外，{% mathjax %}1/\sqrt{d_k}{% endmathjax %}用于扩展{% mathjax %}\omega{% endmathjax %}通过softmax函数对其进行规范化，如下所示：
{% asset_img a_5.png  %}

