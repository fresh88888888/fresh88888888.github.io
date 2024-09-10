---
title: 深度学习(DL)(四) — 探析
date: 2024-09-09 18:15:11
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

#### 自注意力

要将**自注意力**与`CNN`一起使用，需要计算**自注意力**，即为输入句子中的每个单词创建基于**注意力**的表示。示例`Jane, visite, l'Afrique, en, septembre`，我们的目标是为每个单词计算一个基于注意力的表示。最终会得到五个，因为句子有五个单词。即{% mathjax %}A^{<1>},\ldots,A^{<5>}{% endmathjax %}。然后对句子中的各个单词进行计算。表示`l'Afrique`的一种方法是查找`l'Afrique`的词嵌入。根据对`l'Afrique`的理解，可以选择不同的方式来表示它({% mathjax %}A^{<3>}{% endmathjax %})。它将查看周围的单词，试图弄清楚在这个句子中的含义，并找到最合适的表示。就实际计算而言，它与之前在`RNN`上下文中看到的**注意力机制**没有太大区别，只是并行计算句子中所有单词的表示。
<!-- more -->

在`RNN`之上构建注意力时，使用了以下方程{% mathjax %}\alpha^{<t,t'>} = \frac{\text{exp}(e^{<t,t'>})}{\sum_{t'=1}^{T_x}\text{exp}(e^{<t,t'>})}{% endmathjax %}。使用`Transformer`的**自注意力机制**，方程将如下所示{% mathjax %}\mathbf{A}(q,K,V) = \sum_i \frac{\text{exp}(q\cdot k^{<i>})}{\sum_j \text{exp}(q\cdot k^{<j>})}v^{<i>}{% endmathjax %}。可以看到这些方程有一些相似之处。这里的内部项也涉及`softmax`。但主要的区别在于，比如`l'Afrique`，有三个值，称为`Query、Key`和`Value`。这些向量是计算每个单词的**注意力值**的关键输入。让我们逐步完成从单词l`'Afrique`到自注意力表示{% mathjax %}A^{<3>}{% endmathjax %}所需的计算。首先，将每个单词与三个值关联起来，如果{% mathjax %}X^{<3>}{% endmathjax %}是`l'Afrique`的**单词嵌入**，则计算{% mathjax %}Q^{<3>} = WQ \times X^{<3>}{% endmathjax %}是作为一个学习矩阵，键和值也是如此，{% mathjax %}K^{<3>} = WK \times X^{<3>}{% endmathjax %}，{% mathjax %}V^{<3>} = WV \times X^{<3>}{% endmathjax %}。这些矩阵{% mathjax %}WQ,WK{% endmathjax %}和{% mathjax %}WV{% endmathjax %}是此学习算法的参数。那么这些查询，键和值向量应该做什么呢？{% mathjax %}Q^{<3>}{% endmathjax %}是要问的关于非洲的问题。{% mathjax %}Q^{<3>}{% endmathjax %}可能代表这样的问题，例如，那里发生了什么？您可能想知道在计算{% mathjax %}A^{<3>}{% endmathjax %}时，发生了什么。计算{% mathjax %}q^{<3>}\cdot k^{<1>}{% endmathjax %}之间的内积，答案为`1`时问题的答案有多好。{% mathjax %}q^{<3>}\cdot k^{<2>}{% endmathjax %}，答案为`1`时问题的答案有多好，此操作的目的是提取信息，并计算出此处最有用的表示{% mathjax %}A^{<3>}{% endmathjax %}。如果{% mathjax %}k^{<1>}{% endmathjax %}表示这个词是一个人，因为`Jane`是一个人，而{% mathjax %}k^{<2>}{% endmathjax %}表示第二个词`visite`是一个 `action`，那么会发现{% mathjax %}q^{<3>}\cdot k^{<2>}{% endmathjax %}的内积具有最大值，取这一行中的这五个值，并对它们计算`softmax`。{% mathjax %}q^{<3>}\cdot k^{<2>}{% endmathjax %}对应于单词`visite`具有最大值。将这些`softmax`值与{% mathjax %}v^{<1>},v^{<2>}{% endmathjax %}相乘。最后，它们全部加起来。因此将所有这些值相加会得到{% mathjax %}A^{<3>}{% endmathjax %}。这种表示的主要优势在于`l'Afrique`这个词不是某种固定的**词嵌入**。相反，它让**自注意力机制**意识到`l'Afrique`是访问的目的地，从而为这个词计算出更丰富、更有用的表示。如果将这五个计算放在一起，文献中使用的表示法如下所示，其中{% mathjax %}Q,K,V{% endmathjax %}是包含所有这些值的矩阵，这只是此处方程的压缩或矢量化表示{% mathjax %}\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V}) = \text{softmax}(\frac{\mathbf{QK}^{\mathsf{T}}}{\sqrt{d_k}}) \mathbf{V}{% endmathjax %}。分母中的项只缩放点积，因此它不会爆炸。但这种注意力的另一个名称是**缩放点积注意力**。总结一下，与这五个单词中的每一个相关联，您最终会得到一个查询、一个键和一个值。查询可以让您提出有关该单词的问题，例如非洲发生了什么。关键是查看所有其他单词，并通过与查询的相似性，帮助您找出哪些单词给出了与该问题最相关的答案。
{% asset_img dl_1.png %}

