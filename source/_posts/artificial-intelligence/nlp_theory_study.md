---
title: 自然语言处理 (预训练)
date: 2024-05-30 10:04:11
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

人与人之间需要交流。出于人类这种基本需要，每天都有大量的书面文本产生。 比如，社交媒体、聊天应用、电子邮件、产品评论、新闻文章、研究论文和书籍中的丰富文本，使计算机能够理解它们以提供帮助或基于人类语言做出决策变得至关重要。**自然语言处理是指研究使用自然语言的计算机和人类之间的交互**。要理解文本，我们可以从学习它的表示开始。利用来自大型语料库的现有文本序列，自监督学习(`self-supervised learning`)已被广泛用于预训练文本表示，例如通过使用周围文本的其它部分来预测文本的隐藏部分。通过这种方式，模型可以通过有监督地从海量文本数据中学习，而不需要昂贵的标签标注！
<!-- more -->
当每个单词或子词被视为单个**词元**时，可以在大型语料库上使用`word2vec、GloVe`或子词嵌入模型预先训练每个词元的词元。经过预训练后，每个词元的表示可以是一个向量。但是，无论上下文是什么，它都保持不变。例如，“bank”（可以译作银行或者河岸）的向量表示在 “`go to the bank to deposit some money`”（去银行存点钱）和“`go to the bank to sit down`”（去河岸坐下来）中是相同的。因此，许多较新的预训练模型使相同词元的表示适应于不同的上下文，其中包括基于`Transformer`编码器的更深的自监督模型`BERT`。
{% asset_img nlp_1.png "预训练好的文本表示可以放入各种深度学习架构，应用于不同自然语言处理任务" %}

#### 词嵌入（word2vec）

自然语言是用来表达人脑思维的复杂系统。在这个系统中，**词**是意义的基本单元。顾名思义，**词向量**是用于表示单词意义的向量，并且还可以被认为是单词的特征向量或表示。将单词映射到实向量的技术称为**词嵌入**。
##### 自监督的 word2vec

`word2vec`是为了解决上述问题而提出的。它将每个词映射到一个固定长度的向量，这些向量能更好地表达不同词之间的相似性和类比关系。`word2vec`包含两个模型，即跳元模型(`skip-gram`)和连续词袋(`CBOW`)。对于在语义上是有意义的表示，它们的训练依赖于条件概率，条件概率可以被看作使用语料库中一些词来预测另一些单词。由于是不带标签的数据，因此跳元模型和连续词袋都是自监督模型。
##### 跳元模型（Skip-Gram）

跳元模型假设一个词可以用来在文本序列中生成其周围的单词。以文本序列`“the”“man”“loves”“his”“son”`为例。假设中心词选择`“loves”`，并将上下文窗口设置为`2`，如下图所示，给定中心词`“loves”`，跳元模型考虑生成上下文词`“the”“man”“him”“son”`的条件概率：
{% mathjax '{"conversion":{"em":14}}' %}
P(\text{"the","man","his","son"}|\text{"loves"})
{% endmathjax %}
假设上下文词是在给定中心词的情况下独立生成的（即条件独立性）。在这种情况下，上述条件概率可以重写为：
{% mathjax '{"conversion":{"em":14}}' %}
P(\text{"the"}|\text{"loves"})\cdot P(\text{"man"}|\text{"loves"})\cdot P(\text{"his"}|\text{"loves"})\cdot P(\text{"son"}|\text{"loves"})
{% endmathjax %}
{% asset_img nlp_2.png "跳元模型考虑了在给定中心词的情况下生成周围上下文词的条件概率" %}

在跳元模型中，每个词都有两个{% mathjax %}d{% endmathjax %}维向量表示，用于计算条件概率。更具体地说，对于词典中索引为{% mathjax %}i{% endmathjax %}的任何词，分别用{% mathjax %}\mathbf{v}_i\in \mathbb{R}^d{% endmathjax %}和{% mathjax %}\mathbf{u}_i\in \mathbb{R}^d{% endmathjax %}表示其用作中心词和上下文词时的两个向量。给定中心词{% mathjax %}w_c{% endmathjax %}（词典中的索引{% mathjax %}c{% endmathjax %}），生成任何上下文词{% mathjax %}w_o{% endmathjax %}（词典中的索引{% mathjax %}o{% endmathjax %}）的条件概率可以通过对向量点积的`softmax`操作来建模：
{% mathjax '{"conversion":{"em":14}}' %}
p(w_o|w_c) = \frac{\exp(\mathbf{u}_o^{\mathsf{T}}\mathbf{v}_c)}{\sum_{i\in\nu} \exp(\mathbf{u}_i^{\mathsf{T}}\mathbf{v}_c)}
{% endmathjax %}
其中词表索引集{% mathjax %}\nu = \{0,1,\ldots,|\nu|-1\}{% endmathjax %}。给定长度为{% mathjax %}T{% endmathjax %}的文本序列，其中时间步{% mathjax %}t{% endmathjax %}处的词表示为{% mathjax %}w^{(t)}{% endmathjax %}。假设上下文词是在给定任何中心词的情况下独立生成的。对于上下文窗口{% mathjax %}m{% endmathjax %}，跳元模型的似然函数是在给定任何中心词的情况下生成所有上下文词的概率：
{% mathjax '{"conversion":{"em":14}}' %}
\prod_{t=1}^T\;\;\;\prod_{-m\leq j\leq m,j\neq0}\;\;\;P(w^{(t+j)}|w^{(t)})
{% endmathjax %}
其中可以省略小于1或大于{% mathjax %}T{% endmathjax %}的时间步。
###### 训练


##### 连续词袋模型（CBOW）

