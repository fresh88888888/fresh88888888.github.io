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

跳元模型参数是词表中每个词的中心词向量和上下文词向量。在训练中，我们通过最大化似然函数（即极大似然估计）来学习模型参数。这相当于最小化以下损失函数：
{% mathjax '{"conversion":{"em":14}}' %}
-\sum_{t=1}^T\;\;\;\sum_{-m\leq j\leq m, j\neq 0}\;\;\;\log P(w^{(t+j)}|w^{(t)})
{% endmathjax %}
当使用随机梯度下降来最小化损失时，在每次迭代中可以随机抽样一个较短的子序列来计算该子序列的（随机）梯度，以更新模型参数。为了计算该（随机）梯度，我们需要获得对数条件概率关于中心词向量和上下文词向量的梯度。涉及中心词{% mathjax %}w_c{% endmathjax %}和上下文词{% mathjax %}w_o{% endmathjax %}的对数条件概率为：
{% mathjax '{"conversion":{"em":14}}' %}
\log P(w_o|w_c) = \mathbf{u}_o^{\mathsf{T}}\mathbf{v}_c - \log \big( \sum_{i\in \nu} \exp(\mathbf{u}_i^{\mathsf{t}}\mathbf{v}_c) \big)
{% endmathjax %}
通过微分，我们可以获得其相对于中心词向量{% mathjax %}\mathbf{v}_c{% endmathjax %}的梯度为
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\partial \log P(w_o|w_c) & = \mathbf{u}_o - \frac{\sum_{j\in \nu} \exp(\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_c)\mathbf{u}_j}{\sum_{j\in \nu} \exp(\mathbf{u}_i^{\mathsf{T}}\mathbf{v}_c)} \\
& = \mathbf{u}_o - \sum_{j\in \nu}(\frac{\exp(\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_c)}{\sum_{i\in \nu} \exp(\mathbf{u}_i^{\mathsf{T}}\mathbf{v}_c)})\mathbf{u}_j \\
& = \mathbf{u}_o - \sum_{j\in \nu} P(w_j|w_c)\mathbf{u}_j
\end{align}
{% endmathjax %}
注意，以上公式中的计算需要词典中以{% mathjax %}w_c{% endmathjax %}为中心词的所有词的条件概率。其他词向量的梯度可以以相同的方式获得。对词典中索引为{% mathjax %}i{% endmathjax %}的词进行训练后，得到{% mathjax %}\mathbf{v}_i{% endmathjax %}（作为中心词）和{% mathjax %}\mathbf{u}_i{% endmathjax %}（作为上下文词）两个词向量。在自然语言处理应用中，跳元模型的中心词向量通常用作词表示。
##### 连续词袋模型（CBOW）

**连续词袋**(`CBOW`)模型类似于跳元模型。与跳元模型的主要区别在于，连续词袋模型假设中心词是基于其在文本序列中的周围上下文词生成的。例如，在文本序列`“the”“man”“loves”“his”“son”`中，在`“loves”`为中心词且上下文窗口为`2`的情况下，连续词袋模型考虑基于上下文词`“the”“man”“him”“son”`（如下图所示）生成中心词`“loves”`的条件概率，即：
{% mathjax '{"conversion":{"em":14}}' %}
P(\text{"the","man","his","son"}|\text{"loves"})
{% endmathjax %}
{% asset_img nlp_3.png "连续词袋模型考虑了给定周围上下文词生成中心词条件概率" %}

由于连续词袋模型中存在多个上下文词，因此在计算条件概率时对这些上下文词向量进行平均。具体地说，对于字典中索引{% mathjax %}i{% endmathjax %}的任意词，分别用{% mathjax %}\mathbf{v}_i\in \mathbb{R}^d{% endmathjax %}和{% mathjax %}\mathbf{u}_i\in \mathbb{r}^d{% endmathjax %}表示用作上下文词和中心词的两个向量（符号与跳元模型中相反）。给定上下文词{% mathjax %}w_{o_1},\ldots,w_{o_{2m}}{% endmathjax %}（在词表中索引是{% mathjax %}o_1,\ldots,o_{2m}{% endmathjax %}）生成任意中心词{% mathjax %}w_c{% endmathjax %}（在词表中索引是{% mathjax %}c{% endmathjax %}）的条件概率可以由以下公式建模:
{% mathjax '{"conversion":{"em":14}}' %}
P(w_c|w_{o_1,\ldots,w_{o_{2m}}}) = \frac{\exp(\frac{1}{2m}\mathbf{u}_c^{\mathsf{T}}(\mathbf{v}_{o_1}+\ldots,+\mathbf{v}_{o_{2m}}))}{\sum_{i\in \nu}}
{% endmathjax %}
为了简洁起见，我们设为{% mathjax %}\mathcal{W}_o = \{w_{o_1},\ldots,w_{o_{2m}}\}{% endmathjax %}和{% mathjax %}\bar{\mathbf{v}}_o = (\mathbf{v}_{o_1}+ \ldots,+\mathbf{v}_{o_{2m}})/(2m){% endmathjax %}，那么可以简化为：
{% mathjax '{"conversion":{"em":14}}' %}
P(w_c|\mathcal{W}_o) = \frac{\exp(\mathbf{u}_c^{\mathsf{T}}\bar{\mathbf{v}}_o)}{\sum_{i\in \nu}\exp(\mathbf{u}_i^{\mathsf{T}}\bar{\mathbf{v}}_o)}
{% endmathjax %}
给定长度为{% mathjax %}T{% endmathjax %}的文本序列，其中时间步{% mathjax %}t{% endmathjax %}处的词表示为{% mathjax %}w_{(t)}{% endmathjax %}。对于上下文窗口{% mathjax %}m{% endmathjax %}，连续词袋模型的似然函数是在给定其上下文词的情况下生成所有中心词的概率：
{% mathjax '{"conversion":{"em":14}}' %}
\prod_{t=1}^T P(w^{(t)}|w^{(t-m)},\ldots,w^{(t-1)},w^{(t+1)},\ldots,w^{(t+m)})
{% endmathjax %}
###### 训练

训练连续词袋模型与训练跳元模型几乎是一样的。连续词袋模型的最大似然估计等价于最小化以下损失函数：
{% mathjax '{"conversion":{"em":14}}' %}
-\sum_{t=1}^T \log P(w^{(t)}|w^{(t-m)},\ldots,w^{(t-1)},w^{(t+1)},\ldots,w^{(t+m)})
{% endmathjax %}
请注意，
{% mathjax '{"conversion":{"em":14}}' %}
\log P(w_c|\mathcal{W}_o) = \mathbf{u}_c^T\bar{\mathbf{v}}_o- \log(\sum_{i\in \nu} \exp(\mathbf{u}_i^T\bar{\mathbf{v}}_o))
{% endmathjax %}
通过微分，我们可以获得其关于任意上下文词向量{% mathjax %}\mathbf{v}_{o_i}{% endmathjax %}（{% mathjax %}(i=1,\ldots,2m){% endmathjax %}）的梯度，如下：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial \log P(w_c|\mathcal{W}_o)}{\partial \mathbf{v}_{o_i}} = \frac{1}{2m}(\mathbf{u}_c - \sum_{j\in \nu} \frac{\exp(\mathbf{u}_c^{\mathsf{T}}\bar{\mathbf{v}}_o)\mathbf{u}_j}{\sum_{i\in \nu}\exp(\mathbf{u}_i^{\mathsf{T}}\bar{\mathbf{v}}_o)}) = \frac{1}{2m}(\mathbf{u}_c - \sum_{j\in \nu} P(w_j|\mathcal{w}_o)\mathbf{u}_j)
{% endmathjax %}
其他词向量的梯度可以以相同的方式获得。与跳元模型不同，连续词袋模型通常使用上下文词向量作为词表示。
##### 总结

**词向量是用于表示单词意义的向量，也可以看作词的特征向量**。将词映射到实向量的技术称为**词嵌入**。`word2vec`包含**跳元模型和连续词袋模型**。跳元模型假设一个单词可用于在文本序列中，生成其周围的单词；而连续词袋模型假设基于上下文词来生成中心单词。

#### 近似训练

跳元模型的主要思想是使用`softmax`运算来计算基于给定的中心词{% mathjax %}w_c{% endmathjax %}生成上下文字{% mathjax %}w_o{% endmathjax %}的条件概率。由于`softmax`操作的性质，上下文词可以是词表{% mathjax %}\nu{% endmathjax %}中的任意项，与整个词表大小一样多的项的求和。因此，跳元模型的梯度计算和连续词袋模型的梯度计算都包含求和。不幸的是，在一个词典上（通常有几十万或数百万个单词）求和的梯度的计算成本是巨大的！为了降低上述计算复杂度，将介绍两种近似训练方法：负采样和分层`softmax`。由于跳元模型和连续词袋模型的相似性，我们将以跳元模型为例来描述这两种近似训练方法。
##### 负采样

负采样修改了原目标函数。给定中心词{% mathjax %}w_c{% endmathjax %}的上下文窗口，任意上下文词{% mathjax %}w_o{% endmathjax %}来自该上下文窗口的被认为是由下式建模概率的事件：
{% mathjax '{"conversion":{"em":14}}' %}
P(D= 1|w_c,w_o) = \sigma(\mathbf{u}_o^T\mathbf{v_c})
{% endmathjax %}
其中{% mathjax %}\sigma{% endmathjax %}使用了`sigmoid`激活函数的定义：
{% mathjax '{"conversion":{"em":14}}' %}
\sigma(x) = \frac{1}{1 + \exp(-x)}
{% endmathjax %}
让我们从最大化文本序列中所有这些事件的联合概率开始训练词嵌入。具体而言，给定长度为{% mathjax %}T{% endmathjax %}的文本序列，以{% mathjax %}w^{(t)}{% endmathjax %}表示时间步{% mathjax %}t{% endmathjax %}的词，并使上下文窗口为{% mathjax %}m{% endmathjax %}，考虑最大化联合概率：
{% mathjax '{"conversion":{"em":14}}' %}
\prod_{t=1}^T\;\;\;\prod_{-m\leq j\leq m,j\neq 0}\;\;\;P(D=1|w^{(t)},w^{(t+j)})
{% endmathjax %}
然而，以上公式只考虑那些正样本的事件。仅当所有词向量都等于无穷大时，以上公式中的联合概率最大化为`1`。当然，这样的结果毫无意义。为了使目标函数更有意义，负采样添加从预定义分布中采样的负样本。用{% mathjax %}S{% endmathjax %}表示上下文词{% mathjax %}w_o{% endmathjax %}来自中心词{% mathjax %}w_o{% endmathjax %}的上下文窗口的事件。对于这个涉及{% mathjax %}w_o{% endmathjax %}的事件，从预定义分布{% mathjax %}P(w){% endmathjax %}中采样{% mathjax %}K{% endmathjax %}个不是来自这个上下文窗口噪声词。用{% mathjax %}N_k{% endmathjax %}表示噪声词{% mathjax %} {% endmathjax %}w_k（{% mathjax %}(k=1,\ldots,K){% endmathjax %}）不是来自{% mathjax %}w_c{% endmathjax %}的上下文窗口的事件。假设正例和负例{% mathjax %}S,N_1,\ldots,N_K{% endmathjax %}的这些事件是相互独立的。负采样将上面公式中的联合概率（仅涉及正例）重写为：
{% mathjax '{"conversion":{"em":14}}' %}
\prod_{t=1}^T\;\;\;\prod_{-m\leq j\leq m,j\neq 0}\;\;\;P(w^{(t+j)}|w^{(t)})
{% endmathjax %}
通过事件{% mathjax %}S,N_1,\ldots,N_K{% endmathjax %}近似条件概率：
{% mathjax '{"conversion":{"em":14}}' %}
P(w^{(t+j)}|w^{(t)}) = P(D=1|w^{(t)},w^{(t+j)})\;\;\;\prod_{k=1,w_k\in P(w)}^K P(D=0|w^{(t)},w_k)
{% endmathjax %}
分别用{% mathjax %}i_t{% endmathjax %}和{% mathjax %}h_k{% endmathjax %}表示词{% mathjax %}w^{(t)}{% endmathjax %}和噪声词{% mathjax %}w_k{% endmathjax %}在文本序列的时间步{% mathjax %}t{% endmathjax %}处的索引。关于条件概率的对数损失为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
-\log P(w^{(t+j)}|w^{(t)}) & = -\log P(D=1|w^{(t)},w^{(t+j)}) - \sum_{k=1,w_k\sim P(w)}^K \log P(D=0|w^{(t)},w_k) \\
& = -\log \sigma(\mathbf{u}_{i_{t+j}}^{\mathsf{T}}\mathbf{v}_{i_t}) - \sum_{k=1,w_k\sim P(w)}^K \log(1 - \sigma(\mathbf{u}_{h_k}^{\mathsf{t}}\mathbf{v}_{i_t})) \\
& = -\log \sigma(\mathbf{u}_{i_{t+j}}^{\mathsf{T}}\mathbf{v}_{i_t}) - \sum_{k=1,w_k\sim P(w)}^K \log \sigma(\mathbf{u}_{h_k}^{\mathsf{t}}\mathbf{v}_{i_t})
\end{align}
{% endmathjax %}
我们可以看到，现在每个训练步的梯度计算成本与词表大小无关，而是线性依赖于{% mathjax %}K{% endmathjax %}。当将超参数{% mathjax %}K{% endmathjax %}设置为较小的值时，在负采样的每个训练步处的梯度的计算成本较小。
##### 层序Softmax

作为另一种近似训练方法，层序`Softmax`(`hierarchical softmax`)使用二叉树（下图中说明的数据结构），其中树的每个叶节点表示词表{% mathjax %}\nu{% endmathjax %}中的一个词。
{% asset_img nlp_4.png "用于近似训练的分层softmax，其中树的每个叶节点表示词表中的一个词" %}

用{% mathjax %}L(w){% endmathjax %}表示二叉树中表示字{% mathjax %}w{% endmathjax %}的从根节点到叶节点的路径上的节点数（包括两端）。设{% mathjax %}n(w,j){% endmathjax %}为该路径上的{% mathjax %}j^{\text{th}}{% endmathjax %}节点，其上下文字向量为{% mathjax %}\mathbf{u}_{n(w,j)}{% endmathjax %}。例如，上图中的{% mathjax %}L(w_3) = 4{% endmathjax %}。分层`softmax`的条件概率近似为：
{% mathjax '{"conversion":{"em":14}}' %}
P(w_o|w_c) = \;\prod_{j=1}^{L(w_o) - 1} \sigma([n(w_o,j+1) = \text{leftChild(n(w_o,j))}]\cdot \mathbf{u}_{n(w_o,j)}^{\mathsf{T}}\mathbf{v}_c)
{% endmathjax %}
其中函数{% mathjax %}\sigma{% endmathjax %}的定义，{% mathjax %}\text{leftChild}(n){% endmathjax %}是节点{% mathjax %}n{% endmathjax %}的左子节点：如果{% mathjax %}x{% endmathjax %}为真，{% mathjax %}[x] = 1{% endmathjax %};否则{% mathjax %}[x] = -1{% endmathjax %}。为了说明给定词{% mathjax %}w_c{% endmathjax %}生成词{% mathjax %}w_3{% endmathjax %}的条件概率。这需要{% mathjax %}w_c{% endmathjax %}的词向量{% mathjax %}\mathbf{v}_c{% endmathjax %}和从根到{% mathjax %}w_3{% endmathjax %}的路径上的非叶节点向量之间的点积，该路径依次向左、向右和向左遍历：
{% mathjax '{"conversion":{"em":14}}' %}
P(w_3|w_c) = \sigma (\mathbf{u}_{n(w_3,1)}^{mathsf{T}}\mathbf{v}_c)\cdot \sigma (\mathbf{u}_{n(w_3,2)}^{mathsf{T}}\mathbf{v}_c)\cdot \sigma (\mathbf{u}_{n(w_3,3)}^{mathsf{T}}\mathbf{v}_c)
{% endmathjax %}
由{% mathjax %}\sigma(x) + \sigma(-x) = 1{% endmathjax %}，它认为基于任意词{% mathjax %}w_c{% endmathjax %}生成词表{% mathjax %}\nu{% endmathjax %}中所有词的条件概率总和为`1`：
{% mathjax '{"conversion":{"em":14}}' %}
\sum_{w\in \nu} P(w|w_c) = 1
{% endmathjax %}
幸运的是，由于二叉树结构，{% mathjax %}L(w_o) - 1{% endmathjax %}大约与{% mathjax %}\mathcal{O}(\log_2 |\nu|){% endmathjax %}是一个数量级。当词表大小{% mathjax %}\nu{% endmathjax %}很大时，与没有近似训练的相比，使用分层`softmax`的每个训练步的计算代价显著降低。
##### 总结

负采样通过考虑相互独立的事件来构造损失函数，这些事件同时涉及正例和负例。训练的计算量与每一步的噪声词数成线性关系。分层`softmax`使用二叉树中从根节点到叶节点的路径构造损失函数。训练的计算成本取决于词表大小的对数。
