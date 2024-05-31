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
P(w_o|w_c) = \;\prod_{j=1}^{L(w_o) - 1} \sigma([n(w_o,j+1) = \text{leftChild}(n(w_o,j))]\cdot \mathbf{u}_{n(w_o,j)}^{\mathsf{T}}\mathbf{v}_c)
{% endmathjax %}
其中函数{% mathjax %}\sigma{% endmathjax %}的定义，{% mathjax %}\text{leftChild}(n){% endmathjax %}是节点{% mathjax %}n{% endmathjax %}的左子节点：如果{% mathjax %}x{% endmathjax %}为真，{% mathjax %}[x] = 1{% endmathjax %};否则{% mathjax %}[x] = -1{% endmathjax %}。为了说明给定词{% mathjax %}w_c{% endmathjax %}生成词{% mathjax %}w_3{% endmathjax %}的条件概率。这需要{% mathjax %}w_c{% endmathjax %}的词向量{% mathjax %}\mathbf{v}_c{% endmathjax %}和从根到{% mathjax %}w_3{% endmathjax %}的路径上的非叶节点向量之间的点积，该路径依次向左、向右和向左遍历：
{% mathjax '{"conversion":{"em":14}}' %}
P(w_3|w_c) = \sigma (\mathbf{u}_{n(w_3,1)}^{\mathsf{T}}\mathbf{v}_c)\cdot \sigma (\mathbf{u}_{n(w_3,2)}^{\mathsf{T}}\mathbf{v}_c)\cdot \sigma (\mathbf{u}_{n(w_3,3)}^{\mathsf{T}}\mathbf{v}_c)
{% endmathjax %}
由{% mathjax %}\sigma(x) + \sigma(-x) = 1{% endmathjax %}，它认为基于任意词{% mathjax %}w_c{% endmathjax %}生成词表{% mathjax %}\nu{% endmathjax %}中所有词的条件概率总和为`1`：
{% mathjax '{"conversion":{"em":14}}' %}
\sum_{w\in \nu} P(w|w_c) = 1
{% endmathjax %}
幸运的是，由于二叉树结构，{% mathjax %}L(w_o) - 1{% endmathjax %}大约与{% mathjax %}\mathcal{O}(\log_2 |\nu|){% endmathjax %}是一个数量级。当词表大小{% mathjax %}\nu{% endmathjax %}很大时，与没有近似训练的相比，使用分层`softmax`的每个训练步的计算代价显著降低。
##### 总结

负采样通过考虑相互独立的事件来构造损失函数，这些事件同时涉及正例和负例。训练的计算量与每一步的噪声词数成线性关系。分层`softmax`使用二叉树中从根节点到叶节点的路径构造损失函数。训练的计算成本取决于词表大小的对数。

#### 全局向量的词嵌入（GloVe）

上下文窗口内的词共现可以携带丰富的语义信息。例如，在一个大型语料库中，“固体”比“气体”更有可能与“冰”共现，但“气体”一词与“蒸汽”的共现频率可能比与“冰”的共现频率更高。此外，可以预先计算此类共现的全局语料库统计数据：这可以提高训练效率。

##### 带全局语料统计的跳元模型

用{% mathjax %}q_{ij}{% endmathjax %}表示词{% mathjax %}w_j{% endmathjax %}的条件概率{% mathjax %}P(w_j|w_i){% endmathjax %}，在跳元模型给定词{% mathjax %} w_i{% endmathjax %}，我们有：
{% mathjax '{"conversion":{"em":14}}' %}
q_{ij} = \frac{\exp(\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i)}{\sum_{k\in \nu} \exp(\mathbf{u}_k^{\mathsf{T}}\mathbf{v}_i)}
{% endmathjax %}
其中，对于任意索引{% mathjax %}i{% endmathjax %}，向量{% mathjax %}\mathbf{v}_i{% endmathjax %}和{% mathjax %}\mathbf{u}_i{% endmathjax %}分别表示词{% mathjax %}w_i{% endmathjax %}作为中心词和上下文词，且{% mathjax %}\nu = \{0,1,\ldots,|\nu|-1\}{% endmathjax %}是词表的索引集。考虑词{% mathjax %}w_i{% endmathjax %}可能在语料库中出现多次。在整个语料库中，所有以{% mathjax %}w_i{% endmathjax %}为中心词的上下文词形成一个词索引的多重集{% mathjax %}\mathcal{C}_i{% endmathjax %}，该索引允许同一元素的多个实例。对于任何元素，其实例数称为其重数。举例说明，假设词{% mathjax %}w_i{% endmathjax %}在语料库中出现两次，并且在两个上下文窗口中以{% mathjax %}w_i{% endmathjax %}为其中心词的上下文词索引是{% mathjax %}k,j,m,k{% endmathjax %}和{% mathjax %}k,l,k,j{% endmathjax %}。因此，多重集{% mathjax %}\mathcal{C}_i = \{j,j,k,k,k,k,l,m\}{% endmathjax %}，其中元素{% mathjax %}j,k,l,m{% endmathjax %}的重数分别为`2、4、1、1`。现在，让我们将多重集{% mathjax %} \mathcal{C}_i{% endmathjax %}中的元素{% mathjax %}j{% endmathjax %}的重数表示为{% mathjax %}x_{ij}{% endmathjax %}。这是词{% mathjax %}w_j{% endmathjax %}（作为上下文词）和词{% mathjax %}w_i{% endmathjax %}（作为中心词）在整个语料库的同一上下文窗口中的全局共现计数。使用这样的全局语料库统计，跳元模型的损失函数等价于：
{% mathjax '{"conversion":{"em":14}}' %}
-\sum_{i\in \nu}\sum_{j\in \nu} x_{ij}\log q_{ij}
{% endmathjax %}
我们用{% mathjax %}x_i{% endmathjax %}表示上下文窗口中的所有上下文词的数量，其中{% mathjax %}w_i{% endmathjax %}作为它们的中心词出现，这相当于{% mathjax %}|\mathcal{C}_i|{% endmathjax %}。设{% mathjax %}p_{ij}{% endmathjax %}为用于生成上下文词{% mathjax %}w_j{% endmathjax %}的条件概率{% mathjax %}x_{ij}/x_i{% endmathjax %}。给定中心词{% mathjax %}w_i{% endmathjax %}，上面公式可以重写为：
{% mathjax '{"conversion":{"em":14}}' %}
-\sum_{i\in \nu} x_i \sum_{j\in \nu} p_{ij}\log q_{ij}
{% endmathjax %}
{% mathjax %}-\sum_{j\in \nu} p_{ij}\log q_{ij}{% endmathjax %}计算全局语料统计的条件分布{% mathjax %}p_{ij}{% endmathjax %}和模型预测的条件分布{% mathjax %}q_{ij}{% endmathjax %}的交叉熵。如上所述，这一损失也按{% mathjax %}x_i{% endmathjax %}加权。在上个公式中最小化损失函数将使预测的条件分布接近全局语料库统计中的条件分布。

虽然交叉熵损失函数通常用于测量概率分布之间的距离，但在这里可能不是一个好的选择。一方面，规范化{% mathjax %}q_{ij}{% endmathjax %}的代价在于整个词表的求和，这在计算上可能非常昂贵。另一方面，来自大型语料库的大量罕见事件往往被交叉熵损失建模，从而赋予过多的权重。
##### GloVe模型

有鉴于此，`GloVe`模型基于平方损失对跳元模型做了三个修改：
- 使用变量{% mathjax %}p_{ij}' = x_{ij}{% endmathjax %}和{% mathjax %}q_{ij}' = \exp(\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i){% endmathjax %}而非概率分布，并取两者的对数。所以平方损失项是{% mathjax %}(\log p_{ij}' - \log q_{ij}')^2 = (\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i - \log x_{ij})^2{% endmathjax %}。
- 为每个词{% mathjax %}w_i{% endmathjax %}添加两个标量模型参数：中心词偏置{% mathjax %}b_i{% endmathjax %}和上下文词偏置{% mathjax %}c_i{% endmathjax %}。
- 用权重函数{% mathjax %}h(x_{ij}){% endmathjax %}替换每个损失项的权重，其中{% mathjax %}h(x){% endmathjax %}在{% mathjax %}[0,1]{% endmathjax %}的间隔内递增。

整合代码，训练`GloVe`是为了尽量降低以下损失函数：
{% mathjax '{"conversion":{"em":14}}' %}
\sum_{i\in \nu} x_i \sum_{j\in \nu} h(x_{ij})(\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i + b_i + c_i - \log x_{ij})^2
{% endmathjax %}
对于权重函数，建议的选择是：当{% mathjax %}x < c{% endmathjax %}（例如，{% mathjax %}c = 100{% endmathjax %}）时，{% mathjax %}h(x) = (x/c)^{\alpha}{% endmathjax %}（例如{% mathjax %}\alpha = 0.75{% endmathjax %}）；否则{% mathjax %}h(x) = 1{% endmathjax %}。在这种情况下，由于{% mathjax %}h(0) = 0{% endmathjax %}，为了提高计算效率，可以省略任意{% mathjax %}x_{ij} = 0{% endmathjax %}的平方损失项。例如，当使用小批量随机梯度下降进行训练时，在每次迭代中，我们随机抽样一小批量非零的{% mathjax %}x_{ij}{% endmathjax %}来计算梯度并更新模型参数。注意，这些非零的{% mathjax %}x_{ij}{% endmathjax %}是预先计算的全局语料库统计数据；因此，该模型`GloVe`被称为**全局向量**。应该强调的是，当词{% mathjax %}w_i{% endmathjax %}出现在词{% mathjax %}w_j{% endmathjax %}的上下文窗口时，词{% mathjax %}w_j{% endmathjax %}也出现在词{% mathjax %}w_i{% endmathjax %}的上下文窗口。因此，{% mathjax %}x_{ij} = x_{ji}{% endmathjax %}。与拟合非对称条件概率{% mathjax %}p_{ij}{% endmathjax %}的`word2vec`不同，`GloVe`拟合对称概率{% mathjax %}\log x_{ij}{% endmathjax %}。因此，在`GloVe`模型中，任意词的中心词向量和上下文词向量在数学上是等价的。但在实际应用中，由于初始值不同，同一个词经过训练后，在这两个向量中可能得到不同的值：`GloVe`将它们相加作为输出向量。
##### 从条件概率比值理解GloVe模型

我们也可以从另一个角度来理解`GloVe`模型。使用下表中的相同符号，设{% mathjax %}p_{ij}\;\underset{=}{\text{def}}\;P(w_j|w_i){% endmathjax %}为生成上下文词{% mathjax %} w_j{% endmathjax %}的条件概率，给定{% mathjax %}w_i{% endmathjax %}作为语料库中的中心词。`tab_glove`根据大量语料库的统计数据，列出了给定单词`“ice”`和`“steam”`的共现概率及其比值。
<center> 表：label:tab_glove</center>
|{% mathjax %}w_k{% endmathjax %}|solid|gas|water|fashion|
|:-------------------------------|:----|:-----|:-----|:-------|
|{% mathjax %}p_1 = P(w_k|\text{ice}){% endmathjax %}|0.00019|0.000066|0.003|0.000017|
|{% mathjax %}p_2 = P(w_k|\text{steam}){% endmathjax %}|0.000022|0.00078|0.0022|0.000018|
|{% mathjax %}p_1/p_2{% endmathjax %}|8.9|0.085|1.36|0.96|

从`tab_glove中`，我们可以观察到以下几点：
- 对于与`“ice”`相关但与`“steam”`无关的单词{% mathjax %}w_k{% endmathjax %}，例如{% mathjax %}w_k = \text{solid}{% endmathjax %}，我们预计会有更大的共现概率比值，例如`8.9`。
- 对于与`“steam”`相关但与`“ice”`无关的单词{% mathjax %}w_k{% endmathjax %}，例如{% mathjax %}w_k = \text{gas}{% endmathjax %}，我们预计较小的共现概率比值，例如`0.085`。
- 对于同时与`“ice”`和`“steam”`相关的单词{% mathjax %}w_k{% endmathjax %}，例如{% mathjax %}w_k = \text{water}{% endmathjax %}，我们预计其共现概率的比值接近`1`，例如`1.36`.
- 对于与`“ice”`和`“steam”`都不相关的单词{% mathjax %}w_k{% endmathjax %}，例如{% mathjax %}w_k = \text{fashion}{% endmathjax %}，我们预计共现概率的比值接近`1`，例如`0.96`.

由此可见，共现概率的比值能够直观地表达词与词之间的关系。因此，我们可以设计三个词向量的函数来拟合这个比值。对于共现概率{% mathjax %}p_{ij}/p_{ik}{% endmathjax %}的比值，其中{% mathjax %}w_i{% endmathjax %}是中心词，{% mathjax %}w_j{% endmathjax %}和{% mathjax %}w_k{% endmathjax %}是上下文词，我们希望使用某个函数{% mathjax %}f{% endmathjax %}来拟合该比值：
{% mathjax '{"conversion":{"em":14}}' %}
f(\mathbf{u}_j,\mathbf{u}_k,\mathbf{v}_i) \approx \frac{p_{ij}}{p_{ik}}
{% endmathjax %}
在{% mathjax %}f{% endmathjax %}的许多可能的设计中，我们只在以下几点中选择了一个合理的选择。因为共现概率的比值是标量，所以我们要求{% mathjax %}f{% endmathjax %}是标量函数，例如{% mathjax %}f(\mathbf{u}_j,\mathbf{u}_k,\mathbf{v}_i) = f((\mathbf{u}_j - \mathbf{u}_k)^{\mathsf{T}}\mathbf{v}_i){% endmathjax %}。在上面公式中交换词索引{% mathjax %}j{% endmathjax %}和{% mathjax %}k{% endmathjax %}，它必须保持{% mathjax %}f(x)f(-x) = 1{% endmathjax %}，所以一种可能性是{% mathjax %}f(x) = \exp(x){% endmathjax %}，即：
{% mathjax '{"conversion":{"em":14}}' %}
f(\mathbf{u}_j,\mathbf{u}_k,\mathbf{v}_i) = \frac{\exp(\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i)}{\exp(\mathbf{u}_k^{\mathsf{T}}\mathbf{v}_i)} \approx \frac{p_{ij}}{p_{ik}}
{% endmathjax %}
现在让我们选择{% mathjax %}\exp(\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i) \approx \alpha p_{ij}{% endmathjax %}，其中{% mathjax %}\alpha{% endmathjax %}是常数。从{% mathjax %}p_{ij} = x_{ij}/x_i{% endmathjax %}开始，取两边的对数得到{% mathjax %}\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i \approx \log \alpha + \log  x_{ij} - \log x_i{% endmathjax %}。我们可以使用附加的偏置项来拟合{% mathjax %}-\log \alpha + \log x_i{% endmathjax %}，如中心词偏置{% mathjax %}b_i{% endmathjax %}和上下文词偏置{% mathjax %}c_j{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{u}_j^{\mathsf{T}}\mathbf{v}_i + b_i + c_j \approx \log x_{ij}
{% endmathjax %}
通过对上面公式的加权平方误差的度量，得到了的`GloVe`损失函数。
##### 总结

诸如词-词共现计数的全局语料库统计可以来解释跳元模型。交叉熵损失可能不是衡量两种概率分布差异的好选择，特别是对于大型语料库。`GloVe`使用平方损失来拟合预先计算的全局语料库统计数据。对于`GloVe`中的任意词，中心词向量和上下文词向量在数学上是等价的。`GloVe`可以从词-词共现概率的比率来解释。

#### 子词嵌入

在英语中，`“helps”“helped”和“helping”`等单词都是同一个词`“help”`的变形形式。`“dog”`和`“dogs”`之间的关系与`“cat”`和`“cats”`之间的关系相同，`“boy”`和`“boyfriend”`之间的关系与`“girl”`和`“girlfriend”`之间的关系相同。在法语和西班牙语等其他语言中，许多动词有`40`多种变形形式，而在芬兰语中，名词最多可能有`15`种变形。在语言学中，形态学研究单词形成和词汇关系。但是，`word2vec`和`GloVe`都没有对词的内部结构进行探讨。
##### fastText模型

回想一下词在`word2vec`中是如何表示的。在跳元模型和连续词袋模型中，同一词的不同变形形式直接由不同的向量表示，不需要共享参数。为了使用形态信息，`fastTex`t模型提出了一种子词嵌入方法，其中子词是一个字符`n-gram`。`fastText`可以被认为是子词级跳元模型，而非学习词级向量表示，其中每个中心词由其子词级向量之和表示。让我们来说明如何以单词`“where”`为例获得`fastText`中每个中心词的子词。首先，在词的开头和末尾添加特殊字符`“<”`和`“>”`，以将前缀和后缀与其他子词区分开来。然后，从词中提取字符`n-gram`。 例如，值{% mathjax %}n=3{% endmathjax %}时，我们将获得长度为3的所有子词：`“<wh”“whe”“her”“ere”“re>”`和特殊子词`“<where>”`。

在`fastText`中，对于任意词{% mathjax %}w{% endmathjax %}，用{% mathjax %}\mathcal{g}_w{% endmathjax %}表示其长度在`3`和`6`之间的所有子词与其特殊子词的并集。词表是所有词的子词的集合。假设{% mathjax %}\mathbf{z}_g{% endmathjax %}是词典中的子词{% mathjax %}g{% endmathjax %}的向量，则跳元模型中作为中心词的词{% mathjax %}w{% endmathjax %}的向量{% mathjax %}\mathbf{v}_w{% endmathjax %}是其子词向量的和：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{v}_w = \sum_{g\in \mathcal{g}_w} \mathbf{z}_g
{% endmathjax %}

`fastText`的其余部分与跳元模型相同。与跳元模型相比，`fastText`的词量更大，模型参数也更多。此外，为了计算一个词的表示，它的所有子词向量都必须求和，这导致了更高的计算复杂度。然而，由于具有相似结构的词之间共享来自子词的参数，罕见词甚至词表外的词在`fastText`中可能获得更好的向量表示。
##### 字节对编码（Byte Pair Encoding）

在`fastText`中，所有提取的子词都必须是指定的长度，例如`3`到`6`，因此词表大小不能预定义。为了在固定大小的词表中允许可变长度的子词，我们可以应用一种称为**字节对编码**(`Byte Pair Encoding，BPE`)的压缩算法来提取子词。字节对编码执行训练数据集的统计分析，以发现单词内的公共符号，诸如任意长度的连续字符。从长度为1的符号开始，字节对编码迭代地合并最频繁的连续符号对以产生新的更长的符号。请注意，为提高效率，不考虑跨越单词边界的对。最后，我们可以使用像子词这样的符号来切分单词。字节对编码及其变体已经用于诸如`GPT-2`和`RoBERTa`等自然语言处理预训练模型中的输入表示。

##### 总结

`fastText`模型提出了一种**子词嵌入**方法：基于`word2vec`中的跳元模型，它将中心词表示为其子词向量之和。字节对编码执行训练数据集的统计分析，以发现词内的公共符号。作为一种贪心方法，字节对编码迭代合并最频繁的连续符号对。子词嵌入可以提高稀有词和词典外词的表示质量。

#### Transformers的双向编码器表示（BERT）

##### 从上下文无关到上下文敏感

例如，`word2vec`和`GloVe`都将相同的预训练向量分配给同一个词，而不考虑词的上下文（如果有的话）。形式上，任何词元{% mathjax %}x{% endmathjax %}的上下文无关表示是函数{% mathjax %}f(x){% endmathjax %}，其仅将{% mathjax %}x{% endmathjax %}作为其输入。考虑到自然语言中丰富的多义现象和复杂的语义，上下文无关表示具有明显的局限性。例如，在`“a crane is flying”`（一只鹤在飞）和`“a crane driver came”`（一名吊车司机来了）的上下文中，`“crane”`一词有完全不同的含义；因此，同一个词可以根据上下文被赋予不同的表示。