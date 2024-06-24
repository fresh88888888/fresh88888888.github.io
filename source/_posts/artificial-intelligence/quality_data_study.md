---
title: 高质量人类数据—思考（深度学习）
date: 2024-06-21 18:50:11
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

高质量数据是深度学习模型训练的燃料。大多数特定任务的标记数据来自人工标注，例如分类任务或用于`LLM`对齐训练的`RLHF labeling`(基于人类反馈的强化学习标注)（可以构建为分类格式）。文章中的许多`ML`技术可以帮助提高数据质量，但从根本上讲，人工数据收集需要关注细节和谨慎行事。
<!-- more -->

{% asset_img qd_1.png "实现高质量数据的两个方向" %}

#### 人工评估员<->数据质量

收集人类数据涉及一系列操作步骤，每个步骤都会影响数据质量：
- 任务设计：设计任务工作流程以提高清晰度并降低复杂性。详细的指南很有帮助，但非常冗长和复杂的指南需要大量培训才能使用。
- 选择并培训一批评分员：选择具有匹配技能和一致性的注释者。培训课程是必要的。入职后，还需要定期反馈和校准。
- 收集和汇总数据。在这个阶段，可以应用更多 ML 技术来清理、过滤和智能聚合数据以识别真实标签。

{% asset_img qd_2.png "质量保证是指一系列行动，通过对质量模型中确定的质量属性采取行动，从而提高质量" %}

##### 群体的智慧

Vox populi（原意为`“Vox populi, vox Dei”`）是一个拉丁语短语，意为人民的声音。`1907`年，《自然》杂志发表了一篇同名短文。该文追踪了一年一度的展览会上的一项活动，即选出一头肥牛，人们猜测这头牛的重量，如果猜测接近真实数字，即可赢得奖品。最中间的估计值被视为`“vox populi”`，结果非常接近真实值。作者总结道：“我认为，这个结果比人们预期的更能证明民主判断的可信度。”这可能是最早提到众包（“群体的智慧”）如何发挥作用的文献。

近`100`年后，[`Callison-Burch(2009)`](https://aclanthology.org/D09-1030/)进行了一项早期研究，使用`Amazon Mechanical Turk(AMT)`对机器翻译(`MT`)任务进行非专家人工评估，甚至依靠非专家来创建新的黄金参考翻译。人工评估的设置很简单：向每个`turker`展示一个源句子、一个参考翻译和来自`5`个`MT`系统的`5`个翻译。他们被要求从最好到最差对`5`个翻译进行排名。每个任务由`5`个`turker`完成。毫无疑问，有些垃圾评论者会为了优化数量而制作低质量的标注。因此，在衡量专家和非专家之间的一致性时，需要应用不同的加权方案来降低垃圾评论者的贡献：(1)“专家加权”：使用与专家在`10`个示例的黄金数据集上的一致性率；(2)“非专家加权”：依靠与整个数据集上其余`turkers`的一致性率。在一项更艰巨的任务中，非专家级的人工标注者被要求创建新的黄金参考翻译。`Callison-Burch`将这项任务设计为两个阶段，第一阶段参考机器翻译输出创建新的翻译，第二阶段过滤看似由机器翻译系统生成的翻译。专家翻译和众包翻译之间的相关性高于专家翻译和机器翻译系统输出之间的相关性。
{% asset_img qd_3.png "（左）通过比较每对翻译句子（“A > B”、“A=B”、“A < B”）来衡量一致率，因此偶然一致性为1/3。上限由专家之间的一致率设定。（右）不同来源的翻译之间的BLEU分数比较。LCD（语言数据联盟）翻译人员提供专家翻译" %}

##### 评分一致性

我们通常认为标注针对的是单一的基本事实，并尝试根据具有一致标准的黄金答案来评估质量。寻找可靠的基本事实标签的常见做法是从多个评分者那里收集多个标签。假设每个评分者的质量水平不同，我们可以使用标注的加权平均值，但要用熟练度分数加权。这个分数通常由一个评分者同意其他评分者的频率来近似。
- 多数投票：多数投票是最简单的聚合方式。
- 原始一致性：原始一致性计算其他人同意的百分比。这与多数投票间接相关，因为所有成员都有望获得更高的标注者间一致性率。
- `Cohen`的`Kappa`值([`Landis & Koch，1977`](https://www.jstor.org/stable/2529310))：`Cohen`的`Kappa`值以以下形式衡量评分者之间的一致性{% mathjax %}\kappa = (p_o - p_e) / (1 - p_c){% endmathjax %}，在这里{% mathjax %}p_o{% endmathjax %}是原始一致率，{% mathjax %}p_e{% endmathjax %}是偶然的一致性。`Cohen`的`kappa`有一个用于偶然一致的校正项，但如果一个标签更普遍，则此校正可能会被高估。
- 概率图建模：有一系列工作依赖于概率图建模来对标注决策中的不同因素进行建模，例如任务难度、任务潜在主题、评估者偏见、评估者信心，然后据此预测真实标签。比较了众包中`17`种真值推断算法，其中大多数是概率图模型。
##### 评分者分歧和两种范式

上述聚合过程依赖于一个假设，即存在一个潜在的标准答案，因此我们可以据此评估注释者的表现。然而，在许多主题中，特别是在安全、社会或文化领域，人们可能会意见不一，而且这种分歧往往是合理的，然后就归结为我们在多大程度上想要应用严格的规则而不是拥抱多样性。[Aroyo & Welty,2015](https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2564)讨论了人工注释收集实践中的一系列“误区”，发现它们都有些不准确，主要发现包括：
- 一些样本通常有不止一种正确的解释。我们需要多种视角，例如让多个人来审查注释质量。
- 分歧并不总是坏事。我们应该减少由错误或设计不良的流程引起的分歧，但其他分歧也可以为我们提供丰富的信息。如果是因为任务定义不明确，就应该加强指导，但更详细的指导并不能解决固有的意见分歧。
- 专家不一定总是比外行人优秀，但是在考虑什么是重要的方面，他们会有很大的差距。
- 基本事实标注会随着时间而改变，尤其是与及时事件或新闻相关的标注。

后来，[`Rottger`等人,`2021`](https://arxiv.org/abs/2112.07475)将这种差异概括为`NLP`任务数据标注的两个对比范式。
**描述性**：
- 定义：鼓励标注者的主观性，尝试模拟多种信念。
- 优点：有助于识别哪些条目更加主观；拥抱多样性。
- 缺点：评估者分歧等指标不能用于衡量数据质量或注释者表现；不能用于针对输出一种预设行为进行优化的训练模型。

**规定性**：
- 定义：不鼓励标注者的主观性，尝试始终如一地坚持一种信念。
- 优点：更加符合标准`NLP`设置。通过测量分歧或进行标签聚合更容易进行`QC`。
- 缺点：在实践中，创建高质量的标注指南成本高昂且具有挑战性，而且永远不可能完美；培训标注者熟悉指南以便正确应用它也具有挑战性；无法捕捉可解释的多样性或始终如一地编码一种特定的概念。

**描述范式**使我们能够理解许多重要的影响，并解释不同的观点。例如，标注者身份（例如非裔美国人、`LGBTQ`）被发现是他们将与身份相关的内容标记为有毒的统计学上显着的因素([`Goyal`等人，`2022`](https://arxiv.org/abs/2205.00501))。**主题**可能是导致意见分歧的另一个主要驱动因素。([`Wang,2023`](https://research.google/pubs/all-that-agrees-is-not-gold-evaluating-ground-truth-labels-and-dialogue-content-for-safety/))研究了人工智能对话系统安全性的人工评估过程，并比较了信任与安全(`T&S`) 专业人员和众包标注者的标签结果。他们有意收集与人群标注者相关的丰富元数据，例如人口统计或行为信息。通过比较`T&S`专家标签和人群标注，他们发现一致率因语义主题和严重程度而异：
- 不同主题的同意率差别很大；从暴力/血腥主题的`0.96`到个人主题的`0.25`不等。
- 鉴于标记“良性”，“有争议”，“中等”到“极端”的四个标签选项，“极端”和“良性”对话的同意率更高。

{% asset_img qd_4.png "非专家和专家标注之间的相关性在不同主题之间差异很大" %}

[`Zhang`等人,`2023`](https://arxiv.org/abs/2311.04345)提出了一种评估者分歧分类法来分析其根本原因。在列出的原因中，应避免因随机误差或个人层面的不一致而导致的分歧。当评估者在多次被问到同一个任务时给出不同的标签时，其中一些很可能是由人为错误造成的。基于这种直觉，分歧反卷积方法([`Gordon`等人,`2021`](https://dl.acm.org/doi/abs/10.1145/3411764.3445423))通过将每个人的意见锚定到他们自己的主要标签上，将稳定的意见与错误区分开来，从而鼓励评估者内部的一致性。
{% asset_img qd_5.png "评估者意见分歧原因分类" %}

分歧反卷积依赖于概率图建模：
- 估计注释者返回非主要标签的频率，{% mathjax %}p_{\text{flip}}{% endmathjax %}。
- 每个样本获取调整后的标签分布{% mathjax %}p^*{% endmathjax %}主要标签基于{% mathjax %}p_{\text{flip}}{% endmathjax %}。
- 样品来自{% mathjax %}p^*{% endmathjax %}作为新的测试集。
- 根据新的测试集测量性能指标。

鉴于{% mathjax %}C{% endmathjax %}-类别分类，生成模型的采样过程表述如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
y^*\mid x &\sim \text{Categorial}([C], p^*(y\mid x)) \\
y_\text{other}\mid y^* &\sim \text{Categorial}([C]\setminus\{y^*\}, \frac{1}{C-1}) \\
z_\text{flip} \mid x &\sim \text{Bernoulli}(p_\text{flip}(x)) \\
y\mid y^*, y_\text{other}, z_\text{flip} &= y^* (1 - z_\text{flip}) + y_\text{other} z_\text{flip}
\end{aligned}
{% endmathjax %}
鉴于真实情况{% mathjax %}p(y|x){% endmathjax %}和{% mathjax %}p_{\text{flip}}{% endmathjax %}根据数据估计，我们将更新主要标签的标签分布：
{% mathjax '{"conversion":{"em":14}}' %}
p^*(y\mid x) = \frac{p(y\mid x) - \frac{p_\text{flip}(x)}{C-1}}{1 - \frac{C \cdot p_\text{flip}(x)}{C - 1}}
{% endmathjax %}
新的测试集取自{% mathjax %}p^* (y|x){% endmathjax %}表示已消除个体不一致噪声的主标签。它可用作无噪声测试集进行评估。为了捕捉注释者在学习预测标签时的系统性分歧，[`Davani`等人,`2021`](https://arxiv.org/abs/2110.05719)尝试了一个多注释者模型，其中预测每个注释者的标签被视为一个子任务。假设分类任务是在带标注的数据集上定义的{% mathjax %}D=(X,A,Y){% endmathjax %}，在这里{% mathjax %}X{% endmathjax %}是文本实例，{% mathjax %}A{% endmathjax %}是标注者的集合，{% mathjax %}Y{% endmathjax %}是标注矩阵，{% mathjax %}y_{ij}\in Y{% endmathjax %}表示分配的二进制标签{% mathjax %}a_j\in A{% endmathjax %}样品{% mathjax %}x_i\in X{% endmathjax %}。多数人投票赞成{% mathjax %}x_i{% endmathjax %}表示为{% mathjax %}\bar{y}_i{% endmathjax %}。实验是在预先训练的`BERT`模型上训练分类头，并比较`4`种设置：
- 基线：直接预测多数票{% mathjax %}\bar{y}_i{% endmathjax %}，不使用完整标注矩阵{% mathjax %}Y{% endmathjax %}。
- 集成：每个标注器分别训练一个模型来预测{% mathjax %}y_{ij}{% endmathjax %}然后根据多数票汇总结果。
- 多标签：学习预测{% mathjax %}|A|{% endmathjax %}标签表示每个样本所有标注者的标签{% mathjax %}\langle y_{i1}, \dots, y_{i\vert A \vert} \rangle{% endmathjax %}，具有共享的`MLP`层，然后聚合输出。
- 多任务：类似于多标签，但每个标注器的预测头都是从分离的`MLP`层学习的，这样我们分配额外的计算来学习标注器之间的差异。

在`GHC`([`Gab Hate Corpus`](https://osf.io/edua3/))数据集上的实验结果表明，该多任务模型取得了最佳`F1`分数，并且能够自然地提供与标注分歧相关的预测不确定性估计。
{% asset_img qd_6.png "用于对多个标注者标签进行建模的不同架构的说明" %}

**陪审团学习**（[`Gordon`等人，`2022`年](https://arxiv.org/abs/2202.02950)）通过根据不同标注者的特征对其标记行为进行建模来模拟陪审团过程。从包含每个标注者的标签和人口统计特征的数据集开始，我们训练一个模型来学习预测每个标注者（每个标注者都是潜在的陪审员）所标记的标签。在决策时，从业者可以指定一组陪审员的组成，以确定抽样策略。最终决定是通过汇总来自多个审判的陪审员的标签来做出的。
{% asset_img qd_7.png "陪审团学习的工作原理说明" %}

陪审团学习模型是一个`DCN`([深度和交叉网络](https://arxiv.org/abs/2008.13535))，通常用于推荐用例，它经过联合训练以学习评论嵌入、标注者嵌入和组（标注者的特征）嵌入。文本内容由预先训练的`BERT`处理，它也经过联合微调，但时间较短，以避免过度拟合。
{% asset_img qd_8.png "用于陪审团学习的DCN模型架构" %}

他们的实验在毒性多样性数据集上运行，并将陪审团学习与基线模型进行比较，基线模型是经过微调的`BERT`，用于在不使用元数据的情况下预测单个注释者的标签。性能以`MAE`（平均绝对误差）来衡量。陪审团学习在整个测试集以及每个组片段上的表现始终优于标注者者的基线。
{% asset_img qd_9.png "将标注者不可知基线与陪审团学习进行比较的实验结果" %}

#### 数据质量<->模型训练

一旦构建了数据集，许多方法都可以根据训练动态帮助识别错误标签。请注意，我们只关注查找和排除可能带有错误标签的数据集的方法，而不是如何使用嘈杂数据训练模型。
##### Influence Functions (影响函数)

影响函数是稳健统计([`Hampel，1974`](https://www.jstor.org/stable/2285666))中的一种经典技术，它通过描述当我们将训练点的权重增加无穷小量时模型参数如何变化来衡量训练数据点的效果。[`Koh & Liang,2017`](https://arxiv.org/abs/1703.04730)引入了该概念并将其应用于深度神经网络。鉴于{% mathjax %}n{% endmathjax %}训练集中的数据样本，{% mathjax %}z_i = (x_i, y_i){% endmathjax %}为了{% mathjax %}i=1,\ldots,n{% endmathjax %}，模型参数{% mathjax %}\theta{% endmathjax %}进行优化以尽量减少损失：{% mathjax %}\hat{\theta} = \arg\min_{\theta \in \Theta} \frac{1}{n}\sum_{i=1}^n \mathcal{L}(z_i, \theta){% endmathjax %}删除单个数据点后模型参数的变化{% mathjax %}z{% endmathjax %}表示为{% mathjax %}\hat{\theta}_{-z} - \hat{\theta}{% endmathjax %}在这里{% mathjax %}\hat{\theta}_{-z} = \arg\min_{\theta \in \Theta} \frac{1}{n} \sum_{z_i \neq z} \mathcal{L}(z_i, \theta){% endmathjax %}。但是，逐一计算每个样本的计算成本太高。一种近似方法是计算给定小权重的参数变化{% mathjax %}\epsilon{% endmathjax %}在{% mathjax %}z{% endmathjax %}。根据定义，权重提升的影响{% mathjax %}z{% endmathjax %}经过{% mathjax %}\epsilon{% endmathjax %}是：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{I}_{\text{up,params}}(z) = \frac{d\hat{\theta}_{\epsilon,z}}{d\epsilon}\bigg\vert_{\epsilon=0}=-\mathbf{H}^{-1}_{\hat{\theta}} \nabla_\theta \mathcal{L}(z, \hat{\theta})
{% endmathjax %}
在这里{% mathjax %} \hat{\theta}_{\epsilon,z} = \arg\min_{\theta \in \Theta} \frac{1}{n}\sum_{i=1}^n \mathcal{L}(z_i, \theta) + \epsilon L(z, \theta){% endmathjax %}和{% mathjax %}\mathbf{H}^{-1}_{\hat{\theta}} = \frac{1}{n}\sum_{i=1}^n \nabla^2_\theta \mathcal{L}(z_i, \hat{\theta}){% endmathjax %}。删除数据点{% mathjax %}x{% endmathjax %}相当于增加它的权重{% mathjax %}\epsilon = -\frac{1}{n}{% endmathjax %}因此{% mathjax %}\hat{\theta}_{-z} - \hat{\theta} \approx -\frac{1}{n} \mathcal{I}_{\text{up,params}}(z){% endmathjax %}。加权的影响{% mathjax %}z{% endmathjax %}测试点的损失{% mathjax %}z_\text{test}{% endmathjax %}通过应用链式法则给出：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{aligned}
\mathcal{I}_{\text{up,loss}}(z, z_\text{test}) 
&= \frac{d \mathcal{L}(z_\text{test}, \hat{\theta}_{\epsilon,z})}{d\epsilon}\bigg\vert_{\epsilon=0} \\
&= \nabla_\theta \mathcal{L}(z_\text{test}, \hat{\theta})^\top \frac{d \hat{\theta}_{\epsilon,z}}{d\epsilon}\bigg\vert_{\epsilon=0} \\
&= - \nabla_\theta \mathcal{L}(z_\text{test}, \hat{\theta})^\top \mathbf{H}^{-1}_{\hat{\theta}} \nabla_\theta \mathcal{L}(z, \hat{\theta})
\end{aligned}
{% endmathjax %}
使用影响函数，我们可以以封闭形式测量单个数据点对模型参数和损失函数的影响。它可以帮助近似留一法再训练，而无需实际运行所有再训练。为了识别错误标记的数据，我们可以测量{% mathjax %}\mathcal{I}_\text{up,loss}(z_i, z_i){% endmathjax %}，如果{% mathjax %}z_i{% endmathjax %}从训练集中移除则近似预测误差为{% mathjax %}z_i{% endmathjax %}。
{% asset_img qd_10.png "使用影响函数计算得到的值与在10类MNIST数据集上进行逐一剔除训练得到的结果是一致的。这表明影响函数可以有效地近似逐一剔除训练的效果，而无需实际进行所有的重新训练" %}

鉴于封闭形式，影响函数仍然难以扩展，因为逆`Hessian`向量积很难计算。[`Grosse`等人,`2023`](https://arxiv.org/abs/2308.03296)尝试使用`EK-FAC`（特征值校正的`Kronecker`因子近似曲率；[`George`等人,`2018`](https://arxiv.org/abs/1806.03884)）近似。
##### 训练期间的预测变化

另一个方法是跟踪训练过程中模型预测的变化，以识别难以学习的情况。数据图([`Swayamdipta`等人，`2020`年](https://arxiv.org/abs/2009.10795))在训练过程中跟踪模型行为动态的两个属性，以分析数据集的质量：
- **置信度**：模型对真实标签的置信度，定义为模型在各个时期内预测真实标签的平均概率。他们还使用了一个粗粒度指标“正确性”，定义为模型在各个时期内预测正确标签的次数比例。
- **可变性**：置信度的变化，定义为跨时期真实标签的模型概率的标准差。
{% asset_img qd_11.png "基于RoBERTa分类器的SNLI训练集数据图" %}

难以学习（低置信度、低变异性）的样本更容易被错误标记。他们在`WinoGrande`数据集上进行了一项实验，其中包含`1%`的翻转标签数据。重新训练后，翻转的实例移动到置信度较低且变异性略高的区域，这表明难以学习的区域包含错误标记的样本。鉴于此，我们可以仅使用置信度分数在相同数量的标签翻转和干净样本上训练分类器（不确定为什么论文没有同时使用置信度和变异性作为特征）。然后可以在原始数据集上使用这个简单的噪声分类器来识别可能被错误标记的实例。
{% asset_img qd_12.png "最初具有高置信度和低变异性分数的数据点在标签翻转后移动到低置信度、变异性稍高的区域" %}

然而，我们不应该认为所有难以学习的样本都是错误的。事实上，本文假设模糊（高变异性）和难以学习（低置信度、低变异性）的样本对学习更有价值。实验表明，它们有利于`OOD`泛化，在`OOD`评估中能取得更好的结果，甚至与`100%`训练集相比也是如此。

**为了研究神经网络是否有忘记**先前学习到的信息的倾向，[`Mariya Toneva`等人,`2019`年](https://arxiv.org/abs/1812.05159)设计了一个实验：他们在训练过程中跟踪每个样本的模型预测，并计算每个样本从正确分类到错误分类或反之亦然的转变。然后可以相应地对样本进行分类，
- 可遗忘（冗余）样本：如果类别标签在训练期间发生变化。
- 不可遗忘的样本：如果类别标签分配在训练期间保持一致，则这些样本一旦学会就永远不会被遗忘。

他们发现，有大量的难忘示例一旦被学习就永远不会被忘记。带有噪声标签的示例或具有“不常见”特征（视觉上难以分类）的图像是最容易被遗忘的示例之一。实验通过经验验证，可以安全地删除难忘示例而不会影响模型性能。在实现中，只有当前训练批次中包含样本时，才会计算遗忘事件；也就是说，它们会在后续小批次中计算同一示例的呈现遗忘。每个样本的遗忘事件数量在不同种子之间相当稳定，并且可遗忘示例在训练后期首次学习的趋势很小。遗忘事件还被发现在整个训练期间和架构之间是可转移的。[`Pleiss`等人,`2020`年](https://arxiv.org/abs/2001.10528)开发了一种名为`AUM`（边缘下面积）的方法来识别错误标签，该方法基于这样的假设：假设一张`BIRD`图像被错误地标记为`DOG`。梯度更新将鼓励从其他`BIRD`图像推广到这张`BIRD`图像，而`DOG`标签提供了错误的监督信号，鼓励更新朝另一个方向发展。因此，在梯度更新信号中，推广和（错误）预测之间存在矛盾。

给定分类数据集{% mathjax %}(\mathbf{x},y\in \mathcal{D}_{\text{train}}){% endmathjax %}，让{% mathjax %}z^{(t)}_i(\mathbf{x}) \in \mathbb{R}{% endmathjax %}是与类别对应的逻辑回归{% mathjax %}i{% endmathjax %}在时间步步长{% mathjax %} {% endmathjax %}。`epoch`的边距{% mathjax %}t{% endmathjax %}是指定`logit`与最大`logit`之间的差异：
{% mathjax '{"conversion":{"em":14}}' %}
M^{(t)}(\mathbf{x}, y) = z_y^{(t)}(\mathbf{x}) - \max_{i \neq y} z^{(t)}_i(\mathbf{x}),\quad
\text{AUM}(\mathbf{x}, y) = \frac{1}{T} \sum^T_{t=1} M^{(t)}(\mathbf{x}, y)
{% endmathjax %}
负边距表示预测错误，较大的正边距表示对正确预测的信心较高。假设错误标记的样本的边距会比正确样本小，这是由于其他样本引发的通过`SGD`进行泛化的张力。为了确定阈值，他们插入了虚假数据，称为“阈值样本”，以确定阈值：
- 创建阈值样本子集{% mathjax %}\mathcal{D}_{}\text{thr}{% endmathjax %}。如果有{% mathjax %}N{% endmathjax %}训练样本{% mathjax %}C{% endmathjax %}，我们随机抽取{% mathjax %}N/(C+1){% endmathjax %}样本并将其所有标签切换为假的新类别{% mathjax %}C+1{% endmathjax %}。
- 将阈值样本合并到原始数据集中：{% mathjax %}\mathcal{D}’ = { (\mathbf{x}, C+1): \mathbf{x} \in \mathcal{D}_\text{thr}} \cup (\mathcal{D} \setminus\mathcal{D}_\text{thr}){% endmathjax %}。
- 训练模型{% mathjax %}\mathcal{D}'{% endmathjax %}并测量所有数据的`AUM`。
- 计算阈值{% mathjax %}\alpha{% endmathjax %}作为阈值样本`AUM`的第`99`个百分位数。
- 使用以下方法识别错误标记的数据{% mathjax %}\alpha{% endmathjax %}阈值：{% mathjax %}{(\mathbf{x}, y) \in \mathcal{D} \setminus \mathcal{D}_\text{thr}: \text{AUM}_{\mathbf{x}, y} \leq \alpha}{% endmathjax %}
{% asset_img qd_13.png "阈值样本的AUM如何帮助分离出错误标记的样本" %}

{% asset_img qd_14.png "使用随机错误标记的样本在CIFAR 10/100上的测试误差，比较了不同的数据过滤或噪声数据训练方法" %}

##### 噪声交叉验证

`NCV`（噪声交叉验证）方法([`Chen et al, 2019`](https://arxiv.org/abs/1905.05040))将数据集随机分成两半，然后如果数据样本的标签与仅在另一半数据集上训练的模型提供的预测标签相匹配，则将数据样本标识为“干净”。干净的样本预计更值得信赖。`INCV`（迭代噪声交叉验证）迭代运行`NCV`，其中更多干净样本被添加到可信候选集中{% mathjax %}\mathcal{C}{% endmathjax %}并去除了更多噪声样本。
{% asset_img qd_15.png "INCV（迭代噪声交叉验证）算法" %}

