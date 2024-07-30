---
title: 基于LoRA构建稀疏混合专家模型(MoE)的方法(MixLoRA)-探析(微调)
date: 2024-07-29 17:25:11
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

#### 介绍

[`MixLoRA`](https://arxiv.org/pdf/2404.15159v2)是一种用于优化大规模语言模型(`LLMs`)微调的新方法，结合了`LoRA`(`Low-Rank Adaptation`)和专家混合(`Mixture of Experts, MoE`)技术。大规模语言模型的微调通常需要大量的计算资源和显存。`LoRA`通过引入低秩适配器，显著减少了微调时的参数数量和显存需求。然而，`LoRA`在多任务学习场景中的性能仍有提升空间。专家混合模型(`MoE`)在多任务学习中表现出色，但其资源需求对普通消费者级`GPU`来说是一个挑战。
<!-- more -->

**主要特点**：
- **架构设计**：1.`MixLoRA`在冻结的预训练密集模型的前馈网络块中插入多个`LoRA`专家模块；2.使用常见的`top-k`路由器（如`Mixtral`）或`top-1`开关路由器（如`Switch Transformers`）来动态选择合适的专家。
- **性能提升**：1.`MixLoRA`在多任务学习场景下的准确率比现有的参数高效微调(`PEFT`)方法提高了约`9%`；2.通过独立的注意力层`LoRA`适配器增强模型性能。
- **资源效率**：1.引入辅助负载平衡损失来解决路由器的不平衡问题；2.新的高吞吐框架在训练和推理过程中减少了`40%`的`GPU`显存消耗和`30%`的计算延迟。

大型语言模型(`LLM`)的**指令微调**已在自然语言处理(`NLP`)领域中取得了很好的成绩。随着参数规模的增加，`LLM`已被证明能够识别复杂的语言模式，从而实现强大的跨任务泛化能力。然而指令微调将会使其在计算资源和下游任务的性能之间做出权衡。为了减少全参数微调过程所需的计算和内存资源，引入了参数高效微调(`PEFT`)方法，其中，低秩自适应是流行的`PEFT`方法中的一种。`MixLoRA`通过结合`LoRA`和专家混合技术，提供了一种高效的微调大规模语言模型的方法，显著提升了多任务学习的性能，同时大幅减少了计算资源和显存的需求。
{% asset_img ml_1.png  "LoRA-MoE方法发布日期的时间表，包括有关集成位置的详细模型信息、如何使用LoRA-MoE方法进行训练以及它们旨在解决的问题" %}

`PEFT`，**大型语言模型**(`LLM`)在`NLP`任务中表现出了卓越的能力。在此之后，**指令微调**进一步使`LLM`能够理解人类意图并遵循指令，成为聊天系统的基础。但是，随着LLM规模的扩大，对其进行微调成为一个耗时且占用大量内存的过程。为了缓解这个问题，研究探索了不同的方法：**参数高效微调**(`PEFT`)、**蒸馏**、**量化**、**修剪**等。`LoRA`利用**低秩矩阵**分解线性层权重，是最流行的`PEFT`方法之一，不仅提高了模型性能，而且不会在推理过程中引入任何额外的计算开销。例如，`VeRA`结合可学习的缩放向量来调整跨层的共享冻结随机矩阵对。此外，`FedPara`专注于**联邦学习**场景的低秩`Hadamard`积。`Tied-Lora`实现了权重绑定以进一步减少可训练参数的数量。`AdaLoRA`使用**奇异值分解**(`SVD`)来**分解矩阵**并且修剪不太重要的奇异值以简化更新。`DoRA`将预训练权重分解为两个分量，即**幅度**和**方向**，并在微调期间利用`LoRA`进行方向更新，从而有效减少可训练参数的数量。

**专家混合**：专家混合(`MoE`)的概念可以追溯到`1991`年，它引入了一种新颖的监督学习方法，涉及多个网络（专家），每个网络专门处理一组训练示例。`MoE`的现代版本通过合并**稀疏激活的专家**来修改`Transformer`块内的前馈子层，从而能够在不增加计算量的情况下大幅增加**模型宽度**。`LLaVA-MoLE`有效地将`token`路由到`Transformer`层内的特定领域专家，从而缓解数据冲突并实现与普通`LoRA`基线一致的性能提升。对于其他基于`MoE`的架构，`MoRAL`解决了将`LLM`适应新领域/任务的挑战。`LoRAMoE`使用路由器网络集成`LoRA`，以缓解**知识遗忘**。`PESC`使用`MoE`架构将密集模型转换为稀疏模型，从而降低了计算成本和`GPU`内存要求。`MoE-LoRA`提出了一种新颖的参数高效的`MoE`方法，该方法具有分层专家分配(`MoLA`)，适用于`Transformer`的模型。`MoCLE`可根据指令集群激活定制任务的模型参数。`MixLoRA`将`LoRA`作为随机专家进行集成，从而降低了计算成本，同时扩展了模型容量并增强了`LLM`的泛化能力。
{% asset_img ml_2.png  "MixLoRA架构。MixLoRA由n位专家组成，由原始FFN子层与不同的LoRA组合而成，其中FFN子层的权重由所有专家共享" %}

#### MixLoRA架构

`LoRA`仅调整额外的自适应参数并替换原始权重更新。`LoRA`块由两个矩阵组成，{% mathjax %}\mathbf{B}\in \mathbb{R}^{d_1\times r}{% endmathjax %}和{% mathjax %}\mathbf{A}\in \mathbb{R}^{r\times d_2}{% endmathjax %}，其中{% mathjax %}d_1{% endmathjax %}和{% mathjax %}d_2{% endmathjax %}表示`LLM`预训练权重{% mathjax %}\mathbf{W}{% endmathjax %}的维度({% mathjax %}\mathbf{W}\in \mathbb{R}^{d_1\times d_2}{% endmathjax %})，参数{% mathjax %}r{% endmathjax %}表示`LoRA`的隐藏维度，其中{% mathjax %}r\ll \min(d_1, d_2){% endmathjax %}。然后，更新后的权重{% mathjax %}\mathbf{W}{% endmathjax %}，通过以下公式计算：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{W}' = \mathbf{W} + \mathbf{BA}
{% endmathjax %}
**混合专家**(`MoE`)。`MoE`架构最初通过`GShard`引入到语言模型中。`MoE`层由`n`个专家组成，表示为{% mathjax %}\{E_i\}^n_{i=1}{% endmathjax %}，以及路由器{% mathjax %}R{% endmathjax %}。给定隐藏状态{% mathjax %}h{% endmathjax %}的`MoE`层输出{% mathjax %}h'{% endmathjax %}由以下公式确定：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{h}' = \sum_{i=1}^n R(\mathbf{h})_iE_i(\mathbf{h})
{% endmathjax %}
其中，{% mathjax %}R(\mathbf{h})_i{% endmathjax %}表示路由器输出到第{% mathjax %}i{% endmathjax %}位专家，{% mathjax %}E_i(\mathbf{h}){% endmathjax %}是第{% mathjax %}i{% endmathjax %}位专家的结果。`MixLoRA`由两个主要部分构成。第一部分添加了`LoRA`的`vanilla Transformer`块构建稀疏`MoE`块。第二部分利用`top-k`路由器将来自各种任务（例如`ARC、OBQA、PIQA`等）的每个`token`分配给不同的专家模块。假设输入文本为{% mathjax %}s = (s_1, s_2, \ldots, s_n){% endmathjax %}，标签为{% mathjax %}y{% endmathjax %}。让{% mathjax %}h_i^\ell \in \mathbb{R}^{1\times d}(1\leq i\leq n, 1\leq \ell \leq L){% endmathjax %}表示第{% mathjax %}\ell{% endmathjax %}个大型语言模型(`LLM`)层的第{% mathjax %}i{% endmathjax %}个`token`的输出隐藏状态，其中{% mathjax %}L{% endmathjax %}是`LLM`层的总数，{% mathjax %}d{% endmathjax %}是隐藏维度。大型语言模型由堆叠的多头自注意力(`MSA`)和前馈神经网络(`FFN`)组成。每个块内都应用了层归一化(`LN`)和残差连接。普通`transformers`块中第{% mathjax %}\ell{% endmathjax %}个LLM层的输出{% mathjax %}h{% endmathjax %}ℓ 通过以下方式计算：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{split}
\mathbf{h}^0 = [s_1, s_2,\ldots, s_n] \\
\mathbf{x}^{\ell} = \text{MSA}(\text{LN}(\mathbf{h}^{\ell - 1})) + \mathbf{h}^{\ell - 1}, \;\;\;\mathbf{h}^{\ell} = \text{FFN}(\text{LN}(\mathbf{z}^{\ell})) + \mathbf{z}^{\ell}
\end{split}
{% endmathjax %}
`MixLoRA Forward`。`MixLoRA`基于`LoRA`专家构建的。`MixLoRA`利用`LoRA`在微调期间存储每个专家的更新参数，而不是仅使用`LoRA`来构建每个专家。这种方法使`MixLoRA`与现有的预训练`MoE`模型更加一致。在`MixLoRA`的`MoE`块中，这些专家的基本权重由密集模型的单个**前馈网络**(`FFN`)共享，以提高训练和推理效率：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{split}
\mathbf{h}^{\ell} = \text{MixLoRA}(\text{LN}(\mathbf{z}^{\ell})) + \mathbf{z}^{\ell} \\
\text{MixLoRA}(\mathbf{h}^{\ell}) = \sum_{k=1}^K R^{\ell}(\mathbf{h}^{\ell})_k E_k^{\ell}(\mathbf{h}^{\ell}),\;\;\; E_k^{\ell}(\mathbf{h}^{\ell}) = \mathbf{W}^{\ell}\cdot \mathbf{h}^{\ell} + \mathbf{B}_i^{\ell}\mathbf{A}_i^{\ell}\cdot \mathbf{h}^{\ell}
\end{split}
{% endmathjax %}
其中{% mathjax %}\mathbf{W}{% endmathjax %}是`FFN`层的预训练权重，由{% mathjax %}\{E_k\}_{k=1}^K{% endmathjax %}共享，{% mathjax %}R(\cdot){% endmathjax %}表示我们用不同`token`和任务选择特定`LoRA`专家的`Top-K`路由器，{% mathjax %}E_k(\cdot){% endmathjax %}表示`MixLoRA`模块中的第`k`个`LoRA`专家。`MixLoRA`的作用是取代公式中密集模型的`FFN`层，其关键概念是通过路由器为每个`token`选择不同的专家，其中每个专家由不同的`LoRA`和原始`FFN`层组成。

`Top-K`路由器。`MoE`层中的`Top-K`路由器将每个`token`分配给最合适的专家。**路由器是一个线性层**，它计算输入`token`{% mathjax %}x_i{% endmathjax %}被路由到每个专家的概率：{% mathjax %}\mathbf{W}_r(s){% endmathjax %}。在稀疏转换器块中，此路由器根据输入`token`激活最合适的`LoRA`专家。它利用`softmax`激活函数来模拟专家的概率分布。路由器的权重{% mathjax %}\mathbf{W}_r{% endmathjax %}是路由网络的可训练参数。我们的设计采用了`top-2`门控路由器，它为每个输入`token `{% mathjax %}x_i{% endmathjax %}从{% mathjax %}n{% endmathjax %}个可用的{% mathjax %}\{E_k\}_{k=1}^K{% endmathjax %}中选择最佳的两个专家：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
R^{\ell}(\mathbf{h}^{\ell}_i) = \text{KeepTop-2}(\text{Softmax}(\mathbf{W}_r^{ell}\cdot \mathbf{x}_i))
\end{align}
{% endmathjax %}
在推理过程中，`top-k`门控路由器会动态地为每个`token`选择最佳的`k`个专家。通过这种机制，**混合专家**和**路由器**协同工作，使专家能够开发不同的能力并有效地处理不同类型的任务。专家负载平衡。专家负载不平衡是`MoE`面临的重大挑战。这是因为某些专家往往会更频繁地被`top-k`路由器选择。为了解决这种负载不平衡问题，我们在训练时应用负载平衡损失来减轻专家的负载不平衡。受`Switch Transformers`的启发，计算辅助损失并将其添加到总损失中。给定{% mathjax %}N{% endmathjax %}个专家（索引为`i = 1`到`N`）和具有`T`个`token`的批次`B`，辅助损失计算如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathcal{L}_{\text{aux}} = a \cdot N \cdot \sum_{i=1}^N \mathcal{F}_i \cdot \mathcal{P}_i
\end{align}
{% endmathjax %}

{% asset_img ml_3.png  "前向传播过程的比较：(a)原始MIXLORA MoE块中的过程；(b)共享W1和W3计算结果以降低计算复杂度的优化过程" %}

{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathcal{F}_i = \frac{1}{T} \sum_{x\in B} \mathcal{I}\{\text{argmax}_k R(x)_k = i\},\;\; \mathcal{P}_i = \frac{1}{T}\sum_{x\in B} R(x)_i
\end{align}
{% endmathjax %}
其中{% mathjax %}R(\cdot){% endmathjax %}是前{% mathjax %}k{% endmathjax %}个路由器，{% mathjax %}\mathcal{F}_i{% endmathjax %}是分配给专家{% mathjax %}i{% endmathjax %}的`token`分数，{% mathjax %}\mathcal{P}_i{% endmathjax %}是分配给专家{% mathjax %}i{% endmathjax %}的路由器概率的分数。最终损失乘以专家数量{% mathjax %}N{% endmathjax %}，以保持损失在专家数量变化时保持不变。此外，我们使用{% mathjax %}a = 10^{-2}{% endmathjax %}作为辅助损失的乘积系数，该系数足够大以确保负载平衡，同时保持足够小而不会影响主要交叉熵目标。添加带有注意力层的`LoRA`。`MixLoRA`进一步扩展了其微调功能以涵盖注意力层。先前的研究，例如`ST-MoE`，表明微调注意力层可以显着提高性能。为了增强`MixLoRA`的微调过程，将`LoRA`适配器集成到密集模型的注意力层中。实验结果表明，经过`q、k、v`和`o`投影微调的`MixLoRA`模型与仅使用稀疏的`LoRA`专家混合层（即`MixLoRA MoE`）训练的配置相比，始终能够获得更高的平均分数。

#### 性能优化

- **降低计算复杂度**。`MixLoRA`中的每个专家网络包括一个共享的冻结`FFN`和多个`LoRA`，用于在微调期间存储专家的每个线性投影层更新参数。此设置导致计算复杂度根据路由器的`K`设置而变化。这是因为`MixLoRA`中的`token`级`Top-K`路由器将每个`token`发送给`K`个专家进行计算，然后聚合他们的残差并产生输出。以`LLaMA`为例，`LLaMA`的`FFN`块由三个线性投影权重组成：{% mathjax %}W_1,W_2{% endmathjax %}和{% mathjax %}W_3{% endmathjax %}，前向传播过程可以表示为{% mathjax %}H = W_2(\text{SiLU}(W_1(x))\cdot W_3(x)){% endmathjax %}。如图 3 (a) 所示，`MixLoRA`中的每个专家都有相同的前向传播过程，只是每个线性投影层都有一个单独的 LoRA，并且每个专家的输入 x 由 `MixLoRA`路由器预先分配。这在处理长序列输入时会带来显著的开销，对`MixLoRA`的性能优化提出了重大挑战：如何在保持模型精度的同时降低 `MixLoRA`的计算复杂度？考虑到我们已经共享了 FFN 块的权重，我们进一步共享计算结果以降低计算复杂度。如上图所示，方法不是预先分配`MixLoRA`块的输入序列，而是首先将输入直接并行发送到`FFN`块的{% mathjax %}W_1{% endmathjax %}和{% mathjax %}W_3{% endmathjax %}，然后根据`MixLoRA`路由器输出的路由权重对这些线性投影的输出进行**切片**。{% mathjax %}W_2{% endmathjax %}投影的计算复杂度无法降低，因为其计算过程取决于{% mathjax %}{% endmathjax %}和{% mathjax %}W_3{% endmathjax %}投影的输出。这种方法显著降低了`MixLoRA`计算复杂度的三分之一。`MoE`块。在实验中，这种方法保持相同的模型性能的情况下`token`计算延迟比`vanilla MixLoRA`大约低`30%`。
- **优化多**`MixLoRA`**训练和推理**。受`m-LoRA`提出的多`LoRA`优化的启发，我们还优化了`MixLoRA`以实现多模型高吞吐量训练和推理。之前，我们通过消除重复计算来降低`MixLoRA`的计算复杂度。当使用两个或更多`MixLoRA`模型进行训练和推理时，这些模型的多任务输入被打包成一个批次，以提高训练吞吐量。具体来说，我们首先将批处理输入并行发送到{% mathjax %}W_1{% endmathjax %}和{% mathjax %}W_3{% endmathjax %}，然后使用`MixLoRA`模型的单独路由权重对这些线性投影的输出进行切片。这种方法通过共享相同的**预训练模型权重**，显着减少了多个`MixLoRA`模型的内存使用量。这种方法保持了相同的性能的情况下对每个模型的峰值`GPU`内存使用量减少了约`45%`。
{% asset_img ml_4.png  "比较不同的PEFT方法进行单任务学习，使用不同架构和参数数量的基础模型（报告结果：是准确度得分）" %}

#### 结论

这里介绍了`MixLoRA`，这是一种使用多个`LoRA`的专家和一个冻结的共享`FFN`块的`MoE`方法。与传统的`LoRA-MoE`方法不同，`MixLoRA`将多个`LoRA`与共享`FFN`层融合，并在微调期间使用它们为每个专家存储更新的参数，使其与预先训练的`MoE`模型更加一致。它还采用了自注意力`LoRA`适配器和辅助负载平衡损失来提高性能并解决路由器不平衡问题。此外，还设计了一个高性能框架来优化`MixLoRA`中多个`LoRA`的专家的计算过程，以进行训练和推理。因此，该框架将`MixLoRA`的计算复杂度降低了`30%`，并在训练或推理多个`MixLoRA`模型时节省了大约`40%`的`GPU`内存使用量。评估表明，`MixLoRA`在单任务和多任务场景中的表现优于基线。对于单任务学习，`MixLoRA`在`LLaMA-2 7B`上的平均准确率比`LoRA`提高了`6.2%`，比`DoRA`提高了`2.6%`。在多任务学习中，`MixLoRA`的准确率明显优于`LoRA 9.8%`，优于`DoRA 9%`。