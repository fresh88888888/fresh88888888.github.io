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

`MixLoRA`[1]是一种用于优化大规模语言模型(`LLMs`)微调的新方法，结合了`LoRA`(`Low-Rank Adaptation`)和专家混合(`Mixture of Experts, MoE`)技术。大规模语言模型的微调通常需要大量的计算资源和显存。`LoRA`通过引入低秩适配器，显著减少了微调时的参数数量和显存需求。然而，`LoRA`在多任务学习场景中的性能仍有提升空间。专家混合模型(`MoE`)在多任务学习中表现出色，但其资源需求对普通消费者级`GPU`来说是一个挑战。
<!-- more -->

主要特点：
- **架构设计**：1.`MixLoRA`在冻结的预训练密集模型的前馈网络块中插入多个`LoRA`专家模块；2.使用常见的`top-k`路由器（如`Mixtral`）或`top-1`开关路由器（如`Switch Transformers`）来动态选择合适的专家。
- **性能提升**：1.`MixLoRA`在多任务学习场景下的准确率比现有的参数高效微调(`PEFT`)方法提高了约`9%`；2.通过独立的注意力层`LoRA`适配器增强模型性能。
- **资源效率**：1.引入辅助负载平衡损失来解决路由器的不平衡问题；2.新的高吞吐框架在训练和推理过程中减少了`40%`的`GPU`显存消耗和`30%`的计算延迟。

`MixLoRA`通过结合`LoRA`和专家混合技术，提供了一种高效的微调大规模语言模型的方法，显著提升了多任务学习的性能，同时大幅减少了计算资源和显存的需求。

#### 引用

[1]：[MIXLORA: Enhancing Large Language Models Fine-Tuning with LoRA-based Mixture of Experts](https://arxiv.org/pdf/2404.15159v2)