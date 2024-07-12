---
title: LLaMA 2 模型—探析（PyTorch）
date: 2024-07-12 15:00:11
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

`LLaMA 2`是`Meta AI`(原`Facebook AI`)在`2023`年`7`月发布的大型语言模型系列,是`LLaMA`模型的第二代版本。**模型规模**：包含`70`亿、`130`亿和`700`亿参数三种规模的模型。比`LLaMA 1`增加了一个`700`亿参数的大型模型。**训练数据**：使用`2`万亿个`tokens`进行预训练,比`LLaMA 1`增加了`40%`；完全使用公开可用的数据集,不依赖专有数据。**性能改进**：在多数基准测试中,性能超过了同等规模的开源模型；`130`亿参数版本在某些任务上甚至超过了`GPT-3`(`1750`亿参数)。**对话优化**：提供了针对对话场景优化的`LLaMA 2-Chat`版本；使用了超过`100`万人工标注进行微调。**安全性**：在模型训练中加入了安全性改进措施；使用**人类反馈强化学习**(`RLHF`)来确保安全性和有用性。**技术创新**：使用分组查询注意力(`GQA`)机制提高效率；上下文长度增加到`4096 tokens`,是`LLaMA 1`的两倍。
<!-- more -->

`LLaMA 2`采用了经典的`Transformer`架构，但在多个方面进行了优化，以提高模型的性能和效率：
- `Transformer`架构：`LLaMA 2`基于经典的`Transformer`架构，利用注意力机制来理解文本的上下文关系。
- 解码器结构：`LLaMA 2`采用了仅解码器的`Transformer`架构，这种架构在生成任务中表现出色。
- `RMSNorm(Root Mean Square Layer Normalization)`：取代了传统的`Layer Normalization`，`RMSNorm`有助于提高训练的稳定性和效率。
- `SwiGLU`激活函数：采用了`SwiGLU`激活函数，而不是标准的`ReLU`激活函数，这种选择有助于提升模型的表现。
- `RoPE(Rotary Positional Embedding)`位置编码：使用旋转位置编码来处理位置信息，这种方法在处理长序列时表现更好。
- `Grouped Query Attention(GQA)`：引入了**分组查询注意力机制**，以加速推理过程。

{% asset_img ll_1.png %}

