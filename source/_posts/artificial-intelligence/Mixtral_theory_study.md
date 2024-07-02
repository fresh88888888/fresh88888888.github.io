---
title: Mistral / Mixtral：滑动窗口注意力 & 稀疏专家混合 & 滚动缓冲区
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

`Mixtral`是由`Mistral AI`公司开发的一种先进的大型语言模型。`Mixtral`采用混合专家(`Mixture of Experts, MoE`)架构，总参数量为`46.7B`，但每次推理只使用约`12.9B`参数，稀疏混合专家网络架构，每层包含`8`个专家(前馈神经网络块)，对每个`token`,路由器选择`2`个专家处理，`32K tokens`的上下文窗口，支持英语、法语、意大利语、德语和西班牙语，在代码生成方面表现出色。在多项基准测试中表现优异，超越了许多更大规模的模型，推理速度快，效率高；在多数基准测试中优于`Llama 2 70B`和`GPT-3.5`，推理速度是`Llama 2 70B`的`6`倍。
<!-- more -->

#### Transformer vs Mistral

{% asset_img m_1.png %}

{% asset_img m_2.png %}

#### 自注意力机制

自注意力机制允许模型将单词相互关联。假设我们有以下句子：`“The cat is on a chair”`。
{% asset_img m_3.png %}

这里我展示了在应用`softmax`之前{% mathjax %}Q{% endmathjax %}和{% mathjax %}K{% endmathjax %}矩阵的乘积。
{% asset_img m_4.png %}
