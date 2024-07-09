---
title: BERT模型—探析（Transformer）
date: 2024-07-08 18:30:11
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

语言模型是一种概率模型，它为单词序列分配概率。实际上，语言模型允许我们计算以下内容：我们通常训练一个神经网络来预测这些概率。在大量文本上训练的神经网络被称为大型语言模型(`LLM`)。
<!-- more -->
{% asset_img b_1.png %}

怎样训练&推理一个语言模型？假设我们想要训练一个中文诗歌语言模型，例如下面这个：
{% asset_img b_2.png %}

假设你是一个（懒惰的）学生，必须记住李白的诗，但只记得前两个字。你如何背诵出全诗？
{% asset_img b_3.png %}

{% asset_img b_4.png %}