---
title: LoRA模型—探析（PyTorch）
date: 2024-07-18 12:00:11
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

`LoRA`(`Low-Rank Adaptation`) 是一种用于大型语言模型微调的高效技术。`LoRA`旨在解决大语言模型微调时的**计算资源和存储空间**问题。在原始预训练模型中增加一个低秩矩阵作为旁路,只训练这个低秩矩阵,而冻结原模型参数。**工作原理**：在原模型权重矩阵{% mathjax %}W{% endmathjax %}旁边增加一个低秩分解矩阵{% mathjax %}BA{% endmathjax %}；{% mathjax %}B{% endmathjax %}是一个{% mathjax %}d\times r{% endmathjax %}的矩阵,{% mathjax %}A{% endmathjax %}是一个{% mathjax %}r\times k{% endmathjax %}的矩阵,其中{% mathjax %}r\ll \min(d,k){% endmathjax %}；训练时只更新{% mathjax %}A{% endmathjax %}和{% mathjax %}B{% endmathjax %},保持原始权重{% mathjax %}W{% endmathjax %}不变；推理时将{% mathjax %}BA{% endmathjax %}与{% mathjax %}W{% endmathjax %}相加:{% mathjax %}W + BA{% endmathjax %}。
<!-- more -->
{% asset_img l_1.png %}

**优点**：大幅减少可训练参数数量,降低计算和存储开销；训练速度更快,使用内存更少；可以为不同任务训练多个`LoRA`模块,便于切换。总之,`LoRA`通过引入低秩矩阵作为可训练参数,有效解决了大模型微调的资源问题,为特定任务的模型适配提供了高效的解决方案。
