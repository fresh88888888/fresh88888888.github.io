---
title: 视频生成的扩散模型（深度学习）
date: 2024-06-18 14:50:11
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

过去几年，扩散模型在图像合成方面取得了显著成果。现在，研究界开始研究一项更艰巨的任务——将其用于视频生成。这项任务本身是图像情况的超集，因为图像是`1`帧的视频，而且它更具挑战性，因为：
- 它对时间上跨帧的时间一致性有额外的要求，这自然要求将更多的世界知识编码到模型中。
- 相比于文本或图像，收集大量高质量、高维的视频数据更加困难。
<!-- more -->

#### 视频生成建模

##### 参数化和采样

让{% mathjax %}\mathbf{x}\sim q_{\text{real}}{% endmathjax %}是从真实数据分布中采样的数据点。现在我们在时间上添加少量的高斯噪声，从而创建一系列噪声变化{% mathjax %}\mathbf{x}{% endmathjax %}，表示为{% mathjax %}\{\mathbf{z}_t|t=1,\ldots,T\}{% endmathjax %}，随着噪声量增加而t{% mathjax %}t{% endmathjax %}增加，最后{% mathjax %}q(\mathbf{z}_T)\sim \mathcal{N}(\mathbf{0},\mathbf{I}){% endmathjax %}。加噪前向过程为高斯过程。设{% mathjax %} {% endmathjax %}定义高斯过程的可微噪声过程。设{% mathjax %}\alpha_t,\sigma_t{% endmathjax %}定义高斯过程的可微噪声过程：
{% mathjax '{"conversion":{"em":14}}' %}
q(\mathbf{z}_t|\mathbf{x})= \mathcal{N}(\mathbf{z}_t;\alpha_t\mathbf{x},\sigma^2_t\mathbf{I})
{% endmathjax %}
