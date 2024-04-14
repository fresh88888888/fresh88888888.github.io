---
title: Stable Diffusion（KerasCV）
date: 2024-04-14 19:06:11
tags:
  - AI
categories:
  - 人工智能
---

使用`KerasCV`的稳定扩散图像生成。`Stable Diffusion`是一个强大的文本 `->` 图像模型，有`Stability AI`开源。虽然存在多种开源实现，可以轻松地根据文本提示创建图像，但`KerasCV`提供了一些优势：其中包括`XLA`编译和混合精度支持，他们共同实现最优的生成，使用`KerasCV`调用`Stable Diffusion`非常简单。我们传入一个字符串，通常称为**提示**，批量大小为`3`。模型能够生成三张令人惊艳的图片，正如**提示**所描述：
<!-- more -->
{% asset_img sd_1.png %}

