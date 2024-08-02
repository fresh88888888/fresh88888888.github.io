---
title: SAM2模型-探析(深度学习)
date: 2024-08-02 15:44:11
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

[`SAM2(Segment Anything Model 2)`](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)是`Meta AI`最新发布的图像和视频分割模型,是`Segment Anything Model(SAM)`的下一代模型。`SAM2`是一个统一的模型,可以同时处理**图像**和**视频**的**分割任务**。这种统一的架构简化了部署,并在不同媒体类型中实现了一致的性能。`SAM2`采用了**提示式视觉分割**(`Promptable Visual Segmentation, PVS`)的方法。用户可以通过**点击**、**边界框**或**掩码**等方式在视频的任何帧上提供提示,模型会立即生成相应的分割掩码,并将其传播到整个视频中。
<!-- more -->

`SAM2`还可以分割任何视频或图像中的任何对象（通常称为**零样本泛化**），这意味着它可以应用于以前从未见过的视觉内容，而无需进行自定义调整。
{% asset_img ml_1.png  "SAM2模型在SA-V的数据集上进行训练，主要解决基于提示的视觉分割任务" %}

**图像分割**：`Segment Anything`（`Kirillov`等人，`2023`年）引入了一种可以提示的图像分割任务，其目标是在给定输入提示（例如边界框或指向感兴趣对象的点）的情况下输出有效的分割掩码。在`SA-1B`数据集上训练的`SAM`允许使用灵活提示进行零样本分割，从而使其能够应用于广泛的下游应用。最近的工作通过提高其质量扩展了`SAM`。例如，`HQ-SAM`（`Ke`等人，`2024`年）通过引入高质量输出`token`并在细粒度掩码上训练模型来增强`SAM`。另一项工作重点是提高`SAM`的效率，使其在现实世界和移动应用中得到更广泛的应用，例如`EfficientSAM`（`Xiong`等人，`2023`年）、`MobileSAM`（`Zhang`等人，`2023`年）和`FastSAM`（`Zhao`等人，`2023`年）。

