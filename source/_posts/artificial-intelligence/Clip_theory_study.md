---
title: CLIP模型—探析（深度学习）
date: 2024-07-25 17:10:11
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

`CLIP`(`Contrastive Language-Image Pretraining`)是由`OpenAI`开发的一种**多模态学习模型**，旨在通过**自然语言描述来学习视觉概念**。`CLIP`的核心在于将图像和文本嵌入到一个共同的语义空间中，从而实现跨模态的理解和应用。
<!-- more -->

`CLIP`工作原理可以分为以下几个步骤：
- **数据收集与预训练**：`CLIP`在互联网上收集了`4`亿对图像和文本数据进行预训练。模型通过**对比学习**，将图像和文本映射到一个**共享的向量空间**中，使得相关的图像和文本靠近，不相关的则远离。
- **编码器**：使用**图像编码器**（如`ResNet`或`Vision Transformer`）和**文本编码器**（如`CBOW`或`Text Transformer`）分别对图像和文本进行编码。
- **相似度计算**：计算图像和文本向量之间的**余弦相似度**，并通过对比损失函数优化模型，使得正样本对的相似度较高，负样本对的相似度较低。
- **零样本预测**：训练好的模型可以直接进行**零样本预测**，即无需额外训练，直接使用自然语言标签进行图像分类或检索。

`CLIP`的优势：
- **高效的无监督学习**：利用大量的图像-文本对进行训练，减少了对标注数据的依赖。
{% asset_img c_1.png  %}

- **广泛的应用场景**：在图像分类、检索和生成等任务中表现出色。
- **灵活性和通用性**：能够在多个任务和数据集上实现零样本学习，包括细粒度对象分类、地理定位、视频中的动作识别和`OCR`等任务表现出较强的通用性。

虽然`CLIP`通常能很好地识别常见物体，但它在更抽象或系统的任务（例如计算图像中的物体数量）和更复杂的任务（例如预测照片中最近的汽车有多近）上表现不佳。在这两个数据集上，零样本`CLIP`仅比随机猜测略胜一筹。与特定任务模型相比，零样本`CLIP`在非常细粒度的分类上也表现不佳，例如区分汽车型号、飞机变体或花卉种类。`CLIP`对其预训练数据集中未涵盖的图像的泛化能力仍然较差。例如，尽管`CLIP`学习了一个功能强大的`OCR`系统，但在对`MNIST`数据集中的手写数字进行评估时，零样本`CLIP`的准确率仅为`88%`，远低于人类在该数据集上的`99.75%`。最后观察到`CLIP`的零样本分类器可能对措辞或短语很敏感，有时需要反复试验的才能取得良好效果。
{% asset_img c_2.png  'CLIP联合训练图像编码器和文本编码器来预测一批(图像,文本)对。在测试时，学习到的文本编码器通过嵌入目标数据集类别的名称(或描述)来合成零样本线性分类器。' %}

`CLIP`采用了两种不同的图像编码器架构：一种是`ResNet-50`作为图像编码的基础架构（注意力池替换了全局平均池）；另一种是`Vision Transformer(ViT)`，在位置嵌入中添加了一层（层归一化）。
```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```
文本编码器是一个`Transformer`模型（修改于`Radford`）。使用一个`63M`参数、`12`层、`512`维度的模型，带有`8`个注意力头。`Transformer`对文本用小写字节对编码(`BPE`)表示进行操作，词汇量为`49,152`。为了提高计算效率，最大序列长度上限为`76`。文本序列用了`[SOS]`和`[EOS]`两个`token`，Transformer最高层在`[EOS]token`处的激活被视为文本的特征表示，该文本进行层归一化，然后线性投影到多模态嵌入空间中。虽然之前的计算机视觉研究通常对于单独增加宽度(`Mahajan`等人，`2018`)或深度来扩展模型，但对于`ResNet`图像编码器，采用了`Tan & Le (2019)`的方法，该方法发现在宽度、深度和分辨率上分配额外的计算要优于只将其分配给模型的一个维度。虽然`Tan & Le (2019)`调整了分配给其`EfficientNet`架构的每个维度的计算比例，但平等地分配额外的计算以增加模型的宽度、深度和分辨率。对于文本编码器，将模型的宽度缩放为与`ResNet`宽度的计算增加成比例，并且根本不缩放深度，发现`CLIP`的性能对文本编码器的容量不太敏感。
