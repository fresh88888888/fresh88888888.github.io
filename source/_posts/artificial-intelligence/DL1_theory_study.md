---
title: 深度学习(DL)(一) — 探析
date: 2024-09-02 17:15:11
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

#### 递归神经网络

在语音识别中，您将获得一个输入音频片段{% mathjax %}x{% endmathjax %}，并被要求将其映射到文本转录{% mathjax %}y{% endmathjax %}。这里的输入和输出都是**序列数据**，因为{% mathjax %}x{% endmathjax %}是一个音频片段，因此它会随时间播放而输出{% mathjax %}y{% endmathjax %}，{% mathjax %}y{% endmathjax %}是一个单词序列。音乐生成是序列数据问题的另一个例子。在这种情况下，只有输出{% mathjax %}y{% endmathjax %}是一个序列，输入可以是空集，也可以是一个整数，可能指的是您想要生成的音乐类型，也可能是您想要的音乐的前几个音符。但这里的{% mathjax %}x{% endmathjax %}可以是零，也可以只是一个整数，而输出{% mathjax %}y{% endmathjax %}是一个序列。
<!-- more -->

在情绪分类中，输入{% mathjax %}x{% endmathjax %}是一个序列，因此，给定输入短语，如“这部电影没有什么可喜欢的”，您认为这篇评论会得到多少颗星？**序列模型**对于`DNA`序列分析也非常有用。`DNA`通过四个字母`A、C、G`和`T`表示。因此，给定一个`DNA`序列，您能否标记该`DNA`序列的哪一部分对应于蛋白质。在机器翻译中，您会得到一个输入句子，`voulez-vou chante avec moi？`然后要求以不同的语言输出翻译。<span style="color:#295F98;font-weight:900;">在视频活动识别中</span>，您可能会得到一系列视频帧并被要求识别活动。在名称实体识别中，您可能会得到一个句子并被要求识别该句子中的人。因此，所有这些问题都可以作为**监督学习**来解决，标签数据{% mathjax %}x,y{% endmathjax %}作为训练集。但是，从这个示例列表中可以看出，存在许多不同类型的序列问题。在某些情况下，输入{% mathjax %}x{% endmathjax %}和输出{% mathjax %}y{% endmathjax %}都是序列，有时{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}可以具有不同的长度，或者{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}具有相同的长度。或者只有{% mathjax %}x{% endmathjax %}或相反的{% mathjax %}y{% endmathjax %}是序列。

