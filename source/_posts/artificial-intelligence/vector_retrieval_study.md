---
title: 向量检索（PyTorch）
date: 2024-02-26 11:14:32
tags:
  - AI
categories:
  - 人工智能
---

#### 向量检索

举个例子，想象一下，每一条数据（比如一张图片、一段文字和一首歌曲）都被转化成了一个高维空间里的一个小箭头（我们这称之为向量）。这些向量根据他们所携带的信息分布在不同的位置。向量检索就像是给定一个向量，帮你在这个多维迷宫中解开这个相似向量（比如一张图片、一段文字或一首歌）。向量检索的应用场景：
- 图搜商品：在购物平台上，只需要上传照片，系统就能通过向量检索到几乎相同的商品。
- 语音识别后处理：将用户的语音转换为特征向量，然后在数据库中找到最匹配的语句或命令。
- 推荐系统：用户的历史行为被编码成向量，通过检索找出最相关的电影、音乐或新闻推荐给你。
- 生物信息学：基因序列转化为向量，用于寻找相似基因片段或预测蛋白质功能。
<!-- more -->

```python
import torch
import torch.nn.functional as F

# 假设我们有两个向量a和b
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 计算两个向量的余弦相似度
consine_similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)

print(f'The cosine similarity between a and b is: {consine_similarity.item()}')

# 假设我们有一个向量集合，我们想找到与向量a最相似的向量
vector_collection = torch.tensor([[7.0, 8.0, 9.0], [1.5, 2.5, 3.5], [0.5, 1.5, 2.5]])

# 计算向量a与向量集合中每个向量的余璇相似度
similarities = F.cosine_similarity(a.unsqueeze(0), vector_collection, dim=1)

# 找到最相似的向量
most_similar_vector_idex = torch.argmax(similarities).item()
most_similar_vector = vector_collection[most_similar_vector_idex]

print(f'The most similar vector to a is: {most_similar_vector}')
```
输出结果为：
```bash
The most similar vector to a is: tensor([1.5000, 2.5000, 3.5000])
```