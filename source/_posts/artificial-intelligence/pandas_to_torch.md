---
title: 有效地处理从pandas到Pytorch的表格数据集中的数千个特征（PyTorch）
date: 2024-02-27 21:20:32
tags:
  - AI
categories:
  - 人工智能
---

在实践中，传统的表格数据的形状为`(batch_size, feat_1, feat_2,…feat_N)`，其中`N`是任意大的。当有数千个特征（例如，`N>1000`）时，很难知道`Pytorch`张量中的哪些列索引对应于哪个特征。以及如何为神经网络架构应用适当的`Pytorch`模块。
<!-- more -->

- 数据科学家通常使用`pandas DataFrame进`行必要的数据探索、数据处理和特征工程，然后将其转换为 `Pytorch`张量以构建`Pytorch`数据集。使用`pandas`的一些主要好处是它有一个简单的`API`。
- Pytorch模型要求输入数据类型为`torch.Tensor`。但是，当我们将`pandas DataFrame`转换为`Pytorch`张量后，我们失去了能够轻松查找数据集中特征的能力。
- 如果我们想要得到`feature`对应的数据`gender`，使用`pandas`我们会这样做df['gender']，但是对于`Pytorch Tensor`，我们必须计算列数来找到列索引：`X[:,3]`。
- 如果我们有数千个特征，而性别只是其中之一，我们可以清楚地看到它是无法做到的。

学习和实践存在差距：
- 网络资源以简单易懂的方式解释概念，对实际应用的重视不够。
- 专注于深度学习的学习资源往往聚焦于自然语言处理和计算机视觉。

结论：学习高级实践或技术的唯一方法是深入研究在线社区共享的原始`Github`代码或学习行业经验（由导师指导）。

#### 初学者处理从Pandas到PyTorch的数据

通常从在线学习资源中看到的内容，在将`pandas DataFrame`转换为`Pytorch`张量之后，他们下一步会告诉你是创建一个`torch.utils.data.Dataset`用于批量梯度下降，也是为了管理内存。最常见的方法是`torch.utils.data.Dataset`独立输出每个特征，然后我们用它来创建模型：

Torch Dataset：
```python
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class Datasets(Dataset):
    def __init__(self, x, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, idx):
        user_id = self.X[idx, 0]
        movie_id = self.X[idx, 1]
        genres = self.X[idx, 1]
        gender = self.X[idx, 1]
        age = self.X[idx, 1]
        occupation = self.X[idx, 1]
        zip_code = self.X[idx, 1]
        label = self.y[idx]
        return (user_id, movie_id, genres, gender, age, occupation, zip_code, label)
    
    def __len__(self):
        return len(self.X)

X = torch.tensor(df.iloc[:,:-1].values)
y = torch.tensor(df.iloc[:,-1].values)
train_dataset = Datasets(X,y)

BATCH_SIZE = 1
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

next(iter(train_dataloader))
```
Torch Model：
```python
class Model(nn.Module):
    def __init__(self, emb_dim=16):
        super(Model, self).__init__()
        
        # sparse embedings
        self.user_emb = nn.Embedding(235, emb_dim)
        self.gender_emb = nn.Embedding(2, emb_dim)
        self.occupation_emb = nn.Embedding(111, emb_dim)
        self.zip_code_emb = nn.Embedding(526, emb_dim)
        self.movie_emb = nn.Embedding(21, emb_dim)
        self.genres_emb = nn.Embedding(50, emb_dim)
        
    def forward(self, user_id, movie_id, genres, gender, age, occupation, zip_code, label=None):
        
        # user embedding
        
        user_e = self.user_emb(user_id)
        gender_e = self.gender_emb(gender)
        occupation_e = self.occupation_emb(occupation)
        zip_code_e = self.zip_code_emb(zip_code)
        movie_id_e = self.movie_emb(movie_id)
        genres_e = self.genres_emb(genres)
        
        output = torch.cat([user_e, gender_e, occupation_e, zip_code_e, movie_id_e, genres_e, age], dim=-1)
        
        return output
```
- 我们目前只有`7`个特征，代码看起来很长，而且要写很多重复的代码。试想一下，如果我们有`1000`个特征。我们是否要编写`1000`行代码只是为了获取`torch`数据集中的特征，并且我们是否要对 `1000`个参数使用`1000`个实参？
- 我们手动初始化`nn.Embedding`每个分类特征的`a`，同时手动输入词汇量大小。如果我们有`1000`个分类特征，我们要写`1000`行吗?
- 我们通常将这些输入特征连接成一个张量作为线性层的输入。我们是否要手动连接`1000`个特征，从而有效地编写另外`1000`行重复代码？

#### 经验丰富的数据科学家处理从Pandas到PyTorch的数据

他们观察到：
- 获取与自己想要的特征对应的`torch`张量是一个烦人的过程。
- 创建嵌入是一个重复的过程
- 连接他们想要的特征是一个烦人的过程。

他们将利用`Python`数据结构来帮助他们有效地管理代码。

##### 1.创建一个字典来存储词汇总数

##### 2.创建字典来存储嵌入维度

##### 3.创建一个类来存储分类特征元数据

##### 4.将分类列的列表存储为SparseFeat

##### 5.同样，创建一个类来存储数据集的数值特征，并将数值特征列表存储为DenseFeat

##### 6.创建与分类或数值特征对应的 Pytorch 张量的开始和结束索引

##### 7.基于pandasDataFrame构建Pytorch张量feature_columns

##### 8.创建一个函数来查找分类嵌入

##### 9.同样，创建一个函数来查找数值特征

#### 总结

更有经验的数据科学家将利用`Python`数据结构来使他/她的工作变得更轻松，当使用`Pytorch`处理数千个特征时，这些技能在现实世界中极其重要。
{% asset_img pandas_torch.png %}

