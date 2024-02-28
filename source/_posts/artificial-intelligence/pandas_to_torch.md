---
title: 有效地处理从pandas到Pytorch的表格数据集中的数千个特征
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

`Torch Dataset`：
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
`Torch Model`：
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

##### 1.创建一个字典来存储所有词汇

- 当初始化分类特征的`nn.Embedding`时，需要所有词汇量大小。
- 为了不手动跟踪每个分类特征的词汇大小，我们创建一个包含每个分类列的词汇大小的字典。

```python
def get_vocabularies(df: pd.DataFrame, categorical_columns: list):
    vocab_size = {}
    for cat in categorical_columns:
        vocab_size[cat] = df[cat].max() + 1
    
    return vocab_size

categorical_features = ['uid', 'ugender', 'iid', 'igenre']
vocab_sizes = get_vocabularies(df, categorical_features)
vocab_sizes
# {'uid': 3, 'ugender': 2, 'iid': 4, 'igenre': 3}
```

##### 2.创建字典来存储嵌入维度

- 初始化`nn.Embedding`时，我们需要嵌入维度。
- 通常，每个分类特征的嵌入维度是相同的，因此我们可以对多个分类特征执行`Pytorch`操作。

```python
embedding_dim_dict = get_embedding_dim_dict(categorical_features, 6)
embedding_dim_dict
# {'uid': 6, 'ugender': 6, 'iid': 6, 'igenre': 6}
```

##### 3.创建一个类来存储分类特征元数据

- 现在我们有了词汇量大小字典和嵌入维度字典，我们创建一个数据类来存储分类特征的元数据。
- 我们用这个类来创建一个简单的“`API`”，让我们了解分类特征，获取词汇量大小、名称、`nn.Embedding()`的嵌入维度。
- 我们将此类称为`SparseFeat`，因为`nn.Embedding`本质上是一个查找表，就像`One-Hot`编码一样，它本质上是一个稀疏特征，因为`one-hot`编码的特征除了“`1`”就是“`0`”

```python
@dataclass
class SparseFeat:
    name: str
    vocabulary_size: int
    embedding_dim: int
    embedding_name: str = None
    dtype: str = torch.long
    
    def __post_init__(self):
        """Auto fill embedding_name"""
        if self.embedding_name is None:
            self.embedding_name = self.name

embedding_dim = 8
uid_sparse_feat = SparseFeat(name='uid', vocabulary_size=vocab_size['uid'], embedding_dim=embedding_dim)
# get vocabulary size for uid
uid_sparse_feat.vocabulary_size
# get embedding dim for uid
uid_sparse_feat.embedding_dim
```

##### 4.将分类列的列表存储为SparseFeat

- 如果我们有`1,000`个分类特征，那么我们只需要迭代整个列表即可。
```python
sparse_features = [SparseFeat(name=cat,
            vocabulary_size=vocab_sizes[cat],
            embedding_dim=embedding_dim_dict) for cat in categorical_features]

# [SparseFeat(name='uid', vocabulary_size=3, embedding_dim={'uid': 6, 'ugender': 6, 'iid': 6, 'igenre': 6}, embedding_name='uid', group_name='default_group', dtype=torch.int64),
#  SparseFeat(name='ugender', vocabulary_size=2, embedding_dim={'uid': 6, 'ugender': 6, 'iid': 6, 'igenre': 6}, embedding_name='ugender', group_name='default_group', dtype=torch.int64),
#  SparseFeat(name='iid', vocabulary_size=4, embedding_dim={'uid': 6, 'ugender': 6, 'iid': 6, 'igenre': 6}, embedding_name='iid', group_name='default_group', dtype=torch.int64),
#  SparseFeat(name='igenre', vocabulary_size=3, embedding_dim={'uid': 6, 'ugender': 6, 'iid': 6, 'igenre': 6}, embedding_name='igenre', group_name='default_group', dtype=torch.int64)]
```

##### 5.同样，创建一个类来存储数据集的数值特征，并将数值特征列表存储为DenseFeat

- 处理完分类元数据后，我们还创建一个类来存储数字特征的元数据。
- 这也是为了制作一个简单的“`API`”，让我们知道一个数字特征，它的名称是什么以及对应的维度（默认=`1`）
```python
dense_feat = DenseFeat(name='score', dimension=1)
dense_feat
# DenseFeat(name='score', dimension=1, dtype=torch.float32)

# create list of numerical features
numerical_features = ['score']
dense_features = [DenseFeat(name=col, dimension=1) for col in numerical_features]
```

##### 6.创建与分类或数值特征对应的 Pytorch 张量的开始和结束索引

- 请记住，将 `pandas DataFrame` 转换为 `Pytorch` 数据集后，我们失去了能够通过名称轻松查找功能的优势。
- 为了帮助我们解决这个问题，我们创建了一个函数来告诉我们每个特征，`Pytorch` 张量中的开始和结束索引是什么。
- 如果我们没有元数据类，我们必须继续引用`vocabulary_size`字典和`embedding_dim_dict`。起始索引是包含的，而结束索引是排除的，类似于张量切片。
```python
categorical_features = ['uid', 'ugender', 'iid', 'igenre']
sparse_features = [SparseFeat(name=cat,
                              vocabulary_size=vocab_sizes[cat],
                              embedding_dim=embedding_dim_dict[cat]) for cat in categorical_features]

numerical_features = ['score']
# create list of numerical features
dense_features = [DenseFeat(name=col, dimension=1)
                  for col in numerical_features]

feature_columns = sparse_features + dense_features


def build_input_features(feature_columns):

    features = OrderedDict()
    start = 0
    for feat in feature_columns:
        if isinstance(feat, DenseFeat):
            features[feat.name] = (start, start + feat.dimension)
            start += feat.dimension

        elif isinstance(feat, SparseFeat):
            features[feat.name] = (start, start + 1)
            start += 1

        else:
            raise TypeError('Invalid feature columns type, got', type(feat))
    return features


feature_positions = build_input_features(feature_columns)
feature_positions
# OrderedDict([('uid', (0, 1)),
#              ('ugender', (1, 2)),
#              ('iid', (2, 3)),
#              ('igenre', (3, 4)),
#              ('score', (4, 5))])
```

##### 7.基于pandasDataFrame构建Pytorch张量feature_columns

- 请注意上面代码中的 feature_columns =稀疏特征 + 密集特征。这本质上意味着分类特征位于列表的左侧，数字列位于列表的右侧。
- 在上面的 build_input_features 中，与分类或数值特征对应的 Pytorch 张量的开始和结束索引是根据 feature_columns 中的顺序创建的，这可能与 pd.DataFrame 中列的排列方式完全不同。
- 为了解决这个问题，我们根据 feature_columns 中特征的顺序创建 Pytorch 张量。这确保了 Pytorch 张量的开始和结束索引对应于我们通过 build_input_features 创建的 feature_positions。
```python
def build_torch_dataset(df: pd.DataFrame, feature_columns: List):
    """ Create a torch tensor from the pandas dataframe according to the order of the features in feature_columns
    Cannot just use torch.tensor(df.values) because for variable length columns, it contains a list.
    Args:
        df (pandas.DataFrame): dataframe containing the features
        feature_columns (List)
    Returns:
        (torch.Tensor): pytorch tensor from df according to the order of feature_columns
    """
    tensors = []
    df = df.copy()
    feature_length_names = []
    for feat in feature_columns:
        tensor = torch.tensor(df[feat.name].values, dtype=feat.dtype)
        tensors.append(tensor.reshape(-1, 1))
    return torch.concat(tensors, dim=1)


torch_df = build_torch_dataset(df, feature_columns)
torch_df
# tensor([[0.0000, 0.0000, 1.0000, 1.0000, 0.1000],
#         [1.0000, 1.0000, 2.0000, 2.0000, 0.2000],
#         [2.0000, 0.0000, 3.0000, 1.0000, 0.3000]])
```

##### 8.创建一个函数来查找分类嵌入

- 我们首先创建一个字典，其中 key 作为 feature_name，value 作为初始化的 nn.Embedding。
- 我们创建函数，以便我们只能获得所选分类特征的嵌入。
- 在下面的示例中，您可以看到获取 [‘uid’, ‘genre’] 的嵌入是多么容易，我们不必手动考虑‘uid’或‘genre’属于哪个位置索引。
- 如果我们有 1000 个特征，只需将分类特征列表传递给 return_feat_list。
```python
def build_embedding_dict(all_sparse_feature_columns, init_std=0.001):
    embedding_dict = nn.ModuleDict(
        {feat.name: nn.Embedding(feat.vocabulary_size,
                                 feat.embedding_dim) for feat in all_sparse_feature_columns})
    if init_std is not None:
        for tensor in embedding_dict.values():
            # nn.init is in_place
            nn.init.normal_(tensor.weight, mean=0, std=init_std)

    return embedding_dict


embedding_dict = build_embedding_dict(sparse_features)
embedding_dict
# ModuleDict(
#   (uid): Embedding(3, 6)
#   (ugender): Embedding(2, 6)
#   (iid): Embedding(4, 6)
#   (igenre): Embedding(3, 6)
# )


def embedding_lookup(X,
                     feature_positions,
                     embedding_dict,
                     sparse_feature_columns,
                     return_feat_list=()):

    embeddings_list = []
    for feat in sparse_feature_columns:
        feat_name = feat.name
        embedding_name = feat.embedding_name
        if feat_name in return_feat_list or len(return_feat_list) == 0:
            lookup_idx = feature_positions[feat_name]
            input_tensor = X[:, lookup_idx[0]:lookup_idx[1]].long()
            embedding = embedding_dict[embedding_name](input_tensor)
            embeddings_list.append(embedding)
    return embeddings_list


categorical_embeddings = embedding_lookup(torch_df,
                                          feature_positions,
                                          embedding_dict,
                                          sparse_features,
                                          return_feat_list=['uid', 'genre'])
categorical_embeddings
# [tensor([[[-9.1713e-04,  6.5061e-05, -8.2737e-04, -6.2794e-04,  3.2218e-04,
#            -9.5998e-04]],

#          [[-3.6192e-04, -7.2849e-04, -4.4335e-04,  5.4883e-04, -6.2344e-04,
#            -5.5105e-04]],

#          [[ 4.9634e-04,  2.3615e-04, -1.2853e-03, -2.9909e-04,  1.2274e-03,
#            -2.2752e-04]]], grad_fn=<EmbeddingBackward0>)]
```

##### 9.同样，创建一个函数来查找数值特征

- 在下面的示例中，您可以再次看到获取 [‘score’] 的张量是多么容易，我们不必手动考虑‘score’属于哪个位置索引。
```python
def dense_lookup(X, feature_positions, dense_features, return_feat_list=()):
    dense_list = []
    for feat in dense_features:
        feat_name = feat.name
        lookup_idx = feature_positions[feat_name]
        tensor = X[:, lookup_idx[0]:lookup_idx[1]]
        dense_list.append(tensor)
    return dense_list

dense_feats = dense_lookup(torch_df,
                           feature_positions,
                           dense_features,
                           return_feat_list=['score'])
dense_feats
# [tensor([[0.1000],
#          [0.2000],
#          [0.3000]])]
```
#### 总结

更有经验的数据科学家将利用`Python`数据结构来使他/她的工作变得更轻松，当使用`Pytorch`处理数千个特征时，这些技能在现实世界中极其重要。
{% asset_img pandas_torch.png %}

