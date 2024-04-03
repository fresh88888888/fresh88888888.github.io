---
title: NLP（Transformers & BERT）
date: 2024-04-02 11:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 内容

在本文中，我将从`RNN`的基础知识开始，一直到构建最新的深度学习架构来解决`NLP`问题。它将涵盖以下内容：

- 简单的`RNN`(循环神经网络)。
- 词嵌入(`Word Embeddings`)：定义以及如何获取。
- 长短期记忆网络(`LSTM`)。
- 门控循环单元(`GRU`)。
- 双向`RNN`。
- 编码器-解码器模型（`Seq2Seq`模型）。
- 注意力模型(`Attention Models`)。
- `Transformer`-你所需要的就是注意力。
- `BERT`。
<!-- more -->

我将每个主题分为四个小节：
- 基本概况。
- 深入理解。
- 代码实现。
- 代码说明。

#### 配置TPU

我们将使用TPU，因为我们需要构建`BERT`模型。
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU,SimpleRNN
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)

train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
```
我们将删除其他列并将此问题作为二元分类问题来处理，并且我们将在数据集的较小部分（仅`12000`个数据点）上完成练习，以便更轻松地训练模型。
```python
train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)
train = train.loc[:12000,:]

# 我们将检查评论中出现的最大字数，这将有助于我们稍后进行填充
train['comment_text'].apply(lambda x:len(str(x).split())).max()

# 编写一个函数来获取 auc 分数以进行验证
def roc_auc(predictions,target):
    '''
    This methods returns the AUC Score when given the Predictions
    and Labels
    '''
    
    fpr, tpr, thresholds = metrics.roc_curve(target, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

# 数据准备（Data Preparation）
xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values, 
                                                  stratify=train.toxic.values, 
                                                  random_state=42, 
                                                  test_size=0.2, shuffle=True)
```
#### 简单循环神经网络（RNN）

##### 基本概述

什么是`RNN`？**循环神经网络**（`RNN`）是一种神经网络，其中上一步的输出作为当前步骤的输入。在传统的神经网络中，所有的输入和输出都是相互独立的，但是当需要预测句子的下一个单词时，需要前面的单词，因此需要记住前面的单词。于是`RNN`应运而生，它借助**隐藏层**解决了这个问题。

##### 深入理解
根据维基百科，循环神经网络(`RNN`)是一类人工神经网络，其中单元之间的连接沿着序列形成有向图。这使得它能够表现出时间序列的动态时间行为。与前馈神经网络不同，`RNN`可以使用其内部状态（内存）来处理输入序列。这使得它们适用于诸如未分段、连接的手写识别或语音识别等任务。让我们通过一个类比来理解这一点。假设你正在看一部电影，你始终都在看这部电影，你有上下文，因为你已经看过这部电影直到那一点，然后只有你能够正确地将所有内容联系起来。意味着您记住了您看过的所有内容。同样，`RNN`会记住一切。在其他神经网络中，所有输入都是相互独立的。但在`RNN`中，所有输入都是相互关联的。假设您必须预测给定句子中的下一个单词，在这种情况下，所有先前单词之间的关系有助于预测更好的输出。`RNN`在训练自身时会记住所有这些关系。为了实现这一目标，`RNN`创建了带有循环的网络，这使得它能够保存信息。
{% asset_img ntb_1.png %}

这种循环结构允许神经网络获取输入序列。如果你看到展开的版本，你会更好地理解它。[循环神经网络详解(`Recurrent Neural Networks`)](https://www.d2l.ai/chapter_recurrent-neural-networks/rnn.html)

##### 代码实现

```python
# using keras tokenizer here
token = text.Tokenizer(num_words=None)
max_len = 1500

token.fit_on_texts(list(xtrain) + list(xvalid))
xtrain_seq = token.texts_to_sequences(xtrain)
xvalid_seq = token.texts_to_sequences(xvalid)

#zero pad the sequences
xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

word_index = token.word_index

with strategy.scope():
    # A simpleRNN without any pretrained embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     input_length=max_len))
    model.add(SimpleRNN(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
model.summary()

```
结果输出为：
```bash
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 1500, 300)         13049100  
_________________________________________________________________
simple_rnn_1 (SimpleRNN)     (None, 100)               40100     
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 101       
=================================================================
Total params: 13,089,301
Trainable params: 13,089,301
Non-trainable params: 0
_________________________________________________________________
CPU times: user 620 ms, sys: 370 ms, total: 990 ms
Wall time: 1.18 s
```
```python
model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync) #Multiplying by Strategy to run on TPU's
scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
scores_model = []
scores_model.append({'Model': 'SimpleRNN','AUC_Score': roc_auc(scores,yvalid)})
```
结果输出为：
```bash
Epoch 1/5
9600/9600 [==============================] - 39s 4ms/step - loss: 0.3714 - accuracy: 0.8805
Epoch 2/5
9600/9600 [==============================] - 39s 4ms/step - loss: 0.2858 - accuracy: 0.9055
Epoch 3/5
9600/9600 [==============================] - 40s 4ms/step - loss: 0.2748 - accuracy: 0.8945
Epoch 4/5
9600/9600 [==============================] - 40s 4ms/step - loss: 0.2416 - accuracy: 0.9053
Epoch 5/5
9600/9600 [==============================] - 39s 4ms/step - loss: 0.2109 - accuracy: 0.9079

Auc: 0.69%
```
##### 代码说明

我们将每个单词表示为一个维度的热向量：`Vocab`中的单词数`+1`。`keras Tokenizer`的作用是，获取语料库中所有唯一的单词，以单词为键、出现次数为值形成一个字典，然后按计数降序对字典进行排序。然后分配第一个值`1`，第二个值`2`，依此类推。假设单词“`the`”在语料库中出现次数最多，那么它将分配索引`1`，表示“`the`”的向量将是一个热向量，位置`1`处值为`1`，其余为零。打印`xtrain_seq`的前`2`个元素，您将看到每个单词现在都表示为数字。
```python
xtrain_seq[:1]

# [[664,65,7,19,2262,14102,5,2262,20439,6071,4,71,32,20440,6620,39,6,664,65,11,8,20441,1502,38,6072]]
```
###### 构建神经网络

第一行`model.Sequential()`告诉`keras`将按顺序构建网络。然后我们首先添加`Embedding`层。**嵌入层**也是一层神经元，它将每个单词的`n`维一个热向量作为输入，并将其转换为`300`维向量，它为我们提供了类似于`word2vec`的单词嵌入。我们可以使用`word2vec`，但嵌入层在训练过程中进行学习以增强嵌入。接下来我们添加`100`个`LSTM`单元，没有任何`dropout`或正则化，最后我们添加一个具有`sigmoid`函数的神经元，该神经元从`100`个`LSTM`单元（请注意，我们有`100`个`LSTM`单元而不是层）获取输出来预测结果，然后我们编译模型，使用`Adam`优化器。

###### 对模型的评估

我们可以看到模型达到了`1`的准确率，我知道显然**过度拟合**，但这是所有模型中最简单的，我们可以调整很多超参数，例如`RNN`单元，我们可以进行批量归一化、`dropout`等 以获得更好的结果。关键是我们不费吹灰之力就得到了`0.82`的`AUC`分数，并且我们已经了解了`RNN`。

#### 词嵌入(`Word Embeddings`)

在构建简单的`RNN`模型时，我们使用了词嵌入，那么什么是词嵌入，以及我们如何获得词嵌入？**词嵌入是一种学习到的文本表示，其中具有相同含义的单词具有相似的表示**。这种表示单词和文档的方法可能被认为是深度学习在挑战自然语言处理问题方面的关键突破之一。使用密集和低维向量的好处之一是**计算**：大多数神经网络工具包不能很好地处理非常高维的稀疏向量。**密集表示**的主要好处是**泛化能力**，如果我们相信某些特征可能提供相似的线索，那么提供能够捕获这些相似性的表示是值得的。获取词嵌入的最新方法是使用预保留的`GLoVe`或`Fasttext`。
```python
# load the GloVe vectors in a dictionary:

embeddings_index = {}
f = open('/kaggle/input/glove840b300dtxt/glove.840B.300d.txt','r',encoding='utf-8')
for line in tqdm(f):
    values = line.split(' ')
    word = values[0]
    coefs = np.asarray([float(val) for val in values[1:]])
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
```
#### 长短期记忆网络(LSTM)

##### 基本概述

简单的`RNN`比经典的`ML`算法更好，并且给出了最领先的结果，但它无法捕获句子中存在的长期依赖关系。因此在`1998-99`年引入了`LSTM`来克服这些缺点。

##### 深度理解

长期以来，隐变量模型存在着**长期信息保存**和**短期输入缺失**的问题。解决这一问题的最早方法之一是**长短期存储器**（`long short-term memory`，`LSTM`）(`Hochreiter and Schmidhuber, 1997`)。它有许多与**门控循环单元**一样的属性。有趣的是，**长短期记忆网络**的设计比门控循环单元稍微复杂一些，却比门控循环单元早诞生了近20年。

###### 门控记忆元

可以说，**长短期记忆网络**的设计灵感来自于计算机的逻辑门。长短期记忆网络引入了记忆元（`memory cell`），或简称为单元（`cell`）。有些文献认为记忆元是**隐状态**的一种特殊类型，它们与隐状态具有相同的形状，其设计目的是用于记录附加的信息。为了控制记忆元，我们需要许多门。其中一个门用来从单元中输出条目，我们将其称为**输出门**（`output gate`）。另外一个门用来决定何时将数据读入单元，我们将其称为**输入门**（`input gate`）。我们还需要一种机制来重置单元的内容，由**遗忘门**（`forget gate`）来管理，这种设计的动机与门控循环单元相同，能够通过专用机制决定**什么时候记忆或忽略隐状态中的输入**。 让我们看看这在实践中是如何运作的。
{% asset_img ntb_2.png %}

##### 代码实现

```python
# create an embedding matrix for the words we have in the dataset
embedding_matrix = np.zeros((len(word_index) + 1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

with strategy.scope():
    
    # A simple LSTM with glove embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,300,weights=[embedding_matrix],input_length=max_len,trainable=False))

    model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
model.summary()
model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync)

scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))

scores_model.append({'Model': 'LSTM','AUC_Score': roc_auc(scores,yvalid)})
```
结果输出为：
```bash
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_2 (Embedding)      (None, 1500, 300)         13049100  
_________________________________________________________________
lstm_1 (LSTM)                (None, 100)               160400    
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 101       
=================================================================
Total params: 13,209,601
Trainable params: 160,501
Non-trainable params: 13,049,100
_________________________________________________________________
CPU times: user 1.33 s, sys: 1.46 s, total: 2.79 s
Wall time: 3.09 s

Epoch 1/5
9600/9600 [==============================] - 117s 12ms/step - loss: 0.3525 - accuracy: 0.8852
Epoch 2/5
9600/9600 [==============================] - 114s 12ms/step - loss: 0.2397 - accuracy: 0.9192
Epoch 3/5
9600/9600 [==============================] - 114s 12ms/step - loss: 0.1904 - accuracy: 0.9333
Epoch 4/5
9600/9600 [==============================] - 114s 12ms/step - loss: 0.1659 - accuracy: 0.9394
Epoch 5/5
9600/9600 [==============================] - 114s 12ms/step - loss: 0.1553 - accuracy: 0.9470

Auc: 0.96%
```
##### 代码说明

###### 创建模型

第一步，我们根据预训练的`GLoVe`向量计算词汇表的**嵌入矩阵**。然后，在构建嵌入层时，我们将嵌入矩阵作为权重传递给该层，而不是通过词汇对其进行训练，因此我们传递`trainable = False`。模型的其余部分与之前相同，只是我们用`LSTM`单元替换了`SimpleRNN`。

###### 对模型评估

现在我们看到该模型没有过度拟合，并且达到了`0.96`的`auc`分数，这是非常值得称赞的，而且我们也缩小了准确率和`auc`之间的差距。我们看到，在这种情况下，我们使用了`dropout`来防止数据过度拟合。

#### 门控循环单元（GRU）

##### 基本概述

`2014`年，**门控循环单元**（`GRU`）旨在解决标准循环神经网络带来的**梯度消失**问题。`GRU`是`LSTM`的变体，因为两者设计相似，并且在某些情况下产生同样出色的结果。`GRU`的设计比`LSTM`更简单、更快，并且在大多数情况下产生同样好的结果。

##### 深入解释

门控循环单元与普通的循环神经网络之间的关键区别在于：前者支持**隐状态的门控**。这意味着模型有专门的机制来确定应该何时更新**隐状态**，以及应该何时重置隐状态。这些机制是可学习的，并且能够解决了上面列出的问题。例如，如果第一个**词元**非常重要，模型将学会在第一次观测之后不更新隐状态。同样，模型也可以学会跳过不相关的临时观测。最后，模型还将学会在需要的时候重置隐状态。

##### 代码实现

```python
with strategy.scope():
    # GRU with glove embeddings and two dense layers
     model = Sequential()
     model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
     model.add(SpatialDropout1D(0.3))
     model.add(GRU(300))
     model.add(Dense(1, activation='sigmoid'))

     model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])   
    
model.summary()

model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync)

scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))
scores_model.append({'Model': 'GRU','AUC_Score': roc_auc(scores,yvalid)})

```
结果输出为：
```bash
Model: "sequential_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_3 (Embedding)      (None, 1500, 300)         13049100  
_________________________________________________________________
spatial_dropout1d_1 (Spatial (None, 1500, 300)         0         
_________________________________________________________________
gru_1 (GRU)                  (None, 300)               540900    
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 301       
=================================================================
Total params: 13,590,301
Trainable params: 541,201
Non-trainable params: 13,049,100
_________________________________________________________________
CPU times: user 1.3 s, sys: 1.29 s, total: 2.59 s
Wall time: 2.79 s

Epoch 1/5
9600/9600 [==============================] - 191s 20ms/step - loss: 0.3272 - accuracy: 0.8933
Epoch 2/5
9600/9600 [==============================] - 189s 20ms/step - loss: 0.2015 - accuracy: 0.9334
Epoch 3/5
9600/9600 [==============================] - 189s 20ms/step - loss: 0.1540 - accuracy: 0.9483
Epoch 4/5
9600/9600 [==============================] - 189s 20ms/step - loss: 0.1287 - accuracy: 0.9548
Epoch 5/5
9600/9600 [==============================] - 188s 20ms/step - loss: 0.1238 - accuracy: 0.9551

Auc: 0.97%

[{'Model': 'SimpleRNN', 'AUC_Score': 0.6949714081921305},
 {'Model': 'LSTM', 'AUC_Score': 0.9598235453841757},
 {'Model': 'GRU', 'AUC_Score': 0.9716554069114769}]
```

#### 双向RNN

##### 深度理解

如果我们希望在循环神经网络中拥有一种机制，使之能够提供与**隐马尔可夫模型**类似的前瞻能力，我们就需要修改循环神经网络的设计。幸运的是，这在概念上很容易，只需要增加一个“**从最后一个词元开始从后向前运行**”的循环神经网络，而不是只有一个在前向模式下“**从第一个词元开始运行**”的循环神经网络。**双向循环神经网络**（`bidirectional RNNs`）添加了反向传递信息的隐藏层，以便更灵活地处理此类信息。下图描述了具有单个隐藏层的双向循环神经网络的架构。
{% asset_img ntb_3.png %}

事实上，这与隐马尔可夫模型中的动态规划的前向和后向递归没有太大区别。 其主要区别是，在隐马尔可夫模型中的方程具有特定的统计意义。 双向循环神经网络没有这样容易理解的解释， 我们只能把它们当作通用的、可学习的函数。 这种转变集中体现了现代深度网络的设计原则： 首先使用经典统计模型的函数依赖类型，然后将其参数化为通用形式。[双向循环神经网络详解(`Bi-Directional RNN`)](https://www.d2l.ai/chapter_recurrent-neural-networks/rnn.html)

##### 代码实现

```python
# 创建Bi-Directional RNN
with strategy.scope():
    # A simple bidirectional LSTM with glove embeddings and one dense layer
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=max_len,
                     trainable=False))
    model.add(Bidirectional(LSTM(300, dropout=0.3, recurrent_dropout=0.3)))

    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    
model.summary()

model.fit(xtrain_pad, ytrain, nb_epoch=5, batch_size=64*strategy.num_replicas_in_sync)

scores = model.predict(xvalid_pad)
print("Auc: %.2f%%" % (roc_auc(scores,yvalid)))

scores_model.append({'Model': 'Bi-directional LSTM','AUC_Score': roc_auc(scores,yvalid)})
```
结果输出为：
```bash
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_4 (Embedding)      (None, 1500, 300)         13049100  
_________________________________________________________________
bidirectional_1 (Bidirection (None, 600)               1442400   
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 601       
=================================================================
Total params: 14,492,101
Trainable params: 1,443,001
Non-trainable params: 13,049,100
_________________________________________________________________
CPU times: user 2.39 s, sys: 1.62 s, total: 4 s
Wall time: 3.41 s

Epoch 1/5
9600/9600 [==============================] - 322s 34ms/step - loss: 0.3171 - accuracy: 0.9009
Epoch 2/5
9600/9600 [==============================] - 318s 33ms/step - loss: 0.1988 - accuracy: 0.9305
Epoch 3/5
9600/9600 [==============================] - 318s 33ms/step - loss: 0.1650 - accuracy: 0.9424
Epoch 4/5
9600/9600 [==============================] - 318s 33ms/step - loss: 0.1577 - accuracy: 0.9414
Epoch 5/5
9600/9600 [==============================] - 319s 33ms/step - loss: 0.1540 - accuracy: 0.9459

Auc: 0.97%
```
##### 代码说明

代码与以前相同，只是我们为之前使用的`LSTM`单元添加了双向性质。我们已经实现了与之前类似的准确率和`auc`分数，现在我们已经学习了所有类型的典型`RNN`架构。

#### Seq2Seq模型架构

##### 基本概述

`RNN`有多种类型，不同的架构用于不同的目的。这里有一个视频，解释了不同类型的[模型架构](https://www.coursera.org/learn/nlp-sequence-models/lecture/BO8PS/ different-types-of-rnns)。`Seq2Seq`是一个多对多的`RNN`架构，其中输入是一个序列，输出也是一个序列。该架构用于许多应用，如机器翻译、文本摘要、问答等。

##### 深入理解

在一般的**序列到序列**（`Seq2Seq`）问题中，输入和输出的长度不同且未对齐。处理此类数据的标准方法是设计一个**编码器-解码器架构**，它由两个主要组件组成：**一个以可变长度序列作为输入的编码器，另一个充当条件的解码器**。语言模型，接收编码输入和目标序列的左侧上下文，并预测目标序列中的后续标记。
{% asset_img ntb_4.png %}

让我们以从英语到法语的机器翻译为例。给定一个英文输入序列：“`These`”、“`are`”、“`watching`”、“`.`”，这种编码器-解码器架构首先将可变长度输入编码为状态，然后解码该状态以生成翻译序列、标记，作为输出：“`Ils`”、“`regardent`”、“`.`”。**编码器-解码器架构**可以处理由可变长度序列组成的输入和输出，因此适用于序列到序列的问题，例如机器翻译。编码器将可变长度序列作为输入，并将其转换为具有固定形状的状态。解码器将固定形状的编码状态映射到可变长度序列。
```python
# Visualization of Results obtained from various Deep learning models
results = pd.DataFrame(scores_model).sort_values(by='AUC_Score',ascending=False)
results.style.background_gradient(cmap='Blues')

fig = go.Figure(go.Funnelarea(
    text =results.Model,values = results.AUC_Score,title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()
```
{% asset_img ntb_5.png %}
{% asset_img ntb_6.png %}

#### 注意力模型（Attention Models）

如果你能够理解注意力模块的工作原理，那么理解`Transformer`和基于`Transformer`架构（如`BERT`）将是小菜一碟。

#### Transformer - Attention is all you need

最后我们到达了学习曲线的终点，即将开始学习彻底改变`NLP`的技术，这也是最先进的`NLP`技术的原因。`Google`在论文《`Attention is all you need`》中介绍了`Transformer`。`Transformer`是由一个编码器、解码器组件以及他们之间的连接构成。编码组件有一堆编码器组成，解码器组件也是由相同数量的解码器组成。编码器分为两层：**自注意力层、前馈神经网络**。编码器与编码器的结构相同，但彼此不共享权重。编码器的输入首先流入自注意力层，该层帮助编码器在对特定单词进行编码时查看输入句子中的其他单词，自注意力层的输出被馈送到前馈神经网络，完全相同的前馈网络是相互独立的。解码器分为三层：**自注意力层、`Encoder-Decoder`注意力层、前馈神经网络**。`Encoder-Decoder`注意力层的作用是帮助解码器专注于输入的相关部分（类似于`seq2seq`模型中注意力的作用）。
{% asset_img ntb_7.png %}

##### 代码实现

###### 模型架构

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")

class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)
```
{% asset_img ntb_8.png [Transformer 架构] %}

###### 编码器&解码器栈

* 编码器
编码器由6个独立相同的层组成的堆栈。
```python
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

# 两个子层周围 采用残差连接, 然后进行层归一化。
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

# 为了促进这些残差连接，模型中的所有子层以及嵌入层都会产生维度的输出。
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

# 每层有两个子层。第一个是多头自注意力机制，第二个是全连接前馈网络。
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
```
* 解码器
解码器由6个独立相同的层组成的堆栈。
```python
class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

# 除了每个编码器层中的两个子层之外，解码器还插入第三个子层，该子层对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围采用残差连接，然后进行层归一化。
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# 我们还修改了解码器堆栈中的自注意力子层，以防止位置关注后续位置。这种掩蔽与输出嵌入偏移一个位置的事实相结合，确保了位置的预测只能依赖于小于位置的已知输出。
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
```
* 注意力
```python
# 注意力函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。输出被计算为值的加权和，其中分配给每个值的权重是由查询与相应键的兼容性函数计算的。
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```
两种最常用的注意力函数是加性注意力和点积注意力。附加注意力使用具有单个隐藏层的前馈网络来计算兼容性函数。虽然两者在理论复杂性上相似，但点积注意力在实践中更快、更节省空间，因为它可以使用高度优化的矩阵乘法代码来实现。多头注意力允许模型共同关注来自不同位置的不同表示子空间的信息。对于单一注意力头，平均会抑制这种情况。

`Transformer`以三种不同的方式使用多头注意力：
- 在“编码器-解码器注意力”层中，查询来自前一个解码器层，内存键和值来自编码器的输出。这允许解码器中的每个位置都参与输入序列中的所有位置。这模仿了序列到序列模型中典型的编码器-解码器注意机制。
- 编码器包含自注意力层。在自注意力层中，所有键、值和查询都来自同一位置，在本例中是编码器中前一层的输出。编码器中的每个位置可以关注编码器上一层中的所有位置。
- 解码器中的自注意力层允许解码器中的每个位置关注解码器中直到并包括该位置的所有位置。我们需要防止解码器中的左向信息流以保留自回归属性。我们通过屏蔽（设置为-无穷大）`softmax`输入中对应于非法连接的所有值。

* 位置前馈网络
除了注意力子层之外，我们的编码器和解码器中的每个层都包含一个完全连接的前馈网络，该网络单独且相同地应用于每个位置。这由两个线性变换组成，中间有一个`ReLU`激活。
```python
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
```
* Embeddings & Softmax
与其他序列转导模型类似，我们使用学习嵌入将输入标记和输出标记转换为维度向量。我们还使用通常学习的线性变换和`softmax`函数将解码器输出转换为预测的下一个令牌概率。在我们的模型中，我们在两个嵌入层和`pre-softmax`线性变换之间共享相同的权重矩阵。
```python
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
```
* 位置编码
由于我们的模型不包含递归和卷积，为了使模型能够利用序列的顺序，我们必须注入一些有关序列中标记的相对或绝对位置的信息。为此，我们将“位置编码”添加到编码器和解码器堆栈底部的输入嵌入中。位置编码具有相同的维度作为嵌入，以便将两者相加。位置编码有多种选择，有学习的和固定的。
```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

# 我们还尝试使用学习的位置嵌入来代替，发现这两个版本产生几乎相同的结果。我们选择正弦版本，因为它可以允许模型推断出比训练期间遇到的序列长度更长的序列长度。
```
* 完整模型
```python
def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), 
                             c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

# Small example model.
tmp_model = make_model(10, 10, 2)
```
#### BERT

[BERT详解](https://jalammar.github.io/illustrated-bert/)

我们将使用`Hugging Face`和`KERAS`实现`BERT`模型。涉及步骤：
- 数据准备：数据的标记化和编码。
- 配置TPU。
- 构建一个函数用于模型训练并且添加分类输出层。
- 训练模型并得到结果。

```python
# Loading Dependencies
import os
import transformers
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from kaggle_datasets import KaggleDatasets
from tokenizers import BertWordPieceTokenizer

# LOADING THE DATA
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=512):
    """
    Encoder for encoding the text into sequence of integers for BERT Input
    """
    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding(max_length=maxlen)
    all_ids = []
    
    for i in tqdm(range(0, len(texts), chunk_size)):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

#IMP DATA FOR CONFIG
AUTO = tf.data.experimental.AUTOTUNE
# Configuration
EPOCHS = 3
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
MAX_LEN = 192

# Tokenization
# First load the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
# Save the loaded tokenizer locally
tokenizer.save_pretrained('.')
# Reload it with the huggingface tokenizers library
fast_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=False)
fast_tokenizer

x_train = fast_encode(train1.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_valid = fast_encode(valid.comment_text.astype(str), fast_tokenizer, maxlen=MAX_LEN)
x_test = fast_encode(test.content.astype(str), fast_tokenizer, maxlen=MAX_LEN)

y_train = train1.toxic.values
y_valid = valid.toxic.values

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_train, y_train))
    .repeat()
    .shuffle(2048)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((x_valid, y_valid))
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(x_test)
    .batch(BATCH_SIZE)
)

def build_model(transformer, max_len=512):
    """
    function for training the BERT model
    """
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# 开始训练
with strategy.scope():
    transformer_layer = (
        transformers.TFDistilBertModel
        .from_pretrained('distilbert-base-multilingual-cased')
    )
    model = build_model(transformer_layer, max_len=MAX_LEN)
model.summary()

n_steps = x_train.shape[0] // BATCH_SIZE
train_history = model.fit(
    train_dataset,
    steps_per_epoch=n_steps,
    validation_data=valid_dataset,
    epochs=EPOCHS
)

sub['toxic'] = model.predict(test_dataset, verbose=1)
sub.to_csv('submission.csv', index=False)

```
结果输出为：
```bash
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_word_ids (InputLayer)  [(None, 192)]             0         
_________________________________________________________________
tf_distil_bert_model (TFDist ((None, 192, 768),)       134734080 
_________________________________________________________________
tf_op_layer_strided_slice (T [(None, 768)]             0         
_________________________________________________________________
dense (Dense)                (None, 1)                 769       
=================================================================
Total params: 134,734,849
Trainable params: 134,734,849
Non-trainable params: 0
_________________________________________________________________
CPU times: user 34.4 s, sys: 13.3 s, total: 47.7 s
Wall time: 50.8 s

Train for 1746 steps, validate for 63 steps
Epoch 1/3
1746/1746 [==============================] - 255s 146ms/step - loss: 0.1221 - accuracy: 0.9517 - val_loss: 0.4484 - val_accuracy: 0.8479
Epoch 2/3
1746/1746 [==============================] - 198s 114ms/step - loss: 0.0908 - accuracy: 0.9634 - val_loss: 0.4769 - val_accuracy: 0.8491
Epoch 3/3
1746/1746 [==============================] - 198s 113ms/step - loss: 0.0775 - accuracy: 0.9680 - val_loss: 0.5522 - val_accuracy: 0.8500
```
