---
title: NLP（GloVe & BERT & TF-IDF & LSTM）
date: 2024-04-03 14:09:11
tags:
  - AI
categories:
  - 人工智能
---

**自然语言处理**（`NLP`）是人工智能的一个分支，它负责连接机器以自然语言理解人类。自然语言可以是文本或声音的形式。`NLP`可以用人类的方式与机器进行交流。**文本分类**是情感分析中涉及的内容。它是将人类的意见或表达分类为不同的情绪。**情绪**包括正面、中立和负面、评论评级以及快乐、悲伤。 情绪分析可以针对不同的以消费者为中心的行业进行，分析人们对特定产品或主题的看法。自然语言处理起源于`20`世纪`50`年代。早在`1950`年，艾伦·图灵就发表了一篇题为《计算机器与智能》的文章，提出了图灵测试作为智能的标准，这项任务涉及自然语言的自动解释和生成，但当时尚未明确阐述。在此内核中，我们将重点关注文本分类和情感分析部分。
<!-- more -->

#### 加载数据

只需加载数据集和颜色等全局变量即可。
```python
import re
import string
import numpy as np 
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch
from collections import defaultdict
from collections import Counter
import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM, 
                          Embedding, 
                          BatchNormalization,
                          Dense, 
                          TimeDistributed, 
                          Dropout, 
                          Bidirectional,
                          Flatten, 
                          GlobalMaxPool1D)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    accuracy_score
)

# Defining all our palette colours.
primary_blue = "#496595"
primary_blue2 = "#85a1c1"
primary_blue3 = "#3f4d63"
primary_grey = "#c6ccd8"
primary_black = "#202022"
primary_bgcolor = "#f4f0ea"
primary_green = px.colors.qualitative.Plotly[2]

# 加载数据
df = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv", encoding="latin-1")
df = df.dropna(how="any", axis=1)
df.columns = ['target', 'message']
df.head()
```
输出结果为：
```bash
	target	message
0	ham	Go until jurong point, crazy.. Available only ...
1	ham	Ok lar... Joking wif u oni...
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...
3	ham	U dun say so early hor... U c already then say...
4	ham	Nah I don't think he goes to usf, he lives aro...
```
```python
df['message_len'] = df['message'].apply(lambda x: len(x.split(' ')))
df.head()
max(df['message_len'])
```
输出结果为：
```bash
	target	message	message_len
0	ham	Go until jurong point, crazy.. Available only ...	20
1	ham	Ok lar... Joking wif u oni...	6
2	spam	Free entry in 2 a wkly comp to win FA Cup fina...	28
3	ham	U dun say so early hor... U c already then say...	11
4	ham	Nah I don't think he goes to usf, he lives aro...	13

171
```
#### EDA 

现在我们来看看目标分布和消息长度。**平衡数据集**：- 让我们举一个简单的例子，如果在我们的数据集中我们有正值与负值大致相同。然后我们可以说我们的数据集处于**平衡状态**。将橙色视为正值，将蓝色视为负值。可以说正值和负值的数量大致相同。**不平衡数据集**：— 如果正值和负值之间存在很大差异。然后我们可以说我们的数据集是不平衡数据集。
```python
balance_counts = df.groupby('target')['target'].agg('count').values
balance_counts

# array([4825,  747])
```
```python
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['ham'],
    y=[balance_counts[0]],
    name='ham',
    text=[balance_counts[0]],
    textposition='auto',
    marker_color=primary_blue
))
fig.add_trace(go.Bar(
    x=['spam'],
    y=[balance_counts[1]],
    name='spam',
    text=[balance_counts[1]],
    textposition='auto',
    marker_color=primary_grey
))
fig.update_layout(title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>')
fig.show()
```
{% asset_img na_1.png %}

正如我们所看到的，类别是不平衡的，因此我们可以考虑使用某种重采样。
```python
ham_df = df[df['target'] == 'ham']['message_len'].value_counts().sort_index()
spam_df = df[df['target'] == 'spam']['message_len'].value_counts().sort_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ham_df.index,
    y=ham_df.values,
    name='ham',
    fill='tozeroy',
    marker_color=primary_blue,
))
fig.add_trace(go.Scatter(
    x=spam_df.index,
    y=spam_df.values,
    name='spam',
    fill='tozeroy',
    marker_color=primary_grey,
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Data Roles in Different Fields</span>'
)
fig.update_xaxes(range=[0, 70])
fig.show()
```
{% asset_img na_2.png %}

正如我们所看到的，正常邮件的长度往往低于垃圾邮件的长度。

#### 数据预处理

现在我们将对数据进行工程设计，以使模型更容易分类。这一部分对于缩小问题的维度非常重要。

#### 清理语料库

```python
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

df['message_clean'] = df['message'].apply(clean_text)
df.head()
```
输出结果为：
```bash
  target                                            message  message_len                                      message_clean
0    ham  Go until jurong point, crazy.. Available only ...           20  go until jurong point crazy available only in ...
1    ham                      Ok lar... Joking wif u oni...            6                            ok lar joking wif u oni
2   spam  Free entry in 2 a wkly comp to win FA Cup fina...           28  free entry in  a wkly comp to win fa cup final...
3    ham  U dun say so early hor... U c already then say...           11        u dun say so early hor u c already then say
4    ham  Nah I don't think he goes to usf, he lives aro...           13  nah i dont think he goes to usf he lives aroun...
```
##### Stopwords

`Stopwords`是英语中常用的单词，在句子中没有上下文含义。因此，我们在分类之前将它们删除。删除`Stopwords`的一些示例是：
{% asset_img na_3.png %}

```python
stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text
    
df['message_clean'] = df['message_clean'].apply(remove_stopwords)
df.head()
```
输出结果为：
```bash
  target                                            message  message_len                                      message_clean
0    ham  Go until jurong point, crazy.. Available only ...           20  go jurong point crazy available bugis n great ...
1    ham                      Ok lar... Joking wif u oni...            6                              ok lar joking wif oni
2   spam  Free entry in 2 a wkly comp to win FA Cup fina...           28  free entry  wkly comp win fa cup final tkts  m...
3    ham  U dun say so early hor... U c already then say...           11                      dun say early hor already say
4    ham  Nah I don't think he goes to usf, he lives aro...           13        nah dont think goes usf lives around though
```
#### 词干提取

##### 词干提取/特征化

出于语法原因，文档将使用单词的不同形式，例如`write、writing`和`writes`。此外，还有一些具有相似含义的派生相关词族。**词干提取**和**词形还原**的目标都是将单词的屈折形式和有时派生相关的形式减少为共同的基本形式。
- **词干提取**通常是指为了在大多数情况下正确实现目标而砍掉单词末尾的过程，并且通常包括删除派生词缀。
- **词形还原**通常是指使用词汇和单词的形态分析来正确地进行操作，通常旨在仅删除屈折词尾并返回单词的基本形式和字典形式。

{% asset_img na_4.png %}

##### 词干提取算法

`NLTK Python`库中实现了多种词干提取算法：
- `PorterStemmer`使用后缀剥离来生成词干。`PorterStemmer`以其简单和速度而闻名。请注意`PorterStemmer`如何通过简单地删除`cat`后面的'`s`'来给出单词“`cats`”的词根（词干）。这是添加到`cat`上的后缀，使其成为复数。但是，如果你看一下“`trouble`”、“`trouble`”和“`troubled`”，它们就会被归为“`trouble`”，因为`PorterStemmer`算法不遵循语言学，而是遵循一组适用于不同情况的`05`条规则，这些规则分阶段（逐步）应用于生成词干。这就是为什么`PorterStemmer`不经常生成实际英语单词的词干的原因。它不保留单词实际词干的查找表，而是应用算法规则来生成词干。它使用规则来决定删除后缀是否明智。
- 人们可以为任何语言生成自己的一组规则，这就是为什么`Python nltk`引入了`SnowballStemmers`，用于创建非英语`Stemmers`！
- `LancasterStemmer`（`Paice-Husk`词干分析器）是一种迭代算法，规则保存在外部。一个表包含约`120`条规则，按后缀的最后一个字母进行索引。在每次迭代中，它都会尝试通过单词的最后一个字符找到适用的规则。每条规则指定删除或替换结尾部分。如果没有这样的规则，则终止。如果一个单词以元音开头并且只剩下两个单词，或者如果一个单词以辅音开头并且只剩下三个字符，它也会终止。否则，应用该规则并重复该过程。
```python
stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

df['message_clean'] = df['message_clean'].apply(stemm_text)
df.head()
```
输出结果为：
```bash
  target                                            message  message_len                                      message_clean
0    ham  Go until jurong point, crazy.. Available only ...           20  go jurong point crazi avail bugi n great world...
1    ham                      Ok lar... Joking wif u oni...            6                                ok lar joke wif oni
2   spam  Free entry in 2 a wkly comp to win FA Cup fina...           28  free entri  wkli comp win fa cup final tkts  m...
3    ham  U dun say so early hor... U c already then say...           11                      dun say earli hor alreadi say
4    ham  Nah I don't think he goes to usf, he lives aro...           13          nah dont think goe usf live around though
```
#### 全部

```python
def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    # Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    
    return text

df['message_clean'] = df['message_clean'].apply(preprocess_data)
df.head()
```
输出结果为：
```bash
  target                                            message  message_len                                      message_clean
0    ham  Go until jurong point, crazy.. Available only ...           20  go jurong point crazi avail bugi n great world...
1    ham                      Ok lar... Joking wif u oni...            6                                ok lar joke wif oni
2   spam  Free entry in 2 a wkly comp to win FA Cup fina...           28  free entri  wkli comp win fa cup final tkts  m...
3    ham  U dun say so early hor... U c already then say...           11                        dun say ear hor alreadi say
4    ham  Nah I don't think he goes to usf, he lives aro...           13          nah dont think goe usf live around though
```

#### 目标编码

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(df['target'])

df['target_encoded'] = le.transform(df['target'])
df.head()
```
输出结果为：
```bash
  target                                            message  message_len                                      message_clean  target_encoded
0    ham  Go until jurong point, crazy.. Available only ...           20  go jurong point crazi avail bugi n great world...               0
1    ham                      Ok lar... Joking wif u oni...            6                                ok lar joke wif oni               0
2   spam  Free entry in 2 a wkly comp to win FA Cup fina...           28  free entri  wkli comp win fa cup final tkts  m...               1
3    ham  U dun say so early hor... U c already then say...           11                        dun say ear hor alreadi say               0
4    ham  Nah I don't think he goes to usf, he lives aro...           13          nah dont think goe usf live around though               0
```
#### Tokens 可视化

```python
twitter_mask = np.array(Image.open('/kaggle/input/masksforwordclouds/twitter_mask3.jpg'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=twitter_mask,
)
wc.generate(' '.join(text for text in df.loc[df['target'] == 'ham', 'message_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for HAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()
```
