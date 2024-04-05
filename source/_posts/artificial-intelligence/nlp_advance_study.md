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

##### 清理语料库

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
##### 词干提取

###### 词干提取/特征化

出于语法原因，文档将使用单词的不同形式，例如`write、writing`和`writes`。此外，还有一些具有相似含义的派生相关词族。**词干提取**和**词形还原**的目标都是将单词的屈折形式和有时派生相关的形式减少为共同的基本形式。
- **词干提取**通常是指为了在大多数情况下正确实现目标而砍掉单词末尾的过程，并且通常包括删除派生词缀。
- **词形还原**通常是指使用词汇和单词的形态分析来正确地进行操作，通常旨在仅删除屈折词尾并返回单词的基本形式和字典形式。

{% asset_img na_4.png %}

###### 词干提取算法

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
##### 全部

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

##### 目标编码

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
{% asset_img na_5.png %}

```python
twitter_mask = np.array(Image.open('/kaggle/input/masksforwordclouds/twitter_mask3.jpg'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=twitter_mask,
)
wc.generate(' '.join(text for text in df.loc[df['target'] == 'spam', 'message_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for HAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()
```
{% asset_img na_6.png %}

#### 矢量化

目前，我们将消息作为标记列表，现在我们需要将每条消息转换为`SciKit Learn`算法模型可以使用的**向量**。我们将使用`bag-of-words`模型分三个步骤来完成此操作：
- 计算某个单词在每条消息中出现的次数（称为频率）。
- 权衡计数，使频繁的标记获得较低的权重（逆文档频率）。
- 将向量标准化为单位长度，从原始文本长度中抽象出来（`L2`范数）。

让我们开始第一步：
每个向量的维度与`SMS`语料库中唯一单词的维度一样多。我们将首先使用`SciKit Learn`的`CountVectorizer`。该模型会将文本文档集合转换为标记计数矩阵。我们可以将其想象为一个二维矩阵。其中一维是整个词汇表（每个单词`1`行），另一个维度是实际文档，在本例中每条文本消息一列。
{% asset_img na_7.png %}

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
x = df['message_clean']
y = df['target_encoded']

print(len(x), len(y))

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

# instantiate the vectorizer
vect = CountVectorizer()
vect.fit(x_train)

# Use the trained to create a document-term matrix from train and test sets
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)
```
##### 调整CountVectorizer

`CountVectorizer`有一些应该知道的参数。
- `stop_words`：由于`CountVectorizer`只是计算词汇表中每个单词的出现次数，因此像“`the`”、“`and`”等极其常见的单词将成为非常重要的特征，但它们对文本的意义不大。如果您不考虑这些因素，您的模型通常可以改进。停用词只是您不想用作特征的单词列表。您可以设置参数`stop_words='english'`以使用内置列表。或者，您可以将`stop_words`设置为等于某个自定义列表。该参数默认为无。
- `ngram_range`：`n-gram`就是一串连续的`n`个单词。例如。句子“`I am Groot`”包含`2-grams`“`I am”和“am Groot`”。该句子本身就是一个`3`元语法。设置参数`ngram_range=(a,b)`，其中`a`是要包含在特征中的`ngram`的最小值，`b`是最大值。默认`ngram_range为(1,1)`。在最近的一个项目中，我对在线招聘信息进行了建模，我发现将`2-gram`包含在内可以显著提高模型的预测能力。这很直观。许多职位名称，例如“数据科学家”、“数据工程师”和“数据分析师”都是两个词长。
- `min_df`、`max_df`：这些是单词`/n-gram`必须用作特征的最小和最大文档频率。如果这些参数中的任何一个设置为整数，它们将用作每个特征必须位于的文档数量的界限才能被视为特征。如果其中一个设置为浮点数，则该数字将被解释为频率而不是数值限制。`min_df`默认为`1(int)`，`max_df`默认为`1.0(float)`。
- `max_features`：这个参数不言自明。`CountVectorizer`将选择最常出现在其词汇表中的单词/特征，并丢弃其他所有内容。

您可以在初始化`CountVectorizer`对象时设置这些参数，如下所示。
```python
vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), min_df=0.1, max_df=0.7, max_features=100)
```
##### TF-IDF

在信息检索中，`tf–idf`、`TF-IDF`或`TFIDF`是**术语频率–逆文档频率**的缩写，是一种数值统计量，旨在反映一个单词对于集合或语料库中的文档的重要性，经常使作为信息检索、文本挖掘和用户建模搜索中的权重因子。`tf-idf`值与单词在文档中出现的次数成比例增加，并由语料库中包含该单词的文档数量抵消，这有助于调整某些单词通常出现频率更高的事实。`tf–idf`是当今最流行的术语加权方案之一。`2015`年进行的一项调查显示，数字图书馆中`83%`的基于文本的推荐系统使用`tf-idf`。
```python
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(x_train_dtm)
x_train_tfidf = tfidf_transformer.transform(x_train_dtm)

x_train_tfidf

# <4179x5684 sparse matrix of type '<class 'numpy.float64'>' with 32201 stored elements in Compressed Sparse Row format>
```
##### 词嵌入：GloVe

```python
texts = df['message_clean']
target = df['target_encoded']

# Calculate the length of our vocabulary
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(texts)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(texts, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(texts), 
    length_long_sentence, 
    padding='post'
)

train_padded_sentences

# array([[   2, 3179,  274, ...,    0,    0,    0],
#        [   8,  236,  527, ...,    0,    0,    0],
#        [   9,  356,  588, ...,    0,    0,    0],
#        ...,
#        [6724, 1002, 6725, ...,    0,    0,    0],
#        [ 138, 1251, 1603, ...,    0,    0,    0],
#        [1986,  378,  170, ...,    0,    0,    0]], dtype=int32)
```
`GloVe`方法建立在一个重要的想法之上，您可以从`co-occurrence`矩阵导出单词之间的语义关系。为了获得单词的向量表示，我们可以使用一种名为`GloVe`（单词表示的全局向量）的无监督学习算法，该算法专注于整个语料库中单词的`co-occurrence`。它的嵌入与两个单词一起出现的概率有关。**词嵌入**是一种词表示形式，它将人类对语言的理解与机器对语言理解联系起来。他们已经学习了`n`维空间中文本的表示，其中具有相同含义的单词具有相似的表示。这意味着两个相似的单词由向量空间中非常接近的向量表示。因此，当使用词嵌入时，所有单个词都被表示为预定义向量空间中的**实值向量**。每个单词都映射到一个向量，并且类似于神经网络的方式学习向量值。
```python
embeddings_dictionary = dict()
embedding_dim = 100

# Load GloVe 100D embeddings
with open('/kaggle/input/glove6b100dtxt/glove.6B.100d.txt') as fp:
    for line in fp.readlines():
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions

# embeddings_dictionary
# Now we will load embedding vectors of those words that appear in the
# Glove dictionary. Others will be initialized to 0.

embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
embedding_matrix

# array([[ 0.        ,  0.        ,  0.        , ...,  0.        ,
#          0.        ,  0.        ],
#        [-0.57832998, -0.0036551 ,  0.34658   , ...,  0.070204  ,
#          0.44509   ,  0.24147999],
#        [-0.078894  ,  0.46160001,  0.57779002, ...,  0.26352   ,
#          0.59397   ,  0.26741001],
#        ...,
#        [ 0.63009   , -0.036992  ,  0.24052   , ...,  0.10029   ,
#          0.056822  ,  0.25018999],
#        [-0.12002   , -1.23870003, -0.23303001, ...,  0.13658001,
#         -0.61848003,  0.049843  ],
#        [ 0.        ,  0.        ,  0.        , ...,  0.        ,
#          0.        ,  0.        ]])
```
#### 建模

创建多项式朴素贝叶斯模型：
```python
import plotly.figure_factory as ff

x_axes = ['Ham', 'Spam']
y_axes =  ['Spam', 'Ham']

def conf_matrix(z, x=x_axes, y=y_axes):
    
    z = np.flip(z, 0)

    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>',
                      xaxis = dict(title='Predicted value'),
                      yaxis = dict(title='Real value')
                     )

    # add colorbar
    fig['data'][0]['showscale'] = True
    
    return fig

# Create a Multinomial Naive Bayes model
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

# Train the model
nb.fit(x_train_dtm, y_train)
```
##### 朴素贝叶斯 DTM

在统计学中，**朴素贝叶斯分类器**是简单的“**概率分类器**”，基于应用贝叶斯定理以及特征之间的强（朴素）独立性假设。它们是最简单的贝叶斯网络模型之一，但与核密度估计相结合，它们可以达到更高的精度水平。**朴素贝叶斯分类器**具有高度可扩展性，需要学习问题中的许多参数与变量（特征/预测变量）的数量呈线性关系。`Maximum-likelihood`训练可以通过评估封闭表达式来完成，这需要线性时间，而不是像许多其他类型的分类器那样通过昂贵的迭代近似来完成。
{% asset_img na_8.png %}

```python
# Make class anf probability predictions
y_pred_class = nb.predict(x_test_dtm)
y_pred_prob = nb.predict_proba(x_test_dtm)[:, 1]

# calculate accuracy of class predictions
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
# Calculate AUC
metrics.roc_auc_score(y_test, y_pred_prob)

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))

# 0.9784637473079684
# 0.974296765425861
```
{% asset_img na_9.png %}

##### 朴素贝叶斯

```python
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()), ('tfid', TfidfTransformer()), ('model', MultinomialNB())])

# Fit the pipeline with the data
pipe.fit(x_train, y_train)
y_pred_class = pipe.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred_class))

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))

# 0.9597989949748744
```
{% asset_img na_10.png %}

##### XGBoost

```python
import xgboost as xgb

pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=7,
        n_estimators=80,
        use_label_encoder=False,
        eval_metric='auc',
        # colsample_bytree=0.8,
        # subsample=0.7,
        # min_child_weight=5,
    ))
])

# Fit the pipeline with the data
pipe.fit(x_train, y_train)

y_pred_class = pipe.predict(x_test)
y_pred_train = pipe.predict(x_train)

print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))

# Train: 0.9834888729361091
# Test: 0.9662598707824839
```
{% asset_img na_11.png %}

#### 长短期记忆网络(LSTM)

```python
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences, 
    target, 
    test_size=0.25
)

def glove_lstm():
    model = Sequential()
    
    model.add(Embedding(
        input_dim=embedding_matrix.shape[0], 
        output_dim=embedding_matrix.shape[1], 
        weights = [embedding_matrix], 
        input_length=length_long_sentence
    ))
    
    model.add(Bidirectional(LSTM(
        length_long_sentence, 
        return_sequences = True, 
        recurrent_dropout=0.2
    )))
    
    model.add(GlobalMaxPool1D())
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(length_long_sentence, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

model = glove_lstm()
model.summary()
```
输出结果为：
```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, 80, 100)           672600    
_________________________________________________________________
bidirectional (Bidirectional (None, 80, 160)           115840    
_________________________________________________________________
global_max_pooling1d (Global (None, 160)               0         
_________________________________________________________________
batch_normalization (BatchNo (None, 160)               640       
_________________________________________________________________
dropout (Dropout)            (None, 160)               0         
_________________________________________________________________
dense (Dense)                (None, 80)                12880     
_________________________________________________________________
dropout_1 (Dropout)          (None, 80)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 80)                6480      
_________________________________________________________________
dropout_2 (Dropout)          (None, 80)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 81        
=================================================================
Total params: 808,521
Trainable params: 808,201
Non-trainable params: 320
_________________________________________________________________
```
```python
# Load the model and train!!
model = glove_lstm()

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor = 'val_loss', 
    verbose = 1, 
    save_best_only = True
)
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2, 
    verbose = 1, 
    patience = 5,                        
    min_lr = 0.001
)
history = model.fit(
    X_train, 
    y_train, 
    epochs = 7,
    batch_size = 32,
    validation_data = (X_test, y_test),
    verbose = 1,
    callbacks = [reduce_lr, checkpoint]
)
```
让我们看看结果：
```python
def plot_learning_curves(history, arr):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    for idx in range(2):
        ax[idx].plot(history.history[arr[idx][0]])
        ax[idx].plot(history.history[arr[idx][1]])
        ax[idx].legend([arr[idx][0], arr[idx][1]],fontsize=18)
        ax[idx].set_xlabel('A ',fontsize=16)
        ax[idx].set_ylabel('B',fontsize=16)
        ax[idx].set_title(arr[idx][0] + ' X ' + arr[idx][1],fontsize=16)

plot_learning_curves(history, [['loss', 'val_loss'],['accuracy', 'val_accuracy']])
```
{% asset_img na_12.png %}

```python
y_preds = (model.predict(X_test) > 0.5).astype("int32")
conf_matrix(metrics.confusion_matrix(y_test, y_preds))
```
{% asset_img na_13.png %}

#### BERT

`BERT`（来自`Transformer`的双向编码器表示）是`Google AI Language`的研究人员最近发表的一篇论文。它在各种`NLP`任务中展示了优秀的结果，包括问答(`SQuAD v1.1`)、自然语言推理 (`MNLI`) 等，在机器学习社区引起了轰动。`BERT`的关键技术创新是将流行的注意力模型`Transformer`的双向训练应用于语言建模。这与之前的研究形成对比，之前的研究从左到右或从右到左组合训练文本序列。论文结果显示，双向训练的语言模型比单向语言模型能够更深入地了解语言上下文和流程。在论文中，研究人员详细介绍了一种名为`Masked LM (MLM)`的新技术，该技术允许在模型中进行双向训练，而这在以前是无法想象的。
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from tqdm.notebook import tqdm
from tokenizers import BertWordPieceTokenizer
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from transformers import BertTokenizer
from transformers import TFBertModel

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    
except:
    strategy = tf.distribute.get_strategy()
    
print('Number of replicas in sync: ', strategy.num_replicas_in_sync)

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

def bert_encode(data, maximum_length) :
    input_ids = []
    attention_masks = []

    for text in data:
        encoded = tokenizer.encode_plus(
            text, 
            add_special_tokens=True,
            max_length=maximum_length,
            pad_to_max_length=True,

            return_attention_mask=True,
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return np.array(input_ids),np.array(attention_masks)

texts = df['message_clean']
target = df['target_encoded']

train_input_ids, train_attention_masks = bert_encode(texts,60)

def create_model(bert_model):
    
    input_ids = tf.keras.Input(shape=(60,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(60,),dtype='int32')

    output = bert_model([input_ids,attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(32,activation='relu')(output)
    output = tf.keras.layers.Dropout(0.2)(output)
    output = tf.keras.layers.Dense(1,activation='sigmoid')(output)
    
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    return model

bert_model = TFBertModel.from_pretrained('bert-base-uncased')
model = create_model(bert_model)
model.summary()
```
结果输出为：
```bash
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            [(None, 60)]         0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            [(None, 60)]         0                                            
__________________________________________________________________________________________________
tf_bert_model (TFBertModel)     TFBaseModelOutputWit 109482240   input_1[0][0]                    
                                                                 input_2[0][0]                    
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 32)           24608       tf_bert_model[0][1]              
__________________________________________________________________________________________________
dropout_43 (Dropout)            (None, 32)           0           dense_6[0][0]                    
__________________________________________________________________________________________________
dense_7 (Dense)                 (None, 1)            33          dropout_43[0][0]                 
==================================================================================================
Total params: 109,506,881
Trainable params: 109,506,881
Non-trainable params: 0
__________________________________________________________________________________________________
```
```python

# 训练模型
history = model.fit([train_input_ids, train_attention_masks],target,validation_split=0.2, epochs=3,batch_size=10)
```
```bash
Epoch 1/3
446/446 [==============================] - 1803s 4s/step - loss: 0.2607 - accuracy: 0.9095 - val_loss: 0.0709 - val_accuracy: 0.9767
Epoch 2/3
446/446 [==============================] - 1769s 4s/step - loss: 0.0636 - accuracy: 0.9838 - val_loss: 0.0774 - val_accuracy: 0.9767
Epoch 3/3
446/446 [==============================] - 1781s 4s/step - loss: 0.0297 - accuracy: 0.9935 - val_loss: 0.0452 - val_accuracy: 0.9857
```
```python
plot_learning_curves(history, [['loss', 'val_loss'],['accuracy', 'val_accuracy']])
```
{% asset_img na_14.png %}

#### NLP: Disaster Tweets

```python
df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv", encoding="latin-1")
test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv", encoding="latin-1")

df = df.dropna(how="any", axis=1)
df['text_len'] = df['text'].apply(lambda x: len(x.split(' ')))

df.head()
```
结果输出为：
```bash
	id	text	target	text_len
0	1	Our Deeds are the Reason of this #earthquake M...	1	13
1	4	Forest fire near La Ronge Sask. Canada	1	7
2	5	All residents asked to 'shelter in place' are ...	1	22
3	6	13,000 people receive #wildfires evacuation or...	1	9
4	7	Just got sent this photo from Ruby #Alaska as ...	1	17
```
```python
balance_counts = df.groupby('target')['target'].agg('count').values
balance_counts

fig = go.Figure()
fig.add_trace(go.Bar(
    x=['Fake'],
    y=[balance_counts[0]],
    name='Fake',
    text=[balance_counts[0]],
    textposition='auto',
    marker_color=primary_blue
))
fig.add_trace(go.Bar(
    x=['Real disaster'],
    y=[balance_counts[1]],
    name='Real disaster',
    text=[balance_counts[1]],
    textposition='auto',
    marker_color=primary_grey
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by target</span>'
)
fig.show()
```
{% asset_img na_15.png %}

```python
disaster_df = df[df['target'] == 1]['text_len'].value_counts().sort_index()
fake_df = df[df['target'] == 0]['text_len'].value_counts().sort_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=disaster_df.index,
    y=disaster_df.values,
    name='Real disaster',
    fill='tozeroy',
    marker_color=primary_blue,
))
fig.add_trace(go.Scatter(
    x=fake_df.index,
    y=fake_df.values,
    name='Fake',
    fill='tozeroy',
    marker_color=primary_grey,
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Data Roles in Different Fields</span>'
)
fig.show()
```
{% asset_img na_16.png %}

##### 数据预处理

```python
def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'', text)


def remove_emoji(text):
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def remove_html(text):
    html = re.compile(r'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    return re.sub(html, '', text)

# Special thanks to https://www.kaggle.com/tanulsingh077 for this function
def clean_text(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
        '', 
        text
    )
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    
    text = remove_url(text)
    text = remove_emoji(text)
    text = remove_html(text)
    
    return text

# Test emoji removal
remove_emoji("Omg another Earthquake 😔😔")

stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

stemmer = nltk.SnowballStemmer("english")

def preprocess_data(text):
    # Clean puntuation, urls, and so on
    text = clean_text(text)
    # Remove stopwords and Stemm all the words in the sentence
    text = ' '.join(stemmer.stem(word) for word in text.split(' ') if word not in stop_words)

    return text

test_df['text_clean'] = test_df['text'].apply(preprocess_data)

df['text_clean'] = df['text'].apply(preprocess_data)
df.head()
```
结果输出为：
```bash
   id                                               text  target  text_len                                         text_clean
0   1  Our Deeds are the Reason of this #earthquake M...       1        13          deed reason earthquak may allah forgiv us
1   4             Forest fire near La Ronge Sask. Canada       1         7               forest fire near la rong sask canada
2   5  All residents asked to 'shelter in place' are ...       1        22  resid ask shelter place notifi offic evacu she...
3   6  13,000 people receive #wildfires evacuation or...       1         9       peopl receiv wildfir evacu order california 
4   7  Just got sent this photo from Ruby #Alaska as ...       1        17  got sent photo rubi alaska smoke wildfir pour ...
```

##### 词云

```python
def create_corpus_df(tweet, target):
    corpus=[]
    
    for x in tweet[tweet['target']==target]['text_clean'].str.split():
        for i in x:
            corpus.append(i)
    return corpus

corpus_disaster_tweets = create_corpus_df(df, 1)

dic=defaultdict(int)
for word in corpus_disaster_tweets:
    dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
print(top)

# [('fire', 266),('bomb', 179),('kill', 158),('news', 132),('via', 121),('flood', 120),('disast', 116),('california', 115),
#  ('crash', 110),('suicid', 110)]
```
```python
twitter_mask = np.array(Image.open('twitter_mask3.jpg'))

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=twitter_mask,
)
wc.generate(' '.join(text for text in df.loc[df['target'] == 1, 'text_clean']))
plt.figure(figsize=(12,6))
plt.title('Top words for Real Disaster tweets', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()
```
{% asset_img na_17.png %}

```python
corpus_disaster_tweets = create_corpus_df(df, 0)

dic=defaultdict(int)
for word in corpus_disaster_tweets:
    dic[word]+=1
        
top=sorted(dic.items(), key=lambda x:x[1],reverse=True)[:10]
top

wc = WordCloud(
    background_color='white', 
    max_words=200, 
    mask=twitter_mask,
)
wc.generate(' '.join(text for text in df.loc[df['target'] == 0, 'text_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for Fake messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()

[('like', 306),('get', 222),('amp', 192),('new', 168),('go', 142),('dont', 139),('one', 134),('bodi', 116),
 ('love', 115),('bag', 108)]
```
{% asset_img na_18.png %}

##### 建模

```python
# how to define X and y (from the SMS data) for use with COUNTVECTORIZER
x = df['text_clean']
y = df['target']

# Split into train and test sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

pipe = Pipeline([
    ('bow', CountVectorizer()), 
    ('tfid', TfidfTransformer()),  
    ('model', xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='auc',
    ))
])
from sklearn import metrics

# Fit the pipeline with the data
pipe.fit(x_train, y_train)

y_pred_class = pipe.predict(x_test)
y_pred_train = pipe.predict(x_train)

print('Train: {}'.format(metrics.accuracy_score(y_train, y_pred_train)))
print('Test: {}'.format(metrics.accuracy_score(y_test, y_pred_class)))

conf_matrix(metrics.confusion_matrix(y_test, y_pred_class))

# 5709 5709
# 1904 1904

# Train: 0.8567174636538798
# Test: 0.7725840336134454
```
{% asset_img na_19.png %}

##### GloVe - LSTM

我们将使用`LSTM`（长短期记忆）模型。我们需要执行标记化——将文本分割成单词句子的处理。在这个过程中，我们也扔掉了标点符号和额外的符号。标记化的好处在于，它更容易转换为原始数字的格式，这实际上可以用于处理：
```python
train_tweets = df['text_clean'].values
test_tweets = test_df['text_clean'].values
train_target = df['target'].values

# Calculate the length of our vocabulary
word_tokenizer = Tokenizer()
word_tokenizer.fit_on_texts(train_tweets)

vocab_length = len(word_tokenizer.word_index) + 1
vocab_length

def show_metrics(pred_tag, y_test):
    print("F1-score: ", f1_score(pred_tag, y_test))
    print("Precision: ", precision_score(pred_tag, y_test))
    print("Recall: ", recall_score(pred_tag, y_test))
    print("Acuracy: ", accuracy_score(pred_tag, y_test))
    print("-"*50)
    print(classification_report(pred_tag, y_test))
    
def embed(corpus): 
    return word_tokenizer.texts_to_sequences(corpus)

longest_train = max(train_tweets, key=lambda sentence: len(word_tokenize(sentence)))
length_long_sentence = len(word_tokenize(longest_train))

train_padded_sentences = pad_sequences(
    embed(train_tweets), 
    length_long_sentence, 
    padding='post'
)
test_padded_sentences = pad_sequences(
    embed(test_tweets), 
    length_long_sentence,
    padding='post'
)

train_padded_sentences

# 13704

# array([[3635,  467,  201, ...,    0,    0,    0],
#        [ 136,    2,  106, ...,    0,    0,    0],
#        [1338,  502, 1807, ...,    0,    0,    0],
#        ...,
#        [ 448, 1328,    0, ...,    0,    0,    0],
#        [  28,  162, 2637, ...,    0,    0,    0],
#        [ 171,   31,  413, ...,    0,    0,    0]], dtype=int32)
```
为了获得单词的向量表示，我们可以使用一种名为`GloVe`（单词表示的全局向量）的无监督学习算法，该算法专注于整个语料库中单词的共现。它的嵌入与两个单词一起出现的概率有关。
```python
# Load GloVe 100D embeddings
# We are not going to do it here as they were loaded earlier.

# Now we will load embedding vectors of those words that appear in the
# Glove dictionary. Others will be initialized to 0.
embedding_matrix = np.zeros((vocab_length, embedding_dim))

for word, index in word_tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector
        
embedding_matrix

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    train_padded_sentences, 
    train_target, 
    test_size=0.25
)


# Load the model and train!!
model = glove_lstm()

checkpoint = ModelCheckpoint(
    'model.h5', 
    monitor = 'val_loss', 
    verbose = 1, 
    save_best_only = True
)
# 模型 LSTM
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2, 
    verbose = 1, 
    patience = 5,                        
    min_lr = 0.001
)
history = model.fit(
    X_train, 
    y_train, 
    epochs = 7,
    batch_size = 32,
    validation_data = (X_test, y_test),
    verbose = 1,
    callbacks = [reduce_lr, checkpoint]
)

plot_learning_curves(history, [['loss', 'val_loss'],['accuracy', 'val_accuracy']])
```
{% asset_img na_20.png %}

```python
preds = model.predict_classes(X_test)
show_metrics(preds, y_test)
```
结果输出为：
```bash
F1-score:  0.7552910052910053
Precision:  0.6709753231492362
Recall:  0.8638426626323752
Acuracy:  0.805672268907563
--------------------------------------------------
              precision    recall  f1-score   support

           0       0.91      0.77      0.84      1243
           1       0.67      0.86      0.76       661

    accuracy                           0.81      1904
   macro avg       0.79      0.82      0.80      1904
weighted avg       0.83      0.81      0.81      1904
```
