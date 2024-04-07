---
title: NLP（Preprocessing when using embeddings）
date: 2024-04-07 09:17:11
tags:
  - AI
categories:
  - 人工智能
---

我想说明在构建深度学习`NLP`模型时如何进行有意义的预处理。
- 当有预先训练的嵌入时，不要使用标准预处理步骤，例如词干提取或停用词删除。
- 让你的词汇尽可能接近嵌入。
<!-- more -->

我们从一个巧妙的小技巧开始，使我们能够在将函数应用于`pandas Dataframe`时看到进度条。
```python
import re
import operator 
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors

tqdm.pandas()

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
print("Train shape : ",train.shape)
print("Test shape : ",test.shape)

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

# 让我们填充词汇表并显示前5个元素及其计数。请注意，现在我们可以使用progess_apply来查看进度条
sentences = train["question_text"].progress_apply(lambda x: x.split()).values
vocab = build_vocab(sentences)
print({k: vocab[k] for k in list(vocab)[:5]})

# 接下来，我们导入稍后要在模型中使用的嵌入。为了便于说明，我在这里使用GoogleNews。
news_path = '../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin'
embeddings_index = KeyedVectors.load_word2vec_format(news_path, binary=True)

# 接下来，我定义一个函数来检查词汇表和嵌入之间的交集。它将输出一个词汇表外（oov）单词的列表，我们可以用它来改进我们的预处理
def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

oov = check_coverage(vocab,embeddings_index)

# 只有24%的词汇会有嵌入，这使得21%的数据或多或少毫无用处。那么让我们看看并开始改进。为此，我们可以轻松查看最热门的oov单词。
oov[:10]
# [('to', 403183),('a', 402682),('of', 330825),('and', 251973),('India?', 16384),('it?', 12900),('do?', 8753),
# ('life?', 7753),('you?', 6295),('me?', 6202)]

# 第一位是“to”。为什么？仅仅是因为在训练GoogleNews嵌入时“to”被删除了。我们稍后会解决这个问题，现在我们要注意标点符号的分割，因为这似乎也是一个问题。
# 但是我们该如何处理标点符号——我们想要删除还是将其视为一个标记？我会说：这要看情况。如果令牌有嵌入，请保留它，如果没有，我们就不再需要它。让我们检查一下：

'?' in embeddings_index
# False
'&' in embeddings_index
# True

# 虽然“&”位于Google News嵌入中，但“?”不是。所以我们基本上定义了一个函数来拆分“&”并删除其他标点符号。
def clean_text(x):
    x = str(x)
    for punct in "/-'":
        x = x.replace(punct, ' ')
    for punct in '&':
        x = x.replace(punct, f' {punct} ')
    for punct in '?!.,"#$%\'()*+-/:;<=>@[\\]^_`{|}~' + '“”’':
        x = x.replace(punct, '')
    return x

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_text(x))
sentences = train["question_text"].apply(lambda x: x.split())
vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)

# 仅通过处理标点符号，我们就能将嵌入率从24%提高到57%。好的，让我们检查一下oov单词。
oov[:10]
# [('to', 406298),('a', 403852),('of', 332964),('and', 254081),('2017', 8781),('2018', 7373),('10', 6642),
#  ('12', 3694),('20', 2942),('100', 2883)]
# 看来数字也是一个问题。让我们检查前10个嵌入以获得线索。
for i in range(10):
    print(embeddings_index.index2entity[i])
# </s> in for that is on ## The with said
# 为什么里面有“##”？仅仅是因为重新处理，所有大于9的数字都已被哈希值替换。IE。15变为##，而123变为 ### 或15.80€变为##.##€。
# 因此，让我们模仿这个预处理步骤来进一步提高我们的嵌入覆盖率

def clean_numbers(x):
    x = re.sub('[0-9]{5,}', '#####', x)
    x = re.sub('[0-9]{4}', '####', x)
    x = re.sub('[0-9]{3}', '###', x)
    x = re.sub('[0-9]{2}', '##', x)
    return x

train["question_text"] = train["question_text"].progress_apply(lambda x: clean_numbers(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)
# 又增加3%。现在就像处理标点符号一样，但每一点都有帮助。让我们检查一下oov单词
oov[:20]
# [('to', 406298),('a', 403852),('of', 332964),('and', 254081),('favourite', 1247),('bitcoin', 987),('colour', 976),
#  ('doesnt', 918),('centre', 886),('Quorans', 858),('cryptocurrency', 822),('Snapchat', 807),('travelling', 705),
#  ('counselling', 634),('btech', 632),('didnt', 600),('Brexit', 493),('cryptocurrencies', 481),('blockchain', 474),
#  ('behaviour', 468)]

# 现在，我们在使用美国/英国词汇时处理常见的拼写错误，并用“social media”替换一些“modern”单词。对于此任务，我使用我不久前在堆栈溢出发现的多正则表达式脚本。
# 此外，我们将简单地删除单词“a”、“to”、“and”和“of”，因为在训练GoogleNews嵌入时这些词显然已经被采样了。

def _get_mispell(mispell_dict):
    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))
    return mispell_dict, mispell_re

mispell_dict = {'colour':'color',
                'centre':'center',
                'didnt':'did not',
                'doesnt':'does not',
                'isnt':'is not',
                'shouldnt':'should not',
                'favourite':'favorite',
                'travelling':'traveling',
                'counselling':'counseling',
                'theatre':'theater',
                'cancelled':'canceled',
                'labour':'labor',
                'organisation':'organization',
                'wwii':'world war 2',
                'citicise':'criticize',
                'instagram': 'social medium',
                'whatsapp': 'social medium',
                'snapchat': 'social medium'

                }
mispellings, mispellings_re = _get_mispell(mispell_dict)

def replace_typical_misspell(text):
    def replace(match):
        return mispellings[match.group(0)]

    return mispellings_re.sub(replace, text)

train["question_text"] = train["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
sentences = train["question_text"].progress_apply(lambda x: x.split())
to_remove = ['a','to','of','and']
sentences = [[word for word in sentence if not word in to_remove] for sentence in tqdm(sentences)]
vocab = build_vocab(sentences)
oov = check_coverage(vocab,embeddings_index)

# 我们将所有文本的嵌入量从89%提高到了99%。让我们再检查一下oov单词
oov[:20]
# [('bitcoin', 987),('Quorans', 858),('cryptocurrency', 822),('Snapchat', 807),('btech', 632),('Brexit', 493),
#  ('cryptocurrencies', 481),('blockchain', 474),('behaviour', 468),('upvotes', 432),('programme', 402),
#  ('Redmi', 379),('realise', 371),('defence', 364),('KVPY', 349),('Paytm', 334),('grey', 299),
#  ('mtech', 281),('Btech', 262),('bitcoins', 254)]
# 看起来不错。没有明显的oov单词
```

