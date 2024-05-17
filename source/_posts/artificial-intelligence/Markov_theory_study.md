---
title: 序列模型 (循环神经网络)(TensorFlow)
date: 2024-05-16 17:38:11
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

简言之，如果说卷积神经网络可以有效地处理空间信息(图片)，**循环神经网络**(`recurrent neural network，RNN`)则可以更好地处理序列信息(文本)。循环神经网络通过引入状态变量存储过去的信息和当前的输入，从而可以确定当前的输出。
<!-- more -->
想象一下有人正在看电影。一名忠实的用户会对每一部电影都给出评价，毕竟一部好电影需要更多的支持和认可。然而事实证明，事情并不那么简单。随着时间的推移，人们对电影的看法会发生很大的变化。简而言之，电影评分决不是固定不变的。因此，使用时间动力学可以得到更准确的电影推荐。当然，序列数据不仅仅是关于电影评分的。处理序列数据需要统计工具和新的深度神经网络架构。为了简单起见，我们以下图所示的股票价格（富时`100`指数）为例。
{% asset_img mar_1.png "近30年的富时100指数" %}

其中，用{% mathjax %}x_t{% endmathjax %}表示价格，即在时间步(`time step`){% mathjax %}t\in \mathbb{Z}^{+}{% endmathjax %}时，观察到的价格{% mathjax %}x_t{% endmathjax %}。请注意，{% mathjax %}t{% endmathjax %}对于本文中的序列通常是离散的，并在证书或其子集上变化。假设一个交易员想在{% mathjax %}t{% endmathjax %}日的股市中表现良好，于是通过以下途径预测{% mathjax %}x_t{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
x_t \sim P(x_t|x_{t-1},\ldots,x_1)
{% endmathjax %}
#### 自回归模型

为了实现这个预测，交易员可以使用回归模型，输入数据的数量这个数字将会随着我们遇到的数据量的增加而增加，因此需要一个近似方法来使这个计算变得容易处理。简单地说，它归结为以下两种策略。
- 第一种策略，假设在现实情况下相当长的序列{% mathjax %}x_{t-1},\dots,x_1{% endmathjax %}可能是不必要的， 因此我们只需要满足某个长度为{% mathjax %}\tau{% endmathjax %}的时间跨度，即使用观测序列{% mathjax %}x_{t-1},\ldots,x_{t-\tau}{% endmathjax %}。当下获得的最直接的好处就是参数的数量总是不变的，至少在{% mathjax %}t > \tau{% endmathjax %}时如此，这就使我们能够训练一个上面提及的深度网络。 这种模型被称为**自回归模型**(`autoregressive models`)，因为它们是对自己执行回归。
- 第二种策略，如下图所示，是保留一些对过去观测的总结{% mathjax %}h_t{% endmathjax %}，并且同时更新预测{% mathjax %}\hat{x}_t{% endmathjax %}和总结{% mathjax %}h_t{% endmathjax %}。这就产生了基于{% mathjax %}\hat{x}_t= P(x_t|h_t){% endmathjax %}估计{% mathjax %}x_t{% endmathjax %}，以及公式{% mathjax %}h_t = g(h_{t-1},x_{t-1}){% endmathjax %}更新的模型。由于{% mathjax %}h_t{% endmathjax %}从未被观测到，这类模型也被称为**隐变量自回归模型**(`latent autoregressive models`)。
{% asset_img mar_2.png "隐变量自回归模型" %}

这两种情况都有一个显而易见的问题：如何生成训练数据？一个经典方法是使用历史观测来预测下一个未来观测。显然，我们并不指望时间会停滞不前。然而，一个常见的假设是虽然特定值{% mathjax %}x_t{% endmathjax %}可能会改变，但是序列本身的动力学不会改变。这样的假设是合理的，因为新的动力学一定受新的数据影响，而我们不可能用目前所掌握的数据来预测新的动力学。统计学家称不变的动力学为静止的(`stationary`)。因此，整个序列的估计值都将通过以下的方式获得：
{% mathjax '{"conversion":{"em":14}}' %}
P(x_1,\ldots,x_{T}) = \prod_{t=1}^T P(x_t|x_{t-1,\ldots,x_1})
{% endmathjax %}
注意，如果我们处理的是离散的对象（如单词），而不是连续的数字，则上述的考虑仍然有效。唯一的差别是，对于离散的对象， 我们需要使用分类器而不是回归模型来估计{% mathjax %}P(x_t|x_{t-1,\ldots,x_1}){% endmathjax %}。
#### 马尔可夫模型

回想一下，在自回归模型的近似法中，我们使用{% mathjax %}x_{t-1},\ldots,x_{t-\tau}{% endmathjax %}而不是{% mathjax %}x_{t-1},\ldots,x_1{% endmathjax %}来估计{% mathjax %}x_t{% endmathjax %}。只要这种是近似精确的，我们就说序列满足马尔克夫条件(`Markov condition`)。特别是，如果{% mathjax %}\tau= 1{% endmathjax %}，得到一个一阶马尔科夫模型(`first-order Markov model`)，{% mathjax %}P(x){% endmathjax %}由下式给出：
{% mathjax '{"conversion":{"em":14}}' %}
P(x_1,\ldots,x_T) = \prod_{t=1}^T P(x_t|x_{t-1})\;\text{当}\;P(x_1|x_0) = P(x_1)
{% endmathjax %}
当假设{% mathjax %}x_t{% endmathjax %}仅是离散值时，这样的模型特别棒，因为在这种情况下，使用动态规划可以沿着马尔可夫链精确地计算结果。例如，我们可以高效地计算{% mathjax %}P(x_{t+1}|x_{t-1}){% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
P(x_{t+1|x_{t-1}}) & = \frac{\sum_{x_t} P(x_{t+1},x_t,x_{t-1})}{P(x_{t-1})} \\ 
& = \frac{\sum_{x_t} P(x_{t+1|x_t,x_{t-1}})P(x_t,x_{t-1})}{P(x_{t-1})} \\
& = \sum_{x_t} P(x_{t+1}|x_t)P(x_t|x_{t-1}) \\
\end{align}
{% endmathjax %}
利用这一事实，我们只需要考虑过去观察中的一个非常短的历史：{% mathjax %}P(x_{t+1}|x_t,x_{t-1}) = P(x_{t+1}|x_t){% endmathjax %}。隐马尔可夫模型中的动态规划先不讲解，而动态规划这些计算工具已经在控制算法和强化学习算法广泛使用。
#### 因果关系

原则上，将{% mathjax %}P(x_1,\ldots,x_{T}){% endmathjax %}倒序展开也没什么问题。毕竟，基于条件概率公式，我们总是可以写出：
{% mathjax '{"conversion":{"em":14}}' %}
P(x_1,\ldots,x_{T}) = \prod_{t=T}^1 P(x_t|x_{t+1},\ldots,x_T)
{% endmathjax %}
事实上，如果基于一个马尔可夫模型，我们还可以得到一个反向的条件概率分布。然而，在许多情况下，数据存在一个自然的方向，即在时间上是前进的。很明显，未来的事件不能影响过去。因此，如果我们改变{% mathjax %}x_t{% endmathjax %}，可能会影响未来发生的事情{% mathjax %}x_{t+1}{% endmathjax %}，但不能反过来。也就是说，如果我们改变{% mathjax %}x_t{% endmathjax %}，基于过去事件得到的分布不会改变。因此，解释{% mathjax %}P(x_{t+1}|x_t){% endmathjax %}应该比解释{% mathjax %}P(x_t|x_{t+1}){% endmathjax %}更容易。例如，在某些情况下，对于某些可加性噪声{% mathjax %}\epsilon{% endmathjax %}，显然我们可以找到{% mathjax %}x_{t+1}= f(x_t) + \epsilon{% endmathjax %}，而反之则不行 。而这个向前推进的方向恰好也是我们通常感兴趣的方向。彼得斯等人对该主题的更多内容做了详尽的解释，而我们的上述讨论只是其中的冰山一角。
#### 文本预处理

例如，一篇文章可以被简单地看作一串单词序列，甚至是一串字符序列。我们将解析文本的常见预处理步骤包括为：
- 将文本作为字符串加载到内存中。
- 将字符串拆分为词元（如单词和字符）。
- 建立一个词表，将拆分的词元映射到数字索引。
- 将文本转换为数字索引序列，方便模型操作。

##### 词元化

下面的`tokenize`函数将文本行列表（`lines`）作为输入，列表中的每个元素是一个文本序列（如一条文本行）。每个文本序列又被拆分成一个词元列表，**词元（`token`）是文本的基本单位**。 最后，返回一个由词元列表组成的列表，其中的每个词元都是一个字符串（`string`）。
```python
def tokenize(lines, token='word'): 
    """将文本行拆分为单词或字符词元"""
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('错误：未知词元类型：' + token)

tokens = tokenize(lines)
for i in range(11):
    print(tokens[i])

# ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
# ['i']
# []
# ['the', 'time', 'traveller', 'for', 'so', 'it', 'will', 'be', 'convenient', 'to', 'speak', 'of', 'him']
# ['was', 'expounding', 'a', 'recondite', 'matter', 'to', 'us', 'his', 'grey', 'eyes', 'shone', 'and']
# ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
```
##### 词表

词元的类型是字符串，而模型需要的输入是数字，因此这种类型不方便模型使用。现在，让我们构建一个字典，通常也叫做**词表**(`vocabulary`)，用来将字符串类型的词元映射到从`0`开始的数字索引中。我们先将训练集中的所有文档合并在一起，对它们的唯一词元进行统计，得到的统计结果称之为**语料**(`corpus`)。然后根据每个唯一词元的出现频率，为其分配一个数字索引。很少出现的词元通常被移除，这可以降低复杂性。另外，语料库中不存在或已删除的任何词元都将映射到一个特定的未知词元`“<unk>”`。我们可以选择增加一个列表，用于保存那些被保留的词元，例如：填充词元（`“<pad>”`）；序列开始词元（`“<bos>”`）；序列结束词元（`“<eos>”`）。
```python
class Vocab: 
    """文本词表"""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # 按出现频率排序
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

def count_corpus(tokens):  #@save
    """统计词元的频率"""
    # 这里的tokens是1D列表或2D列表
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # 将词元列表展平成一个列表
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

# 我们首先使用时光机器数据集作为语料库来构建词表，然后打印前几个高频词元及其索引。
vocab = Vocab(tokens)
print(list(vocab.token_to_idx.items())[:10])

# [('<unk>', 0), ('the', 1), ('i', 2), ('and', 3), ('of', 4), ('a', 5), 
# ('to', 6), ('was', 7), ('in', 8), ('that', 9)]

# 现在，我们可以将每一条文本行转换成一个数字索引列表。
for i in [0, 10]:
    print('文本:', tokens[i])
    print('索引:', vocab[tokens[i]])

# 文本: ['the', 'time', 'machine', 'by', 'h', 'g', 'wells']
# 索引: [1, 19, 50, 40, 2183, 2184, 400]
# 文本: ['twinkled', 'and', 'his', 'usually', 'pale', 'face', 'was', 'flushed', 'and', 'animated', 'the']
# 索引: [2186, 3, 25, 1044, 362, 113, 7, 1421, 3, 1045, 1]

# 我们将所有功能打包到load_corpus_time_machine函数中，该函数返回corpus（词元索引列表）和vocab（时光机器语料库的词表）。
# 1.为了简化, 我们使用字符（而不是单词）实现文本词元化；
# 2.时光机器数据集中的每个文本行不一定是一个句子或一个段落，还可能是一个单词，因此返回的corpus仅处理为单个列表。
def load_corpus_time_machine(max_tokens=-1): 
    """返回时光机器数据集的词元索引列表和词表"""
    lines = read_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    # 因为时光机器数据集中的每个文本行不一定是一个句子或一个段落，
    # 所以将所有文本行展平到一个列表中
    corpus = [vocab[token] for line in tokens for token in line]
    if max_tokens > 0:
        corpus = corpus[:max_tokens]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()
len(corpus), len(vocab)

# (170580, 28)
```
#### 语言模型

我们了解了如何将文本数据映射为词元，以及将这些词元可以视为一系列离散的观测，例如单词或字符。假设长度为{% mathjax %}T{% endmathjax %}的文本序列中的词元依次为{% mathjax %}x_1,x_2,\ldots,x_T{% endmathjax %}。于是，{% mathjax %}x_t{% endmathjax %}（{% mathjax %}1\leq t \leq T{% endmathjax %}）可以被认为是文本序列在时间步{% mathjax %}t{% endmathjax %}处的观测或标签。在给定这样的文本序列时，**语言模型**(`language model`)的目标是估计序列的联合概率。
{% mathjax '{"conversion":{"em":14}}' %}
P(x_1,x_2,\ldots,x_T)
{% endmathjax %}
例如，只需要一次抽取一个词元{% mathjax %}x_t\sim P(x_t|x_{t-1},\ldots,x_1){% endmathjax %}，一个理想的语言模型就能够基于模型本身生成自然文本。与猴子使用打字机完全不同的是，从这样的模型中提取的文本 都将作为自然语言（例如，英语文本）来传递。只需要基于前面的对话片断中的文本，就足以生成一个有意义的对话。显然，我们离设计出这样的系统还很遥远，因为它需要“理解”文本，而不仅仅是生成语法合理的内容。尽管如此，语言模型依然是非常有用的。例如，短语`“to recognize speech”`和`“to wreck a nice beach”`读音上听起来非常相似。这种相似性会导致语音识别中的歧义，但是这很容易通过语言模型来解决，因为第二句的语义很奇怪。同样，在文档摘要生成算法中，“狗咬人”比“人咬狗”出现的频率要高得多，或者“我想吃奶奶”是一个相当匪夷所思的语句，而“我想吃，奶奶”则要正常得多。

显而易见，我们面对的问题是如何对一个文档，甚至是一个词元序列进行建模。假设在单词级别对文本数据进行词元化，我们可以依靠对序列模型的分析。让我们从基本概率规则开始：
{% mathjax '{"conversion":{"em":14}}' %}
P(x_1,x_2,\ldots,x_T) = \prod_{t=1}^T P(x_t|x_1,\ldots,x_{t-1})
{% endmathjax %}
例如，包含了四个单词的一个文本序列的概率是：
{% mathjax '{"conversion":{"em":14}}' %}
P(\text{deep,learning,is,fun}) = P(\text{deep})P(\text{learning}|\text{deep})P(\text{is}|\text{deep,learning})P(\text{fun}|\text{deep,learning,is})
{% endmathjax %}
为了训练语言模型，我们需要计算单词的概率，以及给定前面几个单词后出现某个单词的条件概率。这些概率本质上就是语言模型的参数。这里，我们假设训练数据集是一个大型的文本语料库。比如，维基百科的所有条目、古登堡计划，或者所有发布在网络上的文本。训练数据集中词的概率可以根据给定词的相对词频来计算。例如，可以将估计值{% mathjax %}\hat{P}(\text{deep}){% endmathjax %}计算为任何以单词`“deep”`开头的句子的概率。一种（稍稍不太精确的）方法是统计单词“`deep`”在数据集中的出现次数，然后将其除以整个语料库中的单词总数。这种方法效果不错，特别是对于频繁出现的单词。接下来，我们可以尝试估计：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{P}(\text{learning}|\text{deep}) = \frac{n(\text{deep,learning})}{n(\text{deep})}
{% endmathjax %}
其中{% mathjax %}n(x){% endmathjax %}和{% mathjax %}n(x,x'){% endmathjax %}分别是单个单词和连续单词对的出现次数。不幸的是，由于连续单词对`“deep learning”`的出现频率要低得多， 所以估计这类单词正确的概率要困难得多。特别是对于一些不常见的单词组合，要想找到足够的出现次数来获得准确的估计可能都不容易。而对于三个或者更多的单词组合，情况会变得更糟。许多合理的三个单词组合可能是存在的，但是在数据集中却找不到。除非我们提供某种解决方案，来将这些单词组合指定为非零计数，否则将无法在语言模型中使用它们。如果数据集很小，或者单词非常罕见，那么这类单词出现一次的机会可能都找不到。一种常见的策略是执行某种形式的**拉普拉斯平滑**(`Laplace smoothing`)，具体方法是在所有计数中添加一个小常量。用{% mathjax %}n{% endmathjax %}表示训练集中的单词总数，用{% mathjax %}m{% endmathjax %}表示唯一单词的数量。此解决方案有助于处理单元素问题，例如通过：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\hat{P}(x) & = \frac{n(x) + \epsilon_1/m}{n + \epsilon_1} \\ 
\hat{P}(x'|x) & = \frac{n(x,x') + \epsilon_2\hat{P}(x')}{n(x) + \epsilon_2} \\
\hat{P}(x''|x,x') & = \frac{n(x,x',x'') + \epsilon_3\hat{P}(x'')}{n(x,x') + \epsilon_3} \\
\end{align}
{% endmathjax %}
其中，{% mathjax %}\epsilon_1,\epsilon_2{% endmathjax %}和{% mathjax %}\epsilon_1{% endmathjax %}是超参数。以{% mathjax %}\epsilon_1{% endmathjax %}为例：当{% mathjax %}\epsilon_1 = 0{% endmathjax %}时，不应用平滑；当{% mathjax %}\epsilon_1{% endmathjax %}接近正无穷大时，{% mathjax %}\hat{P}(x){% endmathjax %}接近均匀概率分布{% mathjax %}1/m{% endmathjax %}。上面的公式是一个相当原始的变形。然而，这样的模型很容易变得无效，原因如下：首先，我们需要存储所有的计数；其次，这完全忽略了单词的意思。例如，“猫”(`cat`)和“猫科动物”(`feline`)可能出现在相关的上下文中，但是想根据上下文调整这类模型其实是相当困难的。最后，长单词序列大部分是没出现过的，因此一个模型如果只是简单地统计先前“看到”的单词序列频率，那么模型面对这种问题肯定是表现不佳的。
#### 马尔可夫模型与n元语法

在讨论包含深度学习的解决方案之前，我们需要了解更多的概念和术语。回想一下我们在马尔可夫模型中，并且将其应用于语言建模。如果{% mathjax %}P(x_{t+1}|x_t,\ldots,x_1) = P(x_{t+1}|x_t){% endmathjax %}，则序列上的分布满足一阶马尔可夫性质。阶数越高，对应的依赖关系就越长。这种性质推导出了许多可以应用于序列建模的近似公式：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
P(x_1,x_2,x_3,x_4) & = P(x_1)P(x_2)P(x_3)P(x_4) \\ 
P(x_1,x_2,x_3,x_4) & = P(x_1)P(x_2|x_1)P(x_3|x_2)P(x_4|x_3) \\
P(x_1,x_2,x_3,x_4) & = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2)P(x_4|x_2,x_3) \\
\end{align}
{% endmathjax %}
通常，涉及一个、两个和三个变量的概率公式分别被称为**一元语法**(`unigram`)、**二元语法**(`bigram`)和**三元语法**(`trigram`)模型。

#### 总结

**内插法**（在现有观测值之间进行估计）和**外推法**（对超出已知观测范围进行预测）在实践的难度上差别很大。因此，对于所拥有的序列数据，在训练时始终要尊重其时间顺序，即最好不要基于未来的数据进行训练。序列模型的估计需要专门的统计工具，两种较流行的选择是自回归模型和隐变量自回归模型。对于时间是向前推进的因果模型，正向估计通常比反向估计更容易。对于直到时间步{% mathjax %}t{% endmathjax %}的观测序列，其在时间步{% mathjax %}t + k{% endmathjax %}的预测输出是“{% mathjax %}k{% endmathjax %}步预测”。随着我们对预测时间{% mathjax %}k{% endmathjax %}值的增加，会造成误差的快速累积和预测质量的极速下降。文本是序列数据的一种最常见的形式之一。为了对文本进行预处理，我们通常将文本拆分为词元，构建词表将词元字符串映射为数字索引，并将文本数据转换为词元索引以供模型操作。语言模型是自然语言处理的关键。{% mathjax %}n{% endmathjax %}元语法通过截断相关性，为处理长序列提供了一种实用的模型。长序列存在一个问题：它们很少出现或者从不出现。齐普夫定律支配着单词的分布，这个分布不仅适用于一元语法，还适用于其他{% mathjax %}n{% endmathjax %}元语法。通过拉普拉斯平滑法可以有效地处理结构丰富而频率不足的低频词词组。读取长序列的主要方式是随机采样和顺序分区。在迭代过程中，后者可以保证来自两个相邻的小批量中的子序列在原始序列上也是相邻的。