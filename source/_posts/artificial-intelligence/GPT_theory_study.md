---
title: GPT模型探析（LLM）(Numpy)
date: 2024-06-04 11:30:11
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

`GPT`代表生成式预训练`Transformer`(`Generative Pre-trained Transformer`)。这是一类基于`Transformer`的神经网络架构。**生成式**(`Generative`)：`GPT`可以生成文本；**预训练**(`Pre-trained`)：`GPT`基于来自于书本、互联网等来源的海量文本进行训练；`Transformer`：`GPT`是一个`decoder-only`的`Transformer`神经网络结构。
<!-- more -->
#### 输入/输出

##### 输入

输入是一些文本字符串，使用`tokenizer`（分词器）将这些文本字符串拆解为”小片段“，我们将这些片段称之为`token`。最后我们使用词汇表(`vocabulary`)将`token`映射为整数。在实际使用中，我们不仅仅使用简单的通过空格分隔去做分词，我们还会使用一些更高级的方法，比如`Byte-Pair Encoding`或者`WordPiece`，但它们的原理是一样的：
- 有一个`vocab`即词汇表，可以将字符串`token`映射为整数索引。
- 有一个`encode`方法，即编码方法，可以实现`str` -> `list[int]`的转化。
- 有一个`decode`方法，即解码方法，可以实现`list[int]` -> `str`的转化。

```python
# the index of a token in the vocab represents the integer id for that token
# the integer id for "heroes" would be 2, since vocab[2] = "heroes"
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]

# a pretend tokenizer that tokenizes on whitespace
tokenizer = WhitespaceTokenizer(vocab)

# the encode() method converts a str -> list[int]
ids = tokenizer.encode("not all heroes wear") # ids = [1, 0, 2, 4]

# we can see what the actual tokens are via our vocab mapping
tokens = [tokenizer.vocab[i] for i in ids] # tokens = ["not", "all", "heroes", "wear"]

# the decode() method converts back a list[int] -> str
text = tokenizer.decode(ids) # text = "not all heroes wear"
```
##### 输出

输出是一个二维数组，其中`output[i][j]`表示模型的预测概率，这个概率代表了词汇表中位于`vocab[j]`的`token`是下一个`tokeninputs[i+1]`的概率。为了针对整个序列获得下一个`token`预测，我们可以简单的选择`output[-1]`中概率最大的那个token, 比如：
```python
vocab = ["all", "not", "heroes", "the", "wear", ".", "capes"]
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[0] =  [0.75    0.1     0.0       0.15    0.0   0.0    0.0  ]
# 在"not"给出的情况下，我们可以看到，(对于下一个token)模型预测"all"具有最高的概率

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[1] =  [0.0     0.0      0.8     0.1    0.0    0.0   0.1  ]
# 在序列["not", "all"]给出的情况下，(对于下一个token)模型预测"heroes"具有最高的概率

#              ["all", "not", "heroes", "the", "wear", ".", "capes"]
# output[-1] = [0.0     0.0     0.0     0.1     0.0    0.05  0.85  ]
# 在整个序列["not", "all", "heroes", "wear"]给出的情况下，(对于下一个token)模型预测"capes"具有最高的概率

# next_token_id = 6
next_token_id = np.argmax(output[-1]) 
# next_token = "capes"
next_token = vocab[next_token_id] 
```
将具有最高概率的`token`作为我们的预测，称为`greedy decoding`。在一个序列中预测下一个逻辑词(`logical word`)的任务被称之为**语言建模**。因此我们可以称`GPT`为语言模型。
#### 生成文本

##### 自回归

我们可以迭代地通过模型获取下一个`token`的预测，从而生成整个句子。这个过程是在预测未来的值（回归），并且将预测的值添加回输入中去（`auto`），这就是为什么你会看到GPT被描述为**自回归模型**。
```python
def generate(inputs, n_tokens_to_generate):
    for _ in range(n_tokens_to_generate): # 自回归的解码循环
        output = gpt(inputs) # 模型前向传递
        next_id = np.argmax(output[-1]) # 贪心采样
        inputs.append(int(next_id)) # 将预测添加回输入
    return inputs[len(inputs) - n_tokens_to_generate :]  # 只返回生成的ids

input_ids = [1, 0] # "not" "all"
output_ids = generate(input_ids, 3) # output_ids = [2, 4, 6]
output_tokens = [vocab[i] for i in output_ids] # "heroes" "wear" "capes"
```
##### 采样

我们可以通过对概率分布进行采样来替代贪心采样，从而为我们的生成引入一些随机性（`stochasticity`）。这样子，我们就可以基于同一个输入产生不同的输出句子啦。当我们结合更多的比如`top-k`，`top-p`和`temperature`这样的技巧的时候（这些技巧能够能更改采样的分布），我们输出的质量也会有很大的提高。这些技巧也引入了一些超参数，通过调整这些超参，我们可以获得不同的生成表现(`behaviors`)。比如提高`temperature`超参，我们的模型就会更加冒进，从而变得更有“创造力”。
```python
# 随机采样
inputs = [1, 0, 2, 4] # "not" "all" "heroes" "wear"
output = gpt(inputs)
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # hats
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # capes
np.random.choice(np.arange(vocab_size), p=output[-1]) # pants
```
#### 训练

与其它神经网络训练一样，针对特定的**损失函数**使用梯度下降训练`GPT`。对于`GPT`，我们使用语言建模任务的交叉熵损失：
```python
def lm_loss(inputs: list[int], params) -> float:
    # the labels y are just the input shifted 1 to the left
    #
    # inputs = [not,     all,   heros,   wear,   capes]
    #      x = [not,     all,   heroes,  wear]
    #      y = [all,  heroes,     wear,  capes]
    #
    # of course, we don't have a label for inputs[-1], so we exclude it from x
    # as such, for N inputs, we have N - 1 langauge modeling example pairs
    x, y = inputs[:-1], inputs[1:]

    # forward pass all the predicted next token probability distributions at each position
    output = gpt(x, params)

    # cross entropy loss we take the average over all N-1 examples
    loss = np.mean(-np.log(output[y]))

    return loss

# 一个极度简化的训练设置
def train(texts: list[list[str]], params) -> float:
    for text in texts:
        inputs = tokenizer.encode(text)
        # 计算语言建模损失
        loss = lm_loss(inputs, params)
        # 损失决定了梯度，我们可以通过反向传播计算梯度
        gradients = compute_gradients_via_backpropagation(loss, params)
        # 梯度来更新模型参数
        params = gradient_descent_update_step(gradients, params)
    return params
```
{% note warning %}
**请注意**，我们在这里并未使用明确的标注数据。取而代之的是，我们可以通过原始文本自身，产生大量的输入/标签对(`input/label pairs`)。这就是所谓的**自监督学习**。
{% endnote %}
**自监督学习的范式**，让我们能够海量扩充训练数据。我们只需要尽可能多的搞到大量的文本数据，然后将其丢入模型即可。比如，`GPT-3`就是基于来自互联网和书籍的`3000`亿`token`进行训练的：这里你就需要一个足够大的模型有能力去从这么大量的数据中学到内容，这就是为什么`GPT-3`模型拥有`1750`亿的参数，并且大概消耗了`100`万–`1000`万美元的计算费用进行训练。
{% asset_img gpt_1.png %}

**自监督训练**的步骤称之为**预训练**，而我们可以重复使用预训练模型权重来训练下游任务上的特定模型，比如对文本进行分类（分类某条推文是有害的还是无害的）。**预训练模型**有时也被称为**基础模型**(`foundation models`)。在下游任务上训练模型被称之为**微调**，由于模型权重已经预训练好了，已经能够理解语言了，那么我们需要做的就是针对特定的任务去微调这些权重。这个所谓“**在通用任务上预训练 + 特定任务上微调**”的策略就称之为**迁移学习**。

#### 提示（prompting）

本质上看，原始的`GPT`论文只是提供了用来迁移学习的`Transformer`模型的预训练。文章显示，一个`117M`的`GPT`预训练模型，在针对下游任务的标注数据上微调之后，它能够在各种`NLP`(`natural language processing`)任务上达到最优性能。一个`GPT`模型只要在足够多的数据上训练，只要模型拥有足够多的参数，那么不需要微调，模型本身就有能力执行各种任务。只要你对模型进行提示，运行自回归语言模型就会得到合适的响应。这就是所谓的**上下文学习**(`in-context learning`)，也就是说模型仅仅根据提示的内容，就能够执行各种任务。**上下文学习**可以是`zero shot`,`one shot`,或者`few shot`（`zero shot`表示我们直接拿着大模型就能用于我们的任务了；`one shot`表示我们需要提供给大模型关于我们特定任务的一个列子；`few shot`表示我们需要提供给大模型关于我们特定任务的几个例子；）。
{% asset_img gpt_2.png %}

基于提示内容生成文本也被称之为条件生成，因为我们的模型是基于特定的输入（条件）进行生成的。当然，`GPT`也不仅限于自然语言处理任务(`NLP`)。你可以将模型用于任何你想要的条件下。比如你可以将`GPT`变成一个聊天机器人(即：`ChatGPT`)，这里的条件就是你的对话历史。你也可以进一步条件化你的聊天机器人，通过提示词进行某种描述，限定其表现为某种行为（比如你可以提示：“你是个聊天机器人，请礼貌一点，请讲完整的句子，不要说有害的东西，等等”）。像这样条件化你的模型，你完全可以得到一个定制化私人助理机器人。但是这样的方式不一定很健壮，你仍然可以对你的模型进行越狱，然后让它表现失常。

#### 实现

```python
import numpy as np

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text

if __name__ == "__main__":
    import fire

    fire.Fire(main)
```
我们将以上代码拆解为四部分：
- `gpt2`函数是我们将要实现的`GPT`代码。你会注意到函数参数中除了`inputs`，还有其它的参数：
    - `wte, wpe, blocks, ln_f`这些都是模型的参数。
    - `n_head`是前向计算过程中需要的超参。
- `generate`函数是我们之前看到的自回归解码算法。为了简洁，我们使用贪心采样算法。`tqdm`是一个进度条库，它可以帮助我们随着每次生成一个`token`，可视化地观察解码过程。
- `main`函数主要处理：
    - 加载分词器(`encoder`)，模型权重（`params`），超参（`hparams`）。
    - 使用分词器将输入提示词编码为`token ID`。
    - 调用生成函数。
    - 将输出`ID`解码为字符串。
- `fire.Fire(main)`将我们的源文件转成一个命令行应用，然后就可以像这样运行我们的代码了：`python gpt2.py "some prompt here"`

##### 编码器

我们的`encoder`使用的是`GPT-2`中使用的`BPE`分词器：
```python
ids = encoder.encode("Not all heroes wear capes.")
ids

# [3673, 477, 10281, 5806, 1451, 274, 13]

encoder.decode(ids)

# "Not all heroes wear capes."

# 使用分词器的词汇表(存储于encoder.decoder)，我们可以看看实际的token到底长啥样：
[encoder.decoder[i] for i in ids]

# ['Not', 'Ġall', 'Ġheroes', 'Ġwear', 'Ġcap', 'es', '.']
```
{% note warning %}
**注意**，有的时候我们的`token`是单词（比如：`Not`），有的时候虽然也是单词，但是可能会有一个空格在它前面（比如`Ġall, Ġ`代表一个空格），有时候是一个单词的一部分（比如：`capes`被分隔为`Ġcap`和`es`），还有可能它就是标点符号（比如：`.`）。
{% endnote %}
`BPE`的一个好处是它可以编码任意字符串。如果遇到了某些没有在词汇表里显示的字符串，那么`BPE`就会将其分割为它能够理解的子串：
```python
[encoder.decoder[i] for i in encoder.encode("zjqfl")]

# ['z', 'j', 'q', 'fl']

# 我们还可以检查一下词汇表的大小
len(encoder.decoder)

# 50257
```
词汇表以及决定字符串如何分解的**字节对组合**(`byte-pair merges`)，是通过训练分词器获得的。当我们加载分词器，就会从一些文件加载已经训练好的词汇表和字节对组合，这些文件在我们运行`load_encoder_hparams_and_params`的时候，随着模型文件被一起下载了。你可以查看`models/124M/encoder.json`(**词汇表**)和`models/124M/vocab.bpe`(**字节对组合**)。
##### 超参数

`hparams`是一个字典，这个字典包含着我们模型的超参：
```python
hparams

# {
#   "n_vocab": 50257, # number of tokens in our vocabulary
#   "n_ctx": 1024, # maximum possible sequence length of the input
#   "n_embd": 768, # embedding dimension (determines the "width" of the network)
#   "n_head": 12, # number of attention heads (n_embd must be divisible by n_head)
#   "n_layer": 12 # number of layers (determines the "depth" of the network)
# }
```
我们将在代码的注释中使用这些符号来表示各种的大小维度等。我们还会使用`n_seq`来表示输入序列的长度(即：`n_seq = len(inputs)`)。
##### 参数

`params`是一个嵌套的`json`字典，该字典具有模型训练好的权重。`json`的叶子节点是`NumPy`数组。如果我们打印`params`，用他们的形状去表示数组，我们可以得到：
```python
print(shape_tree(params))

# {
#     "wpe": [1024, 768],
#     "wte": [50257, 768],
#     "ln_f": {"b": [768], "g": [768]},
#     "blocks": [
#         {
#             "attn": {
#                 "c_attn": {"b": [2304], "w": [768, 2304]},
#                 "c_proj": {"b": [768], "w": [768, 768]},
#             },
#             "ln_1": {"b": [768], "g": [768]},
#             "ln_2": {"b": [768], "g": [768]},
#             "mlp": {
#                 "c_fc": {"b": [3072], "w": [768, 3072]},
#                 "c_proj": {"b": [768], "w": [3072, 768]},
#             },
#         },
#         ... # repeat for n_layers
#     ]
# }
```
在实现`GPT`的过程中，你可能会需要参考这个字典来确认权重的形状。为了一致性，我们将会使代码中的变量名和字典的键值保持对齐。

##### 基础层

###### GELU

`GPT-2`的非线性（激活函数）选择是`GELU`（高斯误差线性单元），这是一种类似`ReLU`的激活函数：
{% asset_img gpt_3.png %}

它的函数函数如下：
```python
def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
```
和`ReLU`类似，`GELU`也对输入进行逐元素操作：
```python
gelu(np.array([[1, 2], [-2, 0.5]]))

# array([[ 0.84119,  1.9546 ],[-0.0454 ,  0.34571]])
```
###### Softmax

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```
{% mathjax '{"conversion":{"em":14}}' %}
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j^{e^{x_j}}}}
{% endmathjax %}
这里我们使用了`max(x)`技巧来保持数值稳定性。`softmax`用来将一组实数（{% mathjax %}-\infty{% endmathjax %}至{% mathjax %}\infty{% endmathjax %}之间）转换为概率（{% mathjax %}0{% endmathjax %}至{% mathjax %}1{% endmathjax %}之间，其求和为`1`）。我们将`softmax`作用于输入的最末轴上。
```python
x = softmax(np.array([[2, 100], [-5, 0]]))
x

# array([[0.00034, 0.99966],[0.26894, 0.73106]])

x.sum(axis=-1)

# array([1., 1.])
```
###### 层归一化

层归一化将数值标准化为均值为`0`，方差为`1`的值：
{% mathjax '{"conversion":{"em":14}}' %}
\text{LayerNorm}(x) = \gamma\cdot \frac{x - \mu}{\sqrt{\sigma^2}} + \beta
{% endmathjax %}
其中{% mathjax %}\mu{% endmathjax %}是{% mathjax %}x{% endmathjax %}的均值，{% mathjax %}\sigma^2{% endmathjax %}为{% mathjax %}x{% endmathjax %}的方差，{% mathjax %}\gamma{% endmathjax %}和{% mathjax %}\beta{% endmathjax %}为可学习的参数。
```python
def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params
```
层归一化确保每层的输入总是在一个一致的范围里，而这将为训练过程的加速和稳定提供支持。与批归一化类似，归一化之后的输出通过两个可学习参数{% mathjax %}\gamma{% endmathjax %}和{% mathjax %}\beta{% endmathjax %}进行缩放和偏移。分母中的小`epsilon`项用来避免计算中的分母为零错误。
###### 线性（变换）

这里是标准的矩阵乘法+偏置：
```python
def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b
```
**线性层**也通常被认为是**投影**操作（因为它们将一个向量空间投影到另一个向量空间）。
```python
x = np.random.normal(size=(64, 784)) # input dim = 784, batch/sequence dim = 64
w = np.random.normal(size=(784, 10)) # output dim = 10
b = np.random.normal(size=(10,))
x.shape # shape before linear projection

# (64, 784)

linear(x, w, b).shape # shape after linear projection

# (64, 10)
```

#### GPT架构

{% asset_img gpt_4.png %}

但它仅仅使用了解码器层（上图中的右边部分）：
{% asset_img gpt_5.png "GPT架构" %}

从宏观的角度来看，`GPT`架构有三个部分组成：
- 文本 `+` 位置嵌入(`positional embeddings`)。
- 基于`Transformer`的解码器层(`decoder stack`)。
- 投影为词汇表(`projection to vocab`)的步骤。

代码层面的话，就像这样：
```python
def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
```
##### 嵌入层

###### Token 嵌入

对于神经网络而言，`token ID`本身并不是一个好的表示。第一，`token ID`的相对大小会传递错误的信息（比如，在我们的词汇表中，如果`Apple = 5，Table=10`，那就意味着`2 * Table = Apple`？显然不对）。其二，单个的数也没有足够的维度喂给神经网络，也就是说单个的数字包含的特征信息不够丰富。为了解决这些限制，我们将利用词向量，即通过一个学习到的嵌入矩阵：
```python
wte[inputs] # [n_seq] -> [n_seq, n_embd]
```
`wte`是一个`[n_vocab, n_emdb]`的矩阵。这就像一个查找表，矩阵中的第{% mathjax %}i{% endmathjax %}行对应我们的词汇表中的第{% mathjax %}i{% endmathjax %}个`token`的向量表示（学出来的）。`wte[inputs]`使用了`integer array indexing`来检索我们输入中每个`token`所对应的向量。就像神经网络中的其他参数，`wte`是可学习的。也就是说，在训练开始的时候它是随机初始化的，然后随着训练的进行，通过梯度下降不断更新。
###### 位置嵌入（Positional Embeddings）

单纯的`Transformer`架构的一个古怪地方在于它并不考虑位置。当我们随机打乱输入位置顺序的时候，输出可以保持不变（输入的顺序对输出并未产生影响）。可是词的顺序当然是语言中重要的部分啊，因此我们需要使用某些方式将位置信息编码进我们的输入。为了这个目标，我们可以使用另一个学习到的嵌入矩阵：
```python
wpe[range(len(inputs))] # [n_seq] -> [n_seq, n_embd]
```
`wpe`是一个`[n_ctx, n_emdb]`矩阵。矩阵的第{% mathjax %}i{% endmathjax %}行包含一个编码输入中第{% mathjax %}i{% endmathjax %}个位置信息的向量。与`wte`类似，这个矩阵也是通过梯度下降来学习到的。需要注意的是，这将限制模型的最大序列长度为`n_ctx`。也就是说必须满足`len(inputs) <= n_ctx`。
###### 组合

现在我们可以将`token`嵌入与位置嵌入联合为一个组合嵌入，这个嵌入将`token`信息和位置信息都编码进来。
```python
# token + positional embeddings
x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

# x[i] represents the word embedding for the ith word + the positional embedding for the ith position
```
##### 解码层

我们将刚才的嵌入通过一连串的`n_layertransformer`解码器模块。一方面，堆叠更多的层让我们可以控制到底我们的网络有多“深”。以`GPT-3`为例，其高达`96`层。另一方面，选择一个更大的`n_embd`值，让我们可以控制网络有多“宽”（还是以`GPT-3`为例，它使用的嵌入大小为`12288`）。
```python
# forward pass through n_layer transformer blocks
for block in blocks:
    x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]
```
##### 投影为词汇表(projection to vocab)

在最后的步骤中，我们将`Transformer`最后一个结构块的输入投影为字符表的一个概率分布：
```python
# projection to vocab
x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]
```
- 在进行投影操作之前，我们先将x通过**最后的层归一化层**。这是`GPT-2`架构所特有的（并没有出现在`GPT`原始论文和`Transformer`论文中）。
- 我们复用了嵌入矩阵`wte`进行投影操作。其它的`GPT`实现当然可以选择使用另外学习到的权重矩阵进行投影，但是权重矩阵共享具有以下一些优势：
    - 你可以节省一些参数。
    - 考虑到这个矩阵作用于转换到词与来自于词的两种转换，理论上，相对于分别使用两个矩阵来做这件事，使用同一个矩阵将学到更为丰富的表征。
- 在最后，我们并未使用`softmax`，因此我们的输出是`logits`而不是`0-1`之间的概率。这样做的理由是：
    - `softmax`是单调的，因此对于贪心采样而言，`np.argmax(logits)`和`np.argmax(softmax(logits))`是等价的，因此使用`softmax`就变得多此一举。
    - `softmax`是不可逆的，这意味着我们总是可以通过`softmax`将`logits`变为`probabilities`，但不能从`probabilities`变为`softmax`，为了让灵活性最大，我们选择直接输出`logits`。
    - 数值稳定性的考量。比如计算交叉熵损失的时候，相对于`log_softmax(logits)，log(softmax(logits))`的数值稳定性就差。

投影为词汇表的过程有时候也被称之为**语言建模头**(`language modeling head`)。这里的“头”是什么意思呢？你的`GPT`一旦被预训练完毕，那么你可以通过更换其他投影操作的语言建模头，比如你可以将其更换为**分类头**，从而在一些分类任务上微调你的模型（让其完成分类任务）。因此你的模型可以拥有多种头。

##### 解码器模块

`Transformer`解码器模块由两个子层组成：
- 多头因果自注意力(`Multi-head causal self attention`)
- 逐位置前馈神经网络(`Position-wise feed forward neural network`)

```python
def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
```
每个子层都在输入上使用了层归一化，也使用了残差连接（即将子层的输入直接连接到子层的输出）。
- **多头因果自注意力机制**便于输入之间的通信。在网络的其它地方，模型是不允许输入相互“看到”彼此的。嵌入层、逐位置前馈网络、层归一化以及投影到词汇表的操作，都是逐位置对我们的输入进行的。建模输入之间的关系完全由注意力机制来处理。
- **逐位置前馈神经网络**只是一个常规的两层全连接神经网络。它只是为我们的模型增加一些可学习的参数，以促进学习过程。
- 在原始的`Transformer`论文中，层归一化被放置在输出层`layer_norm(x + sublayer(x))`上，而我们在这里为了匹配`GPT-2`，将层归一化放置在输入`x + sublayer(layer_norm(x))`上。这被称为**预归一化**，并且已被证明在改善`Transformer`的性能方面尤为重要。
- **残差连接**这这里有几个不同的目的：1.使得深度神经网络（即层数非常多的神经网络）更容易进行优化。其思想是为梯度提供“捷径”，使得梯度更容易地回传到网络的初始的层，从而更容易进行优化；2.如果没有残差连接的话，加深模型层数会导致性能下降（可能是因为梯度很难在没有损失信息的情况下回传到整个深层网络中）。残差连接似乎可以为更深层的网络提供一些精度提升；3.可以帮助解决梯度消失/爆炸的问题。

###### 逐位置前馈网络

**逐位置前馈网络**(`Position-wise Feed Forward Network`)是一个简单的两层的多层感知器：
```python
def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x
```
这里没有什么特别的技巧，我们只是将`n_embd`投影到一个更高的维度`4*n_embd`，然后再将其投影回`n_embd`。回忆一下我们的`params`字典，我们的`mlp`参数如下：
```bash
"mlp": {
    "c_fc":   {"b": [4*n_embd], "w": [n_embd, 4*n_embd]},
    "c_proj": {"b": [n_embd], "w": [4*n_embd, n_embd]},
}
```
###### 多头因果自注意力

这一层可能是理解`Transformer`最困难的部分。因此我们通过分别解释“**多头因果自注意力**”的每个词，一步步理解“**多头因果自注意力**”：
- 注意力（`Attention`）。
- 自身(`Self`)。
- 因果(`Causal`)。
- 多头(`Multi-Head`)。

**注意力**
我从头开始推导了原始`Transformer`论文中提出的缩放点积方程：
{% mathjax '{"conversion":{"em":14}}' %}
\text{attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
{% endmathjax %}
注意力实现：
```python
def attention(q, k, v):  # [n_q, d_k], [n_k, d_k], [n_k, d_v] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1])) @ v
```
**自身**
当`q, k`和`v`来自同一来源时，我们就是在执行自注意力（即让我们的输入序列自我关注）：
```python
def self_attention(x): # [n_seq, n_embd] -> [n_seq, n_embd]
    return attention(q=x, k=x, v=x)
```
例如，如果我们的输入是`“Jay went to the store, he bought 10 apples.”`，我们让单词`“he”`关注所有其它单词，包括`“Jay”`，这意味着模型可以学习到`“he”`指的是`“Jay”`。

我们可以通过为`q、k、v`和注意力输出引入投影来增强自注意力：
```python
def self_attention(x, w_k, w_q, w_v, w_proj): # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projections
    q = x @ w_k # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd]
    k = x @ w_q # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd]
    v = x @ w_v # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd]

    # perform self attention
    x = attention(q, k, v) # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = x @ w_proj # [n_seq, n_embd] @ [n_embd, n_embd] -> [n_seq, n_embd]

    return x
```
这使得我们的模型为`q, k, v`学到一个最好的映射，以帮助注意力区分输入之间的关系。如果我们将`w_q、w_k`和`w_v`组合成一个单独的矩阵`w_fc`，执行投影操作，然后拆分结果，我们就可以将矩阵乘法的数量从`4`个减少到`2`个：
```python
def self_attention(x, w_fc, w_proj): # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projections
    x = x @ w_fc # [n_seq, n_embd] @ [n_embd, 3*n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # perform self attention
    x = attention(q, k, v) # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = x @ w_proj # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x
```
这样会更加高效，因为现代加速器（如`GPU`）可以更好地利用一个大的矩阵乘法，而不是顺序执行`3`个独立的小矩阵乘法。最后，我们添加偏置向量以匹配`GPT-2`的实现，然后使用我们的`linear`函数，并将参数重命名以匹配我们的`params`字典：
```python
def self_attention(x, c_attn, c_proj): # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projections
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # perform self attention
    x = attention(q, k, v) # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x
```
回忆一下，从我们的`params`字典中可知，`attn`参数类似：
```bash
"attn": {
    "c_attn": {"b": [3*n_embd], "w": [n_embd, 3*n_embd]},
    "c_proj": {"b": [n_embd], "w": [n_embd, n_embd]},
}
```
**因果**
我们当前的自注意力设置存在一个问题，就是我们的输入能够“看到”未来的信息！比如，如果我们的输入是`[“not”, “all”, “heroes”, “wear”, “capes”]`，在自注意力中，`“wear”`可以看到`“capes”`。这意味着`“wear”`的输出概率将会受到偏差，因为模型已经知道正确的答案是`“capes”`。这是不好的，因为我们的模型会从中学习到，输入{% mathjax %}i{% endmathjax %}的正确答案可以从输入{% mathjax %}i+1{% endmathjax %}中获取。为了防止这种情况发生，我们需要修改注意力矩阵，以隐藏或屏蔽我们的输入，使其无法看到未来的信息。例如，假设我们的注意力矩阵如下所示：
```bash
        not    all   heroes wear   capes
   not 0.116  0.159  0.055  0.226  0.443
   all 0.180  0.397  0.142  0.106  0.175
heroes 0.156  0.453  0.028  0.129  0.234
  wear 0.499  0.055  0.133  0.017  0.295
 capes 0.089  0.290  0.240  0.228  0.153

```
这里每一行对应一个查询(`query`)，每一列对应一个键值(`key`)。在这个例子中，查看`“wear”`对应的行，可以看到它在最后一列以`0.295`的权重与`“capes”`相关。为了防止这种情况发生，我们要将这项设为`0.0`:
```bash
        not    all    heroes wear   capes
   not 0.116  0.159  0.055  0.226  0.443
   all 0.180  0.397  0.142  0.106  0.175
heroes 0.156  0.453  0.028  0.129  0.234
  wear 0.499  0.055  0.133  0.017  0.
 capes 0.089  0.290  0.240  0.228  0.153
```
通常，为了防止输入中的所有查询看到未来信息，我们将所有满足{% mathjax %}j > i{% endmathjax %}的位置{% mathjax %}i,j{% endmathjax %}都设置为`0`：
```bash
         not    all    heroes wear   capes
   not 0.116  0.     0.     0.     0.
   all 0.180  0.397  0.     0.     0.
heroes 0.156  0.453  0.028  0.     0.
  wear 0.499  0.055  0.133  0.017  0.
 capes 0.089  0.290  0.240  0.228  0.153
```
我们将这称为**掩码**(`masking`)。掩码方法的一个问题是我们的行不再加起来为`1`（因为我们在使用`softmax`后才将它们设为`0`）。为了确保我们的行仍然加起来为`1`，我们需要在使用`softmax`之前先修改注意力矩阵。
```python
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v
```
其中`mask`表示矩阵`（n_seq=5）`：
```bash
0 -1e10 -1e10 -1e10 -1e10
0   0   -1e10 -1e10 -1e10
0   0     0   -1e10 -1e10
0   0     0     0   -1e10
0   0     0     0     0
```
我们用`-1e10`替换`-np.inf`， 因为`-np.inf`会导致`nans`错误。添加`mask`到我们的注意力矩阵中，而不是明确设置值为`-1e10`，是因为在实际操作中，任何数加上`-inf`还是`-inf`。我们可以在`NumPy`中通过`(1 - np.tri(n_seq)) * -1e10`来计算`mask`矩阵。我们得到：
```python
def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def causal_self_attention(x, c_attn, c_proj): # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projections
    x = linear(x, **c_attn) # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    q, k, v = np.split(x, 3, axis=-1) # [n_seq, 3*n_embd] -> 3 of [n_seq, n_embd]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0]), dtype=x.dtype) * -1e10  # [n_seq, n_seq]

    # perform causal self attention
    x = attention(q, k, v, causal_mask) # [n_seq, n_embd] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj) # [n_seq, n_embd] @ [n_embd, n_embd] = [n_seq, n_embd]

    return x
```
**多头**
我们可以进一步改进我们的实现，通过进行`n_head`个独立的注意力计算，将我们的查询(`queries`)，键(`keys`)和值(`values`)拆分到多个头(`heads`)里去：
```python
def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads(拆分q， k， v到n_head个头)
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head(为每个头计算注意力)
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads(合并每个头的输出)
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x
```
这里添加了三步:
- 拆分`q， k， v`到`n_head`个头。
- 为每个头计算注意力。
- 合并每个头的输出。

{% note warning %}
**注意**，这样可以将每个注意力计算的维度从`n_embd`减少到`n_embd/n_head`。这是一个权衡。对于缩减了的维度，我们的模型在通过注意力建模关系时获得了额外的子空间。例如，也许一个注意力头负责将代词与代词所指的人联系起来；也许另一个注意力头负责通过句号将句子分组；另一个则可能只是识别哪些单词是实体，哪些不是。虽然这可能也只是另一个神经网络黑盒而已。我们编写的代码按顺序循环执行每个头的注意力计算（每次一个），当然这并不是很高效。在实践中，你会希望并行处理这些计算。当然在本文中考虑到简洁性，我们将保持这种顺序执行。
{% endnote %}

#### 完整代码

```python
import numpy as np

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params

def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

    return x

def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v

def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]

def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)
```
#### GPU/TPU支持

将`NumPy`替换为`JAX`：
```python
import jax.numpy as np

# 计算梯度
def lm_loss(params, inputs, n_head) -> float:
    x, y = inputs[:-1], inputs[1:]
    output = gpt2(x, **params, n_head=n_head)
    loss = np.mean(-np.log(output[y]))
    return loss

grads = jax.grad(lm_loss)(params, inputs, n_head)

# gpt2函数批量化
gpt2_batched = jax.vmap(gpt2, in_axes=[0, None, None, None, None, None])
gpt2_batched(batched_inputs) # [batch, seq_len] -> [batch, seq_len, vocab]
```
#### 优化

我们的实现相当低效。除了支持`GPU`和批处理之外，最快且最有效的优化可能是实现一个键值缓存。此外，我们顺序地实现了注意力头计算，而实际上我们应该使用并行计算等。

#### 训练

训练`GPT`对于神经网络来说是非常标准的行为（针对损失函数进行梯度下降）。当然，在训练`GPT`时你还需要使用一堆常规的技巧（使用`Adam`优化器，找到最佳的学习率，通过`dropout`和`/`或权重衰减进行正则化，使用学习率规划器，使用正确的权重初始化，进行分批处理等等）。而训练一个好的`GPT`模型的**真正秘诀在于能够扩展数据和模型，这也是真正的挑战所在**。

为了扩展数据量，您需要拥有大规模、高质量、多样化的文本语料库。
- **大规模**意味着拥有数十亿的`token`（数百万`GB`的数据）。例如可以查看`The Pile`，这是一个用于大型语言模型的开源预训练数据集。
- **高质量**意味着需要过滤掉重复的示例、未格式化的文本、不连贯的文本、垃圾文本等等。
- **多样性**意味着序列长度变化大，涵盖了许多不同的主题，来自不同的来源，具有不同的观点等等。当然，如果数据中存在任何偏见，它将反映在模型中，因此您需要谨慎处理。

将模型扩展到数十亿个参数需要超级大量的工程（金钱）可以使用`NVIDIA`的`Megatron Framework, Cohere`的训练框架, `Google`的`PALM`, 开源的`mesh-transformer-jax`（用于训练`EleutherAI`的开源模型）

#### 评估

老实说，这是一个非常困难的问题。`HELM`是一个相当全面且不错的起点，但你应该始终对基准测试和评估指标保持怀疑的态度。

#### 停止生成

我们当前的实现需要事先指定要生成的确切`token`数量。这不是一个很好的方法，因为我们生成的文本可能会太长、太短或在句子中间截断。为了解决这个问题，我们可以引入一个特殊的句子结束（`EOS`）`token`。在预训练期间，我们在输入的末尾附加`EOS token`（比如，`tokens = ["not", "all", "heroes", "wear", "capes", ".", "<|EOS|>"]`）。在生成过程中，我们只需要在遇到`EOS token`时停止（或者达到最大序列长度）：
```python
def generate(inputs, eos_id, max_seq_len):
	prompt_len = len(inputs)
	while inputs[-1] != eos_id and len(inputs) < max_seq_len:
        output = gpt(inputs)
        next_id = np.argmax(output[-1])
        inputs.append(int(next_id))
    return inputs[prompt_len:]
```
`GPT-2`没有使用`EOS token`进行预训练，因此我们无法在我们的代码中使用这种方法，但是现在大多数`LLMs`都已经使用`EOS token`了。

#### 无条件生成

使用我们的模型生成文本需要对其提供提示条件。但是我们也可以让模型执行无条件生成，即模型在没有任何输入提示的情况下生成文本。这是通过在预训练期间在输入开头加上一个特殊的句子开头（`BOS`）`token`来实现的（例如 `tokens = ["<|BOS|>", "not", "all", "heroes", "wear", "capes", "."]`）。要进行无条件文本生成的话，我们就输入一个仅包含`BOS token`的列表：
```python
def generate_unconditioned(bos_id, n_tokens_to_generate):
	inputs = [bos_id]
    for _ in range(n_tokens_to_generate):
        output = gpt(inputs)
        next_id = np.argmax(output[-1])
        inputs.append(int(next_id))
    return inputs[1:]
```
`GPT-2`的预训练是带有`BOS token`的（不过它有一个令人困惑的名字`<|endoftext|>`），因此在我们的实现中要运行无条件生成的话，只需要简单地将这行代码更改为：
```python
input_ids = encoder.encode(prompt) if prompt else [encoder.encoder["<|endoftext|>"]]
```
由于我们使用的是贪心采样，所以输出结果不是很好（重复的内容较多），且每次运行代码的输出结果都是确定的。为了获得更高质量的、不确定性更大的生成结果，我们需要直接从概率分布中进行采样（最好在使用`top-p`之类的方法后进行采样）。无条件生成不是特别有用，但它是演示GPT能力的一种有趣方式。

#### 微调

我们在训练部分简要介绍了微调。回想一下，微调是指我们复用预训练的权重，对模型在某些下游任务上进行训练。我们称这个过程为**迁移学习**。理论上，我们可以使用零样本或少样本提示来让模型完成我们的任务，但是如果您可以访问一个标注的数据集，对`GPT`进行微调将会产生更好的结果（这些结果可以在获得更多数据和更高质量的数据时进行扩展）。
##### 分类微调

在分类微调中，我们会给模型一些文本，并要求它预测它属于哪个类别。以`IMDB`数据集为例，它包含着电影评论，将电影评为好或坏：
```bash
--- Example 1 ---
Text: I wouldn't rent this one even on dollar rental night.
Label: Bad
--- Example 2 ---
Text: I don't know why I like this movie so well, but I never get tired of watching it.
Label: Good
--- Example 3 ---
...
```
为了微调我们的模型，我们需要用分类头替换语言建模头，将其应用于最后一个`token`的输出：
```python
def gpt2(inputs, wte, wpe, blocks, ln_f, cls_head, n_head):
    x = wte[inputs] + wpe[range(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    x = layer_norm(x, **ln_f)

	# project to n_classes [n_embd] @ [n_embd, n_classes] -> [n_classes]
    return x[-1] @ cls_head
```
这里我们只使用最后一个`token`的输出`x[-1]`，因为我们只需要为整个输入产生一个单一的概率分布，而不是像语言模型一样产生`n_seq`个分布。我们特别选择最后一个`token`（而不是第一个`token`或所有`token`的组合），因为最后一个`token`是唯一允许关注整个序列的`token`，因此它具有关于整个输入文本的信息。同往常一样，我们根据交叉熵损失进行优化：
```python
def singe_example_loss_fn(inputs: list[int], label: int, params) -> float:
    logits = gpt(inputs, **params)
    probs = softmax(logits)
    loss = -np.log(probs[label]) # cross entropy loss
    return loss
```
我们还可以执行多标签分类（即一个样本可以属于多个类别，而不仅仅是一个类别），这可以通过使用`sigmoid`替代`softmax`并针对每个类别采用二分交叉熵损失。
##### 生成式微调

有些任务无法被简单地认为是分类，如摘要的任务。我们可以通过对输入和标签拼接进行语言建模，从而实现这类任务的微调。例如，下面就是一个摘要训练样本的示例：
```bash
--- Article ---
This is an article I would like to summarize.
--- Summary ---
This is the summary.
```
我们就像预训练时那样训练这个模型（根据语言建模的损失进行优化）。在预测时，我们将直到`"--- Summary ---"`的输入喂给模型，然后执行自回归语言建模以生成摘要。定界符"`--- Article ---"`和`"--- Summary ---"`的选择是任意的。如何选择文本格式由您决定，只要在训练和推断中保持一致即可。请注意，其实我们也可以将分类任务表述为生成任务（以`IMDB`为例）：
```bash
--- Text ---
I wouldn't rent this one even on dollar rental night.
--- Label ---
Bad
```
然而，这种方法的表现很可能会比直接进行分类微调要差（损失函数包括对整个序列进行语言建模，而不仅仅是对最终预测的输出进行建模，因此与预测有关的损失将被稀释）。
##### 指令微调

目前大多数最先进的大型语言模型在预训练后还需要经过一个额外的指令微调步骤。在这个步骤中，模型在成千上万个由人工标注的指令提示+补全对上进行微调（生成式）。指令微调也可以称为监督式微调，因为数据是人工标记的（即有监督的）。那指令微调的好处是什么呢？虽然在预测维基百科文章中的下一个词时，模型在续写句子方面表现得很好，但它并不擅长遵循说明、进行对话或对文件进行摘要（这些是我们希望`GPT`能够做到的事情）。在人类标记的指令 + 完成对中微调它们是教育模型如何变得更有用，并使它们更容易交互的一种方法。我们将其称为`AI`对齐(`AI alignment`)，因为我们需要模型以我们想要的方式做事和表现。对齐是一个活跃的研究领域，它不仅仅只包括遵循说明（还涉及偏见、安全、意图等）的问题。那么这些指令数据到底是什么样子的呢？`Google`的`FLAN`模型是在多个学术的自然语言处理数据集（这些数据集已经被人工标注）上进行训练的：
{% asset_img gpt_6.png %}

`OpenAI`的`InstructGPT`则使用了从其`API`中收集的提示进行训练。然后他们雇佣工人为这些提示编写补全。下面是这些数据的详细信息：
{% asset_img gpt_7.png %}

##### 参数高效微调（Parameter Efficient Fine-tuning）

当我们在上面的部分讨论微调时，我们是在更新模型的所有参数。虽然这可以获得最佳性能，但成本非常高，无论是在计算方面（需要经过整个模型进行反向传播），还是在存储方面（对于每个微调的模型，您需要存储一份全新的参数副本）。最简单的解决方法是**只更新模型头部并冻结模型的其它部分**。虽然这样做可以加速训练并大大减少新参数的数量，但其表现并不好，因为某种意义上我们损失了深度学习中的深度。相反，我们可以**选择性地冻结特定层**（例如冻结除了最后四层外的所有层，或每隔一层进行冻结，或冻结除多头注意力参数外的所有参数），那么这将有助于恢复深度。这种方法的性能要好得多，但我们也变得不那么参数高效(`parameter efficient`)，同时也失去了一些训练速度的优势。除此之外，我们还可以利用**参数高效微调**(`Parameter Efficient Fine-tuning`)方法。这仍然是一个活跃的研究领域。

我们可以看看`Adapters`论文。在这种方法中，我们在`Transformer`模块的`FFN`和`MHA`层后添加了一个额外的`“adapter”`层。这里的`adapter`层只是一个简单的两层全连接神经网络，其中输入和输出维度是`n_embd`，而隐藏维度小于`n_embd`。适配器方法中，隐藏层的大小是一个我们可以设置的超参数，这使我们能够在参数和性能之间进行权衡。该论文表明，对于`BERT`模型，使用这种方法可以将训练参数数量降低到`2％`，而与完全微调相比仅有少量的性能下降(`<1%`)。
{% asset_img gpt_8.png %}
