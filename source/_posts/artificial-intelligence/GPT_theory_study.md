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
词汇表以及决定字符串如何分解的**字节对组合**(`byte-pair merges`)，是通过训练分词器获得的。当我们加载分词器，就会从一些文件加载已经训练好的词汇表和字节对组合，这些文件在我们运行**load_encoder_hparams_and_params**的时候，随着模型文件被一起下载了。你可以查看**models/124M/encoder.json**(**词汇表**)和**models/124M/vocab.bpe**(**字节对组合**)。
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

##### GPT架构

{% asset_img gpt_4.png %}

但它仅仅使用了解码器层（上图中的右边部分）：
{% asset_img gpt_5.png "GPT架构" %}