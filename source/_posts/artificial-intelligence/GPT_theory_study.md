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
