---
title: 文本生成（KerasNLP）
date: 2024-04-15 14:36:11
tags:
  - AI
categories:
  - 人工智能
---

大语言模型非常流行。而这个大语言模型的核心是预测句子中的下一个单词或标记，这通常被称为`COCO-LM`预训练。大语言模型构建起来很复杂，而且从头开始训练的成本很高，幸运的是有经过预先训练的`LLM`可供使用。`KerasNLP`提供了大量预训练检查点，从而可以实验`SOTA`模型，而无需自行训练。例如，你可以通过`from_preset`方法调用`GPT2CausalLM`加载`GPT-2`模型，除了`GPT-2`模型之外，还有许多其它预训练模型，例如`OPT、ALBERT、RoBeRTa`等。
<!-- more -->
```python
from keras_nlp.models import {GPT2CausalLM, GPT2CausalLMPreprocessor}

preprocessor = GPT2CausalLMPreprocessor.from_preset('gpt2_base_en',sequente_length=128,)
model = GPT2CausalLM.from_preset('gpt2_base_en',preprocessor=preprocessor)
model.compile(...)
model.fit(cnn_dailymail_dataset)

model.generate('Snowfall in Buffalo',max_length=40,)
```
现在你可以调用`generate`方法来生成文本。生成文本的质量还算不错，但我们可以通过微调来改进它。但在进行微调之前，让我们看一下整体架构。与我们上次讨论的`BERT`分类器类似，`GPT2CausalLM`模型也有一个预处理器、分词器和主干，所有这些都可以通过简单的`from_preset`方法轻松加载。为了进行微调我们将使用`Reddit TIFU`数据集，以便输出遵循`Reddit`的写作风格。
{% asset_img tg_1.png %}

这是训练数据的实例：
```python
import os
import keras_nlp
import keras
import tensorflow as tf
import time
import tensorflow_datasets as tfds

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"
keras.mixed_precision.set_global_policy("mixed_float16")

reddit_ds = tfds.load("reddit_tifu", split="train", as_supervised=True)
for document, title in reddit_ds:
    print(document.numpy())
    print(title.numpy())
    break

train_ds = (
    reddit_ds.map(lambda document, _: document)
    .batch(32)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

train_ds = train_ds.take(500)
num_epochs = 1

# Linearly decaying learning rate.
learning_rate = keras.optimizers.schedules.PolynomialDecay(5e-5,decay_steps=train_ds.cardinality() * num_epochs,end_learning_rate=0.0,)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(optimizer=keras.optimizers.Adam(learning_rate),loss=loss,weighted_metrics=["accuracy"],)

gpt2_lm.fit(train_ds, epochs=num_epochs)

start = time.time()

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

end = time.time()
print(f"TOTAL TIME ELAPSED: {end - start:.2f}s")

# 500/500 ━━━━━━━━━━━━━━━━━━━━ 75s 120ms/step - accuracy: 0.3189 - loss: 3.3653
# so i go to the
# TOTAL TIME ELAPSED: 21.13s
```
由于我们正在语言模型中执行下一个单词的预测，因此我们只需要此处的文档特征。接下来，我们定义自定义学习率。并使用`fit`方法开始微调，这需要相当多的时间和`GPU`内存。但完成后生成的文本会更接近`Reddit`的写作风格，生成的长度也更接近我们在训练中预设的长度。您可以做另一件事是：你可以将模型转换为`TensorFlow Lite`，并在`Android`设备上运行。`KerasNLP`提供了多种采样方法，例如贪婪搜索、`Top K`和`BEAM`搜索。编译模型时，你可以通过流轻松设置采样器，默认情况下，`GPT-2`模型使用`Top K`采样，或者你可以传入采样器实例。
```python
# Use a string identifier.
gpt2_lm.compile(sampler="top_k")
output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

# 自定义采样器实例
# Use a `Sampler` instance. `GreedySampler` tends to repeat itself,
greedy_sampler = keras_nlp.samplers.GreedySampler()
gpt2_lm.compile(sampler=greedy_sampler)

output = gpt2_lm.generate("I like basketball", max_length=200)
print("\nGPT-2 output:")
print(output)

# GPT-2 output:
# I like basketball, and this is a pretty good one. 
# so i was playing basketball at my local high school, and i was playing with my friends. 
```
我们还可以在中文数据集上微调`GPT2`。如何在中文诗歌数据集上微调`GPT2`以教授我们的模型成为诗人！由于`GPT2`使用**字节对编码器**，并且原始预训练数据集包含一些汉字，因此我们可以使用原始词汇对中文数据集进行微调。从`json`文件加载文本。我们仅将《全唐诗》用于演示：
```bash
!# Load chinese poetry dataset.
!git clone https://github.com/chinese-poetry/chinese-poetry.git
```
```python
import os
import json

poem_collection = []
for file in os.listdir("chinese-poetry/全唐诗"):
    if ".json" not in file or "poet" not in file:
        continue
    full_filename = "%s/%s" % ("chinese-poetry/全唐诗", file)
    with open(full_filename, "r") as f:
        content = json.load(f)
        poem_collection.extend(content)

paragraphs = ["".join(data["paragraphs"]) for data in poem_collection]

# 与Reddit的例子类似，我们转换为 TF 数据集，并且仅使用部分数据进行训练。
train_ds = (
    tf.data.Dataset.from_tensor_slices(paragraphs)
    .batch(16)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
)

# Running through the whole dataset takes long, only take `500` and run 1
# epochs for demo purposes.
train_ds = train_ds.take(500)
num_epochs = 1

learning_rate = keras.optimizers.schedules.PolynomialDecay(5e-4,
                decay_steps=train_ds.cardinality() * num_epochs,end_learning_rate=0.0,)
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
gpt2_lm.compile(optimizer=keras.optimizers.Adam(learning_rate),loss=loss,weighted_metrics=["accuracy"],)

gpt2_lm.fit(train_ds, epochs=num_epochs)
output = gpt2_lm.generate("昨夜雨疏风骤", max_length=200)
print(output)

# 昨夜雨疏风骤，爲臨江山院短靜。石淡山陵長爲羣，臨石山非處臨羣。美陪河埃聲爲羣，漏漏漏邊陵塘
```