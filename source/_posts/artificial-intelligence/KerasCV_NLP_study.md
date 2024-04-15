---
title: KerasCV & KerasNLP 介绍
date: 2024-04-15 9:06:11
tags:
  - AI
categories:
  - 人工智能
---

在深入了解`KerasCV`和`KerasNLP`之前，先向你介绍`Keras`的一个新功能：`Keras Core`。允许你在任意框架（`TensorFlow、JAX`和`PyTorch`）之上运行`Keras`代码。`Keras`组件，例如层、模型或指标，作为低级`TensorFlow、JAX`和`PyTorch`工作流程的一部分。
<!-- more -->
`KerasCV & KerasNLP`的后台环境变量设置：
{% asset_img cn_1.png %}

```python
# Classify text
model = BertClassifier.from_preset('bert_base_on_uncased', num_classes=2)
model.compile(...)
model.fit(movie_review_dataset)
model.predict(['What an amazing movie!'],)

# Text to image
model = StableDiffusion(img_width=512, img_height=512)
image = model.text_to_image(
    "photograph of an astronaut "
    "riding a horse",
    batch_size=3,
)
```
了解最新模型以保持竞争力至关重要，与此同时，最先进的模型变得越来越复杂，通常需要昂贵的资源。`KerasCV & KerasNLP`可以帮助你完成许多机器学习任务。
{% asset_img cn_2.png %}

`KerasCV & KerasNLP`中最高级别的模型是任务，任务（如图像分类器）是一个`Keras`模型，由主干子模型解决特定问题所需特定任务的层组成。反过来，主干模型是一组重用层。通常在单独的任务上进行预训练，从输入数据中提取信息丰富的特征，从而大大减少在任务中获得有竞争力的性能所需的标记数据量和计算资源。

#### 图像分类

在此示例中我们以`ResNet`架构为主干，在`KerasCV`和`KerasNLP`中加载具有预训练权重的模型，并使用带有预设名称的`from_preset`构造函数。在此示例中`ResNet50 ImageNet`是在`ImageNet`数据集上预训练`50`层`ResNet`模型。
```python
from eras_cv.models import {ResNetBackone, ImageClassifier,}

backbone = ResNetBackone.from_preset('resnet50_imagenet',)
model = ImageClassifier(backbone=backbone, num_classes=2)
model.compile(...)
model.fit(cat_vs_dog_dataset)
```
创建主干模型之后，我们将其连同我们想要预测的类的数量一起传递给图像分类器构造函数，之后编译和拟合模型。

#### 对象检测

`KerasCV`还支持对象检测，这是一项比图像分类更复杂的任务，因为该模型可以检测任意数量的对象，并且必须为每个对象预测一个类和一个边界框。尽管增加了复杂性，但使用`RetinaNet`架构创建对象检测模型与图像分类非常相似。我们再次使用`from_preset`构造函数选择预训练的`ResNet50`主干网络。主要区别在于需要在训练集中标记边界框并在任务构造函数中指定边界框格式之后。与其他`Keras`工作流程一样编译并拟合模型。
```python
from eras_cv.models import {ResNetBackone, RetinaNet}

backbone = ResNetBackone.from_preset('resnet50_imagenet',)
model = RetinaNet(backbone=backbone, num_classes=20, bounding_box_format='xywh')
model.compile(...)
model.fit(dataset)
```
#### 数据增强

数据增强（`Data Augmentation`）是最大限度提高计算机视觉任务准确性所必须的关键预处理步骤。为了避免过度拟合训练集的光照、裁剪和其他特殊性，旋转噪声甚至将原始图像混合在一起增加训练目标的鲁棒性非常重要，这里边包括了用于旋转的`RandomFlip`、用于强度扰动的随机增强（`RandomAugment`）、用于创建合成图像的`CutMix`和`MixUp`等等。只需将您所需的增强层组合到`Keras`模型中，并在训练模型之前映射你的数据集即可。
```python
from keras_cv.layers import {CutMix, MixUp, RandomAugment, RandomFlip}

augmenter = keras.Sequential(
    [
        RandomFlip(),
        RandomAugment(value_range=(0,255)),
        CutMix(),
        MixUp(),
    ],
)
train_dataset = flowers_dataset.map(augmenter)
```
#### 图像生成

文本到图像模型（例如稳定扩散）提供了简单接口，可以根据文本提示生成新颖的图片。使用所需的输出，并提示生成多个输出图片。文本反转是一种通过示例教授稳定扩散模型。提示 `->` 提示是一种将提示修改为稳定扩散同时保持图像视觉一致的方法。
```python
from eras_cv.models import {StableDiffusion}

model = StableDiffusion(jit_compile=True)
images = model.text_to_image(
    "A mysterious dark stranger visits the great pyramids of egypt, "
    "high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting",
    batch_size=3,
)
```
#### 文本分类

让我们首先训练情感分析分类器，来预测电影评论是正面还是负面。我们使用`from_preset`构造函数实例化`BERT`分类器任务模型。与`KerasCV`的区别之一是`KerasNLP`任务模型（如`BERT`分类器）默认包含预处理，在训练和服务时传递原始字符串而不必担心使用正确的标记化和打包方法。因此，最好在任务模型上调用`from_preset`，而不是传递显示主干。这将自动为你提供一个匹配的于处理器类，它将标记并填充输入。该骨干网络已经过千兆字节文本数据的预训练，以理解上下文中单词的含义，并从我们标记的示例中提取更多信息。
```python
from keras_nlp.models import {BertClassifier}

model = BertClassifier.from_preset('bert_base_en_uncased',num_classes=2,activation='softmax',)

model.compile(...)
model.fit(imdb_movie_review_dataset)
model.predict(
    [
        'What an amazing movie!',
        'A total waste of my time',
    ]
)

>>> array([[0.004137, 0.9956],[0.997, 0.0028]], dtype=float16)
```
在根据`IMDB`的情感标记电影评论数据集微调我们的模型后，我们可以预测两条新电影评论的情感，获得积极情绪的概率为`99.6%`。
#### 文本生成

微调文本生成模型，就像分类一样简单，只需传递你希望模型，模仿的文本数据集，预处理就会自动处理。因果LM是一种任务模型，他在给定所有前面的标记的情况下预测输入序列的每个标记，这是训练生成文本模型的规范方法，可以根据用户提示预测新的标记。在此示例中，我们使用`from_preset`构造函数加载预训练的`GPT-2`模型。与其他`Keras`工作流程一样编译并拟合模型。因果任务附带一个生成方法，允许你指定提示和最大输出长度来生成新文本。
```python
from keras_nlp.models import {GPT2CausalLM}

model = BertClassifier.from_preset('gpt2_base_en',)
model.compile(...)
model.fit(cnn_dailymail_dataset)

model.generate('Snowfall in Buffalo',max_length=40,)
```
#### 自定义预处理

{% asset_img cn_3.png %}

我们从一组较低级别的模块构建最高级别的`API`来实现这一目标。例如，假设你的数据集包含相对较短的文本段，并且训练需要多次遍历数据。在这种情况下，`BERT`分类器中的构建预处理器可能不太适合你，因为他将所有序列填充到了`512`个标记并在每个训练周期中重新计算预处理。对于这种情况从头开始主干、`BERT`预处理器和`BERT`分词器类构建而成，每个类都有自己的预设方法，要访问预处理器目录只需使用与分类器相同的预设名称以及你想要指定的任何自定义参数（例如较短的序列），然后你可以自己应用预处理。在此示例中，包含了缓存，在每个时期都不会重新计算标记。为了避免在工作流程中调用预处理器两次，只需要在人物构造函数中设置为`None`即可。与其他`Keras`工作流程一样编译并拟合模型。
```python
from keras_nlp.models import {BertClassifier, BertPreprocessor}

preprocessor = BertPreprocessor.from_preset('bert_base_en_uncased',squence_length=128,)

train_dataset = train_dataset.map(
    preprocessor,
).cache()

model = BertClassifier.from_preset('bert_base_en_uncased',preprocessor=None, num_classes=2,activation='softmax',)
model.compile(...)
model.fit(train_dataset, epochs=10)
```
如果需要更多灵活性，则使用`Keras`函数式`API`：
```python
from keras_nlp.models import {TokenAndPositionEmbedding, TransformerEncoder}

token_id_input = keras.Input(
    shape=(None,),
    dtype = 'int32',
    name = 'token_ids',
)

outputs = TokenAndPositionEmbedding(
    vocabulary_size=30_000,
    sequence_length = 512,
    embedding_dim = 64,
)(token_id_input)

for _in range(num_layers):
    ouputs = TransformerEncoder(
        num_heads=2,
        intermediate_dim =128,
        dropout = 0.1
    )(outputs)

# Use placeholder token embedding to classify 
outputs = keras.layers.Dense(2)(outputs[:,0,:])
model = keras.Model(
    inputs = token_id_input,
    outputs = outputs,
)
```
创建新模型的第一步是声明输入张量。在本例中，我们的输入是令牌`ID`的可变长度序列，然后，我们将此序列传入到嵌入层学习每个标记`ID`和序列位置的唯一向量表示，并返回序列中每个标记的总和。然后我们将嵌入输出传递到一堆可配置的`Transformer`编码器层，该编码器层将一系列多头注意力和前馈层应用于输入。该堆栈的输出使我们最终的序列表示。为了让令牌序列生成单个分类输出，常见的做法是在每个序列的开头放置一个占位符令牌，并将该令牌的表示作为输入传递到前馈层，该前馈层具有与要预测的类相同数量的输出。与任何`Keras`模型一样我们将特征输入和输出传递给构造函数以获取模型实例。与其他`Keras`工作流程一样编译并拟合模型。