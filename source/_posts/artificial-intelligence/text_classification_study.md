---
title: 文本分类（KerasNLP）
date: 2024-04-14 17:22:11
tags:
  - AI
categories:
  - 人工智能
---

`KerasNLP`是一个自然语言处理库，可为用户的整个开发周期提供支持。我们的工作流程由模块化组件构建，这些组件在开箱即用时具有最先进的预设权重和架构，并且在需要更多控制时可轻松定制。该库是核心`Keras API`的扩展；所有高级模块都是层或模型。`KerasNLP`使用`Keras 3`与`TensorFlow、Pytorch`和`Jax`配合一起使用。在下面例子中，我们将使用`jax`后端来训练我们的模型并使用`tf.data`来有效地运行我们的输入预处理。
- 使用预训练分类器进行推理。
- 微调预训练的骨干模型。
- 通过用户控制的预处理进行微调。
- 微调自定义模型。
- 预训练骨干模型。
- 从头开始构建和训练您自己的`Transformer`。

<!-- more -->
自定义预处理：
{% asset_img tc_1.png %}

分类器完成一个分类任务，有一个`from_preset`方法来加载准备使用的模型。在底层，有一个`BertPreprocessor`，它通过调用`BertTokenizer`来执行标记化，以及一些额外的类似于预处理的填充，将字符串映射到张量字典。`BertBackbone`将预处理的张量转化为密集特征。

#### 包 & 配置 & 数据集

首先我们下载斯坦福大型电影评论数据集，这是常用的情感分析数据集，每个训练示例都包含一个电影评论额一个和指示评论是正面还是负面的整数。在此任务中，评论是正面的（整数`=1`）、评论是负面的（整数`=0`）。我们使用`keras.utils.text_dataset_from_directory`加载数据，使用`tf.data.Dataset`格式。
```bash
!# Download the dataset
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
!# Remove unsupervised examples
!rm -r aclImdb/train/unsup
```
导入包：
```python
import os
import keras_nlp
import keras
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "jax"  # or "tensorflow" or "torch"
# Use mixed precision to speed up all training in this guide.
keras.mixed_precision.set_global_policy("mixed_float16")
```

#### 使用预训练分类器进行推理

`KerasNLP`中的最高级别模块是任务。任务是一个`keras.Model`，由（通常是预先训练的）主干模型和特定于任务的层组成。输出是每个类别的`logits`（例如，[`0, 0`]是`50%`的阳性概率）。二元分类的输出为[`negative`, `positive`]。
```python
# 用from_preset方法创建一个BertClassifier，我们使用的是sst2数据集上微调的小型英语模型。
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased_sst2")

# Note: batched inputs expected so must wrap string in iterable
classifier.predict(["I love modular workflows in keras-nlp!"])

# 我们可以使用分类器直接评估测试数据。
classifier.evaluate(imdb_test)

```
#### 微调预训练的BERT主干

如果我们有训练数据我们可以通过微调来提高性能。我们只需要调用`fit`并传递训练和测试数据集。
```python
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased",num_classes=2,)
classifier.fit(imdb_train,validation_data=imdb_test,epochs=1,)
```
在这里，我们看到尽管`IMDB`数据集比`sst2`小得多，但单次训练后验证准确度显着提升(`0.78 -> 0.87`)。

#### 通过用户控制的预处理进行微调

在执行微调之前可以更好的控制预处理，例如，你可以将预处理从分类器中分离出来。
```python
# 我们创建BERT预处理器,并将其传递到映射函数中。以便可以使用自己的自定义逻辑对数据进行预处理和缓存
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased",sequence_length=512,)

# Apply the preprocessor to every sample of train and test data using `map()`.
# `tf.data.AUTOTUNE` and `prefetch()` are options to tune performance, see
# https://www.tensorflow.org/guide/data_performance for details.

# Note: only call `cache()` if you training data fits in CPU memory!
imdb_train_cached = (imdb_train.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE))
imdb_test_cached = (imdb_test.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE))

# 需要关闭预处理器，你可以直接使用分词器，并创立自己的与处理逻辑。
classifier = keras_nlp.models.BertClassifier.from_preset("bert_tiny_en_uncased", preprocessor=None, num_classes=2)
classifier.fit(imdb_train_cached,validation_data=imdb_test_cached,epochs=3,)
```
#### 使用自定义模型进行微调

```python
# 对数据进行预处理，例如，对于自定义模型。
preprocessor = keras_nlp.models.BertPreprocessor.from_preset("bert_tiny_en_uncased")
backbone = keras_nlp.models.BertBackbone.from_preset("bert_tiny_en_uncased")

imdb_train_preprocessed = (imdb_train.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE))
imdb_test_preprocessed = (imdb_test.map(preprocessor, tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE))

# 我们冻结主干权重，并在主干模型顶部堆叠两个Transformer编码器。
backbone.trainable = False
inputs = backbone.input
sequence = backbone(inputs)["sequence_output"]
for _ in range(2):
    sequence = keras_nlp.layers.TransformerEncoder(num_heads=2,intermediate_dim=512,dropout=0.1,)(sequence)

# 然后使用密集层生成分类结果。
# Use [CLS] token output to classify
outputs = keras.layers.Dense(2)(sequence[:, backbone.cls_token_index, :])

# 最后我们就可以像往常一样编译和拟合了。
model = keras.Model(inputs, outputs)
model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.AdamW(5e-5),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
    jit_compile=True,
)
model.summary()
model.fit(imdb_train_preprocessed,validation_data=imdb_test_preprocessed,epochs=3,
)
```
以上这就是微调自定义模型的方法。无论你想使用现成的预训练模型，还是创建自定义模型以提高准确性，`KerasNLP`都能满足你的需求。这里忽略了从头开始创建`Transformer`模型。