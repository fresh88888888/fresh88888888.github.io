---
title: 检索模型（TensorFlow 构建推荐系统）
date: 2024-04-17 10:50:11
tags:
  - AI
categories:
  - 人工智能
---

`TensorFlow Recommenders(TFRS)`是一个用于构建推荐（`Recommender`）系统模型的库，在推荐（`Recommender`）系统的整个构建流程 - 无论是数据准备、模型构建、训练、评估还是部署都可以起到很大的作用。`TFRS`融合了多任务学习、特征交互建模和`TPU`训练的研究成果。推荐系统通常有多个组件来进行检索、排名和后期排名。

推荐系统通常由两个阶段组成：
- **检索阶段**：负责从所有可能的候选者中选择数百个候选者的初始集合。检索模型的主要目标是有效地剔除用户不感兴趣的所有候选者。由于检索模型可能要处理数百万个候选者，因此它必须具有很高的计算效率。
- **排名阶段**：获取检索模型的输出，并对它们进行微调以选择尽可能好的推荐。它的任务是将用户感兴趣的项目集缩小到可能的候选者的候选名单。
<!-- more -->

以下是一个简单的示例：
```bash
# 安装&导入TFRS
!pip install -q tensorflow-recommenders
!pip install -q --upgrade tensorflow-datasets
```
读取数据、定义、训练模型和预测：
```python
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from typing import Dict, Text

# 评分数据
# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train")
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train")

# Select the basic features.
ratings = ratings.map(lambda x: {"movie_title": x["movie_title"],"user_id": x["user_id"]})
movies = movies.map(lambda x: x["movie_title"])

# 构建词汇表以将用户ID和电影标题转换为嵌入层的整数索引：
user_ids_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(ratings.map(lambda x: x["user_id"]))

movie_titles_vocabulary = tf.keras.layers.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies)

# 定义两个模型和检索任务。
# Define user and movie models.
user_model = tf.keras.Sequential([
    user_ids_vocabulary,
    tf.keras.layers.Embedding(user_ids_vocabulary.vocab_size(), 64)
])
movie_model = tf.keras.Sequential([
    movie_titles_vocabulary,
    tf.keras.layers.Embedding(movie_titles_vocabulary.vocab_size(), 64)
])

# Define your objectives.
task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
    movies.batch(128).map(movie_model)
  )
)

# 创建模型、训练模型并生成预测：
# Create a retrieval model.
model = MovieLensModel(user_model, movie_model, task)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))

# Train for 3 epochs.
model.fit(ratings.batch(4096), epochs=3)

# Use brute-force search to set up retrieval using the trained representations.
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
index.index_from_dataset(movies.batch(100).map(lambda title: (title, model.movie_model(title))))

# Get some recommendations.
_, titles = index(np.array(["42"]))
print(f"Top 3 recommendations for user 42: {titles[0, :3]}")

# 定义模型
class MovieLensModel(tfrs.Model):
  # We derive from a custom base class to help reduce boilerplate. Under the hood,
  # these are still plain Keras Models.

  def __init__(self,user_model: tf.keras.Model,movie_model: tf.keras.Model,task: tfrs.tasks.Retrieval):
    super().__init__()

    # Set up user and movie representations.
    self.user_model = user_model
    self.movie_model = movie_model
    # Set up a retrieval task.
    self.task = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # Define how the loss is computed.
    user_embeddings = self.user_model(features["user_id"])
    movie_embeddings = self.movie_model(features["movie_title"])

    return self.task(user_embeddings, movie_embeddings)
```
结果输出为：
```bash
Epoch 1/3
25/25 [==============================] - 34s 1s/step - factorized_top_k/top_1_categorical_accuracy: 7.0000e-05 - factorized_top_k/top_5_categorical_accuracy: 0.0016 - factorized_top_k/top_10_categorical_accuracy: 0.0050 - factorized_top_k/top_50_categorical_accuracy: 0.0457 - factorized_top_k/top_100_categorical_accuracy: 0.1034 - loss: 33069.6692 - regularization_loss: 0.0000e+00 - total_loss: 33069.6692
Epoch 2/3
25/25 [==============================] - 31s 1s/step - factorized_top_k/top_1_categorical_accuracy: 2.8000e-04 - factorized_top_k/top_5_categorical_accuracy: 0.0052 - factorized_top_k/top_10_categorical_accuracy: 0.0150 - factorized_top_k/top_50_categorical_accuracy: 0.1054 - factorized_top_k/top_100_categorical_accuracy: 0.2118 - loss: 31012.9641 - regularization_loss: 0.0000e+00 - total_loss: 31012.9641
Epoch 3/3
25/25 [==============================] - 30s 1s/step - factorized_top_k/top_1_categorical_accuracy: 5.3000e-04 - factorized_top_k/top_5_categorical_accuracy: 0.0088 - factorized_top_k/top_10_categorical_accuracy: 0.0228 - factorized_top_k/top_50_categorical_accuracy: 0.1445 - factorized_top_k/top_100_categorical_accuracy: 0.2675 - loss: 30421.9365 - regularization_loss: 0.0000e+00 - total_loss: 30421.9365

Top 3 recommendations for user 42: [b'Just Cause (1995)' b'Rent-a-Kid (1995)' b'Cobb (1994)']
```
#### 检索模型（Retrieval Model）

检索模型通常由两个子模型组成：
- 使用查询特征计算查询表示（`normally a fixed-dimensionality embedding vector`）的**查询模型**。
- 使用候选特征计算候选表示（`an equally-sized vector`）的**候选模型**。

然后将两个模型的输出相乘以给出查询-候选者亲和力分数，分数越高表示候选者和查询之间的匹配越好。我们接下来的步骤：
- 获取数据并将其分成训练集和测试集。
- 实现检索模型。
- 模型拟合并评估。
- 通过构建近似最近邻(`ANN`)索引将其导出以实现高效检索。

##### 数据集

在本示例中，我们将使用`Movielens`数据集构建和训练这样的两塔模型。`Movielens`数据集是明尼苏达大学`GroupLens`研究小组的经典数据集。它包含一组用户对电影的评分，是推荐系统研究的主要数据集。我们可以通过两种方式处理数据：
- 它可以解释为：用户观看和评价了哪些电影，以及没有观看和评价哪些电影。这是一种**隐式反馈**，用户的`watches`会告诉我们他们喜欢看到哪些内容以及不想看到哪些内容。
- 它也可以被视为用户对他们观看的电影的喜爱程度。这是一种**显示反馈**：假设用户观看了一部电影，我们可以通过查看他们给出的评分来了解他们的喜欢程度。

在本示例中，我们重点关注**检索系统**：从目录中预测用户可能观看的一组电影的模型。通常，隐式数据在这里更有用，因此我们将`Movielens`视为**隐式系统**。这意味着用户观看的每一部电影都是一个正例，而他们没有看过的每一部电影都是一个隐含的反例。

##### 准备数据集

```python
import os
import pprint
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from typing import Dict, Text

# 请注意，由于MovieLens数据集没有预定义的分割，因此所有数据都在训练分割下。
# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

# 在此示例中，我们将重点关注收视率数据。我们只在数据集中保留user_id和movie_title字段。
ratings = ratings.map(lambda x: {"movie_title": x["movie_title"],"user_id": x["user_id"],})
movies = movies.map(lambda x: x["movie_title"])

# 我们使用随机分割，将80%的评级放入训练集中，将20%的评级放入测试集中。
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# 我们还可以找出数据中存在的唯一用户ID和电影标题。
# 我们需要能够将分类特征的原始值映射到模型中的嵌入向量。
# 为此，我们需要一个将原始特征值映射到连续范围内的整数的词汇表：这使我们能够在嵌入表中查找相应的嵌入。
movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])
unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

unique_movie_titles[:10]

# array([b"'Til There Was You (1997)", b'1-900 (1994)',
#        b'101 Dalmatians (1996)', b'12 Angry Men (1957)', b'187 (1997)',
#        b'2 Days in the Valley (1996)',
#        b'20,000 Leagues Under the Sea (1954)',
#        b'2001: A Space Odyssey (1968)',
#        b'3 Ninjas: High Noon At Mega Mountain (1998)',
#        b'39 Steps, The (1935)'], dtype=object)
```
##### 实现检索模型

选择模型的架构是建模的关键部分。因为我们正在构建一个两塔检索模型，所以我们可以单独构建每个塔，然后将它们组合到最终模型中。
{% asset_img tfrs_1.png %}

“塔”这个词的意思是输入层之上的全连接层遵循塔模式，即这些层的宽度逐渐减小，这使得它们看起来像一个堆叠的塔。正如你在上图所看到的，左侧有一座塔将用户特征映射到用户嵌入，我们称之为查询塔。右侧的另一个塔将项目特征映射到项目嵌入，我们称之为候选塔。模型的输出定义为用户嵌入和项目嵌入的点积。这个简单的模型实际上对应于**矩阵分解模型**，我们首先定义查询塔：
```python
# 第一步是确定查询和候选表示的维度：较高的值对应的模型可能更准确，但拟合速度也会较慢并且更容易过度拟合。
embedding_dimension = 32

# 第二步是定义模型本身。在这里，我们将使用Keras预处理层首先将用户ID转换为整数，然后通过嵌入层将其转换为用户嵌入。 
# 请注意，我们使用之前计算的唯一用户ID列表作为词汇表：
user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
  # We add an additional embedding to account for unknown tokens.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

# 候选塔
movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])

# 在我们的训练数据中，我们有正对（user, movie）。 
# 为了弄清楚我们的模型有多好，我们需要将模型为该对计算的亲和力分数与所有其他可能候选者的分数进行比较：
# 如果正对的分数高于所有其他候选者，该模型很准确。

# 为此，我们可以使用tfrs.metrics.FactorizedTopK指标。该指标有一个必需的参数：用作评估隐式否定的候选数据集。
metrics = tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(movie_model))

# 下一个组成部分是用于训练模型的损失。在本例中，我们将利用检索任务对象：一个将损失函数和度量计算捆绑在一起的便捷包装器：
task = tfrs.tasks.Retrieval(
  metrics=metrics
)
```
我们现在可以将它们全部整合到一个模型中。`TFRS`公开了一个基本模型类(`tfrs.models.Model`)，它简化了构建模型的过程：我们需要做的就是在`__init__`方法中设置组件，并实现`compute_loss`方法，获取原始特征并返回损失值。然后，基础模型将负责创建适当的训练循环以适应我们的模型。
```python
class MovielensModel(tfrs.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(user_embeddings, positive_movie_embeddings)

class NoBaseClassMovielensModel(tf.keras.Model):

  def __init__(self, user_model, movie_model):
    super().__init__()
    self.movie_model: tf.keras.Model = movie_model
    self.user_model: tf.keras.Model = user_model
    self.task: tf.keras.layers.Layer = task

  def train_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # Set up a gradient tape to record gradients.
    with tf.GradientTape() as tape:

      # Loss computation.
      user_embeddings = self.user_model(features["user_id"])
      positive_movie_embeddings = self.movie_model(features["movie_title"])
      loss = self.task(user_embeddings, positive_movie_embeddings)

      # Handle regularization losses as well.
      regularization_loss = sum(self.losses)

      total_loss = loss + regularization_loss

    gradients = tape.gradient(total_loss, self.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics

  def test_step(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:

    # Loss computation.
    user_embeddings = self.user_model(features["user_id"])
    positive_movie_embeddings = self.movie_model(features["movie_title"])
    loss = self.task(user_embeddings, positive_movie_embeddings)

    # Handle regularization losses as well.
    regularization_loss = sum(self.losses)

    total_loss = loss + regularization_loss

    metrics = {metric.name: metric.result() for metric in self.metrics}
    metrics["loss"] = loss
    metrics["regularization_loss"] = regularization_loss
    metrics["total_loss"] = total_loss

    return metrics
```
##### 拟合和评估

定义模型后，我们可以使用标准的`Keras`拟合和评估例程来拟合和评估模型。
```python
# 我们首先实例化模型。
model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# 然后对训练和评估数据进行混洗、批处理和缓存。
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# 训练模型：
model.fit(cached_train, epochs=3)

# Epoch 1/3
# 10/10 [==============================] - 6s 309ms/step - factorized_top_k/top_1_categorical_accuracy: 7.2500e-04 - factorized_top_k/top_5_categorical_accuracy: 0.0063 - factorized_top_k/top_10_categorical_accuracy: 0.0140 - factorized_top_k/top_50_categorical_accuracy: 0.0753 - factorized_top_k/top_100_categorical_accuracy: 0.1471 - loss: 69820.5881 - regularization_loss: 0.0000e+00 - total_loss: 69820.5881
# Epoch 2/3
# 10/10 [==============================] - 3s 302ms/step - factorized_top_k/top_1_categorical_accuracy: 0.0011 - factorized_top_k/top_5_categorical_accuracy: 0.0119 - factorized_top_k/top_10_categorical_accuracy: 0.0260 - factorized_top_k/top_50_categorical_accuracy: 0.1403 - factorized_top_k/top_100_categorical_accuracy: 0.2616 - loss: 67457.6612 - regularization_loss: 0.0000e+00 - total_loss: 67457.6612
# Epoch 3/3
# 10/10 [==============================] - 3s 301ms/step - factorized_top_k/top_1_categorical_accuracy: 0.0014 - factorized_top_k/top_5_categorical_accuracy: 0.0189 - factorized_top_k/top_10_categorical_accuracy: 0.0400 - factorized_top_k/top_50_categorical_accuracy: 0.1782 - factorized_top_k/top_100_categorical_accuracy: 0.3056 - loss: 66284.5682 - regularization_loss: 0.0000e+00 - total_loss: 66284.5682

# 最后，我们可以在测试集上评估我们的模型：
model.evaluate(cached_test, return_dict=True)

# {'factorized_top_k/top_1_categorical_accuracy': 0.0010000000474974513,
#  'factorized_top_k/top_5_categorical_accuracy': 0.008700000122189522,
#  'factorized_top_k/top_10_categorical_accuracy': 0.021150000393390656,
#  'factorized_top_k/top_50_categorical_accuracy': 0.121799997985363,
#  'factorized_top_k/top_100_categorical_accuracy': 0.23340000212192535,
#  'loss': 28256.8984375,
#  'regularization_loss': 0,
#  'total_loss': 28256.8984375}

# 模型预测：我们可以使用tfrs.layers.factorized_top_k.BruteForce层来做到这一点。
# Create a model that takes in raw query features, and
index = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
# recommends movies out of the entire movies dataset.
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.movie_model)))
)

# Get recommendations.
_, titles = index(tf.constant(["42"]))
print(f"Recommendations for user 42: {titles[0, :3]}")

# Recommendations for user 42: [b'Christmas Carol, A (1938)' b'Rudy (1993)' b'Bridges of Madison County, The (1995)']
```
##### 服务

模型训练完成后，我们需要一种部署它的方法。在双塔检索模型中，服务有两个组成部分：
- 服务查询模型，接收查询的特征并将其转换为查询嵌入。
- 服务候选人模型。这通常采用近似最近邻(`ANN`)索引的形式，该索引允许响应查询模型生成的查询近似候选对象。

在`TFRS`中，这两个组件都可以打包到单个可导出模型中，该模型采用原始用户`ID`并返回该用户的热门电影的标题。这是通过将模型导出为`SavedModel`格式来完成的，这使得可以使用`TensorFlow Serving`发布服务。要部署这样的模型，我们只需导出上面创建的`BruteForce`层：
```python
# Export the query model.
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  # Save the index.
  tf.saved_model.save(index, path)
  # Load it back; can also be done in TensorFlow Serving.
  loaded = tf.saved_model.load(path)
  # Pass a user id in, get top predicted movie titles back.
  scores, titles = loaded(["42"])

  print(f"Recommendations: {titles[0][:3]}")

# Recommendations: [b'Christmas Carol, A (1938)' b'Rudy (1993)' b'Bridges of Madison County, The (1995)']
```
##### 逐项推荐

在这个模型中，我们创建了一个用户电影模型。但是，对于某些应用（例如，产品详细信息页面），通常会执行逐项（例如，电影到电影）的推荐。像这样的训练模型将遵循本示例中所示的相同模式，但使用不同的训练数据。在这里，我们有一个用户和一个电影塔，并使用（用户，电影）对来训练它们。在项目到项目模型中，我们将有两个项目塔（查询和候选项目），并使用（查询项目、候选项目）对训练模型。