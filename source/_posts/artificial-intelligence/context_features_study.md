---
title: 上下文特征 & 多任务学习（TensorFlow 构建推荐系统）
date: 2024-04-17 18:00:11
tags:
  - AI
categories:
  - 人工智能
---

#### 上下文特征

为了提高模型的准确性，我们可以做的事情之一是利用**上下文特征**，有时称为“辅助特征”。我们之前的例子并未包含上下文特征，而完全依赖于用户和项目`ID`。在推荐模型中，除`ID`之外的特征是否有用：
- **上下文的重要性**：如果用户偏好在上下文和时间上相对平滑，则上下文特征可能不会提升模型的准确性。然而，如果用户偏好与上下文高度相关，则添加上下文将显着改进模型的准确性。例如，在决定是否推荐短片或电影时，星期几可能是一个重要特征：用户可能只有在一周内有时间观看短片内容，但可以在周末放松并欣赏完整的电影。类似地，查询时间戳可能在流行度动态建模中发挥重要作用：一部电影在发行时可能非常受欢迎，但之后很快就会衰退。相反，其他电影可能是常青树，让人重复地观看。
- **数据稀疏性**：如果数据稀疏，使用非`ID`特征可能很关键。由于给定用户或项目的可用观察很少，模型可能难以估计每位用户或每个项目表示。为了构建准确的模型，必须使用项目类别、描述和图像等其他特征来帮助模型泛化到训练数据之外。这在冷启动情况下尤其重要，在冷启动情况下，某些项目或用户的可用数据相对较少。
<!-- more -->

##### 导入包 & 准备数据集
在本示例中，我们将尝试在`MovieLens`模型中使用电影标题和用户`ID`之外的特征。
```python
import os
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# 我们保留用户ID、时间戳和电影标题特征。
ratings = tfds.load("movielens/100k-ratings", split="train")
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])

# 准备特征词汇表。
timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,
)

unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(lambda x: x["user_id"]))))
```
##### 实现模型

作为模型的第一层，任务是将原始输入示例转换为特征嵌入。然而，我们稍微改变它以允许我们打开或关闭时间戳功能。这将使我们能够更轻松地演示时间戳特征对模型的影响。在下面的代码中，`use_timestamps`参数使我们能够控制是否使用时间戳特征。
```python
# 查询模型
class UserModel(tf.keras.Model):

  def __init__(self, use_timestamps):
    super().__init__()

    self._use_timestamps = use_timestamps
    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),
    ])

    if use_timestamps:
      self.timestamp_embedding = tf.keras.Sequential([
          tf.keras.layers.Discretization(timestamp_buckets.tolist()),
          tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
      ])
      self.normalized_timestamp = tf.keras.layers.Normalization(axis=None)
      self.normalized_timestamp.adapt(timestamps)

  def call(self, inputs):
    if not self._use_timestamps:
      return self.user_embedding(inputs["user_id"])

    return tf.concat([
        self.user_embedding(inputs["user_id"]),
        self.timestamp_embedding(inputs["timestamp"]),
        tf.reshape(self.normalized_timestamp(inputs["timestamp"]), (-1, 1)),], axis=1)

# 候选模型
class MovieModel(tf.keras.Model):

  def __init__(self):
    super().__init__()

    max_tokens = 10_000
    self.title_embedding = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
    ])

    self.title_vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens)

    self.title_text_embedding = tf.keras.Sequential([
      self.title_vectorizer,
      tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
      tf.keras.layers.GlobalAveragePooling1D(),
    ])

    self.title_vectorizer.adapt(movies)

  def call(self, titles):
    return tf.concat([self.title_embedding(titles),self.title_text_embedding(titles),], axis=1)

# 定义了UserModel和 MovieModel后，我们可以创建一个组合模型并实现我们的损失和指标逻辑。
# 请注意，我们还需要确保查询模型和候选模型输出嵌入的大小兼容。 
# 因为我们将通过添加更多特征来改变它们的大小，所以实现此目的的最简单方法是在每个模型之后使用密集投影层：
class MovielensModel(tfrs.models.Model):

  def __init__(self, use_timestamps):
    super().__init__()
    self.query_model = tf.keras.Sequential([UserModel(use_timestamps),tf.keras.layers.Dense(32)])
    self.candidate_model = tf.keras.Sequential([MovieModel(),tf.keras.layers.Dense(32)])
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(candidates=movies.batch(128).map(self.candidate_model),),)

  def compute_loss(self, features, training=False):
    # We only pass the user id and timestamp features into the query model. This
    # is to ensure that the training inputs would have the same keys as the
    # query inputs. Otherwise the discrepancy in input structure would cause an
    # error when loading the query model after saving it.
    query_embeddings = self.query_model({"user_id": features["user_id"],"timestamp": features["timestamp"],})
    movie_embeddings = self.candidate_model(features["movie_title"])

    return self.task(query_embeddings, movie_embeddings)
```
##### 实验

```python
# 准备数据，首先将数据分为训练集和测试集。
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()

# 基线：没有时间特征。尝试我们的第一个模型：让我们从不使用时间戳特征。
model = MovielensModel(use_timestamps=False)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy = model.evaluate(cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")

# Top-100 accuracy (train): 0.30.
# Top-100 accuracy (test): 0.21.

# 利用时间特征捕捉时间动态，如果我们添加时间特征，结果会改变吗？
model = MovielensModel(use_timestamps=True)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)

train_accuracy = model.evaluate(cached_train, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]
test_accuracy = model.evaluate(cached_test, return_dict=True)["factorized_top_k/top_100_categorical_accuracy"]

print(f"Top-100 accuracy (train): {train_accuracy:.2f}.")
print(f"Top-100 accuracy (test): {test_accuracy:.2f}.")

# Top-100 accuracy (train): 0.37.
# Top-100 accuracy (test): 0.25.
```
有时间特征比没有时间特征要好得多：不仅训练精度更高，而且测试精度也大幅提高。
#### 多任务学习

我们构建了一个使用电影观看作为积极交互信号的检索系统。然而，在许多应用中，有多种丰富的反馈来源可供利用。例如，电商网站可能会记录用户对产品页面的访问、图片点击、添加到购物车以及最后的购买。它甚至可以记录购买后的信号，例如评论和退货。集成所有这些不同形式的反馈对于构建用户喜欢使用的电商网站至关重要，并且不会以牺牲整体性能为代价来优化任何一项指标。此外，为多个任务构建联合模型比构建多个特定任务模型产生更好的效果。当某些数据丰富（例如点击）而某些数据稀疏（购买、退货、评论）时尤其如此。在这些场景中，联合模型能够使用从丰富任务中学习到的模式，称为**迁移学习**的现象来改进其对稀疏任务的预测。例如，通过添加大量点击日志数据的辅助任务，可以大大改进从稀疏用户调查中预测显式用户评分的模型。在本示例中，我们将使用隐式信号（电影观看）和显式信号（评分）为`Movielens`构建**多目标推荐器**。
{% asset_img cf_1.png %}

**多任务学习**并不是一项新技术，早在`1997`年，`Rich Caruana`就发表了一篇被广泛引用的关于多任务学习的论文。这个想法是通过利用任务之间的共性和差异来同时解决多个机器学习任务。这是有道理的，因为在许多应用中，有多种反馈来源可供利用。例如，在`YouTube`上，用户可以提供各种不同的信号。用户可能会看一些视频，但跳过其它视频，提供了隐式反馈。他们可能喜欢可能不喜欢、在视频上添加评论，甚至将视频分享到其他社交平台。集成所有这些不同形式的反馈构建用户喜欢使用的系统。避免牺牲整体性能为代价来优化单个指标至关重要。此外，为多个任务构建联合模型可能比构建多个特定任务模型产生更好的结果。例如评论和分享。在这些场景中，联合模型从丰富的任务中学习，通过迁移学习改进对稀疏任务的预测。接下来构建一个包含检索任务和使用隐式和显式反馈的排名任务。

##### 导入包 & 准备数据集

```python
import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

ratings = tfds.load('movielens/100k-ratings', split="train")
movies = tfds.load('movielens/100k-movies', split="train")

# 我们使用 Movielens 100K 数据集。
# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
})
movies = movies.map(lambda x: x["movie_title"])

# 重复构建词汇表并将数据拆分为训练集和测试集的准备工作：
# Randomly shuffle data and split between train and test.
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))
```
##### 多任务模型

多任务推荐器有两个关键部分：
- 他们针对两个或更多目标进行优化，因此有两个或更多损失。
- 它们在任务之间共享变量，从而允许迁移学习。

我们将有两个任务，而不是单个任务：一个预测收视率；另一个预测电影观看次数。
```python
user_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_user_ids, mask_token=None),
  # We add 1 to account for the unknown token.
  tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])

movie_model = tf.keras.Sequential([
  tf.keras.layers.StringLookup(
      vocabulary=unique_movie_titles, mask_token=None),
  tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
])

# 现在我们有两个任务。首先是评分任务：其目标是尽可能准确地预测收视率。
tfrs.tasks.Ranking(
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)

# 第二个是检索任务：此任务的目标是预测用户将观看或不会观看哪些电影。
tfrs.tasks.Retrieval(
    metrics=tfrs.metrics.FactorizedTopK(candidates=movies.batch(128))
)
```
##### 模型组合

由于我们有两个任务和两个损失-我们需要决定每个损失的重要性。我们通过给每个损失一个权重，并将这些权重视为超参数。如果我们为评级任务分配较大的损失权重，我们的模型将专注于预测评级（但仍使用检索任务中的一些信息）；如果我们为检索任务分配较大的损失权重，它将转而专注于检索。
```python
class MovielensModel(tfrs.models.Model):

  def __init__(self, rating_weight: float, retrieval_weight: float) -> None:
    # We take the loss weights in the constructor: this allows us to instantiate
    # several model objects with different loss weights.
    super().__init__()

    embedding_dimension = 32
    # User and movie models.
    self.movie_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])
    self.user_model: tf.keras.layers.Layer = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # A small model to take in user and movie embeddings and predict ratings.
    # We can make this as complicated as we want as long as we output a scalar
    # as our prediction.
    self.rating_model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1),
    ])

    # The tasks.
    self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
    )
    self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.movie_model)
        )
    )

    # The loss weights.
    self.rating_weight = rating_weight
    self.retrieval_weight = retrieval_weight

  def call(self, features: Dict[Text, tf.Tensor]) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model.
    movie_embeddings = self.movie_model(features["movie_title"])

    return (
        user_embeddings,
        movie_embeddings,
        # We apply the multi-layered rating model to a concatentation of
        # user and movie embeddings.
        self.rating_model(
            tf.concat([user_embeddings, movie_embeddings], axis=1)
        ),
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    ratings = features.pop("user_rating")
    user_embeddings, movie_embeddings, rating_predictions = self(features)
    # We compute the loss for each task.
    rating_loss = self.rating_task(
        labels=ratings,
        predictions=rating_predictions,
    )
    retrieval_loss = self.retrieval_task(user_embeddings, movie_embeddings)

    # And combine them using the loss weights.
    return (self.rating_weight * rating_loss + self.retrieval_weight * retrieval_loss)
```
##### 评级专用模型

根据我们分配的权重，模型将对任务的不同平衡进行编码。让我们从一个只考虑评级的模型开始。
```python
model = MovielensModel(rating_weight=1.0, retrieval_weight=0.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# 模型训练
model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

# Retrieval top-100 accuracy: 0.060.
# Ranking RMSE: 1.113.
```
该模型在预测收视率方面表现良好（`RMSE`约为`1.11`），但在预测哪些电影将被观看或不被观看方面表现不佳：其准确率几乎比仅训练用于预测观看次数的模型差`4`倍。
##### 检索专用模型

让我们尝试一个仅专注于检索的模型。
```python
model = MovielensModel(rating_weight=0.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

# Retrieval top-100 accuracy: 0.233.
# Ranking RMSE: 3.688.
```
我们得到了相反的结果：模型在检索方面表现良好，但在预测评级方面表现不佳。

##### 联合模型

现在让我们训练一个为这两项任务都分配权重的模型。
```python
model = MovielensModel(rating_weight=1.0, retrieval_weight=1.0)
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(cached_train, epochs=3)
metrics = model.evaluate(cached_test, return_dict=True)

print(f"Retrieval top-100 accuracy: {metrics['factorized_top_k/top_100_categorical_accuracy']:.3f}.")
print(f"Ranking RMSE: {metrics['root_mean_squared_error']:.3f}.")

Retrieval top-100 accuracy: 0.235.
Ranking RMSE: 1.110.
```
结果是一个模型在这两项任务上的表现与每个专用模型大致相同。

##### 预测

我们可以使用经过训练的多任务模型来获得经过用户和电影嵌入训练的预测的评分：
```python
trained_movie_embeddings, trained_user_embeddings, predicted_rating = model({
      "user_id": np.array(["42"]),
      "movie_title": np.array(["Dances with Wolves (1990)"])
  })
print("Predicted rating:")
print(predicted_rating)

# Predicted rating:
# tf.Tensor([[4.604047]], shape=(1, 1), dtype=float32)
```
虽然此处的结果并未表现出联合模型在这种情况下具有明显的准确性优势，但多任务学习通常是一种非常有用的工具。当我们可以将知识从数据密集的任务（例如点击）转移到密切相关的数据稀疏任务（例如购买）时，可以期待更好的结果。
