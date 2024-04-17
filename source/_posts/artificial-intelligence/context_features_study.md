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

