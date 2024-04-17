---
title: 排名模型（TensorFlow 构建推荐系统）
date: 2024-04-17 14:50:11
tags:
  - AI
categories:
  - 人工智能
---

推荐系统通常由两个阶段组成：
- **检索阶段**：负责从所有可能的候选者中选择数百个候选者的初始集合。检索模型的主要目标是有效地剔除用户不感兴趣的所有候选者。由于检索模型可能要处理数百万个候选者，因此它必须具有很高的计算效率。
- **排名阶段**：获取检索模型的输出，并对它们进行微调以选择尽可能好的推荐。它的任务是将用户感兴趣的项目集缩小到可能的候选者的候选名单。
<!-- more -->

#### 排名模型（Ranking Model）

我们接下来的步骤：
- 获取我们的数据并将其分成训练集和测试集。
- 实现排名模型。
- 拟合并评估模型。

##### 导入包 & 准备数据集
评级被视为显示反馈，因为我们可以根据评级数值大致了解用户对电影的喜欢程度。接下来我们对数据集进行清洗，并将其拆分为训练和测试数据集。
```python
import os
import pprint
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

from typing import Dict, Text

# 这次，我们也将保留评级：这是我们预测的目标。
ratings = tfds.load("movielens/100k-ratings", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

# 和以前一样，我们将通过将80%的评分放入训练集，将20%放入测试集来分割数据。
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# 我们还可以找出数据中存在的唯一用户ID和电影标题。
# 我们需要能够将分类特征的原始值映射到模型中的嵌入向量。
# 为此，我们需要一个将原始特征值映射到连续范围内的整数的词汇表：这使我们能够在嵌入表中查找相应的嵌入。
movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))
```
##### 实现模型

排名模型不会面临与检索模型相同的效率限制，因此我们在选择架构方面有更多的自由度。由多个堆叠的密集层组成的模型是排序任务中相对常见的架构。我们可以这样实现：
```python
class RankingModel(tf.keras.Model):

  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    # Compute embeddings for users.
    self.user_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_user_ids, mask_token=None),
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
      tf.keras.layers.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None),
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # Compute predictions.
    self.ratings = tf.keras.Sequential([
      # Learn multiple dense layers.
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(64, activation="relu"),
      # Make rating predictions in the final layer.
      tf.keras.layers.Dense(1)
  ])

  def call(self, inputs):
    user_id, movie_title = inputs
    user_embedding = self.user_embeddings(user_id)
    movie_embedding = self.movie_embeddings(movie_title)

    return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

RankingModel()((["42"], ["One Flew Over the Cuckoo's Nest (1975)"]))

# <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[-0.01534399]], dtype=float32)>

# 下一个部分是用于训练模型的损失，我们将使用排名任务对象：一个将损失函数和度量计算捆绑在一起的便捷包装器。
# 我们将把它与MeanSquaredError Keras损失一起使用来预测评级。
task = tfrs.tasks.Ranking(loss = tf.keras.losses.MeanSquaredError(),metrics=[tf.keras.metrics.RootMeanSquaredError()])

# 我们现在可以将它们全部整合到一个模型中。
# TFRS 公开了一个基本模型类 (tfrs.models.Model)，它简化了构建模型：
# 我们需要做的就是在__init__方法中设置组件，并实现compute_loss方法，获取原始特征并返回损失值。

# 然后，基础模型将负责创建适当的训练循环以适应模型。
class MovielensModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    self.ranking_model: tf.keras.Model = RankingModel()
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
      loss = tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError()]
    )

  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    return self.ranking_model(
        (features["user_id"], features["movie_title"]))

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    labels = features.pop("user_rating")
    rating_predictions = self(features)

    # The task computes the loss and the metrics.
    return self.task(labels=labels, predictions=rating_predictions)
```
##### 拟合和评估模型

定义模型后，我们可以使用标准的`Keras`来拟合和评估模型。
```python
# 首先实例化模型
model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

# 然后对训练和评估数据进行混洗、批处理和缓存。
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# 然后训练模型
model.fit(cached_train, epochs=3)

# Epoch 1/3
# 10/10 [=====] - 4s 166ms/step - root_mean_squared_error: 2.0902 - loss: 4.0368 - regularization_loss: 0.0000e+00 - total_loss: 4.0368
# Epoch 2/3
# 10/10 [=====] - 0s 4ms/step - root_mean_squared_error: 1.1613 - loss: 1.3426 - regularization_loss: 0.0000e+00 - total_loss: 1.3426
# Epoch 3/3
# 10/10 [=====] - 0s 4ms/step - root_mean_squared_error: 1.1140 - loss: 1.2414 - regularization_loss: 0.0000e+00 - total_loss: 1.2414
# <keras.callbacks.History at 0x7fd31445d490>

# 随着模型训练，损失不断下降，RMSE指标不断提高。
# 最后，我们可以在测试集上评估我们的模型：
# RMSE指标越低，我们的模型预测评级就越准确。
model.evaluate(cached_test, return_dict=True)

# 5/5 [========] - 2s 9ms/step - root_mean_squared_error: 1.1009 - loss: 1.2072 - regularization_loss: 0.0000e+00 - total_loss: 1.2072
# {
#  'root_mean_squared_error': 1.100862741470337,
#  'loss': 1.1866925954818726,
#  'regularization_loss': 0,
#  'total_loss': 1.1866925954818726
# }
```
##### 测试排名模型

现在我们可以通过计算一组电影的预测来测试排名模型，然后根据预测对这些电影进行排名：
```python
test_ratings = {}
test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
for movie_title in test_movie_titles:
  test_ratings[movie_title] = model({"user_id": np.array(["42"]),"movie_title": np.array([movie_title])})

print("Ratings:")
for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
  print(f"{title}: {score}")

# Ratings:
# Dances with Wolves (1990): [[3.539769]]
# M*A*S*H (1970): [[3.5356772]]
# Speed (1994): [[3.4501984]]
```
##### 服务

```python
tf.saved_model.save(model, "export")

# 可以加载它并执行预测：
loaded = tf.saved_model.load("export")
loaded({"user_id": np.array(["42"]), "movie_title": ["Speed (1994)"]}).numpy()

# array([[3.4501984]], dtype=float32)
```
在大多数情况下，可以通过使用更**多特征**而不仅仅是用户和候选标识符来显着改进排名模型。