---
title: TPU embeddings（利用TensorFlow构建推荐系统）
date: 2024-04-14 11:40:11
tags:
  - AI
categories:
  - 人工智能
---

如果你正在构建大规模推荐系统，那么最大的挑战必定是**模型当中的大型嵌入表**，这些嵌入表是关键组件。但对他们的嵌入查询操作通常执行起来非常昂贵，这使他们成为性能的瓶颈。因此，接下来我们将讨论：”如何使用`TPU embeddings`来应对这一挑战？“。
<!-- more -->
让我们回顾一下，在现代大规模推荐系统是如何工作的：
- 第一步，训练神经网络模型（离线），例如经典的两塔模型（查询塔、数据库塔），学习如何将查询项目映射到联合嵌入。
- 第二步，映射数据库数据项到嵌入空间（离线），我们映射所有的数据库项目到嵌入空间的项目。
- 第三步，在运行时，我们将计算查询嵌入并且进行向量相似性搜索在嵌入空间中查找最近的项目。

首先，训练嵌入表可能非常具有挑战性，如果你有大量的项目词汇量，比如要推荐超过`1`亿个项目，或者一些高维的稀疏特征。您将需要大型嵌入表来存储，这些嵌入表可能通常不适合单个加速器，因此，您必须在多个加速器之间共享它们，这会带来通信开销，并使查找操作代价变得昂贵。虽然有一些软件的解决方案，但最好从硬件和软件方面解决这个问题，这最终导致我们会使用`TPU embeddings`，专门设计的硬件`Sparse Core`，专用于加速嵌入查找操作。
{% asset_img te_1.png %}

`TPU embeddings`可以显著提高推荐系统的速度。以下是`Google`内部生产推荐模型的性能基准测试：
{% asset_img te_2.png %}

正如你所看到的，通过`TPU v3`和`v4`上使用了`Sparse Core`与`CPU`上的嵌入相比，速度提高了`10~30`倍，接下来了解如何使用`TPU embeddings`。
```python
import os
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import tensorflow_datasets as tfds
from google.colab import auth

# 预定义的TPU策略对象
resolver = tf.distribute.cluster_resolver.TPUClusterResolver('').connect('')
strategy = tf.distribute.TPUStrategy(resolver)

# 需要一个GCS存储桶来讲数据提供给TPU
gcs_bucket = 'gs://YOUR-BUCKET-NAME'  #@param {type:"string"}
auth.authenticate_user()

# 首先我们使用tensorflow_dataset获取数据。我们需要的数据是movie_id、user_id和user_ rating。然后对数据进行预处理并将其转换为整数。
# Ratings data.
ratings = tfds.load(
    "movielens/100k-ratings", data_dir=gcs_bucket, split="train")

# 我们将user_id和movie_id转换为整数。
ratings = ratings.map(
    lambda x: {
        "movie_id": tf.strings.to_number(x["movie_id"]),
        "user_id": tf.strings.to_number(x["user_id"]),
        "user_rating": x["user_rating"],
    })

# 为模型定义一些超参数。
per_replica_batch_size = 16
movie_vocabulary_size = 2048
movie_embedding_size = 64
user_vocabulary_size = 2048
user_embedding_size = 64

# 我们将通过将80%的评级放入训练集，将20%放入测试集来分割数据。
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# 对数据集进行批处理缓存并将其转换为分布式数据集。
train_dataset = train.batch(per_replica_batch_size * strategy.num_replicas_in_sync,drop_remainder=True).cache()
test_dataset = test.batch(per_replica_batch_size * strategy.num_replicas_in_sync,drop_remainder=True).cache()

distribute_train_dataset = strategy.experimental_distribute_dataset(train_dataset,options=tf.distribute.InputOptions        (experimental_fetch_to_device=False))
distribute_test_dataset = strategy.experimental_distribute_dataset(test_dataset,options=tf.distribute.InputOptions
(experimental_fetch_to_device=False))

# 定义优化器和表配置，为了将嵌入放置在Sparse Core上
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)
# 指定词汇表大小和嵌入维度，然后通过特征配置将特征和表配置关联起来。
# movie_id放置在电影表中、user_id放置在用户表中。
user_table = tf.tpu.experimental.embedding.TableConfig(vocabulary_size=user_vocabulary_size, dim=user_embedding_size)
movie_table = tf.tpu.experimental.embedding.TableConfig(vocabulary_size=movie_vocabulary_size, dim=movie_embedding_size)
feature_config = {
    "movie_id": tf.tpu.experimental.embedding.FeatureConfig(table=movie_table),
    "user_id": tf.tpu.experimental.embedding.FeatureConfig(table=user_table)
}

# 在这里我们创建优化器，指定特征和表配置。然后我们创建带有嵌入层的模型。
# Define a ranking model with embedding layer.
class EmbeddingModel(tfrs.models.Model):

  def __init__(self):
    super().__init__()
    # 将特征配置和优化器一起传递到TPU嵌入层
    self.embedding_layer = tfrs.layers.embedding.TPUEmbedding(feature_config=feature_config, optimizer=optimizer)
    self.ratings = tf.keras.Sequential([
        # Learn multiple dense layers.
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        # Make rating predictions in the final layer.
        tf.keras.layers.Dense(1)
    ])
    self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(loss=tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE),
        metrics=[tf.keras.metrics.RootMeanSquaredError()])

  def compute_loss(self, features, training=False):
    embedding = self.embedding_layer({
        "user_id": features["user_id"],
        "movie_id": features["movie_id"]
    })
    rating_predictions = self.ratings(tf.concat([embedding["user_id"], embedding["movie_id"]], axis=1))

    # 我们需要将其缩小全局批量大小的一个因子，该因子等于每个副本的批量大小和strategy.num_replicas_in_sync的乘积
    return tf.reduce_sum(self.task(labels=features["user_rating"], predictions=rating_predictions)) * (
                1 / (per_replica_batch_size * strategy.num_replicas_in_sync))

  def call(self, features, serving_config=None):
    embedding = self.embedding_layer(
        {
            "user_id": features["user_id"],
            "movie_id": features["movie_id"]
        },
        serving_config=serving_config)
    return self.ratings(tf.concat([embedding["user_id"], embedding["movie_id"]], axis=1))

# 确保在TPUStrategy下初始化模型。
with strategy.scope():
  model = EmbeddingModel()
  model.compile(optimizer=optimizer)

# 训练模型
model.fit(distribute_train_dataset, steps_per_epoch=10, epochs=10)

# 测试数据集上评估模型
model.evaluate(distribute_test_dataset, steps=10)

# 训练完成之后，我们可以将模型保存到GCS存储桶之中
model_dir = os.path.join(gcs_bucket, "saved_model")

# 为TPU模型创建检查点并将模型保存到存储桶中。
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
saved_tpu_model_path = checkpoint.save(os.path.join(model_dir, "ckpt"))

# Restore the embedding on TPU
with strategy.scope():
  checkpoint.restore(saved_tpu_model_path)

# 我们还可以在CPU上恢复TPU训练的权重
cpu_model = EmbeddingModel()

# Create the cpu checkpoint and restore the tpu checkpoint.
cpu_checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=cpu_model)
cpu_checkpoint.restore(saved_tpu_model_path)

# 为saved_model提供服务
@tf.function
def serve_tensors(features):
  return cpu_model(features)

signatures = {
    'serving':
        serve_tensors.get_concrete_function(
            features={
                'movie_id':
                    tf.TensorSpec(shape=(1,), dtype=tf.int32, name='movie_id'),
                'user_id':
                    tf.TensorSpec(shape=(1,), dtype=tf.int32, name='user_id'),
            }),
}

tf.saved_model.save(
    cpu_model,
    export_dir=os.path.join(model_dir, 'exported_model'),
    signatures=signatures)

# 我们换可以将movie_id和user_id传递给加载的模型，然后获得评分预测。
imported = tf.saved_model.load(os.path.join(model_dir, 'exported_model'))
predict_fn = imported.signatures['serving']

# Dummy serving data.
input_batch = {
    'movie_id': tf.constant(np.array([100]), dtype=tf.int32),
    'user_id': tf.constant(np.array([30]), dtype=tf.int32)
}
# The prediction it generates.
prediction = predict_fn(**input_batch)['output_0']
```
