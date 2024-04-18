---
title: ScaNN进行高效检索服务（TensorFlow 构建推荐系统）
date: 2024-04-18 18:40:11
tags:
  - AI
categories:
  - 人工智能
---

#### 什么是 ScanNN？

`ScaNN`是可扩展最近邻的缩写。在推荐系统的检索阶段，我们需要快速找到给定查询嵌入的最近数据集的嵌入。通常嵌入集对于穷举搜索来说往往太大。因此，需要像`ScaNN`这样的工具来进行近似邻域搜索。`ScaNN`于`2020.06`开源。他提供高效的向量相似性搜索，即从海量数据库中快速地匹配和检索相似项。他包括基于树的空间分区、非对称哈希、内涵和倒排索引等实现。由于这些高度优化的算法，`ScaNN`在大型和中等规模数据库的最近邻搜索方面提供了显著的加速。
<!-- more -->

#### 构建ScanNN支持的模型

为了在`TFRS`中尝试`ScaNN`，我们将构建一个简单的`MovieLens`检索模型。
```python
import os
import pprint
import tempfile
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
from typing import Dict, Text

# 加载数据
# Load the MovieLens 100K data.
ratings = tfds.load("movielens/100k-ratings",split="train")

# Get the ratings data.
ratings = (ratings
           # Retain only the fields we need.
           .map(lambda x: {"user_id": x["user_id"], "movie_title": x["movie_title"]})
           # Cache for efficiency.
           .cache(tempfile.NamedTemporaryFile().name)
)

# Get the movies data.
movies = tfds.load("movielens/100k-movies", split="train")
movies = (movies
          # Retain only the fields we need.
          .map(lambda x: x["movie_title"])
          # Cache for efficiency.
          .cache(tempfile.NamedTemporaryFile().name))

# 在构建模型之前，我们需要设置用户和电影词汇表：
user_ids = ratings.map(lambda x: x["user_id"])
unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(user_ids.batch(1000))))

# 我们还将设置训练和测试集：
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# 模型定义
# 我们构建了一个简单的两塔模型。
class MovielensModel(tfrs.Model):

  def __init__(self):
    super().__init__()

    embedding_dimension = 32
    # Set up a model for representing movies.
    self.movie_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=unique_movie_titles, mask_token=None),
      # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
    ])

    # Set up a model for representing users.
    self.user_model = tf.keras.Sequential([
      tf.keras.layers.StringLookup(vocabulary=unique_user_ids, mask_token=None),
        # We add an additional embedding to account for unknown tokens.
      tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
    ])

    # Set up a task to optimize the model and compute metrics.
    self.task = tfrs.tasks.Retrieval(
      metrics=tfrs.metrics.FactorizedTopK(
        candidates=(
            movies
            .batch(128)
            .cache()
            .map(lambda title: (title, self.movie_model(title)))
        )
      )
    )

  def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
    # We pick out the user features and pass them into the user model.
    user_embeddings = self.user_model(features["user_id"])
    # And pick out the movie features and pass them into the movie model,
    # getting embeddings back.
    positive_movie_embeddings = self.movie_model(features["movie_title"])

    # The task computes the loss and the metrics.
    return self.task(
        user_embeddings,
        positive_movie_embeddings,
        candidate_ids=features["movie_title"],
        compute_metrics=not training
    )

# 拟合与评估
model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

model.fit(train.batch(8192), epochs=3)
model.evaluate(test.batch(8192), return_dict=True)

# {'factorized_top_k/top_1_categorical_accuracy': 0.0013000000035390258,
#  'factorized_top_k/top_5_categorical_accuracy': 0.009949999861419201,
#  'factorized_top_k/top_10_categorical_accuracy': 0.021900000050663948,
#  'factorized_top_k/top_50_categorical_accuracy': 0.12484999746084213,
#  'factorized_top_k/top_100_categorical_accuracy': 0.23215000331401825,
#  'loss': 28276.328125,
#  'regularization_loss': 0,
#  'total_loss': 28276.328125
# }

# 模型预测
# 检索最佳候选者的最直接方法是通过暴力来完成：计算所有可能的电影的用户电影分数，对它们进行排序，然后选择几个最佳推荐。
# 在TFRS中，这是通过BruteForce层完成的：
brute_force = tfrs.layers.factorized_top_k.BruteForce(model.user_model)
brute_force.index_from_dataset(movies.batch(128).map(lambda title: (title, model.movie_model(title))))

# 一旦创建并填充了候选者，我们就可以调用它来获得预测：
# Get predictions for user 42.
_, titles = brute_force(np.array(["42"]), k=3)

print(f"Top recommendations: {titles[0]}")

# Top recommendations: [b'Angels in the Outfield (1994)' b"Kid in King Arthur's Court, A (1995)" b'Bedknobs and Broomsticks (1971)']

# 在包含 1000 部电影以下的小型数据集上，速度非常快：
%timeit _, titles = brute_force(np.array(["42"]), k=3)

# 1.65 ms ± 6.42 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

# 但如果我们有更多候选人——数百万而不是数千，会发生什么？我们可以通过多次索引所有电影来模拟这一点：
# Construct a dataset of movies that's 1,000 times larger. We 
# do this by adding several million dummy movie titles to the dataset.
lots_of_movies = tf.data.Dataset.concatenate(
    movies.batch(4096),
    movies.batch(4096).repeat(1_000).map(lambda x: tf.zeros_like(x))
)

# We also add lots of dummy embeddings by randomly perturbing
# the estimated embeddings for real movies.
lots_of_movies_embeddings = tf.data.Dataset.concatenate(
    movies.batch(4096).map(model.movie_model),
    movies.batch(4096).repeat(1_000)
      .map(lambda x: model.movie_model(x))
      .map(lambda x: x * tf.random.uniform(tf.shape(x)))
)

# 我们可以在这个更大的数据集上构建BruteForce索引：
brute_force_lots = tfrs.layers.factorized_top_k.BruteForce()
brute_force_lots.index_from_dataset(tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings)))

_, titles = brute_force_lots(model.user_model(np.array(["42"])), k=3)
print(f"Top recommendations: {titles[0]}")

# Top recommendations: [b'Angels in the Outfield (1994)' b"Kid in King Arthur's Court, A (1995)" b'Bedknobs and Broomsticks (1971)']

# 但他们需要更长的时间。对于100万部电影的候选集，暴力预测变得相当慢：
%timeit _, titles = brute_force_lots(model.user_model(np.array(["42"])), k=3)

# 4.03 ms ± 24.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# 随着候选者数量的增加，所需的时间也呈线性增长：如果有1000万候选者，为顶级候选者提供服务将需要250毫秒。这对于实时服务来说显然太慢了。
# 在TFRS中使用ScaNN是通过tfrs.layers.factorized_top_k.ScaNN层完成的。它遵循与其他前k层相同的接口：
scann = tfrs.layers.factorized_top_k.ScaNN(num_reordering_candidates=500,num_leaves_to_search=30)
scann.index_from_dataset(tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings)))

_, titles = scann(model.user_model(np.array(["42"])), k=3)

print(f"Top recommendations: {titles[0]}")

# Top recommendations: [b'Angels in the Outfield (1994)' b"Kid in King Arthur's Court, A (1995)" b'Bedknobs and Broomsticks (1971)']
# 它们的计算速度要快得多：
%timeit _, titles = scann(model.user_model(np.array(["42"])), k=3)

# 22.4 ms ± 44 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# 在这种情况下，我们可以在大约2毫秒内从大约100万部电影中检索出排名前3的电影：比通过暴力计算最佳候选电影快15倍。对于较大的数据集，近似方法的优势甚至更大。
```
#### 评估近似值

当使用近似前`K`个检索机制（例如`ScaNN`）时，检索速度通常是以牺牲准确性为代价的。为了理解这种权衡，在使用`ScaNN`时测量模型的评估指标并将其与基线进行比较非常重要。幸运的是，`TFRS`让这一切变得简单。我们只需使用`ScaNN`的指标覆盖检索任务的指标，重新编译模型并运行评估。为了进行比较，我们首先运行基线结果。仍然需要覆盖我们的指标，以确保它们使用扩大的候选集而不是原始的电影集：
```python
# Override the existing streaming candidate source.
model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(
    candidates=tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings))
)
# Need to recompile the model for the changes to take effect.
model.compile()

%time baseline_result = model.evaluate(test.batch(8192), return_dict=True, verbose=False)

# CPU times: user 24min 23s, sys: 2min, total: 26min 23s
# Wall time: 3min 35s

# 我们可以使用ScaNN做同样的事情：
model.task.factorized_metrics = tfrs.metrics.FactorizedTopK(candidates=scann)
model.compile()

# We can use a much bigger batch size here because ScaNN evaluation
# is more memory efficient.
%time scann_result = model.evaluate(test.batch(8192), return_dict=True, verbose=False)

# CPU times: user 15.6s, sys: 633ms, total: 16.3s
# Wall time: 1.95s

# 基于ScaNN的评估速度要快很多。对于更大的数据集，这种优势会变得更大，因此对于大型数据集，明智的做法是始终运行基于ScaNN的评估以提高模型开发速度。
print(f"Brute force top-100 accuracy: {baseline_result['factorized_top_k/top_100_categorical_accuracy']:.2f}")
print(f"ScaNN top-100 accuracy:       {scann_result['factorized_top_k/top_100_categorical_accuracy']:.2f}")

# Brute force top-100 accuracy: 0.15
# ScaNN top-100 accuracy:       0.14
```
这表明在这个人工数据库上，近似值几乎没有损失。一般来说，所有近似方法都表现出速度与精度的权衡。要更深入地了解这一点，您可以查看`Erik Bernhardsson`的`ANN`基准。

#### 部署近似模型

```python
# 我们可以将其保存为SavedModel对象
# We re-index the ScaNN layer to include the user embeddings in the same model.
# This way we can give the saved model raw features and get valid predictions back.
scann = tfrs.layers.factorized_top_k.ScaNN(model.user_model, num_reordering_candidates=1000)
scann.index_from_dataset(
    tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings))
)

# Need to call it to set the shapes.
_ = scann(np.array(["42"]))
with tempfile.TemporaryDirectory() as tmp:
  path = os.path.join(tmp, "model")
  tf.saved_model.save(
      scann,
      path,
      options=tf.saved_model.SaveOptions(namespace_whitelist=["Scann"])
  )
  loaded = tf.saved_model.load(path)

# 然后加载它并提供服务，得到完全相同的结果：
_, titles = loaded(tf.constant(["42"]))

print(f"Top recommendations: {titles[0][:3]}")

# Top recommendations: [b'Angels in the Outfield (1994)' b"Kid in King Arthur's Court, A (1995)" b'Rudy (1993)']
```
生成的模型可以部署在安装了`TensorFlow`和`ScaNN的``Python`的服务器中提供服务。您还可以从`Dockerfile`自行构建映像。

#### 调整ScaNN

现在让我们研究一下调整`ScaNN`层以获得更好的性能/准确性权衡。为了做到这一点，我们首先需要测量我们的基线性能和准确性。从上面来看，我们已经测量了模型处理单个（非批量）查询的延迟。现在我们需要研究`ScaNN`的准确性，我们通过召回来衡量它。让我们计算当前`ScaNN`搜索器的召回率。首先，我们需要生成`brute force`、`ground truth top-k`：
```python
# Process queries in groups of 1000; processing them all at once with brute force
# may lead to out-of-memory errors, because processing a batch of q queries against
# a size-n dataset takes O(nq) space with brute force.
titles_ground_truth = tf.concat([
  brute_force_lots(queries, k=10)[1] for queries in
  test.batch(1000).map(lambda x: model.user_model(x["user_id"]))], axis=0)

# 我们的变量titles_ground_truth现在包含通过暴力检索返回的前10名电影推荐。现在我们可以在使用ScaNN时计算相同的推荐：
# Get all user_id's as a 1d tensor of strings
test_flat = np.concatenate(list(test.map(lambda x: x["user_id"]).batch(1000).as_numpy_iterator()), axis=0)

# ScaNN is much more memory efficient and has no problem processing the whole
# batch of 20000 queries at once.
_, titles = scann(test_flat, k=10)

# 我们定义计算召回率的函数。对于每个查询，它会计算暴力破解结果与ScaNN结果的交集中有多少个结果，并将其除以暴力破解结果的数量。 
# 所有查询的平均数量就是我们的召回率。
def compute_recall(ground_truth, approx_results):
  return np.mean([
      len(np.intersect1d(truth, approx)) / len(truth)
      for truth, approx in zip(ground_truth, approx_results)
  ])

# 这为我们提供了当前ScaNN配置的基线召回@10： 
print(f"Recall: {compute_recall(titles_ground_truth, titles):.3f}")

# Recall: 0.938

# 我们还可以测量基线延迟：
%timeit -n 1000 scann(np.array(["42"]), k=10)

# 21.9 ms ± 30.5 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```
为此，我们需要一个模型来说明`ScaNN`的调节旋钮如何影响性能。我们当前的模型使用`ScaNN`的`tree-AH`算法。该算法对嵌入数据库（“树”）进行分区，然后使用AH对这些分区中最有希望的进行评分，`AH`是一种高度优化的近似距离计算例程。`TensorFlow Recommenders`的`ScaNN Keras`层的默认参数设置`num_leaves=100`和`num_leaves_to_search=10`。这意味着我们的数据库被划分为`100`个不相交的子集，并且这些分区中最有希望的`10`个用`AH`进行评分。这意味着`10/100=10%`的数据集正在使用`AH`进行搜索。

例如，如果`num_leaves=1000`且`num_leaves_to_search=100`，我们还将使用`AH`搜索数据库的`10%`。然而，与之前的设置相比，我们要搜索的`10%`将包含更高质量的候选者，因为更高的`num_leaves`允许我们对数据集的哪些部分值得搜索做出更细粒度的决策。当`num_leaves=1000`和`num_leaves_to_search=100`时，我们获得了更高的召回率：
```python
scann2 = tfrs.layers.factorized_top_k.ScaNN(
    model.user_model, 
    num_leaves=1000,
    num_leaves_to_search=100,
    num_reordering_candidates=1000)
scann2.index_from_dataset(tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings)))
_, titles2 = scann2(test_flat, k=10)

print(f"Recall: {compute_recall(titles_ground_truth, titles2):.3f}")

# Recall: 0.974

# 然而，作为权衡，我们的延迟也增加了。 
# 这是因为分区步骤变得更加昂贵；scann选择100个分区中的前10个，而scann2选择1000个分区中的前100个。
# 后者更昂贵，因为它需要查看10倍的分区。
%timeit -n 1000 scann2(np.array(["42"]), k=10)

# 22 ms ± 32.4 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)
```
一般来说，调整`ScaNN`搜索就是选择正确的权衡。每个单独的参数更改通常不会使搜索更快、更准确；我们的目标是调整参数在这两个相互冲突的目标之间进行最佳权衡。在我们的例子中，`scann2`相对于`scann`显着提高了召回率，但付出了一定的延迟代价。我们能否调低一些其他旋钮来减少延迟，同时保留我们的大部分召回优势？让我们尝试使用`AH`搜索数据集的`70/1000=7%`，并且仅对最后400个候选者重新评分：
```python
scann3 = tfrs.layers.factorized_top_k.ScaNN(model.user_model,num_leaves=1000,num_leaves_to_search=70,num_reordering_candidates=400)
scann3.index_from_dataset(tf.data.Dataset.zip((lots_of_movies, lots_of_movies_embeddings)))
_, titles3 = scann3(test_flat, k=10)

print(f"Recall: {compute_recall(titles_ground_truth, titles3):.3f}")

# Recall: 0.969

# scann3比scann提供约3%的绝对召回率增益，同时还提供更低的延迟：
%timeit -n 1000 scann3(np.array(["42"]), k=10)

# 21.9 ms ± 26.3 µs per loop (mean ± std. dev. of 7 runs, 1,000 loops each)

```
这些旋钮可以进一步调整，优化`accuracy-performance pareto frontier`的不同方面。`ScaNN`的算法可以在范围更广的召回目标上实现最优的性能。`ScaNN`使用先进的**矢量量化技术**和高度优化来实现。