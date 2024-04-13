---
title: 电影推荐（TensorFlow Ranking）
date: 2024-04-12 11:20:11
tags:
  - AI
categories:
  - 人工智能
---

`TensorFlow Ranking`是一个开源库，用于开发可扩展的神经学习排名 (`LTR`) 模型。 排名模型通常用于搜索和推荐系统，但也已成功应用于各种领域，包括机器翻译、对话系统、`SAT`求解器、智能城市规划，甚至计算生物学。排名模型采用项目列表（网页、文档、产品、电影等）并以优化的顺序生成列表，例如最相关的项目位于顶部，最不相关的项目位于底部，通常应用于用户搜索：
{% asset_img rm_2.png %}
<!-- more -->

该库支持`LTR`模型的标准逐点、成对和列表损失函数。它还支持广泛的排名指标，包括平均倒数排名(`MRR`)和标准化贴现累积增益(`NDCG`)，因此您可以针对排名任务评估和比较这些方法。排名库还提供了由`Google`机器学习工程师研究、测试和构建的增强排名方法的函数。

`TensorFlow Ranking`库可帮助您构建可扩展的神经学习排名模型，使用最新研究中成熟的方法和技术对机器学习模型进行排名。排名模型采用相似项目的列表（例如网页），并生成这些项目的优化列表，例如最相关的页面与最不相关的页面。学习排序模型在搜索、问答、推荐系统和对话系统中都有应用。您可以使用此库通过`Keras API`加速为您的应用程序构建排名模型。排名库还提供工作流实用程序，使您可以更轻松地扩展模型实现，从而使用分布式处理策略有效地处理大型数据集。

#### BERT列表输入排序

`Ranking`库提供了`TFR-BERT`的实现，这是一种将`BERT`与`LTR`建模结合起来的评分架构，以优化列表输入的排序。作为此方法的一个示例应用程序，请考虑一个查询和一个包含`n`个文档的列表，您希望根据该查询对这些文档进行排名。`LTR`模型不是`<query, document>`对学习独立的`BERT`表示，而是应用排名损失来联合学习`BERT`表示，从而最大化整个排名列表相对于真实标签的效果。下图说明了这个过程，首先，我们将包含`n`个文档的列表展平，以响应查询列表`<query,document>`元组进行排名。这些元组被输入到预先训练的语言模型（例如`BERT`）中，然后，将整个文档列表的汇总`BERT`输出与`TF-Ranking`中专门的排名损失之一联合微调。我们的经验表明，这种`TFR-BERT`架构可以显著提高预训练怨言的性能，尤其是在集成多个预训练语言模型时。我们的用户现在可以使用这个简单的示例开始使用`TFR-BERT`。
{% asset_img rm_1.png %}

在传统的检索排名管道中，我们在检索阶段来过滤大量候选者。然后，仅将相关的候选者传递到排名阶段。需要检索阶段是因为候选池太大，如果你直接对所有项目进行排名，则需要花费太多的时间，并且你将受到延迟的影响。这就是为什么我们需要检索阶段来缩小项目的排名。但是，如果你一开始就没有很多项目，换句话说，如果你的排名阶段可以在延迟要求范围内对所有项目进行排名，该怎么办？还需要检索阶段吗？答案是否定的。在这种情况下你可以只进行排名而忽略检索。在工具方面 你可以使用：`TensorFlow Recommenders`进行逐点排名，也可以使用`TensorFlow Ranking`执行更复杂的排名。如果单独使用`TensorFlow Ranking`进行推荐，可以帮助你有效地对候选项目列表进行排名，并且在`Google`内部广泛使用。有了这个背景，我们来看看下面这个例子。我们使用`MovieLens 100K`数据集和`TF-Ranking`构建一个简单的两塔排名模型。我们可以使用这个模型根据给定用户的预测用户评分对电影进行排名和推荐。

#### 可解释的排名学习

透明度和可解释性是在排名系统中部署`LTR`模型的重要因素，排名系统可参与确定贷款资格评估、广告定位或指导医疗决策等流程的结果。在这种情况下，每个特征对最终排名的贡献应该是可检查和可理解的，以确保结果的透明度、问责制和公平性。实现这一目标的一种可能方法是使用广义加性模型（`GAM`）——本质上是"**可解释的机器学习模型**"，由各个特征的平滑函数线性组成。虽然`GAM`在回归和分类任务上得到了广泛的研究，但如何将它们应用到排名环境中还不太清楚。例如，`GAM`可以直接应用于对列表中的每个单独项目进行建模，但对项目交互和这些项目排序的上下文进行建模是一个更具挑战性的研究问题。为此，我们开发了神经排序`GAM`——广义加性模型对排序问题的扩展。与标准`GAM`不同，神经排序`GAM`可以考虑排序项目的特征和上下文特征（例如，查询或用户配置文件），导出可解释的紧凑模型。这确保了不仅每个项目级特征的贡献是可解释的，而且上下文特征的贡献也是可解释的。例如，在下图中，使用神经排名`GAM`可以清楚地看出在给定用户设备的背景下距离、价格和相关性如何影响酒店的最终排名。神经排名`GAM`现在可作为`TF-Ranking`的一部分。
{% asset_img rm_3.png %}

应用神经排序`GAM`进行本地搜索的示例。对于每个输入特征（例如价格、距离），子模型会生成可以检查的子分数，从而提供透明度。可以利用上下文特征（例如，用户设备类型）来导出子模型的重要性权重。

#### 神经排序还是梯度提升？

虽然神经模型已在多个领域实现了最好的性能，但像`LambdaMART`这样的专门梯度增强决策树(`GBDT`)仍然是各种开放`LTR`数据集的基准。`GBDT`在开放数据集中的成功有几个原因。首先，由于神经模型的规模相对较小，因此很容易在这些数据集上过度拟合。其次，由于`GBDT`使用决策树划分其输入特征空间，因此它们自然对排名数据中数值范围的变化更具弹性，这些数据通常包含`Zipfian`或其他倾斜分布的特征。然而，`GBDT`在更现实的排名场景中确实有其局限性，这些场景通常结合了文本和数字特征。例如，`GBDT`不能直接应用于大型离散特征空间，例如原始文档文本。一般来说，它们的可扩展性也低于**神经排序模型**。

#### 安装 & 导入包

安装并导入`TF-Ranking`库：
```bash
pip install -q tensorflow-ranking
pip install -q --upgrade tensorflow-datasets
```
```python
from typing import Dict, Tuple
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr
```
#### 读取数据

通过创建评级数据集和电影数据集来准备训练模型。使用`user_id`作为查询输入特征，`movie_title`作为文档输入特征，`user_rating`作为标签来训练排名模型。构建词汇表，将所有用户`ID`和所有电影标题转换为嵌入层的整数索引：
```python
gcs_utils._is_gcs_disabled = True
# Ratings data.
ratings = tfds.load('movielens/100k-ratings', split="train", try_gcs=False)
# Features of all the available movies.
movies = tfds.load('movielens/100k-movies', split="train", try_gcs=False)

# Select the basic features.
ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"]
})

# 我们为嵌入层构建了用户ID和电影标题词汇表
movies = movies.map(lambda x: x["movie_title"])
users = ratings.map(lambda x: x["user_id"])

user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
user_ids_vocabulary.adapt(users.batch(1000))

movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
movie_titles_vocabulary.adapt(movies.batch(1000))

# 接下来，我们按user_id对评分数据集进行分组，形成排名模型列表
key_func = lambda x: user_ids_vocabulary(x["user_id"])
reduce_func = lambda key, dataset: dataset.batch(100)
ds_train = ratings.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)

# 让我们看一下处理之后的训练数据集，user_id和列表都是四个或者五个，本例中有一个电影标题列表，以及四个或五个用户给出的各自评分
for x in ds_train.take(1):
  for key, value in x.items():
    print(f"Shape of {key}: {value.shape}")
    print(f"Example values of {key}: {value[:5].numpy()}")
    print()

# Shape of movie_title: (100,)
# Example values of movie_title: [b'Man Who Would Be King, The (1975)' b'Silence of the Lambs, The (1991)'
#  b'Next Karate Kid, The (1994)' b'2001: A Space Odyssey (1968)'
#  b'Usual Suspects, The (1995)']

# Shape of user_id: (100,)
# Example values of user_id: [b'405' b'405' b'405' b'405' b'405']

# Shape of user_rating: (100,)
# Example values of user_rating: [1. 4. 1. 5. 5.]

# 接下来我们定义一个辅助函数，来解压评分数据集并分离特征和标签，请注意这里使用的是：dense_to_ragged_batch方法。
# 因为user_id和电影标题列表有时会给出评分小于100的情况
def _features_and_labels(
    x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
  labels = x.pop("user_rating")
  return x, labels

ds_train = ds_train.map(_features_and_labels)
ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32))
```
`ds_train`中生成的`user_id`和`movie_title`张量的形状为`[32, None]`，其中第二个维度在大多数情况下为`100`，但列表中分组的项目少于`100`个时的批次除外。因此使用了研究不规则张量的模型。
```python
# 我们再看一下处理后的特征和标签。
for x, label in ds_train.take(1):
  for key, value in x.items():
    print(f"Shape of {key}: {value.shape}")
    print(f"Example values of {key}: {value[:3, :3].numpy()}")
    print()
  print(f"Shape of label: {label.shape}")
  print(f"Example values of label: {label[:3, :3].numpy()}")

# Shape of movie_title: (32, None)
# Example values of movie_title: [[b'Man Who Would Be King, The (1975)'
#   b'Silence of the Lambs, The (1991)' b'Next Karate Kid, The (1994)']
#  [b'Flower of My Secret, The (Flor de mi secreto, La) (1995)'
#   b'Little Princess, The (1939)' b'Time to Kill, A (1996)']
#  [b'Kundun (1997)' b'Scream (1996)' b'Power 98 (1995)']]

# Shape of user_id: (32, None)
# Example values of user_id: [[b'405' b'405' b'405']
#  [b'655' b'655' b'655']
#  [b'13' b'13' b'13']]

# Shape of label: (32, None)
# Example values of label: [[1. 4. 1.]
#  [3. 3. 3.]
#  [5. 1. 1.]]
```
#### 定义模型

继承`tf.keras.Model`并实现了`call`方法，构建排名模型：
```python
class MovieLensRankingModel(tf.keras.Model):
  # 我们在init函数中定义用户和电影嵌入
  def __init__(self, user_vocab, movie_vocab):
    super().__init__()

    # Set up user and movie vocabulary and embedding.
    self.user_vocab = user_vocab
    self.movie_vocab = movie_vocab
    self.user_embed = tf.keras.layers.Embedding(user_vocab.vocabulary_size(),64)
    self.movie_embed = tf.keras.layers.Embedding(movie_vocab.vocabulary_size(),64)

  # 在此方法中，我们计算用户和电影的嵌入点积
  def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
    # Define how the ranking scores are computed: 
    # Take the dot-product of the user embeddings with the movie embeddings.

    user_embeddings = self.user_embed(self.user_vocab(features["user_id"]))
    movie_embeddings = self.movie_embed(self.movie_vocab(features["movie_title"]))

    return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)
```
创建模型，然后使用排名`tfr.keras.losses`和`tfr.keras.metrics`进行编译，这是`TF-Ranking`包的核心。此示例使用特定于排名的`softmax`损失，这是一种列表损失，旨在提升排名列表中的所有相关项目，以更好的机会超越不相关的项目。与多级分类问题中的`softmax`损失（其中只有一类为正类，其余为负类）相反，`TF-Ranking`库支持查询列表中的多个相关文档和非二元相关标签。对于排名指标，此示例使用特定的标准化折扣累积增益(`NDCG`)和平均倒数排名(`MRR`)，它们计算具有位置折扣的排名查询列表的用户效用。有关排名指标的更多详细信息，审查评估措施离线指标。
```python
# Create the ranking model, trained with a ranking loss and evaluated with
# ranking metrics.
# 接下来我们创建模型并使用优化器、损失和指标进行编译，这里我们使用的是特定于排名的SOFTMAX_LOSS
# 与分类问题中的SOFTMAX_LOSS有所不同，其中只有一类是正类。其它都是负类
model = MovieLensRankingModel(user_ids_vocabulary, movie_titles_vocabulary)
optimizer = tf.keras.optimizers.Adagrad(0.5)
loss = tfr.keras.losses.get(loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
# 对于指标我们使用的是ndcg和mrr，都是常用指标
eval_metrics = [
    tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
    tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
]
model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)

# 训练模型
model.fit(ds_train, epochs=3)
# 可以看到损失正在下降
# Epoch 1/3
# 48/48 [==============================] - 9s 63ms/step - loss: 998.7556 - metric/ndcg: 0.8240 - metric/mrr: 1.0000
# Epoch 2/3
# 48/48 [==============================] - 4s 60ms/step - loss: 997.0884 - metric/ndcg: 0.9172 - metric/mrr: 1.0000
# Epoch 3/3
# 48/48 [==============================] - 4s 61ms/step - loss: 994.8118 - metric/ndcg: 0.9394 - metric/mrr: 1.0000

# 生成预测并评估
# 训练完之后，我们可以通过用户ID和电影标题来获得推荐，根据预测分数对结果进行排序，并将排名靠前的作为推荐项作为返回。
# Get movie title candidate list.
for movie_titles in movies.batch(2000):
  break

# Generate the input for user 42.
inputs = {
    "user_id":tf.expand_dims(tf.repeat("42", repeats=movie_titles.shape[0]), axis=0),
    "movie_title":tf.expand_dims(movie_titles, axis=0)
}

# Get movie recommendations for user 42.
scores = model(inputs)
titles = tfr.utils.sort_by_scores(scores,[tf.expand_dims(movie_titles, axis=0)])[0]
print(f"Top 5 recommendations for user 42: {titles[0, :5]}")
```
电影推荐结果显示为：
```bash
Top 5 recommendations for user 42: 
[
    b'Star Wars (1977)' 
    b'Liar Liar (1997)' 
    b'Toy Story (1995)'
    b'Raiders of the Lost Ark (1981)' 
    b'Sound of Music, The (1965)'
]
```
#### 结论

`TF Recommenders`和`TF Ranking`比较。`TF Recommenders`更专注于推荐系统，他包括专门为推荐器设计的工具和程序，例如检索和排名任务。`TF Ranking`专注于对项目进行排名，并且可以在推荐系统之外使用，例如文档搜索和问答。`TF Recommenders`和`TF Ranking`都是独立的库，但它们在推荐系统的排名阶段有交叉。如果您在进行文档搜索、问答等，只需使用`TF Ranking`；如果你正在构建推荐系统并需要检索阶段，请选择`TF Recommenders`；在推荐器中你可以选择其中任何一个。