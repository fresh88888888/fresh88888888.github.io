---
title: 深度与交叉网络（TensorFlow 构建推荐系统）
date: 2024-04-18 14:40:11
tags:
  - AI
categories:
  - 人工智能
---

#### 介绍

**深度与交叉网络**（`DCN`）旨在有效地学习显式的、有边界的交叉特征，你已经知道大而稀疏的特征空间极难训练。通常我们执行**特征工程**，包括设计交叉特征，这是非常具有挑战性且效率低下的。虽然在这种情况下可以使用其它神经网络，但这并不是最有效的办法。深度与交叉网络是专门为应对这一挑战而设计的。在深入了解DCN之前，我们首先花一点时间回顾一下，什么是特征交叉？假设我们正在构建一个推荐系统来向客户销售搅拌机，那么我们客户过去的购买历史记录，例如购买的香蕉和购买的烹饪书籍或地理特征都是单一特征。如果一个人同时购买了香蕉和烹饪书籍，那么该客户将更有可能点击推荐的搅拌机。购买的香蕉和购买的烹饪书籍的组合被称为特征交叉，它提供了除单个特征之外附加交互信息。你可以添加更多交叉特征。在真实的推荐系统当中，我们通常拥有大而稀疏的特征空间，因此，在这种情况下识别有效的特征，通常需要执行特征工程或详尽的搜索，这是非常低效的。
<!-- more -->
{% asset_img dcn_1.png %}

为了解决这个问题，提出了**深度与交叉网络**（`DCN`）。它从输入层（通常是嵌入层）开始，然后包含多个交叉层的交叉网络，这些交叉层对显式特征交互进行建模，然后对隐式交叉进行建模的深层网络相结合。深层网络只是传统的多层结构。
{% asset_img dcn_2.png %}

但DCN的核心是交叉网络。它明确地在每一层应用特征交叉并且最高多项式次数随着层深的增加而增加，上面的图以数学形式显示了。有多种方法可以将交叉网络和深度网络结合起来，我们可以将深度网络堆叠在交叉网络之上。
{% asset_img dcn_3.png %}

或者我们可以将它们并行放置，正如下图看到的那样，并将深度和交叉网络各自堆叠，再将他们层层连接起来：
{% asset_img dcn_4.png %}

深度网络是传统的前馈多层感知器（`MLP`）。下面我们将首先通过一个示例展示`DCN`的优势，然后我们将引导您了解使用`MovieLen-1M`数据集利用`DCN`的一些常见方法。为了说明`DCN`的优势，让我们通过一个简单的示例来说明。假设我们有一个数据集，我们试图对客户点击搅拌机广告的可能性进行建模，其特征和标签如下所述。
|特征/标签|描述|值类型/范围|
|:--|:--|:--|
|$x_1$=country|该客户居住的国家/地区|Int in [0, 199]|
|$x_2$=bananas|顾客购买的香蕉|Int in [0, 23]|
|$x_3=$|顾客购买的烹饪书籍|Int in [0, 5]|
|$y$|点击搅拌机广告的可能性|--|

然后，我们让数据遵循以下分布：
$$
y= f(x_1, x_2, x_3) = 0.1x_1 + 0.4x_2 + 0.7x_3 + 0.1x_1x_2 + 0.1x_3^2
$$
我们首先定义$f(x_1, x_2, x_3)$：
```python
def get_mixer_data(data_size=100_000, random_seed=42):
  # We need to fix the random seed
  # to make colab runs repeatable.
  rng = np.random.RandomState(random_seed)
  country = rng.randint(200, size=[data_size, 1]) / 200.
  bananas = rng.randint(24, size=[data_size, 1]) / 24.
  cookbooks = rng.randint(6, size=[data_size, 1]) / 6.

  x = np.concatenate([country, bananas, cookbooks], axis=1)
  # # Create 1st-order terms.
  y = 0.1 * country + 0.4 * bananas + 0.7 * cookbooks
  # Create 2nd-order cross terms.
  y += 0.1 * country * bananas + 3.1 * bananas * cookbooks + (0.1 * cookbooks * cookbooks)

  return x, y

# 让我们生成服从分布的数据，并将数据分为90%用于训练，10%用于测试。
x, y = get_mixer_data()
num_train = 90000
train_x = x[:num_train]
train_y = y[:num_train]
eval_x = x[num_train:]
eval_y = y[num_train:]
```
#### 定义模型

由于我们刚刚创建的数据仅包含二阶特征交互，因此用单层交叉网络来说明就足够了。如果我们想要对高阶特征交互进行建模，我们可以堆叠多个交叉层并使用多层交叉网络。我们将构建的两个模型是：
- 交叉网络，只有一个交叉层；
- 具有更宽更深的`ReLU`层的深度网络。

我们首先构建一个统一的模型类，其损失是均方误差。
```python
class Model(tfrs.Model):

  def __init__(self, model):
    super().__init__()
    self._model = model
    self._logit_layer = tf.keras.layers.Dense(1)

    self.task = tfrs.tasks.Ranking(
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
    )

  def call(self, x):
    x = self._model(x)
    return self._logit_layer(x)

  def compute_loss(self, features, training=False):
    x, labels = features
    scores = self(x)

    return self.task(labels=labels,predictions=scores,)

# 然后，我们指定交叉网络（3个交叉层）和基于ReLU的DNN（层大小为 [512, 256, 128]）
crossnet = Model(tfrs.layers.dcn.Cross())
deepnet = Model(
    tf.keras.Sequential([
      tf.keras.layers.Dense(512, activation="relu"),
      tf.keras.layers.Dense(256, activation="relu"),
      tf.keras.layers.Dense(128, activation="relu")
    ])
)

```
#### 模型训练

现在我们已经准备好了数据和模型，我们将训练模型。我们首先对数据进行混洗和批处理，为模型训练做准备。
```python
train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(1000)
eval_data = tf.data.Dataset.from_tensor_slices((eval_x, eval_y)).batch(1000)

# 然后，我们定义训练次数以及学习率。
epochs = 100
learning_rate = 0.4

# 如果您想查看模型的进展情况，可以设置 verbose=True。
crossnet.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))
crossnet.fit(train_data, epochs=epochs, verbose=False)

deepnet.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate))
deepnet.fit(train_data, epochs=epochs, verbose=False)
```
#### 模型评估

我们在评估数据集上验证模型性能并报告均方根误差（`RMSE`，越低越好）。
```python
crossnet_result = crossnet.evaluate(eval_data, return_dict=True, verbose=False)
print(f"CrossNet(1 layer) RMSE is {crossnet_result['RMSE']:.4f} f"using {crossnet.count_params()} parameters.")

deepnet_result = deepnet.evaluate(eval_data, return_dict=True, verbose=False)
print(f"DeepNet(large) RMSE is {deepnet_result['RMSE']:.4f} "f"using {deepnet.count_params()} parameters.")

# CrossNet(1 layer) RMSE is 0.0001 using 16 parameters.
# DeepNet(large) RMSE is 0.0933 using 166401 parameters.
```
我们看到，跟基于`ReLU`的`DNN`相比，交叉网络的`RMSE`显着降低，且参数数量较少。这表明交叉网络在学习特征交叉方面的效果更好。

#### 模型解读

我们已经知道哪些特征交叉的数据中很重要，检查我们的模型是否确实学习了重要的特征交叉很有趣。这可以通过在`DCN`中可视化学习到的**权重矩阵**来完成。权重$W_{ij}$表示特征之间交互的学习重要性$x_i$和$x_j$。
```python
mat = crossnet._model._dense.kernel
features = ["country", "purchased_bananas", "purchased_cookbooks"]

plt.figure(figsize=(9,9))
im = plt.matshow(np.abs(mat.numpy()), cmap=plt.cm.Blues)
ax = plt.gca()
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=10) 
_ = ax.set_xticklabels([''] + features, rotation=45, fontsize=10)
_ = ax.set_yticklabels([''] + features, fontsize=10)
```
{% asset_img dcn_5.png %}

较深的颜色代表更强的学习交互-在这种情况下，很明显，从模型了解到“一起购买香蕉和食谱很重要”。

#### Movielens 1M示例 

我们现在检查`DCN`在真实数据集上的有效性：`Movielens 1M`。 `Movielens 1M`是用于推荐研究的流行数据集。它根据用户相关特征和电影相关特征来预测用户对电影的评分。我们使用此数据集来演示使用`DCN`的一些常见方法。
```python
# 数据预处理
ratings = tfds.load("movie_lens/100k-ratings", split="train")
ratings = ratings.map(lambda x: {
    "movie_id": x["movie_id"],
    "user_id": x["user_id"],
    "user_rating": x["user_rating"],
    "user_gender": int(x["user_gender"]),
    "user_zip_code": x["user_zip_code"],
    "user_occupation_text": x["user_occupation_text"],
    "bucketized_user_age": int(x["bucketized_user_age"]),
})

# 接下来，我们将数据随机分为80%用于训练，20%用于测试。
tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

# 然后，我们为每个特征创建词汇表。
vocabularies = {}
feature_names = ["movie_id", "user_id", "user_gender", "user_zip_code","user_occupation_text", "bucketized_user_age"]

for feature_name in feature_names:
  vocab = ratings.batch(1_000_000).map(lambda x: x[feature_name])
  vocabularies[feature_name] = np.unique(np.concatenate(list(vocab)))

# 定义模型
# 我们将构建的模型架构从嵌入层开始，该嵌入层被输入到交叉网络中，然后是深度网络。 
# 所有特征的嵌入维度均设置为32。您还可以针对不同的特征使用不同的嵌入大小。
class DCN(tfrs.Model):

  def __init__(self, use_cross_layer, deep_layer_sizes, projection_dim=None):
    super().__init__()

    self.embedding_dimension = 32
    str_features = ["movie_id", "user_id", "user_zip_code","user_occupation_text"]
    int_features = ["user_gender", "bucketized_user_age"]
    self._all_features = str_features + int_features
    self._embeddings = {}

    # Compute embeddings for string features.
    for feature_name in str_features:
      vocabulary = vocabularies[feature_name]
      self._embeddings[feature_name] = tf.keras.Sequential(
          [tf.keras.layers.StringLookup(
              vocabulary=vocabulary, mask_token=None),
           tf.keras.layers.Embedding(len(vocabulary) + 1,
                                     self.embedding_dimension)
    ])

    # Compute embeddings for int features.
    for feature_name in int_features:
      vocabulary = vocabularies[feature_name]
      self._embeddings[feature_name] = tf.keras.Sequential(
          [tf.keras.layers.IntegerLookup(
              vocabulary=vocabulary, mask_value=None),
           tf.keras.layers.Embedding(len(vocabulary) + 1,
                                     self.embedding_dimension)
    ])

    if use_cross_layer:
      self._cross_layer = tfrs.layers.dcn.Cross(
          projection_dim=projection_dim,
          kernel_initializer="glorot_uniform")
    else:
      self._cross_layer = None

    self._deep_layers = [tf.keras.layers.Dense(layer_size, activation="relu")
      for layer_size in deep_layer_sizes]

    self._logit_layer = tf.keras.layers.Dense(1)

    self.task = tfrs.tasks.Ranking(
      loss=tf.keras.losses.MeanSquaredError(),
      metrics=[tf.keras.metrics.RootMeanSquaredError("RMSE")]
    )

  def call(self, features):
    # Concatenate embeddings
    embeddings = []
    for feature_name in self._all_features:
      embedding_fn = self._embeddings[feature_name]
      embeddings.append(embedding_fn(features[feature_name]))

    x = tf.concat(embeddings, axis=1)

    # Build Cross Network
    if self._cross_layer is not None:
      x = self._cross_layer(x)

    # Build Deep Network
    for deep_layer in self._deep_layers:
      x = deep_layer(x)

    return self._logit_layer(x)

  def compute_loss(self, features, training=False):
    labels = features.pop("user_rating")
    scores = self(features)
    return self.task(
        labels=labels,
        predictions=scores,
    )

# 模型训练
# 我们对训练和测试数据进行混洗、批处理和缓存。
cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()

# 让我们定义一个函数，该函数多次运行模型并返回多次运行中模型的RMSE平均值和标准差。
def run_models(use_cross_layer, deep_layer_sizes, projection_dim=None, num_runs=5):
  models = []
  rmses = []

  for i in range(num_runs):
    model = DCN(use_cross_layer=use_cross_layer,deep_layer_sizes=deep_layer_sizes,projection_dim=projection_dim)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate))
    models.append(model)
    model.fit(cached_train, epochs=epochs, verbose=False)
    metrics = model.evaluate(cached_test, return_dict=True)
    rmses.append(metrics["RMSE"])

  mean, stdv = np.average(rmses), np.std(rmses)

  return {"model": models, "mean": mean, "stdv": stdv}


epochs = 8
learning_rate = 0.01

# 我们首先训练一个具有堆叠结构的DCN模型，将输入到交叉网络，然后输入深度网络。
dcn_result = run_models(use_cross_layer=True, deep_layer_sizes=[192, 192])

# 为了降低训练和服务成本，我们利用低秩技术来近似DCN权重矩阵。 
# 排名通过参数projection_dim传入；较小的projection_dim导致较低的成本。 
# 请注意，projection_dim < (input size)/2 以降低成本。 
# 在实践中，我们观察到使用秩（input size）/4 的低秩DCN始终保持全秩DCN的准确性。
dcn_lr_result = run_models(use_cross_layer=True, projection_dim=20, deep_layer_sizes=[192, 192])

# 我们训练一个相同大小的深度神经网络(DNN)模型作为参考。
dnn_result = run_models(use_cross_layer=False, deep_layer_sizes=[192, 192, 192])

# 我们根据测试数据评估模型，并重复5次运行的平均值和标准差。
print("DCN            RMSE mean: {:.4f}, stdv: {:.4f}".format(dcn_result["mean"], dcn_result["stdv"]))
print("DCN (low-rank) RMSE mean: {:.4f}, stdv: {:.4f}".format(dcn_lr_result["mean"], dcn_lr_result["stdv"]))
print("DNN            RMSE mean: {:.4f}, stdv: {:.4f}".format(dnn_result["mean"], dnn_result["stdv"]))

# DCN            RMSE mean: 0.9326, stdv: 0.0015
# DCN (low-rank) RMSE mean: 0.9329, stdv: 0.0022
# DNN            RMSE mean: 0.9350, stdv: 0.0032
```
我们看到`DCN`比具有`ReLU`层的相同大小的`DNN`取得了更好的性能。此外，低秩`DCN`能够在保持精度的同时减少参数。除了块范数之外，我们还可以可视化整个矩阵，或每个块的平均值/中值/最大值。
```python
model = dcn_result["model"][0]
mat = model._cross_layer._dense.kernel
features = model._all_features
block_norm = np.ones([len(features), len(features)])
dim = model.embedding_dimension

# Compute the norms of the blocks.
for i in range(len(features)):
  for j in range(len(features)):
    block = mat[i * dim:(i + 1) * dim,j * dim:(j + 1) * dim]
    block_norm[i,j] = np.linalg.norm(block, ord="fro")

plt.figure(figsize=(9,9))
im = plt.matshow(block_norm, cmap=plt.cm.Blues)
ax = plt.gca()
divider = make_axes_locatable(plt.gca())
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax)
cax.tick_params(labelsize=10) 
_ = ax.set_xticklabels([""] + features, rotation=45, ha="left", fontsize=10)
_ = ax.set_yticklabels([""] + features, fontsize=10)
```
{% asset_img dcn_6.png %}

如果您有兴趣了解更多信息，可以查看两篇相关论文：`DCN-v1-paper、DCN-v2-paper`。
