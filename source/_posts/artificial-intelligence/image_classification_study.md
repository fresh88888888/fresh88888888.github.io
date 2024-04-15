---
title: 图像分类（KerasCV）
date: 2024-04-15 18:36:11
tags:
  - AI
categories:
  - 人工智能
---

**图像分类**是预测输入图像的分类标签的过程。虽然分类是一项相对简单的计算机视觉任务，但仍然由几个复杂的组件组成。幸运的是，`KerasCV`提供了`API`来构建常用组件。本例中主要演示了 `KerasCV`的模块化方法来解决三个复杂的图像分类问题：
- 使用预训练分类器进行推理。
- 微调预训练的骨干网络。
- 从头开始训练图像分类器。
<!-- more -->

#### 包导入

```python
import os
import json
import math
import numpy as np
import keras
from keras import losses
from keras import ops
from keras import optimizers
from keras.optimizers import schedules
from keras import metrics
import keras_cv
import tensorflow as tf
import tensorflow_datasets as tfds

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]
```

#### 使用预训练分类器进行推理

```python
classifier = keras_cv.models.ImageClassifier.from_preset("efficientnetv2_b0_imagenet_classifier")

image = keras.utils.load_img("dog.jpg")
image = np.array(image)
keras_cv.visualization.plot_image_gallery(np.array([image]), rows=1, cols=1, value_range=(0, 255), show=True, scale=4)

predictions = classifier.predict(np.expand_dims(image, axis=0))
# 预测以softmax类别排名的形式出现。我们可以使用简单的argsort函数找到顶级类的索引：
top_classes = predictions[0].argsort(axis=-1)

# 为了解码类映射，我们可以构建从类别索引到ImageNet类名的映射。
classes = keras.utils.get_file(
    origin="https://gist.githubusercontent.com/LukeWood/62eebcd5c5c4a4d0e0b7845780f76d55/raw/fde63e5e4c09e2fa0a3436680f436bdcb8325aac/ImagenetClassnames.json"
)
with open(classes, "rb") as f:
    classes = json.load(f)

# 现在我们可以通过索引简单地查找类名
top_two = [classes[str(i)] for i in top_classes[-2:]]
print("Top two classes are:", top_two)

# Top two classes are: ['Egyptian cat', 'velvet']
```
#### 微调预训练分类器

微调自定义分类器可以提高性能。如果我们想训练猫狗分类器，使用显式标记的猫狗数据应该比通用分类器表现更好！对于许多任务，没有相关的预训练模型可用（例如，对特定于您的应用程序的图像进行分类）。
```python
BATCH_SIZE = 32
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()

data, dataset_info = tfds.load("cats_vs_dogs", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
train_dataset = data["train"]

num_classes = dataset_info.features["label"].num_classes

resizing = keras_cv.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)

def preprocess_inputs(image, label):
    image = tf.cast(image, tf.float32)
    # Staticly resize images as we only iterate the dataset once.
    return resizing(image), tf.one_hot(label, num_classes)

# Shuffle the dataset to increase diversity of batches.
# 10*BATCH_SIZE follows the assumption that bigger machines can handle bigger
# shuffle buffers.
train_dataset = train_dataset.shuffle(
10 * BATCH_SIZE, reshuffle_each_iteration=True
).map(preprocess_inputs, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

images = next(iter(train_dataset.take(1)))[0]
keras_cv.visualization.plot_image_gallery(images, value_range=(0, 255))
```
接下来让我们构建我们的模型。预设名称中使用`imagenet`表示主干网络已在`ImageNet`数据集上进行了预训练。预训练的主干网络利用从更大的数据集中提取的模式，从我们的标记示例中提取更多信息。接下来让我们组装我们的分类器：
```python
model = keras_cv.models.ImageClassifier.from_preset(
    "efficientnetv2_b0_imagenet", num_classes=2
)
model.compile(
    loss="categorical_crossentropy",
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)
model.fit(train_dataset)

# 让我们看看我们的模型在微调后的表现如何
predictions = model.predict(np.expand_dims(image, axis=0))
classes = {0: "cat", 1: "dog"}
print("Top class is:", classes[predictions[0].argmax()])

# 1/1 ━━━━━━━━━━━━━━━━━━━━ 3s 3s/step
# Top class is: cat
```
#### 从头开始训练分类器

让我们完成最后一项任务：从头开始训练分类模型！我们使用`CalTech 101`图像分类数据集。虽然我们在本指南中使用更简单的`CalTech 101`数据集，但可以在`ImageNet`上使用相同的训练模板来获得接近最先进的分数。
```python
NUM_CLASSES = 101
# Change epochs to 100~ to fully train.
EPOCHS = 1

def package_inputs(image, label):
    return {"images": image, "labels": tf.one_hot(label, NUM_CLASSES)}


train_ds, eval_ds = tfds.load(
    "caltech101", split=["train", "test"], as_supervised="true"
)
train_ds = train_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(package_inputs, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.shuffle(BATCH_SIZE * 16)

# CalTech101数据集的每个图像都有不同的大小，因此我们使用ragged_batch() API将它们批处理在一起，同时维护每个图像的形状信息。
train_ds = train_ds.ragged_batch(BATCH_SIZE)
eval_ds = eval_ds.ragged_batch(BATCH_SIZE)

batch = next(iter(train_ds.take(1)))
image_batch = batch["images"]
label_batch = batch["labels"]

keras_cv.visualization.plot_image_gallery(
    image_batch.to_tensor(),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```

#### 数据增强

在我们之前的微调示例中，我们执行了静态调整大小操作，并且没有使用任何图像增强。这是因为一次通过训练集就足以取得不错的结果。当训练解决更困难的任务时，您需要在数据管道中包含数据增强。数据增强是一种使模型对输入数据（例如光照、裁剪和方向）变化具有鲁棒性的技术。`KerasCV`在`keras_cv.layers API`中包含一些最有用的增强功能。创建最佳的增强管道是一门艺术，但在本指南的这一部分中，我们将提供一些有关分类最佳实践的提示。关于图像数据增强需要注意的一个警告是，您必须小心，不要将增强的数据分布偏离原始数据分布太远。目标是防止过度拟合并提高泛化能力，但完全不符合数据分布的样本只会给训练过程增加噪声。我们将使用的第一个增强是`RandomFlip`。这种增强的行为或多或少与您所期望的一样：它要么翻转图像，要么不翻转图像。虽然这种增强在`CalTech101`和`ImageNet`中很有用，但应该注意的是，它不应该用于数据分布不是垂直镜像不变的任务。发生这种情况的数据集的一个示例是`MNIST`手写数字。将6翻转到垂直轴将使数字看起来更像`7`而不是`6`，但标签仍会显示`6`。