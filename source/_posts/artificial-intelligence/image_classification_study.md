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

在我们之前的微调示例中，我们执行了静态调整大小的操作，并且没有使用任何图像增强。这是因为一次通过训练集就足以取得不错的结果。当训练更困难的任务时，您需要在数据管道中包含数据增强。数据增强是一种使模型对输入数据（例如光照、裁剪和方向）变化具有鲁棒性的技术。`KerasCV`在`keras_cv.layers API`中包含一些有用的增强功能。创建最佳的增强管道是一门艺术，我们将提供一些有关分类最佳实践的提示。关于图像数据增强需要注意的是，不要将增强的数据分布偏离原始数据分布太远。目标是防止过度拟合并提高泛化能力，但完全不符合数据分布的样本只会给训练过程增加噪声。我们将使用的第一个增强是`RandomFlip`。这种增强的行为与您所期望的一样：它要么翻转图像，要么不翻转图像。虽然这种增强在`CalTech101`和`ImageNet`中很有用，但应该注意的是，它不应该用于数据分布不是垂直镜像不变的任务。发生这种情况的数据集的一个示例是`MNIST`手写数字。将6翻转到垂直轴将使数字看起来更像`9`而不是`6`，但标签仍会显示`6`。
```python
random_flip = keras_cv.layers.RandomFlip()
augmenters = [random_flip]

image_batch = random_flip(image_batch)
keras_cv.visualization.plot_image_gallery(
    image_batch.to_tensor(),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
一半的图已经翻转了，我们将使用的下一个增强是`RandomCropAndResize`。此操作选择图像的随机子集，然后将其大小调整为提供的目标大小。通过使用这种增强，我们迫使我们的分类器的空间不变。此外，该层接受`aspect_ratio_factor`，可用于扭曲图像的纵横比。虽然这可以提高模型性能，但应谨慎使用。纵横比失真很容易使样本偏离原始训练集的数据分布太远。请记住-数据增强的目标是生成更多与训练集的数据分布相符的训练样本！`RandomCropAndResize`还可以处理t`f.RaggedTensor`输入。在`CalTech101`图像数据集中，图像有多种尺寸。因此，它们不能轻易地组合在一起形成密集的训练批次。幸运的是，`RandomCropAndResize`可以为您处理`Ragged -> Dense`转换过程！将`RandomCropAndResize`添加到我们的增强组中：
```python
crop_and_resize = keras_cv.layers.RandomCropAndResize(
    target_size=IMAGE_SIZE,
    crop_area_factor=(0.8, 1.0),
    aspect_ratio_factor=(0.9, 1.1),
)
augmenters += [crop_and_resize]

image_batch = crop_and_resize(image_batch)
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
接下来，让我们在训练集中包含一些基于空间和颜色的抖动。这将使我们能够生成一个对照明闪烁、阴影等具有鲁棒性的分类器。通过改变颜色和空间特征来增强图像的方法有无数种，但也许最经受考验的技术是`RandAugment。 RandAugment`实际上是一组`10`种不同的增强：`AutoContrast、Equalize、Solarize、RandomColorJitter、RandomContrast、RandomBrightness、ShearX、ShearY、TranslateX 和 TranslateY`。在推理时，对每个图像采样`num_augmentations`增强器，并对每个图像采样随机幅度因子。然后依次应用这些增强`KerasCV`使用`augmentations_per_image和magnitude`参数可以随意调整！让我们来试一下：
```python
rand_augment = keras_cv.layers.RandAugment(
    augmentations_per_image=3,
    value_range=(0, 255),
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=1.0,
)
augmenters += [rand_augment]

image_batch = rand_augment(image_batch)
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
看起来很棒, 但我们还没有完成！如果图像缺少某个类别的一个关键特征怎么办？例如，如果一片叶子挡住了猫耳朵，但我们的分类器仅通过观察猫的耳朵就学会了对猫进行分类，该怎么办？解决这个问题的一种简单方法是使用`RandomCutout`，它会随机删除图像的一个子部分：
```python
random_cutout = keras_cv.layers.RandomCutout(width_factor=0.4, height_factor=0.4)
keras_cv.visualization.plot_image_gallery(
    random_cutout(image_batch),
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
虽然这很好地解决了这个问题，但它可能会导致分类器对特征之间的边界由剪切引起的黑色像素区域之间的边界做出响应。`CutMix`通过使用更复杂的技术解决了同样的问题。`CutMix`不是用黑色像素替换剪切区域，而是用从训练集中采样的其他图像区域替换这些区域！在此替换之后，图像的分类标签将更新为原始图像和混合图像的类标签的混合。让我们来看看：
```python
cut_mix = keras_cv.layers.CutMix()
# CutMix needs to modify both images and labels
inputs = {"images": image_batch, "labels": label_batch}

keras_cv.visualization.plot_image_gallery(
    cut_mix(inputs)["images"],
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
接下来，我们来看看`MixUp()`。不幸的是，虽然`MixUp()`已被证明可以显着提高训练模型的鲁棒性和泛化性，但人们并不清楚为什么会出现这种改进。`MixUp()`的工作原理是从一批图像中采样两个图像，然后将它们的像素强度及其分类标签混合在一起。让我们看看它的实际效果：
```python
mix_up = keras_cv.layers.MixUp()
# MixUp needs to modify both images and labels
inputs = {"images": image_batch, "labels": label_batch}

keras_cv.visualization.plot_image_gallery(
    mix_up(inputs)["images"],
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
如果仔细观察，您会发现图像已混合在了一起。我们不是将`CutMix()`和`MixUp()`应用于每个图像，而是随机选择一个应用于每个批次。这可以使用`keras_cv.layers.RandomChoice()`来获取。
```python
cut_mix_or_mix_up = keras_cv.layers.RandomChoice([cut_mix, mix_up], batchwise=True)
augmenters += [cut_mix_or_mix_up]

def create_augmenter_fn(augmenters):
    def augmenter_fn(inputs):
        for augmenter in augmenters:
            inputs = augmenter(inputs)
        return inputs

    return augmenter_fn

augmenter_fn = create_augmenter_fn(augmenters)
train_ds = train_ds.map(augmenter_fn, num_parallel_calls=tf.data.AUTOTUNE)

image_batch = next(iter(train_ds.take(1)))["images"]
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
我们还需要调整评估集的大小，以获得模型预期的密集图像大小。在这种情况下，我们使用`keras_cv.layers.Resizing`以避免给我们的评估指标添加噪音。
```python
inference_resizing = keras_cv.layers.Resizing(
    IMAGE_SIZE[0], IMAGE_SIZE[1], crop_to_aspect_ratio=True
)
eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

image_batch = next(iter(eval_ds.take(1)))["images"]
keras_cv.visualization.plot_image_gallery(
    image_batch,
    rows=3,
    cols=3,
    value_range=(0, 255),
    show=True,
)
```
最后，让我们解压我们的数据集并准备将它们传递给`model.fit()`，它接受（`images, labels`）的元组。
```python
def unpackage_dict(inputs):
    return inputs["images"], inputs["labels"]

train_ds = train_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(unpackage_dict, num_parallel_calls=tf.data.AUTOTUNE)
```
#### 优化器调优

为了实现最佳性能，我们需要使用学习率计划(`learning rate schedule`)，而不是单一学习率。
```python
def lr_warmup_cosine_decay(
    global_step,
    warmup_steps,
    hold=0,
    total_steps=0,
    start_lr=0.0,
    target_lr=1e-2,
):
    # Cosine decay
    learning_rate = (
        0.5
        * target_lr
        * (
            1
            + ops.cos(
                math.pi
                * ops.convert_to_tensor(
                    global_step - warmup_steps - hold, dtype="float32"
                )
                / ops.convert_to_tensor(
                    total_steps - warmup_steps - hold, dtype="float32"
                )
            )
        )
    )

    warmup_lr = target_lr * (global_step / warmup_steps)

    if hold > 0:
        learning_rate = ops.where(global_step > warmup_steps + hold, learning_rate, target_lr)

    learning_rate = ops.where(global_step < warmup_steps, warmup_lr, learning_rate)
    return learning_rate

class WarmUpCosineDecay(schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, hold, start_lr=0.0, target_lr=1e-2):
        super().__init__()
        self.start_lr = start_lr
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.hold = hold

    def __call__(self, step):
        lr = lr_warmup_cosine_decay(
            global_step=step,
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            start_lr=self.start_lr,
            target_lr=self.target_lr,
            hold=self.hold,
        )

        return ops.where(step > self.total_steps, 0.0, lr)
```
{% asset_img ic_1.png %}

接下来我们使用`WarmUpCosineDecay`定义优化器：
```python
total_images = 9000
total_steps = (total_images // BATCH_SIZE) * EPOCHS
warmup_steps = int(0.1 * total_steps)
hold_steps = int(0.45 * total_steps)
schedule = WarmUpCosineDecay(start_lr=0.05,target_lr=1e-2,warmup_steps=warmup_steps,total_steps=total_steps,hold=hold_steps,)
optimizer = optimizers.SGD(weight_decay=5e-4,learning_rate=schedule,momentum=0.9,)
```
最后，我们现在可以构建模型并调用`fit()`。`keras_cv.models.EfficientNetV2B0Backbone()`是`keras_cv.models.EfficientNetV2Backbone.from_preset('efficientnetv2_b0')`的别名。请注意，此预设不附带任何预训练权重。
```python
backbone = keras_cv.models.EfficientNetV2B0Backbone()
model = keras.Sequential(
    [
        backbone,
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(101, activation="softmax"),
    ]
)
```
由于`MixUp()`和`CutMix()`生成的标签有些人为操作，因此我们采用标签平滑来防止模型过度拟合此增强过程的伪影。
```python
loss = losses.CategoricalCrossentropy(label_smoothing=0.1)

# 编译模型
model.compile(loss=loss,optimizer=optimizer,
    metrics=[
        metrics.CategoricalAccuracy(),
        metrics.TopKCategoricalAccuracy(k=5),
    ],
)
# 训练模型
model.fit(train_ds,epochs=EPOCHS,validation_data=eval_ds,)

# 96/96 ━━━━65s 462ms/step - categorical_accuracy: 0.0068 - loss: 6.6096 - top_k_categorical_accuracy: 0.0497 - val_categorical_accuracy: 0.0122 - val_loss: 4.7151 - val_top_k_categorical_accuracy: 0.1596
```
您现在知道如何在`KerasCV`中从头开始训练强大的图像分类器。除了上面讨论的数据增强之外，从头开始的训练可能会或可能不会比使用迁移学习更强大。对于较小的数据集，预训练模型通常会产生高精度和更快的收敛速度。

#### 结论

虽然**图像分类**可能是计算机视觉中最简单的问题，但它具有许多复杂的组成部分。`KerasCV`提供了强大的生产级`API`，可以通过一行代码组装大部分组件。通过使用`KerasCV`的 `ImageClassifier API`、预训练权重和`KerasCV`数据增强，您可以在几百行代码中组装训练强大的分类器所需的一切！