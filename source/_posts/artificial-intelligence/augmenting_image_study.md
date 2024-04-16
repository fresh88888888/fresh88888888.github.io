---
title: 图像数据增强（KerasCV）
date: 2024-04-16 10:00:11
tags:
  - AI
categories:
  - 人工智能
---

`KerasCV`可以轻松组装最先进的工业级数据增强管道，用于图像分类和对象检测任务。`KerasCV`提供了广泛的预处理层，可实现常见的数据增强技术。也许最有用的三个层是`keras_cv.layers.CutMix、keras_cv.layers.MixUp`和`keras_cv.layers.RandAugment`。这些层几乎用于所有最先进的图像分类流程。
<!-- more -->

#### 包导入 && 数据加载

```python
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
import keras_cv

os.environ["KERAS_BACKEND"] = "jax"  # @param ["tensorflow", "jax", "torch"]

# 为了演示如何执行数据增强，我们首先加载`Oxford Flowers102`数据集，其中包含所有种类的花卉我们通过清洗和批处理进一步预处理数据集。
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
tfds.disable_progress_bar()
data, dataset_info = tfds.load("oxford_flowers102", with_info=True, as_supervised=True)
train_steps_per_epoch = dataset_info.splits["train"].num_examples // BATCH_SIZE
val_steps_per_epoch = dataset_info.splits["test"].num_examples // BATCH_SIZE

# 接下来，我们将图像大小调整为恒定大小 (224, 224)，并对标签进行one-hot编码。 
# 请注意，keras_cv.layers.CutMix 和 keras_cv.layers.MixUp 期望目标是one-hot编码的。 
# 这是因为它们以稀疏标签表示无法实现的方式修改目标值。
IMAGE_SIZE = (224, 224)
num_classes = dataset_info.features["label"].num_classes

def to_dict(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    label = tf.one_hot(label, num_classes)
    return {"images": image, "labels": label}

def prepare_dataset(dataset, split):
    if split == "train":
        return (
            dataset.shuffle(10 * BATCH_SIZE)
            .map(to_dict, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
        )
    if split == "test":
        return dataset.map(to_dict, num_parallel_calls=AUTOTUNE).batch(BATCH_SIZE)

def load_dataset(split="train"):
    dataset = data[split]
    return prepare_dataset(dataset, split)

train_dataset = load_dataset()

def visualize_dataset(dataset, title):
    plt.figure(figsize=(6, 6)).suptitle(title, fontsize=18)
    for i, samples in enumerate(iter(dataset.take(9))):
        images = samples["images"]
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
    plt.show()

# 让我们检查一下数据集中的一些样本：
visualize_dataset(train_dataset, title="Before Augmentation")
```
{% asset_img ai_1.png %}

#### RandAugment

`RandAugment`已被证明可以在众多数据集中提供改进的图像分类结果。它对图像执行一组标准的增强。`KerasCV`提供了大量的数据增强层，其中最有用的三个层可能是`RandAugment,CutMix,MixUp`。RandAugment选择一个随机操作。然后它对随机数进行采样，如果随机数小于速率参数，它将对给定图像应用随机操作。除了塑料参数值之外，值范围参数指定图像的值范围，每个图像的增强指定幅度和幅度stddev基本上决定了用于对每个数据增强进行采样的正态分布。应用`RandAugment`之后你可以看到一些示例。
```python
rand_augment = keras_cv.layers.RandAugment(
    value_range=(0, 255),
    augmentations_per_image=3,
    magnitude=0.3,
    magnitude_stddev=0.2,
    rate=1.0,
)

def apply_rand_augment(inputs):
    inputs["images"] = rand_augment(inputs["images"])
    return inputs

train_dataset = load_dataset().map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
# 让我们检查一下结果
visualize_dataset(train_dataset, title="After RandAugment")
```
{% asset_img ai_2.png %}

#### CutMix 和 MixUp

`CutMix`和`MixUp`是另外两个重要的增强操作。`CutMix`随机剪切一张图像的一部分并将其放置在另一张图像上，`MixUp`则在两幅图像之间插入像素值。这两者都可以防止模型过度拟合训练分布，并提高模型泛化的可能性。此外，`CutMix`还可以防止您的模型过度依赖任何特定特征来进行分类。在下边的示例中，我们将在手动创建的预处理管道中独立使用`CutMix`和`MixUp`。在大多数最先进的管道中，图像是通过`CutMix、MixUp`或两者都不随机增强。
```python
cut_mix = keras_cv.layers.CutMix()
mix_up = keras_cv.layers.MixUp()

def cut_mix_and_mix_up(samples):
    samples = cut_mix(samples, training=True)
    samples = mix_up(samples, training=True)
    return samples

train_dataset = load_dataset().map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)
visualize_dataset(train_dataset, title="After CutMix and MixUp")
```
{% asset_img ai_3.png %}

#### 定制增强管道

默写情况下你可能需要自定义它。例如，你想要排除某个增强或添加另一个增强，在这种情况下，你可以随机增强管道。随机增强管道是一个与`RandAugment`类似的层但它可以灵活地自定义增强管道，例如我们在这里删除随机旋转层并在管道中添加网格遮罩层，现在我们可以应用定制的管道。
```python
layers = keras_cv.layers.RandAugment.get_standard_policy(value_range=(0, 255), magnitude=0.75, magnitude_stddev=0.3)

# 删除随机旋转层
layers = [
    layer for layer in layers if not isinstance(layer, keras_cv.layers.RandomRotation)
]

# 添加网格遮罩层
layers = layers + [keras_cv.layers.GridMask()]

pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=layers, augmentations_per_image=3
)

def apply_pipeline(inputs):
    inputs["images"] = pipeline(inputs["images"])
    return inputs

# 让我们看看结果吧
train_dataset = load_dataset().map(apply_pipeline, num_parallel_calls=AUTOTUNE)
visualize_dataset(train_dataset, title="After custom pipeline")
```
{% asset_img ai_4.png %}

正如您所看到的，没有图像被随机旋转。您可以根据需要自定义管道：
```python
# 该管道将​​应用 GrayScale(蒙版和灰度层) 或 GridMask(网格层)：
pipeline = keras_cv.layers.RandomAugmentationPipeline(
    layers=[keras_cv.layers.GridMask(), keras_cv.layers.Grayscale(output_channels=3)],
    augmentations_per_image=1,
)

# 让我们看看结果吧
train_dataset = load_dataset().map(apply_pipeline, num_parallel_calls=AUTOTUNE)
visualize_dataset(train_dataset, title="After custom pipeline")
```
{% asset_img ai_5.png %}

#### 训练一个带有增强功能卷积神经网络（CNN）

我们将使用`CutMix、MixUp`和`RandAugment`在`Oxford`花卉数据集上训练`ResNet50`图像分类器。
```python
def preprocess_for_model(inputs):
    images, labels = inputs["images"], inputs["labels"]
    images = tf.cast(images, tf.float32)
    return images, labels

train_dataset = (
    load_dataset()
    .map(apply_rand_augment, num_parallel_calls=AUTOTUNE)
    .map(cut_mix_and_mix_up, num_parallel_calls=AUTOTUNE)
)

visualize_dataset(train_dataset, "CutMix, MixUp and RandAugment")

train_dataset = train_dataset.map(preprocess_for_model, num_parallel_calls=AUTOTUNE)
test_dataset = load_dataset(split="test").map(preprocess_for_model, num_parallel_calls=AUTOTUNE)

train_dataset = train_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# 我们使用Efficientnetv2作为骨干网络模型&编译。
# 请注意，我们在损失函数中使用label_smoothing=0.1。使用MixUp时，强烈建议进行标签平滑。
def get_model():
    model = keras_cv.models.ImageClassifier.from_preset("efficientnetv2_s", num_classes=num_classes)
    model.compile(
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        optimizer=keras.optimizers.SGD(momentum=0.9),
        metrics=["accuracy"],
    )
    return model

# 定义和训练模型
model = get_model()
model.fit(train_dataset,epochs=1,validation_data=test_dataset,)

# 32/32 ━━━━━━━━━━━━━━━━━━━━ 103s 2s/step - accuracy: 0.0059 - loss: 4.6941 - val_accuracy: 0.0114 - val_loss: 10.4028
```
{% asset_img ai_6.png %}

#### 结论

这就是使用`KerasCV`组装图像增强管道所需的全部！