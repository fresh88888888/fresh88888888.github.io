---
title: 计算机视觉（TensorFlow & Keras）
date: 2024-03-21 16:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 卷积分类器（The Convolutional Classifier）

- 使用现代深度学习网络通过`Keras`构建图像分类器。
- 使用可重复使用的块设计您自己的自定义卷积网络。
- 了解视觉特征提取背后的基本思想。
- 掌握迁移学习的艺术来提升您的模型。
- 利用数据增强来扩展数据集。
<!-- more -->
##### 介绍

将向您介绍计算机视觉的基本思想。我们的目标是了解神经网络如何充分“理解”自然图像，以解决人类视觉系统可以解决的同类问题。最擅长此任务的神经网络称为卷积神经网络（有时我们称为`CNN`）。卷积是一种数学运算，它赋予了`CNN`各层独特的结构。我们将把这些想法应用到图像分类问题上：给定一张图片，我们能否训练计算机告诉我们它是什么？您可能见过可以从照片中识别植物种类的应用程序。这就是图像分类器！最后，您将准备好继续学习更高级的应用程序，例如生成对抗网络和图像分割。

##### 卷积分类器

用于图像分类的卷积网络由两部分组成：**卷积基**和**密集头**。
{% asset_img cv_1.png %}

基础用于从图像中**提取特征**。它主要由执行卷积运算的层组成，但通常也包括其他类型的层。头部用于确定图像的类别。它主要由致密层组成，但可能包括其他层，例如`dropout`。视觉特征是什么意思？特征可以是线条、颜色、纹理、形状、图案——或者一些复杂的组合。整个过程是这样的：
{% asset_img cv_2.png %}

##### 训练分类器

训练期间网络的目标是学习两件事：
- 从图像（基础）中提取哪些特征。
- 哪个类具有哪些特征（头）。

如今，卷积网络很少从头开始训练。更常见的是，我们重用预训练模型的基础。然后我们在预训练的基础上附加一个未经训练的头部。换句话说，我们重用网络中已经学会执行。`1.`提取特征，并附加一些新层来学习；`2.`分类的部分。
{% asset_img cv_3.png %}

由于头部通常仅由几个致密层组成，因此可以从相对较少的数据创建非常准确的分类器。重用预训练模型是一种称为迁移学习的技术。它非常有效，以至于现在几乎每个图像分类器都会使用它。

##### 举例 - 训练Convnet分类器

我们将创建分类器来尝试解决以下问题：这是汽车还是卡车的图片？我们的数据集大约有`10,000`张各种汽车的图片，大约一半是汽车，一半是卡车。

###### 第1步 - 加载数据

下一个隐藏单元将导入一些库并设置我们的数据管道。我们有一个名为`ds_train`的训练分割和一个名为`ds_vali`d的验证分割。
```python
import os, warnings
import matplotlib.pyplot as plt
from matplotlib import gridspec

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
set_seed(31415)

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore") # to clean up output cells


# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    '../input/car-or-truck/train',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=True,
)
ds_valid_ = image_dataset_from_directory(
    '../input/car-or-truck/valid',
    labels='inferred',
    label_mode='binary',
    image_size=[128, 128],
    interpolation='nearest',
    batch_size=64,
    shuffle=False,
)

# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label

AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = (
    ds_train_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
ds_valid = (
    ds_valid_
    .map(convert_to_float)
    .cache()
    .prefetch(buffer_size=AUTOTUNE)
)
```
让我们看一下训练集中的一些示例。
```python
import matplotlib.pyplot as plt
```
###### 第2步 - 定义预训练库

最常用的预训练数据集是`ImageNet`，这是一个包含多种自然图像的大型数据集。`Keras`在其应用程序模块中包含在`ImageNet`上预训练的各种模型。我们将使用的预训练模型称为`VGG16`。
```python
pretrained_base = tf.keras.models.load_model(
    '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
)
pretrained_base.trainable = False
```
###### 第3步 - 连接头部

接下来，我们连接分类器头。对于此示例，我们将使用隐藏单元层（第一个`Dense`层），然后使用一个层将输出转换为`1`类卡车的概率分数。`Flatten`层将基础的二维输出转换为头部所需的一维输入。
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
```
###### 第4步 - 训练

最后，让我们训练模型。由于这是一个二分类问题，我们将使用交叉熵和准确性的二进制版本。`adam`优化器通常表现良好，因此我们也选择它。
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=30,
    verbose=0,
)
```
训练神经网络时，检查损失图和度量图始终是个好主意。历史对象在字典`history.history`中包含此信息。我们可以使用`Pandas`将此字典转换为数据框，并使用内置方法将其绘制出来。
```python
import pandas as pd

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ['loss', 'val_loss']].plot()
history_frame.loc[:, ['binary_accuracy', 'val_binary_accuracy']].plot()
```
{% asset_img cv_4.png %}
{% asset_img cv_5.png %}

##### 结论

我们了解了卷积网络分类器的结构：**在执行特征提取的基础之上充当分类器的头部**。本质上，头部是一个普通的分类器。对于特征，它使用基础提取的那些特征。这是卷积分类器背后的基本思想：我们可以将执行特征工程的单元附加到分类器本身。这是深度神经网络相对于传统机器学习模型的一大优势：给定正确的网络结构，深度神经网络可以学习如何设计解决问题所需的特征。
