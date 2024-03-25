---
title: 基于机器学习 — 花瓣图像分类（TensorFlow & Keras）
date: 2024-03-25 15:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 介绍

构建一个机器学习模型，根据图像对`104`种花朵进行分类。您将学习如何在`Keras`中构建图像分类器并在张量处理单元(`TPU`)上对其进行训练。

#### 第1步：导入包

我们首先导入几个`Python`包。
```python
import math, re, os
import numpy as np
import tensorflow as tf

print("Tensorflow version " + tf.__version__)
```
#### 第2步：分布式策略

`TPU`有八个不同的核心，每个核心都充当自己的加速器。（`TPU`有点像一台机器上有八个`GPU`）我们告诉`TensorFlow`如何通过分布式策略同时使用所有这些核心。
```python
# Detect TPU, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() 
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() 

print("REPLICAS: ", strategy.num_replicas_in_sync)
```
创建神经网络模型时，我们将使用分布式策略。然后，`TensorFlow`将通过创建八个不同的模型副本（每个核心一个）在八个`TPU`核心之间分配训练。

#### 第3步：加载数据

##### 获取GCS路径

与`TPU`一起使用时，数据集需要存储在`Google Cloud Storage`存储桶中。您可以通过提供路径来使用任何公共`GCS`存储桶中的数据，就像使用`“/kaggle/input”`中的数据一样。下面将检索数据集的`GCS`路径。
```python
from kaggle_datasets import KaggleDatasets

GCS_DS_PATH = KaggleDatasets().get_gcs_path('tpu-getting-started')
print(GCS_DS_PATH) # what do gcs paths look like?
```
##### 加载数据

与`TPU`一起使用时，数据集通常会序列化为`TFRecord`。这是一种方便将数据分发到每个`TPU`核心的格式。我们隐藏了读取数据集`TFRecords`的单元格，因为该过程有点长。您可以稍后再回来查看有关自己的数据集与`TPU`结合使用的一些指导。
```python
IMAGE_SIZE = [512, 512]
GCS_PATH = GCS_DS_PATH + '/tfrecords-jpeg-512x512'
AUTO = tf.data.experimental.AUTOTUNE

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test/*.tfrec') 

CLASSES = ['pink primrose',    'hard-leaved pocket orchid', 'canterbury bells', 'sweet pea',     'wild geranium',     'tiger lily',           'moon orchid',              'bird of paradise', 'monkshood',        'globe thistle',         # 00 - 09
           'snapdragon',       "colt's foot",               'king protea',      'spear thistle', 'yellow iris',       'globe-flower',         'purple coneflower',        'peruvian lily',    'balloon flower',   'giant white arum lily', # 10 - 19
           'fire lily',        'pincushion flower',         'fritillary',       'red ginger',    'grape hyacinth',    'corn poppy',           'prince of wales feathers', 'stemless gentian', 'artichoke',        'sweet william',         # 20 - 29
           'carnation',        'garden phlox',              'love in the mist', 'cosmos',        'alpine sea holly',  'ruby-lipped cattleya', 'cape flower',              'great masterwort', 'siam tulip',       'lenten rose',           # 30 - 39
           'barberton daisy',  'daffodil',                  'sword lily',       'poinsettia',    'bolero deep blue',  'wallflower',           'marigold',                 'buttercup',        'daisy',            'common dandelion',      # 40 - 49
           'petunia',          'wild pansy',                'primula',          'sunflower',     'lilac hibiscus',    'bishop of llandaff',   'gaura',                    'geranium',         'orange dahlia',    'pink-yellow dahlia',    # 50 - 59
           'cautleya spicata', 'japanese anemone',          'black-eyed susan', 'silverbush',    'californian poppy', 'osteospermum',         'spring crocus',            'iris',             'windflower',       'tree poppy',            # 60 - 69
           'gazania',          'azalea',                    'water lily',       'rose',          'thorn apple',       'morning glory',        'passion flower',           'lotus',            'toad lily',        'anthurium',             # 70 - 79
           'frangipani',       'clematis',                  'hibiscus',         'columbine',     'desert-rose',       'tree mallow',          'magnolia',                 'cyclamen ',        'watercress',       'canna lily',            # 80 - 89
           'hippeastrum ',     'bee balm',                  'pink quill',       'foxglove',      'bougainvillea',     'camellia',             'mallow',                   'mexican petunia',  'bromelia',         'blanket flower',        # 90 - 99
           'trumpet creeper',  'blackberry lily',           'common tulip',     'wild rose']                                                                                                                                               # 100 - 102


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['class'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['id']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset
```
##### 创建数据管道

在最后一步中，我们将使用`tf.data API`为每个训练、验证和测试拆分定义高效的数据管道。
```python
def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO)
    # statement in the next function (below), this happens essentially
    # for free on TPU. Data pipeline code is executed on the "CPU"
    # part of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
print('Dataset: {} training images, {} validation images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))
```
下一个单元将创建我们将在训练和推理期间与`Keras`一起使用的数据集。请注意我们如何将批次的大小调整为`TPU`核心的数量。
```python
# Define the batch size. This will be 16 with TPU off and 128 (=16*8) with TPU on
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

ds_train = get_training_dataset()
ds_valid = get_validation_dataset()
ds_test = get_test_dataset()

print("Training:", ds_train)
print ("Validation:", ds_valid)
print("Test:", ds_test)
```
结果输出为：
```bash
Training: <PrefetchDataset shapes: ((None, 512, 512, 3), (None,)), types: (tf.float32, tf.int32)>
Validation: <PrefetchDataset shapes: ((None, 512, 512, 3), (None,)), types: (tf.float32, tf.int32)>
Test: <PrefetchDataset shapes: ((None, 512, 512, 3), (None,)), types: (tf.float32, tf.string)>
```
这些数据集是`tf.data.Dataset`对象。您可以将`TensorFlow`中的数据集视为数据记录流。训练集和验证集是（`image, label`）对的流。
```python
np.set_printoptions(threshold=15, linewidth=80)

print("Training data shapes:")
for image, label in ds_train.take(3):
    print(image.numpy().shape, label.numpy().shape)
print("Training data label examples:", label.numpy())
```
结果输出为：
```bash
Training data shapes:
(128, 512, 512, 3) (128,)
(128, 512, 512, 3) (128,)
(128, 512, 512, 3) (128,)
Training data label examples: [ 88  51 102 ...  10  24  14]
```
测试集是（`image，idnum`）对的流；这里的`idnum`是为图像提供的唯一标识符，稍后我们以`csv`文件形式提交时将使用该标识符。
```python
print("Test data shapes:")
for image, idnum in ds_test.take(3):
    print(image.numpy().shape, idnum.numpy().shape)
print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
```
结果输出为：
```python
Test data shapes:
(128, 512, 512, 3) (128,)
(128, 512, 512, 3) (128,)
(128, 512, 512, 3) (128,)
Test data IDs: ['b87e16bc0' 'd8437a7f7' '981396649' ... '15cb0c24a' '1c3a7bc99' 'b20b97998']
```
#### 第 4 步：探索数据

让我们花点时间看一下数据集中的一些图像。
```python
from matplotlib import pyplot as plt

def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case,
                                     # these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is
    # the case for test data)
    return numpy_images, numpy_labels

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)
    
def display_batch_of_images(databatch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]
        
    # auto-squaring: this will drop data that does not fit into square
    # or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images)//rows
        
    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot=(rows,cols,1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))
    else:
        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))
    
    # display
    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3 # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)
    
    #layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.show()


def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])

```
您可以使用我们的另一个辅助函数显示数据集中的一批图像。下一个单元格会将数据集转换为`20`个图像批次的迭代器。
```python
ds_iter = iter(ds_train.unbatch().batch(20))
```
使用`Python next`函数输出流中的下一批，并使用辅助函数显示它。
```python
one_batch = next(ds_iter)
display_batch_of_images(one_batch)
```
{% asset_img cf_1.png %}

通过在单独的单元格中定义`ds_iter`和`one_batch`，您只需重新运行上面的单元格即可看到一批新图像。

#### 第5步：定义模型

现在我们准备创建一个用于图像分类的神经网络！我们将使用所谓的迁移学习。通过迁移学习，您可以重用预训练模型的一部分，以便在新数据集上取得领先。我们将使用在`ImageNet`上预训练的名为`VGG16`的模型。稍后，您可能想尝试`Keras`中包含的其他模型。（`Xception`不会是一个糟糕的选择。）我们之前创建的分部式策略包含一个上下文管理器，`strategy.scope`。该上下文管理器告诉`TensorFlow`如何在八个`TPU`核心之间分配训练工作。将`TensorFlow`与`TPU`结合使用时，在`Strategy.scope()`上下文中定义模型非常重要。
```python
EPOCHS = 12

with strategy.scope():
    pretrained_model = tf.keras.applications.VGG16(
        weights='imagenet',
        include_top=False ,
        input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False
    
    model = tf.keras.Sequential([
        # To a base pretrained on ImageNet to extract features from images...
        pretrained_model,
        # ... attach a new head to act as a classifier.
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])
```
损失和指标的“`sparse_categorical`”版本适用于具有两个以上标签的分类任务，例如这个。
```python
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

model.summary()
```
结果输出为：
```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Model)                (None, 16, 16, 512)       14714688  
_________________________________________________________________
global_average_pooling2d (Gl (None, 512)               0         
_________________________________________________________________
dense (Dense)                (None, 104)               53352     
=================================================================
Total params: 14,768,040
Trainable params: 53,352
Non-trainable params: 14,714,688
_________________________________________________________________
```
#### 第6步：训练

##### 学习率计划

我们将使用特殊的学习率计划来训练该网络。
```python
# Learning Rate Schedule for Fine Tuning #
def exponential_lr(epoch,
                   start_lr = 0.00001, min_lr = 0.00001, max_lr = 0.00005,
                   rampup_epochs = 5, sustain_epochs = 0,
                   exp_decay = 0.8):

    def lr(epoch, start_lr, min_lr, max_lr, rampup_epochs, sustain_epochs, exp_decay):
        # linear increase from start to rampup_epochs
        if epoch < rampup_epochs:
            lr = ((max_lr - start_lr) /
                  rampup_epochs * epoch + start_lr)
        # constant max_lr during sustain_epochs
        elif epoch < rampup_epochs + sustain_epochs:
            lr = max_lr
        # exponential decay towards min_lr
        else:
            lr = ((max_lr - min_lr) *
                  exp_decay**(epoch - rampup_epochs - sustain_epochs) +
                  min_lr)
        return lr
    return lr(epoch,
              start_lr,
              min_lr,
              max_lr,
              rampup_epochs,
              sustain_epochs,
              exp_decay)

lr_callback = tf.keras.callbacks.LearningRateScheduler(exponential_lr, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [exponential_lr(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
```
结果输出为：
```bash
Learning rate schedule: 1e-05 to 5e-05 to 2.05e-05
```
{% asset_img cf_2.png %}

##### 拟合模型

现在我们准备好训练模型了。定义了一些参数后，我们就可以开始了！
```python
# Define training epochs
EPOCHS = 12
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=[lr_callback],
)
```
结果输出为：
```bash
Epoch 00001: LearningRateScheduler reducing learning rate to 0.0010000000474974513.
Epoch 1/12
99/99 [==============================] - 29s 291ms/step - sparse_categorical_accuracy: 0.0879 - loss: 4.1231 - val_sparse_categorical_accuracy: 0.1414 - val_loss: 3.8921 - lr: 0.0010

Epoch 00002: LearningRateScheduler reducing learning rate to 0.0008100000379979611.
Epoch 2/12
99/99 [==============================] - 18s 181ms/step - sparse_categorical_accuracy: 0.1698 - loss: 3.7868 - val_sparse_categorical_accuracy: 0.1975 - val_loss: 3.6811 - lr: 8.1000e-04

Epoch 00003: LearningRateScheduler reducing learning rate to 0.0005060000335611404.
Epoch 3/12
99/99 [==============================] - 17s 175ms/step - sparse_categorical_accuracy: 0.2014 - loss: 3.6228 - val_sparse_categorical_accuracy: 0.2123 - val_loss: 3.5711 - lr: 5.0600e-04

Epoch 00004: LearningRateScheduler reducing learning rate to 0.00023240000449121004.
Epoch 4/12
99/99 [==============================] - 17s 172ms/step - sparse_categorical_accuracy: 0.2196 - loss: 3.5349 - val_sparse_categorical_accuracy: 0.2212 - val_loss: 3.5229 - lr: 2.3240e-04

Epoch 00005: LearningRateScheduler reducing learning rate to 8.648000176530332e-05.
Epoch 5/12
99/99 [==============================] - 17s 170ms/step - sparse_categorical_accuracy: 0.2227 - loss: 3.5044 - val_sparse_categorical_accuracy: 0.2258 - val_loss: 3.5059 - lr: 8.6480e-05

Epoch 00006: LearningRateScheduler reducing learning rate to 5e-05.
Epoch 6/12
99/99 [==============================] - 17s 171ms/step - sparse_categorical_accuracy: 0.2236 - loss: 3.4902 - val_sparse_categorical_accuracy: 0.2249 - val_loss: 3.4963 - lr: 5.0000e-05

Epoch 00007: LearningRateScheduler reducing learning rate to 4.2000000000000004e-05.
Epoch 7/12
99/99 [==============================] - 17s 169ms/step - sparse_categorical_accuracy: 0.2280 - loss: 3.4818 - val_sparse_categorical_accuracy: 0.2290 - val_loss: 3.4885 - lr: 4.2000e-05

Epoch 00008: LearningRateScheduler reducing learning rate to 3.5600000000000005e-05.
Epoch 8/12
99/99 [==============================] - 19s 189ms/step - sparse_categorical_accuracy: 0.2263 - loss: 3.4728 - val_sparse_categorical_accuracy: 0.2290 - val_loss: 3.4818 - lr: 3.5600e-05

Epoch 00009: LearningRateScheduler reducing learning rate to 3.0480000000000006e-05.
Epoch 9/12
99/99 [==============================] - 16s 166ms/step - sparse_categorical_accuracy: 0.2297 - loss: 3.4618 - val_sparse_categorical_accuracy: 0.2309 - val_loss: 3.4760 - lr: 3.0480e-05

Epoch 00010: LearningRateScheduler reducing learning rate to 2.6384000000000004e-05.
Epoch 10/12
99/99 [==============================] - 17s 172ms/step - sparse_categorical_accuracy: 0.2287 - loss: 3.4643 - val_sparse_categorical_accuracy: 0.2309 - val_loss: 3.4711 - lr: 2.6384e-05

Epoch 00011: LearningRateScheduler reducing learning rate to 2.3107200000000005e-05.
Epoch 11/12
99/99 [==============================] - 17s 174ms/step - sparse_categorical_accuracy: 0.2296 - loss: 3.4510 - val_sparse_categorical_accuracy: 0.2301 - val_loss: 3.4668 - lr: 2.3107e-05

Epoch 00012: LearningRateScheduler reducing learning rate to 2.0485760000000004e-05.
Epoch 12/12
99/99 [==============================] - 17s 171ms/step - sparse_categorical_accuracy: 0.2328 - loss: 3.4466 - val_sparse_categorical_accuracy: 0.2309 - val_loss: 3.4630 - lr: 2.0486e-05
```
下一个单元格显示了训练期间损失和指标的进展情况。值得庆幸的是，它收敛了！
```python
display_training_curves(
    history.history['loss'],
    history.history['val_loss'],
    'loss',
    211,
)
display_training_curves(
    history.history['sparse_categorical_accuracy'],
    history.history['val_sparse_categorical_accuracy'],
    'accuracy',
    212,
)
```
{% asset_img cf_3.png %}

#### 第7步：评估预测

在对测试集进行最终预测之前，最好在验证集上评估模型的预测。这可以帮助您诊断训练中的问题或建议改进模型的方法。我们将研究两种常见的验证方法：**绘制混淆矩阵和视觉验证**。
```python
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15,15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring, fontdict={'fontsize': 18, 'horizontalalignment':'right', 'verticalalignment':'top', 'color':'#804040'})
    plt.show()
    
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
```
##### 混淆矩阵

混淆矩阵显示了图像的实际类别与其预测类别的对比。它是评估分类器性能的最佳工具之一。以下单元格对验证数据进行一些处理，然后使用`scikit-learn`中包含的`fusion_matrix`函数创建矩阵。
```python
cmdataset = get_validation_dataset(ordered=True)
images_ds = cmdataset.map(lambda image, label: image)
labels_ds = cmdataset.map(lambda image, label: label).unbatch()

cm_correct_labels = next(iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()
cm_probabilities = model.predict(images_ds)
cm_predictions = np.argmax(cm_probabilities, axis=-1)

labels = range(len(CLASSES))
cmat = confusion_matrix(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
)
cmat = (cmat.T / cmat.sum(axis=1)).T # normalize
```
您可能熟悉`F1`分数或精确率和召回率等指标。该单元格将计算这些指标并用**混淆矩阵图**显示它们。（这些指标在`Scikit-learn`模块`sklearn.metrics`中定义；我们已将它们导入到帮助程序脚本中。）
```python
score = f1_score(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
    average='macro',
)
precision = precision_score(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
    average='macro',
)
recall = recall_score(
    cm_correct_labels,
    cm_predictions,
    labels=labels,
    average='macro',
)
display_confusion_matrix(cmat, score, precision, recall)
```
{% asset_img cf_4.png %}

##### 视觉验证

查看验证集中的一些示例并了解模型预测的类别也很有帮助。这可以帮助揭示模型遇到问题的图像类型的模式。此单元格会将验证集设置为一次显示`20`个图像 - 如果您愿意，您可以更改此设置以显示更多或更少图像。
```python
dataset = get_validation_dataset()
dataset = dataset.unbatch().batch(20)
batch = iter(dataset)
```
这是一组花及其预测的种类。再次运行单元格以查看另一组。
```python
images, labels = next(batch)
probabilities = model.predict(images)
predictions = np.argmax(probabilities, axis=-1)
display_batch_of_images((images, labels), predictions)
```
{% asset_img cf_5.png %}

#### 第8步：做出测试预测

一旦您对一切感到满意，您就可以对测试集进行预测了。
```python
test_ds = get_test_dataset(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)
predictions = np.argmax(probabilities, axis=-1)
print(predictions)
```
结果输出为：
```python
Computing predictions...
[ 67 103 103 ...  49  53  53]
```
我们将生成文件`submission.csv`。
```python
print('Generating submission.csv file...')

# Get image ids from test set and convert to unicode
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')

# Write the submission file
np.savetxt(
    'submission.csv',
    np.rec.fromarrays([test_ids, predictions]),
    fmt=['%s', '%d'],
    delimiter=',',
    header='id,label',
    comments='',
)

# Look at the first few predictions
!head submission.csv
```
