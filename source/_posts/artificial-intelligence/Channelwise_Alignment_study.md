---
title: SETI-BL:通道方向对齐（TensorFlow & TPU）
date: 2024-04-11 15:24:11
tags:
  - AI
categories:
  - 人工智能
---

在本例中，利用数据科学技能帮助识别突`Breakthrough Listen`目标扫描中的异常信号。由于没有确认的外星信号可用于训练机器学习算法，因此在如大海捞针般的望远镜数据中加入了一些模拟信号（称之为“针”）。目前已经识别出一些隐藏的针，以便您可以训练模型以找到更多的”针“。数据由二维数组组成，因此计算机视觉、数字信号处理、异常检测等方法可能很有优势。
<!-- more -->

#### 方法论

- 本实例演示了如何使用`CNN`对信号进行分类。
- 尽管这个问题看起来像是信号检测问题，但我们可以将其转换为图像分类问题。我们可以使用不同的方法将一维信号转换为二维信号（图像），例如傅里叶变换，它基本上将数据从时间域转换为频域。 因此，从[`time, intensity`]我们得到[`time, frequency, intensity`]。
- 该实例使用预先计算的`tfrecord`进行训练。`TFRecord`可以将训练规模扩大至`2`倍。使用`TPU`作为我们的设备，这可以节省大量的时间。
- 对于增强，使用旋转、翻转、剪切、缩放、移位、丢弃、随机抖动。
- 该实例使用`EfficientNet`作为核心模型（分类器）。
- 然后，我们计算该实例的`Out of Fold` (`OOF`)分数。
- 最后，我们提交模型预测值。

#### 包和配置

```python
import os
import shutil
from glob import glob
from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
import tensorflow_addons as tfa
import pandas as pd
import numpy as np
import random
import sklearn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # to avoid too many logging messages
# USE VERBOSE=0 for silent, VERBOSE=1 for interactive, VERBOSE=2 for commit
VERBOSE      = 0
DISPLAY_PLOT = True

DEVICE = "TPU" #or "GPU"

# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
SEED = 42

# NUMBER OF FOLDS. USE 2, 5, 10
FOLDS = 5

# WHICH IMAGE SIZES TO LOAD EACH FOLD
IMG_SIZES = [[273, 256]]*FOLDS

# BATCH SIZE AND EPOCHS
BATCH_SIZES = [32]*FOLDS
EPOCHS      = [12]*FOLDS

# WHICH EFFICIENTNET B? TO USE
EFF_NETS = [6]*FOLDS

# Augmentations
AUGMENT   = True
TRANSFORM = True

# AFFINE TRANSFORMATION
ROT_    = 0.0 # ROTATION
SHR_    = 2.0 # SHEAR
HZOOM_  = 8.0 # HORIZONTAL ZOOM
WZOOM_  = 8.0 # VERTICAL ZOOM
HSHIFT_ = 8.0 # HORIZONTAL SHIFT
WSHIFT_ = 8.0 # VERTICAL SHIFT

# COARSE DROPOUT - MORE ABOUT THIS IS LATER
PROBABILITY = 0.50 # PROBABILITY OF DROPOUT
CT          = 8 # NUMBER OF DROPOUTS
SZ          = 0.05 # SIZE OF THE DROPOUT - CALCULATED AS PERCENT OF IMAGE-DIMENSION

#bri, contrast
sat  = (0.7, 1.3) # SATURATION
cont = (0.8, 1.2) # CONTRAST
bri  =  0.1 # BRIGHTNESS

# WEIGHTS FOR FOLD MODELS WHEN PREDICTING TEST
WGTS = [1/FOLDS]*FOLDS

# TEST TIME AUGMENTATION STEPS
TTA = 11

# 再现性
def seeding(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(SEED)
    tf.random.set_seed(SEED)
    print('seeding done!!!')

seeding(SEED)
```
以下代码自动检测硬件（`tpu`或`gpu`或`cpu`）:
- 什么是`TPU`？：它们是专门用于深度学习任务的硬件加速器。它们配备`128GB`高速内存，允许更大的批次、更大的模型以及更大的训练输入。

{% asset_img ca_1.png %}

```python
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        # detect and init the TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            # instantiate a distribution strategy
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    # instantiate a distribution strategy
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

# connecting to TPU...
# Running on TPU  grpc://10.0.0.2:8470
# initializing  TPU ...
# TPU initialized
# REPLICAS: 8
```
#### 预处理

- 原始信号有`6`个通道。正如这里提到的，我只拿了`3`个。
- 到目前为止还没有进行任何信号处理。把它留给模型。 
- 使用简单的增强，其中一些可能会损害模型。

```python
GCS_PATH = [None]*FOLDS
for i,k in enumerate(IMG_SIZES):
    GCS_PATH[i] = KaggleDatasets().get_gcs_path('setibl-%ix%i-tfrec-dataset'%(k[0],k[1]))
files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/train*.tfrec')))
files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))
num_train_files = len(files_train)
num_test_files  = len(files_test)
print('train_files:',num_train_files)
print('test_files:',num_test_files)

# train_files: 20
# test_files: 10
```
##### 元数据

- `train/` - 以`numpy float16`格式(`v1.20.1`)存储的一组`cadence`片段文件训练集，每个`cadence`片段`id`一个文件，并在`train_labels.csv`文件中找到相应的标签。每个文件都有维度(`6, 273, 256`)，第一个维度代表节奏的`6`个位置，第二个和第三个维度代表`2D`频谱图。
- `test/` - 测试集节奏片段文件；你必须预测节奏中是否含有“针”，这是本次比赛的目标。
- `Sample_submission.csv` - 正确格式的示例提交文件。
- `train_labels` - 与`train/`文件夹中找到的节奏片段文件相对应的目标（按`id`）。
- `old_leaky_data `- 完整的重新启动前数据，包括测试标签。

```python
train_label_df = pd.read_csv('../input/c/seti-breakthrough-listen/train_labels.csv')
test_label_df  = pd.read_csv('../input/c/seti-breakthrough-listen/sample_submission.csv')

train_paths = glob('../input/c/seti-breakthrough-listen/train/**/*.npy')
test_paths = glob('../input/c/seti-breakthrough-listen/test/**/*.npy')

train_df = pd.DataFrame({'filepath':train_paths})
train_df['id'] = train_df.filepath.map(lambda x: x.split('/')[-1].split('.')[0])
train_df['group'] = train_df.filepath.map(lambda x: x.split('/')[-2])
train_df = pd.merge(train_df, train_label_df, on='id', how='left')

print(f'num_train: {len(train_paths)}\nnum_test : {len(test_paths)}')

# num_train: 60000
# num_test : 39995
```
##### 分类分布

`Tfrecord`数据集共有`20`个文件，为了正确的验证方案，每个文件都已分层`target & group feature`。
```python
from plotly.offline import init_notebook_mode, iplot, plot
import plotly as py
import plotly.express as px
init_notebook_mode(connected=True)
import plotly.graph_objs as go

fig = go.Figure(data=[
    go.Bar(name='haystack', 
           y=train_df.target.value_counts().values[0:1],
           x=['haystack'],
           text = train_df.target.value_counts()[0:1],
           orientation='v',
           textposition='outside',),
    go.Bar(name='needle', 
           y=train_df.target.value_counts().values[1:],
           x=['needle'],
           text = train_df.target.value_counts()[1:],
           orientation='v',
           textposition='outside',)
])
# Change the bar mode
fig.update_layout(
                  width=800,
                  height=600,
                  title=f'Class Distribution',
                  yaxis_title='Number of Images',
                  xaxis_title='Class Name',)
iplot(fig)
```
{% asset_img ca_2.png %}

#### 数据增强

使用简单的增强，其中一些可能会损害模型。
- 随机翻转（左右） 
- 无旋转 
- 随机亮度 
- 随机对比 
- 剪切 
- 缩放
- 粗略丢弃/切断

由于这不是典型的图片数据，而是信号，典型的图像增强可能会造成一些损害。
```python
# Augmentations by @cdeotte
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformmatrix which transforms indicies
    # CONVERT DEGREES TO RADIANS
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix,     shift_matrix))

def transform(image, DIM=IMG_SIZES[0]):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    
    # fixed for non-square image thanks to Chris Deotte
    
    if DIM[0]!=DIM[1]:
        pad = (DIM[0]-DIM[1])//2
        image = tf.pad(image, [[0, 0], [pad, pad+1],[0, 0]])
        
    NEW_DIM = DIM[0]
    
    XDIM = NEW_DIM%2 #fix for size 331
    
    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(NEW_DIM//2, -NEW_DIM//2,-1), NEW_DIM)
    y   = tf.tile(tf.range(-NEW_DIM//2, NEW_DIM//2), [NEW_DIM])
    z   = tf.ones([NEW_DIM*NEW_DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -NEW_DIM//2+XDIM+1, NEW_DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([NEW_DIM//2-idx2[0,], NEW_DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
    
    if DIM[0]!=DIM[1]:
        image = tf.reshape(d,[NEW_DIM, NEW_DIM,3])
        image = image[:, pad:DIM[1]+pad,:]
    image = tf.reshape(image, [*DIM, 3])
        
    return image
```
##### Dropout

粗略`Dropout`和`Cutout`增强是防止过度拟合和支持泛化的技术。他们从训练图像中随机删除矩形。通过删除图像的一部分，激励我们的模型关注整个图像，因为它永远不知道图像的哪一部分将出现。（这与`CNN`中的`dropout`层相似但又不同）。
- 剪切是去除`1`个随机大小的矩形的技术。
- 粗略剔除是去除许多大小相似的小矩形的技术。

通过更改下面的参数，我们可以进行粗略的`dropout`或`cutout`。
```python
# by cdotte
def dropout(image,DIM=IMG_SIZES[0], PROBABILITY = 0.6, CT = 5, SZ = 0.1):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)
    if (P==0)|(CT==0)|(SZ==0): 
        return image
    
    for k in range(CT):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM[1]),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM[0]),tf.int32)
        # COMPUTE SQUARE 
        WIDTH = tf.cast( SZ*min(DIM),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM[0],y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM[1],x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.zeros([yb-ya,xb-xa,3], dtype = image.dtype) 
        three = image[ya:yb,xb:DIM[1],:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM[0],:,:]],axis=0) 

    image = tf.reshape(image,[*DIM,3])
    return image
```
##### 读取TFRecord数据

那么，有没有想过为什么人们使用`TFRecord`？
- `TFRecord(.tfrecord/.tfrec)`格式是`TensorFlow`的自定义数据格式，用于存储二进制记录序列。
- `TFRecord`数据在存储磁盘上占用的空间更少，从磁盘读取和写入的时间也更少。
- 使用`TFRecord`有许多优点：
    - 更高效的存储
    - 快速输入/输出
    - 独立文件
    - `TPU`要求您以`TFRecord`格式向其传递数据

要使用`tfrecrod`构建数据管道，我们需要使用`tf.data` `API`。那么，`tf.data`是什么？
- `tf.data` `API`使我们能够从简单、可重用的片段构建复杂的输入管道。例如，图像模型的管道可能会聚合分布式文件系统中文件的数据，对每个图像应用随机变换，并将随机选择的图像合并到一批中进行训练。
- `tf.data` `API`提供了`tf.data.Dataset`抽象，它表示一系列元素，其中每个元素包含一个或多个组件。例如，在图像管道中，元素可能是单个训练示例，具有代表图像及其标签的一对张量分量。

```python
def read_labeled_tfrecord(example):
    tfrec_format = {
        'image'    : tf.io.FixedLenFeature([], tf.string),
        'image_id' : tf.io.FixedLenFeature([], tf.string),
        'target'   : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['target']


def read_unlabeled_tfrecord(example, return_image_id):
    tfrec_format = {
        'image'     : tf.io.FixedLenFeature([], tf.string),
        'image_id'  : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return example['image'], example['image_id'] if return_image_id else 0

 
def prepare_image(img, augment=True, dim=IMG_SIZES[0]):    
    img = tf.image.decode_png(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img/255.0 # rescale image
    
    if augment:
        img = transform(img,DIM=dim) if TRANSFORM else img
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, sat[0], sat[1])
        img = tf.image.random_contrast(img, cont[0], cont[1])
        img = tf.image.random_brightness(img, bri)     
                      
    img = tf.reshape(img, [*dim, 3])
            
    return img

def count_data_items(fileids):
    n = [int(re.compile(r"-([0-9]*)\.").search(fileid).group(1)) 
         for fileid in fileids]
    return np.sum(n)
```
#### 数据管道

- 读取`TFRecord`文件。
- 缓存数据以加快训练速度。
- 重复数据流（仅用于训练和测试时间增强）。
- 打乱数据（仅用于训练）。
- 解析`TFRecord`数据。
- 将数据从`ByteString`解码为图像数据。
- 处理图像数据（重新缩放）。
- 应用增强。
- 批次数据。

```python
def get_dataset(files, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_ids=True, batch_size=16, dim=IMG_SIZES[0]):
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO) # read tfrecord
    ds = ds.cache() # cache data for speedup
    
    if repeat:
        ds = ds.repeat() # repeat the data (for training only)
    
    if shuffle: 
        ds = ds.shuffle(1024*2, seed=SEED) # shuffle the data (for training only)
        opt = tf.data.Options()
        opt.experimental_deterministic = False # order won't be maintained when we shuffle
        ds = ds.with_options(opt)
        
    if labeled: 
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO) # unparse tfrecord data with labels
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_ids), 
                    num_parallel_calls=AUTO)      # unparse tfrecord data without labels
    
    ds = ds.map(lambda img, imgid_or_label: (prepare_image(img, augment=augment, dim=dim), 
                imgid_or_label),  # get img from bytestring, augmentations
                num_parallel_calls=AUTO)
    if labeled and augment:
        ds = ds.map(lambda img, label: (dropout(img, DIM=dim, PROBABILITY = PROBABILITY, CT = CT, SZ = SZ), label),
                    num_parallel_calls=AUTO)  # use dropout only in training
    
    ds = ds.batch(batch_size * REPLICAS) # batch the data
    ds = ds.prefetch(AUTO) # prefatch data for speedup
    return ds
```
#### 可视化

```python
def display_batch(batch, size=3):
    imgs, tars = batch
    for img_idx in range(size):
        plt.figure(figsize=(5*2, 15*2))
        for idx in range(3):
            plt.subplot(size, 3, idx+1)
            plt.title(f'id:{tars[img_idx].numpy().decode("utf-8")}')
            plt.imshow(imgs[img_idx,:, :, idx])
            plt.text(5, 15, str(idx), bbox={'facecolor': 'white'})
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.show() 

fold = 0
ds = get_dataset(files_train, augment=True, shuffle=False, repeat=True,labeled=False,return_image_ids=True,
                dim=IMG_SIZES[fold], batch_size = BATCH_SIZES[fold])
ds = ds.unbatch().batch(20)
batch = next(iter(ds))
display_batch(batch, 3)
```
{% asset_img ca_3.png %}
{% asset_img ca_4.png %}
{% asset_img ca_5.png %}

#### 构建模型

{% asset_img ca_6.png %}

您可以尝试其他模型，例如， 
- `Vision Transformer(ViT)`
- 残差网络(`ResNet`) 
- 创始网络(`InceptionNet`) 
- `XceptionNet`

```python
EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6, efn.EfficientNetB7]

def build_model(dim=IMG_SIZES[0], ef=0):
    inp = tf.keras.layers.Input(shape=(*dim,3)) # input layer with propoer img-dimension
    base = EFNS[ef](input_shape=(*dim,3),weights='imagenet',include_top=False) # get base model (efficientnet), use imgnet weights
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) # use GAP to get pooling result form conv outputs
    x = tf.keras.layers.Dense(32, activation = 'relu')(x) # use activation to apply non-linearity
    x = tf.keras.layers.Dense(1,activation='sigmoid')(x) # use sigmoid to convert predictions to [0-1]
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01)  # label smoothing for robustness
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    return model
```
#### Learning Rate Schedule

- 这是迁移学习的常见训练进度。
- 学习率开始时接近于零，然后增加到最大值，然后随着时间的推移而衰减。考虑改变时间表和/或学习率。请注意，随着批量大小的增加，最大学习率也随之增大。

```python
def get_lr_callback(batch_size=8, plot=False):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr
    if plot:
        plt.figure(figsize=(10,5))
        plt.plot(np.arange(EPOCHS[0]), [lrfn(epoch) for epoch in np.arange(EPOCHS[0])], marker='o')
        plt.xlabel('epoch'); plt.ylabel('learnig rate')
        plt.title('Learning Rate Scheduler')
        plt.show()

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

_=get_lr_callback(BATCH_SIZES[0], plot=True )
```
{% asset_img ca_7.png %}

#### 训练

我们的模型将根据您在上述配置中选择的`FOLDS`和`EPOCHS`数量进行训练。每次折叠(`fold`)时，验证`AUC`最低的模型都会被保存并用于预测`OOF`和测试。
```python
skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
oof_pred = []; oof_tar = []; oof_val = []; oof_f1 = []; oof_ids = []; oof_folds = [] 
preds = np.zeros((count_data_items(files_test),1))

for fold,(idxT,idxV) in enumerate(skf.split(np.arange(num_train_files))):
    # DISPLAY FOLD INFO
    if DEVICE=='TPU':
        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
    
    # CREATE TRAIN AND VALIDATION SUBSETS
    files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxT])
    np.random.shuffle(files_train);
    files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxV])
    files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))
    
    print('#'*25); print('#### FOLD',fold+1)
    print('#### Image Size: (%i, %i) | model: %s | batch_size %i'%
          (IMG_SIZES[fold][0],IMG_SIZES[fold][1],EFNS[EFF_NETS[fold]].__name__,BATCH_SIZES[fold]*REPLICAS))
    train_images = count_data_items(files_train)
    val_images   = count_data_items(files_valid)
    print('#### Training: %i | Validation: %i'%(train_images, val_images))
    
    # BUILD MODEL
    K.clear_session()
    with strategy.scope():
        model = build_model(dim=IMG_SIZES[fold],ef=EFF_NETS[fold])
    print('#'*25)   
    # SAVE BEST MODEL EACH FOLD
    sv = tf.keras.callbacks.ModelCheckpoint(
        'fold-%i.h5'%fold, monitor='val_auc', verbose=0, save_best_only=True,
        save_weights_only=True, mode='max', save_freq='epoch')
   
    # TRAIN
    print('Training...')
    history = model.fit(
        get_dataset(files_train, augment=AUGMENT, shuffle=True, repeat=True,
                dim=IMG_SIZES[fold], batch_size = BATCH_SIZES[fold]), 
        epochs=EPOCHS[fold], 
        callbacks = [sv,get_lr_callback(BATCH_SIZES[fold])], 
        steps_per_epoch=count_data_items(files_train)/BATCH_SIZES[fold]//REPLICAS,
        validation_data=get_dataset(files_valid,augment=False,shuffle=False,
                repeat=False,dim=IMG_SIZES[fold]), 
        #class_weight = {0:1,1:2},
        verbose=VERBOSE
    )
    
    # Loading best model for inference
    print('Loading best model...')
    model.load_weights('fold-%i.h5'%fold)  
    
    # PREDICT OOF USING TTA
    print('Predicting OOF with TTA...')
    ds_valid = get_dataset(files_valid,labeled=False,return_image_ids=False,augment=AUGMENT,
            repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*2)
    ct_valid = count_data_items(files_valid); STEPS = TTA * ct_valid/BATCH_SIZES[fold]/2/REPLICAS
    pred = model.predict(ds_valid,steps=STEPS,verbose=VERBOSE)[:TTA*ct_valid,] 
    oof_pred.append( np.mean(pred.reshape((ct_valid,TTA),order='F'),axis=1) )                 
    
    # GET OOF TARGETS AND idS
    ds_valid = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
            labeled=True, return_image_ids=True)
    oof_tar.append( np.array([target.numpy() for img, target in iter(ds_valid.unbatch())]) )
    oof_folds.append( np.ones_like(oof_tar[-1],dtype='int8')*fold )
    ds = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                labeled=False, return_image_ids=True)
    oof_ids.append( np.array([img_id.numpy().decode("utf-8") for img, img_id in iter(ds.unbatch())]))
    
    # PREDICT TEST USING TTA
    print('Predicting Test with TTA...')
    ds_test = get_dataset(files_test,labeled=False,return_image_ids=False,augment=AUGMENT,
            repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*2)
    ct_test = count_data_items(files_test); STEPS = TTA * ct_test/BATCH_SIZES[fold]/2/REPLICAS
    pred = model.predict(ds_test,steps=STEPS,verbose=VERBOSE)[:TTA*ct_test,] 
    preds[:,0] += np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1) * WGTS[fold]
    
    # REPORT RESULTS
    auc = roc_auc_score(oof_tar[-1],oof_pred[-1])
    oof_val.append(np.max( history.history['val_auc'] ))
    print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f'%(fold+1,oof_val[-1],auc))
    
    # PLOT TRAINING
    if DISPLAY_PLOT:
        plt.figure(figsize=(15,5))
        plt.plot(np.arange(len(history.history['auc'])),history.history['auc'],'-o',label='Train auc',color='#ff7f0e')
        plt.plot(np.arange(len(history.history['auc'])),history.history['val_auc'],'-o',label='Val auc',color='#1f77b4')
        x = np.argmax( history.history['val_auc'] ); y = np.max( history.history['val_auc'] )
        xdist = plt.xlim()[1] - plt.xlim()[0]; ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color='#1f77b4'); plt.text(x-0.03*xdist,y-0.13*ydist,'max auc\n%.2f'%y,size=14)
        plt.ylabel('auc',size=14); plt.xlabel('Epoch',size=14)
        plt.legend(loc=2)
        plt2 = plt.gca().twinx()
        plt2.plot(np.arange(len(history.history['auc'])),history.history['loss'],'-o',label='Train Loss',color='#2ca02c')
        plt2.plot(np.arange(len(history.history['auc'])),history.history['val_loss'],'-o',label='Val Loss',color='#d62728')
        x = np.argmin( history.history['val_loss'] ); y = np.min( history.history['val_loss'] )
        ydist = plt.ylim()[1] - plt.ylim()[0]
        plt.scatter(x,y,s=200,color='#d62728'); plt.text(x-0.03*xdist,y+0.05*ydist,'min loss',size=14)
        plt.ylabel('Loss',size=14)
        plt.title('FOLD %i - Image Size (%i, %i), %s'%
                (fold+1,IMG_SIZES[fold][0],IMG_SIZES[fold][1],EFNS[EFF_NETS[fold]].__name__),size=18)
        plt.legend(loc=3)
        plt.savefig(f'fig{fold}.png')
        plt.show()

```
{% asset_img ca_8.png %}
{% asset_img ca_9.png %}
{% asset_img ca_10.png %}
{% asset_img ca_11.png %}
{% asset_img ca_12.png %}

#### 计算 OOF

`OOF`（未折叠）预测将保存到磁盘。如果您希望集成多个模型，请使用`OOF`来确定混合模型的最佳权重。当用于混合`OOF`时，选择可最大化`OOF CV`分数的权重。然后使用这些相同的权重来混合您的测试预测。

{% note warning %}
**注意**：
- 不要仅仅为了提升`LB`而进行混合，因为大多数时候它最终会变得过度拟合。
- 尝试通过混合不同的模型来提高`CV`，您可以密切关注`LB`。
- 由于只有`20%`的数据将用于计算`LB`分数，因此依靠`CV`应该是一个安全的选择。
{% endnote %}

```python
# COMPUTE OVERALL OOF AUC
oof = np.concatenate(oof_pred); true = np.concatenate(oof_tar);
ids = np.concatenate(oof_ids); folds = np.concatenate(oof_folds)
auc = roc_auc_score(true,oof)
print('Overall OOF AUC with TTA = %.3f'%auc)

# SAVE OOF TO DISK
df_oof = pd.DataFrame(dict(image_id = ids, target=true, pred = oof, fold=folds))
df_oof.to_csv('oof.csv',index=False)
df_oof.head()

# Overall OOF AUC with TTA = 0.669

# 	image_id	target	pred	fold
# 0	707ec47b606a3d0	0	0.083118	0
# 1	743d0594f4e2ff7	0	0.059541	0
# 2	747d5a1f5ab25a9	0	0.086644	0
# 3	786909d18574aeb	0	0.040169	0
# 4	7726ef7381e86c7	0	0.091762	0
```
#### 后期处理

有多种方法可以根据`patient`信息修改预测以增加`CV-LB`。您可以在`OOF`上进行实验。

#### 提交

```python
ds = get_dataset(files_test, augment=False, repeat=False, dim=IMG_SIZES[fold],labeled=False, return_image_ids=True)
image_ids = np.array([img_id.numpy().decode("utf-8") for img, img_id in iter(ds.unbatch())])

submission = pd.DataFrame({'id':image_ids, 'target':preds[:,0]})
submission = submission.sort_values('id') 
submission.to_csv('submission.csv', index=False)
submission.head()

# 	id	target
# 27604	000bf832cae9ff1	0.096986
# 29604	000c74cc71a1140	0.081791
# 18754	000f5f9851161d3	0.085843
# 17054	000f7499e95aba6	0.128763
# 20854	00133ce6ec257f9	0.080760
```
#### 预测分布

```python
plt.figure(figsize=(10,5))
plt.hist(submission.target,bins=100);
```
{% asset_img ca_13.png %}
