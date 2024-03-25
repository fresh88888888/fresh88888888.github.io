---
title: 迁移学习（TensorFlow & Keras）
date: 2024-03-25 16:20:32
tags:
  - AI
categories:
  - 人工智能
---

例如，我可能想要一个可以判断照片是在城市地区还是农村地区拍摄的模型，但我的原始模型不会将图像分为这两个特定类别。我可以为此特定目的从头开始构建一个新模型。但为了获得好的结果，我需要数千张带有城市和乡村标签的照片。一种称为**迁移学习**的方法可以用更少的数据给出良好的结果。**迁移学习**利用模型在解决一个问题时学到的知识（称为**预训练模型**，因为该模型已经在不同的数据集上进行了训练），并将其应用于新的应用程序中。
<!-- more -->
`ImageNet`是一个非常大的图像数据集，由来自数千个类别的超过`1400`万张图像组成。`Keras`在此提供了几个已在此数据集上进行预训练的模型。其中一种模型是`ResNet`。我们将向您展示如何使预训练的`ResNet`模型适应新任务，以预测图像是农村还是城市。您将使用此数据集。

#### 背景

请记住，**深度学习模型的早期层可以识别简单的形状。后面的层可以识别更复杂的视觉模式**，例如道路、建筑物、窗户和开阔的田野。这些层将在我们的新应用程序中有用。
{% asset_img tl_1.png %}

最后一层进行预测。我们将替换`ResNet`模型的最后一层。替换是具有两个节点的密集层。一个节点捕捉照片的城市化程度，另一个节点捕捉照片的乡村化程度。理论上，预测前最后一层的任何节点都可能告知其城市化程度。因此城市度量可以取决于该层中的所有节点。我们画出联系来表明这种可能的关系。出于同样的原因，每个节点的信息可能会影响我们对照片乡村程度的衡量。
{% asset_img tl_2.png %}

我们这里有很多连接，我们将使用训练数据来确定哪些节点表明图像是城市的，哪些节点表明图像是农村的，哪些节点无关紧要。也就是说，我们将训练模型的最后一层。实际上，训练数据将是标记为农村或城市的照片。
{% note warning %}
**注意**：当将某事物仅分为两类时，我们可以在输出处仅使用一个节点。在这种情况下，对照片城市化程度的预测也可以衡量它的乡村化程度。如果一张照片`80%`的可能性是城市，则`20%`的可能性是乡村。但我们在输出层保留了两个独立的节点。在输出层中为每个可能的类别使用单独的节点将有助于我们过渡到想要预测`2`个以上类别的情况。
{% endnote %}

#### Code

##### 指定模型

在此应用程序中，我们将照片分为`2`个类别：城市和农村。我们将其保存为`num_classes`。接下来我们构建模型。我们建立了一个可以添加层的顺序模型。首先我们添加一个预训练的`ResNet`模型。创建`ResNet`模型时，我们编写了`include_top=False`。这就是我们指定要排除进行预测的`ResNet`模型的最后一层的方式。我们还将使用一个不包含该层权重的文件。参数 `pooling='avg'`表示如果在这一步结束时我们的张量中有额外的通道，我们希望通过取平均值将它们折叠为一维张量。现在我们有一个预训练的模型，可以创建您在图形中看到的图层。我们将添加一个密集层来进行预测。我们指定该层中的节点数量，在本例中是类的数量。然后我们应用`softmax`函数来生成概率。
{% asset_img tl_3.png %}

最后，我们将告诉`TensorFlow`不要训练顺序模型的第一层，即`ResNet50`层。这是因为该模型已经使用`ImageNet`数据进行了预训练。
```python
# set random seed / make reproducible
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

num_classes = 2
resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False
my_new_model.summary()
```
```bash
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
resnet50 (Functional)        (None, 2048)              23587712  
_________________________________________________________________
dense (Dense)                (None, 2)                 4098      
=================================================================
Total params: 23,591,810
Trainable params: 4,098
Non-trainable params: 23,587,712
_________________________________________________________________
```

##### 编译模型

编译命令告诉`TensorFlow`如何在训练期间更新网络最后一层的关系。我们有一个衡量损失或不准确性的方法，希望将其最小化。我们将其指定为`categorical_crossentropy`。如果您熟悉对数损失，这是同一事物的另一个术语。我们使用一种称为**随机梯度下降**（`SGD`）的算法来**最小化分类交叉熵损失**。我们要求代码报告准确性指标，即**正确预测的比例**。这比**分类交叉熵分数**更容易解释，因此最好将其打印出来并查看模型的表现。
```python
my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
```
##### 加载图像数据

我们的原始数据分为训练数据目录和验证数据目录。在每个目录中，我们都有一个用于城市图片的子目录，另一个用于乡村图片的子目录。`TensorFlow`提供了一个很棒的工具，用于处理按标签分组到目录中的图像。这是**图像数据生成器**。使用`ImageDataGenerator`有两个步骤。首先我们抽象地创建生成器对象。我们希望在每次读取图像时应用`ResNet`预处理函数。然后我们使用 `flow_from_directory`命令。我们告诉它数据在哪个目录中，我们想要什么大小的图像，一次读入多少图像（批量大小），然后我们告诉它我们正在将数据分类为不同的类别。我们做同样的事情来设置读取验证数据的方法。`ImageDataGenerator`在处理大型数据集时尤其有价值，因为我们不需要立即将整个数据集保存在内存中。但这里还好，数据集很小。 请注意，**这些是生成器，这意味着我们需要迭代它们才能获取数据**。
```python
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

image_size = 224
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/train',
        target_size=(image_size, image_size),
        batch_size=12,
        class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
        '../input/urban-and-rural-photos/val',
        target_size=(image_size, image_size),
        batch_size=20,
        class_mode='categorical')
```
##### 拟合模型

现在我们拟合模型。训练数据来自`train_generator`，验证数据来自`validation_generator`。 由于我们有`72`个训练图像并一次读取`12`个图像，因此我们在单个`epoc`h中使用`6`个步骤 (`steps_per_epoch=6`)。同样，我们有`20`个验证图像，并使用一个验证步骤，因为我们在一步中读取了所有`20`个图像 (`validation_steps=1`)。随着模型训练的运行，我们将看到损失函数和准确性的进度更新。它更新了致密层中的连接，即模型对城市照片和乡村照片的印象。完成后，`78%`的训练数据都是正确的。然后它检查验证数据。`90%`的都答对了。我应该提到，这是一个非常小的数据集，您应该对依赖如此少量数据的验证分数犹豫不决。我们从小型数据集开始，这样您就可以通过可以快速训练的模型获得一些经验。
```python
my_new_model.fit(
        train_generator,
        steps_per_epoch=6,
        validation_data=validation_generator,
        validation_steps=1)
```
即使训练数据集很小，这个准确度分数也非常好。我们用`72`张照片进行训练。您可以轻松地在手机上拍摄那么多照片，并构建一个非常准确的模型来区分几乎所有您关心的东西。

##### 训练结果

在此阶段，打印的验证准确性可能明显优于训练准确性。一开始这可能会令人困惑。发生这种情况是因为随着网络的改进，训练精度是在多个点计算的（卷积中的数字正在更新以使模型更准确）。当模型看到第一张训练图像时，网络不准确，因为权重尚未经过太多训练/改进。 这些第一次训练结果被平均到上面的度量中。模型遍历所有数据后计算验证损失和准确性度量。因此，在计算这些分数时，网络已经经过充分训练。这在实践中并不是一个严重的问题，我们往往不用担心。