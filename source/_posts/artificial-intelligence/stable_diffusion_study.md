---
title: Stable Diffusion（KerasCV）
date: 2024-04-14 19:06:11
tags:
  - AI
categories:
  - 人工智能
---

使用`KerasCV`的稳定扩散图像生成。`Stable Diffusion`是一个强大的文本 `->` 图像模型，有`Stability AI`开源。虽然存在多种开源实现，可以轻松地根据文本提示创建图像，但`KerasCV`提供了一些优势：其中包括`XLA`编译和混合精度支持，他们共同实现最优的生成，使用`KerasCV`调用`Stable Diffusion`非常简单。我们传入一个字符串，通常称为**提示**，批量大小为`3`。模型能够生成三张令人惊艳的图片，正如**提示**所描述：
<!-- more -->
{% asset_img sd_1.png %}

```python
import time
import keras_cv
import keras
import matplotlib.pyplot as plt

# 首先，我们构建一个模型
model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=False)

# 接下来，我们给它一个提示：
images = model.text_to_image("photograph of an astronaut riding a horse", batch_size=3)
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)

# 但这并不是该模型所能做的全部。让我们尝试一个更复杂的提示：
images = model.text_to_image(
    "cute magical flying dog, fantasy art, "
    "golden color, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
plot_images(images)
```
{% asset_img sd_1.png %}
{% asset_img sd_2.png %}

#### 这到底是怎么工作的？

要从潜在扩散文本 `—>` 图像，您需要添加一个关键特征：通过提示关键字控制生成的视觉内容的能力。这是通过“调节”来完成的，这是一种经典的深度学习技术，其中包括将一段文本的向量连接到噪声补丁，然后在`{image:caption}`的数据集上训练模型。这就产生了稳定扩散架构。 稳定扩散由三部分组成：
- 文本编码器，可将您的提示转换为潜在向量。
- 扩散模型，反复对`64x64`潜在图像块进行“去噪”。
- 解码器，将最终的`64x64`潜在补丁转换为更高分辨率的`512x512`图像。

首先，您的文本提示由文本编码器投影到潜在向量空间中，这只是一个预训练的冻结语言模型。然后，将该提示向量连接到随机生成的**噪声块**，该噪声块通过**扩散模型**在一系列“步骤”上重复“去噪”（运行的步骤越多，图像就会越清晰、越好-默认值为`50`次）。最后，`64x64`潜在图像通过解码器发送，用高分辨率渲染它。
{% asset_img sd_3.png %}

上图中有一个文本编码器将，可以将提示字符串转换潜在的向量，该向量连接到随机生成的噪声补丁。新的向量将通过扩散模型重复去噪，最后潜在图像通过解码器，将`64 x 64`**图像块**转换为更高分辨率的`512 x 512`图像。

总而言之，这是一个非常简单的系统 — `Keras`实现包含在四个文件中，总共不到`500`行代码：
- `text_encoder.py: 87 LOC`
- `diffusion_model.py: 181 LOC`
- `decoder.py: 86 LOC`
- `stable_diffusion.py: 106 LOC`

但一旦你训练了数十亿张图片及其标题，这个相对简单的系统就开始看起来像**魔法**一样。 正如费曼对宇宙的评价：**“宇宙并不复杂，只是很多！”**。
```python
benchmark_result = []
start = time.time()
images = model.text_to_image(
    "A cute otter in a rainbow whirlpool holding shells, watercolor",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["Standard", end - start])
plot_images(images)

print(f"Standard model: {(end - start):.2f} seconds")
keras.backend.clear_session()  # Clear session to preserve memory.

# 50/50 ━━━━━━━━━━━━━━━━━━━━ 10s 209ms/step
# Standard model: 10.57 seconds
```
{% asset_img sd_4.png %}

现在我们打开混合精度，使用float16精度执行计算，同时存储float32存储权重。
```python
# Warm up model to run graph tracing before benchmarking.
model.text_to_image("warming up the model", batch_size=3)

start = time.time()
images = model.text_to_image(
    "a cute magical flying dog, fantasy art, "
    "golden color, high quality, highly detailed, elegant, sharp focus, "
    "concept art, character concepts, digital painting, mystery, adventure",
    batch_size=3,
)
end = time.time()
benchmark_result.append(["Mixed Precision", end - start])
plot_images(images)

print(f"Mixed precision model: {(end - start):.2f} seconds")
keras.backend.clear_session()

# 50/50 ━━━━━━━━━━━━━━━━━━━━ 42s 132ms/step
# 50/50 ━━━━━━━━━━━━━━━━━━━━ 6s 129ms/step
# Mixed precision model: 6.65 seconds
```
{% asset_img sd_5.png %}

这更快是因为`NVIDIA GPU`具有专门的`FP16`运算内核。其运行速度比FP32同类产品更快。接下来我们尝试一下`XLA`编译，我们可以通过