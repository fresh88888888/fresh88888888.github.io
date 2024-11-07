---
title: 机器学习(ML)(十五) — 推荐系统探析
date: 2024-11-06 16:10:11
tags:
  - AI
categories:
  - 人工智能
mathjax:
  tex:
    tags: 'ams'
  svg:
    exFactor: 0.03
---

#### 特征交叉 - 因式分解机(FM)

假设有{% mathjax %}d{% endmathjax %}个特征，记作{% mathjax %}\mathbf{x} = \{x_1,\ldots,x_d\}{% endmathjax %}，这是个**线性模型**，记作{% mathjax %}p = b + \sum_{i=1}^d w_ix_i{% endmathjax %}，其中{% mathjax %}b{% endmathjax %}是偏移项，叫做`bias`，第二项{% mathjax %}\sum_{i=1}^d w_ix_i{% endmathjax %}是{% mathjax %}d{% endmathjax %}个特征的连加，其中{% mathjax %}w_i{% endmathjax %}表示每个特征的权重，{% mathjax %}p{% endmathjax %}是线性模型的输出，它是对目标的预估。这个线性模型有{% mathjax %}d + 1{% endmathjax %}个参数，{% mathjax %}\mathbf{w} = [w_1,\ldots,w_d]{% endmathjax %}是权重，{% mathjax %}b{% endmathjax %}是偏移项。线性模型的预测是特征的加权和。特征之间没有交叉，在推荐系统中，特征交叉是很有必要的，可以让模型的预测更准正确。
<!-- more -->

**二阶交叉特征**：假设有{% mathjax %}d{% endmathjax %}个特征，记作{% mathjax %}\mathbf{x} = \{x_1,\ldots,x_d\}{% endmathjax %}，线性模型 + 二阶交叉特征，记作为{% mathjax %}p = b + \sum_{i=1}^d w_ix_i + \sum_{i = 1}^d\sum_{j = i + 1}^d u_{ij}x_i x_j{% endmathjax %}，其中{% mathjax %}x_ix_j{% endmathjax %}是特征的交叉，{% mathjax %}u_{ij}{% endmathjax %}是权重，特征交叉可以提升模型的表达能力，举个例子，假设特征时房屋的大小、周边楼盘每平米的单价，目标是估计房屋的价格，仅仅房屋的大小、每平米的单价的加权和是估不准房屋价格的。如果做特征交叉，房屋的大小和每平米价格这两个特征相乘就能把房屋价格估的很准，这就是为什么交叉特征有用。如果有{% mathjax %}d{% endmathjax %}个特征，则模型参数量为{% mathjax %}\mathcal{O}(d^2){% endmathjax %}，那么模型参数量正比于`dsquared`，是交叉特征的权重{% mathjax %}u{% endmathjax %}，如果{% mathjax %}d{% endmathjax %}比较小，这样的模型没有问题，但如果{% mathjax %}d{% endmathjax %}很大，模型参数数量就太大了，计算代价会很大，而且容易出现**过度拟合**。

可以把所有权重{% mathjax %}u_{ij}{% endmathjax %}组合成{% mathjax %}\mathbf{U}{% endmathjax %}，{% mathjax %}u_{ij}{% endmathjax %}是矩阵{% mathjax %}\mathbf{U}{% endmathjax %}上的第{% mathjax %}i{% endmathjax %}行，第{% mathjax %}j{% endmathjax %}列上的元素。矩阵{% mathjax %}\mathbf{U}{% endmathjax %}有{% mathjax %}d{% endmathjax %}行和{% mathjax %}d{% endmathjax %}列，{% mathjax %}d{% endmathjax %}是参数的数量，{% mathjax %}\mathbf{U}{% endmathjax %}是个对称矩阵，可以对对称矩阵{% mathjax %}\mathbf{U}{% endmathjax %}做近似，可以用矩阵{% mathjax %}\mathbf{V}{% endmathjax %}乘以{% mathjax %}\mathbf{V}^{\mathsf{T}}{% endmathjax %}来近似矩阵{% mathjax %}\mathbf{U}{% endmathjax %}。矩阵{% mathjax %}\mathbf{V}{% endmathjax %}有{% mathjax %}d{% endmathjax %}行，{% mathjax %}k{% endmathjax %}列，{% mathjax %}k{% endmathjax %}是个超参数，由自己设置。{% mathjax %}k{% endmathjax %}越大，{% mathjax %}\mathbf{V}\cdot\mathbf{V}^{\mathsf{T}}{% endmathjax %}就越接近矩阵{% mathjax %}\mathbf{U}{% endmathjax %}。{% mathjax %}u_{ij}{% endmathjax %}可以近似成{% mathjax %}\mathbf{v}^{\mathsf{T}}_i \mathbf{v}_j{% endmathjax %}，那么模型就变成了**因式分解机**(`Factorized Machine`)，记作{% mathjax %}p = b + \sum_{i=1}^d w_ix_i + \sum_{i = 1}^d\sum_{j = i + 1}^d (\mathbf{v}^{\mathsf{T}}_i \mathbf{v}_j)x_i x_j{% endmathjax %}。如下图所示：
{% asset_img ml_1.png %}

**因式分解机**(`Factorized Machine`)模型的参数数量为{% mathjax %}\mathcal{O}(kd){% endmathjax %}({% mathjax %}k \ll d{% endmathjax %})，**因式分解机**(`Factorized Machine`)的好处是参数数量更少，由{% mathjax %}\mathcal{O}(d^2){% endmathjax %}降低到了{% mathjax %}\mathcal{O}(kd){% endmathjax %}，这样使得推理的计算量更小，而且不容易出现过拟合，**因式分解机**(`Factorized Machine`)是线性模型的替代品，凡是能用线性回归、逻辑回归的场景，都可以用**因式分解机**(`Factorized Machine`)。**因式分解机**(`Factorized Machine`)使用**二阶交叉特征**，表达能力比线性模型更强，**因式分解机**(`Factorized Machine`)的效果，显著比线性模型要好。通过做近似{% mathjax %}u_{ij} \approx \mathbf{v}^{\mathsf{T}}_i \mathbf{v}_j{% endmathjax %}，**因式分解机**(`Factorized Machine`)把二阶交叉权重的数量从{% mathjax %}\mathcal{O}(d^2){% endmathjax %}降低到{% mathjax %}\mathcal{O}(kd){% endmathjax %}，**因式分解机**(`Factorized Machine`)的论文，请参考[`Factorization Machines`](https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf)。

#### 特征交叉 - 深度交叉网络(DCN)

**深度交叉网络**(`Deep & Cross Networks`)是一种结合了**深度神经网络**(`DNN`)和**特征交叉**的混合模型，旨在高效处理**高维稀疏数据**。**深度交叉网络**(`Deep & Cross Networks`)的设计初衷是为了克服传统**前馈神经网络**在特征交互建模方面的局限性，尤其是在**推荐系统**和**广告点击预测**等应用中表现出色。在**推荐系统**中，**深度交叉网络**(`DCN`)既可以用于召回也可以用于排序模型。双塔模型是一种框架而不是网络，用户塔和物品塔可以用任意网络结构，比如**深度交叉网络**；多目标模型中的神经网络也可以用任意网络结构所替代，比如**深度交叉网络**。

接下来看一下**深度交叉网络**(`DCN`)的结构：
- **交叉层**(`cross layer`)：假设把输入记作向量{% mathjax %}x_0{% endmathjax %}，经过{% mathjax %}i{% endmathjax %}层之后，输出向量{% mathjax %}x_i{% endmathjax %}，把向量{% mathjax %}x_i{% endmathjax %}输入一个全连接层，全连接层输出向量{% mathjax %}y{% endmathjax %}，把第一层的向量{% mathjax %}x_0{% endmathjax %}与向量{% mathjax %}y{% endmathjax %}做**阿达玛乘积**(`Hadamard Product`)，**阿达玛乘积**(`Hadamard Product`)也称为**逐项乘积**或**舒尔乘积**，是一种二元运算，适用于两个相同维度的矩阵。该运算的结果是一个新矩阵，其每个元素等于输入矩阵对应位置元素的乘积。如果{% mathjax %}x_0{% endmathjax %}和{% mathjax %}y{% endmathjax %}都是`8`维的向量，那么它们的乘积也是`8`维的向量，把输出的向量记作{% mathjax %}z{% endmathjax %}，向量{% mathjax %}x_i{% endmathjax %}和{% mathjax %}z{% endmathjax %}分别是输入和输出，两个向量的形状是一样的，把向量{% mathjax %}x_i{% endmathjax %}和{% mathjax %}z{% endmathjax %}相加，得到向量{% mathjax %}x_{i+1}{% endmathjax %}，向量{% mathjax %}x_{i+1}{% endmathjax %}是第{% mathjax %}i{% endmathjax %}个交叉层的输出，向量{% mathjax %}x_0,x_i{% endmathjax %}是这个交叉层的输入，这个交叉层的参数全都在全连接层里边，交叉层可以记作{% mathjax %}x_{i+1} = x_0\odot (w\cdot x_i + b) + x_i{% endmathjax %}，其中{% mathjax %}x_0{% endmathjax %}是神经网络最底层额输入，{% mathjax %}x_i{% endmathjax %}是第{% mathjax %}i{% endmathjax %}层的输入，{% mathjax %}w\cdot x_i + b{% endmathjax %}是全连接层，而全连接层的输出是一个向量，跟输入{% mathjax %}x_i{% endmathjax %}的大小是一样的。矩阵{% mathjax %}w{% endmathjax %}和{% mathjax %}b{% endmathjax %}是全连接层的参数，也是这个交叉层中全部的参数，参数需要再训练的过程中通过梯度去更新，把向量{% mathjax %}x_0{% endmathjax %}与全连接层做**阿达玛乘积**(`Hadamard Product`)，最后把**阿达玛乘积**(`Hadamard Product`)的结果与向量{% mathjax %}x_i{% endmathjax %}相加，把输入与输出相加，这就是`ResNet`中的跳跃连接，这样可以防止梯度消失。向量{% mathjax %}x_{i+1}{% endmathjax %}是交叉层的输出，形状跟{% mathjax %}x_0,x_i{% endmathjax %}是一样的，也就是说每个交叉层的输入与输出都是向量，而且形状相同。
{% asset_img ml_2.png "交叉层" %}
- **交叉网络**(`cross network`)：向量{% mathjax %}x_0{% endmathjax %}是**交叉网络**的输入，把{% mathjax %}x_0{% endmathjax %}送入交叉层(参数：{% mathjax %}w_0,b_0{% endmathjax %})，交叉层输出向量{% mathjax %}x_1{% endmathjax %}，记作{% mathjax %}x_1 = x_0\odot (w_0\cdot x_0 + b_0) + x_0 {% endmathjax %}，其中{% mathjax %}w_0,b_0{% endmathjax %}是这个交叉层中的参数，把上一层的输出向量{% mathjax %}x_1{% endmathjax %}和{% mathjax %}x_0{% endmathjax %}一起输入下一个交叉层(参数：{% mathjax %}w_1,b_1{% endmathjax %})，这个交叉层输出向量{% mathjax %}x_2{% endmathjax %}，记作{% mathjax %}x_2 = x_0\odot (w_1\cdot x_1 + b_1) + x_1{% endmathjax %}，把向量{% mathjax %}x_2{% endmathjax %}和最底层向量{% mathjax %}x_0{% endmathjax %}一起输入下一个交叉层，该交叉层的参数是{% mathjax %}w_2,b_2{% endmathjax %}，这个交叉层输出向量{% mathjax %}x_3{% endmathjax %}，重复这个过程，可以加更多的交叉层，如果不加更多的交叉层，那么向量{% mathjax %}x_3{% endmathjax %}就是这个神经网络的输出。交叉网络相关的最新论文请参考：[`DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems`](https://arxiv.org/pdf/2008.13535)。如下图所示：
{% asset_img ml_3.png "交叉网络" %}

**推荐系统**召回、排序模型的输入是用户特征、物品特征、其它特征。把这些特征向量做`concatenation`，输入到两个**神经网络**（**全连接网络** `&` **交叉网络**），两个神经网络并联，两个神经网络各输出一个向量，把两个向量做`concatenation`，并输入到一个**全连接层**，全连接层输出一个向量，全连接网络、交叉网络、全连接层拼接到一起就是**深度交叉网络**(`DCN`)。其比单个全连接网络效果更好，**深度交叉网络**(`DCN`)既可以用于召回，也可以用于排序，双塔模型中的用户塔和物品塔都可以使用**深度交叉网络**(`DCN`)，多目标排序模型中的`shared bottom`、`MMoE`中的专家神经网络也都可以是**深度交叉网络**(`DCN`)。如下图所示：
{% asset_img ml_4.png "深度交叉网络" %}

#### 特征交叉 - HLUC

`LHUC`(`Learning Hidden Unit Contributions`)是一种**神经网络**结构，最初用于语音识别领域，旨在实现说话人自适应(`speaker adaptation`)。该方法通过为每个说话人学习特定的隐式单位贡献，从而提升不同说话人的语音识别效果。`LHUC`的核心思想是通过调整**隐藏层单元**的输出，使模型能够更好地适应不同的输入特征。`LHUC`只能应用在**精排模型**当中。在语音识别中的`LHUC`，语音识别的输入是一段语音信号，我们希望神经网络输入做变换得到更有效的表征，然后识别出语音中的文字，不同的声音会有所区别，最好是加入一些说话者的特征，最简单的特征就是说话者`ID`，对`ID`做`Embedding`，得到一个向量，作为这个说话者的表征。把语音信号输入到一个全连接层，输出一个向量；把说话者的特征输入到另一个神经网络，并输出一个向量。这个神经网络包含多个全连接层，最后一个全连接层的激活函数是{% mathjax %}sigmoid \times 2{% endmathjax %}，单独作用到每个元素上。也就是说这个向量每个元素都介于`0~2`之间。这两个向量的形状必须完全一致，这样也就可以计算这两个向量的**阿达玛乘积**(`Hadamard Product`)，对这2个向量逐元素相乘。这样就可以吧语音信号的特征，说话者的特征相融合。语音信号有的特征被放大，有的被缩小，这样可以做到个性化。**阿达玛乘积**(`Hadamard Product`)得到一个向量，形状与输入的特征是相通的，把这个向量输入到下一个全连接层，并输出一个形状相同的向量，把说话者特征也输入一个神经网络，这个神经网络有多个全连接层，最后一层的激活函数也是{% mathjax %}sigmoid \times 2{% endmathjax %}，神经网络输出一个向量，是对说话正的表征，可以计算这两个向量的**阿达玛乘积**(`Hadamard Product`)，并输出一个形状相同的向量。这个向量是从语音信号提取的特征，还结合了说话者的特征，做到了个性化，最后的向量就`是HLUC`的输出。这两个神经网络的输入都是刷花这特征，两个神经网络都是由多个全连接层组成，最后一层的激活函数都是{% mathjax %}sigmoid \times 2{% endmathjax %}，这样的向量跟**阿达玛乘积**(`Hadamard Product`)，会放大某些特征、缩小另一些些特征，从而实现个性化。在语音识别的应用中，两个输入的向量分别是：语音信号和说话者的特征。在**推荐系统**的排序模型中，两个输入变成了用户特征和物品特征。物品特征相当于语音信号特征、用户特征相当于说话者的特征，`HLUC`的结构还是跟之前一样的。如下图所示：
{% asset_img ml_5.png "LHUC网络结构" %}
