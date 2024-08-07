---
title: 生成对抗网络(GAN)（机器学习）
date: 2024-06-24 15:00:11
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

**生成对抗网络**(`GAN`)由`Goodfellow`等人在[`NeurIPS,2014`](https://arxiv.org/abs/1406.2661)中提出，是机器学习领域一项令人兴奋的最新创新。`GAN`是一种生成模型：它们会创建与您的训练数据相似的新数据实例。例如，即使这些脸不属于任何真实的人，`GAN`也可以创建看起来像人脸照片的图像。
<!-- more -->

#### 基本原理

`GAN`通过将生成器（学习生成目标输出）与鉴别器（学习区分真实数据和生成器输出）配对来实现这种真实性。生成器试图欺骗鉴别器，而鉴别器则试图不被欺骗。

#### 什么是生成模型？

“生成对抗网络”中的`“generative”`是什么意思？“生成”描述了一类与**判别模型形**成对比的统计模型。生成模型可以生成看起来像真实动物的新动物照片，而判别模型可以区分狗和猫。`GAN`只是生成模型的一种。
- 非正式地：生成模型可以生成新的数据实例；判别模型区分不同类型的数据实例。
- 更正式地：给定一组数据实例{% mathjax %}X{% endmathjax %}和一组标签{% mathjax %}Y{% endmathjax %}。生成模型捕捉联合概率为{% mathjax %}P(X,Y){% endmathjax %}，如果没有标签则概率为{% mathjax %}P(X){% endmathjax %}；判别模型捕捉条件概率为{% mathjax %}P(Y|X){% endmathjax %}。

生成模型包括数据本身的分布，并告诉您给定示例的可能性。例如，预测序列中下一个单词的模型通常是生成模型（通常比`GAN`简单得多），因为它们可以为单词序列分配概率。判别模型忽略了给定实例是否可能的问题，而只是告诉您标签应用于该实例的可能性有多大。

#### 概率建模

这两种模型都不必返回代表概率的数字。您可以通过模仿数据分布来对数据分布进行建模。例如，决策树之类的判别分类器可以标记实例而不为该标签分配概率。这样的分类器仍然是一个模型，因为所有预测标签的分布将模拟数据中标签的实际分布。类似地，生成模型可以通过生成看起来像是从该分布中提取的令人信服的“虚假”数据来对分布进行建模。

生成模型比类似的判别模型处理的任务更困难。生成模型必须进行更多建模。图像生成模型可能会捕捉到这样的相关性：“看起来像船的东西可能会出现在看起来像水的东西附近”和“眼睛不太可能出现在额头上”。这些都是非常复杂的分布。相比之下，判别模型可能只需寻找一些明显的模式就能学会“帆船”与“非帆船”之间的区别。它可能会忽略生成模型必须正确处理的许多相关性。判别模型试图在数据空间中划定边界，而生成模型则试图模拟数据在整个空间中的放置方式。例如，下图显示了手写数字的判别模型和生成模型：
{% asset_img g_1.png "判别模型和生成模型的手写示例" %}

判别模型通过在数据空间中画一条线来尝试区分手写的`0`和`1`。如果画对了线，它就可以区分`0`和`1`，而不必对实例在线两侧的数据空间中的位置进行精确建模。相比之下，生成模型则试图通过生成接近数据空间中真实数字的数字来产生令人信服的`1`和`0`。它必须对整个数据空间的分布进行建模。`GAN`提供了一种有效的方法来训练这些丰富的模型以类似于真实分布。

#### GAN结构概述

生成对抗网络(`GAN`)由两部分组成：
- 生成器学习生成可信的数据。生成的实例将成为鉴别器的反面训练示例。
- 鉴别器学会区分生成器的虚假数据和真实数据。鉴别器会惩罚产生不合理结果的生成器。

当训练开始时，生成器会生成明显是假的数据，而鉴别器很快就能分辨出这是假的：
{% asset_img g_2.png %}

随着训练的进行，生成器越来越接近产生可以欺骗鉴别器的输出：
{% asset_img g_3.png %}

在上图中，生成的数据现在有一个绿色矩形，左上角有数字10，还有一张简单的脸部图像。最后，如果生成器训练进展顺利，鉴别器分辨真假的能力就会变差。它开始将假数据归类为真数据，其准确率也会下降。
{% asset_img g_4.png %}

这是整个系统的图片：
{% asset_img g_5.png %}

生成器和鉴别器都是神经网络。生成器的输出直接连接到鉴别器的输入。通过反向传播，鉴别器的分类提供给生成器用来更新其权重的信号。
##### 生成器

`GAN`的生成器部分通过结合来自鉴别器的反馈来学习创建虚假数据。它学会让鉴别器将其输出分类为真实数据。与鉴别器训练相比，生成器训练需要生成器和鉴别器之间更紧密的集成。`GAN`中训练生成器的部分包括：
- 随机输入。
- 生成器网络，将随机输入转换为数据实例。
- 鉴别器网络，对生成的数据进行分类。
- 鉴别器输出
- 生成器损失，对未能欺骗鉴别器的生成器进行惩罚。

该图展示了生成器训练中的反向传播：
{% asset_img g_6.png %}

###### 随机输入

神经网络需要某种形式的输入。通常，我们输入想要处理的数据，例如我们想要分类或预测的实例。但是，对于输出全新数据实例的网络，我们使用什么作为输入呢？最基本的`GAN`形式是将随机噪声作为输入。然后，生成器将这种噪声转换为有意义的输出。通过引入噪声，我们可以让`GAN`生成各种各样的数据，从目标分布的不同位置进行采样。实验表明噪声的分布并不重要，因此我们可以选择易于采样的分布，例如均匀分布。为方便起见，噪声采样空间的维度通常小于输出空间的维度。请注意，有些`GAN`使用非随机输入来塑造输出。请参阅`GAN`变体。使用鉴别器训练生成器。为了训练神经网络，我们会改变网络的权重以减少其输出的错误或损失。然而，在我们的`GAN`中，生成器与我们试图影响的损失没有直接联系。生成器将数据输入到鉴别器网络中，鉴别器产生我们试图影响的输出。生成器损失会惩罚生成器，因为它生成了鉴别器网络认为是假的样本。网络的这个额外部分必须包含在反向传播中。反向传播通过计算权重对输出的影响（即如果改变权重，输出将如何变化）来将每个权重调整到正确的方向。但生成器权重的影响取决于它输入的鉴别器权重的影响。因此反向传播从输出开始，并通过鉴别器流回到生成器。同时，我们不希望判别器在生成器训练期间发生变化。试图击中移动目标会让生成器的难题变得更加困难。因此我们按照以下步骤训练生成器：
- 采样随机噪声。
- 从采样的随机噪声中产生生成器输出。
- 获取生成器输出的鉴别器“真实”或“假”分类。
- 计算鉴别器分类的损失。
- 通过鉴别器和生成器进行反向传播以获得梯度。
- 使用梯度改变生成器权重。

##### 鉴别器

`GAN`中的鉴别器只是一个分类器。它试图区分真实数据和生成器创建的数据。它可以使用任何适合其分类数据类型的网络架构。该图展示了鉴别器训练中的反向传播：
{% asset_img g_7.png %}

鉴别器的训练数据有两个来源：
- 真实数据实例，例如真实的人物图片。判别器在训练期间使用这些实例作为正例。
- 生成器创建的虚假数据实例。鉴别器在训练期间使用这些实例作为反面示例。

在上图中，两个“样本”框代表输入判别器的两个数据源。在判别器训练期间，生成器不会进行训练。生成器的权重保持不变，同时为判别器生成训练样本。鉴别器连接到两个损失函数。在鉴别器训练期间，鉴别器忽略生成器损失，仅使用鉴别器损失。我们在生成器训练期间使用生成器损失。在鉴别器训练期间：鉴别器对来自生成器的真实数据和虚假数据进行分类；鉴别器损失惩罚鉴别器将真实实例错误地分类为假实例或将假实例错误地分类为真实实例；鉴别器通过鉴别器网络的鉴别器损失反向传播来更新其权重。

##### GAN训练

由于`GAN`包含两个单独训练的网络，因此其训练算法必须解决两个复杂问题：`GAN`必须兼顾两种不同的训练（生成器和鉴别器）；`GAN`收敛性很难识别。
###### 交替训练

生成器和鉴别器的训练过程是不同的。那么我们如何训练整个`GAN`？`GAN`训练按交替周期进行：1.鉴别器训练一个或多个周期；2.生成器训练一个或多个周期；重复步骤1和2，继续训练生成器和鉴别器网络。在判别器训练阶段，我们让生成器保持不变。当判别器训练试图找出如何区分真实数据和虚假数据时，它必须学会如何识别生成器的缺陷。对于经过全面训练的生成器来说，这是一个与产生随机输出的未经训练的生成器不同的问题。类似地，我们在生成器训练阶段保持鉴别器不变。否则，生成器将试图击中**移动目标**，并且可能永远不会收敛。正是这种来回反复的过程让`GAN`能够解决原本难以解决的生成问题。我们从一个简单得多的分类问题开始，从而在困难的生成问题中获得立足点。相反，如果您无法训练分类器来区分真实数据和生成的数据，即使是初始随机生成器输出，您也无法开始`GAN`训练。
###### 收敛

随着生成器的训练不断改进，鉴别器的性能会越来越差，因为鉴别器无法轻易区分真假。如果生成器很完美，那么鉴别器的准确率只有`50%`。实际上，鉴别器通过抛硬币来做出预测。这种进展给整个`GAN`的收敛带来了问题：随着时间的推移，鉴别器反馈的意义越来越小。如果`GAN`在鉴别器给出完全随机反馈之后继续训练，那么生成器就会开始在垃圾反馈上进行训练，其自身的质量可能会下降。对于`GAN`来说，收敛往往是一种短暂的状态，而不是稳定的状态。

##### 损失函数

`GAN`试图复制概率分布。因此，它们应该使用反映`GAN`生成的数据分布与真实数据分布之间距离的损失函数。如何捕捉`GAN`损失函数中两个分布之间的差异？这个问题是一个活跃的研究领域，已经提出了许多方法。我们将在这里讨论两个常见的`GAN`损失函数，它们都已实现：
- **Minimax损失**：[`Goodfellow`等人](https://arxiv.org/abs/1406.2661)在介绍`GAN`的论文中所使用的损失函数。
- **Wasserstein损失**：`TF-GAN`估算器的默认损失函数。[`Frogner`等人,`2015`](https://arxiv.org/abs/1506.05439)中首次描述了该函数。

`GAN`可以有两个损失函数：一个用于生成器训练，一个用于鉴别器训练。两个损失函数如何协同工作以反映概率分布之间的距离度量？在我们这里要介绍的损失方案中，生成器和鉴别器损失源自概率分布之间的单一距离度量。然而，在这两种方案中，生成器只能影响距离度量中的一个项：反映虚假数据分布的项。因此，在生成器训练期间，我们会删除另一个反映真实数据分布的项。尽管生成器和鉴别器的损失源自同一个公式，但它们最终看起来是不同的。

###### Minimax损失

生成器试图最小化以下函数，而鉴别器则试图最大化它：
{% mathjax '{"conversion":{"em":14}}' %}
E_{x}[\log (D(x))]+E_{z}[\log (1-D(G(z)))]
{% endmathjax %}
在这里：
- {% mathjax %}D(x){% endmathjax %}是判别器对真实数据实例概率的估计。
- {% mathjax %}E_x{% endmathjax %}是所有真实数据实例的预期值。
- {% mathjax %}G(z){% endmathjax %}是给定噪声时生成器的输出。
- {% mathjax %}D(G(z)){% endmathjax %}是鉴别器对假实例真实的概率的估计。
- {% mathjax %}E_z{% endmathjax %}是生成器所有随机输入的预期值（实际上，是所有生成的虚假实例{% mathjax %}G(z){% endmathjax %}的预期值）。
- 该公式源自真实分布和生成分布之间的交叉熵。

生成器不能直接影响函数中的项{% mathjax %}\log(D(x)){% endmathjax %} 因此，对于生成器来说，最小化损失相当于最小化{% mathjax %}\log(1 - D(G(z))){% endmathjax %}。
###### 改进的Minimax损失

`GAN`论文指出，上述`Minimax`损失函数可能会导致`GAN`在`GAN`训练的早期阶段陷入困境，此时鉴别器的工作非常轻松。因此，该论文建议修改生成器损失，以便生成器尝试最大化{% mathjax %}\log(D(G(z))){% endmathjax %}
###### Wasserstein损失

`Wasserstein`损失是包括`TF-GAN`在内的多个库的默认损失函数。该损失函数依赖于`GAN`方案的修改（称为`“Wasserstein GAN”`或`“WGAN”`），其中鉴别器实际上并不对实例进行分类。对于每个实例，它都会输出一个数字。这个数字不必小于`1`或大于`0`，因此我们不能使用`0.5`作为阈值来判断实例是真实的还是虚假的。鉴别器训练只是试图使真实实例的输出大于虚假实例的输出。由于无法真正区分真假，`WGAN`鉴别器实际上被称为“批评者”而不是“鉴别器”。这种区别具有理论重要性，但从实际目的来看，我们可以将其视为承认损失函数的输入不是概率。损失函数本身看似简单：
{% mathjax '{"conversion":{"em":14}}' %}
D(x) - D(G(z))
{% endmathjax %}
鉴别器试图最大化这个函数。换句话说，它试图最大化其在真实实例上的输出与在虚假实例上的输出之间的差异。生成器损失：
{% mathjax '{"conversion":{"em":14}}' %}
D(G(z))
{% endmathjax %}
生成器试图最大化这个函数。换句话说，它试图最大化鉴别器对其虚假实例的输出。在这里：
- {% mathjax %}D(x){% endmathjax %}是评估者针对真实实例的输出。
- {% mathjax %}G(z){% endmathjax %}是给定噪声{% mathjax %}z{% endmathjax %}时生成器的输出。
- {% mathjax %}D(G(z)){% endmathjax %}是评估者对虚假实例的输出。
- 评估者的输出{% mathjax %}D{% endmathjax %}在{% mathjax %}[0,1]{% endmathjax %}之间。
- 该公式源自真实分布和生成分布之间的移动距离。

`Wasserstein GAN`（或`WGAN`）的理论依据要求对整个`GAN`中的权重进行修剪，以使其保持在受限范围内。`Wasserstein GAN`比基于极小极大值的`GAN`更不容易陷入困境，并避免了梯度消失的问题。移动距离还具有作为真实度量的优势：在概率分布空间中测量距离。从这个意义上讲，交叉熵不是度量。`GAN`有许多常见的故障模式。所有这些常见问题都是活跃的研究领域。虽然这些问题都还没有完全解决，但我们会提到一些人们尝试过的方法。
- 消失梯度：研究表明，如果你的鉴别器太好，那么生成器训练可能会因梯度消失而失败。实际上，最佳鉴别器无法为生成器提供足够的信息来取得进展。`Wasserstein` 损失：`Wasserstein`损失旨在防止在训练鉴别器达到最优状态时出现梯度消失。修改后的极小最大损失：`GAN`论文提出了对极小最大损失的修改，以处理梯度消失的问题。
- 模式崩溃：通常，你希望`GAN`能够产生各种各样的输出。例如，你希望人脸生成器的每个随机输入都产生一张不同的人脸。然而，如果生成器产生了一个特别合理的输出，生成器可能会学会只产生那个输出。事实上，生成器总是试图找到一个对鉴别器来说最合理的输出。如果生成器开始一遍又一遍地产生相同的输出（或一小部分输出），则鉴别器的最佳策略是学会始终拒绝该输出。但是，如果下一代鉴别器陷入局部最小值并且找不到最佳策略，那么下一次生成器迭代就很容易为当前鉴别器找到最合理的输出。生成器的每次迭代都会针对特定鉴别器进行过度优化，而鉴别器永远无法学会如何摆脱困境。因此，生成器会轮流使用一小部分输出类型。这种形式的`GAN`失败称为**模式崩溃**。以下方法尝试通过阻止生成器针对单个固定鉴别器进行优化来强制生成器扩大其范围。`Wasserstein`损失：`Wasserstein`损失可缓解模式崩溃问题，它让鉴别器训练到最优状态，而无需担心梯度消失。如果鉴别器没有陷入局部最小值，它就会学会拒绝生成器稳定的输出。因此，生成器必须尝试一些新的东西;展开式`GAN`：展开式`GAN`使用生成器损失函数，该函数不仅包含当前判别器的分类，还包含未来判别器版本的输出。因此，生成器无法针对单个判别器进行过度优化。
- 无法收敛：正如`GAN`训练部分所讨论的那样，`GAN`经常无法收敛。研究人员尝试使用各种形式的正则化来改善`GAN`收敛，包括：向鉴别器输入添加噪声；惩罚鉴别器权重。

##### GAN变体

研究人员不断寻找改进的`GAN`技术和`GAN`的新用途。
###### 渐进式GAN

在渐进式`GAN`中，生成器的第一层会生成非常低分辨率的图像，而后续层则会添加细节。这种技术使`GAN`的训练速度比同类非渐进式`GAN`更快，并能生成更高分辨率的图像。
###### 条件GAN

条件`GAN`在标记数据集上进行训练，并允许您指定每个生成的实例的标签。例如，无条件`MNIST GAN`会产生随机数字，而条件`MNIST GAN`会让您指定`GAN`应生成哪个数字。而不是对联合概率进行建模{% mathjax %}P(X,Y){% endmathjax %}，条件`GAN`模型的条件概率为{% mathjax %}P(X|Y){% endmathjax %}。
###### 图像到图像的转换

图像到图像转换`GAN`将图像作为输入，并将其映射到具有不同属性的生成输出图像。例如，我们可以获取带有汽车形状的色块的蒙版图像，然后GAN可以用逼真的汽车细节填充该形状。类似地，你可以训练图像到图像的`GAN`来拍摄手提包的草图并将其转换为手提包的逼真图像：
{% asset_img g_8.png %}

上图显示了一个`3x3`的手提包图片表。每行显示不同的手提包款式。在每行中，最左边的图像是简单的手提包线条画，中间的图像是真实手提包的照片，最右边的图像是`GAN`生成的逼真图片。三列分别标记为“输入”、“真实情况”和“输出”。在这些情况下，损失是通常的基于鉴别器的损失和逐像素损失的加权组合，对偏离源图像的生成器进行惩罚。
###### CycleGAN

`CycleGAN`学习将一组图像转换为可能属于另一组的图像。例如，当输入左侧图像时，`CycleGAN`生成了下面右侧的图像。它把一张马的图像变成了一张斑马的图像。下图显示了一匹奔跑的马，第二张图像除了马是斑马外，其他方面都完全相同：
{% asset_img g_9.png %}

`CycleGAN`的训练数据只是两组图像（在本例中，一组马图像和一组斑马图像）。该系统不需要图像之间的标签或成对对应关系。有关更多信息，请参阅[`Zhu et al, 2017`](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)，其中说明了如何使用`CycleGAN`在没有配对数据的情况下执行图像到图像的转换。
###### 文本到图像合成

文本转图像`GAN`将文本作为输入，并生成可信且由文本描述的图像。例如，下面的花朵图像是通过将文本描述输入到`GAN`生成的。
{% asset_img g_10.png %}

请注意，在这个系统中，`GAN`只能从一小部分类别中生成图像。
###### 超分辨率

超分辨率`GAN`可提高图像的分辨率，在需要的地方添加细节以填充模糊区域。例如，下面中间的模糊图像是左侧原始图像的下采样版本。给定模糊图像，`GAN`会生成右侧更清晰的图像：
{% asset_img g_11.png %}

请注意，原始图像显示了一位戴着精致头饰的女孩的画作。头饰的头带以复杂的图案编织而成。给定该画作的模糊版本，在`GAN`的输出处会获得清晰的版本。`GAN`生成的图像看起来与原始图像非常相似，但她头饰和衣服上的图案的一些细节略有不同 - 例如，如果你仔细观察头带，你会发现`GAN`并没有重现原始图像中的星爆图案。相反，它自己制作了可信的图案来取代被下采样抹去的图案。有关更多信息，请参阅[`Ledig`等人，`2017`年](https://arxiv.org/pdf/1609.04802.pdf)。

###### 脸部修复

`GAN`已用于语义图像修复任务。在修复任务中，图像的块被涂黑，系统会尝试填充缺失的块。[`Yeh`等人,`2017`年](https://aman.ai/primers/ai/gans/)使用`GAN`修复人脸图像，效果优于其他技术。下面显示的是一组图像，其中每张图像都是一张人脸照片，但部分区域被替换为黑色。每张图像都是一张人脸照片，与“输入”列中的一张图像完全相同，只是没有黑色区域。
{% asset_img g_12.png %}

###### 文字转语音

并非所有`GAN`都能生成图像。例如，研究人员还使用`GAN`从文本输入生成合成语音。有关更多信息，请参阅[`Yang et al, 2017`](https://arxiv.org/pdf/1607.07539.pdf)。