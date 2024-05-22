---
title: 注意力机制 (Transformer)(TensorFlow)
date: 2024-05-21 17:12:11
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

灵长类动物的视觉系统接受了大量的感官输入，这些感官输入远远超过了大脑能够完全处理的程度。然而，并非所有刺激的影响都是相等的。意识的聚集和专注使灵长类动物能够在复杂的视觉环境中将注意力引向感兴趣的物体，例如猎物和天敌。只关注一小部分信息的能力对进化更加有意义，使人类得以生存和成功。自`19`世纪以来，科学家们一直致力于研究认知神经科学领域的注意力。基于注意力机制的`Transformer`架构，该架构中使用了**多头注意力**(`multi-head attention`)和**自注意力**(`self-attention`)。自`2017`年横空出世，`Transformer`一直都普遍存在于现代的深度学习应用中，例如语言、视觉、语音和强化学习领域。
<!-- more -->
#### 注意力提示

自经济学研究稀缺资源分配以来，人们正处在“注意力经济”时代，即人类的注意力被视为可以交换的、有限的、有价值的且稀缺的商品。许多商业模式也被开发出来去利用这一点：在音乐或视频流媒体服务上，人们要么消耗注意力在广告上，要么付钱来隐藏广告；为了在网络游戏世界的成长，人们要么消耗注意力在游戏战斗中，从而帮助吸引新的玩家，要么付钱立即变得强大。总之，注意力不是免费的。注意力是稀缺的，而环境中的干扰注意力的信息却并不少。比如人类的视觉神经系统大约每秒收到{% mathjax %}10^8{% endmathjax %}位的信息，这远远超过了大脑能够完全处理的水平。幸运的是，人类的祖先已经从经验（也称为数据）中认识到 “并非感官的所有输入都是一样的”。在整个人类历史中，这种只将注意力引向感兴趣的一小部分信息的能力，使人类的大脑能够更明智地分配资源来生存、成长和社交，例如发现天敌、找寻食物和伴侣。

##### 生物学中的注意力提示
注意力是如何应用于视觉世界中的呢？这要从当今十分普及的双组件(`two-component`)的框架开始讲起：这个框架的出现可以追溯到`19`世纪`90`年代的威廉·詹姆斯，他被认为是“美国心理学之父”。在这个框架中，受试者基于非自主性提示和自主性提示 有选择地引导注意力的焦点。非自主性提示是基于环境中物体的突出性和易见性。想象一下，假如我们面前有五个物品：一份报纸、一篇研究论文、一杯咖啡、一本笔记本和一本书，就像下图。所有纸制品都是黑白印刷的，但咖啡杯是红色的。换句话说，这个咖啡杯在这种视觉环境中是突出和显眼的，不由自主地引起人们的注意。所以我们会把视力最敏锐的地方放到咖啡上，如下图所示。
{% asset_img at_1.png "由于突出性的非自主性提示（红杯子），注意力不自主地指向了咖啡杯" %}

喝咖啡后，我们会变得兴奋并想读书，所以转过头，重新聚焦眼睛，然后看看书，就像下图中描述那样。与下图中由于突出性导致的选择不同，此时选择书是受到了认知和意识的控制，因此注意力在基于自主性提示去辅助选择时将更为谨慎。受试者的主观意愿推动，选择的力量也就更强大。
{% asset_img at_2.png "依赖于任务的意志提示（想读一本书），注意力被自主引导到书上" %}

##### 查询、键和值

自主性的与非自主性的注意力提示解释了人类的注意力的方式，下面来看看如何通过这两种注意力提示，用神经网络来设计注意力机制的框架，首先，考虑一个相对简单的状况，即只使用非自主性提示。 要想将选择偏向于感官输入，则可以简单地使用参数化的全连接层，甚至是非参数化的最大汇聚层或平均汇聚层。

因此，“是否包含自主性提示”将注意力机制与全连接层或汇聚层区别开来。在注意力机制的背景下，自主性提示被称为查询(`query`)。给定任何查询，注意力机制通过注意力汇聚(`attention pooling`)将选择引导至感官输入（`sensory inputs`，例如中间特征表示）。在注意力机制中，这些感官输入被称为值(`value`)。更通俗的解释，每个值都与一个键(`key`)配对，这可以想象为感官输入的非自主提示。如下图所示，可以通过设计注意力汇聚的方式，便于给定的查询（自主性提示）与键（非自主性提示）进行匹配，这将引导得出最匹配的值（感官输入）。
{% asset_img at_3.png "注意力机制通过注意力汇聚将查询（自主性提示）和键（非自主性提示）结合在一起，实现对值（感官输入）的选择倾向" %}

注意力机制的设计有许多替代方案。例如可以设计一个不可微的注意力模型，该模型可以使用强化学习方法进行训练。
##### 总结

人类的注意力是有限的、有价值和稀缺的资源。受试者使用非自主性和自主性提示有选择性地引导注意力。前者基于突出性，后者则依赖于意识。注意力机制与全连接层或者汇聚层的区别源于增加的自主提示。由于包含了自主性提示，注意力机制与全连接的层或汇聚层不同。注意力机制通过注意力汇聚使选择偏向于值（感官输入），其中包含查询（自主性提示）和键（非自主性提示）。键和值是成对的。可视化查询和键之间的注意力权重是可行的。

#### 注意力汇聚

查询（自主提示）和键（非自主提示）之间的交互形成了注意力汇聚；注意力汇聚有选择地聚合了值（感官输入）以生成最终的输出。接下来介绍注意力汇聚的更多细节，以便从宏观上了解注意力机制在实践中的运作方式。具体来说，`1964`年提出的`Nadaraya-Watson`核回归模型是一个简单但完整的例子，可以用于演示具有注意力机制的机器学习。
##### 生生数据集

简单起见，考虑下面这个回归问题：给定成对的输入-输出数据集{% mathjax %}\{(x_1,y_1),\dots,(x_n,y_n)\}{% endmathjax %}，如何学习{% mathjax %}f{% endmathjax %}来预测任意新输入{% mathjax %}x{% endmathjax %}的输出{% mathjax %}\hat{y} = f(x){% endmathjax %}？根据下面的非线性函数生成一个人工数据集，其中加入的噪声项为{% mathjax %}\epsilon{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon
{% endmathjax %}
其中{% mathjax %}\epsilon{% endmathjax %}服从均值为`0`和标准差为`0.5`的正态分布。在这里生成了`50`个训练样本和50个测试样本。为了更好地可视化之后的注意力模式，需要将训练样本进行排序。
```python
import tensorflow as tf
import matplotlib.pyplot as plt

tf.random.set_seed(seed=1322)

def plot_kernel_reg(y_hat):
    plt.plot(x_test, [y_truth, y_hat], 'x', 'y', legend=['Truth', 'Pred'],xlim=[0, 5], ylim=[-1, 5])
    plt.plot(x_train, y_train, 'o', alpha=0.5)
    plt.show()

def f(x):
    return 2 * tf.sin(x) + x**0.8

n_train = 50
x_train = tf.sort(tf.random.uniform(shape=(n_train,), maxval=5))
y_train = f(x_train) + tf.random.normal((n_train,), 0.0, 0.5)  # 训练样本的输出
x_test = tf.range(0, 5, 0.1)  # 测试样本
y_truth = f(x_test)  # 测试样本的真实输出
n_test = len(x_test)  # 测试样本数
n_test

# 50
```
上面的函数`plot_kernel_reg`将绘制所有的训练样本（样本由圆圈表示），不带噪声项的真实数据生成函数{% mathjax %}f{% endmathjax %}标记为`“Truth”`），以及学习得到的预测函数（标记为`“Pred”`）。
##### 平均汇聚

先使用最简单的估计器来解决回归问题。基于**平均汇聚**来计算所有训练样本输出值的平均值：
{% mathjax '{"conversion":{"em":14}}' %}
f(x) = \frac{1}{n}\sum_{i=1}^n y_i
{% endmathjax %}
如下图所示，这个估计器确实不够聪明。真实函数{% mathjax %}f{% endmathjax %}(`“Truth”`)和预测函数(`“Pred”`)相差很大。
```python
y_hat = tf.repeat(tf.reduce_mean(y_train), repeats=n_test)
plot_kernel_reg(y_hat)
```
{% asset_img at_5.png %}

##### 非参数注意力汇聚

显然，**平均汇聚**忽略了输入{% mathjax %}x_i{% endmathjax %}。于是Nadaraya和Watson提出了一个更好的想法，根据输入的位置对输出{% mathjax %}y_i{% endmathjax %}进行加权：
{% mathjax '{"conversion":{"em":14}}' %}
f(x) = \sum_{i=1}^n\frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)}y_i
{% endmathjax %}
其中{% mathjax %}K{% endmathjax %}是核(`kernel`)。以上公式所描述的估计器被称为`Nadaraya-Watson`核回归(`Nadaraya-Watson kernel regression`)。
{% mathjax '{"conversion":{"em":14}}' %}
f(x) = \sum_{i=1}^n \alpha(x,x_i)y_i
{% endmathjax %}
其中{% mathjax %}x{% endmathjax %}是查询，{% mathjax %}(x_i,y_i){% endmathjax %}是键值对，注意力汇聚是{% mathjax %}y_i{% endmathjax %}的加权平均。将查询{% mathjax %}x{% endmathjax %}和{% mathjax %}x_i{% endmathjax %}之间的关系建模为**注意力权重**(`attention weight`){% mathjax %}\alpha(x,x_i){% endmathjax %}，这个权重将被分配给每一个对应值{% mathjax %}y_i{% endmathjax %}。对于任何查询，模型在所有键值对注意力权重都是一个有效的概率分布：它们是非负的，并且总和为`1`。为了更好地理解注意力汇聚，下面考虑一个**高斯核**(`Gaussian kernel`)，其定义为：
{% mathjax '{"conversion":{"em":14}}' %}
K(u) = \frac{1}{\sqrt{2\pi}}\exp(-\frac{u^2}{2})
{% endmathjax %}
将高斯核代入可以得到：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
f(x) & = \sum_{i=1}^n \alpha(x,x_i)y_i \\ 
& = \sum_{i=1}^n\frac{\exp(\frac{1}{2}(x-x_i)^2)}{\sum_{j=1}^n\exp(-\frac{1}{2}(x-x_i)^2)} y_i \\
& = \sum_{i=1}^n \text{softmax}(-\frac{1}{2}(x-x_i)^2) y_i \\
\end{align}
{% endmathjax %}
在以上公式中，如果一个键{% mathjax %}x_i{% endmathjax %}越是接近给定的查询{% mathjax %}x{% endmathjax %}，那么分配给这个键对应值{% mathjax %}y_i{% endmathjax %}的注意力权重就会越大，也就获得了更多的注意力。值得注意的是，`Nadaraya-Watson`核回归是一个非参数模型。也就意味着，它是**非参数的注意力汇聚**(`nonparametric attention pooling`)模型。接下来，我们将基于这个非参数的注意力汇聚模型来绘制预测结果。从绘制的结果会发现新的模型预测线是平滑的，并且比平均汇聚的预测更接近真实。
```python
# X_repeat的形状:(n_test,n_train),
# 每一行都包含着相同的测试输入（例如：同样的查询）
X_repeat = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# x_train包含着键。attention_weights的形状：(n_test,n_train),
# 每一行都包含着要在给定的每个查询的值（y_train）之间分配的注意力权重
attention_weights = tf.nn.softmax(-(X_repeat - tf.expand_dims(x_train, axis=1))**2/2, axis=1)
# y_hat的每个元素都是值的加权平均值，其中的权重是注意力权重
y_hat = tf.matmul(attention_weights, tf.expand_dims(y_train, axis=1))

plot_kernel_reg(y_hat)
```
{% asset_img at_6.png %}

现在来观察注意力的权重。这里测试数据的输入相当于查询，而训练数据的输入相当于键。因为两个输入都是经过排序的，因此由观察可知“查询-键”对越接近，注意力汇聚的注意力权重就越高。
##### 带参数注意力汇聚

非参数的`Nadaraya-Watson`核回归具有一致性(`consistency`)的优点：如果有足够的数据，此模型会收敛到最优结果。尽管如此，我们还是可以轻松地将可学习的参数集成到注意力汇聚中。例如，与非参数的注意力汇聚不同，在下面的查询{% mathjax %}x{% endmathjax %}和键{% mathjax %}x_i{% endmathjax %}之间的距离乘以可学习参数{% mathjax %}w{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
f(x) & = \sum_{i=1}^n \alpha(x,x_i)y_i \\ 
& = \sum_{i=1}^n\frac{\exp(\frac{1}{2}((x-x_i)w)^2)}{\sum_{j=1}^n\exp(-\frac{1}{2}((x-x_i)w)^2)} y_i \\
& = \sum_{i=1}^n \text{softmax}(-\frac{1}{2}((x-x_i)w)^2) y_i \\
\end{align}
{% endmathjax %}
###### 批量矩阵乘法
为了更有效地计算小批量数据的注意力，我们可以利用深度学习开发框架中提供的批量矩阵乘法。假设第一个小批量数据包含{% mathjax %}n{% endmathjax %}个矩阵{% mathjax %}\mathbf{X}_1,\ldots,\mathbf{X}_n{% endmathjax %}，形状为{% mathjax %}a\times b{% endmathjax %}，第二个小批量包含{% mathjax %}n{% endmathjax %}个矩阵{% mathjax %}\mathbf{Y}_1,\ldots,\mathbf{Y}_n{% endmathjax %}，形状为{% mathjax %}b\times c{% endmathjax %}，它们的批量矩阵乘法得到{% mathjax %}n{% endmathjax %}个矩阵{% mathjax %}\mathbf{X}_1\mathbf{Y}_1,\ldots,\mathbf{X}_n\mathbf{Y}_n{% endmathjax %}，形状为{% mathjax %}a\times c{% endmathjax %}。因此，假定两个张量的形状分别是{% mathjax %}(n,a,b){% endmathjax %}和{% mathjax %}(n,b,c){% endmathjax %}，它们的批量矩阵乘法输出的形状为{% mathjax %}(n,a,c){% endmathjax %}。
```python
X = tf.ones((2, 1, 4))
Y = tf.ones((2, 4, 6))
tf.matmul(X, Y).shape

# TensorShape([2, 1, 6])

# 在注意力机制的背景中，我们可以使用小批量矩阵乘法来计算小批量数据中的加权平均值。
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()

# array([[[ 4.5]],[[14.5]]], dtype=float32)
```
###### 定义模型

基于带参数的注意力汇聚，使用小批量矩阵乘法，定义`Nadaraya-Watson`核回归的带参数版本为：
```python
class NWKernelRegression(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(initial_value=tf.random.uniform(shape=(1,)))

    def call(self, queries, keys, values, **kwargs):
        # 对于训练，“查询”是x_train。“键”是每个点的训练数据的距离。“值”为'y_train'。
        # queries和attention_weights的形状为(查询个数，“键－值”对个数)
        queries = tf.repeat(tf.expand_dims(queries, axis=1), repeats=keys.shape[1], axis=1)
        self.attention_weights = tf.nn.softmax(-((queries - keys) * self.w)**2 /2, axis =1)
        # values的形状为(查询个数，“键－值”对个数)
        return tf.squeeze(tf.matmul(tf.expand_dims(self.attention_weights, axis=1), tf.expand_dims(values, axis=-1)))
```
###### 训练

接下来，将训练数据集变换为键和值用于训练注意力模型。在带参数的注意力汇聚模型中，任何一个训练样本的输入都会和除自己以外的所有训练样本的“键－值”对进行计算，从而得到其对应的预测输出。
```python
# X_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输入
X_tile = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_train, axis=0)
# Y_tile的形状:(n_train，n_train)，每一行都包含着相同的训练输出
Y_tile = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_train, axis=0)
# keys的形状:('n_train'，'n_train'-1)
keys = tf.reshape(X_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))
# values的形状:('n_train'，'n_train'-1)
values = tf.reshape(Y_tile[tf.cast(1 - tf.eye(n_train), dtype=tf.bool)], shape=(n_train, -1))

# 训练带参数的注意力汇聚模型时，使用平方损失函数和随机梯度下降。
net = NWKernelRegression()
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.5)
animator = plt.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])


for epoch in range(5):
    with tf.GradientTape() as t:
        loss = loss_object(y_train, net(x_train, keys, values)) * len(y_train)
    grads = t.gradient(loss, net.trainable_variables)
    optimizer.apply_gradients(zip(grads, net.trainable_variables))
    print(f'epoch {epoch + 1}, loss {float(loss):.6f}')
    animator.add(epoch + 1, float(loss))
```
{% asset_img at_7.png %}

如下所示，训练完带参数的注意力汇聚模型后可以发现：在尝试拟合带噪声的训练数据时，预测结果绘制的线不如之前非参数模型的平滑。
```python
# keys的形状:(n_test，n_train)，每一行包含着相同的训练输入（例如，相同的键）
keys = tf.repeat(tf.expand_dims(x_train, axis=0), repeats=n_test, axis=0)
# value的形状:(n_test，n_train)
values = tf.repeat(tf.expand_dims(y_train, axis=0), repeats=n_test, axis=0)
y_hat = net(x_test, keys, values)
plot_kernel_reg(y_hat)
```
{% asset_img at_8.png %}

与非参数的注意力汇聚模型相比，带参数的模型加入可学习的参数后，曲线在注意力权重较大的区域变得更不平滑。
##### 总结

`Nadaraya-Watson`核回归是具有注意力机制的机器学习范例。`Nadaraya-Watson`核回归的注意力汇聚是对训练数据中输出的加权平均。从注意力的角度来看，分配给每个值的注意力权重取决于将值所对应的键和查询作为输入的函数。**注意力汇聚可以分为非参数型和带参数型**。

#### 注意力评分函数

高斯核指数部分可以视为**注意力评分函数**(`attention scoring function`)，简称评分函数(`scoring function`)，然后把这个函数的输出结果输入到`softmax`函数中进行运算。通过上述步骤，将得到与键对应的值的概率分布（即注意力权重）。最后，注意力汇聚的输出就是基于这些注意力权重的值的加权和。说明了如何将注意力汇聚的输出计算成为值的加权和，其中{% mathjax %}a{% endmathjax %}表示注意力评分函数。由于注意力权重是概率分布，因此加权和其本质上是加权平均值。
{% asset_img at_4.png "计算注意力汇聚的输出为值的加权和" %}

