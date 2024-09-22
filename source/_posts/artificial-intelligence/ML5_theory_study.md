---
title: 机器学习(ML)(五) — 探析
date: 2024-09-18 12:24:11
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

#### 介绍

构建**逻辑回归模型**的第一步是指定如何根据输入特征{% mathjax %}x{% endmathjax %}和参数{% mathjax %}w,b{% endmathjax %}计算输出。**逻辑回归函数**预测{% mathjax %}f_{\vec{W},b}(\vec{x}) = g\;,\;g = \vec{W}\cdot\vec{x} + b{% endmathjax %}。那么{% mathjax %}g(z) = \frac{1}{1+ e^{-z}}{% endmathjax %}，代码实现为：`z = np.dot(w , x) + b, f_x = 1/(1 + np.exp(-z))`。
<!-- more -->

第一步是指定**逻辑回归**的输入到输出函数是什么？这取决于输入{% mathjax %}\vec{x}{% endmathjax %}和模型的参数。训练识字回归模型的第二步是指定**损失函数**和**成本函数**，损失函数表示为{% mathjax %}L(f_{\vec{W},b}(\vec{x}),y){% endmathjax %}，标签和训练集为{% mathjax %}y{% endmathjax %}，则该单个训练示例的损失为：`loss = -y * np.log(f_x) - (1-y) * np.log(1 - f_x)`。这是衡量**逻辑回归**在单个训练示例{% mathjax %}(x,y){% endmathjax %}上表现如何的**指标**。给定**损失函数**的定义，然后定义**成本函数**，**成本函数**是参数{% mathjax %}W,b{% endmathjax %}的函数，这只是对{% mathjax %}m{% endmathjax %}个训练示例{% mathjax %}\mathbf{J}(\vec{W},b) = \frac{1}{m}\sum_{i=1}^m L(f_{\vec{W},b}(\vec{x}^{(i)}),y^{(i)}){% endmathjax %}，我们使用的**损失函数**是学习算法输出和基本事实标签的函数，它是在单个训练示例中计算的，而**成本函数**{% mathjax %}\mathbf{J}{% endmathjax %}是在整个训练集上计算的**损失函数**的平均值。这是构建**逻辑回归**的第二步。然后，**训练逻辑回归模型**的第三步也是使用一种算法，**梯度下降法**，最小化成本函数{% mathjax %}\mathbf{J}(\mathbf{W},b){% endmathjax %}。使用**梯度下降法**最小化成本{% mathjax %}\mathbf{J}{% endmathjax %}作为参数的函数，代码实现为：`w = w - alpha * dj_dw, b = b - alpha * dj_db`。第一步，指定如何根据输入{% mathjax %}\vec{x}{% endmathjax %}和参数计算输出，第二步指定**损失**和**成本**，第三步最小化**训练逻辑回归**的**成本函数**。同样的三个步骤也是在`TensorFlow`中训练神经网络的方法。现在让我们看看这三个步骤如何映射到训练神经网络。第一步是指定如何根据输入{% mathjax %}\vec{x}{% endmathjax %}和参数{% mathjax %}W,b{% endmathjax %}计算输出，这可以通过此代码片段完成。
```python
model = Squential([Dense(...),Dense(...),Dense(...)])
```

第二步是编译模型并告诉它要使用什么损失，这是用于指定此损失函数的代码：`model =.compile(loss = BinaryCrossentropy())`，即**二元交叉熵损失函数**，一旦指定此损失，对整个训练集取平均值，然后第三步是调用函数将成本最小化为神经网络参数的函数：`model.fit(x,y,epochs = 100)`。接下来更详细地了解这三个步骤。第一步，指定如何根据输入{% mathjax %}x{% endmathjax %}和参数{% mathjax %}w,b{% endmathjax %}计算输出。此代码片段指定了神经网络的整个架构。第一个隐藏层有`25`个隐藏单元，下一个隐藏层有`15`个隐藏单元，然后是一个输出单元，使用`S`型激活值。
```python
import tensorflow as tf
from tf.keras import Sequential
from tf.keras.layouts import Dense

model = Squential([
    Dense(units = 25,activation='sigmoid'),
    Dense(units = 15,activation='sigmoid'),
    Dense(units = 1,activation='sigmoid')])
```
基于此代码片段，我们还知道第一层的参数{% mathjax %}W^{[1]},\vec{b}^{[1]}{% endmathjax %}第二层的参数{% mathjax %}W^{[2]},\vec{b}^{[2]}{% endmathjax %}和第三层的参数{% mathjax %}W^{[3]},\vec{b}^{[3]}{% endmathjax %}。此代码片段指定了神经网络的整个架构。为了计算{% mathjax %}\vec{x}{% endmathjax %}的输出{% mathjax %}\vec{a}^{[3]}{% endmathjax %}，这里我们写了{% mathjax %}W^{[l]},\vec{b}^{[l]}{% endmathjax %}。在第二步中，必须指定**损失函数**是什么？这也将定义**成本函数**。对于手写数字分类问题，其中图像要么是`0`，要么是`1`，到目前为止最常见的**损失函数**是{% mathjax %}L(f(\vec{x}),y) = -y\log(f(\vec{x})) - (1 - y)\log(1 - f(\vec{x})){% endmathjax %}，{% mathjax %}y{% endmathjax %}称为**目标标签**，而{% mathjax %}f(\vec{x}){% endmathjax %}是神经网络的输出。在`TensorFlow`中，这称为**二元交叉熵损失函数**。这个名字从何而来？在统计学中，这个函数被称为**交叉熵损失函数**，而二元这个词只是再次强调这是一个**二元分类**问题，因为每个图像要么是`0`，要么是`1`。`TensorFlow`使用此**损失函数**，`keras`最初是一个独立于`TensorFlow`开发库，实际上是与`TensorFlow`完全独立的项目。但最终它被合并到`TensorFlow`中，这就是为什么有`tf.Keras`库。指定了单个训练示例的损失后，`TensorFlow`知道您想要最小化的成本是**平均值**，即对所有训练示例的损失取所有`m`个训练示例的平均值。如果想解决**回归问题**而不是**分类问题**。可以告诉`TensorFlow`使用其它的损失函数编译模型。例如，如果有一个**回归问题**，并且如果想**最小化平方误差损失**。如果学习算法输出{% mathjax %}f(\vec{x}){% endmathjax %}，目标为{% mathjax %}y{% endmathjax %}，则损失是平方误差的`1/2`。然后在`TensorFlow`中使用此损失函数。在这个表达式中，成本函数表示为表示为：{% mathjax %}\mathbf{J}(\mathbf{W,B}) = \frac{1}{m}\sum_{i=1}^m L(f(\vec{x}^{(1)}),y^{(i)}){% endmathjax %}。**成本函数**是神经网络中所有参数的函数。将大写{% mathjax %}\mathbf{W}{% endmathjax %}视为包括{% mathjax %}W^{[1]},W^{[2]},W^{[3]}{% endmathjax %}和{% mathjax %}b^{[1]},b^{[2]},b^{[3]}{% endmathjax %}。将{% mathjax %}f(\vec{x}){% endmathjax %}作为**神经网络**的输出，但如果想强调神经网络的输出作为{% mathjax %}\vec{x}{% endmathjax %}的函数取决于神经网络所有层中的所有参数，可以写为{% mathjax %}f_{\vec{W},b}(\vec{x}){% endmathjax %}。这就是**损失函数**和**成本函数**。最后，将用`TensorFlow`**最小化交叉函数**。使用**梯度下降法**训练神经网络的参数，那么对于每个层{% mathjax %}l{% endmathjax %}和每个单元{% mathjax %}j{% endmathjax %}，需要根据{% mathjax %}w_j^{[l]} = w_j^{[l]} - \alpha\frac{\partial}{\partial w_j}\mathbf{J}(\vec{W},b)\;,\;b_j^{[l]} = b_j^{[l]} - \alpha\frac{\partial}{\partial b_j}\mathbf{J}(\vec{W},b){% endmathjax %}。进行`100`次**梯度下降**迭代之后，希望能得到一个好的参数值。为了使用**梯度下降法**，需要计算的关键是这些**偏导数项**。神经网络训练的标准做法，是使用一种称为**反向传播**的算法来计算这些**偏导数项**。`TensorFlow`可以完成所有这些工作。称为`fit`的函数中实现了**反向传播**。您所要做的就是调用`model.fit(X、y, epochs=100)`作为训练集，并告诉它进行`100`次迭代。
{% asset_img ml_1.png %}

#### 激活函数

到目前为止，在**隐藏层**和**输出层**的所有节点中都使用了`S`型**激活函数**。之所以这样做，是因为**逻辑回归**构建**神经网络**，并创建了大量**逻辑回归单元**并将它们串联在一起。如果使用**激活函数**，**神经网络**会变得更加强大。例如给定价格、运费、营销、材料，尝试预测某件商品是否非常实惠。如果知名度高且感知质量高，则尝试预测它是畅销品。但知名度可能是二元的，即人们知道或不知道。但似乎潜在买家对你所销售的T恤的了解程度可能不是二元的，他们可能有点了解，了解，非常了解，或者它可能已经完全流行起来。因此，不应将意识建模为二进制数`0、1`，而应估计意识的概率。也许意识应该是任何非负数，因为意识可以是任何非负值，从`0`到非常大的数字。因此，之前曾使用这个方程来计算第二个隐藏单元的激活（估计意识）{% mathjax %}z = \vec{W}_2^{[1]}\cdot\vec{x} + b_2^{[1]}\;,\;a_2^{[1]} = g(z) = \frac{1}{1+e^{-z}}\;,\;0 < g(z) < 1{% endmathjax %}。
{% asset_img ml_2.png %}

如果想让{% mathjax %}a_2^{[1]}{% endmathjax %}取更大的正值，可以换成其它**激活函数**。它看起来像这样，如下图所示。如果{% mathjax %}z < 0{% endmathjax %}，则{% mathjax %}g(z) = 0{% endmathjax %}，是左半部分直线。如果{% mathjax %}z \geq 0{% endmathjax %}，则{% mathjax %}g(z) = z{% endmathjax %}，是有半部分`45°`的直线。
{% asset_img ml_3.png %}

数学方程为{% mathjax %}g(z) = \max(0,z){% endmathjax %}。此激活函数有一个名称为`ReLU`，**整流线性单元**。还有一个值得一提的**激活函数**，称为**线性激活函数**，也就是{% mathjax %}g(z) = z{% endmathjax %}。如果{% mathjax %}\vec{a} = g(z)\;,\; g(z) = z{% endmathjax %}，那么{% mathjax %}\vec{a} = z = \vec{w}\cdot\vec{x} + b{% endmathjax %}。好像里面根本没有使用{% mathjax %} g{% endmathjax %}一样。所以没有使用任何**激活函数**。

如何为**神经网络**中的**神经元**选择不同的**激活函数**。首先介绍如何为**输出层**选择**激活函数**，然后介绍**隐藏层激活函数**的选择。当考虑输出层的**激活函数**时，通常会有一个很自然的选择，具体取决于**目标标签**{% mathjax %}y{% endmathjax %}是什么。如果您正在处理分类问题，其中{% mathjax %}y{% endmathjax %}为`0`或`1`，即**二元分类问题**，`S`型激活函数是最自然的选择，因为神经网络会学习预测{% mathjax %}y=1{% endmathjax %}的概率，就像对**逻辑回归**所做的那样。如果正在处理**二元分类问题**，在输出层使用`S`型函数。如果正在解决**回归问题**，那么可以选择其它的**激活函数**。例如，如果预测明天的股价与今天的股价相比会如何变化。它可以上涨也可以下跌，在这种情况下，建议使用**线性激活函数**。使用**线性激活函数**，{% mathjax %}g(z){% endmathjax %}可以取正值或负值。所以{% mathjax %}\hat{y}{% endmathjax %}可以是正数也可以是负数。最后，如果 {% mathjax %}\hat{y}{% endmathjax %}只能取非负值，例如，如果预测房价，它永远不会为负数，那么最自然的选择就是`ReLU`激活函数，因为这个激活函数只取非负值，要么是零，要么是正值。在选择用于**输出层**的激活函数时，通常取决于预测的标签{% mathjax %}\hat{y}{% endmathjax %}是什么，会有一个很自然的选择。
{% asset_img ml_4.png %}

神经网络的**隐藏层**呢？`ReLU`激活函数是许多训练神经网络的最常见选择。在神经网络发展的早期历史中，人们在许多地方使用`S`型激活函数，但该领域已经发展到更频繁地使用`ReLU`。唯一的例外是，如果有一个**二元分类问题**，那么在输出层使用`S`型激活函数。那么为什么呢？首先，如果比较`ReLU`和`S`型激活函数，`ReLU`的计算速度要快一点，因为它只需要计算{% mathjax %}g(z) = \max(0,z){% endmathjax %}，而`S`型激活函数需要先求幂，然后求逆等等，所以效率稍低一些。但第二个原因更为重要，那就是`ReLU`激活函数只在图的一部分中平坦；这里左边是完全平坦的，而`S`型激活函数在两边都平坦。如果你使用**梯度下降**来训**练神经网络**，那么当在很多地方都很胖的函数时，**梯度下降**会非常慢。**梯度下降**优化的是**成本函数**{% mathjax %}\mathbf{J}(\vec{W},b){% endmathjax %}，而不是**激活函数**，但**激活函数**是计算的一部分，这会导致**成本函数**{% mathjax %}\mathbf{J}(\vec{W},b){% endmathjax %}中更多地方也是平坦的，梯度很小，这会减慢学习速度。使用`ReLU`激活函数可以让神经网络学习得更快一些，在**隐藏层**中使用`ReLU`激活函数已经成为最常见的选择。
{% asset_img ml_5.png %}

总而言之，对于**输出层**，使用`S`型函数，如果你有一个**二元分类问题**；如果{% mathjax %}\hat{y}{% endmathjax %}是一个可以取正值或负值的数字，则使用**线性激活函数**；如果{% mathjax %}\hat{y}{% endmathjax %}只能取正值，则使用`ReLU`激活函数。对于**隐藏层**，建议`ReLU`作为默认激活函数，在`TensorFlow`中，您可以这样实现它。
```python
import tensorflow as tf
from tf.keras import Sequential
from tf.keras.layouts import Dense

model = Squential([
    Dense(units = 25,activation='relu'),     # layer 1
    Dense(units = 15,activation='relu'),     # layer 2
    Dense(units = 1, activation='sigmoid')]) # layer 3 , activation is sigmoid (binary classification)、relu (y >= 0) or linear (y is negative/positive).
```
顺便说一句，如果你看看研究文献，有时会看到作者使用其他的**激活函数**，例如`tan h`激活函数、`LeakyReLU`激活函数或`swish`激活函数。

为什么**神经网络**需要**激活函数**？如果对**神经网络**中的所有节点都使用**线性激活函数**，会发生什么？这个神经网络将变得与**线性回归**没有什么不同。看一个神经网络的例子，其中输入{% mathjax %}\vec{x}{% endmathjax %}是一个数值，有一个带有参数{% mathjax %}W_1^{[1]},b_1^{[1]}{% endmathjax %}的**隐藏单元**，输出{% mathjax %}a^{[1]}{% endmathjax %}，然后第二层是**输出层**，只有一个带有参数{% mathjax %}W_1^{[2]},b_1^{[2]}{% endmathjax %}的**输出单元**，然后输出{% mathjax %}a^{[2]}{% endmathjax %}，只是一个标量，它是神经网络{% mathjax %}f(\vec{x}) = \vec{w}\cdot\vec{x} + \vec{b}{% endmathjax %}的输出。使用线性激活函数{% mathjax %}g(z) = z{% endmathjax %}会怎样。计算{% mathjax %}a^{[1]} = w_1^{[1]}x + b_1^{[1]}\;,\;a^{[2]} = w_1^{[2]}a^{[1]} + b_1^{[2]}{% endmathjax %}。取{% mathjax %}a^{[1]}{% endmathjax %}表达式并将其代入其中。因此它变成{% mathjax %}a^{[2]} = w_1^{[2]}(w_1^{[1]}x + b_1^{[1]}) + b_1^{[2]}{% endmathjax %}。如果简化，它变成 {% mathjax %}a^{[2]} = (w_1^{[1]}w_1^{[2]})x + w_1^{[2]}b_1^{[1]} + b_1^{[2]}{% endmathjax %}。事实证明，如果我将{% mathjax %}w_1^{[1]},w_1^{[2]}{% endmathjax %}设置为{% mathjax %}w{% endmathjax %}，并将{% mathjax %}w_1^{[2]}b_1^{[1]} + b_1^{[2]}{% endmathjax %}设置为{% mathjax %}b{% endmathjax %}，那么{% mathjax %}a^{[2]} = wx + b{% endmathjax %}。所以{% mathjax %}a^{[2]}{% endmathjax %}只是输入{% mathjax %}x{% endmathjax %}的**线性函数**。与其使用具有一个**隐藏层**和一个**输出层**的**神经网络**，还不如使用**线性回归模型**。如果有一个像这样的多层神经网络，假设所有**隐藏层**使用**线性激活函数**，对输出层也使用**线性激活函数**，那么这个模型将计算出一个完全等同于**线性回归**的输出。如果对所有**隐藏层**使用**线性激活函数**，对**输出层**使用**逻辑激活函数**，那么可以证明这个模型等同于**逻辑回归**，因此，不要在**神经网络**的**隐藏层**中使用**线性激活函数**。使用`ReLU`激活函数就可以了。
{% asset_img ml_6.png %}

#### 多元分类（softmax）

**多元分类**是指分类问题，其中的输出标签不止两个。对于手写数字分类问题，我们只是试图区分手写数字`0`和`1`。但是，如果阅读信封中的邮政编码，则想要识别`10`个可能的数字。患者是否可能患有三种或五种不同的疾病进行分类。这也是一个多元分类问题，您可能会查看制药公司生产的药丸的图片，并试图弄清楚它是否有划痕效果、变色缺陷或芯片缺陷。这又将有多种不同类型的缺陷，您可以对这种药丸进行分类。因此，**多元分类问题**仍然是分类问题，因为{% mathjax %}\hat{y}{% endmathjax %}只能取少数离散类别，而不是任意数字，但现在{% mathjax %}\hat{y}{% endmathjax %}可以取两个以上的值。因此，之前对于分类，可能有一个这样的数据集，其中包含特征{% mathjax %}x_1,x_2{% endmathjax %}。在这种情况下，**逻辑回归**将**拟合**模型，预测给定特征{% mathjax %}\vec{x}{% endmathjax %}时{% mathjax %}\mathbf{P}(y=1|\vec{x}){% endmathjax %}的概率。对于**多元分类问题**，y 要么是 01，要么是 01，所以您将得到一个可能看起来像这样的数据集。有四个类别，其中`O`代表一个类，`x`代表另一个类。三角形代表第三类，正方形代表第四类。那么{% mathjax %}\mathbf{P}(y=1|\vec{x}){% endmathjax %}的概率是多少？或者{% mathjax %}\mathbf{P}(y=2|\vec{x}){% endmathjax %}的概率是多少？或者{% mathjax %}\mathbf{P}(y=3|\vec{x}){% endmathjax %}的概率是多少？或者{% mathjax %}\mathbf{P}(y=4|\vec{x}){% endmathjax %}的概率是多少？
{% asset_img ml_7.png %}

`softmax`回归算法是**逻辑回归**的**泛化**，**逻辑回归**是一种二元分类算法，适用于**多元分类**场景。让我们来看看它是如何工作的。回想一下，当{% mathjax %}\hat{y}{% endmathjax %}可以采用两个可能的输出值（`0`或`1`）时，适用于**逻辑回归**，首先计算{% mathjax %}z = \vec{W}\cdot\vec{x} + b{% endmathjax %}，然后计算{% mathjax %}\vec{a} = g{z} = \frac{1}{1 + e^{-z}}{% endmathjax %}，这是一个应用于{% mathjax %}z{% endmathjax %}的`S`型激活函数。给定输入特征{% mathjax %}\vec{x}{% endmathjax %}，预测**逻辑回归**为{% mathjax %}\hat{y} = 1{% endmathjax %}的概率。如果{% mathjax %}\mathbf{P}(y=1|\vec{x}) = 0.71{% endmathjax %}，那么{% mathjax %}\mathbf{P}(y=0|\vec{x}){% endmathjax %}的概率是多少？{% mathjax %}\mathbf{P}(y=1|\vec{x}){% endmathjax %}和{% mathjax %}\mathbf{P}(y=0|\vec{x}) = 0.29{% endmathjax %}，它们加起来必须等于`1`。将**逻辑回归**视为计算两个数值：第一个是{% mathjax %}a_1 = \mathbf{P}(y=1|\vec{x}){% endmathjax %}；第二个**逻辑回归**视为计算{% mathjax %}a_2 = 1 - \mathbf{P}(y=1|\vec{x}){% endmathjax %}，且{% mathjax %}a_1 + a_2 = 1{% endmathjax %}。现在将其推广到`softmax`回归，使用一个具体的例子来说明当{% mathjax %}\hat{y}{% endmathjax %}采用四种可能的输出时。以下是`softmax`回归执行的操作，计算{% mathjax %}z_1 = w_1\cdot\vec{x} + b_1{% endmathjax %}，然后计算{% mathjax %}z_2 = w_2\cdot\vec{x} + b_2{% endmathjax %}，对于{% mathjax %}z_3, z_4{% endmathjax %}依此类推，{% mathjax %}w_1,w_2,w_3,w_4{% endmathjax %}以及{% mathjax %}b_1,b_2,b_3,b_4{% endmathjax %}是`softmax`回归的参数。接下来，`softmax`回归的公式，计算{% mathjax %}a_1 = \mathbf{P}(y=1|\vec{x}) = \frac{e^{z_1}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}{% endmathjax %}。接下来将计算{% mathjax %}a_2 = \mathbf{P}(y=2|\vec{x}) = \frac{e^{z_2}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}{% endmathjax %}。接下来将计算{% mathjax %}a_3 = \mathbf{P}(y=3|\vec{x}) = \frac{e^{z_3}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}{% endmathjax %}，同样计算{% mathjax %}a_4 = \mathbf{P}(y=4|\vec{x}) = \frac{e^{z_4}}{e^{z_1} + e^{z_2} + e^{z_3} + e^{z_4}}{% endmathjax %}。

参数{% mathjax %}w_1,\ldots,w_4{% endmathjax %}和{% mathjax %}b_1,\ldots,b_4{% endmathjax %}，您可能已经意识到，由于{% mathjax %}\mathbf{P}(y=1,\ldots,4|\vec{x}){% endmathjax %}，它们加起来必须等于`1`，因此{% mathjax %}\mathbf{P}(y=4|\vec{x}) = 1 - 0.3 - 0.2 - 0.15 = 0.35{% endmathjax %}。一般情况下，{% mathjax %}y{% endmathjax %}可以取`n`个可能值，因此{% mathjax %}y\in {1,2,3,\ldots,n}{% endmathjax %}。`softmax`回归计算为{% mathjax %}z_j = \vec{W}_j\cdot\vec{x} + b_j\;j\in {1,2,\ldots,n}{% endmathjax %}，`softmax`回归的参数为{% mathjax %}w_1,\ldots,w_n{% endmathjax %}，以及{% mathjax %}b_1,b_2,\ldots,b_n{% endmathjax %}。最后计算{% mathjax %}a_j = \frac{e^{z_j}}{\sum_{k=1}^n e^{z_k}}{% endmathjax %}。{% mathjax %}j{% endmathjax %}指的是一个固定数值，例如{% mathjax %}j = 1{% endmathjax %}。则{% mathjax %}a_j = \mathbf{P}(y = j|\vec{x}){% endmathjax %}。{% mathjax %}a_1 + a_2 + \ldots + a_n = 1{% endmathjax %}。如果是{% mathjax %}n = 2{% endmathjax %}的`softmax`回归，那么只有两个输出类，`softmax`回归计算的结果与**逻辑回归**基本相同。这就是`softmax`回归模型是**逻辑回归**的**泛化**原因。在定义了`softmax`回归如何计算其输出，现在看一下如何指定`softmax`回归的**成本函数**。回想一下**逻辑回归**。之前{% mathjax %}a^{[1]} = g(z) = \mathbf{P}(y = 1|\vec{x})\;,\;a^{[2]} = \mathbf{P}(y =0|\vec{x}){% endmathjax %}。**逻辑回归**的损失写为{% mathjax %}\text{loss} = -y\log a_1 - (1-y)\log (1-a_1){% endmathjax %}，因为{% mathjax %}a_2 = 1 - a_1{% endmathjax %}。可以简化**逻辑回归**的损失，将其写为{% mathjax %}\text{loss} = -y\log (a_1 - 1) - y\log a_2{% endmathjax %}。换句话说，如果{% mathjax %}y = 1{% endmathjax %}，则损失为{% mathjax %} - \log a_1{% endmathjax %}。如果{% mathjax %}y = 0{% endmathjax %}，则损失为{% mathjax %}-\log a_2{% endmathjax %}，模型中所有参数的**成本函数**是**平均损失**，即整个训练集的**平均损失**。也是**逻辑回归**的**成本函数**{% mathjax %}\mathbf{J}(\vec{W},b) = \text{average loss}{% endmathjax %}。对于`softmax`回归的**成本函数**，如果在{% mathjax %}a_1,\ldots,a_n{% endmathjax %}中，则损失为{% mathjax %}\text{loss}(a_1,\ldots,a_N,y) = \begin{cases}-\log a_1 & \text{if}\;y = 1\\-\log a_2 & \text{if}\;y = 2\\ \vdots \\-\log a_N & \text{if}\;y = N \end{cases}{% endmathjax %}。如果{% mathjax %}y = j{% endmathjax %}，则损失为{% mathjax %}-\log a_j{% endmathjax %}。{% mathjax %}-\log a_j{% endmathjax %}是一条如下所示的曲线。
{% asset_img ml_8.png %}

如果{% mathjax %}j \approx 1{% endmathjax %}，则超出曲线的这一部分，损失将非常小。但是，如果{% mathjax %}j = 0.5{% endmathjax %}，则损失会稍微大一些。{% mathjax %}j{% endmathjax %}越小，损失越大。这会激励算法使{% mathjax %}j{% endmathjax %}尽可能大，尽可能接近`1`。请注意，每个训练示例中的{% mathjax %}y{% endmathjax %}只能取一个值。如果{% mathjax %}y = j{% endmathjax %},则计算损失只能为{% mathjax %}-\log a_j{% endmathjax %}。例如，如果{% mathjax %}y = 2{% endmathjax %}，只会计算{% mathjax %}-\log a_2{% endmathjax %}，但不会计算{% mathjax %}-\log a_1{% endmathjax %}。这是模型的形式以及`softmax`回归的**成本函数**。如果您要训练此模型，则可以构建**多元分类算法**。

为了构建一个可以进行**多元分类**的**神经网络**，我们将采用`Softmax`回归模型，并将其放入**神经网络**的**输出层**。如果要对`10`个类（`0~9`）进行手写数字分类，那么神经网络具有`10`个输出单元，如下图所示。这个新的**输出层**将是一个`softmax`输出层。有时会说这个**神经网络**有一个`softmax`输出。这个**神经网络**中的**前向传播**工作方式是给定一个输入{% mathjax %}\vec{x}{% endmathjax %}，{% mathjax %}\vec{a}^{[1]}{% endmathjax %}的计算方式与之前完全相同。然后是{% mathjax %}\vec{a}^{[2]}{% endmathjax %}，第二个隐藏层的激活也与之前完全相同。现在计算这个输出层的激活{% mathjax %}\vec{a}^{[3]}{% endmathjax %}。然后计算{% mathjax %}z_1 = \vec{W}_1\cdot\vec{a}^{[2]} + b_1,z_2 = \vec{W}_2\cdot\vec{a}^{[2]} + b_1,\ldots,z_{10} = \vec{W}_{10}\cdot\vec{a}^{[2]} + b_{10}{% endmathjax %}。然后{% mathjax %}a_1 = \frac{e^{z_1}}{\sum_{m=1}^{10} e^{z_m}} = \mathbf{P}(y= 1|\vec{x}) = g(z_1),a_2 = \frac{e^{z_2}}{\sum_{m=1}^{10} e^{z_m}} = \mathbf{P}(y= 2|\vec{x}) = g(z_2),\ldots,a_{10} = \frac{e^{z_{10}}}{\sum_{m=1}^{10} e^{z_m}} = \mathbf{P}(y= 10|\vec{x}) = g(z_10){% endmathjax %}。这就是{% mathjax %}\hat{y} = 1{% endmathjax %}的概率。`softmax`层有时也称为`softmax`激活函数，对于`softmax`激活函数，{% mathjax %}\vec{a}^{[1]}{% endmathjax %}是{% mathjax %}z_1,\ldots,z_{10}{% endmathjax %}的函数。因此，每个激活值都取决于所有{% mathjax %}z{% endmathjax %}值。这是`softmax`激活函数独有的属性。
{% asset_img ml_9.png %}

最后，让我们看看如何在`TensorFlow`中实现它。训练模型有三个步骤。第一步是告诉`TensorFlow`按顺序串联三层。第一层是`25`个单元的`ReLU`激活函数。第二层是`15`个单元的`ReLU`激活函数。然后第三层是`10`个**输出单元**的`softmax`激活函数。`TensorFlow`中的**成本函数**称之为`SparseCategoricalCrossentropy`。而对于逻辑回归，有`BinaryCrossEntropy`函数。稀疏分类指的是将{% mathjax %}\hat{y}{% endmathjax %}分类到{% mathjax %}{1,\ldots,10}{% endmathjax %}类别中。**稀疏**指的是{% mathjax %}\hat{y}{% endmathjax %}只能取这`10`个值中的一个。你不会看到一幅图像同时是2和7，所以稀疏是指每个数字只是这些类别中的一个。
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layouts import Dense
from tensorflow.keras.loss import SparseCategoricalCrossentropy

model = Squential([
    Dense(units = 25,activation='relu'),      # layer 1
    Dense(units = 15,activation='relu'),      # layer 2
    Dense(units = 10, activation='softmax')]) # layer 3

model.compile(loss = loss.SparseCategoricalCrossentropy())
model.fit(X,y,epochs=100)
```
在计算机中计算相同数值的两种不同方法。选项一，可以将{% mathjax %}x{% endmathjax %}设置为{% mathjax %}x = \frac{2}{10000}{% endmathjax %}。选项二，我们可以将{% mathjax %}x{% endmathjax %}设置为 {% mathjax %}x = (1 + \frac{1}{10000}) - (1 - \frac{1}{10000}){% endmathjax %}。首先，们将{% mathjax %}x{% endmathjax %}设置为{% mathjax %}x = \frac{2}{10000}{% endmathjax %}，并将结果打印到小数点后几位。看起来相当不错。其次，将{% mathjax %}x{% endmathjax %}设置为{% mathjax %}x = (1 + \frac{1}{10000}) - (1 - \frac{1}{10000}){% endmathjax %}。将其打印出来。它看起来存在一些舍入误差。由于计算机只有有限的内存来存储每个数字（称为浮点数），决定如何计算{% mathjax %}\frac{2}{10000}{% endmathjax %}的值，结果可能或多或少的数值舍入误差。虽然计算`softmax`成本函数的方法是正确的，但还有一种不同的方法可以减少这些数值舍入误差，从而在`TensorFlow`中进行更准确的计算。让我首先使用逻辑回归更详细地解释这一点。然后我们将展示如何将这些想法应用于改进我们的 softmax 实现。首先，让我使用逻辑回归来说明这些想法。然后我们将继续展示如何改进您的 softmax 实现。回想一下，对于逻辑回归，如果要计算给定示例的**损失函数**，则首先要计算此输出激活{% mathjax %}a= g(z) = \frac{1}{1 + e^{-z}}{% endmathjax %}。然后，使用此处的表达式计算**损失**({% mathjax %}\text{loss} = -y\log(a) - (1 -y)\log(1 - a)\;,\;\text{loss} = -y\log(\frac{1}{1 + e^{-z}}) - (1 -y)\log(1 - \frac{1}{1 + e^{-z}}){% endmathjax %})。实际上，对于具有**二元交叉熵损失**的逻辑输出层，代码如下所示。
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layouts import Dense

model = Squential([
    Dense(units = 25,activation='relu'),      # layer 1
    Dense(units = 15,activation='relu'),      # layer 2
    Dense(units = 10, activation='sigmoid')]) # layer 3

model  = compile(loss = loss.BinaryCrossEntropy())
```
对于**逻辑回归**，通常数值舍入误差不会那么严重。`TensorFlow`不用将{% mathjax %}\vec{a}{% endmathjax %}作为中间项进行计算。相反，TensorFlow可以重新排列此表达式中的项，有一种更精确的方法来计算此**损失函数**。而原始程序计算中间值{% mathjax %}1 + \frac{1}{10000}{% endmathjax %}和另一个中间值{% mathjax %}1 - \frac{1}{10000}{% endmathjax %}，然后让这两个值相加得到{% mathjax %}\frac{2}{10000}{% endmathjax %}。此部分实现明确{% mathjax %}a{% endmathjax %}作为计算中间量。通过将此表达式指定为**损失函数**，`TensorFlow`提供了更大的灵活性。代码如下所示，它的作用是将输出层设置为**线性激活函数**，并将激活函数{% mathjax %}\frac{1}{1+e^{-z}}{% endmathjax %}，以及**交叉熵损失**放入此处的**损失函数**中。
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layouts import Dense

model = Squential([
    Dense(units = 25,activation='relu'),      # layer 1
    Dense(units = 15,activation='relu'),      # layer 2
    Dense(units = 10, activation='linear')])  # layer 3

model  = compile(loss = loss.BinaryCrossEntropy(from_logits = True))
```
`from_logits = True`参数。如果想知道`logits`是什么，它就是这个{% mathjax %}z{% endmathjax %}。 `TensorFlow`会将{% mathjax %}z{% endmathjax %}计算为中间值，但它可以重新排列，使其计算得更准确。此代码的一个缺点是它变得有点复杂。但这会使`TensorFlow`的数值舍入误差小一些。在**逻辑回归**的情况下，这两种实现实际上都可以正常工作，但当涉及到`softmax`时，数值舍入误差会变得更糟。现在将这个想法应用于`softmax`回归。回想一下上一个例子，按如下方式计算激活。激活是{% mathjax %}(a_1,a_2,\ldots,a_{10}) = g(z_1,z_2,\ldots,z_{10}){% endmathjax %}，其中{% mathjax %}a_1 = \frac{e^{z_1}}{\sum_{m=1}^{10} e^{z_m}}{% endmathjax %}，然后{% mathjax %}\text{loss}= L(\vec{a},y) = \begin{cases}-\log a_1 & \text{if}\;y = 1\\-\log a_2 & \text{if}\;y = 2\\ \vdots \\-\log a_{10} & \text{if}\;y = {10} \end{cases}{% endmathjax %}，将其代入{% mathjax %}\text{loss}= L(\vec{a},y) = \begin{cases}-\log \frac{e^{z_1}}{\sum_{m=1}^{10} e^{z_m}} & \text{if}\;y = 1\\-\log \frac{e^{z_2}}{\sum_{m=1}^{10} e^{z_m}} & \text{if}\;y = 2\\ \vdots \\-\log \frac{e^{z_10}}{\sum_{m=1}^{10} e^{z_m}} & \text{if}\;y = {10} \end{cases}{% endmathjax %}。但是，如果你指定损失函数为{% mathjax %}y = 1{% endmathjax %}时，则为{% mathjax %}-\log a_1{% endmathjax %}，依此类推。`TensorFlow`可以重新排列项，并以精确的方式进行计算。如果`z`非常小，{% mathjax %}e^z{% endmathjax %}变得非常小，相反，{% mathjax %}z{% endmathjax %}值非常大，那么{% mathjax %}e^z{% endmathjax %}可以通过重新排列，`TensorFlow`可以避免一些非常小或非常大的值，从而为损失函数提供更精确的计算。执行此操作的代码显示在输出层中，现在只使用**线性激活函数**，输出层只计算{% mathjax %}z_1,\ldots,z_{10}{% endmathjax %}，然后将整个损失计算捕获在**损失函数**中，其中使用`from_logists = True`参数。`Logist`回归的数值舍入误差并不是那么糟糕，但建议您改用此实现，从概念上讲，此代码与您之前的第一个版本的功能相同，只是它在数值上更精确一些。虽然缺点可能也更难理解。现在还有一个细节，就是将神经网络中的输出层更改为使用**线性激活函数**，而不是`softmax`激活函数。**神经网络**的**输出层**不再输出{% mathjax %}a_1,\ldots,a_{10}{% endmathjax %}。而是{% mathjax %}z_1,\ldots,z_{10}{% endmathjax %}。
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layouts import Dense

model = Squential([
    Dense(units = 25,activation='relu'),      # layer 1
    Dense(units = 15,activation='relu'),      # layer 2
    Dense(units = 10, activation='linear')])  # layer 3

model  = compile(loss = loss.BinaryCrossEntropy(from_logits = True))
model.fit(X,y,epochs = 100)

logits = model(X)
f_x = tf.nn.sigmoid(logits)
```
还有一种不同类型的分类问题，称为**多标签分类问题**，即每个图像可能有多个标签。如果您正在构建自动驾驶汽车或驾驶员辅助系统，那么给出一张汽车前方的图片，您可能想问一个问题，例如，有汽车吗？或者有公共汽车吗？或者有行人吗？在这种情况下，有汽车，没有公共汽车，至少有一名行人，或者在第二张图片中，没有汽车，没有公共汽车，有行人；有汽车，有公共汽车，没有行人。这些是多标签分类问题的示例，因为与单个输入图像{% mathjax %}\vec{x}{% endmathjax %}相关联，有三个不同的标签，分别对应于图像中是否有汽车、公共汽车或行人。在这种情况下，{% mathjax %}\vec{y}{% endmathjax %}的目标是一个由三个数字组成的向量，这与**多元分类问题**不同，在多元分类问题中，例如手写数字分类，{% mathjax %}y{% endmathjax %}只是一个数字，即使这个数字可以取`10`个不同的值。如何构建**多标签分类**的**神经网络**？一种方法是将其视为三个完全独立的机器学习问题。您可以构建一个神经网络来判断是否有汽车？第二个神经网络用于检测公交车，第三个神经网络用于检测行人。其实这是一个不合理的方法。还有另一种方法，即训练单个神经网络同时检测汽车、公共汽车和行人这三种情况，如果神经网络架构如下图所示，输入{% mathjax %}vec{x}{% endmathjax %}。第一个隐藏层提供{% mathjax %}\vec{a}^{[1]}{% endmathjax %}，第二个隐藏层提供{% mathjax %}\vec{a}^{[2]}{% endmathjax %}，最后一个输出层提供{% mathjax %}\vec{a}^{[3]}{% endmathjax %}，它将是由三个数值组成的向量。这是三个**二元分类问题**，你可以在**输出层**中对这三个节点中的每一个使用`S`型激活函数{% mathjax %}\vec{a}^{[3]} = \begin{bmatrix}a_1^{[3]} \\a_2^{[3]} \\a_3^{[3]} \end{bmatrix}{% endmathjax %}，结果如下图所示。
{% asset_img ml_10.png %}
