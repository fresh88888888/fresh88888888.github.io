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

构建**逻辑回归模型**的第一步是指定如何根据输入特征{% mathjax %}x{% endmathjax %}和参数{% mathjax %}w,b{% endmathjax %}计算输出。**逻辑回归函数**预测{% mathjax %}f_{\vec{W},b}(\vec{x}) = g{% endmathjax %}。{% mathjax %}g = \vec{W}\cdot\vec{x} + b{% endmathjax %}，那么{% mathjax %}g(z) = \frac{1}{1+ e^{-z}}{% endmathjax %}，代码实现为：`z = np.dot(w , x) + b, f_x = 1/(1 + np.exp(-z))`。
<!-- more -->

第一步是指定**逻辑回归**的输入到输出函数是什么？这取决于输入{% mathjax %}\vec{x}{% endmathjax %}和模型的参数。训练识字回归模型的第二步是指定**损失函数**和**成本函数**，损失函数表示为{% mathjax %}L(f_{\vec{W},b}(\vec{x}),y){% endmathjax %}，标签和训练集为{% mathjax %}y{% endmathjax %}，则该单个训练示例的损失为：`loss = -y * np.log(f_x) - (1-y) * np.log(1 - f_x)`。这是衡量**逻辑回归**在单个训练示例{% mathjax %}(x,y){% endmathjax %}上表现如何的**指标**。给定**损失函数**的定义，然后定义**成本函数**，**成本函数**是参数{% mathjax %}W,b{% endmathjax %}的函数，这只是对{% mathjax %}m{% endmathjax %}个训练示例{% mathjax %}\mathbf{J}(\vec{W},b) = \frac{1}{m}\sum_{i=1}^m L(f_{\vec{W},b}(\vec{x}^{(i)}),y^{(i)}){% endmathjax %}，我们使用的**损失函数**是学习算法输出和基本事实标签的函数，它是在单个训练示例中计算的，而**成本函数**{% mathjax %}\mathbf{J}{% endmathjax %}是在整个训练集上计算的**损失函数**的平均值。这是构建**逻辑回归**的第二步。然后，**训练逻辑回归模型**的第三步也是使用一种算法，**梯度下降法**，最小化成本函数{% mathjax %}\mathbf{J}(\mathbf{W},b){% endmathjax %}。使用**梯度下降法**最小化成本{% mathjax %}\mathbf{J}{% endmathjax %}作为参数的函数，代码实现为：`w = w - alpha * dj_dw, b = b - alpha * dj_db`。第一步，指定如何根据输入{% mathjax %}\vec{x}{% endmathjax %}和参数计算输出，第二步指定**损失**和**成本**，第三步最小化**训练逻辑回归**的**成本函数**。同样的三个步骤也是在`TensorFlow`中训练神经网络的方法。现在让我们看看这三个步骤如何映射到训练神经网络。第一步是指定如何根据输入{% mathjax %}\vec{x}{% endmathjax %}和参数{% mathjax %}W,b{% endmathjax %}计算输出，这可以通过此代码片段完成。
```python
model = Squential([Dense(...),Dense(...),Dense(...)])
```

第二步是编译模型并告诉它要使用什么损失，这是用于指定此损失函数的代码：`model =.compile(loss = BinaryCrossentropy())`，即**二元交叉熵损失函数**，一旦指定此损失，对整个训练集取平均值，然后第三步是调用函数将成本最小化为神经网络参数的函数：`model.fit(x,y,epochs = 100)`。接下来更详细地了解这三个步骤。第一步，指定如何根据输入{% mathjax %}x{% endmathjax %}和参数{% mathjax %}w,b{% endmathjax %}计算输出。此代码片段指定了神经网络的整个架构。第一个隐藏层有`25`个隐藏单元，下一个隐藏层有`15`个隐藏单元，然后是一个输出单元，使用`S`型激活值。
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layouts import Dense

model = Squential([Dense(units = 25,activation='sigmoid'),Dense(units = 15,activation='sigmoid'),Dense(units = 1,activation='sigmoid')])
```
基于此代码片段，我们还知道第一层的参数{% mathjax %}W^{[1]},\vec{b}^{[1]}{% endmathjax %}第二层的参数{% mathjax %}W^{[2]},\vec{b}^{[2]}{% endmathjax %}和第三层的参数{% mathjax %}W^{[3]},\vec{b}^{[3]}{% endmathjax %}。此代码片段指定了神经网络的整个架构。为了计算{% mathjax %}\vec{x}{% endmathjax %}的输出\vec{a}^{[3]}，这里我们写了{% mathjax %}W^{[l]},\vec{b}^{[l]}{% endmathjax %}。在第二步中，必须指定**损失函数**是什么？这也将定义**成本函数**。对于手写数字分类问题，其中图像要么是`0`，要么是`1`，到目前为止最常见的**损失函数**是{% mathjax %}L(f(\vec{x}),y) = -y\log(f(\vec{x})) - (1 - y)\log(1 - f(\vec{x})){% endmathjax %}，{% mathjax %}y{% endmathjax %}称为**目标标签**，而{% mathjax %}f(\vec{x}){% endmathjax %}是神经网络的输出。在`TensorFlow`中，这称为**二元交叉熵损失函数**。这个名字从何而来？在统计学中，这个函数被称为**交叉熵损失函数**，而二元这个词只是再次强调这是一个**二元分类**问题，因为每个图像要么是`0`，要么是`1`。`TensorFlow`使用此**损失函数**，`keras`最初是一个独立于`TensorFlow`开发库，实际上是与`TensorFlow`完全独立的项目。但最终它被合并到`TensorFlow`中，这就是为什么有`tf.Keras`库。指定了单个训练示例的损失后，`TensorFlow`知道您想要最小化的成本是**平均值**，即对所有训练示例的损失取所有`m`个训练示例的平均值。如果想解决**回归问题**而不是**分类问题**。可以告诉`TensorFlow`使用其它的损失函数编译模型。例如，如果有一个**回归问题**，并且如果想**最小化平方误差损失**。如果学习算法输出{% mathjax %}f(\vec{x}){% endmathjax %}，目标为{% mathjax %}y{% endmathjax %}，则损失是平方误差的`1/2`。然后在`TensorFlow`中使用此损失函数。在这个表达式中，成本函数表示为表示为：{% mathjax %}\mathbf{J}(\mathbf{W,B}) = \frac{1}{m}\sum_{i=1}^m L(f(\vec{x}^{(1)}),y^{(i)}){% endmathjax %}。**成本函数**是神经网络中所有参数的函数。将大写{% mathjax %}\mathbf{W}{% endmathjax %}视为包括{% mathjax %}W^{[1]},W^{[2]},W^{[3]}{% endmathjax %}和{% mathjax %}b^{[1]},b^{[2]},b^{[3]}{% endmathjax %}。将{% mathjax %}f(\vec{x}){% endmathjax %}作为**神经网络**的输出，但如果想强调神经网络的输出作为{% mathjax %}\vec{x}{% endmathjax %}的函数取决于神经网络所有层中的所有参数，可以写为{% mathjax %}f_{\vec{W},b}(\vec{x}){% endmathjax %}。这就是**损失函数**和**成本函数**。最后，将用`TensorFlow`**最小化交叉函数**。使用**梯度下降法**训练神经网络的参数，那么对于每个层{% mathjax %}l{% endmathjax %}和每个单元{% mathjax %}j{% endmathjax %}，需要根据{% mathjax %}w_j^{[l]} = w_j^{[l]} - \alpha\frac{\partial}{\partial w_j}\mathbf{J}(\vec{W},b)\;,\;b_j^{[l]} = b_j^{[l]} - \alpha\frac{\partial}{\partial b_j}\mathbf{J}(\vec{W},b){% endmathjax %}。进行`100`次**梯度下降**迭代之后，希望能得到一个好的参数值。为了使用**梯度下降法**，需要计算的关键是这些**偏导数项**。神经网络训练的标准做法，是使用一种称为**反向传播**的算法来计算这些**偏导数项**。`TensorFlow`可以完成所有这些工作。称为`fit`的函数中实现了**反向传播**。您所要做的就是调用`model.fit(X、y, epochs=100)`作为训练集，并告诉它进行`100`次迭代。
{% asset_img ml_1.png %}

