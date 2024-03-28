---
title: TensorFlow & JAX
date: 2024-03-27 18:10:32
tags:
  - AI
categories:
  - 人工智能
---

`JAX`是一个**高性能机器学习库**。`JAX`在加速器（例如`GPU`和`TPU`）上编译并运行`NumPy`代码。您可以使用`JAX`（以及为`JAX`构建的神经网络库`FLAX`）来构建和训练深度学习模型。
<!-- more -->
#### 环境设置

```python
import numpy as np
import tensorflow as tf
import keras
from keras import layers, callbacks

seed=1234
np.random.seed(seed)
tf.random.set_seed(seed)
```
#### 张量（Tensors）

什么是**张量**？尽管张量的含义与我们在机器学习中通常使用的含义有很大不同，但每当我们在机器学习中提到张量时，我们的意思是它是一个**多维数组**，其中所有值都具有统一的数据类型。创建`TF`张量的方法有很多种。我们将看看其中的一些重要的。`tf.constant(..)`：这是创建张量对象的最简单方法，但存在一些问题。首先，让我们尝试用它创建一个张量，然后我们稍后再看看其中的问题。
```python
# A zero rank tensor. A zero rank tensor is nothing but a single value
x = tf.constant(5.0)
print(x)
```
结果输出为：
```bash
tf.Tensor(5.0, shape=(), dtype=float32)
```
正如您在上面看到的，**张量对象**具有**形状和数据类型**。还有其他与张量对象关联的`attributes/properties`。
- 形状(`Shape`)：张量每个轴的长度（元素数量）。
- 等级(`Rank`)：轴的数量。例如，矩阵是`2`阶张量。
- 轴或维度(`Axis or Dimension`)：张量的特定维度。
- 大小(`Size`)：张量中的项目总数。

```python
# We can convert any tensor object to `ndarray` by calling the `numpy()` method
y = tf.constant([1, 2, 3], dtype=tf.int8).numpy()
print(f"`y` is now a {type(y)} object and have a value == {y}")
```
结果输出为：
```bash
`y` is now a <class 'numpy.ndarray'> object and have a value == [1 2 3]
```
{% note warning %}
**注意**
- 人们将`tf.constant(..)`与创建常量张量的操作混淆。不存在这样的关系。这与我们如何在`tf.Graph`中嵌入节点有关。
- 默认情况下，`TensorFlow`中的任何张量都是不可变的，即张量一旦创建就无法更改其值。你必须创造一个新的。这与`numpy`和`pytorch`不同，在`numpy`和`pytorch`中您可以修改。
- 与`tf.constant`最接近的成员之一是`tf.convert_to_tensor()`方法，有一些差异。
- `tf.constant(..)`只是创建张量的多种方法之一。
{% endnote %}

```python
# Immutability check

# Rank-1 tensor
x = tf.constant([1, 2], dtype=tf.int8)

# Try to modify the values
try:
    x[1] = 3
except Exception as ex:
    print(type(ex).__name__, ex)

# tf.constant(..) is no special. Let's create a tensor using a diff method
x = tf.ones(2, dtype=tf.int8)
print(x)

try:
    x[0] = 3
except Exception as ex:
    print("\n", type(ex).__name__, ex)

# Check all the properties of a tensor object
print(f"Shape of x : {x.shape}")
print(f"Another method to obtain the shape using `tf.shape(..)`: {tf.shape(x)}")

print(f"\nRank of the tensor: {x.ndim}")
print(f"dtype of the tensor: {x.dtype}")
print(f"Total size of the tensor: {tf.size(x)}")
print(f"Values of the tensor: {x.numpy()}")
```
结果输出为：
```bash
TypeError 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment

tf.Tensor([1 1], shape=(2,), dtype=int8)
TypeError 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment

Shape of x : (2,)
Another method to obtain the shape using `tf.shape(..)`: [2]
Rank of the tensor: 1
dtype of the tensor: <dtype: 'int8'>
Total size of the tensor: 2
Values of the tensor: [1 1]
```
无法在`Tensor`对象中进行赋值有点令人沮丧。那有什么办法解决呢？我发现的一直适用于我的用例的最佳方法是创建一个掩码或使用`tf.tensor_scatter_nd_update`。让我们看一个例子。
`Original tensor` -> `[1, 2, 3, 4, 5]`和`Output tensor we want` -> `[1, 200, 3, 400, 5]`
```python
# Create a tensor first. Here is another way
x = tf.cast([1, 2, 3, 4, 5], dtype=tf.float32)
print("Original tensor: ", x)

mask = x%2 == 0
print("Original mask: ", mask)

mask = tf.cast(mask, dtype=x.dtype)
print("Mask casted to original tensor type: ", mask)

# Some kind of operation on an tensor that is of same size 
# or broadcastable to the original tensor. Here we will simply
# use the range object to create that tensor
temp = tf.cast(tf.range(1, 6) * 100, dtype=x.dtype)

# Output tensor
# Input tensor -> [1, 2, 3, 4, 5]
# Mask -> [0, 1, 0, 1, 0]
out = x * (1-mask) + mask * temp
print("Output tensor: ", out)
```
结果输出为：
```bash
Original tensor:  tf.Tensor([1. 2. 3. 4. 5.], shape=(5,), dtype=float32)
Original mask:  tf.Tensor([False  True False  True False], shape=(5,), dtype=bool)
Mask casted to original tensor type:  tf.Tensor([0. 1. 0. 1. 0.], shape=(5,), dtype=float32)
Output tensor:  tf.Tensor([  1. 200.   3. 400.   5.], shape=(5,), dtype=float32)
```
```python
# Another way to achieve the same thing
indices_to_update = tf.where(x % 2 == 0)
print("Indices to update: ", indices_to_update)

# Update the tensor values
updates = [200., 400.]
out = tf.tensor_scatter_nd_update(x, indices_to_update, updates)
print("\nOutput tensor")
print(out)
```
```bash
Indices to update:  tf.Tensor([[1][3]], shape=(2, 1), dtype=int64)

Output tensor
tf.Tensor([  1. 200.   3. 400.   5.], shape=(5,), dtype=float32)
```
现在让我们看看另一件有趣的事情。
```python
# This works!
arr = np.random.randint(5, size=(5,), dtype=np.int32)
print("Numpy array: ", arr)
print("Accessing numpy array elements based on a  condition with irregular strides", arr[[1, 4]])

# This doesn't work
try:
    print("Accessing tensor elements based on a  condition with irregular strides", x[[1, 4]])
except Exception as ex:
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
Numpy array:  [3 4 4 0 1]
Accessing numpy array elements based on a  condition with irregular strides [4 1]
InvalidArgumentError Index out of range using input dim 1; input has only 1 dims [Op:StridedSlice] name: strided_slice
```
现在怎么办？如果您想从具有不规则步长或定义不明确的步长的张量中提取多个元素，那么`tf.gather`和`tf.gather_nd`是您的朋友。让我们再试一次！
```python
print("Original tensor: ", x.numpy())

# Using the indices that we used for mask
print("\nIndices to update: ", indices_to_update.numpy())

# This works!
print("\n Accesing tensor elements using gather")
print("\n", tf.gather(x, indices_to_update).numpy())
```
```bash
Original tensor:  [1. 2. 3. 4. 5.]
Indices to update:  [[1][3]]
Accesing tensor elements using gather
[[2.][4.]]
```
还有另一种方法`tf.convert_to_tensor(..)`来创建张量。这与`tf.constant(..)`非常相似，但有一些细微的差别：
- 每当您将非`tf.Tensor`对象（例如`Python`列表或`ndarray`）传递给操作时，始终会自动调用`Convert_to_tensor(..)`
- 它不会形成为输入参数。
- 它甚至允许传递符号张量。

何时使用`tf.convert_to_tensor(..)`？ 这取决于你的思维模式！
```python
#  An example with a python list
y = tf.convert_to_tensor([1, 2, 3])
print("Tensor from python list: ", y)

#  An example with a ndarray
y = tf.convert_to_tensor(np.array([1, 2, 3]))
print("Tensor from ndarray: ", y)

#  An example with symbolic tensors
with tf.compat.v1.Graph().as_default():
    y = tf.convert_to_tensor(tf.compat.v1.placeholder(shape=[None, None, None], dtype=tf.int32))
print("Tensor from python list: ", y)
```
结果输出为：
```bash
Tensor from python list:  tf.Tensor([1 2 3], shape=(3,), dtype=int32)
Tensor from ndarray:  tf.Tensor([1 2 3], shape=(3,), dtype=int64)
Tensor from python list:  Tensor("Placeholder:0", shape=(None, None, None), dtype=int32)
```
##### String Tensors

```python
# String as a tensor object with dtype==tf.string
string = tf.constant("abc", dtype=tf.string)
print("String tensor: ", string)

# String tensors are atomic and non-indexable. 
# This doen't work as expected!
print("\nAccessing second element of the string")
try:
    print(string[1])
except Exception as ex:
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
String tensor:  tf.Tensor(b'abc', shape=(), dtype=string)
Accessing second element of the string
InvalidArgumentError Index out of range using input dim 0; input has only 0 dims [Op:StridedSlice] name: strided_slice
```
###### Ragged Tensors

简而言之，一个沿某个轴具有可变数量元素的张量。 
```python
# This works!
y = [[1, 2, 3],
     [4, 5],
     [6]
    ]

ragged = tf.ragged.constant(y)
print("Creating ragged tensor from python sequence: ", ragged)

# This won't work
print("Trying to create tensor from above python sequence\n")
try:
    z = tf.constant(y)
except Exception as ex:
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
Creating ragged tensor from python sequence:  <tf.RaggedTensor [[1, 2, 3], [4, 5], [6]]>
Trying to create tensor from above python sequence
ValueError Can't convert non-rectangular Python sequence to Tensor.
```
##### Sparse tensors

```python
# Let's say you have a an array like this one
# [[1 0 0]
#  [0 2 0]
#  [0 0 3]]
# If there are too many zeros in your `huge` tensor, then it is wise to use `sparse`
# tensors instead of `dense` one. Let's say how to create this one. We need to specify:
# 1. Indices where our values are
# 2. The values 
# 3. The actual shape

sparse_tensor = tf.SparseTensor(indices=[[0, 0], [1, 1], [2, 2]],
                                values=[1, 2, 3],
                                dense_shape=[3, 3]
                               )
print(sparse_tensor)

# You can convert sparse tensors to dense as well
print("\n", tf.sparse.to_dense(sparse_tensor))
```
结果输出为：
```bash
SparseTensor(indices=tf.Tensor([[0 0][1 1][2 2]], shape=(3, 2), dtype=int64), values=tf.Tensor([1 2 3], shape=(3,), dtype=int32), dense_shape=tf.Tensor([3 3], shape=(2,), dtype=int64))

tf.Tensor([[1 0 0][0 2 0][0 0 3]], shape=(3, 3), dtype=int32)
```
#### 变量（Variables）

**变量**是一种“特殊”的张量。它用于表示或存储可变状态。`tf.Variable`表示一个张量，其值可以通过对其运行操作来更改。想一想您会使用`Variable`对象的情况吗？**神经网络的权重**是变量使用的最好例子之一。我们将首先了解如何创建变量对象，然后我们将研究其属性和一些陷阱。

只有一种方法可以创建`Variable`对象：`tf.Variable(..)`
```python
# Variables with an integer value of 2 as initial value
x = tf.Variable(2)
x
# Nested list as initial value
y = tf.Variable([[2, 3]], dtype=tf.int32)
y
# Tuples also work but beware it isn't the same as a nested list.
# Check the difference between the current output and the previous cell output
w = tf.Variable(((2, 3)), dtype=tf.int32)
w
# You can even pass a tensor object as an initial value
t = tf.constant([1, 2,], dtype=tf.int32)
z = tf.Variable(t)
z

# An interesting thing to note. 
# You can't change the values of the tensor `t` in the above example
# but you can change the values of the variable created using it

# This won't work
try:
    t[0] = 1
except Exception as ex:
    print(type(ex).__name__, ex)
    
# This also won't work
try:
    z[0] = 10
except Exception as ex:
    print(type(ex).__name__, ex)
    
# This works though
print("\nOriginal variable: ", z)
z[0].assign(5)
print("Updated variable: ", z)
```
结果输出为：
```bash
<tf.Variable 'Variable:0' shape=() dtype=int32, numpy=2>

<tf.Variable 'Variable:0' shape=(1, 2) dtype=int32, numpy=array([[2, 3]], dtype=int32)>

<tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([2, 3], dtype=int32)>

<tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([1, 2], dtype=int32)>

TypeError 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
TypeError 'ResourceVariable' object does not support item assignment

Original variable:  <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([1, 2], dtype=int32)>
Updated variable:  <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([5, 2], dtype=int32)>
```
{% note info %}
**有几点需要注意**：
- 您可以通过传递初始值来创建变量，该初始值可以是`Tensor`或可转换为`Tensor`的`Python`对象。
- 您传递的张量对象是不可变的，但使用它创建的变量是可变的。
- **变量是一种特殊的张量**，但张量和变量的底层数据结构都是`tf.Tensor`。
- 由于数据结构相同，因此两者的大多数属性都是相同的。
- 直接赋值（如`z[0]=5`）也不适用于`tf.Variable`。要更改值，您需要调用`allocate(...)`、`assign_add(...)`或`allocate_sub(...)`等方法。
- 任何变量都与任何其他`Python`对象具有相同的生命周期。当没有对变量的引用时，它会自动释放。
{% endnote%}

```python
# Most of the properties that we saw for tensors in part1 are the same for variables
print(f"Shape of variable : {z.shape}")
print(f"Another method to obtain the shape using `tf.shape(..)`: {tf.shape(z)}")

print(f"dtype of the variable: {z.dtype}")
print(f"Total size of the variable: {tf.size(z)}")
print(f"Values of the variable: {z.numpy()}")

try:
    print(f"Rank: {z.ndim}")
except Exception as ex:
    print(type(ex).__name__, ex)

# Crap! How to find out the no of dimensions then?
print(f"Rank: {tf.shape(z)} or like this {z.shape}")

# Whatever operator overloading is available for a Tensor, is also available for a Variable
# We have a tensor `t` and a varibale `z`. 
t = tf.constant([1, 2,], dtype=tf.int32)
z = tf.Variable(t)
print("Tensor t: ", t)
print("Variable z: ", z)

print("\nThis works: ", (t+5))
print("So does this: ", (z +5))

print(f"Another example just for demonstration: {(t*5).numpy()}, {(z*5).numpy()}")

# Gather works as well
tf.gather(z, indices=[1])

# Here is another interesting difference between the properties of 
# a tensor and a variable
try:
    print("Is variable z trainable? ", z.trainable)
    print("Is tensor t trainable? ", t.trainable)
except Exception as ex:
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
Shape of variable : (2,)
Another method to obtain the shape using `tf.shape(..)`: [2]
dtype of the variable: <dtype: 'int32'>
Total size of the variable: 2
Values of the variable: [5 2]

AttributeError 'ResourceVariable' object has no attribute 'ndim'

Rank: [2] or like this (2,)

Tensor t:  tf.Tensor([1 2], shape=(2,), dtype=int32)
Variable z:  <tf.Variable 'Variable:0' shape=(2,) dtype=int32, numpy=array([1, 2], dtype=int32)>

This works:  tf.Tensor([6 7], shape=(2,), dtype=int32)
So does this:  tf.Tensor([6 7], shape=(2,), dtype=int32)
Another example just for demonstration: [ 5 10], [ 5 10]

<tf.Tensor: shape=(1,), dtype=int32, numpy=array([2], dtype=int32)>

Is variable z trainable?  True
AttributeError 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'trainable'
```
让我们来谈谈为什么对于`Variable`对象而言，`trainable`是一个有趣的属性。
- 任何变量都会被自动跟踪（如果它在范围内），除非它不可训练。
- 在继承`tf.Module`的类范围内定义的任何变量都会被自动跟踪，并且可以通过`trainable_variables`、变量或子模块属性进行收集。
- 有时我们不想要某个变量的梯度。在这种情况下，我们可以通过设置`trainable=False`来关闭跟踪。

```python
x = tf.Variable(2.0, name="x")
y = tf.Variable(4.0, trainable=False, name="y")
z = tf.Variable(6.0, name="z")

with tf.GradientTape() as tape:
    x = x + 2
    y = y + 5

print([variable.name for variable in tape.watched_variables()])
# ['x:0']
```
变量的最大优点是可以重用内存。您可以修改这些值而不创建新值，但需要记住一些事项。
```python
# Create a variable instance
z = tf.Variable([1, 2], dtype=tf.int32, name="z")
print(f"Variable {z.name}: ", z)

# Can we change the dtype while changing the values?
try:
    z.assign([1.0, 2.0])
except Exception as ex:
    print("\nOh dear...what have you done!")
    print(type(ex).__name__, ex)
    
# Can we change the shape while assigning a new value?
try:
    z.assign([1, 2, 3])
except Exception as ex:
    print("\nAre you thinking clearly?")
    print(type(ex).__name__, ex)
    
# A way to create variable with an arbitrary shape
x = tf.Variable(5, dtype=tf.int32, shape=tf.TensorShape(None), name="x")
print("\nOriginal Variable x: ", x)

# Assign a proper value with a defined shape
x.assign([1, 2, 3])
print("Modified Variable x: ", x)

# Try assigning a value with a diff shape now.
try:
    x.assign([[1, 2, 3], [4, 5, 6]])
    print("\nThis works!!")
    print("Variable value modified with a diff shape: ", x)
except Exception as ex:
    print("\nDid you forget what we just learned?")
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
Variable z:0:  <tf.Variable 'z:0' shape=(2,) dtype=int32, numpy=array([1, 2], dtype=int32)>

Oh dear...what have you done!
TypeError Cannot convert [1.0, 2.0] to EagerTensor of dtype int32

Are you thinking clearly?
ValueError Cannot assign to variable z:0 due to variable shape (2,) and value shape (3,) are incompatible

Original Variable x:  <tf.Variable 'x:0' shape=<unknown> dtype=int32, numpy=5>
Modified Variable x:  <tf.Variable 'x:0' shape=<unknown> dtype=int32, numpy=array([1, 2, 3], dtype=int32)>

This works!!
Variable value modified with a diff shape:  <tf.Variable 'x:0' shape=<unknown> dtype=int32, numpy=
array([[1, 2, 3],[4, 5, 6]], dtype=int32)>
```

#### 自动微分和梯度（Automatic Differentiation and Gradients）

**自动微分和梯度**是非常重要的概念，没有必要理解它的每一点，但你越深入它，你就会越欣赏它的美丽。假设您在前向传递中对输入应用一系列操作。要自动区分，您需要某种机制来弄清楚：
- 前向传递中应用了哪些操作？ 
- 应用操作的顺序是什么？

对于`autodiff`，你需要记住上面两条。不同的框架可以以不同的方式实现相同的想法，但基本原理保持不变。`TensorFlow`提供了`tf.GradientTape API`用于**自动微分**。在 `GradientTape`上下文中执行的任何相关操作都会被记录下来以进行**梯度计算**。要计算梯度，您需要执行以下操作：
- 记录`tf.GradientTape`上下文中的操作 。
- 使用`GradientTape.gradient(target, sources)`计算梯度。

让我们为此编写几个示例。
```python
# We will initialize a few Variables here

x = tf.Variable(3.0)
y = tf.Variable(4.0)

print(f"Variable x: {x}")
print(f"Is x trainable?: {x.trainable}")
print(f"\nVariable y: {y}")
print(f"Is y trainable?: {y.trainable}")
```
结果输出为：
```bash
Variable x: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.0>
Is x trainable?: True

Variable y: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>
Is y trainable?: True
```
我们将在这里做一个简单的操作：`z = x * y`。我们将计算`z`对于`x`和`y`的梯度。
```python
# Remember we need to execute the operations inside the context
# of GradientTape so that we can record them

with tf.GradientTape() as tape:
    z = x * y
    
dx, dy = tape.gradient(z, [x, y])

print(f"Input Variable x: {x.numpy()}")
print(f"Input Variable y: {y.numpy()}")
print(f"Output z: {z}\n")

# dz / dx
print(f"Gradient of z wrt x: {dx}")

# dz / dy
print(f"Gradient of z wrt y: {dy}")
```
结果输出为：
```bash
Input Variable x: 3.0
Input Variable y: 4.0
Output z: 12.0

Gradient of z wrt x: 4.0
Gradient of z wrt y: 3.0
```
您可以通过以嵌套方式传递该计算中涉及的所有可训练变量（例如可以是列表或字典）来计算某些变量的梯度。返回的梯度将遵循输入传递到`tape`相同的嵌套结构。如果我们分别计算上述代码中关于 `x`和`y`的梯度，会发生什么？
```python
with tf.GradientTape() as tape:
    z = x * y

try:
    dx = tape.gradient(z, x)
    dy = tape.gradient(z, y)

    print(f"Gradient of z wrt x: {dx}")
    print(f"Gradient of z wrt y: {dy}")
except Exception as ex:
    print("ERROR! ERROR! ERROR!\n")
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
ERROR! ERROR! ERROR!
RuntimeError A non-persistent GradientTape can only be used tocompute one set of gradients (or jacobians)
```
一旦调用`GradientTape.gradient(...)`，`GradientTape`所持有的所有资源都会被释放。因此，如果您计算了一次梯度，那么您将无法再次调用它。解决方案是将持久参数设置为`True`。这允许多次调用`gradient()`方法，因为当`tape`对象被垃圾收集时资源被释放。
```python
# Set the persistent argument
with tf.GradientTape(persistent=True) as tape:
    z = x * y

try:
    dx = tape.gradient(z, x)
    dy = tape.gradient(z, y)

    print(f"Gradient of z wrt x: {dx}")
    print(f"Gradient of z wrt y: {dy}")
except Exception as ex:
    print("ERROR! ERROR! ERROR!\n")
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
Gradient of z wrt x: 4.0
Gradient of z wrt y: 3.0
```
如果一个变量不可训练，会如何？
```python
# What if one of the Variables is non-trainable?
# Let's make y non-trainable in the above example and run
# the computation again

x = tf.Variable(3.0)
y = tf.Variable(4.0, trainable=False)

with tf.GradientTape() as tape:
    z = x * y
    
dx, dy = tape.gradient(z, [x, y])

print(f"Variable x: {x}")
print(f"Is x trainable?: {x.trainable}")
print(f"\nVariable y: {y}")
print(f"Is y trainable?: {y.trainable}\n")

print(f"Gradient of z wrt x: {dx}")
print(f"Gradient of z wrt y: {dy}")
```
结果输出为：
```bash
Variable x: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.0>
Is x trainable?: True

Variable y: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>
Is y trainable?: False

Gradient of z wrt x: 4.0
Gradient of z wrt y: None
```
{% note warning %}
**注意**：要记住的重要一点是，**切勿混合`AD`数据类型的拓扑和计算梯度**。当我说拓扑时，我的意思是不要混合使用`float、int、string`类型。事实上，您不能对数据类型为`int`或 `string`的任何操作采用梯度。
{% endnote %}

```python
# Note the dtypes
x = tf.Variable(3.0, dtype=tf.float32)
y = tf.Variable(4, dtype=tf.int32)

with tf.GradientTape() as tape:
    z = x * tf.cast(y, x.dtype)
    
dx, dy = tape.gradient(z, [x, y])

print(f"Input Variable x: {x}")
print(f"Input Variable y: {y}")
print(f"Output z: {z}\n")

print(f"Gradient of z wrt x: {dx}")
print(f"Gradient of z wrt y: {dy}")
```
结果输出为：
```bash
Input Variable x: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=3.0>
Input Variable y: <tf.Variable 'Variable:0' shape=() dtype=int32, numpy=4>
Output z: 12.0

Gradient of z wrt x: 4.0
Gradient of z wrt y: None
```
您可能会说您的代码是正确的，从某种意义上来说是正确的，但是您将得到`None`作为源和目标数据类型不同的变量的梯度值。在我们讨论一个非常重要的概念之前，让我们总结一下到目前为止我们学到的所有内容：
- `tf.GradientTape`是在`TensorFlow`中进行`AD`的`API`。
- 为了使用`Tape`计算梯度，我们需要：
    - 在`Tape`上下文中记录相关操作
    - 通过调用`GradientTape.gradient(...)`方法计算渐变
- 如果您希望多次调用`gradient(...)`方法，请确保将持久参数设置为`GradientTape`。
- 如果计算中涉及任何不可训练的变量，则该变量的梯度将为`None`。
- 混合数据类型拓扑是一个错误。

**精细增益控制**(`Fine-gain control`)
- 如何访问所有正在监视的对象？
- 如何停止梯度通过特定变量/路径的流动？
- 如果您不想查看`GradientTape`上下文中的所有变量怎么办？
- 如果您想观看上下文之外的内容怎么办？

我们将为上述每个案例举几个例子，以便更好地理解它。
##### 访问所有被监视的对象

```python
x = tf.Variable(3.0, name="x")
y = tf.Variable(4.0, name="y")
t = tf.Variable(tf.random.normal(shape=(2, 2)), name="t")

with tf.GradientTape() as tape:
    z = x * y

print("Tape is watching all of these:")
for var in tape.watched_variables():
    print(f"{var.name} and it's value is {var.numpy()}")
```
结果输出为：
```bash
Tape is watching all of these:
x:0 and it's value is 3.0
y:0 and it's value is 4.0
```
##### 停止梯度

```python
# The ugly way

x = tf.Variable(3.0, name="x")
y = tf.Variable(4.0, name="y")

with tf.GradientTape(persistent=True) as tape:
    z = x * y
    
    # Stop the grasdient flow
    with tape.stop_recording():
        zz = x*x + y*y

dz_dx, dz_dy = tape.gradient(z, [x, y])
dzz_dx, dzz_dy = tape.gradient(zz, [x, y])

print(f"Gradient of z wrt x: {dz_dx}")
print(f"Gradient of z wrt y: {dz_dy}\n")
print(f"Gradient of zz wrt x: {dzz_dx}")
print(f"Gradient of zz wrt y: {dzz_dy}")
```
结果输出为：
```bash
Gradient of z wrt x: 4.0
Gradient of z wrt y: 3.0

Gradient of zz wrt x: None
Gradient of zz wrt y: None
```
停止梯度流的更好方法是使用`tf.stop_gradient(...)`。为什么？
- 不需要访问`tape `
- 干净且具有更好的语义

```python
# The better way!
x = tf.Variable(3.0, name="x")
y = tf.Variable(4.0, name="y")

with tf.GradientTape() as tape:
    z = x * tf.stop_gradient(y)

dz_dx, dz_dy = tape.gradient(z, [x, y])
print(f"Gradient of z wrt x: {dz_dx}")
print(f"Gradient of z wrt y: {dz_dy}")
```
结果输出为：
```bash
Gradient of z wrt x: 4.0
Gradient of z wrt y: None
```
##### 选择上下文中要观看的内容

默认情况下，`GradientTape将`自动监视在上下文中访问的任何可训练变量，但如果您只需要选定变量的梯度，则可以通过将`watch_accessed_variables=False`传递给`tape`构造函数来禁用自动跟踪。
```python
# Both variables are trainable
x = tf.Variable(3.0, name="x")
y = tf.Variable(4.0, name="y")

# Telling the tape: Hey! I will tell you what to record.
# Don't start recording automatically!
with tf.GradientTape(watch_accessed_variables=False) as tape:
    # Watch x but not y
    tape.watch(x)
    z = x * y

dz_dx, dz_dy = tape.gradient(z, [x, y])
print(f"Gradient of z wrt x: {dz_dx}")
print(f"Gradient of z wrt y: {dz_dy}")
```
结果输出为：
```bash
Gradient of z wrt x: 4.0
Gradient of z wrt y: None
```
```python
# What if something that you wanted to watch,
# wasn't present in the computation done inside the context?

x = tf.Variable(3.0, name="x")
y = tf.Variable(4.0, name="y")
t = tf.Variable(5.0, name="t")

# Telling the tape: Hey! I will tell you what to record.
# Don't start recording automatically!
with tf.GradientTape(watch_accessed_variables=False) as tape:
    # Watch x but not y
    tape.watch(x)
    z = x * y
    
    # `t` isn't involved in any computation here
    # but what if we want to record it as well
    tape.watch(t)

print("Tape watching only these objects that you asked it to watch")
for var in tape.watched_variables():
    print(f"{var.name} and it's value is {var.numpy()}")
```
结果输出为：
```bash
Tape watching only these objects that you asked it to watch
x:0 and it's value is 3.0
t:0 and it's value is 5.0
```
##### Multiple Tapes

您可以使用多个`GradientTape`来记录不同的对象。
```python
x = tf.Variable(3.0, name="x")
y = tf.Variable(4.0, name="y")

with tf.GradientTape() as tape_for_x, tf.GradientTape() as tape_for_y:
    # Watching different variables with different tapes
    tape_for_x.watch(x)
    tape_for_y.watch(y)
    
    z = x * y

dz_dx = tape_for_x.gradient(z, x)
dz_dy = tape_for_y.gradient(z, y)
print(f"Gradient of z wrt x: {dz_dx}")
print(f"Gradient of z wrt y: {dz_dy}")
```
结果输出为：
```bash
Gradient of z wrt x: 4.0
Gradient of z wrt y: 3.0
```
##### 高阶梯度

在`GradientTape`上下文中完成的任何计算都会被记录。如果计算涉及**梯度计算**，它也会被记录下来。这使得使用相同的`API`可以轻松计算**高阶梯度**。
```python
x = tf.Variable(3.0, name="x")

with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape0:
        y = x * x * x
    first_order_grad = tape0.gradient(y, x)
second_order_grad = tape1.gradient(first_order_grad, x)

print(f"Variable x: {x.numpy()}")
print("\nEquation is y = x^3")
print(f"First Order Gradient wrt x (3 * x^2): {first_order_grad}")
print(f"Second Order Gradient wrt x (6^x): {second_order_grad}")
```
结果输出为：
```bash
Variable x: 3.0
Equation is y = x^3
First Order Gradient wrt x (3 * x^2): 27.0
Second Order Gradient wrt x (6^x): 18.0
```
##### 陷阱（Gotchas）

我们已经知道`int`或`string`数据类型的梯度未定义。
```python
# What happens when you tries to take gradient wrt a Tensor?
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    y = x * x
    
dy_dx = tape.gradient(y, x)

print(x)
print("\nGradient of y wrt x: ", dy_dx)

# tf.Tensor(3.0, shape=(), dtype=float32)
# Gradient of y wrt x:  None

# Let's modify the above code a bit
x = tf.constant(3.0)

with tf.GradientTape() as tape:
    tape.watch(x)
    y = x * x
    
dy_dx = tape.gradient(y, x)

print(x)
print("\nGradient of y wrt x: ", dy_dx)

# tf.Tensor(3.0, shape=(), dtype=float32)
# Gradient of y wrt x:  tf.Tensor(6.0, shape=(), dtype=float32)
```
##### 状态和梯度

`GradientTape`只能从当前状态读取，而不能从它的历史记录中读取。状态阻止梯度计算回溯到更远的地方。
```python
x = tf.Variable(3.0)
y = tf.Variable(4.0)

with tf.GradientTape() as tape:
    # Change the state of x by making x = x + y
    x.assign_add(y)
    
    # Let's do some computation e.g z = x * x 
    # This is equivalent to z = (x + y) * (x + y) because of above assign_add
    z = x * x
    
dy = tape.gradient(z, y)
print("Gradients of z wrt y: ", dy)
```
结果输出为：
```bash
Gradients of z wrt y:  None

```