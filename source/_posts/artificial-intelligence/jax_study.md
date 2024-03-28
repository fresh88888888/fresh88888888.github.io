---
title: JAX（DeviceArray）
date: 2024-03-28 10:10:32
tags:
  - AI
categories:
  - 人工智能
---

`JAX`是一个**高性能机器学习库**。`JAX`在加速器（例如`GPU`和`TPU`）上编译并运行`NumPy`代码。您可以使用`JAX`（以及为`JAX`构建的神经网络库`FLAX`）来构建和训练深度学习模型。
<!-- more -->
#### JAX是什么？

**`JAX`**是一个特别适合机器学习研究的框架。关于`JAX`的几点：
- 它就像`numpy`一样，但使用编译器(`XLA`)编译本机`Numpy`代码，并在加速器(`GPU/TPU`)上运行。
- 对于**自动微分**，`JAX`使用`Autograd`。它自动区分原生`Python`和`Numpy`代码。
- `JAX`用于将数值程序表示为组合，但具有某些约束，例如`JAX`转换和编译设计为仅适用于纯函数的`Python`函数。如果一个函数在使用相同的参数调用时始终返回相同的值，则该函数是**纯函数**，并且该函数没有副作用，例如，改变非局部变量的状态。
- 就语法而言，`JAX`与`numpy`非常相似，但您应该注意一些细微的差异。

让我们举几个例子来看看`JAX`的实际应用！
```python
import time

import jax
import numpy as np
import jax.numpy as jnp
from jax import random

# We will create two arrays, one with numpy and other with jax
# to check the common things and the differences
array_numpy = np.arange(10, dtype=np.int32)
array_jax = jnp.arange(10, dtype=jnp.int32)

print("Array created using numpy: ", array_numpy)
print("Array created using JAX: ", array_jax)

# What types of array are these?
print(f"array_numpy is of type : {type(array_numpy)}")
print(f"array_jax is of type : {type(array_jax)}")
```
结果输出为：
```bash
Array created using numpy:  [0 1 2 3 4 5 6 7 8 9]
Array created using JAX:  [0 1 2 3 4 5 6 7 8 9]

array_numpy is of type : <class 'numpy.ndarray'>
array_jax is of type : <class 'jaxlib.xla_extension.DeviceArray'>
```
`array_numpy`是`ndarray`对象，而`array_jax`是`DeviceArray`对象。

#### DeviceArray

有关`DeviceArray`的几点：
- 它是`JAX`数组对象的核心，与`ndarray`类似，但有细微的差别。
- 与`ndarray`不同，`DeviceArray`由单个设备（`CPU/GPU/TPU`）上的内存缓冲区支持。
- 它与设备无关，即`JAX`不需要跟踪阵列所在的设备，并且可以避免数据传输。
- 由于它与设备无关，因此可以轻松在`CPU、GPU`或`TPU`上运行相同的`JAX`代码，而无需更改代码。
- `DeviceArray`是惰性的，即`JAX DeviceArray`的值不会立即可用，并且仅在请求时才拉取。
- 尽管`DeviceArray`是惰性的，您仍然可以执行诸如检查`DeviceArray`的形状或类型之类的操作，而无需等待生成它的计算完成。我们甚至可以将其传递给另一个`JAX`计算。

**延迟计算和与设备无关**的两个属性为`DeviceArray`提供了巨大的优势。

#### Numpy vs JAX-numpy

`jax numpy`在`API`方面与`numpy`非常相似。您在`numpy`中执行的大多数操作也可以在`jax numpy`中使用，具有类似的语义。我只是列出了一些操作来展示这一点，但还有更多操作。注意：并非所有`Numpy`函数都在`JAX numpy`中实现。
```python
# Find the max element. Similarly you can find `min` as well
print(f"Maximum element in ndarray: {array_numpy.max()}")
print(f"Maximum element in DeviceArray: {array_jax.max()}")

# Reshaping
print("Original shape of ndarray: ", array_numpy.shape)
print("Original shape of DeviceArray: ", array_jax.shape)

array_numpy = array_numpy.reshape(-1, 1)
array_jax = array_jax.reshape(-1, 1)

print("\nNew shape of ndarray: ", array_numpy.shape)
print("New shape of DeviceArray: ", array_jax.shape)

# Absoulte pairwise difference
print("Absoulte pairwise difference in ndarray")
print(np.abs(array_numpy - array_numpy.T))

print("\nAbsoulte pairwise difference in DeviceArray")
print(jnp.abs(array_jax - array_jax.T))

# Are they equal?
print("\nAre all the values same?", end=" ")
print(jnp.alltrue(np.abs(array_numpy - array_numpy.T) == jnp.abs(array_jax - array_jax.T)))

# Matrix multiplication
print("Matrix multiplication of ndarray")
print(np.dot(array_numpy, array_numpy.T))

print("\nMatrix multiplication of DeviceArray")
print(jnp.dot(array_jax, array_jax.T))
```
结果输出为：
```bash
Maximum element in ndarray: 9
Maximum element in DeviceArray: 9

Original shape of ndarray:  (10,)
Original shape of DeviceArray:  (10,)
New shape of ndarray:  (10, 1)
New shape of DeviceArray:  (10, 1)

Absoulte pairwise difference in ndarray
[[0 1 2 3 4 5 6 7 8 9]
 [1 0 1 2 3 4 5 6 7 8]
 [2 1 0 1 2 3 4 5 6 7]
 [3 2 1 0 1 2 3 4 5 6]
 [4 3 2 1 0 1 2 3 4 5]
 [5 4 3 2 1 0 1 2 3 4]
 [6 5 4 3 2 1 0 1 2 3]
 [7 6 5 4 3 2 1 0 1 2]
 [8 7 6 5 4 3 2 1 0 1]
 [9 8 7 6 5 4 3 2 1 0]]
Absoulte pairwise difference in DeviceArray
[[0 1 2 3 4 5 6 7 8 9]
 [1 0 1 2 3 4 5 6 7 8]
 [2 1 0 1 2 3 4 5 6 7]
 [3 2 1 0 1 2 3 4 5 6]
 [4 3 2 1 0 1 2 3 4 5]
 [5 4 3 2 1 0 1 2 3 4]
 [6 5 4 3 2 1 0 1 2 3]
 [7 6 5 4 3 2 1 0 1 2]
 [8 7 6 5 4 3 2 1 0 1]
 [9 8 7 6 5 4 3 2 1 0]]
Are all the values same? True

Matrix multiplication of ndarray
[[ 0  0  0  0  0  0  0  0  0  0]
 [ 0  1  2  3  4  5  6  7  8  9]
 [ 0  2  4  6  8 10 12 14 16 18]
 [ 0  3  6  9 12 15 18 21 24 27]
 [ 0  4  8 12 16 20 24 28 32 36]
 [ 0  5 10 15 20 25 30 35 40 45]
 [ 0  6 12 18 24 30 36 42 48 54]
 [ 0  7 14 21 28 35 42 49 56 63]
 [ 0  8 16 24 32 40 48 56 64 72]
 [ 0  9 18 27 36 45 54 63 72 81]]

Matrix multiplication of DeviceArray
[[ 0  0  0  0  0  0  0  0  0  0]
 [ 0  1  2  3  4  5  6  7  8  9]
 [ 0  2  4  6  8 10 12 14 16 18]
 [ 0  3  6  9 12 15 18 21 24 27]
 [ 0  4  8 12 16 20 24 28 32 36]
 [ 0  5 10 15 20 25 30 35 40 45]
 [ 0  6 12 18 24 30 36 42 48 54]
 [ 0  7 14 21 28 35 42 49 56 63]
 [ 0  8 16 24 32 40 48 56 64 72]
 [ 0  9 18 27 36 45 54 63 72 81]]
```
现在，让我们看一下在`Numpy`中可以执行但在`Jax-numpy`中不能执行的一些操作，反之亦然。

#### 不变性（Immutability）

`JAX`数组是不可变的，就像`TensorFlow`张量一样。这意味着，`JAX`数组不支持像`ndarray`中那样的项目分配。
```python
array1 = np.arange(5, dtype=np.int32)
array2 = jnp.arange(5, dtype=jnp.int32)

print("Original ndarray: ", array1)
print("Original DeviceArray: ", array2)

# Item assignment
array1[4] = 10
print("\nModified ndarray: ", array1)
print("\nTrying to modify DeviceArray-> ", end=" ")

try:
    array2[4] = 10
    print("Modified DeviceArray: ", array2)
except Exception as ex:
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
Original ndarray:  [0 1 2 3 4]
Original DeviceArray:  [0 1 2 3 4]
Modified ndarray:  [ 0  1  2  3 10]

Trying to modify DeviceArray->  TypeError '<class 'jaxlib.xla_extension.DeviceArray'>' object does not support item assignment. JAX arrays are immutable; perhaps you want jax.ops.index_update or jax.ops.index_add instead?
```
这种情况与我们使用`TensorFlow Tensors`时的情况完全相同。与`TensorFlow`中的`tf.tensor_scatter_nd_update`类似，我们有索引更新运算符（之前曾经有`jax.ops.index_update(..)`但现在已弃用）。语法非常简单，例如`DeviceArray.at[idx].op(val)`。但这不会修改原始数组，而是返回一个新数组，其中元素已按指定更新一个自然而然地浮现在脑海中的问题？为什么是不变性？问题是`JAX`依赖于纯函数。允许项目分配或就地更新与该理念相反。但是为什么`TF`张量是不可变的，因为它不需要纯函数？如果您要对`DAG`进行任何优化，强烈建议避免更改计算中使用的操作的状态，以避免任何副作用。
```python
# Modifying DeviceArray elements at specific index/indices
array2_modified = array2.at[4].set(10)

# Equivalent => array2_modified = jax.ops.index_update(array2, 4, 10)
print("Original DeviceArray: ", array2)
print("Modified DeviceArray: ", array2_modified)

# Of course, updates come in many forms!
print(array2.at[4].add(6))
print(array2.at[4].max(20))
print(array2.at[4].min(-1))

# Equivalent but depecated. Just to showcase the similarity to tf scatter_nd_update
print("\nEquivalent but deprecatd")
print(jax.ops.index_add(array2, 4, 6))
print(jax.ops.index_max(array2, 4, 20))
print(jax.ops.index_min(array2, 4, -1))
```
#### 异步调度

`ndarrays`和`DeviceArrays`之间最大的区别之一在于它们的执行力和可用性。`JAX`使用异步调度来隐藏`Python`开销。
```python
# Create two random arrays sampled from a uniform distribution
array1 = np.random.uniform(size=(8000, 8000)).astype(np.float32)
array2 = jax.random.uniform(jax.random.PRNGKey(0), (8000, 8000), dtype=jnp.float32) # More on PRNGKey later!
print("Shape of ndarray: ", array1.shape)
print("Shape of DeviceArray: ", array2.shape)

# Shape of ndarray:  (8000, 8000)
# Shape of DeviceArray:  (8000, 8000)
```
现在，让我们对每个数组进行一些计算，看看会发生什么以及每个计算需要多少时间。
```python
# Dot product on ndarray
start_time = time.time()
res = np.dot(array1, array1)
print(f"Time taken by dot product op on ndarrays: {time.time()-start_time:.2f} seconds")

# Dot product on DeviceArray
start_time = time.time()
res = jnp.dot(array2, array2)
print(f"Time taken by dot product op on DeviceArrays: {time.time()-start_time:.2f} seconds")

# Time taken by dot product op on ndarrays: 7.95 seconds
# Time taken by dot product op on DeviceArrays: 0.02 seconds
```
看来`DeviceArray`计算很快就完成了。
- 与`ndarray`的结果不同，`DeviceArray`上完成的计算结果尚不可用。这是加速器上可用的未来值。
- 您可以通过打印或将其转换为普通的旧`numpy ndarray`来检索此计算的值。
- `DeviceArray`的计时是调度工作所花费的时间，而不是实际计算所花费的时间。
- 异步调度很有用，因为它允许`Python`代码“运行在加速器设备之前”，从而使`Python`代码远离关键路径。如果`Python`代码在设备上排队的速度比执行速度快，并且`Python`代码实际上不需要检查主机上计算的输出，则`Python`程序可以将任意数量的工作排队并避免使用加速器等待。
- 要衡量任何此类操作的真实成本：将其转换为普通`numpy ndarray`（不推荐）;使用`block_until_ready()`等待它的计算完成（基准测试的首选方法）。

#### Types promotion

这是需要牢记的另一个方面。与`numpy`相比，`JAX`中的`dtype`提升不那么激进：
- 在提升`Python`标量时，`JAX`始终更喜欢`JAX`值的精度。
- 在针对浮点或复杂类型提升整数或布尔类型时，`JAX`始终优先选择浮点或复杂类型。
- `JAX`使用更适合`GPU/TPU`等现代加速器设备的浮点提升规则。

```python
print("Types promotion in numpy =>", end=" ")
print((np.int8(32) + 4).dtype)

print("Types promtoion in JAX =>", end=" ")
print((jnp.int8(32) + 4).dtype)

# Types promotion in numpy => int64
# Types promtoion in JAX => int8

array1 = np.random.randint(5, size=(2), dtype=np.int32)
print("Implicit numpy casting gives: ", (array1 + 5.0).dtype)

# Check the difference in semantics of the above function in JAX
array2 = jax.random.randint(jax.random.PRNGKey(0),
                            minval=0,
                            maxval=5,
                            shape=[2],
                            dtype=jnp.int32
                           )
print("Implicit JAX casting gives: ", (array2 + 5.0).dtype)

# Implicit numpy casting gives:  float64
# Implicit JAX casting gives:  float32
```
#### 自动微分（Automatic Differentiation）

自动微分是我最喜欢讨论的主题之一。与我们在`TensorFlow`中深入介绍`AD`的方式类似，这里我们将通过一个简单的示例来了解它与`JAX`的集成有多紧密。
```python
def squared(x):
    return x**2

x = 4.0
y = squared(x)

dydx = jax.grad(squared)
print("First order gradients of y wrt x: ", dydx(x))
print("Second order gradients of y wrt x: ", jax.grad(dydx)(x))
```
结果输出为：
```bash
First order gradients of y wrt x:  8.0
Second order gradients of y wrt x:  2.0
```