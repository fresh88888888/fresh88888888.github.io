---
title: JAX（PRNG）
date: 2024-03-29 10:10:32
tags:
  - AI
categories:
  - 人工智能
---

#### 什么是PRNG？

如果我们按照定义，那么**伪随机数生成**是通过算法生成随机数序列的过程，使得生成的随机数的属性近似于从分布中采样的随机数序列的属性。当我们说随机时，意味着预测这个序列的概率并不比随机猜测更好。尽管我们担心这里的随机性，但伪随机数生成并不是真正的随机过程。为什么？因为序列是由提供给算法的初始值或初始状态决定的。用于生成这些随机数序列的算法称为**伪随机数生成器**（`Pseudo Random Number Generator`）。
<!-- more -->
#### PRNG有何用途？

`PRNG`有很多用途，但最有趣的是密码学、模拟、游戏、数据科学和机器学习等。您可能已经注意到，大多数人在他们的数据科学和机器学习工作流程中埋下了**种子**。种子被作为初始值！

#### 为什么要关心PRNG？

当我们设置种子时，我们试图解决的是**可重复性**问题。尽管可重复性取决于很多因素。我们在机器学习工作中处理随机状态的频率比我们想象的要多。例如，将数据集分为训练集和验证集，从神经网络中的给定分布中对隐藏层的权重进行采样，从高斯分布中采样噪声向量等。因此，当我们在这种情况下说可重复性时，什么是可重复性的？无论我运行同一个进程多少次，我都应该得到相同的随机数序列。这就是为什么埋下种子变得很重要。注意：埋下种子并不能解决工作流程的可重复性危机，它只是确保它的第一步。让我们举个例子来看一下可重复性的问题！

#### Numpy中的随机数
```python
import jax
from jax import jit
import jax.numpy as jnp

# If I set the seed, would I get the same sequence of random numbers every time?
for i in range(10):
    # Set initial value by providing a seed value
    seed = 0 
    np.random.seed(seed)
    
    # Generate a random integer from a range of [0, 5)
    random_number = np.random.randint(0, 5)
    print(f"Seed: {seed} -> Random number generated: {random_number}")

# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
# Seed: 0 -> Random number generated: 4
```
让我们举一个有点复杂的例子。我们将获取一个数组并将其拆分为两个数组。
```python
# Array of 10 values
array = np.arange(10)

for i in range(5):
    # Set initial value by providing a seed value
    seed = 1234
    np.random.seed(seed)
    
    # Choose array1 and array2 indices
    array_1_idx = np.random.choice(array, size=8)
    array_2_idx = np.random.choice(array, size=2)
    
    # Split the array into two sets
    array_1 = array[array_1_idx]
    array_2 = array[array_2_idx]
    
    print(f"Iteration: {i+1}  Seed value: {seed}\n")
    print(f"First array: {array_1}  Second array: {array_2}")
    print("="*50)
    print("")
```
结果输出为：
```bash
Iteration: 1  Seed value: 1234

First array: [3 6 5 4 8 9 1 7]  Second array: [9 6]
==================================================

Iteration: 2  Seed value: 1234

First array: [3 6 5 4 8 9 1 7]  Second array: [9 6]
==================================================

Iteration: 3  Seed value: 1234

First array: [3 6 5 4 8 9 1 7]  Second array: [9 6]
==================================================

Iteration: 4  Seed value: 1234

First array: [3 6 5 4 8 9 1 7]  Second array: [9 6]
==================================================

Iteration: 5  Seed value: 1234

First array: [3 6 5 4 8 9 1 7]  Second array: [9 6]
==================================================
```
{% note warning %}
**注意**：我们上面看到的是在`numpy`中生成随机数序列的传统方法。它使用`numpy RandomState(...)`提供的原有生成器。但这也是使用最广泛的一种。还有另一个函数（首选）`np.random.default_rng()`使用默认的`BitGenerator`来生成随机序列。
{% endnote %}

让我们使用`default_rng(...)`重复上面的示例。因为这是一个不同的`RNG`。
```python
# Array of 10 values
array = np.arange(10)

# Same example but with a different kind of random number generator
for i in range(5):
    # Set initial value by providing a seed value
    seed = 0
    rng = np.random.default_rng(seed)
    
    # Choose array1 and array2 indices
    array_1_idx = rng.choice(array, size=8)
    array_2_idx = rng.choice(array, size=2)
    
    # Split the array into two sets
    array_1 = array[array_1_idx]
    array_2 = array[array_2_idx]
    
    print(f"Iteration: {i+1}  Seed value: {seed}\n")
    print(f"First array: {array_1}  Second array: {array_2}")
    print("="*50)
    print("")
```
结果输出为：
```bash
Iteration: 1  Seed value: 0
First array: [8 6 5 2 3 0 0 0]  Second array: [1 8]
==================================================

Iteration: 2  Seed value: 0

First array: [8 6 5 2 3 0 0 0]  Second array: [1 8]
==================================================

Iteration: 3  Seed value: 0

First array: [8 6 5 2 3 0 0 0]  Second array: [1 8]
==================================================

Iteration: 4  Seed value: 0

First array: [8 6 5 2 3 0 0 0]  Second array: [1 8]
==================================================

Iteration: 5  Seed value: 0

First array: [8 6 5 2 3 0 0 0]  Second array: [1 8]
==================================================
```
##### Numpy PRNG：优点和缺点

我们看到了一些如何在`numpy`中生成伪随机数的示例。
##### 优点

- 从大多数用户的角度来看，设置全局种子很容易。设置一次即可完成。
- 使用新的生成器和`SeedSequencing`，可以跨多个进程生成可重复的伪随机数。
- **顺序等效保证**：`numpy`中随机数生成的好处之一是它确保了顺序等效保证。意味着无论您一次对包含`n`个元素的向量进行采样，还是一次对`n`个元素进行采样，最终的序列始终是相同的。让我们看看它的实际效果。

```python
# Set the seed
seed = 1234
np.random.seed(seed)

# Sample a vector of size 10 
array1 = np.random.randint(0, 10, size=10)

# Sample 10 elements one at a time
np.random.seed(seed)
array2 = np.stack([np.random.randint(0, 10) for _ in range(10)])

print(f"Sampled all at once    => {array1}")
print(f"Sampled one at a time  => {array2}")
```
结果输出为：
```bash
Sampled all at once    => [3 6 5 4 8 9 1 7 9 6]
Sampled one at a time  => [3 6 5 4 8 9 1 7 9 6]
```
##### 缺点

- 全局状态不利于**可重复性**：全局状态是有问题的，尤其是当您要在代码中实现某种并发性时。这就是为什么不再鼓励在`numpy`中设置全局种子的原始方法。
- 有了共享的全局状态，就很难推断它是如何在不同的线程、进程和设备之间使用和更新的，而且当熵产生和消耗的细节对最终用户隐藏时，很容易搞砸。
- 大多数`python`和`numpy`代码中使用的`Mersenne Twister PRNG`存在几个初始化问题。
- 当涉及并发时，`SeedSequencing`可以轻松获得可重复性的随机数序列，但它仍然不能用于`JAX`。

在我们转向`JAX PRNG`设计之前，让我们先看一个`SeedSequencing`的例子。
```python
def get_sequence(seed, size=5):
    rng = np.random.default_rng(seed)
    array = np.arange(10)
    return rng.choice(array, size=size)

# Instantiate SeedSequence
seed = 1234
ss = np.random.SeedSequence(seed)

# Spawn 2 child seed sequence
child_seeds = ss.spawn(2)

# Run the function a few times in parallel to check if we get
# same RNG sequence
for i in range(5):
    res = []
    for child_seed in child_seeds:
        res.append(delayed(get_sequence)(child_seed))
    res = Parallel(n_jobs=2)(res)
    print(f"Iteration: {i+1} Sequences: {res}")
    print("="*70)

# Iteration: 1 Sequences: [array([4, 5, 4, 2, 5]), array([7, 7, 7, 5, 1])]
# ======================================================================
# Iteration: 2 Sequences: [array([4, 5, 4, 2, 5]), array([7, 7, 7, 5, 1])]
# ======================================================================
# Iteration: 3 Sequences: [array([4, 5, 4, 2, 5]), array([7, 7, 7, 5, 1])]
# ======================================================================
# Iteration: 4 Sequences: [array([4, 5, 4, 2, 5]), array([7, 7, 7, 5, 1])]
# ======================================================================
# Iteration: 5 Sequences: [array([4, 5, 4, 2, 5]), array([7, 7, 7, 5, 1])]
# ======================================================================
```

#### JAX中的随机数

`JAX`中的`RNG`与`numpy`中的`RNG`有很大不同。人们自然想到的一个问题是：当`JAX`团队可以重用`numpy`中的相同代码库时，为什么他们要在`JAX`中实现全新的`PRNG`？我们举几个例子来回答这个问题，使用`numpy`代码的函数的执行由`Python`强制执行。假设`A`和`B`是两个函数。`A`和`B`的返回值被分配给`C`。因此，代码如下所示：`C = A() + B()`。
```python
# Global seed
np.random.seed(1234)

def A():
    return np.random.choice(["a", "A"])

def B():
    return np.random.choice(["b", "B"])

for i in range(2):
    C = A() + B()
    print(f"Iteration: {i+1}  C: {C}")

# Iteration: 1  C: AB
# Iteration: 2  C: aB
```
这里的执行定义的顺序。`A()`总是在`B()`之前调用。但是如果你在`JAX`中做同样的事情并`jit`，那么你不知道是先调用`A()`还是`B()`将首先被调用。
- `XLA`将按照最有效的顺序执行它们，不一定按照相同的顺序。还记得我们以前使用的`tf.control_dependency(...)`吗？`TensorFlow`没有任何问题，它只是指导编译器的一种方式。
- 如果强制执行顺序，则与`JAX`的原理相矛盾，即如果两个转换彼此独立，则它们的执行可以并行。

这看起来像是一场危机。如果您使用全局状态，您将无法推断首次调用哪个函数，因此生成的随机数序列是不可重现的。那有什么方法解决呢？

#### JAX中的RNG设计

为了确保我们可以并行转换，并且仍然获得可重现的结果，`JAX`应用了两个规则：
- 不依赖全局种子来生成随机序列。
- 随机函数应该显式地消耗一个状态（种子），这将确保这些函数在使用相同种子时会重现相同的结果。

{% note warning %}
注意：当人们在`PRNG`上下文中说状态、种子或密钥时，它们的意思是相同的。`JAX`使用“键”和“子键”一词比“种子”一词更频繁。为了与文档保持一致，我们将在这里使用相同的术语。
{% endnote %}

```python
from jax import random

# Define a state
seed = 1234
key = random.PRNGKey(1234)
key

# DeviceArray([   0, 1234], dtype=uint32)
```
因此，`key`是形状为(2, )的`DeviceArray`。然后将该`key`传递给随机函数。**随机函数消耗状态但不改变它**，这意味着如果您不断将相同的`key`传递给相同的函数，它将始终返回相同的输出。因为函数不会改变状态，所以每次我们调用一个新的随机函数时，我们都需要传递一个新的`key`。新的`key`是如何生成的？通过拆分原始`key`。看看下面的例子：
```python
# Passing the original key to a random function
random_integers = random.randint(key=key, minval=0, maxval=10, shape=[5])
print(random_integers)

# [2 4 9 9 4]

# What if we want to call another function?
# Don't use the same key. Split the original key, and then pass it
print("Original key: ", key)

# Split the key. By default the number of splits is set to 2
# You can specify explicitly how many splits you want to do
key, subkey = random.split(key, num=2)

print("New key: ",  key)
print("Subkey: ", subkey)

# Original key:  [   0 1234]
# New key:  [2113592192 1902136347]
# Subkey:  [603280156 445306386]

# Call another random function with the new key
random_floats = random.normal(key=key, shape=(5,), dtype=jnp.float32)
print(random_floats)

# [ 5.2179128e-01  1.4659788e-03 -5.9906763e-01 -3.9343226e-01 -1.9224551e+00]
```
注意：虽然我们称它们为`key`和`subkey`，但它们都是状态，您可以将它们中的任何一个传递给随机函数，或者`split`函数。

#### JAX PRNG：优点和缺点

现在我们已经了解了JAX中PRNG的设计以及它是如何实现和使用的，是时候讨论这种方法的优缺点了。

##### 优点

- `JAX PRNG`是基于计数器的`PRNG`设计，它使用`Threefry`哈希函数。这种设计允许`JAX`摆脱顺序执行约束，允许所有内容可向量化和可并行化，而无需放弃可重复性。
- 每个随机函数都会消耗状态但不会改变它。函数不必返回`key`。
- 分割方法是确定性的。因此，如果您从一个随机`key`开始，并将其拆分为代码中的n个`key`，可以放心，每次运行代码时，您都会得到相同的拆分。 
- 您可以一次从一个`key`生成n个`key`并不断传递它们。

```python
# Splitting is deterministic!
for i in range(3):
    key = random.PRNGKey(1234)
    print(f"Iteration: {i+1}\n")
    print(f"Original key: {key}")
    key, subkey = random.split(key)
    print(f"First subkey: {key}")
    print(f"Second subkey: {subkey}")
    print("="*50)
    print("")

# Iteration: 1

# Original key: [   0 1234]
# First subkey: [2113592192 1902136347]
# Second subkey: [603280156 445306386]
# ==================================================

# Iteration: 2

# Original key: [   0 1234]
# First subkey: [2113592192 1902136347]
# Second subkey: [603280156 445306386]
# ==================================================

# Iteration: 3

# Original key: [   0 1234]
# First subkey: [2113592192 1902136347]
# Second subkey: [603280156 445306386]
# ==================================================

# You can generate multiple keys at one go with one split
key = random.PRNGKey(111)
print(f"Original key: {key}\n")

subkeys = random.split(key, num=5)

for i, subkey in enumerate(subkeys):
    print(f"Subkey no: {i+1}  Subkey: {subkey}")

# Original key: [  0 111]
# Subkey no: 1  Subkey: [2149343144 3788759061]
# Subkey no: 2  Subkey: [1263116805 2203640444]
# Subkey no: 3  Subkey: [ 260051842 2161001049]
# Subkey no: 4  Subkey: [ 450316230 2080109636]
# Subkey no: 5  Subkey: [2532194002 3516360950]
```
##### 缺点

- `JAX`中`PRNG`设计只有在我们放弃顺序等效保证的情况下才有可能实现。由于该属性与矢量化不兼容，因此后者实际上是`JAX`的优先事项。
- 这本身并不是一个缺点，但用户很容易忘记这一点。这里需要考虑两件事：
    - 如果您使用相同的`key`一次又一次地调用一个函数，您将始终得到相同的输出。假设您想要从均匀分布中抽取`5`个随机数。如果将相同的`key`传递给采样函数，最终将得到5个重复的数字。
    - 如果将相同的`key`传递给不同的函数，在某些情况下您将得到高度相关的结果。用户在传递给使用随机函数的任何东西之前都应分割`key`。

```python
# No more Sequential Equivalent Guarantee unlike numpy

key = random.PRNGKey(1234)
random_integers_1 = random.randint(key=key, minval=0, maxval=10, shape=(5,))

key = random.PRNGKey(1234)
key, *subkeys = random.split(key, 5)
random_integers_2 = []

for subkey in subkeys:
    num = random.randint(key=subkey, minval=0, maxval=10, shape=(1,))
    random_integers_2.append(num)

random_integers_2 = np.stack(random_integers_2, axis=-1)[0]

print("Generated all at once: ", random_integers_1)
print("Generated sequentially: ", random_integers_2)

# Generated all at once:  [2 4 9 9 4]
# Generated sequentially:  [1 5 8 7]

# Possible highly correlated outputs. 
# Not a very good example but serves the demonstration purpose
def sampler1(key):
    return random.uniform(key=key, minval=0, maxval=1, shape=(2,))

def sampler2(key):
    return 2 * random.uniform(key=key, minval=0, maxval=1, shape=(2,))

key = random.PRNGKey(0)
sample_1 = sampler1(key=key)
sample_2 = sampler2(key=key)

print("First sample: ", sample_1)
print("Second sample: ", sample_2)

# First sample:  [0.21629536 0.8041241 ]
# Second sample:  [0.43259072 1.6082482 ]
```
在numpy中尝试一下！
```python
def sampler1():
    return np.random.uniform(low=0, high=1, size=(2,))

def sampler2():
    return 2 * np.random.uniform(low=0, high=1, size=(2,))

np.random.seed(0)
sample_1 = sampler1()
sample_2 = sampler2()

print("First sample: ", sample_1)
print("Second sample: ", sample_2)

# First sample:  [0.5488135  0.71518937]
# Second sample:  [1.20552675 1.08976637]
```
您会看到，在`JAX`代码中，两个采样器的输出高度相关，而在`numpy`代码中我们没有获得相关性。除非您想要相同的输出，否则切勿通过将`key`传递给JAX中的不同随机函数来重复使用`key`。
