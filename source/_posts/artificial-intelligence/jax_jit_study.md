---
title: JAX（JIT）
date: 2024-03-29 14:10:32
tags:
  - AI
categories:
  - 人工智能
---

#### 什么是即时(JIT)编译？

如果我们按照`JIT`的定义，那么`JIT`就是在执行期间编译代码的一种方式。实现`JIT`编译器的系统通常会连续分析正在执行的代码，并识别代码中从编译或重新编译获得的加速将超过编译该代码的开销的部分。
<!-- more -->
#### JAX中的JIT

`JAX`使用`XLA`进行编译。`jax.jit(...)`进行即时编译并转换为普通的`JAX Python`函数，以便它们可以在`XLA`中更有效地执行。
```python
import os
import time
import requests
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.config import config

def apply_activation(x):
    return jnp.maximum(0.0, x)

def get_dot_product(W, X):
    return jnp.dot(W, X)

# Always use a seed
key = random.PRNGKey(1234)
W = random.normal(key=key, shape=[1000, 10000], dtype=jnp.float32)

# Never reuse the key
key, subkey = random.split(key)
X = random.normal(key=subkey, shape=[10000, 20000], dtype=jnp.float32)

# JIT the functions we have
dot_product_jit  = jit(get_dot_product)
activation_jit = jit(apply_activation)

for i in range(3):
    start = time.time()
    # Don't forget to use `block_until_ready(..)`
    # else you will be recording dispatch time only
    Z = dot_product_jit(W, X).block_until_ready()
    end = time.time()
    print(f"Iteration: {i+1}")
    print(f"Time taken to execute dot product: {end - start:.2f} seconds", end="")
    
    start = time.time()
    A = activation_jit(Z).block_until_ready()
    print(f", activation function: {time.time()-start:.2f} seconds")
```
结果输出为：
```bash
Iteration: 1
Time taken to execute dot product: 6.48 seconds, activation function: 0.05 seconds
Iteration: 2
Time taken to execute dot product: 3.17 seconds, activation function: 0.03 seconds
Iteration: 3
Time taken to execute dot product: 3.19 seconds, activation function: 0.03 seconds
```
将上面的示例分解为几个步骤，以详细了解幕后发生的情况:
- 我们定义了两个函数，即`get_dot_product(...)`和`apply_activation(...)`，前者对权重和输入进行点积，后者对先前的结果应用`relu`。
- 然后我们使用`jit(function_name)`定义了两个转换，并获得了函数的编译版本。
- 当您第一次使用指定的参数调用已编译的函数时，执行时间非常长。为什么？ 因为第一次调用是**预热阶段**。预热阶段只不过是JAX跟踪所花费的时间。根据输入，跟踪器将代码转换为中间语言 `jaxprs`，然后编译该语言在`XLA`中执行。
- 后续调用运行代码的编译版本。
{% note warning %}
注意：如果您使用其他对函数的`jit`版本进行基准测试，请首先进行预热以进行公平比较，否则您将在基准测试中包含编译时间。
{% endnote %}

在继续进一步讨论`JIT`转换之前，我们将在这里休息一下，首先要理解`jaxprs`的概念。

#### Jaxprs

`Jaxpr`是一种用于表示普通`Python`函数的中间语言。当您转换函数时，`Jaxpr`语言首先将函数转换为简单的静态类型中间表达式，然后将转换直接应用于`jaxpr`。
- `jaxpr`实例表示具有一个或多个类型化参数（输入变量）和一个或多个类型化结果的函数。
- 输入和输出具有类型并表示为抽象值。
- 并非所有`Python`程序都可以用`jaxprs`表示，但许多科学计算和机器学习程序可以。

`JAX`中的每个转换都会具体化为某种形式的`jaxpr`。如果您想了解`JAX`内部是如何工作的，或者如果您想了解`JAX`跟踪的结果，了解`jaxprs`很有用。让我们举几个例子来说明`jaxpr`是如何工作的。我们首先看看我们上面定义的函数是如何用`jaxpr`来表达的。
```python
# Make jaxpr for the activation function
print(jax.make_jaxpr(activation_jit)(Z))
```
结果输出为：
```bash
{ lambda  ; 
    a.let b = xla_call[ backend=None
                    call_jaxpr={ lambda ; a.let b = max 0.0 a in (b,) }
                    device=None
                    donated_invars=(False,)
                    name=apply_activation ] a in (b,) }
```
如何解释`jaxpr`？
- 第一行告诉您该函数接收一个参数`a`。
- 第二行告诉您，这将在`XLA`上执行，即`(0, a)`的最大值。
- 最后一行告诉您返回的输出。

让我们看一下应用点积的函数的`jaxpr`。

```python
# Make jaxpr for the activation function
print(jax.make_jaxpr(dot_product_jit)(W, X))
```
结果输出为：
```bash
{ lambda  ; a b.
  let c = xla_call[ backend=None
                    call_jaxpr={ lambda  ; a b.
                                 let c = dot_general[ dimension_numbers=(((1,), (0,)), ((), ()))
                                                      precision=None
                                                      preferred_element_type=None ] a b
                                 in (c,) }
                    device=None
                    donated_invars=(False, False)
                    name=get_dot_product ] a b in (c,) }
```
与上面类似：
- 第一行告诉函数接收两个输入变量`a`和`b`，对应于我们的`W`和`X`
- 第二行是`XLA`调用，我们在其中执行点操作。（检查点积使用的尺寸）
- 最后一行是要返回的结果，用`c`表示

我们再举一个有趣的例子:
```python
# We know that `print` introduces but impurity but it is
# also very useful to print values while debugging. How does
# jaxprs interpret that?

def number_squared(num):
    print("Received: ", num)
    return num ** 2

# Compiled version
number_squared_jit = jit(number_squared)

# Make jaxprs
print(jax.make_jaxpr(number_squared_jit)(2))
```
结果输出为：
```bash
Received:  Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/1)>
{ lambda  ; a.
  let b = xla_call[ backend=None
                    call_jaxpr={ lambda  ; a.
                                 let b = integer_pow[ y=2 ] a
                                 in (b,) }
                    device=None
                    donated_invars=(False,)
                    name=number_squared ] a in (b,) }
```
注意`print`语句中的`num`是如何被追踪的。没有什么可以阻止您运行不纯的函数，但您应该准备好遇到此类副作用。事实上，打印语句在第一次调用时被跟踪，但在后续调用中可能不会被跟踪，这是因为您的`python`代码将至少运行一次。让我们看看它的实际效果。
```python
# Subsequent calls to the jitted function
for i, num in enumerate([2, 4, 8]):
    print("Iteration: ", i+1)
    print("Result: ", number_squared_jit(num))
    print("="*50)
```
结果输出为：
```bash
Iteration:  1
Received:  Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>
Result:  4
==================================================
Iteration:  2
Result:  16
==================================================
Iteration:  3
Result:  64
==================================================
```
```python
squared_numbers = []

# An impure function (using a global state)
def number_squared(num):
    global squared_numbers
    squared = num ** 2
    squared_numbers.append(squared)
    return squared

# Compiled verison
number_squared_jit = jit(number_squared)

# Make jaxpr
print(jax.make_jaxpr(number_squared_jit)(2))
```
结果输出为：
```bash
{ lambda ; a:i32[]. let
    b:i32[] = pjit[
      name=number_squared
      jaxpr={ lambda ; c:i32[]. let d:i32[] = integer_pow[y=2] c in (d,) }
    ] a
  in (b,) }
```
有几点需要注意：
- 第一行统计数据与往常一样，显示我们有一个输入变量`a`，对应于`num`参数。
- 第二行是`XLA`调用，用于对输入数字进行平方。
- 最后一行返回由`b`表示的`XLA`调用的结果。

`jaxp`未捕获副作用。`jaxpr`依赖于跟踪。任何转换函数的行为都取决于跟踪值。您可能会注意到第一次运行时的副作用，但不一定会注意到后续调用的副作用。因此，在这种情况下，`jaxpr`甚至不关心全局列表。
{% note warning %}
**注意**：需要注意的另一件重要的事情是`jaxprs`中的设备值。尽管除非您在jit转换期间指定了设备（如`jit(fn_name, device=)`），否则该参数就存在，但此处不会列出任何设备。有时这可能会令人困惑，因为计算将在某些加速器上运行，但这里不会反映设备名称。这背后的逻辑是`jaxpr`只是一个表达式，独立于它要运行的逻辑。它更关心`XLA`的布局，而不是运行表达式的设备。
{% endnote %}

```python
# Subsequent calls to the jitted function
for i, num in enumerate([4, 8, 16]):
    print("Iteration: ", i+1)
    print("Result: ", number_squared_jit(num))
    print("="*50)
    
# What's in the list?
print("\n Results in the global list")
squared_numbers
```
结果输出为：
```bash
Iteration:  1
Result:  16
==================================================
Iteration:  2
Result:  64
==================================================
Iteration:  3
Result:  256
==================================================

Results in the global list
[Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=1/1)>,
 Traced<ShapedArray(int32[], weak_type=True)>with<DynamicJaxprTrace(level=0/1)>]
```
您可能想知道如果副作用出现在第一次调用时，为什么全局列表中有两个跟踪值。原因是副作用可能会、也可能不会出现在后续调用中。这是一种不可预测的行为。

#### JIT有多大价值？

在深入研究与`JIT`相关的细微差别之前，我们假设有两个可以执行`jit`的函数，例如，我们的`get_dot_product(...)`和`apply_activation(..)`函数。您应该将它们两者都合并起来，还是应该将它们运用到一个函数/模块中并且合并该函数/模块？
```python
# Calling the two functions into a single function
# so that we can jit this function instead of jitting them
def forward_pass(W, X):
    Z = get_dot_product(W, X)
    A = apply_activation(Z)
    return Z, A

# Always use a seed
key = random.PRNGKey(1234)

# We will use much bigger array this time
W = random.normal(key=key, shape=[2000, 10000], dtype=jnp.float32)

# Never reuse the key
key, subkey = random.split(key)
X = random.normal(key=subkey, shape=[10000, 20000], dtype=jnp.float32)

# JIT the functions we have individually
dot_product_jit  = jit(get_dot_product)
activation_jit = jit(apply_activation)

# JIT the function that wraps both the functions
forward_pass_jit = jit(forward_pass)

for i in range(3):
    start = time.time()
    # Don't forget to use `block_until_ready(..)`
    # else you will be recording dispatch time only
    Z = dot_product_jit(W, X).block_until_ready()
    end = time.time()
    print(f"Iteration: {i+1}")
    print(f"Time taken to execute dot product: {end - start:.2f} seconds", end="")
    
    start = time.time()
    A = activation_jit(Z).block_until_ready()
    print(f", activation function: {time.time()- start:.2f} seconds")
    
    # Now measure the time with a single jitted function that calls
    # the other two functions
    Z, A = forward_pass_jit(W, X)
    Z, A = Z.block_until_ready(), A.block_until_ready()
    print(f"Time taken by the forward pass function: {time.time()- start:.2f} seconds")
    print("")
    print("="*50)
```
结果输出为：
```bash
Iteration: 1
Time taken to execute dot product: 8.83 seconds, activation function: 0.08 seconds
Time taken by the forward pass function: 6.30 seconds

==================================================
Iteration: 2
Time taken to execute dot product: 6.16 seconds, activation function: 0.06 seconds
Time taken by the forward pass function: 6.54 seconds

==================================================
Iteration: 3
Time taken to execute dot product: 6.12 seconds, activation function: 0.06 seconds
Time taken by the forward pass function: 6.17 seconds

==================================================
```
遵循哪种方法？这取决于你。另外，我无法确认第二种方法是否总是有效?

#### JIT和Python控制流

在这个阶段，我们自然会想到一个问题：为什么我们不直接`JIT`一切？这将在执行方面带来巨大的收益。虽然在某种意义上是正确的，但你不能搞砸一切。在某些情况下，`jitting`无法开箱即用。 让我们举几个例子来理解这一点：
```python
def square_or_cube(x):
    if x % 2 == 0:
        return x ** 2
    else:
        return x * x * x

# JIT transformation
square_or_cube_jit = jit(square_or_cube)

# Run the jitted version on some sample data
try:
    val = square_or_cube_jit(2)
except Exception as ex:
    print(type(ex).__name__, ex)
```
那么为什么这段代码不起作用呢？让我们再次分解一下`JIT`的整个流程：
- 当我们`jit`一个函数时，我们的目标是获得该函数的编译版本，以便我们可以针对不同的参数值缓存和重用编译后的代码。
- 为了实现这一点，`JAX`在可能输入集的抽象值上进行跟踪。
- 跟踪期间使用不同级别的抽象，用于特定函数跟踪的抽象类型取决于所完成的转换类型。
- 默认情况下，`jit`在`ShapedArray`抽象级别上跟踪代码，其中每个抽象值具有固定形状和数据类型的所有数组值的集合。例如，如果我们使用抽象值`ShapedArray((3,), jnp.float32)`进行跟踪，我们会得到一个函数视图，该函数可以重用于数组集中的任何具体值。这意味着我们可以节省编译时间。

回到上面的代码以及它失败的原因，在这种情况下，`x`的值在跟踪时并不具体。因此，当我们遇到像`if x % 2 == 0`这样的行时，表达式`x % 2`的计算结果表示集合`{True, False}`的抽象 `ShapedArray((), jnp.bool_)`。当`Python`将其强制为`True`或`False`时，我们会收到错误：我们不知道要采用哪个分支，并且无法继续跟踪！让我们再举一个例子，这次涉及一个循环:
```python
def multiply_n_times(x, n):
    count = 0
    res = 1
    while count < n:
        res = res * x
        count +=1 
    return x

try:
    val = jit(multiply_n_times)(2, 5)
except Exception as ex:
    print(type(ex).__name__, ex)
```
如果循环内的计算非常昂贵，仍然可以抛弃昂贵的计算部分
```python
# Jitting the expensive computational part
def multiply(x, i):
    return x * i

# Specifying the static args
multiply_jit = jit(multiply, static_argnums=0)

# Leaving it as it as
def multiply_n_times(x, n):
    count = 0
    res = 1
    while count < n:
        res = multiply_jit(x, res)
        count += 1
    return res

# 236 µs ± 20.9 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
```
#### 缓存

当你`jit`一个函数时，它会在第一次调用时被编译。对`jitted`函数的任何后续调用都会复用缓存的代码。如果我们需要`JIT`一个对输入值有条件的函数，我们可以通过指定`static_argnums`告诉`JAX`为特定输入创建一个不太抽象的跟踪器。这样做的代价是生成的`jaxpr`不太灵活，因此`JAX`必须指定输入的每个新值重新编译该函数。保证函数获得有限的不同值，这才是一个好的策略。这样做可以有效地在每次调用时创建一个新的`jit`转换对象，每次都会对其进行编译，而不是复用相同的**缓存函数**。

{% note warning %}
**注意**：如果输入的形状发生变化，在这种情况下也会发生重新编译。例如，如果您的批量大小发生变化，那么在这种情况下它将重新编译该函数。
{% endmote %}
