---
title: JAX（Autodiff）
date: 2024-03-31 16:20:32
tags:
  - AI
categories:
  - 人工智能
---

今天，我们将研究另一个重要概念**自动微分**。我们已经在`TensorFlow`中看到了**自动微分**。**自动微分**的想法在所有框架中都非常相似，但`IMO JAX`比所有框架都做得更好。
<!-- more -->
#### 梯度

`JAX`中的`grad`函数用于**计算梯度**。 我们知道`JAX`背后的基本思想是使用**函数组合**，`grad`也将**可调用对象**作为输入并返回可调用对象。因此，每当我们想要计算梯度时，我们需要首先将可调用对象传递给`grad`。让我们举个例子来更清楚地说明：
```python
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from jax import make_jaxpr
from jax import vmap, pmap, jit
from jax import grad, value_and_grad
from jax.test_util import check_grads

def product(x, y):
    z = x * y
    return z

x = 3.0
y = 4.0

z = product(x, y)

print(f"Input Variable x: {x}")
print(f"Input Variable y: {y}")
print(f"Product z: {z}\n")

# dz / dx
dx = grad(product, argnums=0)(x, y)
print(f"Gradient of z wrt x: {dx}")

# dz / dy
dy = grad(product, argnums=1)(x, y)
print(f"Gradient of z wrt y: {dy}")

# Input Variable x: 3.0
# Input Variable y: 4.0
# Product z: 12.0
# Gradient of z wrt x: 4.0
# Gradient of z wrt y: 3.0
```
让我们分解上面的例子并尝试逐步理解梯度计算。
- 我们有一个名为`Product(...)`的函数，它接受两个位置参数作为输入并返回这些参数的乘积。
- 我们将`Product(...)`函数传递给`grad`来计算梯度。`grad`中的`argnums`参数告诉`grad`区分函数与位置参数。因此，我们通过`0`和`1`来相应地计算`x`和`y`的梯度。

您还可以一次性计算函数值和梯度。为此，我们将使用`value_and_grad(...)`函数。
```python
z, dx = value_and_grad(product, argnums=0)(x, y)
print("Product z:", z)
print(f"Gradient of z wrt x: {dx}")

# Product z: 12.0
# Gradient of z wrt x: 4.0
```
#### Jaxprs 和 grad

由于我们可以在`JAX`中组合函数转换，因此我们可以从`grad`函数生成`jaxprs`来了解幕后发生的情况。举个例子：
```python
# Differentiating wrt first positional argument `x`
print("Differentiating wrt x")
print(make_jaxpr(grad(product, argnums=0))(x, y))


# Differentiating wrt second positional argument `y`
print("\nDifferentiating wrt y")
print(make_jaxpr(grad(product, argnums=1))(x, y))
```
结果输出为：
```bash
Differentiating wrt x
{ lambda ; a:f32[] b:f32[]. let _:f32[] = mul a b; c:f32[] = mul 1.0 b in (c,) }

Differentiating wrt y
{ lambda ; a:f32[] b:f32[]. let _:f32[] = mul a b; c:f32[] = mul a 1.0 in (c,) }
```
请注意，除我们要微分的`1`之外的参数是值为`1`的常数。

#### 停止梯度计算

有时我们不希望梯度流过特定计算中涉及的某些变量。在这种情况下，我们需要明确告诉`JAX`我们不希望梯度流经指定的变量集。稍后我们将研究这方面的复杂示例，但现在，我将修改我们的`Product(...)`函数，其中我们不希望梯度流经`y`。
```python
# Modified product function. Explicity stopping the
# flow of the gradients through `y`
def product_stop_grad(x, y):
    z = x * jax.lax.stop_gradient(y)
    return z

# Differentiating wrt y. This should return 0
grad(product_stop_grad, argnums=1)(x, y)

# DeviceArray(0., dtype=float32)
```
#### 每个样本的梯度

在**反向模式**下，仅为输出标量的函数定义梯度，例如反向传播损失值以更新机器学习模型的参数。损失始终是标量值。如果您的函数返回一个批次并且您想要计算该批次的每个样本的梯度该怎么办？
这些在`JAX`中非常简单。
- 编写一个接受输入应用`tanh`的函数。
- 我们将检查是否可以计算单个示例的梯度。
- 我们将传递一批输入并计算整批的梯度。

```python
def activate(x):
    """Applies tanh activation."""
    return jnp.tanh(x)

# Check if we can compute the gradients for a single example
grads_single_example = grad(activate)(0.5)
print("Gradient for a single input x=0.5: ", grads_single_example)

# Now we will generate a batch of random inputs, and will pass
# those inputs to our activate function. And we will also try to
# calculate the grads on the same batch in the same way as above
# Always use the PRNG
key = random.PRNGKey(1234)
x = random.normal(key=key, shape=(5,))
activations = activate(x)

print("\nTrying to compute gradients on a batch")
print("Input shape: ", x.shape)
print("Output shape: ", activations.shape)

try:
    grads_batch = grad(activate)(x)
    print("Gradients for the batch: ", grads_batch)
except Exception as ex:
    print(type(ex).__name__, ex)
```
结果输出为：
```bash
Gradient for a single input x=0.5:  0.7864477
Trying to compute gradients on a batch
Input shape:  (5,)
Output shape:  (5,)
TypeError Gradient only defined for scalar-output functions. Output had shape: (5,).
```
那么解决办法是什么呢？`vmap`和`pmap`是几乎所有问题的解决方案，让我们看看它的实际效果：
```python
grads_batch = vmap(grad(activate))(x)
print("Gradients for the batch: ", grads_batch)

# Gradients for the batch:  [0.48228705 0.45585024 0.99329686 0.0953269  0.8153717 ]
```
让我们分解一下我们上面为达到预期结果所做的所有修改。
- `grad(activate)(...)`适用于单个示例。
- 添加`vmap`组合为我们的输入和输出添加批量维度（默认为`0`）。

从单个示例到批量示例就这么简单，反之亦然。您所需要的只是专注于使用`vmap`。让我们看看这个转换的`jaxpr`是什么样子的。
```python
make_jaxpr(vmap(grad(activate)))(x)

# { lambda  ; a.
#   let b = tanh a
#       c = sub 1.0 b
#       d = mul 1.0 c
#       e = mul d b
#       f = add_any d e
#   in (f,) }
```
#### 其他变换的组合

我们可以将任何其他转换与`grad`组合起来。我们已经看到`vmap`与`grad`一起使用。让我们将jit应用到上面的转换中，以使其更加高效。
```python
jitted_grads_batch = jit(vmap(grad(activate)))

for _ in range(3):
    start_time = time.time()
    print("Gradients for the batch: ", jitted_grads_batch(x))
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("="*50)
    print()

# Gradients for the batch:  [0.48228705 0.45585027 0.99329686 0.09532695 0.8153717 ]
# Time taken: 0.03 seconds
# ==================================================

# Gradients for the batch:  [0.48228705 0.45585027 0.99329686 0.09532695 0.8153717 ]
# Time taken: 0.00 seconds
# ==================================================

# Gradients for the batch:  [0.48228705 0.45585027 0.99329686 0.09532695 0.8153717 ]
# Time taken: 0.00 seconds
# ==================================================
```
#### 验证有限差分

很多时候，我们想要用有限差分来验证梯度的计算，以再次检查我们所做的一切是否正确。因为这是处理导数时非常常见的健全性检查，所以JAX提供了一个方便的函数`check_grads`来检查任意阶梯度的有限差分。让我们来看看：
```python
try:
    check_grads(jitted_grads_batch, (x,),  order=1)
    print("Gradient match with gradient calculated using finite differences")
except Exception as ex:
    print(type(ex).__name__, ex)

# Gradient match with gradient calculated using finite differences
```
#### 高阶梯度

`grad`函数接受一个可调用函数作为输入并返回另一个函数。我们可以一次又一次地将变换返回的函数与`grad`组合起来，以计算任意阶的高阶导数。让我们举一个例子来看看它的实际效果。我们将使用`activate(...)`函数来演示这一点。
```python
x = 0.5

print("First order derivative: ", grad(activate)(x))
print("Second order derivative: ", grad(grad(activate))(x))
print("Third order derivative: ", grad(grad(grad(activate)))(x))

# First order derivative:  0.7864477
# Second order derivative:  -0.726862
# Third order derivative:  -0.5652091
```
#### 梯度和数值稳定性

**下溢**和**溢出**是我们多次遇到的常见问题，尤其是在计算梯度时。我们将举一个例子（这个例子直接来自`JAX`文档，这是一个非常好的例子）来说明我们如何遇到数值不稳定以及`JAX`如何尝试帮助您克服它。当您计算某个值的梯度时会发生什么？
```python
# An example of a mathematical operation in your workflow
def log1pexp(x):
    """Implements log(1 + exp(x))"""
    return jnp.log(1. + jnp.exp(x))

# This works fine
print("Gradients for a small value of x: ", grad(log1pexp)(5.0))

# But what about for very large values of x for which the
# exponent operation will explode
print("Gradients for a large value of x: ", grad(log1pexp)(500.0))

# Gradients for a small value of x:  0.9933072

# Gradients for a large value of x:  nan
```
刚刚发生了什么？让我们对其进行分解，以了解预期的输出以及返回`nan`的`JAX`幕后`gpoing`的内容。我们知道上述函数的导数可以写成这样：
{% asset_img ja_1.png %}

对于非常大的值，您会期望导数的值为`1`，但是当我们将`grad`与我们的函数实现结合起来时，它返回`nan`。为了获得更多信息，我们可以通过查看转换的`jaxpr`来分解梯度计算。
```python
make_jaxpr(grad(log1pexp))(500.0)

# { lambda  ; a.
#   let b = exp a
#       c = add b 1.0
#       _ = log c
#       d = div 1.0 c
#       e = mul d b
#   in (e,) }
```
如果您仔细观察，您会发现计算等效于：
{% asset_img ja_2.png %}

对于较大的值，右侧的项将四舍五入为`inf`，并且梯度计算将返回`nan`，如我们在上面看到的。在这种情况下，我们知道如何正确计算梯度，但`JAX`不知道。它正在研究标准自动差异规则。那么，我们如何告诉`JAX`，我们的函数应该按照我们想要的方式进行区分呢？我们可以使用`JAX`中的`custom_vjp`或`custom_vjp`函数来实现这一点。让我们看看它的实际效果。
```python
from jax import custom_jvp

@custom_jvp
def log1pexp(x):
    """Implements log(1 + exp(x))"""
    return jnp.log(1. + jnp.exp(x))

@log1pexp.defjvp
def log1pexp_jvp(primals, tangents):
    """Tells JAX to differentiate the function in the way we want."""
    x, = primals
    x_dot, = tangents
    ans = log1pexp(x)
    # This is where we define the correct way to compute gradients
    ans_dot = (1 - 1/(1 + jnp.exp(x))) * x_dot
    return ans, ans_dot

# Let's now compute the gradients for large values
print("Gradients for a small value of x: ", grad(log1pexp)(500.0))

# What about the Jaxpr?
make_jaxpr(grad(log1pexp))(500.0)

# Gradients for a small value of x:  1.0
# { lambda  ; a.
#   let _ = custom_jvp_call_jaxpr[ fun_jaxpr={ lambda  ; a.
#                                              let b = exp a
#                                                  c = add b 1.0
#                                                  d = log c
#                                              in (d,) }
#                                  jvp_jaxpr_thunk=<function _memoize.<locals>.memoized at 0x7f79cc3f2dd0>
#                                  num_consts=0 ] a
#       b = exp a
#       c = add b 1.0
#       d = div 1.0 c
#       e = sub 1.0 d
#       f = mul e 1.0
#   in (f,) }
```
让我们分解一下步骤。
- 我们用计算**雅可比向量积**（前向模式）的`custom_vjp`装饰了`log1pexp(...)`。
- 然后我们定义了`log1pexp_jvp(...)`来计算梯度。重点关注该函数中的这行代码：`ans_dot = (1 - 1/(1 + jnp.exp(x))) * x_dot`。简单来说，我们所做的就是以这种方式**重新排列导数**：
{% asset_img ja_3.png %}

我们用`log1pexp.defjvp`装饰`logp1exp_jvp(...)`函数，告诉`JAX`计算`JVP`，请使用我们定义的函数并返回预期的输出。