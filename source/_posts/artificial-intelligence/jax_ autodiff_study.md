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

有时我们不希望梯度流过特定计算中涉及的某些变量。在这种情况下，我们需要明确告诉JAX我们不希望梯度流经指定的变量集。稍后我们将研究这方面的复杂示例，但现在，我将修改我们的`Product(...)`函数，其中我们不希望梯度流经`y`。
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
