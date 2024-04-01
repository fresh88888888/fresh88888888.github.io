---
title: JAX（Pytrees）
date: 2024-04-01 13:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### Pytrees

什么是`Pytree`？如果我们按照文档中提供的`Pytress`定义，那么`Pytree`是指由类似容器的`Python`对象构建的**树状结构**。
<!-- more -->
什么是类似容器的`Python`对象？看到名称，您可能已经猜到类似容器的`Python`对象包括列表、元组、字典、`namedtuple`、`OrderedDict`和`None`。这些数据结构是默认的类似容器的对象，被视为`Pytree`。我们还可以告诉`JAX`将视为类似容器的对象，但我们需要首先将它们包含在`Pytree`注册表中。让我们看一下`Pytree`的几个例子。
```python
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from jax import make_jaxpr
from jax import vmap, pmap, jit
from jax import grad, value_and_grad
from jax.test_util import check_grads

# A list as a pytree
example_1 = [1, 2, 3]

# As in normal Python code, a list that represents pytree
# can contain obejcts of any type
example_2 = [1, 2, "a", "b", (3, 4)]

# Similarly we can define pytree using a tuple as well
example_3 = (1, 2, "a", "b", (3, 4))

# We can define the same pytree using a dict as well
example_4 = {"k1": 1, "k2": 2, "k3": "a", "k4": "b", "k5": (3, 4)}

# Let's check the number of leaves and the corresponding values in the above pytrees
example_pytrees = [example_1, example_2, example_3, example_4]
for pytree in example_pytrees:
    leaves = jax.tree_leaves(pytree)
    print(f"Pytree: {repr(pytree):<30}")
    print(f"Number of leaves: {len(leaves)}")
    print(f"Leaves are: {leaves}\n")
    print("="*50)

# Pytree: [1, 2, 3]                     
# Number of leaves: 3
# Leaves are: [1, 2, 3]

# ==================================================
# Pytree: [1, 2, 'a', 'b', (3, 4)]      
# Number of leaves: 6
# Leaves are: [1, 2, 'a', 'b', 3, 4]

# ==================================================
# Pytree: (1, 2, 'a', 'b', (3, 4))      
# Number of leaves: 6
# Leaves are: [1, 2, 'a', 'b', 3, 4]

# ==================================================
# Pytree: {'k1': 1, 'k2': 2, 'k3': 'a', 'k4': 'b', 'k5': (3, 4)}
# Number of leaves: 6
# Leaves are: [1, 2, 'a', 'b', 3, 4]

# ==================================================
```
简而言之，`Pytree`只是节点（类似容器的`Python`对象）和叶子（所有其他`Python`对象）的组合。`JAX`还允许您将自定义类型注册为`Pytree`。此时您应该问的一个问题是`JAX`中的核心数据结构`DeviceArray`是否可以用作`Pytree`。答案是否定的。任何`ndarray`都被视为`Pytree`中的叶子，让我们举个例子来说明这一点。
```python
# Check if we can make a pytree from a DeviceArray
example_5 = jnp.array([1, 2, 3])
leaves = jax.tree_leaves(example_5)
print(f"DeviceArray: {repr(example_5):<30}")
print(f"Number of leaves: {len(leaves)}")
print(f"Leaves are: {leaves}")

# DeviceArray: DeviceArray([1, 2, 3], dtype=int32)
# Number of leaves: 1
# Leaves are: [DeviceArray([1, 2, 3], dtype=int32)]
```
如您所见，`ndarray`仅被视为一片叶子。对于数值也是如此。另一个需要注意的重要事项是`Pytree`是树状数据结构，而不是`DAG`或图状数据结构。他们假设引用透明，并且不存在引用循环。因此，不要在`Pytree`的多个叶子中使用相同的对象。我们可以在每一层压平树，得到叶子，以及原始的树结构。让我们看看它的实际效果。
