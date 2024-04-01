---
title: JAX（Pytrees）
date: 2024-04-01 13:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### Pytrees

什么是`Pytree`？如果我们按照文档中提供的`Pytress`定义，那么`Pytree`是指由类似容器的`Python`对象构建的**树状结构**。什么是类似容器的`Python`对象？看到名称，您可能已经猜到类似容器的`Python`对象包括列表、元组、字典、`namedtuple`、`OrderedDict`和`None`。这些数据结构是默认的类似容器的对象，被视为`Pytree`。我们还可以告诉`JAX`将视为类似容器的对象，但我们需要首先将它们包含在`Pytree`注册表中。让我们看一下`Pytree`的几个例子。
<!-- more -->
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
```python
# We will use the `example_2` pytree for this purpose.
# Our pytree looks like this: [1, 2, 'a', 'b', (3, 4)]
# We will unflatten it, obtain the leaves, and the tree structure as well

example_2_leaves, example_2_treedef = jax.tree_flatten(example_2)
print(f"Original Pytree: {repr(example_2)}")
print(f"Leaves: {repr(example_2_leaves)}")
print(f"Pytree structure: {repr(example_2_treedef)}")

# Original Pytree: [1, 2, 'a', 'b', (3, 4)]
# Leaves: [1, 2, 'a', 'b', 3, 4]
# Pytree structure: PyTreeDef([*, *, *, *, (*, *)])
```
现在我们已经提取了叶子，我们可以修改它们并使用原始树结构（`treedef`）再次重建树。
{% note info %}
**注意**：我们可以使用`tree_map(...)`和`tree_multimap(...)`对叶子进行操作，因为这是更好方法。上面的示例是为了展示您可以通过一种想要更多地控制应用于树的不同叶子的操作的方式来实现这一点。
{% endnote %}

```python
def change_even_positioned_leaf(x, pos):
    if pos % 2 == 0:
        return x * 2
    else:
        return x
    
transformed_leaves = [
    change_even_positioned_leaf(leaf, pos+1) for pos, leaf in enumerate(example_2_leaves)
]

print(f"Original leaves:    {repr(example_2_leaves)}")
print(f"Transformed leaves: {repr(transformed_leaves)}")

# Original leaves:    [1, 2, 'a', 'b', 3, 4]
# Transformed leaves: [1, 4, 'a', 'bb', 3, 8]
```
我们现在可以使用原始的树结构来重建具有变换后的叶子的树。我们看到`Pytree`是类似容器的`Python`对象，如列表、元组、字典等。但是如果您想扩展这组被视为`Pytree`节点的`Python`对象怎么办？ 例如，如果您想将您的类视为`Pytree`节点怎么办？要将一个类视为`Pytree`节点，我们需要：
- 通过在内部注册表中注册它，告诉`JAX`您希望将其视为节点而不是叶子。
- 因为这是一个自定义对象，`JAX`不知道如何展开和取消展开它，我们也需要告诉`JAX`。
- 在某些情况下，我们需要比较两个`treedef`结构是否相等。因此，我们需要确保添加自定义对象不会破坏相等性检查。

让我们看一个例子。
```python
from jax.tree_util import register_pytree_node
from jax.tree_util import register_pytree_node_class

class Counter:
    def __init__(self, count, name):
        self.count = count
        self.name = name
        
    def __repr__(self):
        return f"Counter value = {self.count}"
    
    def increment(self):
        return self.count + 1
    
    def decrement(self):
        return self.count - 1
    

# Because JAX doesn't know how to flattent and unflatten these custom objects
# hence we need to define those methods for these objects

def flatten_counter(tree):
    """Specifies how to flatten a Counter class object.
    
    Args:
        tree: Counter class object represented as Pytree node
    Returns:
        A pair of an iterable with the children to be flattened recursively,
        and some opaque auxiliary data to pass back to the unflattening recipe.
        The auxiliary data is stored in the treedef for use during unflattening.
        The auxiliary data could be used, e.g., for dictionary keys.
    """
    
    children = (tree.count,)
    aux_data = tree.name # We don't want to treat the name as a child
    return (children, aux_data)


def unflatten_counter(aux_data, children):
    """Specifies how to unflattening a Counter class object.

    Args:
        aux_data: the opaque data that was specified during flattening of the
            current treedef.
        children: the unflattened children
    Returns:
        A re-constructed object of the registered type, using the specified
        children and auxiliary data.
    """
    return Counter(*children, aux_data)

# Now all we need to do is to tell JAX that we need to Register our class as
# a Pytree node and it need to treat all the objects of that class as such
register_pytree_node(
    Counter,
    flatten_counter,    # tell JAX what are the children nodes
    unflatten_counter   # tell JAX how to pack back into a `Counter`
)

# An instance of the Counter class
my_counter = Counter(count=5, name="Counter_class_as_pytree_node")

# Flatten the custom object
my_counter_leaves, my_counter_treedef = jax.tree_flatten(my_counter)

# Unflatten
my_counter_reconstructed = jax.tree_unflatten(
    treedef=my_counter_treedef, leaves=my_counter_leaves
)
print(f"Original Pytree: {repr(my_counter)}")
print(f"Leaves: {repr(my_counter_leaves)}")
print(f"Pytree structure: {repr(my_counter_treedef)}")
print(f"Reconstructed Pytree: {repr(my_counter_reconstructed)}")
```
结果输出为：
```bash
Original Pytree: Counter value = 5
Leaves: [5]
Pytree structure: PyTreeDef(CustomNode(<class '__main__.Counter'>[Counter_class_as_pytree_node], [*]))
Reconstructed Pytree: Counter value = 5
```
{% note info %}
**注意**：定义函数来展开自定义对象时，请重新检查传递给该函数的参数顺序。第一个位置参数始终表示辅助数据，而第二个参数始终表示子数据。请检查上面的`flatten_counter(...)`作为示例。
{% endnote %}

我们尚未检查的一件事是我们的`Pytree`是否可以进行相等性检查。让我们检查一下。
```python
# Another instance
my_counter_2 = Counter(count=5, name="Counter_class_as_pytree_node")

# Flatten the custom object
my_counter_2_leaves, my_counter_2_treedef = jax.tree_flatten(my_counter)

# Check if the treedef are same for both the pytrees
my_counter_treedef == my_counter_2_treedef
```
#### Pytree和JAX转换

到目前为止，我们讨论了`Pytree`是什么以及如何创建它，包括使用自定义的类似容器的对象。在我们开始使用`Pytree`进行一些高级示例之前，让我们退后一步，了解如何将`grad、vmap`等`JAX`函数转换应用于`pytree`。
```python
def activate(x):
    """Applies tanh activation."""
    return jnp.tanh(x["weights"])

# Always use the PRNG
key = random.PRNGKey(1234)
example_pytree = {"weights": random.normal(key=key, shape=(5,))}

# We will now use vmap and grad to compute the gradients per sample
grads_example_pytree = vmap(grad(activate), in_axes=({"weights":0},))(example_pytree)

print("Original pytree:")
print(f" {repr(example_pytree)}\n")
print("Leaves in the pytree:")
print(f"{repr(jax.tree_leaves(example_pytree))}\n")
print("Gradients per example:")
print(f"{grads_example_pytree}\n")

# Original pytree:
#  {'weights': DeviceArray([ 0.90665466, -0.9453377 ,  0.08205649, -1.8436366 ,0.45950893], dtype=float32)}

# Leaves in the pytree:
# [DeviceArray([ 0.90665466, -0.9453377 ,  0.08205649, -1.8436366 ,0.45950893], dtype=float32)]

# Gradients per example:
# {'weights': DeviceArray([0.48228705, 0.45585024, 0.99329686, 0.0953269 , 0.8153717 ], dtype=float32)}
```
{% note warning %}
**有几点需要注意**：
- 我们可以组合任何`JAX`转换并将其应用到`Pytree`。
- 某些转换（例如`vmap`和`pmap`）采用可选参数（例如`in_axes`和`out_axes`），这些参数指定应如何处理某些输入或输出值。这些参数也可以是`pytree`，它们的结构必须与相应参数的`Pytree`结构相对应。例如，检查如何在上面的示例中传递`in_axes`的值。
{% endnote %}

#### Jaxprs 和 Pytree

与任何其他`JAX`代码一样，您也可以将`jaxprs`与`Pytree`一起使用。例如，我们检查一下上面例子的`jaxpr`。
```python
make_jaxpr(vmap(grad(activate), in_axes=({"weights":0},)))(example_pytree)

# { lambda ; a:f32[5]. let
#     b:f32[5] = tanh a
#     c:f32[5] = sub 1.0 b
#     d:f32[5] = mul 1.0 c
#     e:f32[5] = mul d b
#     f:f32[5] = add_any d e
#   in (f,) }
```
#### 为什么是Pytree？

到目前为止，我们讨论了`Pytree`，但我们尚未回答的一个问题:为什么应该了解`Pytree`？以及使用`Pytree`的常见用例是什么? 尽管`Pytree`有很多用例，但最常见的用例是**指定模型参数**。例如，如果你想构建一个`DNN`，你可以将每层对应的**权重和偏差**存储为`Pytree`。您甚至可以将`DNN`特征的整个模块定义为`Pytree`。让我们看一个相同的例子。这取自`JAX`文档。
```python
key = random.PRNGKey(111)
key, subkey = random.split(key)

# Generate some random data
x = random.normal(key=key, shape=(128, 1))
# Let's just do y = 10x + 20
y = 10 * x + 20

plt.plot(x, y, marker='x', label='Generated linear function')
plt.legend()
plt.show()
```
{% asset_img jp_1.png %}

```python
def initialize_params(key, dims):
    """Initialize the weights and biases of the MLP.
    
    Args:
        key: PRNG key
        dims: List of integers
    Returns:
        A pytree of initialized paramters for each layer
    """
    
    params = []
    
    for dim_in, dim_out in zip(dims[:-1], dims[1:]):
        key, subkey = random.split(key)
        weights = random.normal(key=key, shape=(dim_in, dim_out)) * jnp.sqrt(2 / dim_in)
        biases = jnp.zeros(shape=(dim_out))
        params.append({"weights": weights, "biases":biases})
    
    return params


# Initialize the parameters
params = initialize_params(key=subkey, dims=[1, 128, 128, 1])

# We can inspect the shape of the intialized params as well
shapes = jax.tree_map(lambda layer_params: layer_params.shape, params)

for i, shape in enumerate(shapes):
    print(f"Layer {i+1} => Params shape: {shape}")

def forward(params, x):
    """Forward pass for the MLP
    
    Args:
        params: A pytree containing the parameters of the network
        x: Inputs
    """
    *hidden, last = params
    for layer in hidden:
        x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
    return x @ last['weights'] + last['biases']

def loss_fn(params, x, y):
    """Mean squared error loss function."""
    return jnp.mean((forward(params, x) - y) ** 2)

@jax.jit
def update(params, x, y):
    """Updates the parameters of the network.
    
    Args:
        params: A pytree containing the parameters of the network
        x : Inputs
        y:  Outputs
    Returns:
        Pytree with updated values
    """
    
    # 1. Calculate the gradients based on the loss
    grads = jax.grad(loss_fn)(params, x, y)
    
    # 2. Update the parameters using `tree_multi_map(...)`
    return jax.tree_multimap(lambda p, g: p - 0.0001 * g, params, grads)

# Run the model for a few iterations
for _ in range(2000):
    params = update(params, x, y)
    
# Plot the predictions and the ground truth
plt.plot(x, y, marker='x', label='Generated linear function')
plt.plot(x, forward(params, x), marker="x", label="Predictions")
plt.legend()
plt.show()
```
```bash
Layer 1 => Params shape: {'biases': (128,), 'weights': (1, 128)}
Layer 2 => Params shape: {'biases': (128,), 'weights': (128, 128)}
Layer 3 => Params shape: {'biases': (1,), 'weights': (128, 1)}
```
{% asset_img jp_2.png %}
