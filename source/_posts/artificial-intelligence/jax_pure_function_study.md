---
title: JAX（Pure Functions）
date: 2024-03-28 10:10:32
tags:
  - AI
categories:
  - 人工智能
---

#### 纯函数（Pure Functions）

函数就是纯函数：
- 当使用相同的输入调用时，该函数返回相同的值。
- 函数调用没有观察到副作用。

虽然这个定义看起来很简单，但如果没有示例，它可能很难理解，而且听起来很模糊（尤其是对于初学者）。第一点很清楚，但是副作用是什么意思呢？什么构成或被标记为副作用？您可以做什么来避免副作用？虽然我可以在这里陈述所有内容，并且您可以尝试将它们“适合”您的头脑，以确保您写的内容不会产生副作用，但我更喜欢举一些例子，以便每个人都能理解“为什么”部分以更简单的方式。那么，让我们举几个例子，看看一些可能产生副作用的常见错误。
<!-- more -->
#### 案例1：全局变量
```python
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad
from jax import jit
from jax import lax
from jax import random

# A global variable
counter = 5

def add_global_value(x):
    """
    A function that relies on the global variable `counter` for
    doing some computation.
    """
    return x + counter

x = 2

# We will `JIT` the function so that it runs as a JAX transformed
# function and not like a normal python function
y = jit(add_global_value)(x)
print("Global variable value: ", counter)
print(f"First call to the function with input {x} with global variable value {counter} returned {y}")

# Someone updated the global variable value later in the code
counter = 10

# Call the function again
y = jit(add_global_value)(x)
print("\nGlobal variable changed value: ", counter)
print(f"Second call to the function with input {x} with global variable value {counter} returned {y}")
```
结果输出为：
```bash
Global variable value:  5
First call to the function with input 2 with global variable value 5 returned 7

Global variable changed value:  10
Second call to the function with input 2 with global variable value 10 returned 7
```
当您执行jit函数时，JAX跟踪就会启动。在第一次调用时，结果将符合预期，但在后续函数调用中，您将获得缓存的结果，除非：
- 参数的类型已更改。
- 论证的形式已经改变。

```python
# Change the type of the argument passed to the function
# In this case we will change int to float (2 -> 2.0)
x = 2.0
y = jit(add_global_value)(x)
print(f"Third call to the function with input {x} with global variable value {counter} returned {y}")

# Change the shape of the argument
x = jnp.array([2])

# Changing global variable value again
counter = 15

# Call the function again
y = jit(add_global_value)(x)
print(f"Third call to the function with input {x} with global variable value {counter} returned {y}")
```
结果输出为：
```bash
Third call to the function with input 2.0 with global variable value 10 returned 12.0
Third call to the function with input [2] with global variable value 15 returned [17]
```
如果我一开始就不执行`jit`函数怎么办？
```python
def apply_sin_to_global():
    return jnp.sin(jnp.pi / counter)

y = apply_sin_to_global()
print("Global variable value: ", counter)
print(f"First call to the function with global variable value {counter} returned {y}")


# Change the global value again
counter = 90
y = apply_sin_to_global()
print("\nGlobal variable value: ", counter)
print(f"Second call to the function with global variable value {counter} returned {y}")
```
结果输出为：
```bash
Global variable value:  15
First call to the function with global variable value 15 returned 0.20791170001029968

Global variable value:  90
Second call to the function with global variable value 90 returned 0.03489949554204941
```
我们使用`JAX`来转换原生`Python`代码，使其运行得更快。如果我们编译代码，以便它可以在`XLA`（`JAX`使用的编译器）上运行。因此，避免在计算中使用全局变量，因为全局变量会引入杂质(`impurity`)。

#### 案例2：迭代器

我们将举一个非常简单的例子来看看。我们将以两种不同的方式将0到5之间的数字相加：
- 将实际的数组传递给函数。
- 将迭代器对象传递给同一函数。

```python
# A function that takes an actual array object
# and add all the elements present in it
def add_elements(array, start, end, initial_value=0):
    res = 0
    def loop_fn(i, val):
        return val + array[i]
    return lax.fori_loop(start, end, loop_fn, initial_value)


# Define an array object
array = jnp.arange(5)
print("Array: ", array)
print("Adding all the array elements gives: ", add_elements(array, 0, len(array), 0))


# Redefining the same function but this time it takes an 
# iterator object as an input
def add_elements(iterator, start, end, initial_value=0):
    res = 0
    def loop_fn(i, val):
        return val + next(iterator)
    return lax.fori_loop(start, end, loop_fn, initial_value)
    
    
# Create an iterator object
iterator = iter(np.arange(5))
print("\n\nIterator: ", iterator)
print("Adding all the elements gives: ", add_elements(iterator, 0, 5, 0))
```
结果输出为：
```bash
Array:  [0 1 2 3 4]
Adding all the array elements gives:  10

Iterator:  <iterator object at 0x7ff9e82205d0>
Adding all the elements gives:  0
```
为什么第二种情况的结果为`0`？这是因为迭代器引入了**外部状态**来检索下一个值。

#### 案例3：IO

让我们再举一个例子，一个非常不寻常的例子，它可能会使你的函数变得不纯粹（`impure`）。
```python
def return_as_it_is(x):
    """Returns the same element doing nothing. A function that isn't
    using `globals` or any `iterator`
    """
    print(f"I have received the value")
    return x

# First call to the function
print(f"Value returned on first call: {jit(return_as_it_is)(2)}\n")

# Second call to the fucntion with different value
print(f"Value returned on second call: {jit(return_as_it_is)(4)}")
```
结果输出为：
```bash
I have received the value
Value returned on first call: 2

Value returned on second call: 4
```
你注意到了吗？收到该值的声明没有在后续调用中打印。此时，大多数人都会说，这太疯狂了！我没有使用全局变量，没有迭代器，什么都没有，还有副作用吗？这怎么可能呢？问题是你的功能仍然依赖于外部状态。它使用标准输出流进行打印。如果由于某种原因该流在后续调用中不可用怎么办？当使用相同的输入调用时，这将违反“返回相同的输出”的第一原则。简而言之，**为了保持函数的纯粹性，不要使用任何依赖于外部状态的东西**。**外部**这个词很重要，因为您可以在内部使用有状态对象，并且仍然保持函数的纯净。我们也举个例子:

#### 带有状态对象的纯函数

```python
# Function that uses stateful objects but internally and is still pure
def pure_function_with_stateful_obejcts(array):
    array_dict = {}
    for i in range(len(array)):
        array_dict[i] = array[i] + 10
    return array_dict

array = jnp.arange(5)

# First call to the function
print(f"Value returned on first call: {jit(pure_function_with_stateful_obejcts)(array)}")

# Second call to the fucntion with different value
print(f"\nValue returned on second call: {jit(pure_function_with_stateful_obejcts)(array)}")
```
结果输出为：
```bash
Value returned on first call: {0: DeviceArray(10, dtype=int32), 1: DeviceArray(11, dtype=int32), 2: DeviceArray(12, dtype=int32), 3: DeviceArray(13, dtype=int32), 4: DeviceArray(14, dtype=int32)}

Value returned on second call: {0: DeviceArray(10, dtype=int32), 1: DeviceArray(11, dtype=int32), 2: DeviceArray(12, dtype=int32), 3: DeviceArray(13, dtype=int32), 4: DeviceArray(14, dtype=int32)}
```
因此，为了保持简洁，请记住不要在函数内部使用任何依赖于任何外部状态（包括IO）的内容。如果这样做，转换函数会给你带来意想不到的结果，并且当转换后的函数返回缓存结果时，你最终会浪费大量时间来调试代码，这很讽刺，因为**纯函数**很容易调试。

#### 为什么是纯函数？

我想到的一个问题是，为什么`JAX`首先使用纯函数？没有其他框架（如`TensorFlow、PyTorch、mxnet`等）使用它。您必须正确思考的另一件事可能是：使用纯函数真是令人头疼，我从来不需要在`TF/Torch`中处理这些细微差别。如果您有这种想法，那么您并不孤单，但在得出任何结论之前，请考虑一下依赖纯函数的优势。
- **易于调试**：函数是纯函数意味着您不需要超出纯函数的范围。您需要关注的只是参数、函数内部的逻辑以及返回值，相同的输入 => 相同的输出。
- **易于并行化**：假设您有三个函数`A、B`和`C`，并且涉及如下计算：`res = A(x) + B(y) + C(z)`。因为所有函数都是纯函数，所以您不必担心对外部状态或共享状态的依赖。`A、B`和 `C`之间的执行不存在依赖关系。每个函数接收一些参数并返回相同的输出。因此，您可以轻松地将计算部署到许多线程、内核、设备等。编译器必须确保所有函数（在本例中为`A、B`和`C`）的结果在项分配之前可用。
- **缓存或记忆**：我们在上面的示例中看到，一旦编译了纯函数，该函数将在后续调用时返回缓存的结果。我们可以缓存函数的结果，使整个程序更快。
- **功能组合**：当函数是纯函数时，您可以将它们组合起来以更简单的方式解决复杂的问题。例如，在`JAX`中您会经常看到这些模式：`jit(vmap(grad(..)))`。
- **参考透明度**：如果一个表达式可以被其对应的值替换而不改变程序的行为，则该表达式被称为**引用透明**。这只有当函数是纯函数时才能实现。它在做代数时特别有用（这是我们在机器学习中所做的所有事情）。例如，考虑表达式：`x = 5`，`y = 5`，`z = x + y`。现在，考虑到z的值来自**纯函数**，您可以在代码中的任何位置将`x + y`替换为`z`。
