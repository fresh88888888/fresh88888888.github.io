---
title: 基础知识（机器学习）(PyTorch)
date: 2024-04-22 09:22:11
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

#### 数据操作

为了能够完成各种数据操作，我们需要某种方法来存储和操作数据。通常，我们需要做两件重要的事：`1`.获取数据；`2`.将数据读入计算机后对其进行处理。如果没有某种方法来存储数据，那么获取数据是没有意义的。首先，我们介绍`n`维数组，也称为张量（`tensor`）。使用过`Python`中`NumPy`计算包。无论使用哪个深度学习框架，它的张量类（在`MXNet`中为`ndarray`，在`PyTorch`和`TensorFlow`中为`Tensor`）都与`Numpy`的`ndarray`类似。但深度学习框架又比`Numpy`的`ndarray`多一些重要功能：首先，`GPU`很好地支持加速计算，而`NumPy`仅支持`CPU`计算；其次，张量类支持自动微分。这些功能使得张量类更适合深度学习。
<!-- more -->
张量表示一个由数值组成的数组，这个数组可能有多个维度。具有一个轴的张量对应数学上的向量（`vector`）；具有两个轴的张量对应数学上的矩阵（`matrix`）；具有两个轴以上的张量没有特殊的数学名称。首先，我们可以用`arange`创建一个行向量`x`。这个行向量包含以`0`开始的前`12`个整数。他们默认创建为整数。也可以指定创建类型为浮点数。张量中的每个值都称为张量的元素（`element`）。例如，张量`x`中有`12`个元素。除非额外指定，新的张量将存储在内存中。并采用基于`CPU`的计算。
```python
import torch as torch

x = torch.arange(12)
x

# tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

# 可以通过张量shape的属性来访问张量（沿每个轴的长度）的形状。
x.shap

# torch.Size([12])

# 如果只想知道张量的元素总数，即形状的所有元素乘积，可以检查它的大小（size）。因为这里处理的是一个向量，所以它的shpae与它的size相同。
x.numel()

# 12

# 要想改变一个张量的形状而不改变元素数量和元素值，可以调用reshape函数。例如可以把张量x从形状为（12,）的行向量转换形状为（3,4）的矩阵。
# 这个新的张量包含转换前相同的值，但是他被看成一个3行4列的矩阵，要重点说明一下，虽然张量的形状发生了改变，但其元素值并没有变。
# 注意：通过改变张量的形状，张量的大小不会改变。
X = x.reshape(3,4)
X

# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11]])

# 我们不需要通过手动指定每个维度来改变形状。也就是说，我们的目标形状的是（高度,宽度），难在知道宽度后，高度会被自动计算得出，不必我们自己做除法。
# 在上面的例子中，为了获得一个3行的矩阵，我们手动指定了它有3行和4列。幸运的是，我们可以通过-1来调用此自动计算出维度的功能。 
# 即我们可以用x.reshape(-1,4)或x.reshape(3,-1)来取代x.reshape(3,4)。

# 有时,我们希望使用全0、全1、其他常量，或者从特定分布中随机采样的数字来初始化矩阵，我们可以创建一个形状(2,3,4)的张量，其中所有元素都设置为0，代码如下：
torch.zeros((2,3,4))

# tensor([[[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]],

#         [[0., 0., 0., 0.],
#          [0., 0., 0., 0.],
#          [0., 0., 0., 0.]]])

# 同样，我们可以创建一个形状为(2,3,4)的张量，其中所有元素都设置为1：
torch.ones((2,3,4))

# tensor([[[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]],

#         [[1., 1., 1., 1.],
#          [1., 1., 1., 1.],
#          [1., 1., 1., 1.]]])

# 有时我们想通过从某个特定概率分布中随机采样来得到张量中的每个元素值。例如，当我们构造数组来作为神经网络中的参数时，我们通常会随机初始化参数的值。
# 以下代码创建一个形状为（3,4）的张量。其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样。
torch.randn(3,4)

# tensor([[-0.0135,  0.0665,  0.0912,  0.3212],
#         [ 1.4653,  0.1843, -1.6995, -0.3036],
#         [ 1.7646,  1.0450,  0.2457, -0.7732]])

# 我们还可以通过提供包含数值的Python列表（或嵌入列表），来为所需张量中的每个元素赋予确定值。在这里，最外层的列表对应于轴0，内层列表对应于轴1。
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# tensor([[2, 1, 4, 3],
#         [1, 2, 3, 4],
#         [4, 3, 2, 1]])
```
##### 运算符

我们的兴趣不仅限于读取数据和写入数据。我们想在这些数据上执行数学运算，其中最简单且最有用的操作是按元素（`elementwise`）运算。它们将标准标量运算符应用于数组的每个元素。对于将两个数组作为输入的函数，按元素运算将二元运算符应用于两个数组中的每对位置对应的元素。我们可以基于任何从标量到标量的函数来创建按元素函数。

在数学的表示中，我们将通过符号{% mathjax %}f:\mathbb {R} \rightarrow \mathbb {R} {% endmathjax %}来表示一元标量运算符(只接收一个输入)。这意味着该函数从任何实数({% mathjax %}\mathbb {R}{% endmathjax %})映射到另一个实数。同样，我们通过符号{% mathjax %} f:\mathbb {R},\mathbb {R} \rightarrow \mathbb {R}{% endmathjax %}表示二元标量运算符，这意味着，该函数接收两个输入，并产生一个输出。给定同一形状的任意两个向量{% mathjax %}u{% endmathjax %}和{% mathjax %}v{% endmathjax %}和二元运算符{% mathjax %}f{% endmathjax %}，我们可以得到向量{% mathjax %}c=F(u,v){% endmathjax %}。具体计算方法是{% mathjax %}c_i \leftarrow f(u_i,v_i){% endmathjax %}，其中{% mathjax %}c_i{% endmathjax %}、{% mathjax %}u_i{% endmathjax %}和{% mathjax %}v_i{% endmathjax %}分别是向量{% mathjax %}c{% endmathjax %}、{% mathjax %}u{% endmathjax %}和{% mathjax %}v{% endmathjax %}中的元素，在这里，我们将通过标量函数升级为按元素向量运算来生成向量值{% mathjax %}F:\mathbb {R^d},\mathbb {R^d}\rightarrow \mathbb {R^d}{% endmathjax %}。

对于任意具有相同形状的张量，常见的标准算术运算符（`+、-、*、/`和`**`）都可以被升级为按元素运算。我们可以在同一形状的任意两个张量上调用按元素操作。在下面的例子中，我们使用逗号来表示一个具有`5`个元素的元组，其中每个元素都是按元素操作的结果。
```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y  # **运算符是求幂运算

# x+y=tensor([ 3.,  4.,  6., 10.])
# x-y=tensor([-1.,  0.,  2.,  6.])
# x*y=tensor([ 2.,  4.,  8., 16.])
# x/y=tensor([0.5000, 1.0000, 2.0000, 4.0000])
# x^y=tensor([ 1.,  4., 16., 64.])

# 按元素方式可以应用更多的计算，包括像求幂这样的一元运算符。
torch.exp(x)

# tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03])

# 除了按元素之外，我们还可以执行线性代数运算，包括向量点积和矩阵乘法。我们也可以把多个张量连接(concatenate)在一起，把它们端对端地叠起来形成一个更大的张量。
# 我们只需要提供张量列表，并给出沿哪个轴连结。下面的例子分别演示了当我们沿行（轴-0，形状的第一个元素）和按列（轴-1，形状的第二个元素）连结两个矩阵时，会发生什么情况。
# 我们可以看到，第一个输出张量的轴-0长度（6）是两个输入张量轴-0长度的总和（3 + 3）；第二个输出张量的轴-1长度（8）是两个输入张量轴-1长度的总和（4 + 4）。
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  6.,  7.],
#         [ 8.,  9., 10., 11.],
#         [ 2.,  1.,  4.,  3.],
#         [ 1.,  2.,  3.,  4.],
#         [ 4.,  3.,  2.,  1.]])
# tensor([[ 0.,  1.,  2.,  3.,  2.,  1.,  4.,  3.],
#         [ 4.,  5.,  6.,  7.,  1.,  2.,  3.,  4.],
#         [ 8.,  9., 10., 11.,  4.,  3.,  2.,  1.]])

# 有时，我们想通过逻辑运算构建二元张量。以X==Y为例：对于每个位置，如果X和Y在该位置相等，则新张量中相应项的值为1。这意味着逻辑语句X==Y在该位置处为真，否则该位置为0.
X==Y

# tensor([[False,  True, False,  True],
#         [False, False, False, False],
#         [False, False, False, False]])

# 对张量中的所有元素求和，会产生一个单元素张量。
X.sum()

# tensor(66.)
```
##### 广播机制

在上面的部分中，我们看到了如何在相同形状的两个张量上执行按元素操作。在某些情况下，即使形状不同，我们仍然可以通过调用**广播机制**(`broadcasting mechanism`)来执行按元素操作。这种机制的工作方式如下：
- 通过适当复制元素来扩展一个或两个数组，以便在转换之后，两个张量具有相同的形状。
- 对生成的数组执行按元素操作。

在大多数情况下我么将沿着数组中长度为1的轴进行广播，如下例子：
```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a + b

# tensor([[0, 1],
#         [1, 2],
#         [2, 3]])
```
##### 索引和切片

就像在其他任何`Python`数组中一样，张量中的元素可以同索引来访问，第一个元素的索引为`0`，做后一个元素索引为`-1`。可以指定范围以包含第一个元素和最后一个之前的元素。如下所示，我们可以用[-1]选择最后一个元素，可以用[1:3]选择第二个和第三个元素：
```python
X[-1], X[1:3]

# X[-1]:tensor([ 8.,  9., 10., 11.])
# X[1:3]:tensor([[ 4.,  5.,  6.,  7.],
#                [ 8.,  9., 10., 11.]])

除读取外，我们还可以指定索引来将元素写入矩阵。
X[1,2]=9
X

# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  5.,  9.,  7.],
#         [ 8.,  9., 10., 11.]])

# 如果我们想为多个元素赋相同的值，我们只需要索引所有元素，然后为他们赋值。例如，[0:2,:]访问第1行和第2行，其中”：“代表沿轴1（列）的所有元素。
# 虽然我们讨论的是矩阵的索引，但这也使用向量和超过2个维度的张量。
X[0:2,:] = 12
X

# tensor([[12., 12., 12., 12.],
#         [12., 12., 12., 12.],
#         [ 8.,  9., 10., 11.]])
```
##### 节省内存

运行一些操作可能会导致新结果分配内存。例如，如果我们用`Y = X + Y`，我们将取消引用Y指向的张量，而是指向新分配的内存处的张量。在下面的例子中，我们用`Python`的`id()`函数演示了这一点，它给我们提供了内存中引用对象的确切地址。运行`Y = Y + X`后，我们会发现id(Y)指向另一个位置。这是因为`Python`首先计算`Y + X`，为结果分配新的内存，然后使`Y`指向内存中的这个新位置。
```python
before = id(Y)
Y = Y + X
id(Y) == before

# False

# 这可能是不可取的，原因有两个：
# 1.首先，我们不想总是不必要的分配内存。在机器学习中，我们可能有数百兆的参数，通常情况下，我们希望原地执行这些更新。
# 2.如果我们不原地更新，其他引用仍然会指向旧的内存位置，这样我们的代码可能会无意中引用了旧的参数。

# 执行原地操作非常简单。我们可以使用切片表示法将操作的结果分配给先前分配的数组，例如Y[:] = <expression>。 
# 为了说明这一点，我们首先创建一个新的矩阵Z，其形状与另一个Y相同，使用zeros_like来分配一个全0的块。
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

# id(Z): 4668984304
# id(Z): 4668984304

# 如果在后续的计算中没有重复使用X，我们也可以使用X[:] = X + Y或X+=Y来减少操作的内存开销。
before = id(X)
X += Y
id(X) == before

# True
```
##### 转换为其它Python对象

将深度学习框架定义的张量转化为`Numpy`张量(`ndarray`)很容易，反之也同样容易。`torch`张量和`numpy`数组将共享它们的底层内存，就地操作更改一个张量也会同时更改另一个张量。
```python
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

# (numpy.ndarray, torch.Tensor)

# 要将大小为1的张量转化为Python标量，我们可以调用item(.)函数或Python的内置函数。
a = torch.tensor([3.5])
print(f'a:{a}, a.item:{a.item()}, int(a):{int(a)}, float(a):{float(a)}')

# a:tensor([3.5000]), a.item:3.5, int(a):3, float(a):3.5
```
深度学习存储和操作数据的主要接口是张量（`n`维数组）。

#### 数据预处理

为了能用机器学习来解决现实世界的问题，我们经常从预处理原始数据开始，而不是哪些准备好的张量格式数据开始。在`python`常用的数据分析工具中，我们经常使用`pandas`软件包。像庞大的`python`生态系统中的许多其它扩展包一样，`pandas`可以与张量兼容。
##### 读取数据集

举一个例子，我们首先创建一个人工数据集，并存储在`CSV`（逗号分隔值）文件.`./data/house_tiny.csv`中。以其他格式存储的数据也可以通过类似的方式进行处理。下面我们将数据集按行写入`CSV`文件中。
```python
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

# 要从创建的CSV文件中加载原始数据集，我们导入pandas包并调用read_csv函数。该数据集有四行三列。
# 其中每行描述了房间数量（“NumRooms”）、巷子类型（“Alley”）和房屋价格（“Price”）。
data = pd.read_csv(data_file)
print(data)

#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000
```
##### 处理缺失值

注意，”`NaN`“项代表缺失值。为了处理缺失的数据，典型的方法包括**插值法和删除法**，其中插值法用一个替代值来弥补缺失值，而删除法直接忽略缺失值。通过位置索引`iloc`，我们将`data`分为`inputs`和`outputs`，其中前者为`data`的前两列，而后者为`data`的最后一列，对于`inputs`中缺少的数值，我们用同一列的均值替换”`NaN`“项。
```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

#    NumRooms Alley
# 0       3.0  Pave
# 1       2.0   NaN
# 2       4.0   NaN
# 3       3.0   NaN

# 对于inputs中的类别值和离散值，我们将NaN视为一个类别。
# 由于“巷子类型”（“Alley”）列只接受两种类型的类别值“Pave”和“NaN”，pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。
# 巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。 缺少巷子类型的行会将“Alley_Pave”和“Alley_nan”分别设置为0和1。

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

#    NumRooms  Alley_Pave  Alley_nan
# 0       3.0           1          0
# 1       2.0           0          1
# 2       4.0           0          1
# 3       3.0           0          1

```
##### 转换为张量格式

现在`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式。当数据采用张量格式之后，可以用张量函数进一步操作。
```python
X = torch.tensor(inputs.to_numpy(dtype=float))
Y = torch.tensor(outputs.to_numpy(dtype=float))
X,Y

# (tensor([[3., 1., 0.],
#          [2., 0., 1.],
#          [4., 0., 1.],
#          [3., 0., 1.]], dtype=torch.float64),
#  tensor([127500., 106000., 178100., 140000.], dtype=torch.float64))
```
#### 线性代数

##### 标量

如果你曾经在餐厅支付餐费，那么应该已经知道一些基本的线性代数，比如在数字间相加或相乘。例如，北京的温度为{% mathjax %}52\,^{\circ}\mathrm{F}{% endmathjax %}(华氏度，除摄氏度之外一种计量单位)。严格来说，仅包含一个数值被称为标量(`scalar`)。如果要将华氏度转换为更常用的摄氏度值，则可以计算表达式{% mathjax %}c=\frac{5}{9}(f-32){% endmathjax %}并将{% mathjax %}f{% endmathjax %}赋为52。在此等式中每一项`5、9`和`32`都是标量值。符号{% mathjax %}c{% endmathjax %}和{% mathjax %}f{% endmathjax %}称为变量(variable)，他们表示未知的标量值。在这里采用了数学表示法，其中变量用普通小写字母表示（例如{% mathjax %}x,y{% endmathjax %}和{% mathjax %}z{% endmathjax %}）用{% mathjax %}\mathbb{R}{% endmathjax %}表示所有（连续）实数标量的空间之后将严格定义空间(space)是什么？但现在只要记住表达式{% mathjax %}x\in\mathbb{R}{% endmathjax %}是表示{% mathjax %}x{% endmathjax %}是一个是值标量的正式形式。符号{% mathjax %}\in{% endmathjax %}称为”属于“，它表示是集合中的成员。例如{% mathjax %}x,y\in{0,1}{% endmathjax %}可以用来表明{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}只能为0或1的数字。标量由只有一个元素的张量表示，下面的代码将实例化两个标量，并执行一些熟悉的算术运算，即加法、乘法、除法和指数。
```python
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y

# (tensor(5.), tensor(6.), tensor(1.5000), tensor(9.))
```
##### 向量

向量可以被视为标量值组成的列表。这些标量值被称为向量的元素(`element`)或分量(`component`)。当向量表示数据集中的样本时，它们的值具有一定的现实意义。例如，如果我们正在训练一个模型来预测贷款违约风险，可能会将每个申请人与一个向量相关联，其分量与其收入、工作年限、过往违约次数和其他因素相对应。如果我们正在研究医院患者可能面临的心脏病发作风险，可能会用一个向量来表示每个患者，其分量为最近的生命体征、胆固醇水平、每天运动时间等。在数学表示法中，向量通常记为粗体、小写的符号（例如，{% mathjax %}\mathbf{x},\mathbf{y}{% endmathjax %}和{% mathjax %}\mathbf{z}{% endmathjax %}）。人们通过一维张量来表示向量。一般来说张量可以具有任意长度，取决于机器的内存限制。我们可以使用下标来引用向量的任意元素例如可以通过{% mathjax %}x_i{% endmathjax %}来引用第{% mathjax %}i{% endmathjax %}个元素。注意，{% mathjax %}x_i{% endmathjax %}是一个标量，索引我们在引用它时不会加粗，大量文献认为列向量是向量的默认方向。在数学中，向量{% mathjax %}\mathbf{x}{% endmathjax %}可以写为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{x}=\begin{bmatrix}x_1 \\x_2 \\ \vdots \\ x_n\end{bmatrix}
{% endmathjax %}
其中{% mathjax %}x_1,\ldots,x_n{% endmathjax %}是向量的元素。在代码中，我们通过张量的索引来访问任一元素。
```python
x[3]

# tensor(3)

# 向量只有一个数字数组，就像每个数组都有一个长度一样，每个向量也是如此。在数学表示法中，如果我们想说一个向量
# {% mathjax %}\mathbf{x}{% endmathjax %}由n个实际标量组成，可以将其表示为{% mathjax %}\mathbf{x}\in\mathbb{R}^n{% endmathjax %}。
# 向量的长度通常称为向量的维度(dimension)。与普通的Python数组一样，我们可以通过调用Python的内置len()函数来访问张量的长度。
len(x)

# 4

# 当用一个张量表示一个向量（只有一个轴）时，我们也可以通过.shape属性访问向量的长度。
# 形状(shape)是一个元组，列出了张量沿每个轴的长度（维数）。对于只有一个轴的张量，形状只有一个元素。
x.shape

# torch.Size([4])
```
{% note warning %}
**注意**：维度(`dimension`)在不同上下文中会有不同的含义，经常会使人感到困惑。为了清楚起见，在此明确一下：向量或轴的维度被用来表示向量或轴的长度，即向量或轴的元素数量。然而张量的维度用来表示张量具有的轴数。在这个意义上张量的某个轴的维数就是这个轴的长度。
{% endnote %}
##### 矩阵

  正如向量将标量从零阶推广到一阶，矩阵将向量从一阶推广到了二阶。矩阵，我们通常用粗体、大写字母来表示（例如，{% mathjax %}X,Y{% endmathjax %}和{% mathjax %}Z{% endmathjax %}），在代码中表示为具有两个轴的张量。数学表示法使用{% mathjax %}\mathbf{A}\in\mathbb{R}^{m\times n}{% endmathjax %}来表示矩阵{% mathjax %}\mathbf{A}{% endmathjax %}，其由{% mathjax %}m{% endmathjax %}行{% mathjax %}n{% endmathjax %}列的实值标量组成。我们可以将任意矩阵{% mathjax %}\mathbf{A}\in\mathbb{R}^{m\times n}{% endmathjax %}视为一个表格，其中每个元素{% mathjax %}a_{ij}{% endmathjax %}属于第{% mathjax %}i{% endmathjax %}行第{% mathjax %}j{% endmathjax %}列：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1n} \\ a_{21} & a_{22} & \ldots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \ldots & a_{mn}\end{bmatrix}
{% endmathjax %}
对于任意{% mathjax %}\mathbf{A}\in\mathbb{R}^{m\times n}{% endmathjax %}，{% mathjax %}\mathbf{A}{% endmathjax %}的形状是{% mathjax %}(m,n){% endmathjax %}或{% mathjax %}m\times n{% endmathjax %}。当矩阵具有相同数量的行和列时其形状将变为正方形。因此，它被称为”方正“(`square matrix`)。当调用函数来实例化张量时，我们可以通过指定两个分量{% mathjax %}m{% endmathjax %}和{% mathjax %}n{% endmathjax %}来创建一个形状为{% mathjax %}m\times n{% endmathjax %}的矩阵。
```python
A = torch.arange(20).reshape(5, 4)
A

# tensor([[ 0,  1,  2,  3],
#         [ 4,  5,  6,  7],
#         [ 8,  9, 10, 11],
#         [12, 13, 14, 15],
#         [16, 17, 18, 19]])
```
我们可以通过行索引({% mathjax %}i{% endmathjax %})和列索引({% mathjax %}j{% endmathjax %})来访问矩阵中的标量元素{% mathjax %}a_{ij}{% endmathjax %}，例如，{% mathjax %}[\mathbf{A}]_{ij}{% endmathjax %}。如果没有给出矩阵{% mathjax %}\mathbf{A}{% endmathjax %}的标量元素，我们可以简单的使用矩阵{% mathjax %}\mathbf{A}{% endmathjax %}的小写字母索引下标{% mathjax %}a_{ij}{% endmathjax %}来引用{% mathjax %}[\mathbf{A}]_{ij}{% endmathjax %}。为了表示起来简单，只有在必要时才会将逗号插入到单独的索引中，例如{% mathjax %}a_{2,3j}{% endmathjax %}。当我们交换矩阵的行和列时，结果称为矩阵的转置(`transpose`)。通常用{% mathjax %}a_{\mathsf{T}}{% endmathjax %}来表示矩阵的转置，如果{% mathjax %}\mathbf{B}= \mathbf{A}_{\mathsf{T}}{% endmathjax %}，则对于任何的{% mathjax %}i{% endmathjax %}和{% mathjax %}j{% endmathjax %}，都有{% mathjax %}b_{ij}= a_{ji}{% endmathjax %}，因此{% mathjax %}\mathbf{A}{% endmathjax %}的转置是一个形状为{% mathjax %}n\times m{% endmathjax %}的矩阵：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{A}^{\mathsf{T}}=\begin{bmatrix} a_{11} & a_{21} & \ldots & a_{m1} \\ a_{12} & a_{22} & \ldots & a_{m2} \\ \vdots & \vdots & \ddots & \vdots \\ a_{1n} & a_{2n} & \ldots & a_{mn}\end{bmatrix}
{% endmathjax %}
在代码中访问矩阵的转置。
```python
A.T

# tensor([[ 0,  4,  8, 12, 16],
#         [ 1,  5,  9, 13, 17],
#         [ 2,  6, 10, 14, 18],
#         [ 3,  7, 11, 15, 19]])
```
作为方阵的一种特殊类型，对称矩阵(`symmetric matrix`) {% mathjax %}\mathbf{A}{% endmathjax %}等于其转置：{% mathjax %}\mathbf{A}=\mathbf{A}^{\mathsf{T}}{% endmathjax %}。这里定义一个对称矩阵{% mathjax %}\mathbf{B}{% endmathjax %}：
```python
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
B

# tensor([[1, 2, 3],
#         [2, 0, 4],
#         [3, 4, 5]])

# 现在我们将B与它的转置进行比较。
B == B.T

# tensor([[True, True, True],
#         [True, True, True],
#         [True, True, True]])
```
矩阵是有用的数据结构：它们允许我们组织具有不同模式的数据。例如，矩阵中的行可能对应于不同的房屋（数据样本），而列可能对应于不同的属性。因此，尽管单个向量的默认方向是列向量，但在表示表格数据集的矩阵中，将每个数据样本作为矩阵中的行向量更为常见。
##### 张量

就像向量是标量的推广，矩阵是向量的推广一样，我们可以构建更多轴的数据结构。张量是描述具有任意数量轴的n为数组的通用方法，例如，向量是一阶张量，矩阵是二阶张量。张量用特殊字体的大写字母表示（例如，{% mathjax %}\mathsf{X,Y}{% endmathjax %}和{% mathjax %}\mathsf{Z}{% endmathjax %}），它们的索引机制（例如{% mathjax %}x_{ijk}{% endmathjax %}和{% mathjax %}[\mathsf{X}_{1,2i-1,3}]{% endmathjax %}）与矩阵类似。当我们开始处理图像时，张量将变得更加重要，图像以n维数组形式出现，其中3个轴对应于高度、宽度以及一个通道(channel)轴，用于表示颜色通道（红色，绿色和蓝色）。
```python
X = torch.arange(24).reshape(2, 3, 4)
X

# tensor([[[ 0,  1,  2,  3],
#          [ 4,  5,  6,  7],
#          [ 8,  9, 10, 11]],

#         [[12, 13, 14, 15],
#          [16, 17, 18, 19],
#          [20, 21, 22, 23]]])
```
标量、向量、矩阵和任意数量轴的张量有一些实用的属性。例如，从按元素操作的定义中可以注意到，任何按元素的一元运算都不会改变其操作数的形状。同样，给定具有相同形状的任意两个张量，任何按元素二元运算的结果都将是相同形状的张量。例如，将两个相同形状的矩阵相加，会在这两个矩阵上执行元素加法。
```python
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B
A, A + B

# (tensor([[ 0.,  1.,  2.,  3.],
#          [ 4.,  5.,  6.,  7.],
#          [ 8.,  9., 10., 11.],
#          [12., 13., 14., 15.],
#          [16., 17., 18., 19.]]),

#  tensor([[ 0.,  2.,  4.,  6.],
#          [ 8., 10., 12., 14.],
#          [16., 18., 20., 22.],
#          [24., 26., 28., 30.],
#          [32., 34., 36., 38.]]))
```
具体而言，连个矩阵的按元素乘法称为`Hadamard`积(`Hadamard product`) (数学符号为{% mathjax %}\odot{% endmathjax %})。对于矩阵{% mathjax %}\mathbf{B}\in\mathbb{R}^{m\times n}{% endmathjax %}，其中第{% mathjax %}i{% endmathjax %}和第{% mathjax %}j{% endmathjax %}列的元素是{% mathjax %}b_{ij}{% endmathjax %}。矩阵{% mathjax %}\mathbf{A}{% endmathjax %}和{% mathjax %}\mathbf{B}{% endmathjax %}的`Hadamard`积为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{A}\odot\mathbf{B}=\begin{bmatrix} a_{11}b{11} & a_{12}b{12} & \ldots & a_{1n}b{1n} \\ a_{21}b{21} & a_{22}b_{22} & \ldots & a_{2n}b_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1}b_{m1} & a_{m2}b_{m2} & \ldots & a_{mn}b_{mn}\end{bmatrix}
{% endmathjax %}
```python
A * B

# tensor([[  0.,   1.,   4.,   9.],
#         [ 16.,  25.,  36.,  49.],
#         [ 64.,  81., 100., 121.],
#         [144., 169., 196., 225.],
#         [256., 289., 324., 361.]])

# 将张量乘以或加上一个标量不会改变张量的形状，其中张量的每个元素都将与标量相加或相乘。
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape

# (tensor([[[ 2,  3,  4,  5],
#           [ 6,  7,  8,  9],
#           [10, 11, 12, 13]],

#          [[14, 15, 16, 17],
#           [18, 19, 20, 21],
#           [22, 23, 24, 25]]]),
#  torch.Size([2, 3, 4]))
```
##### 降维

我们可以对任意张量进行的一个有用的操作是计算其元素的和。数学表示法使用{% mathjax %}\sum{% endmathjax %}来表示求和，为了表示长度为{% mathjax %}d{% endmathjax %}的向量中元素的总和，可以记为{% mathjax %}\sum_{i=1}^{x_i}{% endmathjax %}。在代码中可以调用计算求和的函数：
```python
x = torch.arange(4, dtype=torch.float32)
x, x.sum()

# (tensor([0., 1., 2., 3.]), tensor(6.))
```
我们可以表示任意形状张量的元素和。例如，矩阵{% mathjax %}\mathbf{A}{% endmathjax %}中元素的和可以记为{% mathjax %}\sum_{i=1}^m \sum_{j=1}^{n} a_{ij}{% endmathjax %}。
```python
A.shape, A.sum()

# (torch.Size([5, 4]), tensor(190.))
```
默认情况下，调用求和函数会沿所有的轴降低张量的维度，使它变为一个标量。我们还可以指定张量沿哪一个轴来通过求和降低维度。以矩阵为例，为了通过求和所有行的元素来降维（轴`0`），可以在调用函数时指定`axis=0`。由于输入矩阵沿`0`轴降维以生成输出向量，因此输入轴`0`的维数在输出形状中消失。
```python
A_sum_axis0 = A.sum(axis=0)
A_sum_axis0, A_sum_axis0.shape

# (tensor([40., 45., 50., 55.]), torch.Size([4]))

# 指定axis=1将通过汇总所有列的元素降维（轴1）。因此，输入轴1的维数在输出形状中消失。
A_sum_axis1 = A.sum(axis=1)
A_sum_axis1, A_sum_axis1.shape

# (tensor([ 6., 22., 38., 54., 70.]), torch.Size([5]))

# 沿着行和列对矩阵求和，等价于对矩阵的所有元素进行求和。
A.sum(axis=[0, 1])  # 结果和A.sum()相同

# tensor(190.)

# 一个与求和相关的量是平均值（mean或average）。我们通过将总和除以元素总数来计算平均值。在代码中，我们可以调用函数来计算任意形状张量的平均值。
A.mean(), A.sum() / A.numel()

# (tensor(9.5000), tensor(9.5000))

# 同样，计算平均值的函数也可以沿指定轴降低张量的维度。
A.mean(axis=0), A.sum(axis=0) / A.shape[0]

# (tensor([ 8.,  9., 10., 11.]), tensor([ 8.,  9., 10., 11.]))
```
##### 非降维求和

但是，有时在调用函数来计算总和或均值时保持轴数不变会很有用。
```python
sum_A = A.sum(axis=1, keepdims=True)
sum_A

# tensor([[ 6.],
#         [22.],
#         [38.],
#         [54.],
#         [70.]])

# 例如，由于sum_A在对每行进行求和后仍保持两个轴，我们可以通过广播将A除以sum_A。
A / sum_A

# tensor([[0.0000, 0.1667, 0.3333, 0.5000],
#         [0.1818, 0.2273, 0.2727, 0.3182],
#         [0.2105, 0.2368, 0.2632, 0.2895],
#         [0.2222, 0.2407, 0.2593, 0.2778],
#         [0.2286, 0.2429, 0.2571, 0.2714]])

# 如果我们想沿某个轴计算A元素的累积总和，比如axis=0（按行计算），可以调用cumsum函数。此函数不会沿任何轴降低输入张量的维度。
A.cumsum(axis=0)

# tensor([[ 0.,  1.,  2.,  3.],
#         [ 4.,  6.,  8., 10.],
#         [12., 15., 18., 21.],
#         [24., 28., 32., 36.],
#         [40., 45., 50., 55.]])
```
##### 点积

我们已经学习了按元素操作，、求和和平均值，另一个最基本的操作之一是**点积**。给定两个向量{% mathjax %}\mathbf{x,y}\in\mathbb{R}^d{% endmathjax %},它们的点积(`dot product`){% mathjax %}\mathbf{x}^{\mathsf{T}} \mathbf{y}{% endmathjax %}(或{% mathjax %}\langle\mathbf{x,y}\rangle{% endmathjax %})是相同位置按元素乘积的和：{% mathjax %}\mathbf{x}^{\mathsf{T}} \mathsf{y}=\sum_{i=1}^d x_iy_i{% endmathjax %}。
```python
x = torch.tensor(4, dytype=torch.float32)
y = torch.ones(4, dtype = torch.float32)
x, y, torch.dot(x, y)

# (tensor([0., 1., 2., 3.]), tensor([1., 1., 1., 1.]), tensor(6.))

# 注意，我们可以执行按元素乘法，然后进行求和来表示两个向量的点积：
torch.sum(x * y)

# tensor(6.)
```
点积在很多场合很有用。例如，给定一组向量{% mathjax %}\mathbf{x}\in \mathbb{R}^d{% endmathjax %}表示的值，和一组由{% mathjax %}\mathbf{w}\in \mathbb{R}^d{% endmathjax %}表示的权重。{% mathjax %}\mathbf{x}{% endmathjax %}中的值根据权重{% mathjax %}\mathbf{w}{% endmathjax %}的加权和，可以表示为点积{% mathjax %}\mathbf{x}^{\mathsf{T}} \mathbf{w}{% endmathjax %}。当权重为非负数且和为`1`（即{% mathjax %}\sum_{i=1}^d w_i = 1{% endmathjax %}）时，点击表示加权平均(`weighted average`)。将两个向量规范化得到单位长度后，点积表示它们夹角的余弦。
##### 矩阵-向量积

现在我们知道如何计算点积，可以开始理解矩阵-向量积（`matrix-vector product`）。让我们将矩阵{% mathjax %}\mathbf{A}{% endmathjax %}用它的行向量表示：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{A}=\begin{bmatrix}a_1^{\mathsf{T}} \\a_2^{\mathsf{T}} \\ \vdots \\ a_m^{\mathsf{T}}\end{bmatrix}
{% endmathjax %}
其中每个{% mathjax %}a_i^{\mathsf{T}} \in\mathbb{R}^n{% endmathjax %}都是行向量，表示矩阵的第{% mathjax %}i{% endmathjax %}。矩阵向量积{% mathjax %}\mathbf{Ax}{% endmathjax %}是一个长度为{% mathjax %}m{% endmathjax %}的列向量，其第{% mathjax %}i{% endmathjax %}个元素的点积{% mathjax %}a_i^{\mathsf{T}}x{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{Ax}=\begin{bmatrix}a_1^{\mathsf{T}} \\a_2^{\mathsf{T}} \\ \vdots \\ a_m^{\mathsf{T}}\end{bmatrix} \mathbf{x}=\begin{bmatrix}a_1^{\mathsf{T}}\mathbf{x} \\a_2^{\mathsf{T}}\mathbf{x} \\ \vdots \\ a_m^{\mathsf{T}}\mathbf{x}\end{bmatrix}
{% endmathjax %}
我们可以把一个矩阵{% mathjax %}\mathbf{A}\in \mathbb{R}^{m\times n}{% endmathjax %}乘法看做一个从{% mathjax %}\mathbb{R}^n{% endmathjax %}到{% mathjax %}\mathbb{R}^m{% endmathjax %}向量的转换。这些转换是非常有用的，例如可以用方阵的乘法来表示旋转。我们也可以使用矩阵-向量积来描述在给定前一层的值时，求解神经网络每一层所需的复杂计算。在代码中使用张量表示矩阵-向量积，我们使用`mv`函数。当我们为矩阵A和向量`x`调用`torch.mv(A, x)`时，会执行矩阵-向量积。注意，`A`的列维数（沿轴`1`的长度）必须与`x`的维数（其长度）相同。
```python
A.shape, x.shape, torch.mv(A, x)

# (torch.Size([5, 4]), torch.Size([4]), tensor([ 14.,  38.,  62.,  86., 110.]))
```
##### 矩阵-矩阵乘法

在掌握点积和矩阵-向量积后，那么矩阵-矩阵乘法（`matrix-matrix multiplication`）应该很简单。假设有两个矩阵{% mathjax %}\mathbf{A}\in \mathbb{R}^{n\times k}{% endmathjax %}和{% mathjax %}\mathbf{B}\in \mathbb{R}^{k\times m}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{A}\odot\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \ldots & a_{1k} \\ a_{21} & a_{22} & \ldots & a_{2k} \\ \vdots & \vdots & \ddots & \vdots \\ b_{n1} & a_{n2} & \ldots & a_{nk}\end{bmatrix}, \mathbf{B}\odot\mathbf{B}=\begin{bmatrix} b_{11} & b_{12} & \ldots & b_{1m} \\ b_{21} & b_{22} & \ldots & a_{2m} \\ \vdots & \vdots & \ddots & \vdots \\ b_{k1} & b_{k2} & \ldots & b_{km}\end{bmatrix}
{% endmathjax %}
用行向量{% mathjax %}\mathbf{a}_i^{\mathsf{T}}\in \mathbb{R}^k{% endmathjax %}表示矩阵{% mathjax %}\mathbf{A}{% endmathjax %}的第{% mathjax %}i{% endmathjax %}行，并让列向量{% mathjax %}\mathbf{b}_j\in \mathbb{R}^k{% endmathjax %}作为矩阵{% mathjax %}\mathbf{B}{% endmathjax %}的第{% mathjax %}j{% endmathjax %}列。要生成矩阵积{% mathjax %}\mathbf{C}= \mathbf{AB}{% endmathjax %}，最简单的方法是考虑{% mathjax %}\mathbf{A}{% endmathjax %}的行向量和{% mathjax %}\mathbf{B}{% endmathjax %}的列向量：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{A}=\begin{bmatrix}a_1^{\mathsf{T}} \\a_2^{\mathsf{T}} \\ \vdots \\ a_n^{\mathsf{T}}\end{bmatrix} , \mathbf{B}= \begin{bmatrix}\mathbf{b}_1 & \mathbf{b}_2 & \ldots & \mathbf{b}_m\end{bmatrix}
{% endmathjax %}
当我们将简单的每个元素{% mathjax %}c_{ij}{% endmathjax %}计算为点积{% mathjax %}\mathbf{a}_i^{\mathsf{T}} \mathbf{b}_j{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{C}=\mathbf{AB}=\begin{bmatrix}a_1^{\mathsf{T}} \\a_2^{\mathsf{T}} \\ \vdots \\ a_n^{\mathsf{T}}\end{bmatrix} \mathbf{B}= \begin{bmatrix}\mathbf{b}_1 & \mathbf{b}_2 & \ldots & \mathbf{b}_m\end{bmatrix} = \begin{bmatrix} a_{1}^{\mathsf{T}}b_1 & a_{1}^{\mathsf{T}}b_2 & \ldots & a_{1}^{\mathsf{T}}b_m \\ a_{2}^{\mathsf{T}}b_1 & a_{2}^{\mathsf{T}}b_2 & \ldots & a_{2}^{\mathsf{T}}b_m \\ \vdots & \vdots & \ddots & \vdots \\ a_{n}^{\mathsf{T}}b_1 & a_{n}^{\mathsf{T}}b_2 & \ldots & a_{n}^{\mathsf{T}}b_m\end{bmatrix}
{% endmathjax %}
我么可以将矩阵-矩阵乘法{% mathjax %}\mathbf{AB}{% endmathjax %}看做简单的执行{% mathjax %}m{% endmathjax %}次矩阵向量积，并将结果拼接在一起，形成一个{% mathjax %}n\times m{% endmathjax %}的矩阵。在下面的代码中，我们在{% mathjax %}\mathbf{A}{% endmathjax %}和{% mathjax %}\mathbf{B}{% endmathjax %}上执行矩阵乘法。这里的{% mathjax %}\mathbf{A}{% endmathjax %}是一个`5`行`4`列的矩阵，{% mathjax %}\mathbf{B}{% endmathjax %}是一个`4`行`3`列的矩阵。 两者相乘后，我们得到了一个`5`行`3`列的矩阵。
```python
B = torch.ones(4, 3)
torch.mm(A, B)

# tensor([[ 6.,  6.,  6.],
#         [22., 22., 22.],
#         [38., 38., 38.],
#         [54., 54., 54.],
#         [70., 70., 70.]])
```
**矩阵-矩阵乘法**可以简单地称为矩阵乘法，不应与“`Hadamard积`”混淆。
##### 范数

线性代数中最有用的一些运算符是范数（`norm`）。非正式地说，向量的范数是表示一个向量有多大。这里考虑的大小（`size`）概念不涉及维度，而是分量的大小。在线性代数中，向量范数是将向量映射到标量的函数{% mathjax %}f{% endmathjax %}。给定任意向量{% mathjax %}\mathbf{x}{% endmathjax %}，向量范数要满足一些属性。 第一个性质是：如果我们按常数因子{% mathjax %}\alpha{% endmathjax %}缩放向量的所有元素，其范数也会按相同常数因子的绝对值缩放：
{% mathjax '{"conversion":{"em":14}}' %}
f(\alpha x) = \lvert\alpha\rvert f(x)
{% endmathjax %}
第二个性质是熟悉的三角不等式:
{% mathjax '{"conversion":{"em":14}}' %}
f(x+y)\leq f(x) + f(y)
{% endmathjax %}
第三个性质简单地说范数必须是非负的:
{% mathjax '{"conversion":{"em":14}}' %}
f(x)\geq 0
{% endmathjax %}
这是有道理的。因为在大多数情况下，任何东西的最小的大小是`0`。 最后一个性质要求范数最小为`0`，当且仅当向量全由`0`组成。
{% mathjax '{"conversion":{"em":14}}' %}
\forall i, [\mathbf{x}]_i = 0 \Leftrightarrow f(x) = 0
{% endmathjax %}
范数听起来很像距离的度量。欧几里得距离和毕达哥拉斯定理中的非负性概念和三角不等式可能会给出一些启发。事实上，欧几里得距离是一个{% mathjax %}L_2{% endmathjax %}范数：假设{% mathjax %}n{% endmathjax %}维向量{% mathjax %}\mathbf{x}{% endmathjax %}中的元素是{% mathjax %}x_1, \ldots ,x_n{% endmathjax %}，其{% mathjax %}L_2{% endmathjax %}范数是向量元素平方和的平方根：
{% mathjax '{"conversion":{"em":14}}' %}
\lVert\mathbf{x}\rVert_2 = \sqrt{\sum_{i=1}^n x_i^2}
{% endmathjax %}
其中，在{% mathjax %}L_2{% endmathjax %}范数中常常省略下标{% mathjax %}2{% endmathjax %}，也就是说{% mathjax %}\lVert\mathbf{x}\rVert{% endmathjax %}等同于{% mathjax %}\lVert\mathbf{x}\rVert_2{% endmathjax %}。 在代码中，我们可以按如下方式计算向量的{% mathjax %}L_2{% endmathjax %}范数。
```python
u = torch.tensor([3.0, -4.0])
torch.norm(u)

# tensor(5.)
```
深度学习中经常使用{% mathjax %}L_2{% endmathjax %}范数的平方，也会经常遇到{% mathjax %}L_1{% endmathjax %}范数，他表示为向量元素的绝对值求和：
{% mathjax '{"conversion":{"em":14}}' %}
\lVert\mathbf{x}\rVert_1 = \sum_{i=1}^n \lvert x_i \rvert
{% endmathjax %}
与{% mathjax %}L_2{% endmathjax %}范数相比{% mathjax %}L_1{% endmathjax %}范数，我们将绝对值函数和按元素求和组合起来。
```python
torch.abs(u).sum()

# tensor(7.)
```
{% mathjax %}L_2{% endmathjax %}范数和{% mathjax %}L_1{% endmathjax %}范数都是更一般的{% mathjax %}L_p{% endmathjax %}范数的特例：
{% mathjax '{"conversion":{"em":14}}' %}
\lVert\mathbf{x}\rVert_p = \Big( \sum_{i=1}^n \lvert x_i \rvert_p \Big)^{1/p}
{% endmathjax %}
类似于向量的{% mathjax %}L_2{% endmathjax %}范数，矩阵{% mathjax %}\mathbf{X}\in \mathbb{R}^{m\times n}{% endmathjax %}的`Frobenius`范数（`Frobenius norm`）是矩阵元素平方和的平方根：
{% mathjax '{"conversion":{"em":14}}' %}
\lVert\mathbf{x}\rVert_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}
{% endmathjax %}
`Frobenius`范数满足向量范数的所有性质，它就像是矩阵形向量的{% mathjax %}L_2{% endmathjax %}范数。调用以下函数将计算矩阵的`Frobenius`范数。
```python
torch.norm(torch.ones((4, 9)))

# tensor(6.)
```
在深度学习中，我们经常试图解决优化问题：最大化分配给观测数据的概率; 最小化预测和真实观测之间的距离。用向量表示物品（如单词、产品或新闻文章），以便最小化相似项目之间的距离，最大化不同项目之间的距离。目标，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为范数。线性代数还有很多，其中很多数学对于机器学习非常有用。例如，矩阵可以分解为因子，这些分解可以显示真实世界数据集中的低维结构。机器学习的整个子领域都侧重于使用矩阵分解及其向高阶张量的泛化，来发现数据集中的结构并解决预测问题。当开始动手尝试并在真实数据集上应用了有效的机器学习模型，你会更倾向于学习更多数学。

#### 微积分

在`2500`年前，古希腊人把一个多边形分成三角形，并把它们面积相加，才找到计算多边形面积的方法，为了求出曲线形状的面积，古希腊人在这样的形状上内接多边形。内接多边形的等长越多，就越接近圆。这个过程也被称为**逼近法**(`method of exhaustion`)。
{% asset_img p_1.png "用逼近法求圆的面积" %}

事实上，**逼近法就是积分**（`integral calculus`）的起源。`2000`多年后，微积分的另一支，微分（`differential calculus`）被发明出来。**在微分学最重要的应用是优化问题**，即考虑如何把事情做到最好。这种问题在深度学习中是无处不在的。在深度学习中，我们“训练”模型，不断更新它们，使它们在看到越来越多的数据时变得越来越好。通常情况下，变得更好意味着最小化一个**损失函数**（`loss function`），即一个衡量“模型有多糟糕”这个问题的分数。最终，我们真正关心的是生成一个模型，它能够在从未见过的数据上表现良好。但“训练”模型只能将模型与我们实际能看到的数据相拟合。因此，我们可以将拟合模型的任务分解为两个关键问题：
- **优化**（`optimization`）：用模型拟合观测数据的过程；
- **泛化**（`generalization`）：数学原理和实践者的智慧，能够指导我们生成出有效性超出用于训练的数据集本身的模型。

##### 导数和微分

我们首先讨论导数的计算，这是几乎所有深度学习优化算法的关键步骤。在深度学习中，我们通常选择对于模型参数可微的损失函数。简而言之，对于每个参数，如果我们把这个参数增加或减少一个无穷小的量，可以知道损失会以多快的速度增加或减少。假设我们有一个函数{% mathjax %}f: \mathbb{R}\rightarrow \mathbb{R}{% endmathjax %}，其输入和输出都是标量。如果{% mathjax %}f{% endmathjax %}的导数存在，这个极限被定义为：
{% mathjax '{"conversion":{"em":14}}' %}
f'(x)=\lim_{h \to 0} \frac{f(x+h)-f(x)}{h}
{% endmathjax %}
如果{% mathjax %}f'(a){% endmathjax %}存在，则称{% mathjax %}f{% endmathjax %}在{% mathjax %}a{% endmathjax %}处可微(`differentiable`)的。如果{% mathjax %}f{% endmathjax %}在一个区间内的每个数上都是可微的，则此函数在此区间中是可微的。我们可以将导数{% mathjax %}f'(x){% endmathjax %}解释为{% mathjax %}f(x){% endmathjax %}相对于{% mathjax %}x{% endmathjax %}的瞬时(`instantaneous`)变化率。所谓的瞬时变化率是基于{% mathjax %}x{% endmathjax %}中的变化{% mathjax %}h{% endmathjax %}，且{% mathjax %}h{% endmathjax %}接近`0`。为了更好的解释导数，让我们做一个实验。定义{% mathjax %}u=f(x)=3x^2-4x{% endmathjax %}如下：
```python
import numpy as np
import torch as torch
from matplotlib_inline import backend_inline

def f(x):
    return 3 * x ** 2 - 4 * x

def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

h = 0.1
for i in range(5):
    print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
    h *= 0.1

# h=0.10000, numerical limit=2.30000
# h=0.01000, numerical limit=2.03000
# h=0.00100, numerical limit=2.00300
# h=0.00010, numerical limit=2.00030
# h=0.00001, numerical limit=2.00003
```
通过{% mathjax %}x=1{% endmathjax %}并让{% mathjax %}h{% endmathjax %}接近于`0`，{% mathjax %}\frac{f(x+h) - f(x)}{h}{% endmathjax %}结果接近于`2`.虽然这个实验不是数学证明。但当{% mathjax %}x=1{% endmathjax %}时，导数{% mathjax %}u'{% endmathjax %}是`2`。让我们熟悉一下导数的几个等价符号。给定{% mathjax %}y=f(x){% endmathjax %}，其中{% mathjax %}x{% endmathjax %}和{% mathjax %}y{% endmathjax %}分别是{% mathjax %}f{% endmathjax %}的自变量和因变量。以下表达式是等价的：
{% mathjax '{"conversion":{"em":14}}' %}
f'(x)=y'=\frac{dy}{dx}=\frac{df}{dx}=\frac{d}{dx}f(x)=Df(x)=D_xf(x)
{% endmathjax %}
其中符号{% mathjax %}\frac{d}{dx}{% endmathjax %}和{% mathjax %}D{% endmathjax %}是微分运算符，表示微分操作，我们可以使用以下规则来对常见函数求微分：
- {% mathjax %}DC=0{% endmathjax %}({% mathjax %}C{% endmathjax %}是一个常熟)。
- {% mathjax %}Dx^n=nx^{n-1}{% endmathjax %}(幂率(power rule)，{% mathjax %}n{% endmathjax %}是任意实数)。
- {% mathjax %}De^x=e^x{% endmathjax %}。
- {% mathjax %}D\ln(x)=1/x{% endmathjax %}。

为了微分一个由一些常见函数组成的函数，下面的一些法则方便使用。假设函数{% mathjax %}f{% endmathjax %}和{% mathjax %}g{% endmathjax %}都是可微的，{% mathjax %}C{% endmathjax %}是一个常数，则：
{% mathjax '{"conversion":{"em":14}}' %}
常数相乘法：
\begin{equation*}
\frac{d}{dx}[Cf(x)]=C\frac{d}{dx}f(x)
\end{equation*}
{% endmathjax %}

{% mathjax '{"conversion":{"em":14}}' %}

\begin{equation*}
\frac{d}{dx}[f(x)+g(x)]=\frac{d}{dx}f(x) + \frac{d}{dx}g(x)
\end{equation*}
{% endmathjax %}

{% mathjax '{"conversion":{"em":14}}' %}

\begin{equation*}
\frac{d}{dx}[f(x)g(x)]=f(x)\frac{d}{dx}[g(x)] + g(x)\frac{d}{dx}[f(x)]
\end{equation*}
{% endmathjax %}

{% mathjax '{"conversion":{"em":14}}' %}

\begin{equation*}
\frac{d}{dx}[\frac{f(x)}{g(x)}]=\frac{g(x)\frac{d}{dx}[f(x)]-f(x)\frac{d}{dx}[g(x)]}{[g(x)]^2}
\end{equation*}
{% endmathjax %}
现在我们可以应用上述几个法则来计算{% mathjax %}u'=f'(x)=3\frac{d}{dx}x^2 - 4\frac{d}{dx}x = 6x-4{% endmathjax %}。令{% mathjax %}x=1{% endmathjax %}，我们有{% mathjax %}u'=2{% endmathjax %}：在这个实验中，数值结果接近`2`，这一点得到了前面的实验的支持。当{% mathjax %}x=1{% endmathjax %}时，此导数也是曲线{% mathjax %}u=f(x){% endmathjax %}切线的斜率。为了对导数的这种解释进行可视化，我们将使用`matplotlib`，这是一个`Python中`流行的绘图库。要配置`matplotlib`生成图形的属性，我们需要定义几个函数。在下面，`use_svg_display`函数指定`matplotlib`软件包输出`svg`图表以获得更清晰的图像。
```python
def use_svg_display():  
    """使用svg格式在Jupyter中显示绘图"""
    backend_inline.set_matplotlib_formats('svg')

# 我们定义set_figsize函数来设置图表大小。
def set_figsize(figsize=(3.5, 2.5)): 
    """设置matplotlib的图表大小"""
    use_svg_display()

# 下面的set_axes函数用于设置由matplotlib生成图表的轴的属性。
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

# 通过这三个用于图形配置的函数，定义一个plot函数来简洁地绘制多条曲线，因为我们需要可视化许多曲线。
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# 现在我们可以绘制函数u=f(x)及其在x=1处的切线y=2x-3，其中系数2是切线的斜率。
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```
{% asset_img p_2.png %}

##### 偏导数

到目前为止，我们只讨论了仅含一个变量的函数的微分。在深度学习中，函数通常依赖于许多变量。因此，我们需要将微分的思想推广到多元函数(`multivariate function`)上。设{% mathjax %} y=f(x_1,x_2,\dots,x_n){% endmathjax %}是一个具有{% mathjax %}n{% endmathjax %}个变量的函数，{% mathjax %}y{% endmathjax %}关于第{% mathjax %}i{% endmathjax %}个参数{% mathjax %}x_i{% endmathjax %}的偏导数(`partial derivative`)为：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial y}{\partial x_i}=\lim_{h \to 0}\frac{f(x_1,\dots,x_{i-1},x_i+h,x_{i+1},\dots,x_n)-f(x_1,\dots,x_i,\dots,x_n)}{h}
{% endmathjax %}
为了计算{% mathjax %}\frac{\partial y}{\partial x_i}{% endmathjax %}，我们可以简单地将{% mathjax %}x_1,\dots,x_{i-1},x_{i+1},\dots,x_n{% endmathjax %}看做常熟，并计算{% mathjax %}y{% endmathjax %}关于{% mathjax %}x_i{% endmathjax %}的导数。对于偏导数的表示以下是等价的：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial y}{\partial x_i}=\frac{\partial f}{\partial x_i}=f_{x_i}=f_i=D_if=D_{x_i}f
{% endmathjax %}
##### 梯度

我们可以连结一个多元函数对其所有变量的偏导数，以得到该函数的梯度(`gradient`)向量。具体而言，设函数{% mathjax %}f:\mathbb{R}\rightarrow \mathbb{R}{% endmathjax %}的输入是一个`n`维向量{% mathjax %}\mathbf{x}=[x1,x2,\dots,x_n]^{\mathsf{T}}{% endmathjax %}，并且输出是一个标量。函数{% mathjax %}f(x){% endmathjax %}相对于{% mathjax %}x{% endmathjax %}的梯度是一个包含n个偏导数的向量：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla_x f(x)=[\frac{\partial f(x)}{\partial x_1},\frac{\partial f(x)}{\partial x_2},\dots,\frac{\partial f(x)}{\partial x_n}]^{\mathsf{T}}
{% endmathjax %}
其中{% mathjax %}\nabla_x f(x){% endmathjax %}通常在没有歧义时被{% mathjax %}\nabla f(x){% endmathjax %}取代。假设{% mathjax %}x{% endmathjax %}为{% mathjax %}n{% endmathjax %}维向量，在微分多元函数时经常使用以下规则：
- 对于所有{% mathjax %}\mathbf{A}\in \mathbb{R}^{m\times n}{% endmathjax %},都有{% mathjax %}\nabla_x \mathbf{Ax}=\mathbf{A}^{\mathsf{T}}{% endmathjax %}。
- 对于所有{% mathjax %}\mathbf{A}\in \mathbb{R}^{n\times m}{% endmathjax %},都有{% mathjax %}\nabla_x \mathbf{x}^{\mathsf{T}}\mathbf{A}=\mathbf{A}{% endmathjax %}。
- 对于所有{% mathjax %}\mathbf{A}\in \mathbb{R}^{n\times n}{% endmathjax %},都有{% mathjax %}\nabla_x \mathbf{x}^{\mathsf{T}}\mathbf{Ax}=(\mathbf{A}+\mathbf{a}^{\mathsf{T}})\mathbf{x}{% endmathjax %}。
- {% mathjax %}\nabla_x \lVert \mathbf{x}\rVert^2 = \nabla_x\mathbf{x}^{\mathsf{T}}\mathbf{x} = 2\mathbf{x}{% endmathjax %}。

同样，对于任何矩阵{% mathjax %}\mathbf{X}{% endmathjax %}，都有{% mathjax %}\nabla_x \lVert \mathbf{X} \rVert_F^2 = 2\mathbf{X}{% endmathjax %}，正如我们之后将看到的，梯度对于设计深度学习中的优化算法有很大用处。
##### 链式法则

然而，上面方法可能很难找到梯度。这是因为在深度学习中，**多元函数**通常是复合(`composite`)的，所以难以应用上述任何规则来微分这些函数。幸运的是，链式法则可以被用来微分复合函数。让我们先考虑单变量函数。假设函数{% mathjax %}y=f(u){% endmathjax %}和{% mathjax %}u=g(x){% endmathjax %}都是可微的，根据链式法则：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}
{% endmathjax %}
现在考虑一个更一般的场景，即函数具有任意数量的变量的情况。假设可微分函数{% mathjax %}y{% endmathjax %}有变量{% mathjax %}u_1,u_2,\dots,u_m{% endmathjax %}，其中每个可微分函数{% mathjax %}u_i{% endmathjax %}都有变量{% mathjax %}x_1,x_2,\dots,x_n{% endmathjax %}。注意{% mathjax %}y{% endmathjax %}是{% mathjax %}x_1,x_2,\dots,x_n{% endmathjax %}的函数。对于任意{% mathjax %}i=1,2,\dots,n{% endmathjax %}，链式法则给出：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial y}{\partial x_i}= \frac{\partial y}{\partial u_1}\frac{\partial u_1}{\partial x_i} + \frac{\partial y}{\partial u_2}\frac{\partial u_2}{\partial x_i} + \dots + \frac{\partial y}{\partial u_m}\frac{\partial u_m}{\partial x_i}
{% endmathjax %}

##### 总结

**微分和积分是微积分的两个分支，前者可以应用于深度学习中的优化问题。导数可以被解释为函数相对于其变量的瞬时变化率，它也是函数曲线的切线的斜率。梯度是一个向量，其分量是多变量函数相对于其所有变量的偏导数。链式法则可以用来微分复合函数。**

#### 自动微分

求导是几乎所有深度学习优化算法的关键步骤。虽然求导的计算很简单，只需要一些基本的微积分。但对于复杂的模型，手工进行更新是一件很痛苦的事情（而且经常容易出错）。深度学习框架通过自动计算导数，即**自动微分**(`automatic differentiation`)来加快求导。实际中，根据设计好的模型，系统会构建一个**计算图**(`computational graph`)，来跟踪计算是哪些数据通过哪些操作组合起来产生输出。自动微分使系统能够随后反向传播梯度。这里，**反向传播**(`backpropagate`)意味着跟踪整个计算图，填充关于每个参数的偏导数。

作为一个演示例子，假设我们想对函数{% mathjax %}y=2x^{\mathsf{T}}x{% endmathjax %}关于列向量{% mathjax %}x{% endmathjax %}求导。首先，我们创建变量`x`并为其分配一个初始值。
```python
import torch

x = torch.arange(4.0)
x.requires_grad_(True)  # 等价于x=torch.arange(4.0,requires_grad=True)
y = 2 * torch.dot(x, x)
y.backward()
x.grad
x.grad == 4 * x
# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()
x.grad

# tensor([ 0.,  4.,  8., 12.])
# tensor([True, True, True, True])
# tensor([1., 1., 1., 1.])
```
在我们计算{% mathjax %}y{% endmathjax %}关于{% mathjax %}x{% endmathjax %}的梯度之前，需要一个地方来存储梯度。重要的是，我们不会在每次对一个参数求导时都分配新的内存。因为我们经常会成千上万次地更新相同的参数，每次都分配新的内存可能很快就会将内存耗尽。注意，一个标量函数关于向量{% mathjax %}x{% endmathjax %}的梯度是向量，并且与{% mathjax %}x{% endmathjax %}具有相同的形状。现在计算{% mathjax %}y{% endmathjax %}。`x`是一个长度为`4`的向量，计算`x`和`x`的点积，得到了我们赋值给`y`的标量输出。 接下来，通过调用反向传播函数来自动计算`y`关于`x`每个分量的梯度，并打印这些梯度。函数{% mathjax %}y=2x^{\mathsf{T}}x{% endmathjax %}关于{% mathjax %}x{% endmathjax %}的梯度应为{% mathjax %}4x{% endmathjax %}。让我们快速验证这个梯度是否计算正确。现在计算`x`的另一个函数。
##### 非标量变量的反向传播

当`y`不是标量时，向量`y`关于向量`x`的导数的最自然解释是一个矩阵。对于高阶和高维的`y`和`x`，求导的结果可以是一个高阶张量。然而，虽然这些更奇特的对象确实出现在高级机器学习中（包括深度学习中），但当调用向量的反向计算时，我们通常会试图计算一批训练样本中每个组成部分的损失函数的导数。这里，我们的目的不是计算微分矩阵，而是单独计算批量中每个样本的偏导数之和。
```python
# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于self的梯度。
# 本例只想求偏导数的和，所以传递一个1的梯度是合适的
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
x.grad

# tensor([0., 2., 4., 6.])
```
##### 分离计算

有时，我们希望将某些计算移动到记录的计算图之外。例如，假设`y`是作为`x`的函数计算的，而`z`则是作为`y`和`x`的函数计算的。想象一下，我们想计算`z`关于`x`的梯度，但由于某种原因，希望将`y`视为一个常数，并且只考虑到`x`在`y`被计算后发挥的作用。这里可以分离`y`来返回一个新变量`u`，该变量与`y`具有相同的值，但丢弃计算图中如何计算`y`的任何信息。换句话说，梯度不会向后流经`u`到`x`。因此，下面的反向传播函数计算{% mathjax %}z=u\times x{% endmathjax %}关于`x`的偏导数，同时将`u`作为常数处理，而不是{% mathjax %}z=x\times x\times x{% endmathjax %}关于`x`的偏导数。
```python
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u

# tensor([True, True, True, True])

# 由于记录了y的计算结果，我们可以随后在y上调用反向传播，得到y=x*x关于的x的导数，即2*x。
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x

# tensor([True, True, True, True])
```
##### Python控制流的梯度计算

使用自动微分的一个好处是：即使构建函数的计算图需要通过`Python`控制流（例如，条件、循环或任意函数调用），我们仍然可以计算得到的变量的梯度。在下面的代码中，while循环的迭代次数和if语句的结果都取决于输入`a`的值。
```python
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

# 让我们计算梯度。
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

# 我们现在可以分析上面定义的f函数。请注意，它在其输入a中是分段线性的。 
# 换言之，对于任何a，存在某个常量标量k，使得f(a)=k*a，其中k的值取决于输入a，因此可以用d/a验证梯度是否正确。
a.grad == d / a

# tensor(True)
```
##### 总结

深度学习框架可以自动计算导数：我们首先将梯度附加到想要对其计算偏导数的变量上，然后记录目标值的计算，执行它的反向传播函数，并访问得到的梯度。

#### 概率统计

简单地说，机器学习就是做出预测。根据病人的临床病史，我们可能想预测他们在下一年心脏病发作的概率。在飞机喷气发动机的异常检测中，我们想要评估一组发动机读数为正常运行情况的概率有多大。 在强化学习中，我们希望**智能体**(`agent`)能在一个环境中智能地行动。这意味着我们需要考虑在每种可行的行为下获得高奖励的概率。当我们建立推荐系统时，我们也需要考虑概率。例如，假设我们为一家大型在线书店工作，我们可能希望估计某些用户购买特定图书的概率。为此，我们需要使用概率学。

##### 基本概率论

##### 处理多个随机变量

##### 期望和方差