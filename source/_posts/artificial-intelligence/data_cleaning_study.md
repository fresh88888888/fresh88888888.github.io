---
title: 数据清理（Pandas）
date: 2024-03-26 09:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 处理缺失值

数据清理是数据科学的关键部分，但它可能会令人深感沮丧。为什么有些文本字段出现乱码？对于那些缺失的值你应该做什么？为什么您的日期格式不正确？如何快速清理不一致的数据输入？您将学习如何解决一些最常见的数据清理问题，以便您可以更快地分析数据。 您将使用真实、混乱的数据完成五个实践练习，并解决一些最常见的数据清理问题。
<!-- more -->
##### 先看一下数据

我们需要做的第一件事是加载我们将使用的库和数据集。为了进行演示，我们将使用美式橄榄球比赛中发生的事件的数据集。在以下练习中，您将把新技能应用于旧金山颁发的建筑许可证数据集。
```python
# modules we'll use
import pandas as pd
import numpy as np

# read in all our data
nfl_data = pd.read_csv("../input/nflplaybyplay2009to2016/NFL Play by Play 2009-2017 (v4).csv")

# set seed for reproducibility
np.random.seed(0) 
```
当您获得新数据集时要做的第一件事就是查看其中的内容。这可以让您正确读取所有内容，并了解数据的情况。在这种情况下，让我们看看是否有任何缺失值，这些值将用`NaN`或`None`表示。
```python
# look at the first five rows of the nfl_data file. 
# I can see a handful of missing data already!
nfl_data.head()
```
{% asset_img dc_1.png %}

##### 我们有多少个缺失的数据？

现在我们知道我们确实有一些缺失值。让我们看看每列中有多少个。
```python
# get the number of missing data points per column
missing_values_count = nfl_data.isnull().sum()

# look at the # of missing points in the first ten columns
missing_values_count[0:10]
```
结果输出为：
```bash
Date                0
GameID              0
Drive               0
qtr                 0
down            61154
time              224
TimeUnder           0
TimeSecs          224
PlayTimeDiff      444
SideofField       528
dtype: int64
```
看起来好像很多啊！查看数据集中缺失的值的百分比可能会有所帮助，以便我们更好地了解此问题的规模：
```python
# how many total missing values do we have?
total_cells = np.product(nfl_data.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)
```
结果输出为：
```bash
24.87214126835169
```
这个数据集中几乎四分之一的单元格是空的！在下一步中，我们将仔细查看一些缺少值的列，并尝试找出它们可能发生的情况。

##### 找出数据丢失的原因

让我们来看一个例子。查看`nfl_data`数据框中缺失值的数量，我注意到“`TimesSec`”列中有很多缺失值：
```python
# look at the # of missing points in the first ten columns
missing_values_count[0:10]
```
结果输出为：
```bash
Date                0
GameID              0
Drive               0
qtr                 0
down            61154
time              224
TimeUnder           0
TimeSecs          224
PlayTimeDiff      444
SideofField       528
dtype: int64

```
通过查看文档，我可以看到该列包含有关游戏进行时剩余秒数的信息。这意味着这些值可能丢失，因为它们没有被记录，而不是因为它们不存在。因此，我们尝试猜测它们应该是什么而不是仅仅将它们保留为`NA`。另一方面，还有其他字段，例如“`PenalizedTeam`”，也有很多缺失字段。但在这种情况下，该字段缺失，因为如果没有处罚，那么说哪支球队受到处罚就没有意义。对于此列，将其留空或添加第三个值（例如“两者都不是”）并使用它来替换`NA`会更有意义。
{% note info %}
**提示**：如果您还没有阅读数据集文档，这是一个阅读数据集文档的好地方！ 如果您正在使用从其他人那里获得的数据集，您也可以尝试联系他们以获取更多信息。
{% endnote %}
如果您正在进行非常仔细的数据分析，此时您需要单独查看每一列，以找出填充这些缺失值的最佳策略。

##### 删除缺失值

如果您很着急或没有找出值缺失的原因，您可以选择删除包含缺失值的任何行或列。如果您确定要删除缺少值的行，`pandas`确实有一个方便的函数`dropna()`来帮助您执行此操作。让我们在`NFL`数据集上尝试一下！
```python
# remove all the rows that contain a missing value
nfl_data.dropna()
```
```python
# remove all columns with at least one missing value
columns_with_na_dropped = nfl_data.dropna(axis=1)
columns_with_na_dropped.head()
```
我们一共丢失了多少数据：
```python
# just how much data did we lose?
print("Columns in original dataset: %d \n" % nfl_data.shape[1])
print("Columns with na's dropped: %d" % columns_with_na_dropped.shape[1])
```
结果输出为：
```bash
Columns in original dataset: 102 
Columns with na's dropped: 41
```
我们丢失了相当多的数据，但此时我们已经成功地从数据中删除了所有`NaN`。

##### 自动填充缺失值

另一种选择是尝试填写缺失的值。对于接下来的部分，我将获取`NFL`数据的一小部分，以便它可以很好地打印。
```python
# get a small subset of the NFL dataset
subset_nfl_data = nfl_data.loc[:, 'EPA':'Season'].head()
subset_nfl_data
```
我们可以使用`Panda`的`fillna()`函数来填充数据框中的缺失值。我们的一种选择是指定我们想要用什么来替换`NaN`值。在这里，我想说的是我想用`0`替换所有`NaN`值。
```python
# replace all NA's with 0
subset_nfl_data.fillna(0)
```
我还可以更精明一点，用同一列中紧随其后的任何值替换缺失值。
```python
# replace all NA's the value that comes directly after it in the same column, 
# then replace all the remaining na's with 0
subset_nfl_data.fillna(method='bfill', axis=0).fillna(0)
```
结果输出为：
{% asset_img dc_2.png %}

#### 缩放和标准化

##### 设置环境

```python
# modules we'll use
import pandas as pd
import numpy as np

# for Box-Cox Transformation
from scipy import stats

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling

# plotting modules
import seaborn as sns
import matplotlib.pyplot as plt

# set seed for reproducibility
np.random.seed(0)
```
##### 缩放与标准化：有什么区别？

缩放和标准化之间很容易混淆的原因之一:是因为这些术语有时可以互换使用，而且更令人困惑的是，它们非常相似！在这两种情况下，您都会转换数值变量的值，以便转换后的数据点具有特定的有用属性。不同之处在于：
- 在缩放中，您正在更改数据的范围。
- 在标准化过程中，您正在改变数据分布的形状。

让我们更深入地讨论一下每个选项。

##### 缩放（Scaling）

这意味着您正在转换数据，使其适合特定的范围，例如`0-100`或`0-1`。当您使用基于数据点距离度量的方法（例如支持向量机(`SVM`)或`k`最近邻(`KNN`)）时，您需要缩放数据。使用这些算法，任何数字特征中“`1`”的变化都被赋予相同的重要性。例如，您可能会查看某些产品的日元和美元价格。`1`美元大约值`100`日元，但如果你不调整价格，`SVM`或`KNN`等方法会认为`1`日元的价格差异与`1`美元的差异一样重要！这显然不符合我们对世界的直觉。使用货币，您可以在货币之间进行转换。但是如果您要查看身高和体重之类的数据怎么办？目前尚不完全清楚多少磅应等于一英寸（或多少公斤应等于一米）。通过缩放变量，您可以帮助在平等的基础上比较不同的变量。为了帮助巩固缩放的外观，让我们看一个虚构的示例。
```python
# generate 1000 data points randomly drawn from an exponential distribution
original_data = np.random.exponential(size=1000)

# mix-max scale the data between 0 and 1
scaled_data = minmax_scaling(original_data, columns=[0])

# plot both together to compare
fig, ax = plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(scaled_data, ax=ax[1], kde=True, legend=False)
ax[1].set_title("Scaled data")
plt.show()
```
{% asset_img dc_3.png %}

{% note warning %}
**请注意**，数据的形状没有改变，但范围不再是`0`到`8`，而是现在的范围是 `0`到`1`。
{% endnote %}

##### 标准化（Normalization）

缩放只会改变数据的范围。**标准化**是一种更彻底的转变。标准化的目的是改变您的观察结果，以便将它们描述为**正态分布**。正态分布：也称为“**钟形曲线**”，这是一种特定的统计分布，其中大致相等的观测值落在平均值之上和之下，平均值和中位数相同，并且接近平均值的观测值较多。正态分布也称为**高斯分布**。一般来说，如果您要使用假设数据呈正态分布的机器学习或统计技术，则需要对数据进行标准化。其中的一些示例包括**线性判别分析**(`LDA`)和**高斯朴素贝叶斯**。（专业提示：名称中带有“高斯”的任何方法都可能假设正态分布。）我们在这里用来标准化的方法称为`Box-Cox`变换。让我们快速浏览一下一些数据的标准化是什么样子的：
```python
# normalize the exponential data with boxcox
normalized_data = stats.boxcox(original_data)

# plot both together to compare
fig, ax=plt.subplots(1, 2, figsize=(15, 3))
sns.histplot(original_data, ax=ax[0], kde=True, legend=False)
ax[0].set_title("Original Data")
sns.histplot(normalized_data[0], ax=ax[1], kde=True, legend=False)
ax[1].set_title("Normalized data")
plt.show()
```
{% asset_img dc_4.png %}

{% note warning %}
**请注意**，我们的数据形状已经改变。在**标准化**之前它几乎是**L形**的。但标准化后，它看起来更像**钟形的轮廓**（因此称为“**钟形曲线**”）。
{% endnote %}

#### 解析日期（Parsing Dates）

##### 设置环境

我们需要做的第一件事是加载我们将使用的库和数据集。我们将使用包含`2007`年至`2016`年期间发生的山体滑坡信息的数据集。在下面的练习中，您将把新技能应用于全球地震数据集。
```python
# modules we'll use
import pandas as pd
import numpy as np
import seaborn as sns
import datetime

# read in our data
landslides = pd.read_csv("../input/landslide-events/catalog.csv")

# set seed for reproducibility
np.random.seed(0)
```
##### 检查日期列的数据类型

我们首先查看数据的前五行。
```python
landslides.head()
```
我们将使用`landslides`数据框中的“日期”列。让我们确保它实际上看起来包含日期。
```python
# print the first few rows of the date column
print(landslides['date'].head())
```
结果输出为：
```bash
0     3/2/07
1    3/22/07
2     4/6/07
3    4/14/07
4    4/15/07
Name: date, dtype: object
```
请注意，在`head()`输出的底部，您可以看到该列的数据类型是“`object`”。`Pandas`使用“`object`”数据类型来存储各种类型的数据类型，但大多数情况下，当您看到数据类型为“`object`”的列时，它会包含字符串。如果您在此处查看`pandas dtype`文档，您会注意到还有一个特定的`datetime64 dtypes`。因为我们列的数据类型是`object`而不是`datetime64`，所以我们可以看出`Python`不知道该列包含日期。我们还可以只查看列的`dtype`，而不打印前几行：
```python
# check the data type of our date column
landslides['date'].dtype
```
结果输出为：
```bash
dtype('O')
```
您可能需要检查`numpy`文档以将字母代码与对象的数据类型相匹配。“`O`”是“`object`”的代码，所以我们可以看到这两个方法给了我们相同的信息。

##### 将日期列转换为日期时间

现在，我们知道我们的日期列并未被视为日期，现在将其转换为日期了，以便将其视为日期。这称为“**解析日期**”，因为我们正在使用一个字符串并识别其组件部分。我们可以通过称为“`Strftime`指令”的指南来确定日期的格式，您可以在此链接中找到更多信息。基本思想是，您需要指出日期的哪些部分在哪里以及它们之间的标点符号是什么。日期有很多可能的部分，但最常见的是一天`％d`，一个月的`％m`，两位数的`％y`和四位数的`％y`。
- `1/17/07`的格式为“`%m/%d/%y`”。
- `17-1-2007`的格式为“`%d-%m-%Y`”。

回顾一下山体滑坡数据集中“日期”列的头部，我们可以看到它的格式是“月/日/两位数年份”，因此我们可以使用与第一个示例相同的语法来解析我们的日期。
```python
# create a new column, date_parsed, with the parsed dates
landslides['date_parsed'] = pd.to_datetime(landslides['date'], format="%m/%d/%y")
```
现在，当我检查新列的前几行时，我可以看到`dtype`是`datetime64`。我还可以看到我的日期已稍微重新排列，以便它们符合默认顺序日期时间对象（`year-month-day`）。
```python
# print the first few rows
landslides['date_parsed'].head()
```
结果输出为：
```bash
0   2007-03-02
1   2007-03-22
2   2007-04-06
3   2007-04-14
4   2007-04-15
Name: date_parsed, dtype: datetime64[ns]
```
现在我们的日期已正确解析，我们可以以有用的方式与它们交互。
- 如果我遇到多种日期格式错误怎么办？虽然我们在此处指定日期格式，但有时当单列中有多种日期格式时，您会遇到错误。如果发生这种情况，您可以让`pandas`尝试推断正确的日期格式应该是什么。你可以这样做：`landslides['date_parsed'] = pd.to_datetime(landslides['Date'], infer_datetime_format=True)`。
- 为什么不总是使用`infer_datetime_format = True`？不总是让`pandas`猜测时间格式有两个重要原因。首先，`pandas`并不总是能够找出正确的日期格式，特别是如果有人在数据输入方面发挥了创意。第二个是它比指定日期的确切格式慢得多。

##### 选择该月的某一天

现在我们有一列已解析的日期，我们可以提取信息，例如山体滑坡发生的月份中的哪一天。
```python
# get the day of the month from the date_parsed column
day_of_month_landslides = landslides['date_parsed'].dt.day
day_of_month_landslides.head()
```
结果输出为：
```bash
0     2.0
1    22.0
2     6.0
3    14.0
4    15.0
Name: date_parsed, dtype: float64
```
如果我们尝试从原始“日期”列中获取相同的信息，我们会收到错误：`AttributeError`：只能将`.dt`访问器与类似日期时间的值一起使用。这是因为`dt.day`不知道如何处理数据类型为“`object`”的列。尽管我们的数据帧中有日期，但我们必须先解析它们，然后才能以有用的方式与它们交互。

##### 绘制该月的日期来检查日期解析

解析日期的最大危险之一是混淆月份和日期。`to_datetime()`函数确实有非常有用的错误消息，但仔细检查我们提取的月份中的日期是否有意义也没有什么坏处。为此，我们绘制该月各天的直方图。我们预计它的值在`1`到`31`之间，并且由于没有理由认为山体滑坡在每月的某些日子比其他日子更常见，因此分布相对均匀。（`31`日有所下降，因为并非所有月份都有`31`天。）让我们看看情况是否如此：
```python
# remove na's
day_of_month_landslides = day_of_month_landslides.dropna()

# plot the day of the month
sns.distplot(day_of_month_landslides, kde=False, bins=31)
```
{% asset_img dc_5.png %}

看起来我们确实正确解析了日期，并且这张图对我来说很有意义。
