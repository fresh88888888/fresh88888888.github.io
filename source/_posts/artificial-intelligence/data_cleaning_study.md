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
