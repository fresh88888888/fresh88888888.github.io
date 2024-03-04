---
title: 机器学习
date: 2024-03-04 18:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 基础数据探索

##### Pandas

任何机器学习项目的第一步都是熟悉数据。 为此，您将使用`Pandas`库。 `Pandas`是数据科学家用于探索和操作数据的主要工具。大多数人在代码中将`pandas`缩写为`pd`。我们用命令来做到这一点。
```python
import pandas as pd
```
<!-- more -->
`Pandas`库最重要的部分是`DataFrame`。 `DataFrame`保存您可能认为是表格的数据类型。这类似于`Excel`中的工作表或`SQL`数据库中的表。`Pandas`拥有强大的方法来处理您想要对此类数据执行的大多数操作。例如，我们将查看澳大利亚墨尔本的房价数据。在实践练习中，您将向新数据集应用相同的过程，该数据集包含爱荷华州的房价。示例（墨尔本）数据位于文件路径 `../input/melbourne-housing-snapshot/melb_data.csv`，我们使用以下命令加载并探索数据：
```python
import oandas as pd

# save filepath to variable for easier access
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 
# print a summary of the data in Melbourne data
melbourne_data.describe()
```
结果输出为：
{% asset_img ml_1.png %}

##### 解释数据描述

结果显示原始数据集中每列`8`个数字。第一个数字是计数，显示有多少行具有非缺失值。缺失值的产生有多种原因。例如，在测量一卧室房屋时，不会收集第二卧室的尺寸。我们将回到丢失数据的主题。第二个值是`mean`，即平均值。其中，`std`是标准差，它衡量值在数值上的分布情况。要解释最小值、`25%、50%、75%`和最大值，请想象将每列从最低值到最高值排序。 第一个（最小）值是最小值。 如果您浏览列表的四分之一，您会发现一个大于值的`25%`且小于值的`75%`的数字。即`25%`的值。 第`50`个百分位数和第`75`个百分位数的定义类似，其中`max`是最大的数字。

#### 您的第一个机器学习模型

##### 选择建模数据

您的数据集有太多变量，难以理解，甚至无法很好地打印出来。如何将如此大量的数据简化为您可以理解的内容？我们将首先根据直觉选择一些变量。后面的课程将向您展示自动对变量进行优先级排序的统计技术。要选择变量/列，我们需要查看数据集中所有列的列表。这是通过`DataFrame`的`columns`属性完成的（下面的代码的底行）。
```python
import pandas as pd

melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns
```
输出结果为：
```bash
Index(['Suburb', 'Address', 'Rooms', 'Type', 'Price', 'Method', 'SellerG',
       'Date', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car',
       'Landsize', 'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude',
       'Longtitude', 'Regionname', 'Propertycount'],
      dtype='object')
```

```python
# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.  
# Your Iowa data doesn't have missing values in the columns you use. 
# So we will take the simplest option for now, and drop houses from our data. 
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)
```
有多种方法可以选择数据子集。但我们现在将重点关注两种方法:
- 点表示法，我们用它来选择“`prediction target`”
- 使用列列表进行选择，我们用它来选择“`features`”

##### 选择预测目标

您可以使用点符号提取变量。这个单列存储在一个`Series`中，它大致就像一个只有单列数据的`DataFrame`。我们将使用点符号来选择我们想要预测的列，这称为预测目标。按照惯例，预测目标称为`y`。所以我们需要保存墨尔本数据中的房价的代码是:
```python
y = melbourne_data.Price
```

##### 选择“Features”

输入到我们的模型中（随后用于进行预测）的列称为“特征”。在我们的例子中，这些列将用于确定房价。有时，您将使用除目标之外的所有列作为特征。其他时候，使用更少的功能会更好。现在，我们将构建一个仅包含几个特征的模型。稍后您将了解如何迭代和比较使用不同功能构建的模型。我们通过在括号内提供列名称列表来选择多个功能。该列表中的每个项目都应该是一个字符串（带引号）。
```python
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
```
按照惯例，该数据称为`X`:
```python
X = melbourne_data[melbourne_features]
```
让我们快速回顾一下我们将使用describe方法和head方法来预测房价的数据，该方法显示了前几行。
```python
X.describe()
```
输出结果为：
{% asset_img ml_2.png %}

```python
X.head()
```
输出结果为：
{% asset_img ml_3.png %}
使用这些命令直观地检查数据是数据科学家工作的重要组成部分。您经常会在数据集中发现值得进一步检查的惊喜。

##### 建立你的模型

您将使用`scikit-learn`库来创建模型。编码时，该库被编写为`sklearn`，正如您将在示例代码中看到的那样。`Scikit-learn`无疑是最流行的库，用于对通常存储在`DataFrame`中的数据类型进行建模。构建和使用模型的步骤是：
- 定义：它将是什么类型的模型？决策树？其他类型的模型？还指定了模型类型的一些其他参数。
- 拟合：从提供的数据中捕获模式。这是建模的核心。
- 预测：正如听起来的那样。
- 评估：确定模型预测的准确性。
以下是使用`scikit-learn`定义决策树模型并将其与特征和目标变量进行拟合的示例。
```python
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)
```
输出结果为：
```bash
DecisionTreeRegressor(random_state=1)
```
许多机器学习模型允许模型训练具有一定的随机性。为random_state指定一个数字可确保您在每次运行中获得相同的结果。 这被认为是一个很好的做法。您使用任何数字，模型质量不会完全取决于您选择的值。我们现在有了一个可以用来进行预测的拟合模型。在实践中，您需要对市场上即将上市的新房屋进行预测，而不是对我们已经有价格的房屋进行预测。 但我们将对训练数据的前几行进行预测，以了解预测函数的工作原理。
```python
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
```
输出结果为：
```bash
Making predictions for the following 5 houses:
   Rooms  Bathroom  Landsize  Lattitude  Longtitude
1      2       1.0     156.0   -37.8079    144.9934
2      3       2.0     134.0   -37.8093    144.9944
4      4       1.0     120.0   -37.8072    144.9941
6      3       2.0     245.0   -37.8024    144.9993
7      2       1.0     256.0   -37.8060    144.9954
The predictions are
[1035000. 1465000. 1600000. 1876000. 1636000.]
```

#### 模型验证

