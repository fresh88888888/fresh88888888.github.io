---
title: 机器学习（初级）
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

#### 第一个机器学习模型

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

##### 什么是模型验证

您将需要评估几乎您构建的每个模型。在大多数（尽管不是全部）应用中，模型质量的相关衡量标准是预测准确性。 换句话说，模型的预测是否会接近实际发生的情况。许多人在衡量预测准确性时犯了一个巨大的错误。他们利用训练数据进行预测，并将这些预测与训练数据中的目标值进行比较。稍后您就会看到这种方法的问题以及如何解决它，但让我们首先考虑一下如何做到这一点。您首先需要以易于理解的方式总结模型质量。如果您比较`10,000`栋房屋的预测价值和实际价值，您可能会发现预测的好坏参半。查看包含`10,000`个预测值和实际值的列表是没有意义的。我们需要将其总结为一个指标。总结模型质量的指标有很多，但我们将从平均绝对误差（也称为`MAE`）开始。让我们从最后一个词“错误”开始分解这个指标。每栋房屋的预测误差为:
```python
error=actual−predicted
```
因此，如果一栋房子的价格为`150,000`美元，而您预测它将花费`100,000`美元，则错误为`50,000`美元。使用`MAE`指标，我们取每个误差的绝对值。这会将每个错误转换为正数。然后我们取这些绝对误差的平均值。这是我们衡量模型质量的标准。用简单的英语来说，可以说是:"平均而言，我们的预测偏差大约`X`"，为了计算`MAE`，我们首先需要一个模型。
```python
# Data Loading Code Hidden Here
import pandas as pd

# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)
```
一旦我们有了模型，我们就可以计算平均绝对误差：
```python
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
```
输出结果为：
```bash
434.71594577146544
```

##### “样本内”分数的问题

我们刚刚计算的度量可以称为“样本内”分数。我们使用单个房屋“样本”来构建模型并对其进行评估。这就是为什么这很糟糕。想象一下，在庞大的房地产市场中，门的颜色与房价无关。然而，在用于构建模型的数据样本中，所有带有绿色门的房屋都非常昂贵。 该模型的工作是找到预测房价的模式，因此它会看到这种模式，并且总是会预测带有绿色门的房屋的高价格。由于该模式是从训练数据中得出的，因此该模型在训练数据中将显得准确。但如果当模型看到新数据时这种模式不成立，那么该模型在实践中使用时就会非常不准确。由于模型的实用价值来自对新数据的预测，因此我们衡量未用于构建模型的数据的性能。最直接的方法是从模型构建过程中排除一些数据，然后使用这些数据来测试模型对以前从未见过的数据的准确性。该数据称为验证数据。

`scikit-learn`库有一个函数`train_test_split`将数据分成两部分。我们将使用其中一些数据作为训练数据来拟合模型，并使用其他数据作为验证数据来计算`mean_absolute_error`。
```python
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
```
输出结果为：
```bash
265806.91478373145
```
样本内数据的平均绝对误差约为`500`美元。样本外的金额超过`250,000`美元。这就是几乎完全正确的模型与无法用于大多数实际目的的模型之间的区别。作为参考，验证数据中的平均房屋价值为`110`万美元。因此，新数据的误差约为平均房价的四分之一。有很多方法可以改进这个模型，例如尝试寻找更好的特征或不同的模型类型。

#### 欠拟合和过拟合

您将了解欠拟合和过拟合的概念，并且您将能够应用这些想法使您的模型更加准确。

##### 尝试不同的模型

现在您已经有了衡量模型准确性的可靠方法，您可以尝试替代模型，看看哪个模型可以提供最佳预测。但是对于模型你还有什么选择呢？您可以在`scikit-learn`的文档中看到，决策树模型有很多选项（超出您长期以来想要或需要的选项）。
{% asset_img ml_4.png %}

在此步骤结束时，您将了解欠拟合和过拟合的概念，并且您将能够应用这些想法使您的模型更加准确。尝试不同的模型,现在您已经有了衡量模型准确性的可靠方法，您可以尝试替代模型，看看哪个模型可以提供最佳预测。但是对于模型你还有什么选择呢？深度`2`树实际上，一棵树在顶层（所有房屋）和叶子之间有`10`个裂缝的情况并不罕见。随着树变得更深，数据集被分割成具有更少房屋的叶子。如果一棵树只有`1`次分裂，它会将数据分为`2`组。如果将每组再次分割，我们将得到`4`组房屋。再次将每个组分开将创建`8`组。如果我们通过在每个级别添加更多拆分来继续将组数加倍，我们将有`210`个, 当我们到达第`10`层时，房屋群就会出现。那是`1024`片叶子。当我们将房屋划分为许多叶子时，每个叶子中的房屋也会减少。拥有很少房屋的叶子将做出与这些房屋的实际价值非常接近的预测，但它们可能对新数据做出非常不可靠的预测（因为每个预测仅基于少数房屋）。这是一种称为过度拟合的现象，其中模型与训练数据几乎完美匹配，但在验证和其他新数据中表现不佳。另一方面，如果我们把树做得很浅，它就不会把房子分成非常不同的组。在极端情况下，如果一棵树只将房屋分为`2`或`4`个，那么每个组中仍然有各种各样的房屋。对于大多数房屋来说，即使在训练数据中，最终的预测也可能相差很远（并且出于同样的原因，在验证中也会很糟糕）。当模型无法捕获数据中的重要区别和模式时，即使在训练数据中它也表现不佳，这称为欠拟合。由于我们关心新数据的准确性，这是我们根据验证数据估计的，因此我们希望找到欠拟合和过度拟合之间的最佳平衡点。从视觉上看，我们想要下图中（红色）验证曲线的低点。
{% asset_img ml_5.png %}

##### 举个例子

有几种控制树深度的替代方法，并且许多方法允许通过树的某些路线比其他路线具有更大的深度。但是`max_leaf_nodes`参数提供了一种非常明智的方法来控制过度拟合与欠拟合。我们允许模型制作的叶子越多，我们从上图中的欠拟合区域移动到过拟合区域的程度就越多。我们可以使用实用函数来帮助比较`max_leaf_nodes`不同值的`MAE`分数：
```python
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
```
使用您已经见过的代码（以及您已经编写的代码）将数据加载到`train_X、val_X、train_y 和 val_y`中。
```python
# Data Loading Code Runs At This Point
import pandas as pd
    
# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```
我们可以使用`for`循环来比较使用不同`max_leaf_nodes`值构建的模型的准确性。
```python
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
```
输出结果为：
```bash
Max leaf nodes: 5  		 Mean Absolute Error:  347380
Max leaf nodes: 50  		 Mean Absolute Error:  258171
Max leaf nodes: 500  		 Mean Absolute Error:  243495
Max leaf nodes: 5000  		 Mean Absolute Error:  254983
```
在列出的选项中，`500`是最佳叶子数。

##### 结论

要点如下：模型可能会遇到以下任一问题：
- 过度拟合：捕获未来不会重复出现的虚假模式，导致预测不太准确。
- 拟合不足：未能捕获相关模式，再次导致预测不太准确。
我们使用模型训练中未使用的验证数据来衡量候选模型的准确性。这让我们可以尝试许多候选模型并保留最好的一个。

#### 随机森林

##### 介绍

决策树会让您做出艰难的决定。一棵有很多叶子的深树会过度拟合，因为每个预测都来自于其叶子上的少数房屋的历史数据。但是，叶子很少的浅树会表现不佳，因为它无法捕获原始数据中的许多区别。即使当今最复杂的建模技术也面临着欠拟合和过拟合之间的紧张关系。但是，许多模型都有巧妙的想法，可以带来更好的性能。我们将以随机森林为例。随机森林使用许多树，它通过平均每个组成树的预测来进行预测。它通常比单个决策树具有更好的预测准确性，并且在使用默认参数时效果很好。如果继续建模，您可以学习更多具有更好性能的模型，但其中许多模型对获取正确的参数很敏感。

##### 举个例子

您已经看过几次加载数据的代码。数据加载结束时，我们有以下变量:
- `train_X`
- `val_X`
- `train_y`
- `val_y`

```python
import pandas as pd
    
# Load data
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
```
我们构建一个随机森林模型，类似于在`scikit-learn`中构建决策树的方式 - 这次使用`RandomForestRegressor`类而不是`DecisionTreeRegressor`。
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
```
输出结果为：
```bash
191669.7536453626
```
##### 结论

可能还有进一步改进的空间，但这比最佳决策树误差`250,000`有了很大的改进。有些参数允许您更改随机森林的性能，就像我们更改单个决策树的最大深度一样。但随机森林模型的最佳特征之一是，即使没有这种调整，它们通常也能正常工作。