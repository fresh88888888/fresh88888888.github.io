---
title: 机器学习（中级）
date: 2024-03-07 10:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 介绍

- 处理现实世界数据集中常见的数据类型（缺失值、分类变量）。
- 设计管道以提高机器学习代码的质量。
- 使用先进的技术进行模型验证（交叉验证）。
- 构建最先进的模型，广泛用于赢得`Kaggle`比赛(`XGBoost`)。
- 避免常见且重要的数据科学错误（泄漏）。

#### 缺失值（Missing Values）

您将学习三种处理缺失值的方法。然后，您将在现实数据集上比较这些方法的有效性。
<!-- more -->

##### 介绍

数据最终可能会出现缺失值的情况有很多。例如：
- 两居室的房子不包括第三间卧室的尺寸值。
- 调查受访者可以选择不分享他的收入。

如果您尝试使用缺失值的数据构建模型，大多数机器学习库（包括`scikit-learn`）都会出错。因此，您需要选择以下策略之一。

##### 缺失值处理的三种方法

###### 1.删除缺失值的列

最简单的选择是删除缺失值的列。
{% asset_img iml_1.png %}

除非删除的列中的大多丢失数值，否则模型将无法使用此方法访问大量（可能有用！）信息。作为一个极端的示例，请考虑一个包含`10,000`行的数据集，其中一个重要列缺少单个条目。这种方法会完全删除该列！

###### 2.更好的选择：插补

插补用一些数字填充缺失值。例如，我们可以填写每列的平均值。
{% asset_img iml_2.png %}

在大多数情况下，估算值并不完全正确，但与完全删除列相比，它通常会产生更准确的模型。

###### 3.插补的扩展

插补是标准方法，通常效果很好。但是，估算值可能系统地高于或低于其实际值（未在数据集中收集）。或者，具有缺失值的行可能以其他方式是唯一的。在这种情况下，您的模型将通过考虑最初丢失的值来做出更好的预测。
{% asset_img iml_3.png %}

在这种方法中，我们像以前一样估算缺失值。此外，对于原始数据集中缺少条目的每一列，我们添加一个新列来显示估算条目的位置。就我而言，这将显着改善结果。在其他情况下，它根本没有帮助。

##### 举例

在示例中，我们将使用墨尔本住房数据集。我们的模型将使用房间数量和土地面积等信息来预测房价。我们不会关注数据加载步骤。相反，您可以想象您已经在`X_train、X_valid、y_train 和 y_valid`中拥有训练和验证数据。
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select target
y = data.Price

# To keep things simple, we'll use only numerical predictors
melb_predictors = data.drop(['Price'], axis=1)
X = melb_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)
```
###### 定义衡量每种方法质量的函数

我们定义一个函数`Score_dataset()`来比较处理缺失值的不同方法。此函数报告随机森林模型的平均绝对误差(`MAE`)。
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```
###### 方法一的得分（删除具有缺失值的列）

由于我们同时使用训练集和验证集，因此我们会小心地在两个`DataFrame`中删除相同的列。
```python
# Get names of columns with missing values
cols_with_missing = [col for col in X_train.columns
                     if X_train[col].isnull().any()]

# Drop columns in training and validation data
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)

print("MAE from Approach 1 (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))
```
输出结果为：
```bash
MAE from Approach 1 (Drop columns with missing values):
183550.22137772635
```
###### 方法二的得分（插补）

接下来，我们使用`SimpleImputer`将缺失值替换为每列的平均值。虽然很简单，但填充平均值通常效果很好（但这因数据集而异）。虽然统计学家尝试了更复杂的方法来确定估算值（例如回归估算），但一旦将结果插入复杂的机器学习模型，复杂的策略通常不会带来额外的好处。
```python
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

print("MAE from Approach 2 (Imputation):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))
```
输出结果为：
```bash
MAE from Approach 2 (Imputation):
178166.46269899711
```
我们看到方法`2`的`MAE`低于方法`1`，因此方法`2`在此数据集上表现更好。

###### 方法三的得分（插补的扩展）

接下来，我们估算缺失值，同时还跟踪估算了哪些值。
```python
# Make copy to avoid changing original data (when imputing)
X_train_plus = X_train.copy()
X_valid_plus = X_valid.copy()

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

print("MAE from Approach 3 (An Extension to Imputation):")
print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))
```
输出结果为：
```bash
MAE from Approach 3 (An Extension to Imputation):
178927.503183954
```
正如我们所看到的，方法`3`的表现比方法`2`稍差。

###### 那么，为什么插补比删除列表现更好呢？

训练数据有`10864`行和`12`列，其中`3`列包含缺失数据。对于每一列，缺失的条目不到一半。因此，删除列会删除很多有用的信息，因此插补会表现得更好是有道理的。
```python
# Shape of training data (num_rows, num_columns)
print(X_train.shape)

# Number of missing values in each column of training data
missing_val_count_by_column = (X_train.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0])
```
```bash
(10864, 12)
Car               49
BuildingArea    5156
YearBuilt       4307
dtype: int64
```
##### 结论

通常，相对于我们简单地删除包含缺失值的列（在方法`1`中），估算缺失值（在方法`2`和方法`3`中）会产生更好的结果。
