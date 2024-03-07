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

#### 分类变量（Categorical Variables）

##### 介绍

分类变量仅采用有限数量的值。
- 考虑一项调查，询问您吃早餐的频率，并提供四个选项：“从不”、“很少”、“大多数天”或“每天”。在这种情况下，数据是分类的，因为响应属于一组固定的类别。
- 如果人们回答关于他们拥有什么品牌的汽车的调查，那么回答将分为“本田”、“丰田”和“福特”等类别。在这种情况下，数据也是分类的。

如果您尝试将这些变量插入到`Python`中的大多数机器学习模型中而不首先对其进行预处理，则会出现错误。我们将比较可用于准备分类数据的三种方法。

##### 三种方法

###### 删除分类变量

处理分类变量的最简单方法是将它们从数据集中删除。仅当列不包含有用信息时，此方法才有效。

###### 序数编码

序数编码将每个唯一值分配给不同的整数。
{% asset_img iml_4.png %}

此方法假设类别的顺序为：“从不”`(0)`<“很少”`(1)`<“大多数天”`(2)`<“每天”`(3)`。这个假设在这个例子中是有意义的，因为类别有无可争议的排名。并非所有类别变量的值都有明确的排序，但我们将那些具有明确排序的变量称为序数变量。对于基于树的模型（例如决策树和随机森林），您可以期望序数编码能够很好地处理序数变量。

###### 一次性编码

`One-hot`编码创建新列，指示原始数据中每个可能值的存在（或不存在）。为了理解这一点，我们将通过一个示例来进行操作。
{% asset_img iml_5.png %}

在原始数据集中，“颜色”是一个分类变量，具有三个类别：“红色”、“黄色”和“绿色”。相应的`one-hot`编码包含原始数据集中每个可能值的一列和每一行的一行。只要原始值为“`Red`”，我们就在“`Red`”列中输入`1`； 如果原始值为“黄色”，我们在“黄色”列中输入`1`，依此类推。与序数编码相反，`one-hot`编码不假设类别的顺序。因此，如果分类数据中没有明确的排序（例如，“红色”既不大于也不小于“黄色”），您可以预期这种方法会特别有效。我们将没有内在排名的分类变量称为名义变量。如果分类变量采用大量值（即，您通常不会将其用于采用超过`15`个不同值的变量），`One-hot`编码通常表现不佳。

###### 举例

我们将使用墨尔本住房数据集。我们不会关注数据加载步骤。相反，您可以想象您已经在`X_train、X_valid、y_train`和`y_valid`中拥有训练和验证数据。
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# Drop columns with missing values (simplest approach)
cols_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()] 
X_train_full.drop(cols_with_missing, axis=1, inplace=True)
X_valid_full.drop(cols_with_missing, axis=1, inplace=True)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
```
我们使用下面的`head()`方法查看训练数据。
```python
X_train.head()
```
{% asset_img iml_6.png %}

接下来，我们获得训练数据中所有分类变量的列表。我们通过检查每列的数据类型（或`dtype`）来做到这一点。对象数据类型指示列有文本（理论上它还可以是其他东西，但这对我们的目的来说并不重要）。对于此数据集，带有文本的列表示分类变量。
```python
# Get list of categorical variables
s = (X_train.dtypes == 'object')
object_cols = list(s[s].index)

print("Categorical variables:")
print(object_cols)
```
结果输出为：
```bash
Categorical variables:
['Type', 'Method', 'Regionname']
```
###### 定义衡量每种方法质量的函数

我们定义一个函数`Score_dataset()`来比较处理`calcategori`变量的三种不同方法。此函数报告随机森林模型的平均绝对误差(`MAE`)。一般来说，我们希望`MAE`尽可能低！
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Function for comparing different approaches
def score_dataset(X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds)
```
###### 方法一的得分（删除类别变量）

我们使用`select_dtypes()`方法删除对象列。
```python
drop_X_train = X_train.select_dtypes(exclude=['object'])
drop_X_valid = X_valid.select_dtypes(exclude=['object'])

print("MAE from Approach 1 (Drop categorical variables):")
print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
```
输出结果为：
```bash
MAE from Approach 1 (Drop categorical variables):
175703.48185157913
```
###### 方法二的得分（序数编码）

`Scikit-learn`有一个`OrdinalEncoder`类，可用于获取序数编码。我们循环分类变量并将序数编码器分别应用于每一列。
```python
from sklearn.preprocessing import OrdinalEncoder

# Make copy to avoid changing original data 
label_X_train = X_train.copy()
label_X_valid = X_valid.copy()

# Apply ordinal encoder to each column with categorical data
ordinal_encoder = OrdinalEncoder()
label_X_train[object_cols] = ordinal_encoder.fit_transform(X_train[object_cols])
label_X_valid[object_cols] = ordinal_encoder.transform(X_valid[object_cols])

print("MAE from Approach 2 (Ordinal Encoding):") 
print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
```
输出结果为：
```bash
MAE from Approach 2 (Ordinal Encoding):
165936.40548390493
```
在上面的代码单元中，对于每一列，我们将每个唯一值随机分配给不同的整数，这是一种常见的方法，比提供自定义标签更简单；然而，如果我们为所有序数变量提供更明智的标签，我们可以期待性能的进一步提升。

###### 方法三的得分（One-Hot 编码）

我们使用`scikit-learn`中的`OneHotEncoder`类来获取`one-hot`编码。有许多参数可用于自定义其行为。
- 我们设置`handle_unknown ='ignore'`以避免当验证数据包含训练数据中未表示的类时出现错误。
- 设置稀疏`= False`确保编码列作为`numpy`数组（而不是稀疏矩阵）返回。

为了使用编码器，我们只提供我们想要进行`one-hot`编码的分类列。例如，为了对训练数据进行编码，我们提供`X_train[object_cols]`。（下面代码单元中的`object_cols`是包含分类数据的列名称列表，因此`X_train[object_cols]`包含训练集中的所有分类数据。）
```python
from sklearn.preprocessing import OneHotEncoder

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index
OH_cols_valid.index = X_valid.index

# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

# Ensure all columns have string type
OH_X_train.columns = OH_X_train.columns.astype(str)
OH_X_valid.columns = OH_X_valid.columns.astype(str)

print("MAE from Approach 3 (One-Hot Encoding):") 
print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))
```
输出结果为：
```bash
MAE from Approach 3 (One-Hot Encoding):
166089.4893009678
```
##### 哪种方法最好？

在这种情况下，删除分类列（方法`1`）效果最差，因为它的`MAE`得分最高。至于其他两种方法，由于返回的`MAE`分数的值非常接近，因此其中一种方法似乎没有比另一种方法有任何有意义的好处。一般来说，`one-hot`编码（方法`3`）通常会表现最佳，而删除分类列（方法`1`）通常表现最差，但具体情况会有所不同。

##### 总结

世界充满了分类数据。如果您知道如何使用这种常见数据类型，您将成为一名更高效的数据科学家！

#### 管道

##### 介绍

管道是保持数据预处理和建模代码井井有条的简单方法。具体来说，管道捆绑了预处理和建模步骤，因此您可以像使用单个步骤一样使用整个捆绑包。许多数据科学家在没有管道的情况下组合模型，但管道有一些重要的好处。其中包括：
- 更清晰的代码：在预处理的每个步骤中计算数据可能会变得混乱。使用管道，您无需在每个步骤中手动跟踪训练和验证数据。
- 错误更少：误用步骤或忘记预处理步骤的机会更少。
- 更容易生产：将模型从原型转变为可大规模部署的模型可能非常困难。我们不会在这里讨论许多相关的问题，但管道可以提供帮助。
- 模型验证的更多选项

##### 举例

我们将使用墨尔本住房数据集。我们不会关注数据加载步骤。相反，您可以想象您已经在`X_train、X_valid、y_train`和`y_valid`中拥有训练和验证数据。
```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Separate target from predictors
y = data.Price
X = data.drop(['Price'], axis=1)

# Divide data into training and validation subsets
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
```
我们使用下面的`head()`方法查看训练数据。请注意，数据包含分类数据和具有缺失值的列。有了管道，就可以轻松处理这两件事！
```python
X_train.head()
```
输出结果为：
{% asset_img iml_7.png %}
我们分三步构建完整的管道。

###### 第一步：定义预处理步骤

与管道如何将预处理和建模步骤捆绑在一起类似，我们使用`ColumnTransformer`类将不同的预处理步骤捆绑在一起。代码如下：
- 估算数值数据中的缺失值。
- 估算缺失值并对分类数据应用`one-hot`编码。

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='constant')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```
###### 第二步：定义模型

我们使用熟悉的`RandomForestRegressor`类定义随机森林模型
```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, random_state=0)
```
###### 第三步：创建并评估管道

最后，我们使用`Pipeline`类来定义捆绑预处理和建模步骤的管道。有一些重要的事情需要注意：
- 通过管道，我们预处理训练数据并在一行代码中拟合模型。（相反，如果没有管道，我们必须在单独的步骤中进行插补、`one-hot`编码和模型训练。如果我们必须处理数值变量和分类变量，这会变得特别混乱！）
- 通过管道，我们将`X_valid`中未处理的特征提供给`Predict()`命令，管道在生成预测之前自动预处理这些特征。（但是，如果没有管道，我们必须记住在进行预测之前对验证数据进行预处理。）

```python
from sklearn.metrics import mean_absolute_error

# Bundle preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)
```
输出结果为：
```bash
MAE: 160679.18917034855
```
###### 结论

管道对于清理机器学习代码和避免错误非常有价值，对于具有复杂数据预处理的工作流程尤其有用。

#### 交叉验证（Cross-Validation）

##### 介绍

机器学习是一个迭代过程。您将面临有关使用哪些预测变量、使用什么类型的模型、为这些模型提供哪些参数等的选择。到目前为止，您已经通过验证来衡量模型质量，以数据驱动的方式做出了这些选择（ 或坚持）设置。但这种方法有一些缺点。 要看到这一点，假设您有一个包含`5000`行的数据集。您通常会保留大约`20%`的数据作为验证数据集，即`1000`行。但这在确定模型分数时留下了一些随机机会。也就是说，模型可能在一组`1000`行上表现良好，即使它在不同的`1000`行上可能不准确。在极端情况下，您可以想象验证集中只有`1`行数据。如果您比较其他模型，哪个模型对单个数据点做出最好的预测将主要取决于运气！一般来说，验证集越大，我们衡量模型质量的随机性（也称为“噪声”）就越少，并且越可靠。不幸的是，我们只能通过从训练数据中删除行来获得大的验证集，而较小的训练数据集意味着更差的模型！

##### 什么是交叉验证？

在交叉验证中，我们对不同的数据子集运行建模过程，以获得模型质量的多种度量。例如，我们可以首先将数据分为`5`部分，每部分占完整数据集的`20%`。在本例中，我们说我们已将数据分成`5`个“折叠”。
{% asset_img iml_8.png %}

然后，我们为每个折叠运行一个实验：
- 在实验`1`中，我们使用第一次折叠作为验证（或保留）集，其他所有内容作为训练数据。这为我们提供了基于`20%`保留集的模型质量衡量标准。
- 在实验`2`中，我们保留第二次折叠中的数据（并使用除第二次折叠之外的所有数据来训练模型）。然后使用保留集来获得模型质量的第二次估计。
- 我们重复这个过程，使用每个折叠一次作为保留集。 总而言之。`100%`的数据在某个时刻被用作保留，我们最终得到基于数据集中所有行的模型质量度量（即使我们不同时使用所有行）。

##### 什么时候应该使用交叉验证？

交叉验证可以更准确地衡量模型质量，如果您要做出大量建模决策，这一点尤其重要。但是，它可能需要更长的时间来运行，因为它估计多个模型（每个折叠一个）。那么，考虑到这些权衡，您应该何时使用每种方法？
- 对于小型数据集，额外的计算负担并不是什么大问题，您应该运行交叉验证。
- 对于较大的数据集，单个验证集就足够了。您的代码将运行得更快，并且您可能拥有足够的数据，几乎不需要重复使用其中的一些数据来保留。

对于什么构成大数据集和小数据集，没有简单的阈值。但是，如果您的模型需要几分钟或更短的时间才能运行，则可能值得切换到交叉验证。或者，您可以运行交叉验证，看看每个实验的分数是否看起来很接近。如果每个实验产生相同的结果，则单个验证集可能就足够了。

##### 举例

我们将使用与上一个教程中相同的数据。我们将输入数据加载到`X`中，将输出数据加载到`y`中。
```python
import pandas as pd

# Read the data
data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price
```
然后，我们定义一个管道，使用输入器来填充缺失值，并使用随机森林模型来进行预测。虽然可以在没有管道的情况下进行交叉验证，但这非常困难！使用管道将使代码变得非常简单。
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
                              ('model', RandomForestRegressor(n_estimators=50,
                                                              random_state=0))])
```
我们使用`scikit-learn`中的`cross_val_score()`函数获取交叉验证分数。我们使用`cv`参数设置折叠次数。
```python
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)
```
结果输出为：
```bash
MAE scores:[301628.7893587  303164.4782723  287298.331666   236061.84754543 260383.45111427]
```
评分参数选择要报告的模型质量度量：在本例中，我们选择负平均绝对误差 (`MAE`)。`scikit-learn`的文档显示了选项列表。我们指定负`MAE`有点令人惊讶。`Scikit-learn`有一个约定，其中定义了所有指标，因此数字越大越好。在这里使用负数可以使它们与该约定保持一致，尽管负`MAE`在其他地方几乎闻所未闻。我们通常需要单一的模型质量度量来比较替代模型。所以我们取实验的平均值。
```python
print("Average MAE score (across experiments):")
print(scores.mean())
```

```bash
Average MAE score (across experiments):
277707.3795913405
```
##### 结论

使用交叉验证可以更好地衡量模型质量，并具有清理代码的额外好处：请注意，我们不再需要跟踪单独的训练集和验证集。因此，特别是对于小型数据集，这是一个很好的改进！

