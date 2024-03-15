---
title: 特征工程实践之——房价
date: 2024-03-15 10:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 第1步 - 准备工作

##### 导入和配置

我们将首先导入使用的包并设置一些笔记本默认值。
```python
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor


# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

# Mute warnings
warnings.filterwarnings('ignore')
```
##### 数据预处理

在进行任何特征工程之前，我们需要对数据进行预处理，以使其成为适合分析的形式。我们在课程中使用的数据比比赛数据简单一些。对于艾姆斯竞赛数据集，我们需要：
- 从`CSV`文件加载数据。
- 清理数据以修复任何错误或不一致。
- 对统计数据类型（数字、分类）进行编码。
- 估算任何缺失值。

###### 加载数据

我们将把所有这些步骤包装在一个函数中，这将使您可以在需要时轻松获得新的数据帧。读取`CSV`文件后，我们将应用三个预处理步骤：清理、编码和插补，然后创建数据分割：一个 (`df_train`) 用于训练模型，另一个 (`df_test`) 用于进行预测。
```python
def load_data():
    # Read data
    data_dir = Path("../input/house-prices-advanced-regression-techniques/")
    df_train = pd.read_csv(data_dir / "train.csv", index_col="Id")
    df_test = pd.read_csv(data_dir / "test.csv", index_col="Id")
    # Merge the splits so we can process them together
    df = pd.concat([df_train, df_test])
    # Preprocessing
    df = clean(df)
    df = encode(df)
    df = impute(df)
    # Reform splits
    df_train = df.loc[df_train.index, :]
    df_test = df.loc[df_test.index, :]
    return df_train, df_test
```
###### 清理数据

该数据集中的一些分类特征在其类别中存在明显的拼写错误：
```python
data_dir = Path("../input/house-prices-advanced-regression-techniques/")
df = pd.read_csv(data_dir / "train.csv", index_col="Id")

df.Exterior2nd.unique()
```
结果输出为：
```bash
array(['VinylSd', 'MetalSd', 'Wd Shng', 'HdBoard', 'Plywood', 'Wd Sdng',
       'CmentBd', 'BrkFace', 'Stucco', 'AsbShng', 'Brk Cmn', 'ImStucc',
       'AsphShn', 'Stone', 'Other', 'CBlock'], dtype=object)
```
将它们与`data_description.txt`进行比较可以向我们展示哪些内容需要清理。我们将在这里解决几个问题，但您可能需要进一步评估这些数据。
```python
def clean(df):
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    # Some values of GarageYrBlt are corrupt, so we'll replace them
    # with the year the house was built
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    # Names beginning with numbers are awkward to work with
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
    }, inplace=True,
    )
    return df
```
###### 对统计数据类型进行编码

`Pandas`具有与标准统计类型（数值、分类等）相对应的`Python`类型。使用正确的类型对每个特征进行编码有助于确保我们使用的任何函数都适当地处理每个特征，并使我们更容易一致地应用转换。该隐藏单元定义了编码函数：
```python
# The numeric features are already encoded correctly (`float` for
# continuous, `int` for discrete), but the categoricals we'll need to
# do ourselves. Note in particular, that the `MSSubClass` feature is
# read as an `int` type, but is actually a (nominative) categorical.

# The nominative (unordered) categorical features
features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]


# The ordinal (ordered) categorical features 

# Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Add a None level for missing values
ordered_levels = {key: ["None"] + value for key, value in
                  ordered_levels.items()}


def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name] = df[name].cat.add_categories("None")
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    return df
```
###### 处理缺失值

现在处理缺失值将使特征工程进行得更加顺利。我们将缺失数值归为`0`，将缺失分类值归为“无”。您可能想尝试其他插补策略。特别是，您可以尝试创建“缺失值”指标：每当估算值时为`1`，否则为 `0`。
```python
def impute(df):
    for name in df.select_dtypes("number"):
        df[name] = df[name].fillna(0)
    for name in df.select_dtypes("category"):
        df[name] = df[name].fillna("None")
    return df
```

##### 加载数据

现在我们可以调用数据加载器并获取处理后的数据分割：
```python
df_train, df_test = load_data()
```
如果您想查看它们包含的内容，请取消注释并运行此单元格。请注意，`df_test`缺少`SalePrice`值。（在插补步骤中`NA`被设为`0`。）

##### 建立基线

最后，让我们建立一个基线分数来判断我们的特征工程。它将计算功能集的交叉验证`RMSLE`分数。我们的模型使用了`XGBoost`，但您可以想尝试其他模型。
```python
def score_dataset(X, y, model=XGBRegressor()):
    # Label encoding for categoricals
    #
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
```
当我们想要尝试新的特征集时，我们可以随时重用这个评分函数。我们现在将在没有附加功能的处理数据上运行它并获得基线分数：
```python
X = df_train.copy()
y = X.pop("SalePrice")

baseline_score = score_dataset(X, y)
print(f"Baseline score: {baseline_score:.5f} RMSLE")
```
结果输出为：
```bash
Baseline score: 0.14302 RMSLE
```
这个基线分数可以帮助我们了解我们组装的某些特征是否实际上带来了任何改进。

#### 第2步 - 特征有效分值

我们了解了如何使用互信息来计算某个特征的有效分值，让您了解该特征的潜力有多大。
```python
def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores


def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
```
让我们再次看看我们的特征得分:
```python
X = df_train.copy()
y = X.pop("SalePrice")

mi_scores = make_mi_scores(X, y)
mi_scores
```
结果输出为：
```bash
OverallQual     0.571457
Neighborhood    0.526220
GrLivArea       0.430395
YearBuilt       0.407974
LotArea         0.394468
                  ...   
PoolQC          0.000000
MiscFeature     0.000000
MiscVal         0.000000
MoSold          0.000000
YrSold          0.000000
Name: MI Scores, Length: 79, dtype: float64
```
您可以看到，我们有许多信息丰富的特征，但也有一些特征似乎根本没有信息（至少其本身）。得分最高的特征通常会在特征开发过程中获得最大的回报，因此将精力集中在这些特征上可能是个好主意。 另一方面，对无信息特征的训练可能会导致过度拟合。因此，我们将完全放弃得分为`0.0`的特征：
