---
title: 时间序列（Python）
date: 2024-03-19 16:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 介绍

预测可能是机器学习在现实世界中最常见的应用。企业预测产品需求，政府预测经济和人口增长，气象学家预测天气。对未来事物的理解是科学、政府和工业界的迫切需求，这些领域的从业者越来越多地应用机器学习来满足这一需求。时间序列预测是一个广阔的领域，有着悠久的历史。 
- 工程师对主要时间序列组成部分（趋势、季节和周期）进行建模。
- 使用多种时间序列图可视化时间序列。
- 创建结合互补模型优势​​的预测混合体。
- 使机器学习方法适应各种预测任务。
<!-- more -->
##### 什么是时间序列

预测的基本对象是**时间序列**，它是随时间记录的一组观测值。在预测应用中，通常以固定频率（例如每天或每月）记录观测结果。
```python
import pandas as pd

df = pd.read_csv(
    "../input/ts-course-data/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

df.head()
```
结果输出为：
```bash
	    Hardcover
Date	
2000-04-01	139
2000-04-02	128
2000-04-03	172
2000-04-04	139
2000-04-05	191
```
该系列记录了零售店`30`天内的精装书销售数量。请注意，我们有一个带有时间索引日期的观察精装列。

##### 时间序列的线性回归

我们将使用线性回归算法来构建预测模型。线性回归在实践中广泛使用，并且自然地适应复杂的预测任务。线性回归算法学习如何根据其输入特征进行加权和。对于两个特征，我们将有：
```python
target = weight_1 * feature_1 + weight_2 * feature_2 + bias
```
在训练期间，回归算法会学习最适合目标的参数`weight_1、weight_2`和偏差的值。（该算法通常称为普通最小二乘法，因为它选择最小化目标与预测之间的平方误差的值。）权重也称为**回归系数**，偏差也称为**截距**，因为它告诉您该图的位置函数与`y`轴交叉。

###### 时间步长特征

时间序列特有的特征有两种：**时间步长特征和滞后特征**。时间步特征是我们可以直接从时间索引导出的特征。最基本的时间步长功能是`time dummy`，它计算从开始到结束的时间步长。
```python
import numpy as np

df['Time'] = np.arange(len(df.index))

df.head()
```
结果输出为：
```bash
	   Hardcover  Time
Date		
2000-04-01	139	0
2000-04-02	128	1
2000-04-03	172	2
2000-04-04	139	3
2000-04-05	191	4
```
使用`time dummy`变量的线性回归生成模型：
```bash
target = weight * time + bias
```
然后，`time dummy`对象让我们将曲线拟合到时间图中的时间序列，其中时间形成`x`轴。
```python
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
```
{% asset_img ts_1.png %}

时间步特征可让您对时间依赖性进行建模。如果序列的值可以从发生的时间预测，则序列是时间相关的。在精装销售系列中，我们可以预测本月晚些时候的销量通常会高于本月早些时候的销量。

###### 滞后特征

为了制作滞后特征，我们改变了目标序列的观察结果，使它们看起来是在较晚的时间发生的。在这里，我们创建了`1`步滞后特征，尽管也可以进行多步移动。
```python
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

df.head()
```
结果输出为：
```bash
	Hardcover	Lag_1
Date		
2000-04-01	139	NaN
2000-04-02	128	139.0
2000-04-03	172	128.0
2000-04-04	139	172.0
2000-04-05	191	139.0
```
具有滞后特征的线性回归产生模型：
```bash
target = weight * lag + bias
```
因此，滞后特征让我们可以将曲线拟合到滞后图上，其中系列中的每个观测值都根据先前的观测值进行绘制。
```python
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales');
```
{% asset_img ts_2.png %}

您可以从滞后图中看到，一天的销售额（精装本）与前一天的销售额 (`Lag_1`) 相关。当您看到这样的关系时，您就知道滞后特征会很有用。更一般地说，滞后特征可让您对序列依赖性进行建模。当可以根据先前的观察来预测观察时，时间序列具有序列依赖性。在精装销售中，我们可以预测一天的高销量通常意味着第二天的高销量。将机器学习算法应用于时间序列问题主要是关于时间索引和滞后的特征工程。我们使用线性回归，因为它简单，但无论您为预测任务选择哪种算法，这些特征都将很有用。

##### 举例 - 隧道流量

`Tunnel Traffic`是一个时间序列，描述从`2003`年`11`月到`2005`年`11`月每天通过瑞士`Baregg`隧道的车辆数量。在这个示例中，我们将进行一些将线性回归应用于时间步长特征和滞后特征的练习。
```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 4))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'


# Load Tunnel Traffic dataset
data_dir = Path("../input/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])

# Create a time series in Pandas by setting the index to a date
# column. We parsed "Day" as a date type by using `parse_dates` when
# loading the data.
tunnel = tunnel.set_index("Day")

# By default, Pandas creates a `DatetimeIndex` with dtype `Timestamp`
# (equivalent to `np.datetime64`, representing a time series as a
# sequence of measurements taken at single moments. A `PeriodIndex`,
# on the other hand, represents a time series as a sequence of
# quantities accumulated over periods of time. Periods are often
# easier to work with, so that's what we'll use in this course.
tunnel = tunnel.to_period()

tunnel.head()
```
结果输出为：
```bash
	NumVehicles
Day	
2003-11-01	103536
2003-11-02	92051
2003-11-03	100795
2003-11-04	102352
2003-11-05	106569
```
###### 时间步长特征

如果时间序列没有任何缺失的日期，我们可以通过计算序列的长度来创建`time dummy`值。
```python
df = tunnel.copy()

df['Time'] = np.arange(len(tunnel.index))

df.head()
```
结果输出为：
```bash
	NumVehicles	Time
Day		
2003-11-01	103536	0
2003-11-02	92051	1
2003-11-03	100795	2
2003-11-04	102352	3
2003-11-05	106569	4
```
拟合线性回归模型的过程遵循`scikit-learn`的标准步骤。
```python
from sklearn.linear_model import LinearRegression

# Training data
X = df.loc[:, ['Time']]  # features
y = df.loc[:, 'NumVehicles']  # target

# Train the model
model = LinearRegression()
model.fit(X, y)

# Store the fitted values as a time series with the same time index as
# the training data
y_pred = pd.Series(model.predict(X), index=X.index)
```
实际创建的模型为（大约）：车辆 = `22.5` * 时间 + `98176`。绘制随时间变化的拟合值向我们展示了如何将线性回归拟合到`time dummy`变量来创建由该方程定义的趋势线。
```python
ax = y.plot(**plot_params)
ax = y_pred.plot(ax=ax, linewidth=3)
ax.set_title('Time Plot of Tunnel Traffic');
```
{% asset_img ts_3.png %}

###### 滞后特征

`Pandas`为我们提供了一种简单的滞后序列的方法，即移位方法。
```python
df['Lag_1'] = df['NumVehicles'].shift(1)
df.head()
```
结果输出为：
```bash
	NumVehicles	Time	Lag_1
Day			
2003-11-01	103536	0	NaN
2003-11-02	92051	1	103536.0
2003-11-03	100795	2	92051.0
2003-11-04	102352	3	100795.0
2003-11-05	106569	4	102352.0
```
创建滞后特征时，我们需要决定如何处理产生的缺失值。填充它们是一种选择，可能使用`0.0`或使用第一个已知值“回填”。相反，我们将删除缺失的值，确保也删除目标中相应日期的值。
```python
from sklearn.linear_model import LinearRegression

X = df.loc[:, ['Lag_1']]
X.dropna(inplace=True)  # drop missing values in the feature set
y = df.loc[:, 'NumVehicles']  # create the target
y, X = y.align(X, join='inner')  # drop corresponding values in target

model = LinearRegression()
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
```
滞后图向我们展示了我们能够如何很好地拟合一天的车辆数量与前一天的车辆数量之间的关系。
```python
fig, ax = plt.subplots()
ax.plot(X['Lag_1'], y, '.', color='0.25')
ax.plot(X['Lag_1'], y_pred)
ax.set_aspect('equal')
ax.set_ylabel('NumVehicles')
ax.set_xlabel('Lag_1')
ax.set_title('Lag Plot of Tunnel Traffic');
```
{% asset_img ts_4.png %}

滞后特征的预测对于我们预测随时间变化的序列的效果意味着什么？下面的时间图向我们展示了我们的预测现在如何响应该系列最近的行为。
```python
ax = y.plot(**plot_params)
ax = y_pred.plot()
```
结果输出为：
{% asset_img ts_5.png %}

最好的时间序列模型通常包含时间步长特征和滞后特征的某种组合。在接下来，我们将学习如何使用本课程中的特征作为起点来设计特征，对时间序列中最常见的模式进行建模。

#### 趋势（Trend）

##### 什么是趋势？

**时间序列**的趋势成分代表该序列平均值的持续、长期变化。趋势是一系列中移动最慢的部分，该部分代表了最大的重要时间尺度。在产品销售的时间序列中，随着越来越多的人逐年了解该产品，增长趋势可能是市场扩张的影响。
{% asset_img ts_6.png %}

我们将重点关注均值趋势。但更一般地说，序列中任何持续且缓慢变化的变化都可能构成趋势——例如，时间序列通常在其变化中具有趋势。

##### 移动平均图

要查看时间序列可能具有什么样的趋势，我们可以使用**移动平均图**。 为了计算时间序列的移动平均值，我们计算某个定义宽度的滑动窗口内的值的平均值。图表上的每个点代表落在两侧窗口内的系列中所有值的平均值。这个想法是为了消除系列中的任何短期波动，以便只保留长期变化。
{% asset_img ts_7.gif %}

说明线性趋势的移动平均图。曲线上的每个点（蓝色）都是大小为`12`的窗口内点（红色）的平均值。请注意上面的莫纳罗亚系列如何年复一年地重复上下运动——这是一种短期的季节性变化。要使变化成为趋势的一部分，它发生的时间应该比任何季节变化都要长。因此，为了可视化趋势，我们在比该系列中任何季节周期更长的时期内取平均值。对于`Mauna Loa`系列，我们选择了`12`号的窗口，以平滑每年的季节。

##### 工程趋势

一旦我们确定了趋势的形状，我们就可以尝试使用时间步长特征对其进行建模。我们已经了解了如何使用时间虚拟(`time dummy`)变量本身来模拟线性趋势：
```bash
target = a * time + b
```
我们可以通过时间虚拟(`time dummy`)变量的变换来拟合许多其他类型的趋势。如果趋势看起来是二次的（抛物线），我们只需将时间虚拟(`time dummy`)的平方添加到特征集中，即可得到：
```bash
target = a * time ** 2 + b * time + c
```
线性回归将学习系数`a`、`b` 和`c`。下图中的趋势曲线都是使用这些特征和`scikit-learn`的`LinearRegression`拟合的：
{% asset_img ts_8.png %}

如果您以前没有见过这个技巧，您可能还没有意识到线性回归可以拟合曲线而不是直线。这个想法是，如果您可以提供适当形状的曲线作为特征，那么线性回归可以学习如何以最适合目标的方式组合它们。

##### 举例 - 隧道流量

在此示例中，我们将为隧道流量数据集创建趋势模型。
```python
from pathlib import Path
from warnings import simplefilter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

simplefilter("ignore")  # ignore warnings to clean up output cells

# Set Matplotlib defaults
plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
%config InlineBackend.figure_format = 'retina'


# Load Tunnel Traffic dataset
data_dir = Path("../input/ts-course-data")
tunnel = pd.read_csv(data_dir / "tunnel.csv", parse_dates=["Day"])
tunnel = tunnel.set_index("Day").to_period()
```
我们来做一个移动平均图，看看这个系列有什么样的趋势。由于该系列有每日观察，因此我们选择`365`天的窗口来平滑一年内的任何短期变化。要创建移动平均线，首先使用滚动方法开始窗口计算。按照平均值方法计算窗口上的平均值。正如我们所看到的，隧道流量的趋势似乎是线性的。
```python
moving_average = tunnel.rolling(
    window=365,       # 365-day window
    center=True,      # puts the average at the center of the window
    min_periods=183,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)

ax = tunnel.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Tunnel Traffic - 365-Day Moving Average", legend=False,
)
```
{% asset_img ts_9.png %}

我们直接在`Pandas`中设计了时间虚拟(`time dummy`)对象。然而，从现在开始，我们将使用`statsmodels`库中名为确定性进程 (`DeterministicProcess`) 的函数。使用此函数将帮助我们避免时间序列和线性回归可能出现的一些棘手的失败情况。`order`参数指的是多项式阶数：`1`表示线性，`2`表示二次，`3`表示三次，依此类推。
```python
from statsmodels.tsa.deterministic import DeterministicProcess

dp = DeterministicProcess(
    index=tunnel.index,  # dates from the training data
    constant=True,       # dummy feature for the bias (y_intercept)
    order=1,             # the time dummy (trend)
    drop=True,           # drop terms if necessary to avoid collinearity
)
# `in_sample` creates features for the dates given in the `index` argument
X = dp.in_sample()

X.head()
```
结果输出为：
```bash
	const	trend
Day		
2003-11-01	1.0	1.0
2003-11-02	1.0	2.0
2003-11-03	1.0	3.0
2003-11-04	1.0	4.0
2003-11-05	1.0	5.0
```
（顺便说一句，确定性过程是一个技术术语，指的是非随机或完全确定的时间序列，例如`const`和趋势序列。从时间索引导出的特征通常是确定性的。）我们基本上像以前一样创建趋势模型，但请注意添加了`fit_intercept=False`参数。
```python
from sklearn.linear_model import LinearRegression

y = tunnel["NumVehicles"]  # the target

# The intercept is the same as the `const` feature from
# DeterministicProcess. LinearRegression behaves badly with duplicated
# features, so we need to be sure to exclude it here.
model = LinearRegression(fit_intercept=False)
model.fit(X, y)

y_pred = pd.Series(model.predict(X), index=X.index)
```
我们的线性回归模型发现的趋势几乎与移动平均图相同，这表明在这种情况下线性趋势是正确的决定。
```python
ax = tunnel.plot(style=".", color="0.5", title="Tunnel Traffic - Linear Trend")
_ = y_pred.plot(ax=ax, linewidth=3, label="Trend")
```
{% asset_img ts_10.png %}

为了进行预测，我们将模型应用于“**样本外**”特征。 “样本外”是指训练数据观察期之外的时间。以下是我们如何进行`30`天的预测：
```python
X = dp.out_of_sample(steps=30)

y_fore = pd.Series(model.predict(X), index=X.index)

y_fore.head()
```
结果输出为：
```bash
2005-11-17    114981.801146
2005-11-18    115004.298595
2005-11-19    115026.796045
2005-11-20    115049.293494
2005-11-21    115071.790944
Freq: D, dtype: float64
```
让我们绘制该系列的一部分来查看未来`30`天的趋势预测：
```python
ax = tunnel["2005-05":].plot(title="Tunnel Traffic - Linear Trend Forecast", **plot_params)
ax = y_pred["2005-05":].plot(ax=ax, linewidth=3, label="Trend")
ax = y_fore.plot(ax=ax, linewidth=3, label="Trend Forecast", color="C3")
_ = ax.legend()
```
{% asset_img ts_11.png %}

除了充当更复杂模型的基线或起点之外，我们还可以将它们用作“混合模型”中的组件，其中算法无法学习趋势（例如`XGBoost`和随机森林）。