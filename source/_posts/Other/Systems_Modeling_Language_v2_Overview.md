---
title: Systems Modeling Language™ v2 Overview
date: 2024-01-08 10:20:41
tags:
  - SysML
category:
  - other
---

### 基于模型系统工程(MBSE) 的未来

- 作为数字化转型的一部分。
- 从SoS到组件级的完整生命周期。
- 自动化的工作流程和数字线程的配置基于敏捷的系统开发方法。
- 建模模式和重用。

实现的目标：

- 管理复杂性和风险。
- 更快地对变化作出反应。
- 重用和设计演进。
- 推理和分析系统。
- 利益相关者共享认知。
- 自动化的文档 & 报告。

<!-- more -->

### SysML v2 的目标

- 语言的精确性和广泛性。
- 语言概念之间的一致性和统一性。
- 与其他工程模型和工具的互操作性。
- 模型开发人员和客户的可用性。
- 支持特定领域应用程序的可扩展性。
- `SysML v1`用户和实现者的迁移路径。

### SysML v2 的关键成员

- 不受`UML`约束的新元模型，保留了大部分`UML`建模功能，重点放在系统建模上, 基于形式语义学构建。
- 基于灵活的视图和视角的鲁棒可视化，图形、表格、文本。
- 标准化的`API`来访问模型。

### 示例车辆模型SysML v2 文本和图形语法

{% asset_img sysml_1.png %}

### SysML v2 vs SysML v1的定义和用法

- 重用概念, 个元素只定义一次，然后在不同的上下文中使用它。
- `SysML v1`非正式地介绍了定义和使用的概念。
- 定义和使用元素是`SysML v2`的正式组成部分。支持一致的分解和特殊化。
- 优点：实现了有效的重用、有助于学习和使用语言、能够自动化。

{% asset_img sysml_2.png %}

### SysML v1 and SysML v2 vehicle block vs part 分解

{% asset_img sysml_3.png %}

### SysML v2 的需求

- `SysML v1`的需求是基于属性的概念上构建的。
- 有效的设计方案必须满足的约束定义:
    - 标识符
    - 可以计算为真或假的约束表达式
    - 约束表达式的属性
    - 约束表达式必须为`true`，才能满足需求

### SysML v1实例 vs SysML v2个体和快照

`SysML v2` 将个体的概念与个体在其生命周期中的某个时刻的快照区分开来。
{% asset_img sysml_4.png %}

### SysML v2 别名和短名

{% asset_img sysml_5.png %}

### 语言扩展 SysML v2 vs SysML v1

`SysML v2`中的库扩展机制可以自动地将专门化功能与构造型结合起来。
{% asset_img sysml_6.png %}

### 示例车辆的模型（SysML v2）

{% asset_img sysml_7.png %}

### 通过标准API连接SysML v2

{% asset_img sysml_8.png %}

### SysML v2 vs SysML v1 比较

- 更容易学习和使用
    - 设计成元模型的系统工程概念
    - 定义和用法模式的一致应用
    - 更一致的术语
    - 使用包过滤器更灵活地组织模型
- 更精确
    - 文本语法和表达语言
    - 形式语义分类
    - 作为约束的需求
- 更多表达
    - 不同的建模
    - 分析用例
    - 权衡分析
    - 个体、快照、时间片
    - 更稳健的定量性质
    - 简单的几何
    - 查询/筛选器表达式
    - 元数据
- 更具可扩展性
    - 更简单的语言扩展能力(基于模型库)
- 更多可互操作性
    - 标准的`API`

### SysML v1 转换为 SysML v2

{% asset_img sysml_9.png %}

### 概念统一

{% asset_img sysml_10.png %}

### 在上下文中的专业化

{% asset_img sysml_11.png %}
{% asset_img sysml_12.png %}

### 具体关系

{% asset_img sysml_13.png %}

### 紧凑的符号

{% asset_img sysml_14.png %}

### 语义库模型

{% asset_img sysml_15.png %}