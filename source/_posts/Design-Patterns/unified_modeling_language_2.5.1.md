---
title: Unified Modeling Langusge 2.5.1
date: 2024-01-03 12:34:32
tags:
  - UML
categories:
  - 系统建模
---

## 符号约定

### 需求陈述的关键词

本规范中的词语`SHALL`、`SHALL NOT`、`SHOULD`、`SHOULD NOT`、`MAY`、`NEED NOT`、`CAN和CANNOT`应根据 `ISO` 附录 `H` 进行解释。

### 示例图注释

本规范中的一些图表示例包含解释性注释，不应将其与成为正式 `UML` 图形符号的一部分。在这些情况下，解释性文本源自 `UML` 图边界之外，并且有一个箭头指向由注释解释的图表的特征。 该规格的色彩再现显示了这些注释为红色。

## 公共结构

### 概括

本节规定了 UML 中所有结构建模的基本建模概念。 许多元类这里定义的类是抽象的，为后续子句中定义的专门的、具体的类提供基础。 然而，为了提供如何在 `UML` 中应用这些基本概念的示例，有必要使用这些具体的建模构造，即使它们在后面的条款中指定。 提供了适当的前向参考：必要的。

<!-- more -->

### 根

#### 概括

元素和关系的根概念为 `UML` 中所有其他建模概念提供了基础。

#### 抽象语法

{% asset_img uml_1.png Unified Modeling Language 2.5.1 %}

#### 语义

##### Elements

元素是模型的组成部分。 元素的后代提供适合它们的概念的语义代表。每个元素都具有拥有其他元素的固有能力。 当一个元素从模型中删除时，它的所有元素`ownedElements` 也必须从模型中删除。 每种元素的抽象语法指定它可能拥有哪些其他类型的元素。 模型中的每个元素必须完全属于该模型中的另一个元素模型，该模式的顶级包除外。

##### Comments

每种元素都可以拥有注释。 元素的`ownedComments` 不添加任何语义，但可以表示对模型读者有用的信息。

##### Relationships

关系是指定其他元素之间某种关系的元素。 `DirectedRelationship` 表示源模型元素集合与目标模型元素。 `DirectedRelationship` 被认为是从源元素指向到目标元素。

##### Notation

`relatedElements` 对于定向关系，线路通常以某种方式从源定向到目标。注释显示为右上角弯曲的矩形（也称为“注释符号”）。 这矩形包含评论的正文。 与每个 `annotatedElement` 的连接由单独的虚线显示线。 如果从注释符号可以清楚地看出，连接注释符号和注释元素的虚线可以被抑制上下文，或者在此图中不重要。

### 模版

#### 概括

模板是由其他模型元素参数化的模型元素。 本子条款规定了一般适用于各种模板的概念。

#### 抽象语法

{% asset_img uml_2.png Templates %}


{% asset_img uml_3.png Template bindings %}

#### 语义

##### Templates 

`TemplateableElement` 是一种可以选择定义为模板并绑定到其他模板的元素。 `A template` 是使用 `TemplateSignature` 进行参数化的 `TemplateableElement`。 这样的模板可以用来使用 `TemplateBinding` 关系生成其他模型元素。

模板不能以与同类非模板元素相同的方式使用（例如，模板类不能用作 `TypedElement` 的类型）。 模板元素只能用于生成绑定元素或作为另一个模板规范的一部分（例如，一个模板类可以转化为另一个模板类）。

模板的 `TemplateSignature` 定义了一组可以绑定到实际模型元素的 `TemplateParameters` 在模板的绑定元素中。 绑定元素是具有一个或多个此类的 `TemplateableElement` 模板绑定。

完全绑定元素是其所有 `TemplateBindings` 都绑定了该元素的所有 `TemplateParameter` 的绑定元素。模板正在绑定。 完全绑定元素是普通元素，可以像使用完全绑定元素一样使用同类非绑定（和非模板）元素。 例如，类模板的完全绑定元素可以用作类型化元素的类型。

部分绑定元素是至少其中一个 `TemplateBindings` 不绑定所绑定模板的`TemplateParameter`。 部分绑定的元素仍被视为模板，由其 `TemplateBindings` 未绑定的剩余 `TemplateParameters` 进行参数化。

##### Template Signatures

`TemplateSignature` 的 `TemplateParameters` 指定将被实际替换的形参绑定中的参数（或默认值）。 `TemplateParameter` 是根据包含的 `ParameterableElement`定义的在拥有 `TemplateSignature` 的模板内，`TemplateParameter` 是其中的一部分。 这样的元素被称为由 `TemplateParameter` 公开。

