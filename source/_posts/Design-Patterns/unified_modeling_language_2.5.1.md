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

公开的 `ParameterableElement` 可以由模板直接或间接拥有，也可以由模板拥有`TemplateParameter` 本身，在元素不具有所有权关联的情况下模板模型。 无论哪种情况，`ParameterableElement` 仅在模板的上下文中才有意义 - 它将被绑定上下文中的实际元素有效替换。 因此，由 `ParameterableElement` 公开不能在其所属模板或有权访问该模板的其他模板之外引用 `TemplateParameter`原始模板的内部结构（例如，如果模板是专用的）。 `TemplateSignature`的子类还可以添加附加规则，限制在上下文中可将哪种类型的 `ParameterableElement` 用于 `TemplateParameter`一种特定类型的模板。

`TemplateParameter` 还可以引用 `ParameterableElement` 作为任何形式参数的默认值`TemplateBinding` 不为参数提供显式 `TemplateParameterSubstitution`。 类似于公开的 `ParameterableElement`，默认的 `ParameterableElement` 可以直接由模板拥有，也可以由模板拥有模板参数本身。 即使在以下情况下，`TemplateParameter` 也可以拥有此默认 `ParameterableElement`：公开的 `ParameterableElement` 不属于 `TemplateParameter`。

##### Template Bindings

`TemplateBinding` 是 `TemplateableElement` 和指定替换的模板之间的关系模板的正式 `TemplateParameters` 的实际 `ParameterableElements`。 模板参数替换指定要替换 `TemplateBinding` 上下文中的形式 `TemplateParameter` 的实际参数。如果在此绑定中没有为形式参数指定实际参数，则该参数的默认 `ParameterableElement`使用正式的 `TemplateParameter`（如果指定）。

一个绑定元素可以有多个绑定，可能绑定到同一个模板。 此外，绑定元素可以包含除绑定之外的元素。 多个绑定的扩展以及绑定元素拥有的任何其他元素如何组合在一起以完全指定绑定元素的详细信息特定于 `TemplateableElement` 的子类。 一般原则是，单独评估绑定以产生中间结果（每个绑定一个），然后将其合并以产生最终结果。 这就是融合的方式所做的工作是针对每种 `TemplateableElement` 的。

`TemplateableElement` 可以包含 `TemplateSignature` 和 `TemplateBindings`。 因此是一个 `TemplateableElement`可以既是模板又是绑定元素。

一致的工具可能要求所有正式的 `TemplateParameters` 必须作为 `TemplateBinding` 的一部分进行绑定（完全绑定）或者可以仅允许绑定正式模板参数的子集（部分绑定）。 在里面完全绑定的情况下，绑定元素可能有自己的`TemplateSignature`，并且`TemplateParameters`来自这可以作为 `TemplateBinding` 的实际参数提供。 在部分绑定的情况下，未绑定的形式`TemplateParameters` 充当绑定元素的正式 `TemplateParameters`，因此它仍然是模板。

{% note danger %}
**注意**：具有默认值的 `TemplateParameter` 永远无法取消绑定，因为它具有到默认值的隐式绑定，即使没有给出显式的 `TemplateParameterSubstitution`。
{% endnote %}

##### Bound Element Semantics

`TemplateBinding` 意味着绑定元素具有相同的格式良好的约束和语义，就像拥有目标 `TemplateSignature` 的模板的内容被复制到绑定元素中，替换任何`ParameterableElements` 通过指定的相应 `ParameterableElements` 公开为正式 `TemplateParameters`作为 `TemplateBinding` 中的实际模板参数。 但是，绑定元素并不显式包含模型通过扩展其绑定的模板而隐含的元素。 尽管如此，还是可以定义一个扩展的将绑定元素的 `TemplateParameterSubstitution` 实际应用到目标所产生的绑定元素模板。

如果一个绑定元素有多个`TemplateBinding`，那么可以根据模板定义一个特定的扩展绑定元素在每个 `TemplateBinding` 上。 然后通过合并所有扩展边界元素来构造整体扩展边界元素
`TemplateBinding` 特定的扩展绑定元素与原始绑定元素包含的任何其他元素。如前所述，如何执行此合并取决于所绑定的 `TemplateableElement` 的类型。

在模型中包含绑定元素并不自动要求相应的扩展绑定元素是包含在模型中。 然而，如果按上面给出的方式构造的扩展绑定元素违反了任何格式良好的约束，则原始绑定元素也被认为是格式不正确的。

另一方面，如果绑定元素用于命名空间模板，则可能需要能够引用绑定元素的成员被视为命名空间本身。 例如，对于类模板的绑定元素，可能需要引用该类的操作，例如从 `CallOperationAction` 中。
{% note danger %}
**注意**：从模板引用操作是不够的，因为模板类的每个绑定元素都是被认为拥有自己的有效模板操作副本。
{% endnote %}

为了适应这种情况，允许在模型中包含扩展的绑定元素除了绑定元素本身之外，还包括绑定元素。 在这种情况下，扩展的绑定元素必须有一个实现与其扩展的绑定元素的依赖关系。 扩展的绑定元素必须是根据上面给出的规则构建（由建模者手动或由工具自动）。 参考然后照常从其他模型元素到扩展绑定元素的可见成员被认为是在语义上等同于对原始绑定的相应隐式成员进行的有效引用元素。 直接与扩展绑定元素建立的任何关系在语义上等同于关系对绑定元素本身进行了修改。

#### Notation（符号）

如果 `TemplateableElement` 具有 `TemplateParameters`，则会在符号上叠加一个小虚线矩形`TemplateableElement`，通常位于符号的右上角（如果可能）。 虚线矩形包含正式模板参数的列表。 参数列表不能为空，尽管它可能被抑制在演示中。 `TemplateableElement` 符号中的任何其他部分均正常显示。

正式的 `TemplateParameter` 列表可以显示为逗号分隔的列表，也可以是一个正式的 `TemplateParameter` 列表。每行模板参数。 `TemplateParameter` 的一般表示法是显示在模板的`Template`参数列表：

```c++
<template-parameter> ::= <template-param-name> [‘:’ <parameter-kind> ] [‘=’ <default>]

```
其中 `<parameter-kind>` 是公开元素的元类名称。 <模板参数名称> 的语法和 `<default>` 取决于此 `TemplateParameter` 的 `ParameteredElement` 类型。

绑定元素与其他同类元素具有相同的图形符号。 `TemplateBinding` 显示为虚线箭头，尾部位于绑定元素上，箭头位于模板上，关键字`“bind”`。 这绑定信息可以显示为模板参数替换的逗号分隔列表：

```c++
<template-param-substitution> ::= <template-param-name> ‘->’ <actual-template-parameter>

```
其中 `<template-param-name>` 的语法是形参的`parameteredElement` 的名称或`qualifiedNameTemplateParameter` 和 `<actual-template-parameter>` 的类型取决于 `ParameteredElement` 的类型那个模板参数。

绑定元素的绑定的另一种表示方式是将绑定信息包含在绑定元素的符号。 绑定元素的名称被扩展为包含绑定表达式以下语法：

```c++
[<element-name> ‘:’] <binding-expression> [‘,’ <binding-expression>]*
<binding-expression> ::= <template-element-name> ‘<‘ <template-param-substitution> [‘,’<template-paramsubstitution]*‘>’
and <template-param-substitution> is defined as above.
```
### NameSpaces

#### 概括

命名空间是模型中的一个元素，包含一组可以通过名称标识的命名元素。 包（参见第 12 条）是命名空间，其特定目的是包含其他命名元素以组织模型，但许多其他类型的模型元素也是命名空间，包括分类器（参见子条款 9.2），其中包含命名特征和嵌套分类器，以及行为特征（参见子条款 9.4），其中包含命名特征参数。

#### 抽象语法

{% asset_img uml_4.png Namespaces %}

