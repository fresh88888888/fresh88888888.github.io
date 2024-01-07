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

#### 语义

##### NameSpaces

命名空间为命名元素提供容器，称为其拥有的成员。 命名空间也可以从其他命名空间导入 `NamedElements`，在这种情况下，这些元素与`ownedMembers` 一起都是导入命名空间。 如果名称为 `N` 的命名空间的成员是名称为 `x` 的 `NamedElement`，则成员可以通过 `N::x` 形式的限定名称来引用。

当需要区分时，不符合命名空间名称的简单名称可以称为不合格的名字。 在命名空间内，非限定名称可用于引用该命名空间的成员，并且到未隐藏的外部名称。 外部名称是 `NamedElement` 的名称，可以使用直接封闭的命名空间中的非限定名称。 外部名称被隐藏，除非它可以与所有名称区分开来内部命名空间的成员。 （请参阅下面“命名元素”下关于可区分性的讨论。）

由于命名空间本身就是一个 `NamedElement`，因此 `NamedElement` 的完全限定名称可能包括多个命名空间名称，例如 N1::N2::x。

命名空间的`ownedRule`约束表示受约束元素的格式良好的规则（参见子第 `7.6` 条关于约束）。 在确定受约束元素是否格式良好时，将评估这些约束。

##### Named Elements

`NamedElement` 是模型中可能有名称的元素。 该名称可用于识别命名空间内的 `NamedElement` 可以访问其名称。

{% note danger %}
**注意**：`NamedElement` 的名称是可选的，它提供了缺少名称的可能性（即与空名称不同）。
{% endnote %}

`NamedElements` 可能会根据指定 `NamedElement` 的规则出现在命名空间中与另一个有区别。 默认规则是，如果两个成员具有不同的名称或者如果它们具有相同的名称，但它们的元类不同，并且都不是另一个的（直接或间接）子类。对于特定情况，例如通过签名区分的操作（请参阅子命令），此规则可能会被覆盖第 9.6 条）。

`NamedElement` 的可见性提供了一种限制元素使用的方法，无论是在命名空间还是在访问元素。 它旨在与导入、泛化和访问机制结合使用。

除了具有显式名称之外，`NamedElement` 还可以与 `StringExpression` 关联（参见第 8.3 节）可用于指定 `NamedElement` 的计算名称。 在模板中（参见第 7.3 条），`NamedElement` 可能有一个关联的 `StringExpression`，其子表达式可能是 `ParameteredElements` 公开的通过模板参数。 绑定模板时，公开的子表达式将替换为实际值替换模板参数。 `StringExpression` 的值是连接后产生的字符串子表达式的值，然后提供 `NamedElement` 的名称。

`NamedElement` 可以具有与其关联的名称和名称表达式。 在这种情况下，名称可以用作`NamedElement` 的别名，例如，可用于引用约束表达式中的元素。（这避免了在文本表面表示法中使用 `StringExpressions` 的需要，这通常很麻烦，尽管它确实不排除它。）

##### Packageable Elements and Imports

`PackageableElement` 是可以直接由包拥有的 `NamedElement`（请参阅有关包的第 12 条）。 任何这样的元素可以充当模板参数（参见有关模板的子条款 7.3）。

`ElementImport` 是导入命名空间和 `PackageableElement` 之间的 `DirectedRelationship`。 它添加了`PackageableElement` 的名称到导入命名空间。 `ElementImport` 的可见性可以是与导入元素相同或更受限制。

如果名称与外部名称（在封闭的命名空间中定义的元素，可使用它在封闭的命名空间中的非限定名称）在导入命名空间中，外部名称被隐藏`ElementImport`，非限定名称指的是导入的元素。 外部名称可以使用其访问合格名称。

`PackageImport` 是导入命名空间和 `Package` 之间的 `DirectedRelationship`，表明`importing Namespace` 会将 `Package` 的成员名称添加到其自己的命名空间中。 从概念上讲，一个包`import` 相当于对导入的命名空间的每个单独成员都有一个 `ElementImport`，除非有单独定义的 `ElementImport`。 如果某个元素有 `ElementImport`，那么它优先于通过 `PackageImport` 可能导入相同的元素。

如果无法区分的元素由于 `ElementImports` 或`PackageImports`，元素不会添加到导入命名空间中，并且这些元素的名称必须是符合资格才能在该命名空间中使用。 如果导入元素的名称与元素归导入命名空间所有，该元素未添加到导入命名空间且名称为该元素必须经过限定才能使用。

公开导入的元素是导入命名空间的公共成员。 这意味着，如果命名空间是一个包，另一个命名空间对它的 PackageImport 将导致进一步公开这些内容除了包的公共成员之外，还将成员导入到其他命名空间中。
{% note danger %}
**注意**：命名空间不能导入自身，也不能导入任何它自己拥有的成员。 这意味着它不是`NamedElement` 可以在其所属的命名空间中获取别名。
{% endnote %}

#### Notation

##### Namespaces

命名空间没有通用的符号。 特定种类的命名空间有其自己特定的符号。符合标准的工具可以选择允许使用第 12.2.4 条中定义的“圆加号”符号来显示封装成员资格也可用于显示其他类型命名空间中的成员资格（例如，显示嵌套分类器和拥有的类的行为）。

##### Name Expressions

与 `NamedElement` 关联的 `nameExpression` 可以通过两种方式显示，具体取决于别名是否为是否需要。 两种表示法如图 7.6 所示.
- 无别名：`StringExpression` 显示为模型元素的名称。
- 使用别名：无论名称出现在何处，都会显示 `StringExpression` 和别名。 别名下面给出 `StringExpression`。

在这两种情况下，`StringExpression` 都出现在`“$”`符号之间。 `UML` 中的表达式规范支持在抽象语法中使用替代字符串表达式语言——它们必须以 `String` 作为其类型，并且可以是带有操作数的运算符表达式的一些结构。在模板的上下文中，在 `a` 中参数化的 `StringExpression`（通常是 `LiteralStrings`）的子表达式模板显示在尖括号之间。

##### Imports

`PackageImport` 或 `ElementImport` 使用虚线箭头显示，带有来自导入的空心箭头导入的包或元素的命名空间。 如果可见性，则关键字 `«import»` 将显示在虚线箭头附近是公开的； 否则，将显示关键字`“access”`以指示私有可见性。 别名可能显示在之后或下方关键字“导入”。 如果 `ElementImport` 的导入元素是 `Package`，则关键字可以选择为前面是`“element”，即“element import”`。

作为虚线箭头的替代方案，可以通过以下文本来显示 `PackageImport` 或 `ElementImport`：在大括号内唯一标识导入的包或元素，位于名称下方或之后命名空间。 `PackageImport` 的文本语法是：
```c++
‘{import ’ <qualified-name> ‘}’ | ‘{access ’ <qualified-name> ‘}’
The textual syntax for an ElementImport is:
 ‘{element import’ <qualified-name> ‘}’ | ‘{element access ’ <qualified-name> ‘}’
 ```
 或者，也可以显示别名（如果有）：

```c++
‘{element import ’ <qualified-name> ‘ as ’ <alias> ‘}’ | ‘{element access ’ <qualified-name> ‘as’ <alias> ‘}’
```
#### Examples

##### Name Expressions

下图显示了一个资源分配包模板，其中前两个正式模板参数是字符串表达式参数。 这些正式的模板参数在包模板中使用来命名一些`Classes` 和 `Association`结束。 该图还显示了一个绑定包（名为 `TrainingAdmin`），它有两个绑定到此 `ResourceAllocation` 模板。 第一个绑定用字符串`“Instructor”`替换资源，`ResourceKind` 的字符串`“Qualification”`，`System` 的 `Class TrainingAdminSystem`。 第二个绑定将字符串`“Facility”`替换为 `Resource`，将字符串`“FacilitySpecification”`替换为 `ResourceKind`，将 `Class` 替换为`TrainingAdminSystem` 再次替换为 `System`。

绑定的结果包括 `Classes Instructor、Qualification、InstructorAllocation` 以及 `Classes Facility`，设施规格和设施分配。 这些关联也被类似地复制。

{% note danger %}
**注意**：请求将具有从单个`“the<ResourceKind>”`属性派生的两个属性（此处通过箭头），即规格和设施规格。
{% endnote %}

{% asset_img uml_5.png Template package with string parameters %}

##### Imports

下图中，所示的 `ElementImport` 允许包程序中的元素通过名称引用 `DataType`时间类型无限定。 然而，他们仍然需要显式引用 `Types::Integer`，因为这个 `Element` 不是进口的。 数据类型字符串被导入到程序包中，但它作为以下成员不公开可见程序在该包之外，并且不能通过其他命名空间进一步从该程序包导入。

{% asset_img uml_6.png Example of element import %}

在下图中，`ElementImport` 与别名相结合，这意味着将引用 `DataType Types::Real`在 `Shapes` 包中名为 `Double`。

{% asset_img uml_7.png Example of element import with aliasing %}

在下图中，显示了许多 `PackageImport`。 `Types` 的公共成员被导入到 `ShoppingCart` 中然后进一步导入到`WebShop`。 不过，辅助队的成员只是由私人引进的。`ShoppingCart` 不能使用 `WebShop` 中的不合格名称进行引用。

{% asset_img uml_8.png Examples of public and private package imports %}

### Types and Multiplicity

#### 概括

类型和多重性在包含值的元素的声明中使用，以约束类型和可能包含的值的数量。

#### 抽象语法

{% asset_img uml_9.png Abstract syntax of types and multiplicity elements %}

#### Semantics

##### Types and Typed Elements

类型指定一组允许的值，称为类型的实例。 根据类型、实例的种类随着时间的推移，该类型可能会被创建或销毁。 然而，类型实例的构成规则由该类型的定义保持固定。 `UML` 中的所有类型都是分类器。

`TypedElement` 是 `NamedElement`，它以某种方式表示特定值。 取决于种类`TypedElement`，它所代表的实际值可能会随着时间的推移而改变。 `TypedElement` 种类的示例包括
`ValueSpecification`，它直接指定值的集合，以及 `StructuralFeature`，它表示作为拥有它的分类器实例的结构的一部分而保存的值。

如果 `TypedElement` 有关联的 `Type`，则 `TypedElement` 表示的任何值（在任何时间点）都应是给定类型的实例。 没有关联 `Type` 的 `TypeElement` 可以表示任何值。

##### Multiplicities

`MultiplicityElement` 是可以以某种方式实例化以表示值集合的元素。根据 `MultiplicityElement` 的类型，集合中的值可能会随时间而变化。 种类的例子`MultiplicityElement` 包括 `StructuralFeature`，它在拥有的 `Classifier` 实例的上下文中具有它和变量，它在活动执行的上下文中有值。

集合的基数是该集合中包含的值的数量。 `MultiplicityElement` 指定它所代表的集合的有效基数。 多重性是对基数，不得小于为指定的下界且不大于指定的上限（除非多重性是无限的，在这种情况下，上限没有限制）。

`MultiplicityElement` 的重数的下限和上限由 `ValueSpecifications` 指定，`lowerBound` 必须为 `Integer` 值，`upperBound` 必须为 `UnlimitedNatural` 值。 如果`MultiplicityElement` 的 `upperBound` 有无限自然值（“*”）。 如果 `MultiplicityElement` 的 `upperBound`大于 1，则它是多值的（包括无界）。 非多值的 `MultiplicityElement` 最多只能表示一个值。

`MultiplicityElement` 可以定义其边界均为零的重数。 这将允许的基数限制为为 `0`； 也就是说，它要求该元素的实例化不包含任何值。 这在以下情况下很有用概括以约束更一般分类器的基数。 它适用于（但不仅限于）重新定义更通用的分类器中存在的属性。如果 `MultiplicityElement` 被指定为有序（即 `isOrdered` 为 `true`），则实例化中的值集合该元素的值是有序的。 这种排序意味着存在从正整数到元素的映射值的集合。 如果 `MultiplicityElement` 不是多值，则 `isOrdered` 的值没有语义效果。如果 `MultiplicityElement` 被指定为无序（即 `isOrdered` 为 `false`），则不能对该元素的实例化中的值排序。

如果 `MultiplicityElement` 被指定为唯一（即 `isUnique` 为 `true`），则实例化中的值集合该元素必须是唯一的。 也就是说，集合中没有两个值可以相等，其中对象（实例）相等类）基于对象标识，而数据值（数据类型的实例）和信号实例的相等性是基于值。 如果一个`MultiplicityElement` 不是多值的，因此 `isUnique` 的值没有语义效果。总而言之，`isOrdered` 和 `isUnique` 属性可用于指定`MultiplicityElement` 的实例化属于四种类型之一。 

{% asset_img uml_10.png Collection types for MultiplicityElements %}

#### Notation

##### Multiplicity Element

`MultiplicityElement` 的特定符号是为每种具体类型的 `MultiplicityElement` 定义的。 一般来说，符号将包括多重性规范，它显示为包含边界的文本字符串多重性和用于显示可选排序和唯一性规范的符号。

多重界限可以用以下格式显示：

```c++
<lower-bound> ‘..’ <upper-bound>
```
其中 `<lower-bound>` 是 `Integer` 类型的 `ValueSpecification`，`<upper-bound>` 是UnlimitedNatural类型的 `ValueSpecification`。 星号 `(*)` 用作多重性规范的一部分，表示无限的上限边界。如果多重性与表示法是文本字符串（例如属性）的 `MultiplicityElement` 关联，则多重字符串作为该文本字符串的一部分放置在方括号 `([ ])`内。如果多重性与显示为符号（例如关联端）的多重性元素相关联，则多重性字符串显示时不带方括号，并且可以放置在元素符号附近。

如果下限等于上限，则另一种表示法是使用仅包含上限的字符串边界。 例如，“1”在语义上等同于“1..1”多重性。 以零为下界的重数未指定的上限可以使用包含单星“*”而不是“0..*”的替代符号多重性。排序和唯一性规范的具体符号可能会根据具体类型而有所不同多重性元素。 一般的表示法是使用包含“有序”或“无序”的文本注释来定义排序，以及“唯一”或“非唯一”来定义唯一性。

下面的`BNF`定义了多重性字符串的一般语法，包括支持顺序和唯一性指示符：

```c++
<multiplicity> ::= <multiplicity-range> [ [ ‘{‘ <order-designator> [‘,’ <uniqueness-designator> ] ‘}’ ] |
[ ‘{‘ <uniqueness-designator> [‘,’ <order-designator> ] ‘}’ ] ]
<multiplicity-range> ::= [ <lower> ‘..’ ] <upper>
<lower> ::= <value-specification>
<upper> ::= <value-specification>
<order-designator> ::= ‘ordered’ | ‘unordered’
<uniqueness-designator> ::= ‘unique’ | ‘nonunique’
```

#### Examples

下图中，显示两个多重字符串作为类符号中属性规范的一部分。

{% asset_img uml_11.png Multiplicity within a textual specification %}

下图中，显示两个多重字符串作为两个关联端规范的一部分。

{% asset_img uml_12.png Multiplicity as an adornment to a symbol %}

### Constraints

#### Summary

约束是一个断言，指示模型的任何有效实现都必须满足的限制包含约束。 `Constraint` 附加到一组 `constrainedElements`，它表示附加语义有关这些元素的信息。

#### Abstract Syntax

{% asset_img uml_13.png Abstract Syntax of Constraints %}

#### Semantics

约束的规范由布尔类型的 `ValueSpecification`给出。 计算规范可以引用约束的 `constrainedElements` 以及约束的上下文。一般来说，约束有多种可能的上下文。 约束的上下文决定了何时评估约束规范。 例如，作为操作前提条件的约束在操作调用开始时，而作为后置条件的约束在操作结束时评估调用。通过评估其规范来评估约束。 如果规范评估为真，则约束为当时就满足了。 如果规范评估为 `false`，则不满足 `Constraint`，并且实现进行评估的模型无效。

#### Notation

某些类型的约束是在 `UML` 中预定义的，其他类型的约束可以是用户定义的。 用户定义的规范约束通常表示为某种语言中的文本字符串，其语法和解释如下所定义语言。 在某些情况下，形式语言（例如 `OCL`）或编程语言（例如 `Java`）可能是适当的，在其他情况下可以使用自然语言。 这样的规范可以表示为具有适当语言和正文的 `OpaqueExpression`。 可以将约束标记为根据以下 `BNF`，文本在大括号 `({})` 内：

```c++
<constraint> ::= ‘{‘ [ <name> ‘:’ ] <boolean-expression> ‘ }’
```
其中 `<name>` 是约束的名称，`<boolean-expression>` 是约束的适当文本表示法约束规范。最常见的是，约束字符串放置在注释符号中，并附加到每个符号`constrainedElements`由虚线表示。对于应用于单个 `constrainedElement`（例如单个类或关联）的约束，约束字符串可以直接放置在 `constrainedElement` 的符号附近，最好靠近名称（如果有）。 一个工具就能做到可以确定`constrainedElement`。对于表示法是文本字符串（例如属性等）的 `Element`，约束字符串可以跟在 `Element` 后面文本字符串。 这样注释的元素就是约束的单个 `constrainedElement`。

对于适用于两个元素（例如两个类或两个关联）的约束，可以显示约束作为约束字符串标记的元素之间的虚线。如果约束显示为两个元素之间的虚线，则可以在一端放置一个箭头。 这箭头方向是约束内的相关信息。 箭头尾部的 Element 映射到 constrainedElement 中的第一个位置，箭头头部的元素映射到 constrainedElement 中的第二个位置。对于三个或更多相同类型的路径（例如泛化路径或关联路径），约束字符串可以附加到穿过所有路径的虚线。

#### Examples

{% asset_img uml_14.png Constraint in a note symbol %}

下图中，显示附加到属性的约束字符串。

{% asset_img uml_15.png Constraint attached to an attribute %}

下图中，显示两个关联之间的 {xor} 约束

{% asset_img uml_16.png {xor} constraint %}

### Dependencies

#### Summary

依赖关系表示模型元素之间的供应商/客户关系，其中供应商的修改可能影响客户端模型元素。

#### Abstract Syntax

{% asset_img uml_17.png Abstract syntax of dependencies %}

#### Semantics

##### Dependency

依赖关系意味着如果没有供应商，客户端的语义就不完整。 模型中的依赖关系没有任何运行时语义含义。 语义全部给出参与关系的 `NamedElements` 的术语，而不是它们的实例。

##### Usage

`Usage`是一种依赖关系，其中一个 `NamedElement` 需要另一个 `NamedElement`（或一组 `NamedElement`）其全面实施或运作。 使用并未指定客户如何使用供应商，除了以下事实：供应商由客户的定义或实现使用。

##### Abstraction

抽象是一种依赖关系，它关联两个 `NamedElements` 或代表相同内容的 `NamedElements` 集不同抽象层次或不同观点的概念。 该关系可以被定义为映射供应商和客户之间。 根据抽象的具体构造型，映射可以是形式的或非正式的，可以是单向的，也可以是双向的。 抽象具有预定义的构造型（例如`“Derive”，“Refine”和“Trace”`）在标准配置文件中定义。 如果一个抽象有多个客户，供应商作为一个组映射到客户集合中。 例如，一个分析级类可能会被分成几个设计级类。 如果有多个供应商，情况也类似。

##### Realization

`Realization`是两组 `NamedElement` 之间的特殊抽象依赖关系，一组代表一个规范（供应商）和另一个代表该规范的实现（客户）。 实现可以用于建模逐步细化、优化、转换、模板、模型合成、框架实现表示客户端集合是供应商集合的实现，供应商集合充当规范。 `“Realization”`的含义并没有严格定义，而是暗示了更细化或关于特定建模环境的复杂形式。 可以指定规范和之间的映射实现元素，尽管这不一定是可计算的。

#### Notation

依赖关系显示为两个模型元素之间的虚线箭头。 箭头尾部的模型元素（客户）箭头处的模型元素（供应商）。 箭头可以标有可选的关键字或构造型以及可选名称。

{% asset_img uml_18.png Notation for a Dependency between two elements %}

可以为客户或供应商提供一组元素。 在这种情况下，一个或多个箭头的尾部位于客户与一个或多个箭头的尾部相连，其头指向供应商。 可以在上面放置一个小点如果需要的话。 应在连接点附上有关依赖关系的注释。用法显示为依赖项，并附加有`“use”`关键字。抽象显示为带有“抽象”关键字或附加的特定预定义构造型的依赖项到它。实现显示为一条虚线，末端有一个三角形箭头，对应于已实现的元素。

#### Examples

下图中，`CarFactory` 类依赖于 `Car` 类。 在这种情况下，依赖关系是一个用法应用了标准构造型`“Instantiate”`，表明 `CarFactory` 类的实例创建了汽车类。

{% asset_img uml_19.png  An example of an «Instantiate» Dependency %}

下图中，`Order`类需要`Line Item`类才能完整实现。

{% asset_img uml_20.png  An example of a «use» Dependency %}

下图中，举例说明了 `Business` 类由 `Owner` 和 `Employee` 组合实现的示例类。

{% asset_img uml_21.png An example of a realization Dependency %}

### Classifier Descriptions

#### Abstraction [Class]

##### Description

抽象是一种关系，它将表示同一概念的两个元素或元素集联系起来。不同的抽象层次或不同的`viewpoints`。

#### Comment [Class]

##### Description

注释是可以附加到一组元素的文本注释。

#### Constraint [Class]

##### Description

约束是用自然语言文本或机器可读语言表达的条件或限制声明一个元素或一组元素的一些语义的目的。

#### Dependency [Class]

##### Description

依赖关系是一种关系，表示单个模型元素或一组模型元素需要其他元素其规范或实现的模型元素。 这意味着客户端的完整语义元素在语义上或结构上取决于供应商元素的定义。

#### DirectedRelationship [Abstract Class]

##### Description

`DirectedRelationship` 表示源模型元素集合和目标模型元素。

#### Element [Abstract Class]

##### Description

元素是模型的组成部分。 因此，它有能力拥有其他元素。

#### ElementImport [Class]

##### Description

`ElementImport` 标识命名空间中的 `NamedElement`，而不是拥有该 `NamedElement` 的命名空间，并且允许在拥有 `ElementImport` 的命名空间中使用非限定名称来引用 `NamedElement`。

#### MultiplicityElement [Abstract Class]

##### Description

重数是非负整数的包含区间的定义，从下限开始到结束具有（可能是无限的）上限。 `MultiplicityElement` 嵌入此信息来指定允许的元素实例化的基数。

#### NamedElement [Abstract Class]

##### Description

`NamedElement` 是模型中可能有名称的元素。 该名称可以直接给出和/或通过使用一个字符串表达式。

#### Namespace [Abstract Class]

##### Description

命名空间是模型中的一个元素，它拥有和/或导入一组可以通过以下方式识别的命名元素姓名。

### Values

#### Summary

一般来说，`ValueSpecification` 是一个模型元素，被认为是在语义上产生零个或多个值。 值的类型和数量应适合于上下文使用 `ValueSpecification`（由该上下文中给出的约束确定）。以下子条款描述了 `UML` 中可用的各种 `ValueSpecification`。

### Literals

#### Summary

`LiteralSpecification` 是指定文字值的 `ValueSpecification`。 有一种不同的 `UML` 标准 `PrimitiveTypes` 的 `LiteralSpecification`，带有相应的文字符号，加上`“null”`字面意思是“缺少值”。

#### Abstract Syntax

{% asset_img uml_22.png Literals %}

#### Semantics

`LiteralSpecification` 有六种：
1. `LiteralNull` 旨在用于显式建模缺少值的情况。 在一个背景下`MultiplicityElement` 的重数下界为 `0`，这对应于空集（即，一组没有值）。 这相当于没有为 `Element` 指定任何值。
2. `LiteralString` 指定 `PrimitiveType String` 的常量值。 虽然 `String` 被指定为字符序列，字符串值在 `UML` 中被认为是原始的，因此它们的内部结构不是指定为 `UML` 语义的一部分。
3. `LiteralInteger` 指定 `PrimitiveType Integer` 的常量值。
4. `LiteralBoolean` 指定 `PrimitiveType Boolean` 的常量值。
5. `LiteralUnlimitedNatural` 指定 `PrimitiveType UnlimitedNatural` 的常量值。
6. `LiteralReal` 指定 `PrimitiveType Real` 的常量值。

#### Notation

`LiteralSpecifications` 以文本方式标注。

`LiteralNull` 的表示法根据其使用位置的不同而有所不同。 它通常显示为`“null”一词`。 其他地方描述了符号的具体用途。
- `LiteralString` 显示为双引号内的字符序列。 String 值是以下序列字符，不包括引号。 使用的字符集未指定。
- `LiteralInteger` 显示为表示 `Integer` 值的十进制数字的数字序列。
- `LiteralBoolean` 显示为单词`“true”`或单词`“false”`，与其值相对应。
- `LiteralUnlimitedNatural` 显示为数字序列或星号 `(*)`，其中星号表示无限制。 请注意，“无限制”表示对某些元素的值没有限制（例如多重性上限），而不是“无穷大”值。
- `LiteralReal` 以十进制记数法或科学记数法显示。 十进制表示法由可选符号组成字符 `(+/-)` 后跟零个或多个数字，可选地后跟一个点 `(.)`，后跟一个或多个数字。科学记数法由十进制记数法后跟字母`“e”`或`“E”`和指数组成由可选的符号字符后跟一个或多个数字组成。 科学记数法表达的是实数等于指数前面的小数表示的数字乘以 `10` 的幂指数。
该表示法由以下 EBNF 规则指定：
```c++
<natural-literal> ::= ('0'..'9')+
<decimal-literal> ::= ['+' | '-' ] <natural-literal> | ['+' | '-' ] [<natural-literal>] '.' <natural-literal>
<real-literal> ::= <decimal-literal> [ ('e' | 'E') ['+' | '-' ] <natural-literal> ]
```
### Expressions

#### Summary

表达式是指定计算结果值的 `ValueSpecifications`。

#### Abstract Syntax

{% asset_img uml_23.png Expressions %}

#### Semantics

##### Expressions

表达式被指定为树结构。 该树结构中的每个节点都包含一个符号和一组可选的操作数。 如果没有操作数，则表达式表示终端节点。 如果有操作数，则表达式表示由应用于这些操作数的符号给出的运算符。表达式的计算方法是首先计算其每个操作数，然后执行由结果操作数值的表达式符号。 然而，该符号的实际解释取决于表达式的使用上下文和本规范不提供任何标准符号定义。 一个合格的工具可以定义一组特定的符号并为其提供解释，也可以简单地将所有表达式视为未解释的。

##### String Expressions

`StringExpression` 是一个表达式，它指定通过连接子字符串列表而派生的字符串值。子字符串以 `LiteralString` 操作数列表或 `StringExpression` 子表达式列表的形式给出（但它是不允许将两者混合）。 `StringExpression` 的 `String` 值是通过按顺序连接 `String` 获得的操作数或子表达式的值，具体取决于给定的值。`StringExpressions` 旨在用于在模板上下文中指定 `NamedElements` 的名称。 任何一个整个 `StringExpression` 或其一个或多个子表达式可以用作 `ParameterableElements TemplateParameters`，允许在模板中参数化 `NamedElement` 的名称。

##### Opaque Expressions

`OpaqueExpression` 指定根据 `UML` 行为或基于使用除 `UML` 之外的语言的文本语句。 `OpaqueExpression` 可能有一个由一系列文本字符串组成的主体，这些文本字符串表示替代方法计算 `OpaqueExpression` 的值。 相应的语言字符串序列可用于指定每个正文字符串要解释的语言。 语言按顺序与正文字符串匹配。`UML` 规范没有定义正文字符串相对于任何语言的解释方式，尽管其他语言规范可以定义特定的语言字符串，用于指示对这些语言的解释规范（例如，`“OCL”`表示根据 `OCL` 规范解释的表达式）。 另请注意，它不是需要指定语言。 如果未指定，则必须确定任何正文字符串的解释隐含地来自主体的形式或 `OpaqueExpression` 的使用上下文。
