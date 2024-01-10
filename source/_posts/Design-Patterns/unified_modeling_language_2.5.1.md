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

## Values

### Summary

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

#### Examples

##### Expressions
```c++
xor、else、plus(x,1)、x+1
```
##### Opaque Expressions
```c++
a > 0 {OCL} i > j and self.size > i
average hours worked per week
```

### Time

#### Summary

该子条款定义了基于简单时间模型生成值的时间表达式和持续时间。 这简单的时间模型旨在作为时间和时间的更复杂方面的情况的近似值可以安全地忽略测量。 例如，在许多分布式系统中，没有全局的时间概念，只有相对于系统的每个分布式元素的本地时间概念。 时间的相对性没有被考虑在内简单的时间模型，也不是由具有有限分辨率的不完美时钟、溢出、漂移、倾斜等造成的影响。假设与这些特征相关的应用程序将使用更复杂的时间模型由适当的配置文件提供。

#### Abstract Syntax

{% asset_img uml_24.png Time and Duration %}

#### Semantics

##### Time

`UML` 的结构建模用于对特定时间的实体属性进行建模。 在相比之下，行为建模构造用于模拟这些属性如何随时间变化。 一个事件是一个当感兴趣的事情发生时，特定时间点可能发生的事情的规范
正在建模的属性和行为，例如属性值的变化或开始执行活动。在这个概念中，时间只是一个安排事件发生的坐标。 每个事件的发生都可以给出时间坐标值，基于此，可以说是在另一个事件之前、之后或同时发生。持续时间是两个事件发生之间的时间段，计算为时间坐标的差那些事件。 如果模型元素具有行为效果，则该效果可能会在一段时间内发生。 开始持续时间的事件称为进入元素，结束事件称为退出元素。

##### Observations

`Observation`表示对相对于模型的其他部分可能发生的事件的观察。 一个对模型内的 `NamedElement` 进行观察。 感兴趣的事件是当参考进入和退出`NamedElement`。 如果引用的 `NamedElement` 不是行为元素，则持续时间进入和退出 `NamedElement` 之间的值被认为为零，但本规范不另外规定定义在元素上观察到哪些特定事件。观察有两种，时间观察和持续时间观察。`TimeObservation` 观察进入或退出特定的 `NamedElement`。 如果`firstEvent为true`，则它是条目观察到进入的事件，否则观察到退出事件。 `TimeObservation` 的结果是观察到的事件发生。`DurationObservation` 观察相对于一个或两个 `NamedElement` 的持续时间。 如果单个元素是观察到，则观察到的持续时间是元素的进入和退出事件连续发生之间的时间。 如果观察到两个元素，则持续时间介于第一个元素的进入或退出事件与第二个元素的后续进入或退出事件。 在后一种情况下，两个相应的`firstEvent`值也必须是为 `DurationObservation` 给出，这样，如果观察到的元素的`firstEvent=true`，那么它就是入口事件观察到，否则观察到的是退出事件。

##### TimeExpression

`TimeExpression` 是一个 `ValueSpecification`，它计算出某个时刻的时间坐标，可能是相对的对于某些给定的观察集。如果 `TimeExpression` 有 `expr`，则会对其求值以生成 `TimeExpression` 的结果。 表达式必须评估为单个值，但 `UML` 没有定义该值必须具有的任何特定类型或单位。 表达式可能引用与 `TimeExpression` 相关的观察结果，但没有为此类引用定义标准符号。 如果`TimeExpression` 有一个 `expr` 但没有观测值，那么 `expr` 的计算结果为时间常数。如果 `TimeExpression` 没有 `expr`，那么它必须有一个 `TimeObservation` 及其结果观察值是 `TimeExpression` 的值。

#####  Duration

`Duration` 是一个 `ValueSpecification`，它评估某个持续时间，可能相对于某些给定的一组观察。如果 `Duration` 有一个 expr，则对其求值以生成 `DurationExpression` 的结果。 表达式必须评估为单个值，但 `UML` 没有定义该值必须具有的任何特定类型或单位。 表达式可能参考与持续时间相关的观察结果，但没有为此类参考定义标准符号。 如果`Duration` 有一个 `expr` 但没有观测值，然后 `expr` 计算结果为一个持续时间的常量。如果持续时间没有 `expr`，那么它必须有一个 `DurationObservation` 并且该观察的结果是持续时间的值。

#### Notation

##### Observations

观察可以用一条附加到它所引用的 `NamedElement` 的直线来表示。 给出了观察结果显示在该行未连接端附近的名称。 给出了关于观察的附加符号约定相对于通常使用它们的建模结构的其他地方。

##### Time Expressions and Durations

时间表达式或持续时间由其 `expr` 的文本形式表示（如果有的话）。该表示是用于计算时间或持续时间值的公式，其中可以包括相关的名称观察结果和常数。 如果 `TimeExpression`或 `Duration` 没有 `expr`，则它仅由其表示单一相关观察。持续时间是以特定于实现的文本格式给出的相对时间值。 通常，持续时间是一个非负整数表达式，表示在此持续时间内可能经过的“时间刻度”数。

#### Examples

时间通常使用数字坐标表示，在这种情况下，`TimeExpression` 的 `expr` 应计算为数值，其单位可以按照模型中的惯例假定（例如，时间始终以秒为单位）。或者，数据类型可用于对具有特定单位（例如，秒、日等）的时间值和`expr`进行建模然后，`TimeExpression` 应该具有这些类型中适当的一种。持续时间是相对时间的值，因此通常表示为非负数，例如整数持续时间内参考时钟上经过的“时间滴答”数量的计数。 在这种情况下，`expr`的 `DurationExpression`计算结果应为非负数值。 `Duration` 值也可以用来表示自某个固定的时间“原点”以来，时间坐标值的持续时间。

### Intervals

#### Summary

间隔是两个值之间的范围，主要用于断言某些其他元素具有给定范围内的值。 可以为任何类型的值定义间隔，但它们对于时间特别有用持续时间值作为相应 `TimeConstraints` 和 `DurationConstraints` 的一部分。

### Abstract Syntax

{% asset_img uml_25.png Intervals %}

### Semantics

#### Intervals

间隔是使用其他两个 `ValueSpecification`（最小值和最大值）指定的 `ValueSpecification`。 间隔是通过首先评估其每个组成 `ValueSpecifications` ，每个`ValueSpecifications` 必须评估单个值。`Interval` 的值就是从最小值到最大值的范围，即所有大于或等于的值的集合等于最小值且小于或等于最大值（可能是空集）。 请注意，虽然从语法上讲，任何类型的 `ValueSpecifications` 都允许用于 `Interval` 的最小值和最大值，这是一种标准语义仅针对最小和最大 `ValueSpecifications` 具有相同类型和该类型的间隔给出解释其上定义了排序。`Interval` 有两种特殊形式可用于时间约束。 `TimeInterval` 指定两个时间之间的范围由 `TimeExpressions` 给出的时间值。 `DurationInterval` 指定两个持续时间值之间的范围：持续时间。

#### IntervalConstraint

`IntervalConstraint` 定义了一个 `Constraint`，其规范 `Interval` 给出约束。 `IntervalConstraint` 的 `constrainedElements` 被断言为具有范围内的值由 `IntervalConstraint` 的间隔指定。 如果 `constrainedElement` 的值超出此范围，则违反了 `IntervalConstraint`。 如果任何 `constrainedElement` 无法被解释为值，或者其值不相同类型为 `IntervalConstraint` 给定的范围，则 `IntervalConstraint` 没有标准语义解释。`IntervalConstraint` 有两种专门化用于指定时序约束。 `TimeConstraint` 定义了单个 `constrainedElement` 上的 `IntervalConstraint`，其中约束 `Interval` 是 `TimeInterval`。 `DurationConstraint` 在一个或两个 `constrainedElement` 上定义`IntervalConstraint`。

#### Notation

##### Intervals

间隔在文本上由两个用`“..”`分隔的 `ValueSpecifications` 的文本表示形式表示：
```c++
<interval> ::= <min-value> ‘ ..’ <max-value>
```
`TimeInterval` 用 `Interval` 表示法显示，其中每个 `ValueSpecification` 元素都是一个 `TimeExpression`。 `DurationInterval` 使用 `Interval` 表示法显示，其中每个 `ValueSpecification` 元素都是一个 `Duration`。

## Classification

### Summary

分类是组织的一项重要技术。 本节规定了与分类相关的概念。 这核心概念是分类器，一个抽象元类，其具体子类用于对不同类型的值进行分类。本节中的其他元类表示分类器的组成部分，以及如何实例化分类器的模型使用 `InstanceSpecifications` 以及所有这些概念之间的各种关系。

### Classifiers

#### Summary

分类器表示根据实例的特征对实例进行分类。 分类器按层次结构组织。 `RedefinableElements` 可以在泛化层次结构的上下文中重新定义。

#### Abstract Syntax

{% asset_img uml_26.png Classifiers %}

#### Semantics

##### Classifiers

分类器具有一组特征，其中一些是属性，称为分类器的属性。 每个功能是分类器的成员。分类器分类的值称为分类器的实例, 可以重新定义分类器。

##### Generalization

泛化定义了分类器之间的泛化/专业化关系。 每个概括都涉及一个特定的分类器到更通用的分类器。 给定一个分类器，其一般分类器的传递闭包通常是称为其泛化，其特定分类器的传递闭包称为其特化。 即时的泛化也称为分类器的父类，如果分类器是一个类，则称为它的超类。
{% note danger %}
**注意**：父级（分类器之间的泛化关系）的概念与所有者（分类器之间的泛化关系）的概念无关。
{% endnote %}

分类器的实例也是其每个泛化的（间接）实例。 任何适用于的限制泛化的实例也适用于分类器的实例。当分类器被泛化时，其泛化的某些成员会被继承，也就是说，它们的行为就好像它们一样在继承的分类器本身中定义。 例如，作为属性的继承成员可能具有值或继承分类器的任何实例中的值的集合，并且作为操作的继承成员可以是在继承分类器的实例上调用。继承的成员集称为继承成员。 除非对特定种类有不同的规定分类器中，继承的成员是不具有私有可见性的成员。类型一致性意味着如果一个类型符合另一个类型，则第一个类型的任何实例都可以用作`TypedElement` 的值，其类型被声明为第二个`Type`。

`Classifier` 的 `isAbstract` 属性为 `true` 时，指定该 `Classifier` 是抽象的，即没有直接实例：每个抽象分类器的实例应是其专业化之一的实例。如果一个分类器（父级）概括另一个分类器（子级），则子级的实例不一定是这样的在任何可能的情况下都可以替代父实例。 例如，`Circle` 可以定义为`Ellipse` 的专业化，并且它的实例在涉及访问的每种情况下都是可替换的椭圆的性质。 但是，如果 `Ellipse` 要定义修改其长轴长度的拉伸行为只有这样，`Circle` 对象将无法实现这样的行为。 `isSubstitutable` 属性可用于指示特定分类器是否可以在可以使用通用分类器的所有情况下使用。

##### Redefinition

专业分类器的泛化的任何成员（即一种 `RedefinableElement`）都可以重新定义而不是被继承。 重新定义是为了增加、约束或覆盖重新定义的成员专业分类器实例的上下文。 当这种情况发生时，重新定义成员将有助于代替重新定义的成员的专门分类器的结构或行为； 具体来说，任何对在专业分类器实例的上下文中重新定义的成员应解析为重新定义的成员（注意为了避免循环，这里的“任何引用”不包括`redefineElement`引用本身）。可以重新定义成员的分类器称为 `redefinitionContext`。 虽然在元模型中`redefinitionContext` 具有多重性“*”，`UML` 规范中没有出现多个“*”的情况重新定义`Context`。 `redefinitionContext`是为每种`RedefinableElement`定义的。重定义元素应与其重定义的 `RedefinableElement` 一致，但可以添加特定约束或专业化 `redefinitionContext` 实例特有的其他细节。可以重定义多个 `RedefinableElements`。 此外，`RedefinableElement` 可以是多次重新定义，只要明确哪个定义适用于哪个特定实例即可。

当 `isLeaf` 属性对于特定 `RedefinableElement` 为 `true` 时，指定它不应被重新定义。重定义的详细语义因 `RedefinableElement` 的每个专业化而异。 有各种各样的重定义元素与其重定义元素之间的兼容性，例如名称兼容性（重定义元素与重新定义的元素具有相同的名称），结构兼容性（客户端可见属性重定义元素也是重定义元素的属性），或行为兼容性（重定义元素是可替换重新定义的元素）。 任何类型的兼容性都涉及对重新定义的约束。分类器本身就是一个 `RedefinableElement`。 当分类器嵌套在类或接口中时，这可以发挥作用，这成为重新定义上下文。 在专门的类或接口的上下文中重新定义分类器具有从专门的类或接口解析的实例对重新定义的分类器进行任何引用。

##### Substitution

替换是两个分类器之间的关系，表示替换分类器符合合约分类器指定的合约。 这意味着 `substitutingClassifier` 的实例是运行时的情况下,在需要合同分类器实例时可以替换。 替换依赖表示运行时不基于专业化的可替代性。 与专业化不同，替代并不意味着有继承结构，但仅遵守公开可用的合同。 它要求：
- 由合约分类器实现的接口, 也由 `substitutingClassifier` 实现，或者由`substitutingClassifier` 实现了更专业的接口类型。
- 合约分类器拥有的任何端口，都有一个替代分类器所拥有的匹配端口。

#### Notation

##### Classifiers

分类器是一个抽象元类。 尽管如此，在一个地方定义一个可用的默认符号还是很方便的。 分类器的一些专业化有其自己独特的符号。分类器的默认表示法是一个包含分类器名称的实线矩形，并带有隔间名称下方用水平线分隔。 分类器的名称应以粗体居中。 对于那些区分大小写字符的语言，分类器名称应以大写字母。如果分类器使用默认符号，则应显示与分类器元类相对应的关键字在名称上方的 `guillemets`中。 每个元类的关键字在附录 `C` 中列出并在符号中指定对于分类器的每个子类。 不需要关键字来表明元类是 `Class`。任何关键字（包括构造型名称）也应在分类器上方的 `guillemets` 内以普通面居中姓名。 如果多个关键字和/或构造型名称适用于同一模型元素，则每个关键字和/或构造型名称都可以包含在单独的一对 `guillemets` 并一个接一个地列出。 或者，它们可能全部出现在同一对之间`guillemets`，用逗号分隔。在使用的字体允许的情况下，抽象分类器的名称以斜体显示。 替代或附加，抽象分类器可以使用其名称之后或下方的文本注释 `{abstract}` 来显示。分类器形状中的一些隔间是强制性的，并且应得到显示具体语法的工具的支持一致性。 其他是可选的，因为一致性工具可能不支持此类隔间。任何隔室都可以被抑制。 不为抑制隔室绘制分隔线。 如果一个隔间是如果被抑制，则无法推断其中是否存在元素。名为`“attributes”`的部分包含通过 `attribute` 属性访问的属性的符号。 这属性隔间是强制性的，并且如果不被抑制，则始终显示在其他隔间上方。

名为“操作”的部分包含操作的符号。 操作舱是强制性的，并且如果未限制，则始终显示在属性隔间下方。 操作室用于拥有操作的分类器，包括类、数据类型和接口。名为“接待”的隔间包含接待的符号。 接待室是强制性的，并且如果未限制，则始终出现在操作室下方。 接待室用于拥有接待的分类器，包括类。任何包含特征符号的隔间都可以显示那些分组在公共文字下的特征，私有和受保护，代表他们的可见性。 可见性文字在隔间中左对齐特征的符号在其下方缩进显示。 这些组可以按任何顺序出现。 可见性分组是可选：一致性工具不需要支持它。一致性工具可以提供选项来抑制包含符号的隔间中的各个特征。一致性工具可以选择支持隔间命名。 可能会显示隔间的名称以删除含糊不清，或者可能被隐藏。 分区名称应居中并以小写字母开头。 隔间名称可以包含空格，并且不应包含标点符号（包括 `guillemets`）。如果分类器拥有属于分类器的成员（包括行为），则符合标准的工具可以提供显示拥有的分类器以及它们之间的关系的选项，以图表方式嵌套在单独的分类器中所属分类器矩形的隔间。 除非另有规定，该隔室的名称应为从相应的元模型属性派生，如果该属性的重数大于 1，则为复数。因此，对于例如，显示类的属性`nestedClassifier`的内容的隔间应被调用“嵌套分类器；” 显示`BehavioredClassifier` 的属性`ownedBehavior` 的内容的隔间应被称为“拥有的行为”。如果分类器拥有约束，一致工具可以实现一个隔间来显示列出的拥有的约束在所属分类器矩形的单独隔间内。 该可选隔间的名称是`“constraints”`。

##### Other elements

泛化显示为一条线，在表示概括的符号之间有一个空心三角形作为箭头。涉及分类器。 箭头指向代表通用分类器的符号。引用相同通用分类器的多个泛化关系可以显示为单独的行单独的箭头。 这种表示法称为“单独目标样式”。 或者，它们可以连接到“共享目标样式”中的相同箭头。`RedefinableElement` 没有通用符号。 有关特定符号，请参阅 `RedefinableElement` 的子类。替换显示为依赖项，并附有关键字`“substitute”`。由分类器继承的成员可以通过在前面添加`“^”`符号来显示在该分类器的图表上，如果该成员不是继承的，则将显示文本表示形式。 因此继承的符号属性定义如下：
```c++
<inherited-property> ::= ’^’ <property>
where <property> 

<inherited-connector> ::= ’^’ <connector>
where <connector>
```
类似的符号可以用于所有继承了分类器成员的命名元素，以表明它们是继承的。继承的成员也可以显示为较浅的颜色，以帮助将它们与非继承的成员区分开。

##### Examples

{% asset_img uml_27.png Generalization notation showing different target styles %}

在特定环境中，通用 Window 类可以由 Resizing Window 类替代。

{% asset_img uml_28.png Example of Substitution notation %}

### Classifier Templates

#### Summary

分类器是一种`TemplateableElement`，表示分类器可以参数化。 这也是（通过`PackageableElement`）一种 `ParameterableElement`，因此分类器可以是正式的 `TemplateParameter` 并且可以被指定为模板绑定中的实际参数。

#### Abstract Syntax

{% asset_img uml_29.png Classifier Templates %}

