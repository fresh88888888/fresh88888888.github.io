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