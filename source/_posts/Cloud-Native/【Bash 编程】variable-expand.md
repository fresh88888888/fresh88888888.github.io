---
title: 【Bash 编程】变量和扩展
date: 2023-08-02 20:23:01
tag: 
   - bash programing
   - Cloud Native
category:
   - Bash Programing
---

假设我们要删除目录中的文件Downloads。根据我们迄今为止收集的知识，我们可以查看其中有哪些文件并删除它们：
```bash
$ cd ~/Downloads
$ ls
05 Between Angels and Insects.ogg
07 Wake Up.ogg
$ rm -v '05 Between Angels and Insects.ogg' '07 Wake Up.ogg'
removed '05 Between Angels and Insects.ogg'
removed '07 Wake Up.ogg'
$ ls
$ 
```
<!-- more -->
如果我们不必把所有事情都说得那么明确，那不是很好吗？毕竟，我们的目的是清空我们的`Downloads`目录。为此，我们必须手动前往那里，找出存在哪些文件，并发出`rm`枚举每个文件名的命令。让我们改进这个工作流程，让我们的代码更加动态一些。我们想要实现的是一种可以反复使用的工作模板。职位描述描述了执行我们意图的方式，无论我们当前所处的具体情况如何。
#### 扩展
##### Pathname Expansion

bash 为我们提供的多种形式的扩展中的第一种。欢迎使用路径名扩展：
```bash
$ cd ~/Downloads
$ rm -v *
removed '05 Between Angels and Insects.ogg'
removed '07 Wake Up.ogg'
$ ls
$ 
```
我们想要删除的文件名发生了什么？我们用一种模式替换了它们，告诉 bash为我们扩展路径名。扩展是用特定情况的代码替换部分命令代码的做法。在本例中，我们希望替换*为下载目录中每个文件的路径名。因此，用路径名替换模式称为路径名扩展。bash 注意到您已在命令行上期望看到参数的位置放置了路径名模式。然后，它采用此路径名模式，并在文件系统中查找它可以找到的与我们的此模式匹配的每个路径名。碰巧该模式\*与当前目录中每个文件的名称相匹配。因此，bash 将命令行中的模式替换为当前目录中每个文件的路径名。我们不必再自己做这些工作了！结果，我们的下载目录按预期被清空。

{% note warning %}
重要的是要理解，虽然我们\*在代码中看到明显的参数对于 `rm`，但我们实际上并没有传递\*给`rm`。事实上，该rm命令甚至永远不会看到我们的路径名扩展模式。该模式在启动之前就由 `bash` 进行评估和扩展`rm`。据了解`rm`，它只是接收一个`-v`参数，后跟目录中每个文件的确切完整名称。扩展始终由`bash`本身执行，并且始终在实际运行命令之前执行！
{% endnote %}
`Bash` 可以为我们执行各种路径名扩展。要执行路径名扩展，我们只需在要扩展路径名的位置编写语法`glob` 模式即可。`glob` 是 `bash shell` 支持的模式类型的名称。以下是 `bash shell` 支持的各种基本 `glob` 模式：

|    Glob    |     含义     |
|:-----------|:-------------|
|`*`       |星号或星号匹配任何类型的文本，甚至根本没有文本。|
|`?`       |问号与任意一个字符匹配。  |
|`[characters]`  |方括号内的一组字符与单个字符匹配，前提是该字符位于给定的集合中。|
|`[[:classname:]]` |当矩形大括号内直接有一组冒号时，您可以指定一类字符的名称，而不必自己枚举每个字符。Bash 了解各种字符类。例如，如果您使用该模式，则仅当该字符是字母数字时，`bash` 才会将其与字符匹配。支持的字符类包括：`alnum、alpha、ascii、blank、cntrl、digit、graph、lower、print、punct、space、upper、word、xdigit[[:alnum:]]`  |

我们可以将这些 glob 模式组合在一起来描述各种路径名组合。我们还可以将它与文字字符结合起来，告诉 bash 模式的一部分应该包含确切的文本：
```bash
$ ls                                # Without arguments, ls simply lists the full contents of a directory.
myscript.txt
mybudget.xsl
hello.txt
05 Between Angels and Insects.ogg
07 Wake Up.ogg
$ ls *                              # While the effect is the same, this command actually enumerates every single file
myscript.txt                        # in the directory to the ls in its arguments!
mybudget.xsl
hello.txt
05 Between Angels and Insects.ogg
07 Wake Up.ogg
$ ls *.txt                          # When we include the literal string .txt, the only pathnames that still match the pattern
myscript.txt                        # are those that start with any kind of text and end with the literal string .txt.
hello.txt
$ ls 0?' '*.ogg                     # Here we're combining patterns, looking for any pathname start starts with a 0,
05 Between Angels and Insects.ogg   # followed by any single character, followed by a literal space, ending in .ogg.
07 Wake Up.ogg
$ ls [0-9]*                         # In a character set, we can use - to indicate a range of characters.  This will match
05 Between Angels and Insects.ogg   # a pathname starting with one character between 0 and 9 followed by any other text.
07 Wake Up.ogg
$ ls [[:digit:]][[:digit:]]*        # Character classes are really nice because they speak for us: they tell us exactly
05 Between Angels and Insects.ogg   # what our intent here is.  We want any pathname that start with two digits.
07 Wake Up.ogg
$ ls [[:digit:]][[:digit:]]         # Your pattern needs to be complete!  None of our filenames is only just two digits.
$ 
```
同样重要的是要了解这些 `glob` 永远不会跳转到子目录中。它们仅与自己目录中的文件名匹配。如果我们希望 `glob` 去查看不同目录中的路径名，我们需要用文字路径名显式地告诉它：
```bash
$ ls ~/Downloads/*.txt                    # Enumerate all pathnames in ~/Downloads that end with .txt.
/Users/lhunath/Downloads/myscript.txt
/Users/lhunath/Downloads/hello.txt
$ ls ~/*/hello.txt                        # Globs can even search through many directories!  Here bash will search
/Users/lhunath/Documents/hello.txt        # through all directories in our home directory for a file that's called hello.txt.
/Users/lhunath/Downloads/hello.txt
```
路径名扩展是一个非常强大的工具，可以避免在参数中指定确切的路径名，或者在文件系统中查找我们需要的文件。最后，`bash` 还内置了对更高级的 `glob` 模式的支持。这些 `glob` 称为：扩展 `glob`。默认情况下，对它们的支持是禁用的，但我们可以使用以下命令在当前 `shell` 中轻松启用它：
```bash
$ shopt -s extglob
```
启用扩展 glob 后，上面的 glob 模式运算符表将扩展为以下附加运算符：
|          全局扩展         |        意义        |
|:------------------------|:-------------------|
|`+(pattern[ \| pattern ... ])`|当列表中的任何模式出现一次或多次时匹配。读作：至少其中之一......|
|`*(pattern[ \| pattern ... ])`|当列表中的任何模式出现一次、根本不出现或出现多次时匹配。读作：然而许多......。|
|`?(pattern[ \| pattern ... ])`|当列表中的任何模式出现一次或根本不出现时匹配。读作：也许其中之一......|
|`@(pattern[ \| pattern ... ])`|当列表中的任何模式仅出现一次时匹配。读作：……之一。|
|`!(pattern[ \| pattern ... ])`|仅当列表中没有出现任何模式时才匹配。读作：没有一个……。|

这些运算符起​​初有点难以理解，但它们是向模式添加逻辑的好方法：
```bash
$ ls +([[:digit:]])' '*.ogg               # Filenames that start with one or more digits.
05 Between Angels and Insects.ogg
07 Wake Up.ogg
$ ls *.jp?(e)g                            # Filenames that end either in .jpg or .jpeg.
img_88751.jpg
igpd_45qr.jpeg
$ ls *.@(jpg|jpeg)                        # Same thing, perhaps written more clearly!
img_88751.jpg
igpd_45qr.jpeg
$ ls !(my*).txt                           # All the .txt files that do not begin with my.
hello.txt
$ ls !(my)*.txt                           # Can you guess why this one matches myscript.txt?
myscript.txt
hello.txt
```
扩展的全局模式有时非常有用，但它们也可能令人困惑和误导。让我们关注最后一个例子：为什么要`!(my)*.txt`扩展路径名`myscript.txt？`是不是`!(my)`应该只在路径名在此位置没有时才匹配`？my`你说得对，就是这样！然而，`bash` 扩展了以my!开头的路径名。

这里的答案是 `bash` 很乐意将模式的这一部分与m开头的 匹配（与 不同`my`），甚至与文件名开头的空白进行匹配。这意味着为了使路径名仍然可以扩展，模式的其余部分需要与路径名的其余部分匹配。碰巧我们在 `glob*`后面有一个 `glob !(my)`，它很乐意匹配整个文件名。在这种情况下，该部分与名称开头的字符`!(my)`匹配，该部分与该部分匹配，模式的后缀与尾随的字符匹配`m*yscript.txt.txt`我们的路径名。模式与名称相匹配，因此名称被扩展！当我们在模式*内部包含该内容时`!()`，这不再有效，并且针对此路径名的匹配失败：
```bash
$ ls !(my)*.txt
myscript.txt
hello.txt
$ ls !(my*).txt
hello.txt
```
##### 波形符扩展

它称为波形符扩展，它涉及将路径名中的波形符 ( ~) 替换为当前用户主目录的路径：
```bash
$ echo 'I live in: ' ~           # Note that expansions must not be quoted or they will become literal!
I live in: /Users/lhunath
```
与路径名扩展相比，波浪线扩展在 `bash` 中稍微特殊，因为它发生在解析器阶段的早期。这只是一个细节，但重要的是要注意波浪号扩展与路径名扩展不同。我们不会执行搜索并尝试将文件名与全局模式进行匹配。我们只是用显式路径名替换波形符。除了简单的波浪号之外，我们还可以通过将用户名放在波浪号后面来扩展另一个用户的主目录：
```bash
$ echo 'My boss lives in: ' ~root
My boss lives in: /var/root
```
##### 命令替换

扩展的用途远不止于此。我们可以使用扩展将几乎任何类型的数据扩展到命令的参数中。命令替换是将数据扩展为命令参数的一种非常流行的方法。通过`Command Substitution`，我们可以有效地在命令中编写命令，并要求 `bash` 将内部命令展开到其输出中，并使用该输出作为主命令的参数数据：
```bash
$ echo 'Hello world.' > hello.txt
$ cat hello.txt
Hello world.
$ echo "The file <hello.txt> contains: $(cat hello.txt)"
The file <hello.txt> contains: Hello world.
```
我们开始非常简单：我们创建一个名为 的文件`hello.txt`并将字符串放入`Hello world`.其中。然后我们使用cat命令输出文件的内容。我们可以看到该文件包含我们保存到其中的字符串。

但接下来事情就变得有趣了：我们想要在这里做的是向用户输出一条消息，用一个漂亮的句子解释我们文件中的字符串是什么。为此，我们希望使文件的内容成为我们`echo`输出的句子的“一部分”。然而，当我们为这句话编写代码时，我们并不知道文件的内容是什么，那么我们如何在脚本中打出正确的句子呢？答案是扩展：我们知道如何使用 获取文件的内容`cat`，因此这里我们将命令的输出扩展为我们的句子。`Bash` 将首先运行，获取该命令的输出（这是我们的字符串），然后扩展我们的命令替换语法`$(cat ...)`进入该输出。只有在这个扩展之后，`bash`才会尝试运行该`echo`命令。您能猜出在我们的命令替换就地扩展之后该命令的参数`echo`变成了什么吗？答案是：`echo "The file <hello.txt> contains: Hello world`."

这是我们了解到的第一种价值扩展。值扩展允许我们将数据扩展为命令参数。它们非常有用，您将一直使用它们。`Bash` 在值扩展方面具有相当一致的语法：它们都以`$`符号开头。命令替换实质上扩展了在子 shell 中执行的命令的价值。因此，语法是值扩展前缀$后跟要扩展的子 shell 的组合：(...)。子 shell 本质上是一个小型的新 bash 进程，用于在主 bash shell 等待结果时运行命令。
{% note success %}
非常细心的读者可能已经注意到，本指南倾向于使用单引号来引用其字符串，但在最新的示例中，对包含扩展语法的句子改用双引号。这是有意为之的：如果参数是双引号的，则所有值扩展（即所有带有`$`前缀的语法）只能在带引号的参数内扩展。单引号会将`$`语法转换为文字字符，导致 `bash` 输出美元而不是就地扩展其值！因此，对所有包含值扩展的参数加双引号非常重要。

值扩展 ( $...) 必须始终用双引号引起来。
{% endnote %}
{% note warning %}
切勿将价值扩展不加引号。如果这样做，`bash` 将使用分词来分割该值，删除其中的所有空格，并对其中的所有单词执行隐藏的路径名扩展！
{% endnote %}

#### 存储和重复使用数据

我们现在知道如何使用 bash 来编写和管理简单的命令。这些命令使我们能够访问系统上的许多强大的操作。我们已经了解了命令如何通过为程序创建新进程来告诉 bash 执行程序。我们甚至学会了操纵这些进程的基本输入和输出，以便我们可以读取和写入任意文件。那些真正密切关注的人甚至会发现我们如何使用诸如此处文档和此处字符串之类的结构将任意数据传递到进程中。现在最大的限制是我们无法灵活处理数据。我们可以将其写入文件，然后通过使用许多文件重定向再次将其读入，并且我们可以使用此处文档和此处字符串传入静态预定义数据。

**bash 参数**

简单地说，bash 参数是内存中的区域，您可以在其中临时存储一些信息以供以后使用。与文件不同，我们写入这些参数，并在稍后需要检索信息时从中读取。但由于我们使用系统内存而不是磁盘来写入这些信息，因此访问速度要快得多。与将输入和输出重定向到文件或从文件重定向相比，使用参数也更容易，语法也更强大。Bash 提供了几种不同类型的参数：位置参数、特殊参数和 shell 变量。后者是最有趣的类型，前两者主要使我们能够访问 bash 提供的某些信息。我们将通过变量介绍参数的实际方面和用法，然后解释位置参数和特殊参数的不同之处。

##### shell 变量

`shell` 变量本质上是一个有名称的 `bash` 参数。您可以使用变量来存储值，并在以后修改或读回该值以供重复使用。使用变量很容易。您可以通过变量分配在其中存储信息，并在以后随时使用参数扩展来访问该信息：
```bash
$ name=lhunath                         # Assign the value lhunath to the variable name
$ echo "Hello, $name. How are you?"    # Expand the value of name into the echo argument
Hello, lhunath.  How are you?
```
正如您所看到的，赋值创建了一个名为 的变量`name`，并在其中放入了一个值。参数值的扩展是通过在名称前添加符号来完成的`$`，这会导致我们的值被注入到 `echo` 参数中。赋值使用`=`运算符。您必须了解运算符周围不能有语法空间。虽然其他语言可能允许这样做，但 bash 却不允许。请记住上一章中的空格在 `bash` 中具有特殊含义：它们将命令拆分为参数。如果我们在运算符周围放置空格=，它们会导致 `bash` 将命令拆分为命令名称和参数，认为您想要执行程序而不是分配变量值：=要修复此代码，我们只需删除导致分词的运算符周围的空格即可。如果我们想给以几个文字空格字符开头的变量分配一个值，我们需要使用引号来向 `bash` 发出信号，表明我们的空格是文字的，不应该触发分词：
```bash
$ name=lhunath
$ item='    4. Milk'
```
我们甚至可以将此赋值语法与其他值扩展结合起来：
```bash
$ contents="$(cat hello.txt)"
```
在这里，我们执行命令替换，将文件的内容扩展hello.txt为我们的赋值语法，这随后导致该内容被分配给内容变量。给变量赋值很简洁，但并不是立即有用。能够随时重用这些值使得参数变得如此有趣。重用参数值是通过扩展它们来完成的。参数扩展有效地从参数中取出数据并将其内联到命令的数据中。正如我们之前简要看到的，我们通过在参数名称前加上符号来扩展参数$。每当您在 bash 中看到此符号时，可能是因为某些内容正在扩展。它可以是参数、命令的输出或算术运算的结果。此外，参数扩展允许您将大括号 ({和}) 括在扩展周围。这些大括号用于告诉 bash 参数名称的开头和结尾是什么。它们通常是可选的，因为 bash 通常可以自行计算出名称。尽管有时它们成为必需品：
```bash
$ name=Britta time=23.73                        # We want to expand time and add an s for seconds
$ echo "$name's current record is $times."      # but bash mistakes the name for times which holds nothing
Britta's current record is .
$ echo "$name's current record is ${time}s."    # Braces explicitly tell bash where the name ends
Britta's current record is 23.73s.
```
参数扩展非常适合将用户或程序数据插入到我们的命令指令中，但它们还有一个额外的王牌：参数扩展运算符。扩展参数时，可以将运算符应用于扩展值。该运算符可以通过多种有用方法之一修改该值。请记住，该运算符仅更改扩展的值；它不会改变变量中的原始值。
```bash
$ name=Britta time=23.73
$ echo "$name's current record is ${time%.*} seconds and ${time#*.} hundredths."
Britta's current record is 23 seconds and 73 hundredths.
$ echo "PATH currently contains: ${PATH//:/, }"
PATH currently contains: /Users/lhunath/.bin, /usr/local/bin, /usr/bin, /bin, /usr/libexec
```
上面的示例使用`%`,`#`和`//`运算符在扩展结果之前对参数值执行各种操作。参数本身没有改变；运算符仅影响扩展到位的值。您还会注意到，我们可以在这里使用 `glob` 模式，就像我们在路径名扩展期间所做的那样，来匹配参数中的值。

在第一种情况下，我们在展开之前使用%运算符从 的值中删除.及其后面的数字。time这样我们就只剩下了 前面的部分.，即秒。第二种情况做了类似的事情，我们使用#运算符从值的开头删除一部分`time`。最后，我们使用`//`运算符（这实际上是运算符的特例`/`）将的值:中的每个字符替换为。结果是一个目录列表，比原来的冒号分隔的目录更容易阅读。 `PATH, PATH`
|    Operator      |       Example        |           Result       |
|:-----------------|:---------------------|:-----------------------|
|`${parameter#pattern}`如果与模式匹配的最短字符串位于值的开头，则删除它。|`"${url#*/}"`|https:///guide.bash.academy/variables.html > /guide.bash.academy/variables.html |
|`${parameter##pattern}`如果与模式匹配的最长字符串位于值的开头，则删除它。|`"${url##*/}"`|https://guide.bash.academy/variables.html > variables.html|
|`${parameter%pattern}`如果与模式匹配的最短字符串位于值的末尾，则删除它。|`"${url%/*}"`|https://guide.bash.academy/variables.html > https://guide.bash.academy|
|`${parameter%%pattern}`如果与模式匹配的最长字符串位于值的末尾，则删除它。|`"${url%%/*}"`|https：//guide.bash.academy/variables.html > https：|
|`${parameter/pattern/replacement}`将与模式匹配的第一个字符串替换为替换内容。|`"${url/./-}"`|https://guide.bash.academy/variables.html > https://guide-bash.academy/variables.html|
|`${parameter//pattern/replacement}`将与模式匹配的每个字符串替换为替换内容。|`"${url//./-}"`|https://guide.bash.academy/variables.html > https://guide-bash-academy/variables-html|
|`${parameter/#pattern/replacement}`将与值开头的模式匹配的字符串替换为替换值。|`"${url/#*:/http:}"`|https：//guide.bash.academy/variables.html > http://guide.bash.academy/variables.html|
|`${parameter/%pattern/replacement}`将与值末尾的模式匹配的字符串替换为替换值。|`"${url/%.html/.jpg}"`|https://guide.bash.academy/variables.html > https://guide.bash.academy/variables.jpg|
|`${#parameter}`扩展值的长度（以字节为单位）。|`"${#url}"`|https://guide.bash.academy/variables.html > 40|
|`${parameter:start[:length]}`展开值的一部分，从start开始，长度字节长。您甚至可以使用负值（空格后跟一个）从末尾而不是从开头开始计数。|`"${url:8}"`|https://guide.bash.academy/variables.html > guide.bash.academy/variables.html|
|`${parameter[^\|^^\|,\|,,][pattern]}`展开转换后的值，将与模式匹配的第一个或所有字符大写或小写。您可以省略模式来匹配任何字符。|`"${url^^[ht]}"`|http://guide.bash.academy/variables.html > HTTps://guide.basH.academy/variables.HTml|

`Shell` 变量是您可以自由赋值的参数。赋值是使用语法完成的`var=value`。可以扩展参数以将其数据内联到命令的参数中。参数扩展是通过在变量名前加上`$`符号来完成的。有时，您需要在参数名称周围添加`{}`大括号，以明确告诉 bash 参数名称的开始和结束位置（例如 `"${time}s"`）。
**参数扩展应始终用双引号引起来，以保持一致性**，并防止其中任何潜在的空格导致分词以及触发意外的路径名完成。在扩展参数时，您可以应用特殊的参数扩展运算符以某种方式改变扩展值。

1. 分配hello给变量greeting。
```bash
$ greeting=hello
```
2. 显示变量`greeting`的内容。
```bash
$ echo "$greeting"
```
3. 将字符串分配 world到变量当前内容的末尾。
```bash
$ greeting="$greeting world"
$ greeting+=" world"
```
4. 显示变量greeting中的最后一个单词。
```bash
$ echo "${greeting##* }"
```
5. 显示变量问候语的内容，第一个字符大写，.末尾有一个句点 ( )。
```bash
$ echo "${greeting^}."
```
6. 将变量内容中的第一个空格字符替换为big。
```bash
$ greeting=${greeting/ / big }
```
7. 将变量greeting的内容重定向到一个文件中，该文件的名称是该变量的值，并且末尾的空格替换为下划线( _) 和 a 。.txt
```bash
$ echo "$greeting" > "${greeting// /_}.txt"
```
8. 显示变量问候语的内容，中间单词完全大写。
```bash
$ middle=${greeting% *} middle=${middle#* }; echo "${greeting%% *} ${middle^^} ${greeting##* }"
```

#### 环境

有两个单独的空间保存变量。这些独立的空间经常被混淆，导致许多误解。您已经熟悉了第一个：shell 变量。保存变量的第二个空间是进程环境。我们将介绍环境变量并解释它们与 shell 变量的区别。
##### 环境变量

与 shell 变量不同，环境变量存在于进程级别。这意味着它们不是 bash shell 的功能，而是系统上任何程序进程的功能。如果我们将流程想象成您购买的一块土地，那么我们在这块土地上建造的建筑物将是在您的流程中运行的代码。你可以在土地上建造一座bash房子、一间grep棚屋或一座塔。firefox环境变量是存储在您的进程土地本身上的变量，而 shell 变量存储在您的土地上构建的 bash house 内。
您可以在环境中存储变量，也可以在 shell 中存储变量。环境是每个进程都拥有的，而 shell 空间仅适用于 bash 进程。通常，您应该将变量放在 shell 空间中除非您明确要求环境变量的行为。
```
    ╭──── bash────────────────────────╮ 
    │ ╭────────────────── ─╮ │ 
    │ 环境 │ SHELL │ │ │ 
    │ shell_var1=值 │ │ 
    │ │ shell_var2=值 │ │ 
    │ ╰──────────────────╯ │ 
    │ ENV_VAR1=值 │ 
    │ ENV_VAR2=值 │ 
    ╰──────────────────────────────────╯
```
当您从 shell 运行新程序时，bash 将在新进程中运行该程序。当它发生时，这个新进程将拥有自己的环境。但与shell进程不同的是，普通进程没有shell变量。他们只有环境变量。更重要的是，当创建一个新进程时，它的环境是通过创建创建进程的环境的副本来填充的：
```
    ╭──── bash ──────────────────────╮ 
    │ ╭────────────────╮ │ 
    │环境 │ SHELL │ │ 
    │ │greeting=hello │ │ 
    │ ╰────────────────╯ │ 
    │ HOME=/home/lhunath │ 
    │ PATH=/bin:/usr/bin │ 
    ╰─┬────────────────────────────╯ 
      ╎ ╭──── ls ────────── ────────────────╮ 
      └╌╌┥ │ 
         │ 环境 │ 
         │ │ 
         │ HOME=/home/lhunath │ 
         │ PATH=/bin:/usr/bin │ 
         ╰─── ──────────────────────────────╯
```
一个常见的误解是，环境是所有进程共享的系统全局变量池。这种错觉通常是由于在子进程中看到相同的变量而导致的。当您在 shell 中创建自定义环境变量时，您之后创建的任何子进程都将继承该变量，因为该变量会从您的 shell 复制到子进程的环境中。但是，由于环境特定于每个进程，因此在子进程中更改或创建新变量绝不会影响父进程：
```
    ╭─── bash ───────────────────────╮
    │             ╭────────────────╮ │
    │ ENVIRONMENT │ SHELL          │ │
    │             │ greeting=hello │ │
    │             ╰────────────────╯ │
    │ HOME=/home/lhunath             │
    │ PATH=/bin:/usr/bin             │
    │ NAME=Bob                       │
    ╰─┬──────────────────────────────╯
      ╎  ╭─── bash ───────────────────────╮
      └╌╌┥             ╭────────────────╮ │
         │ ENVIRONMENT │ SHELL          │ │
         │             ╰────────────────╯ │
         │ HOME=/home/lhunath             │
         │ PATH=/bin:/usr/bin             │
         │ NAME=Bob                       │
         ╰────────────────────────────────╯

$ NAME=John

    ╭─── bash ───────────────────────╮
    │             ╭────────────────╮ │
    │ ENVIRONMENT │ SHELL          │ │
    │             │ greeting=hello │ │
    │             ╰────────────────╯ │
    │ HOME=/home/lhunath             │
    │ PATH=/bin:/usr/bin             │
    │ NAME=Bob                       │
    ╰─┬──────────────────────────────╯
      ╎  ╭─── bash ───────────────────────╮
      └╌╌┥             ╭────────────────╮ │
         │ ENVIRONMENT │ SHELL          │ │
         │             ╰────────────────╯ │
         │ HOME=/home/lhunath             │
         │ PATH=/bin:/usr/bin             │
         │ NAME=John                      │
         ╰────────────────────────────────╯
```
这种区别也清楚地表明了为什么人们会选择将某些变量放入环境中。虽然大多数变量都是普通的 shell 变量，但您可以选择将一些 shell 变量“导出”到 shell 的进程环境中。这样做时，您可以有效地将变量的数据导出到您创建的每个子进程，而这些子进程又将其环境变量导出到它们的子进程。您的系统将环境变量用于各种用途，主要是为某些进程提供状态信息和默认配置。

#例如，login传统上用于将用户登录到系统的程序将有关用户的信息导出到环境中（USER包含您的用户名，HOME包含您的主目录，PATH包含标准命令搜索路径等） 。现在，您登录后运行的所有进程都可以通过查看环境来了解它们正在为哪个用户运行。

您可以将自己的变量导出到环境中。这通常是为了配置您运行的任何程序的行为。例如，您可以导出LANG并为其分配一个值，告诉程序应该使用什么语言和字符集。环境变量通常仅对那些明确了解并支持它们的程序有用。有些变量的用途非常狭窄，例如 某些程序可以使用LSCOLORSls来对系统上文件的输出进行着色。
```
    ╭─── bash ───────────────────────╮
    │             ╭────────────────╮ │
    │ ENVIRONMENT │ SHELL          │ │
    │             │ greeting=hello │ │
    │             ╰────────────────╯ │
    │ HOME=/home/lhunath             │
    │ PATH=/bin:/usr/bin             │
    │ LANG=en_CA                     │
    │ PAGER=less                     │
    │ LESS=-i -R                     │
    ╰─┬──────────────────────────────╯
      ╎  ╭─── rm ─────────────────────────╮           # rm uses just LANG if present to determine
      ├╌╌┥                                │           # the language of its error messages.
      ╎  │ ENVIRONMENT                    │
      ╎  │                                │
      ╎  │ HOME=/home/lhunath             │
      ╎  │ PATH=/bin:/usr/bin             │
      ╎  │ LANG=en_CA                     │
      ╎  │ PAGER=less                     │
      ╎  │ LESS=-i -R                     │
      ╎  ╰────────────────────────────────╯
      ╎  ╭─── man ────────────────────────╮           # In addition to LANG, man uses PAGER to determine
      └╌╌┥                                │           # what program to use for paginating long manuals.
         │ ENVIRONMENT                    │
         │                                │
         │ HOME=/home/lhunath             │
         │ PATH=/bin:/usr/bin             │
         │ LANG=en_CA                     │
         │ PAGER=less                     │
         │ LESS=-i -R                     │
         ╰─┬──────────────────────────────╯
           ╎  ╭─── less ───────────────────────╮      # less makes use of the LESS variable to supply
           └╌╌┥                                │      # an initial configuration for itself.
              │ ENVIRONMENT                    │
              │                                │
              │ HOME=/home/lhunath             │
              │ PATH=/bin:/usr/bin             │
              │ LANG=en_CA                     │
              │ PAGER=less                     │
              │ LESS=-i -R                     │
              ╰────────────────────────────────╯
```

##### Shell Initialization

当您启动交互式 `bash` 会话时，`bash` 将通过从系统上的不同文件读取一些初始化命令来准备使用。您可以使用这些文件来告诉 `bash` 如何行为。其中一个特别旨在让您有机会将变量导出到环境中。该文件被调用`.bash_profile`并且位于您的主目录中。您很可能还没有此文件；如果是这种情况，您只需创建该文件，`bash` 下次查找它时就会找到它。

在你的最后`~/.bash_profile`，你应该有命令`source ~/.bashrc`。这是因为当`.bash_profile`存在时，`bash` 的行为有点奇怪，因为它停止寻找其标准 `shell` 初始化文件`~/.bashrc`。该source命令解决了这个奇怪的问题。

请注意，如果没有`~/.bash_profile`文件，`bash` 将尝试读取`~/.profile`（如果存在）。后者是通用的 `shell` 配置文件，其他 `shell` 也可以读取该文件。您可以选择将环境配置放在那里，但如果这样做，您需要注意应该限制自己使用 `POSIX sh` 语法，而不是在文件中使用任何特定于 `bash` 的 `shell` 语法。
```
    login                  The login program signs the user in
      │
      ╰─ -bash             The login command starts the user's login shell
         │
         ╰─ screen         The user runs the screen program from his login shell
              │
              ╰─ weechat   The screen program creates multiple windows
              │            and allows the user to switch between them. 
              ╰─ bash      The first runs an IRC client, two others run a 
              │            non-login bash shell. 
              ╰─ bash
```
该进程树描述了一个用户，该用户使用 bash 作为登录 shell，并多路复用他的终端以创建多个单独的“屏幕”，从而允许他与多个同时运行的程序进行交互。登录后，系统（login程序）确定用户的登录shell。例如，它可以通过查看来做到这一点/etc/passwd。在本例中，用户的登录 shell 设置为 bash。login继续运行 bash 并将其名称设置为-bash. 程序的标准过程是login在登录 shell 的名称前添加一个-（破折号）前缀，指示该 shell 它应该充当登录 shell。

一旦用户拥有正在运行的 bash 登录 shell，他就运行该screen程序。当 screen 运行时，它会接管用户的整个终端并模拟其中的多个终端，允许用户在它们之间切换。在每个模拟终端中，屏幕运行一个新程序。在这种情况下，用户将屏幕配置为启动一个运行 IRC 客户端的模拟终端，以及两个运行交互式（但非登录）bash shell 的模拟终端。实际情况如下：
我们来看看这个场景中初始化是如何发生的，以及环境变量来自哪里：
```
    login
      │ TERM=dumb
      │ USER=lhunath
      │ HOME=/home/lhunath
      │ PATH=/usr/bin:/bin
      │
      ╰─ -bash
         │ TERM=dumb
         │ USER=lhunath
         │ HOME=/home/lhunath
         │ PATH=/usr/bin:/bin
         │ PWD=/home/lhunath
         │ SHLVL=1
         │╭──────────────╮     ╭────────────────────────╮╭──────────────────╮
         ┝┥ login shell? ┝─yes─┥ source ~/.bash_profile ┝┥ source ~/.bashrc │
         │╰──────────────╯     ╰────────────────────────╯╰──────────────────╯
         │ PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/libexec
         │ EDITOR=vim
         │ LANG=en_CA.UTF-8
         │ LESS=-i -M -R -W -S
         │ GREP_COLOR=31
         │
         ╰─ screen
              │ TERM=dumb
              │ TERM=screen-bce
              │ USER=lhunath
              │ HOME=/home/lhunath
              │ PATH=/usr/bin:/bin
              │ PWD=/home/lhunath
              │ SHLVL=1
              │ PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/libexec
              │ EDITOR=vim
              │ LANG=en_CA.UTF-8
              │ LESS=-i -M -R -W -S
              │ GREP_COLOR=31
              │ WINDOW=0
              │
              ╰─ weechat
              │
              ╰─ bash
              │    │╭──────────────╮
              │    ╰┥ login shell? ┝
              │     ╰──────┰───────╯
              │            no
              │     ╭──────┸───────╮     ╭──────────────────╮
              │     │ interactive? ┝─yes─┥ source ~/.bashrc │
              │     ╰──────────────╯     ╰──────────────────╯
              ╰─ bash
                   │╭──────────────╮
                   ╰┥ login shell? ┝
                    ╰──────┰───────╯
                           no
                    ╭──────┸───────╮     ╭──────────────────╮
                    │ interactive? ┝─yes─┥ source ~/.bashrc │
                    ╰──────────────╯     ╰──────────────────╯
```
正如您所看到的，不同级别将自己的变量导出到环境中。每个子进程都从其父进程的环境中继承变量。反过来，它可以覆盖其中一些值或添加新变量。

请注意第一个（登录）bash 如何获取两者的源~/.bash_profile，而~/.bashrc底部的两个仅获取源~/.bashrc。这是因为只有第一个 bash 进程作为“登录 shell”启动（通过-在其名称前面添加 ）。下面的两个 bash 进程是普通的交互式 shell。他们不需要采购的原因~/.bash_profile现在变得更加明显：他们的职责 ~/.bash_profile是设置 bash 的环境，而底部的两个 shell 已经从其登录 shell 祖先继承了环境。

##### 位置参数

其中变量是带有名称的参数，位置参数是带有数字（更具体地说，正整数）的参数。我们使用正常的参数扩展语法来扩展这些参数：`$1, $3`。但值得注意的是，`bash` 要求您在超过一位数字的位置参数周围使用大括号： , `${10}（${22}`实际上，您很少需要显式引用这么高的位置参数）。

位置参数扩展到由父进程创建时作为参数发送到进程的值。例如，当您`grep`使用以下命令启动进程时：
```bash
$ grep Name registrations.txt
```
您正在有效地运行`grep`带有参数Name的命令`registrations.txt`。如果`grep`是 `bash` 脚本，则第一个参数可以通过扩展在脚本中使用`$1`，第二个参数可以通过扩展使用`$2`。高于的位置参数2将被取消设置。

很高兴知道还有第零个位置参数。该位置参数扩展为进程的名称。进程的名称是由创建它的程序选择的，因此第零个参数实际上可以包含任何内容，并且完全取决于脚本的父级。大多数 `shell` 将使用它们运行的​​文件的绝对路径来启动进程作为进程的名称，或者用户执行的启动进程的命令。请注意，这绝不是一个要求，您不能对第零个参数的内容做出任何可靠的假设：出于所有意图和目的，最好避免这样做。

很好而且非常方便：到目前为止我们学到的关于可变参数的大部分知识也适用于位置参数：我们可以扩展它们，并且可以在这些扩展上应用参数扩展运算符来改变结果值：
```bash
#!/usr/bin/env bash
echo "The Name Script"
echo "usage: names 'My Full Name'"; echo

first=${1%% *} last=${1##* } middle=${1#$first} middle=${middle%$last}
echo "Your first name is: $first"
echo "Your last name is: $last"
echo "Your middle names are: $middle"
```
如果您将此脚本保存在名为 的文件中`names`，并根据使用说明运行它，通过向其传递单个参数，您将看到该脚本分析您的姓名并告知您姓名的哪一部分构成了第一个、最后一个和中间部分名称。当我们在语句中扩展变量时，我们使用变量`first`、`last`和`middle`来存储这些信息`echo`。

```bash
$ chmod +x names
$ ./names 'Maarten Billemont'
The Name Script
usage: names 'My Full Name'

Your first name is: Maarten
Your last name is: Billemont
Your middle names are: 
$ ./names 'James Tiberius "Jim" Kirk'
The Name Script
usage: names 'My Full Name'

Your first name is: James
Your last name is: Kirk
Your middle names are:  Tiberius "Jim"
```
重要的是要理解，与大多数变量不同，位置参数是只读参数。经过反思，您可能会认为人们无法从脚本内部更改脚本的参数。因此，这是一个语法错误：
```bash
$ 1='New First Argument'
-bash: 1=New First Argument: command not found
```
虽然错误消息有点令人困惑，但它表明 bash 甚至不认为该语句是试图为变量赋值（因为参数不是变量1），而是认为您已经给了它命令的名称你想跑。

然而，我们可以使用一个内置命令来更改位置参数集的值。虽然这是缺乏 bash 更高级功能的古代 shell 中的常见做法，但在 bash 中您很少需要这样做。要修改当前的位置参数集，请使用该set命令并在参数后面指定新的位置参数作为参数--：
```bash
$ set -- 'New First Argument' Second Third 'Fourth Argument'
$ echo "1: $1, 2: $2, 4: $4"
1: New First Argument, 2: Second, 4: Fourth Argument
```
除了更改位置参数集之外，还有shift可用于“推送”我们的位置参数集的内置函数。当我们移动位置参数时，我们实际上将它们全部推向开头，导致前几个位置参数被撞掉，为其他参数让路：
```bash
New First Argument Second Third Fourth Argument
$ shift 2                              # Push the positional parameters back 2.
Third Fourth Argument <----
```
最后，当使用命令启动新的 `bash shell` 时`bash`，有一种方法可以传入位置参数。这是将参数列表传递给内联 `bash` 脚本的非常有用的方法。稍后，当您将内联 `bash` 代码与其他实用程序结合使用时，您将使用此方法，但就目前而言，这是试验位置参数的好方法，而无需创建单独的脚本来调用和传递参数（就像我们在示例中所做的那样 `names`）多于）。以下是如何运行内联 `bash` 命令并传入参数列表来填充位置参数：
```bash
$ bash -c 'echo "1: $1, 2: $2, 4: $4"' -- 'New First Argument' Second Third 'Fourth Argument'
1: New First Argument, 2: Second, 4: Fourth Argument
```
我们运行该bash命令，传递-c选项，后跟包含一些 bash shell 代码的参数。这将告诉 bash，您不想启动新的交互式 bash shell，而只想让 shell 运行提供的 bash 代码并完成。在 shell 代码之后，我们指定用于填充位置参数的参数。我们示例中的第一个参数是--，虽然从技术上讲该参数用于填充第零个位置参数，但--为了兼容性并明确 bash 参数与 bash 参数之间的区别，始终使用它是一个好主意。你的外壳代码。在此参数之后，每个参数都会按照您的预期填充标准位置参数。
{% note success %}
请注意，我们包含 bash 代码的参数是'single-quoted'：

每当我们将代码放入字符串中时（例如将其作为参数传递的情况），代码都应该用单引号引起来。不要用于`"double quotes"`包装代码字符串。这很重要，因为单引号在制作包装数据文字方面比双引号可靠得多。
{% endnote %}
如果我们在上面的示例中使用双引号，则我们在其中键入命令的 `shell bash`会展开`$1,$2`和`$4`扩展，从而导致该`-c`选项的参数损坏。
```bash
$ bash -vc 'echo "1: $1, 2: $2, 4: $4"' -- \          # We pass the -v argument to bash to show us the code it is going to run before the result.
'New First Argument' Second Third 'Fourth Argument'   # We can use \ at the end of a line to resume on a new line.
$ echo "1: $1, 2: $2, 4: $4"                            # Here is the code it is going to run.
1: New First Argument, 2: Second, 4: Fourth Argument  # And this is the result.
```

`-c`如果我们在参数周围使用双引号而不是单引号会发生什么：
```bash
$ bash -vc "echo "1: $1, 2: $2, 4: $4"" -- \          # The outer double-quotes conflict with the inner double-quotes, leading to ambiguity.
'New First Argument' Second Third 'Fourth Argument'
echo 1:                                               # As a result, the argument to -c is no longer the entire bash code but only the first word of it.
1:
$ bash -vc "echo \"1: $1, 2: $2, 4: $4\"" -- \        # Even if we fix the quoting ambiguity, the $1, $2 and $4 are now evaluated by the shell we're typing this command into,
'New First Argument' Second Third 'Fourth Argument'   # not the shell we pass the arguments to.
echo "1: , 2: , 4: "                                  # Since $1, $2 and $4 are likely empty in your interactive shell, they will expand empty and disappear from the -c argument.
1: , 2: , 4:
```
我们甚至可以通过反斜杠转义所有特殊字符（包括双引号和美元符号）来解决双引号内的所有问题。这可以解决问题，但会使 shell 代码看起来极其复杂且难以阅读。维护像这样以特殊方式转义的 shell 代码是一场噩梦，并且会导致难以发现的意外错误：
```bash
$ bash -vc "echo \"1: \$1, 2: \$2, 4: \$4\"" -- \
'New First Argument' Second Third 'Fourth Argument'
echo "1: $1, 2: $2, 4: $4"
1: New First Argument, 2: Second, 4: Fourth Argument
```

##### 特殊参数

了解位置参数对理解特殊参数变得更加容易：它们非常相似。特殊参数是名称为单个符号字符的参数，它们用于从 `bash shell` 请求某些状态信息。以下是不同类型的特殊参数及其包含的信息：
|    Parameter    |   Example        |       Description     |
|:----------------|:-----------------|:----------------------|
|`"$*"`|`echo "Arguments: $*"`|扩展单个`string`，将所有位置参数连接成一个，并用IFS中的第一个字符（默认情况下为空格）分隔。注意：除非您明确打算连接所有参数，否则切勿使用此参数。你几乎总是想用它`@`来代替。|
|`"$@"`|`rm "$@"`|将位置参数扩展为单独参数的列表。|
|`"$#"`|`echo "Count: $#"`|展开为一个数字，指示可用位置参数的数量。|
|`"$?"`|`(( $? == 0 )) \| echo "Error: $?"`|展开刚刚完成的最后一个（同步）命令的退出代码。退出代码 0 表示命令成功，任何其他数字表示命令失败的原因。|
|`"$-"`|`[[ $- = *i* ]]`|扩展为当前在 shell 中处于活动状态的选项标志集。选项标志配置 `shell` 的行为，该示例测试该标志是否存在i，表明 `shell` 是交互式的（有提示）并且没有运行脚本。|
|`"$$"`|`echo "$$" > /var/run/myscript.pid`|扩展一个数字，该数字是 shell 进程（正在解析代码）的唯一进程标识符。|
|`"$!"`|`kill "$!"`|展开一个数字，该数字是在后台（异步）启动的最后一个进程的唯一进程标识符。该示例向后台进程发出信号，表明该终止了。|
|`"$_"`|`mkdir -p ~/workspace/projects/myscripts && cd "$_"`|扩展到上一个命令的最后一个参数。|

就像位置参数一样，特殊参数是只读的：您只能使用它们来扩展信息，而不能存储信息。

##### Shell 内部变量

您已经知道什么是 `shell` 变量。您是否知道 `bash shell` 还为您创建了一些变量？这些变量用于各种任务，并且可以方便地从 `shell` 查找某些状态信息或更改某些 `shell` 行为。

{% note success %}
内部 `shell` 变量是名称全部大写的 `shell` 变量。几乎所有环境变量都是如此。重要的是要确保当我们开始创建自己的 `shell` 变量时，我们不会意外地使用我们不知道的 `shell` 变量的名称，这个错误将导致各种危险和意外的行为。值得庆幸的是，`shell` 变量名称区分大小写，因此为了避免意外覆盖 `shell` 内部变量或同名的系统导出变量，一般规则是：

您应该将所有自己的 `shell` 变量设置为小写。如果创建环境变量，请为其指定一个全大写名称。
{% endnote %}
虽然 `bash` 实际上定义了相当多的内部 `shell` 变量，但其中大多数都不是很有用。其他的可以使用，但仅在非常特定的情况下使用。其中许多变量要求您了解更高级的 `bash` 概念。我将简要提及一些现阶段值得了解的内部 `shell` 变量。内部 `shell` 变量的完整列表可以在 中找到`man bash`。
```
BASH	                     /usr/local/bin/bash
This variable contains the full pathname of the command that started the bash you are currently in.

BASH_VERSION	            4.4.0(1)-release
A version number that describes the currently active version of bash.

BASH_VERSINFO	            [ 4, 4, 0, 1, release, x86_64-apple-darwin16.0.0 ]
An array of detailed version information on the currently active version of bash.

BASH_SOURCE	               myscript
This contains all the filenames of the scripts that are currently running. The first is the script that's currently running.Usually it is either empty (no scripts running) or contains just the pathname of your script.

BASHPID	                  5345
This contains the process ID of the bash that is parsing the script code.

UID	                     501
Contains the ID number of the user that's running this bash shell.

HOME	                     /Users/lhunath
Contains the pathname of the home directory of the user running the bash shell.

HOSTNAME	                  myst.local
The name of your computer.

LANG	                     en_CA.UTF-8
Used to indicate your preferred language category.

MACHTYPE	                  x86_64-apple-darwin16.0.0
A full description of the type of system you are running.

PWD	                     /Users/lhunath
The full pathname of the directory you are currently in.

OLDPWD	                  /Users/lhunath
The full pathname of the directory you were in before you came to the current directory.

RANDOM	                  12568
Expands a new random number between 0 and 32767, every time.

SECONDS	                  338217
Expands the number of seconds your bash shell has been running for.

LINES	                     48
Contains the height (amount of rows or lines) of your terminal display.

COLUMNS	                  178
Contains the width (amount of single-character spaces) of a single row in your terminal display.

IFS	                     $' \t\n'
The "Internal Field Separator" is a string of characters that bash uses for word-splitting of data. By default, bash splits on spaces, tabs and newlines.

PATH	                     /Users/lhunath/.bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/libexec
The list of paths bash will search for executable programs when you run a command.

PS1	                     \s-\v\$ 
This contains a string that describes what the interactive bash shell's prompt should look like.

PS2	                     > 
This contains a string that describes what the interactive bash shell's secondary prompt should look like. The secondary prompt is used when you finish a command line but the command isn't yet complete.
```
正如我所提到的，还有许多其他内部 `shell` 变量，但它们每个都服务于目前并不有趣的非常具体的高级案例。如果您正在寻找有关 `bash` 当前如何运行的一些信息，您很可能可以在其内部 `shell` 变量之一中找到它。

1. 启动一个新的 bash shell，输出其第一个参数并Hello World!作为参数传递给它。
```bash
$ bash -c 'echo "$1"' -- 'Hello World!'
```
2. 启动一个 bash shell，输出传入的参数数量并传入参数1,2和The Third。
```bash
$ bash -c 'echo "$#"' -- 1 2 'The Third'
```
3. 启动一个 bash shell，将位置参数移走，然后输出第一个。传入参数1,2和The Third。
```bash
$ bash -c 'shift; echo "$1"' -- 1 2 'The Third'
```
4. 启动一个 bash shell，输出最后传入的参数并传入参数1,2和The Third。
```bash
$ bash -c 'echo "${@: -1}"' -- 1 2 'The Third'
```

#### 数组

最后但同样重要的是，我们可能会遇到最有趣的 bash 参数：数组。

##### 数组的作用

数组是一个奇特的词，指的是参数，它不能容纳一个字符串，而是容纳一整串字符串。存储事物列表的概念并不新鲜 - 我们之前在本指南中已经看到过它，例如PATH存储 bash 的目录路径名列表以查找命令程序。然而，引入数组是为了解决使用简单字符串变量存储事物列表时出现的一个非常重要的问题。

在简单变量中存储事物列表的问题是，当您对此列表的单独元素感兴趣时，您不可避免地需要将这个单个变量拆分为这些单独的元素。然而，我们大多数人甚至没有注意到这是一个问题：作为人类，我们非常擅长根据具体情况做到这一点。当我们看到诸如 的名字时Leonard Cohen，我们认识到它由两个单独的名字组成，这两个名字一起构成一个人的全名。现在，当我们查看诸如 之类的字符串时Leonard Cohen - Adam Cohen - Lorca Cohen，我们立即将其识别为三个不同名称的列表：我们立即识别出该字符串中用破折号分隔名称的模式。事实上，我们非常擅长这一点，以至于当我们看到诸如此类的名称列表时，我们通常甚至不需要停下来思考Susan Q. - Mary T. - Steven S. - Anne-Marie D. - Peter E.。我们甚至擅长在较大的字符串中找到相关的上下文单元，例如由行和段落组成的诗歌。但不幸的是，当我们开始考虑让计算机为我们处理数据时，我们需要停止思考我们优秀的人类抽象，并穿上我们的认知婴儿鞋。计算机不知道这Susan Q. - Mary T. - Steven S. - Anne-Marie D. - Peter E.是一个名称列表，它当然也不知道这些名称是用破折号分隔的，并且它绝对无法猜测这是一个单一的名称，Anne-Marie而不是其中两个不同的人的名称。列表。
明确列表中单独元素的一个好方法是使用命令的参数。还记得我们什么时候学过引用吗？事实上，这是回顾我们的引用课程的好时机。
```bash
$ ls -l 05 Between Angels and Insects.ogg
```
在此命令中，我们向命令传递ls一个包含许多参数的列表，bash 会将每个参数假定为单独的文件名。这显然不是预期的效果，但 bash 并不像我们人类那样擅长从任意数据中获取上下文意义。因此，我们必须明确列表中的元素是什么，这一点很重要：
```bash
$ ls -l "05 Between Angels and Insects.ogg"
```
现在我们已经清楚地向 bash 表明我们的列表仅包含一个文件名，并且该文件名本身包含多个单词，该ls命令能够正确完成其工作。变量也存在同样的问题。如果我们想创建一个包含我们要删除的所有文件的列表的变量怎么办？我们如何创建这样一个列表，然后我们可以将该列表中的每个不同元素传递给命令进行rm删除，而不会冒 bash 误解我们的文件名需要如何解释的风险？
答案是数组：
```bash
$$ _files=( myscript hello.txt "05 Between Angels and Insects.ogg" )
rm -v "${files[@]}"
```
为了创建数组变量，bash 引入了一个略有不同的赋值运算符`：=( )`。与标准一样=，我们将变量的名称放在运算符的左侧，但是，分配给该变量的值列表应该很好地放在大括号和大(括号之间)。

您可能还记得我们关于变量赋值的部分，重要的是我们不要在赋值值周围放置语法空格：`bash` 之后的空格将赋值拆分为命令名称和参数对；我们的赋值值中不带引号的空格会导致 bash 将值拆分为部分赋值，后跟命令名称。使用这种新的数组赋值语法，大括号内可以自由地允许空格，事实上，它们用于分隔数组值列表的许多元素。但就像常规变量赋值一样，当空格需要成为变量数据的一部分时，必须用引号引起来，以便 `bash` 将空格解释为文字。请注意，在上面的示例中，我们使用语法myscript和之间的间距hello.txt，允许 bash 将这两个单词理解为列表中不同的元素，而我们在单词和之间使用文字间距- 这里的空格是文件名的一部分，它不应该导致 bash 将单词分成单独的列表elements：空格是字面意思，因此我们引用了它。

事实上，这些语法规则并不是什么新鲜事。我们已经知道如何将不同的参数传递给我们的命令，并且将不同的元素传递给我们的数组赋值运算符也没有什么不同。

#最后，创建文件列表后，我们扩展命令的参数rm。如果您还记得上面的参数扩展部分，扩展是通过在参数名称前加上$- 符号来进行的。然而，与常规参数扩展相反，我们对扩展单个参数不感兴趣：我们想要做的是将列表中的每个元素扩展为命令的单独且不同的参数rm。为此，我们将参数名称添加为后缀`[@]`，现在需要使用花括号 ( ) 将整个参数括起来{ }，以确保 bash 将整个参数理解为单个参数扩展单元。files使用语法扩展参数`"${files[@]}"`：
```bash
$
rm 'myscript'
rm 'hello.txt'
rm '05 Between Angels and Insects.ogg'      #rm -v myscript hello.txt "05 Between Angels and Insects.ogg"
```
Bash 巧妙地将数组列表中的每个单独元素扩展为命令的单独参数`rm!`
{% note warning %}
与所有参数扩展一样，将所有数组参数扩展用双引号括起来至关重要。与常规参数扩展一样，如果无法对扩展进行双引号引用，则会导致 `bash` 对完全不同的数组项列表中的所有值进行分词，从而导致命令的单词参数列表损坏：
```bash
$ rm -v ${files[@]}
removed 'myscript'
removed 'hello.txt'
rm: cannot remove '05': No such file or directory
rm: cannot remove 'Between': No such file or directory
rm: cannot remove 'Angels': No such file or directory
rm: cannot remove 'and': No such file or directory
rm: cannot remove 'Insects.ogg': No such file or directory
```
回想一下我们之前学过的引用规则：如果你的参数中有空格或符号，你必须引用双引号。
```bash
$ rm -v "${files[@]}"
```
{% endnote %}

除了数组赋值和数组扩展之外，bash 还提供了一些我们可以对数组执行的其他操作：
```bash
$files+=( selfie.png )        # 使用+=( )运算符，我们可以将项目列表附加到数组的末尾。
$files=( *.txt )              # 就像在命令的参数中一样，我们可以在这里扩展全局模式。
$echo "${files[0]}"           # 要扩展数组中的单个项目，请指定该项目的序号。
$echo "$files"                # 如果我们忘记了数组扩展语法，bash 将仅扩展第一项。
$unset "files[3]"             # 要从数组中删除特定项目，我们使用unset.
                              # 但请注意：我们在这里不使用 @ $，因为我们没有扩展该值！
```
除了[@]将数组元素扩展为不同参数的后缀之外，bash 还可以将所有数组元素扩展为单个参数。这是使用`[*]`后缀完成的。bash 如何将所有单独的元素合并到一个参数中？我们可能希望它通过多种方式来做到这一点——它是否创建一个以空格分隔的字符串？它是否将所有元素挤压在一起形成一个长字符串，没有任何元素分隔？也许它可以创建一个字符串，其中每个元素都位于单独的行上？事实是，由于上述所有原因，没有一种策略可以将不同的元素合并到单个字符串中而不出现问题。因此该运营商非常可疑`[@]`并且在几乎所有情况下都应该避免！

事实上，`bash` 允许您在使用`[*]`: 时通过查看IFS内部 shell 变量的当前值来选择如何将元素合并到单个字符串中。`Bash` 使用此变量的第一个字符（默认情况下是空格）来分隔结果字符串中的元素：
```bash
$ names=( "Susan Quinn" "Anne-Marie Davis" "Mary Tate" )
$ echo "Invites sent to: <${names[*]}>."                    # Results in a single argument where elements are separated by a literal space.
Invites were sent to: <Susan Quinn Anne-Marie Davis Mary Tate>.
$ ( IFS=','; echo "Invites sent to: <${names[*]}>." )       # When we change IFS to a ,, the distinct elements become more clear.
Invites were sent to: <Susan Quinn,Anne-Marie Davis,Mary Tate>.
```
由于包含多个不同元素的单个字符串几乎总是有缺陷的，并且不如那些元素很好分离的数组变量有用，因此后缀的实际用途很少`[*]`。但有一个例外：该运算符对于向用户显示元素列表非常有用。当我们尝试向人类显示数组的值时，我们不必太担心输出的语法正确性。在上面的示例中，`IFS`被更改以,说明向用户显示数组中的值的常见方法。
{% note warning %}
在修改内部 bash shell 变量时（例如上面使用IFS 的示例），务必要非常小心：
当我们更改内部 shell 变量时，我们需要认识到我们正在改变 bash 的运行方式。将IFS的值更改为逗号 ( ,) 以便使用[*]后缀扩展文件是可以的，但是如果您继续执行脚本，同时将IFS设置为其非默认值“ ,”，则许多其他事情基于IFS值的 bash 会突然出现故障。

正是由于这个原因，您应该始终将内部 shell 配置更改的范围限制在尽可能缩小的脚本区域内。您可能已经注意到，在上面的示例中，我们( )在代码周围引入了大括号。这些大括号创建一个子 shell，在其中执行代码，当大括号结束时，具有非标准IFS值的子 shell 也随之结束。因此，原始脚本的 bash shell 从未修改过其 IFS变量，我们避免了这些意外故障。
{% endnote %}
最后，我们之前了解的所有特殊参数扩展运算符也可以应用于数组扩展，但我们将重新迭代其中一些，因为它们的效果在扩展多个不同元素的上下文中非常有趣。对于初学者来说，该`${parameter[@]/pattern/replacement}`运算符及其所有变体在扩展时都将其替换逻辑明确应用于每个元素：
```bash
$ names=( "Susan Quinn" "Anne-Marie Davis" "Mary Tate" )
$ echo "${names[@]/ /_}"                                    # Replace spaces by underscores in each name.
Susan_Quinn Anne-Marie_Davis Mary_Tate
$ ( IFS=','; echo "${names[*]/#/Ms }" )                     # More interestingly: replace the start of each name with Ms ,
Ms Susan Quinn,Ms Anne-Marie Davis,Ms Mary Tate             # effectively prefixing every element with a string as we expand them
```
运算`${#parameter}`符与后缀相结合`[@]`给出了元素的计数：
```bash
$ echo "${#names[@]}"
3
$ echo "${#names[1]}"            # But we can still get the length of a string by specifying directly
16                               # which string element in the array we want to get the length of.
```
最后，该`${parameter[@]:start:length}`运算符可用于获取数组的切片或“子集”：
```bash
$ echo "${names[@]:1:2}"
Anne-Marie Davis Mary Tate
$ echo "${names[@]: -2}"         # Specifying a negative start allows us to count backwards from the end!
Anne-Marie Davis Mary Tate       # While omitting the length yields "all remaining elements" from the start
```
请注意，在负起始值前面包含一个空格很重要：如果我们省略空格，`bash` 会感到困惑，并认为您正在尝试调用运算符，只要参数的值为空，该运算符就会替换默认值`${parameter:-value}`。`value`这显然不是我们想要的。

就是这样！您已经牢牢掌握了 `bash shell` 语言绝对最重要和最有用的方面：它的参数和扩展，以及我们可以将运算符应用于扩展值并根据我们的各种需求塑造它们的多种方式。
