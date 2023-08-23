---
title: 【Bash 编程】Tests And Conditionals
date: 2023-08-03 20:23:01
tag: 
   - bash programing
   - Cloud Native
category:
   - Bash Programing
---

与没有条件分支的线性脚本相比，带有条件分支的脚本更加广泛且多用途，就像游戏与书籍的线性叙述相比具有多用途一样。那么，为什么我们需要条件语句呢？我们需要他们编写能够动态处理不同情况的脚本，并根据情况改变其运行方式。让我们从一个非常简单的条件开始，以帮助我们正确地开始新的一天：
```bash
$ read -p "Would you like some breakfast? [y/n] "
Would you like some breakfast? [y/n] n
$ if [[ $REPLY = y ]]; then
>     echo "Here you go, an egg sandwich."              # Branch #1
> else
>     echo "Here, you should at least have a coffee."   # Branch #2
> fi
Here, you should at least have a coffee.
```
<!-- more -->
与我们之前编写的所有代码相比，条件语句的关键在于，除非情况发生变化，否则我们现在的代码永远不会被执行。仅执行了第二个分支中的代码，即使我们在第一个分支中有实际代码，bash 也从未真正执行过它。只有当情况（在这种情况下，我们对前面问题的答案）发生变化时，脚本的执行分支才会发生变化，并且我们会看到第一个分支中的代码被执行，但同时，这将导致第二个分支分支变得“死亡”。

##### The if compound.

该if语句在编程语言中非常普遍，几乎可以肯定，当我们考虑在代码中构建一个条件时，首先想到的就是它。这并非偶然：这些陈述清晰、简单且明确。这也使它们成为我们熟悉 bash 条件语句的绝佳起点。
```
if list [ ;|<newline> ] then list ;|<newline>
[ elif list [ ;|<newline> ] then list ;|<newline> ] ...
[ else list ;|<newline> ]
fi
```
```bash
if ! rm hello.txt; then 
    echo "Couldn't delete hello.txt." >&2; 
    exit 1; 
fi

if rm hello.txt; then 
    echo "Successfully deleted hello.txt."
else 
    echo "Couldn't delete hello.txt." >&2; 
    exit 1; 
fi

if mv hello.txt ~/.Trash/; then 
    echo "Moved hello.txt into the trash."
elif rm hello.txt; then 
    echo "Deleted hello.txt."
else 
    echo "Couldn't remove hello.txt." >&2; 
    exit 1; 
fi
```
该复合词的语法if虽然一开始有点冗长，但本质上非常简单。我们从关键字开始`if`，然后是命令列表。该命令列表将由 `bash` 执行，完成后，`bash` 会将最终的退出代码交给`if`要评估的化合物。如果退出代码为零`（0= success）`，则将执行第一个分支。否则，将跳过第一个分支。

如果跳过第一个分支，`if`复合会将执行机会传递到下一个分支。如果`elif`有一个或多个分支可用，这些分支将依次执行并评估它们自己的命令列表，如果成功，则执行它们的分支。请注意，一旦`if`执行复合的任何分支，其余分支就会自动跳过：仅执行一个分支。如果`if`两个 `elif`分支都没有资格执行，`else`则将执行该分支（如果存在）。

实际上，`if`复合语句是表达一系列要执行的潜在分支的语句，每个分支前面都有一个命令列表，用于评估是否应选择该分支。大多数if语句只有一个分支或一个主分支和一个`else`分支。

##### Conditional command lists

如上所述，该`if`语句与大多数其他条件语句类似，评估`List`命令的最终退出代码，以确定是否应采用或跳过其相应的条件分支。您将遇到的几乎所有`if`条件语句和其他条件语句都只不过是一个简单命令作为其条件，但仍然可以提供简单命令的完整列表。当我们这样做时，重要的是要理解只有执行整个列表后的最终退出代码才与分支的评估相关：
```bash
$ read -p "Breakfast? [y/n] "; if [[ $REPLY = y ]]; then echo "Here are your eggs."; fi
Breakfast? [y/n] y
Here are your eggs.
$ if read -p "Breakfast? [y/n] "; [[ $REPLY = y ]]; then echo "Here are your eggs."; fi
Breakfast? [y/n] y
Here are your eggs.
```
两者在操作上是相同的。在第一种情况下，我们的read命令先于if语句；在后者中，我们的read命令嵌入在初始分支条件中。本质上，风格或偏好的选择将决定您更喜欢哪种方法。关于此事的一些想法：

- 嵌入数据收集命令为条件创建了一种“完整”方法：条件成为由其所有依赖项组成的单元。
- 数据收集命令之前的条件语句将两个不同的操作分开。`elif`当其他分支成为语句的一部分时，它还使条件更加对称或“平衡” 。

##### Conditional test commands

最常用作条件的命令是`testcommand`，也称为`[command`。这两个是同义词：它们是相同的命令，但名称不同。唯一的区别是，当您用作`[`命令名称时，必须使用尾随`]`参数来终止命令。

然而，在现代 `bash` 脚本中，`test`出于所有意图和目的，该命令已被它的两个弟弟所取代： `[[`和`((`关键字。该`test`命令实际上已经过时，其有缺陷且脆弱的语法无法与`[[bash((`解析器授予的特殊权力相匹配。

`It may seem strange at first thought, but it is actually quite interesting a revelation to notice that [ and [[, as we've seen them appear several times in if and other sample statements in this guide, are not some special form of if-syntax - no! They are simple, ordinary commands, just like any other command. The [[ command name takes a list of arguments and its final argument must be ]]. Similarly, [ is a command name which takes test arguments and must be closed with a trailing ] argument. This is especially noticable when we make a mistake and omit spaces between these command names and their arguments:`
```bash
$ [[ Jack = Jane ]] && echo "Jack is Jane" || echo "Jack is not Jane"
Jack is not Jane
$ [[Jack = Jane ]] && echo "Jack is Jane" || echo "Jack is not Jane"
-bash: [[Jack: command not found
$ [[ Jack=Jane ]] && echo "Jack is Jane" || echo "Jack is not Jane"
Jack is Jane
```
第一条语句编写正确，我们得到了预期的输出。在第二条语句中，我们忘记将`[[`命令名称与第一个参数分开，导致 bash 解析器去寻找名为`[[Jack`的命令。毕竟，当 `bash` 解析此命令并将命令的名称和参数分词为标记时，第一个以空格分隔的标记是整个字符串`[[Jack`。

