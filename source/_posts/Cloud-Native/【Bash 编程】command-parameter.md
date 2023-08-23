---
title: 【Bash 编程】命令行和参数
date: 2023-08-01 20:23:01
tag: 
   - bash programing
   - Cloud Native
category:
   - Bash Programing
---

bash shell 是一个二进制程序，可以交互式或非交互式运行，通常在终端仿真器程序提供的基于文本的界面中运行。

当您从图形用户界面启动终端仿真器程序时，您将看到一个打开的窗口，其中包含文本。此窗口中显示的文本既是终端中运行的程序的输出，也是您使用键盘等发送到这些程序的字符。bash 程序只是可以在终端中运行的众多程序之一，因此请务必注意，bash 并不是使文本出现在屏幕上的原因。终端程序会处理这个问题，从 bash 中获取文本并将其放置在窗口中供您查看。终端可以对终端中运行的其他与 bash 完全无关的程序执行相同的操作，例如邮件程序或 IRC 客户端。

<!-- more -->
{% asset_img bash_flow.png %}

简而言之，程序是一组可以由系统内核执行的预先编写的指令。程序直接向内核发出指令。从技术上讲，内核也是一个程序，但它会不断运行并与硬件进行通信。
{% asset_img bash_flow2.png %}

如果一个程序需要其输出到另一个程序的输入（而不是您的显示），它将指示内核将其标准输出连接到另一个程序的标准输入。现在，它发送到其标准输出文件描述符的所有信息都将流入另一个程序的标准输入文件描述符。文件、设备和进程之间的这些信息流称为流。

流是在运行系统中的文件、设备和进程之间的链接中流动的信息（具体来说，字节）。它们可以传输任何类型的字节，并且接收端只能按照发送的顺序消耗它们的字节。如果我有一个程序输出连接到另一个程序的名称，则第二个程序只能在第一次从流中读取第一个名称后才能看到第二个名称。读取完第二个名称后，流中的下一个内容是第三个名称。从流中读取名称后，程序可以将其存储在某处，以备以后再次需要时使用。从流中读取名称会消耗流中的这些字节，并且流会前进。流无法倒回，名称也无法重新读取。

{% asset_img bash_flow3.png %}

在上面的示例中，两个 bash 进程通过流链接。第一个 bash 进程从键盘读取输入。它在标准输出和标准错误上发送输出。标准错误的输出连接到终端显示器，而标准输出的输出连接到第二个进程。请注意:第一个进程' 如何FD 1连接到第二个进程' FD 0? 第二个进程在从其标准输入读取时会消耗第一个进程的标准输出。第二个进程的标准输出又连接到终端的显示器。要尝试这种动态，您可以在终端中运行以下代码，其中(和)符号创建两个子 shell，并且|符号将前者连接FD 1到后者FD 0：
```bash
$( echo "Your name?" >&2; read name; echo "$name" ) | ( while read name; do echo "Hello, $name"; done )
Your name？
Maarten Billemont
Hello，Maarten Billemont
```
请注意，终端中显示的唯一文本是连接到终端显示屏的命令的输出，以及终端发送到程序的输入。
重要的是要理解文件描述符是特定于进程的：只有在引用特定进程时才有意义，“标准输出”才有意义。在上面的示例中，您会注意到第一个进程的标准输入与第二个进程的标准输入不同。您还会注意到，第一个进程的 FD 0（标准输出）连接到第二个进程的 FD 1（标准输入）。文件描述符不描述连接进程的流，它们仅描述可以连接这些流的进程的插头。

#### 命令和参数

##### 什么是 bash 命令呢？

`bash shell` 语言的核心是它的命令。你的命令一步一步、一个命令一个命令地告诉 `bash` 你需要它做什么。Bash 通常一次接受您的一个命令，执行该命令，完成后返回给您执行下一个命令。我们称之为同步命令执行。重要的是要理解，当 bash 忙于处理您给它的命令时，您无法直接与 bash 交互：您必须等待它准备好执行其命令并返回到脚本。对于大多数命令，您几乎不会注意到这一点：它们执行得如此之快，bash 将在您意识到之前返回下一个命令。

不过，某些命令可能需要很长时间才能完成。特别是启动可以与之交互的其他程序的命令。例如，命令可能会启动文件编辑器。当您与文件编辑器交互时，bash 会退居二线并等待文件编辑器结束（这通常意味着您退出它）。当文件编辑器程序停止运行时，命令结束，bash 通过询问您下一步要做的事情来恢复操作。您会注意到，当编辑器运行时，您不再处于 bash 提示符处。一旦编辑器退出，bash 提示符就会重新出现：
```bash
$ exbash command to run the "ex" program.
: iex command to "insert" some text.
Hello!
.A line with just a dot tells ex to stop inserting text.
: w greeting.txtex command to "write" the text to a file.
"greeting.txt" [New] 1L, 7C written
: qex command to "quit" the program.
$ cat greeting.txtAnd now we're back in bash!
Hello!The "cat" program shows the contents of the file.
$ 
```
我们首先向 `bash` 提供启动ex文件编辑器的命令。发出此命令后，我们的提示符发生了变化：我们现在输入的任何文本都会发送到 `ex`，而不是 `bash`。当 ex 运行时，`bash` 处于睡眠状态，等待您的 `ex` 会话结束。当您使用该命令退出 `ex` 时`q`，`exbash` 命令结束，并且 `bash` 已准备好接收新命令。为了告诉您这一点，它会再次向您显示提示符，允许您输入下一个 `bash` 命令。我们使用 `bash` 命令来完成该示例cat `greetings.txt`，该命令告诉 `bash` 运行 `cat` 程序。`cat` 命令非常适合输出文件内容。`greetings.txt`示例中的 `cat` 命令用于在使用 `ex` 程序编辑完文件后查找文件中的内容 。

> bash 命令是 bash 可以独立执行的最小代码单元。执行命令时，您无法与 bash shell 交互。一旦 bash 执行完一个命令，它就会返回给您执行下一个命令。

##### Bash 命令如何读取？

`Bash` 主要是一种基于行的语言。因此，当 `bash` 读取您的命令时，它会逐行执行。大多数命令仅构成一行，除非 `bash` 命令的语法明确表明您的命令尚未完成，否则一旦您结束该行，`bash`将立即认为这是命令的结束。因此，输入一行文本并按下回车键通常会导致 `bash` 开始执行该行文本所描述的命令。

然而，有些命令跨越多行。这些通常是块命令或带引号的命令：
```bash
$ read -p "Your name? " nameThis command is complete and can be started immediately.
Your name? Maarten Billemont
$ if [[ $name = $USER ]]; thenThe "if" block started but wasn't finished.
>     echo "Hello, me."
> else
>     echo "Hello, $name."
> fiNow the "if" block ends and bash knows enough to start the command.
Hello, Maarten Billemont.
```
从逻辑上讲，bash 在拥有足够的信息来完成其工作之前无法执行命令。上面示例中命令的第一行`if`（我们稍后将更详细地介绍这些命令的作用）没有包含足够的信息，让 `bash` 无法知道测试成功或失败时该怎么做。结果，`bash` 显示了一个特殊的提示：`>`。这个提示的实质意思是：你给我的命令还没有结束。我们继续为命令提供额外的行，直到到达`fi`。当我们结束该行时，bash 知道您已完成提供条件。它立即开始运行整个块中的所有代码。我们很快就会看到 `bash` 语法中定义的不同类型的命令，但是`if`我们刚刚看到的命令称为复合命令，因为它将一堆基本命令组合成一个更大的逻辑块。

在每种情况下，我们都会将命令传递给交互式 bash 会话。正如我们之前所解释的，bash 还可以在非交互模式下运行，它从文件或流中读取命令，而不是询问您命令。在非交互模式下，`bash` 没有提示符。除此之外，它的操作几乎相同。我们可以复制上面示例中的 `bash` 代码并将其放入文本文件中：使用您最喜欢的文本编辑器再次打开文件，并在其顶部`hello.txt`添加一个`hashbang` ，作为脚本的第一行：`#!/usr/bin/env bash`。
```bash
#!/usr/bin/env bash
read -p "Your name? " name
if [[ $name = $USER ]]; then
    echo "Hello, me."
else
    echo "Hello, $name."
fi
```
您已经创建了第一个 `bash` 脚本。什么是 `bash` 脚本？它是一个包含 `bash` 代码的文件，可以像计算机上的任何其他程序一样由内核执行。从本质上讲，它本身就是一个程序，尽管它确实需要 `bash` 解释器来完成将 bash 语言翻译成内核可以理解的指令的工作。这就是我们刚刚添加到文件中的“`hashbang`”行的用处：它告诉内核需要使用什么解释器来理解该文件中的语言，以及在哪里可以找到它。我们称其为“`hashbang`”，因为它总是以“`hash`”开头，#后跟“`bang`”!。然后，您的 `hashbang` 必须为任何能够理解文件中的语言并且可以采用单个参数的程序指定绝对路径名。不过，我们的 `hashbang` 有点特别：我们引用了程序`/usr/bin/env`，它并不是真正理解 `bash` 语言的程序。它是一个可以查找并启动其他程序的程序。在我们的例子中，我们使用一个参数告诉它找到程序`bash`并使用它来解释脚本中的语言。为什么我们使用这个名为 的“中间”程序env？它与名称之前的内容密切相关：路径。我们相对确定地知道该`env`程序位于该`/usr/bin`路径中。然而，鉴于操作系统和配置多种多样，我们无法确定`bash`程序已安装。这就是为什么我们使用该 `env`程序来为我们找到它。这有点复杂！但是现在，我们的文件在添加 `hashbang` 之前和之后有什么区别呢？`

```bash
$ chmod +x hello.txt       # Mark hello.txt as an executable program.
$ ./hello.txt              # Tell bash to start the hello.txt program.
```
大多数系统要求您将文件标记为可执行文件，然后内核才允许您将其作为程序运行。一旦我们这样做了，我们就可以hello.txt像启动任何其他程序一样启动该程序。内核将查看文件内部，找到 hashbang，使用它来追踪 bash 解释器，最后使用 bash 解释器开始运行文件中的指令。Bash 通过读取行​​来获取命令。一旦读取了足够的行来组成完整的命令，bash 就会开始运行该命令。通常，命令只有一行长。交互式 bash 会话会在提示符下读取您的行。非交互式 bash 进程从文件或流中读取命令。以hashbang作为第一行（和可执行权限）的文件可以像任何其他程序一样由系统内核启动。

##### bash命令的基本语法

```bash
[ var=value ... ] name [ arg ... ] [ redirection ... ]
echo "Hello world."
IFS=, read -a fields < file
```
在命令名称之前，您可以选择放置一些`var`赋值。这些变量分配仅适用于该命令的环境。稍后我们将更深入地讨论变量和环境。命令的名称是第一个单词（在可选分配之后）。`Bash` 找到具有该名称的命令并启动它。稍后我们将详细了解有哪些类型的命名命令以及 `bash` 如何找到它们。命令名称后面可以选择跟随一个`arg`单词列表，即命令参数。我们很快就会了解什么是参数及其语法。最后，命令还可以应用一组重定向操作。如果您还记得我们在前面部分中对文件描述符的解释，重定向是更改文件描述符插入指向的内容的操作。他们改变连接到我们的命令进程的流。我们将在以后的部分中了解重定向的威力。

##### 管道

Bash 附带了大量“语法糖”，使常见任务比仅使用基本语法更容易执行。管道是您将大量使用的糖的一个例子。它们是通过将第一个进程的标准输出链接到第二个进程的标准输入来“连接”两个命令的便捷方法。这是终端命令相互通信和传递信息的最常见方式。
```bash
[时间[ -p ]][ ! ]命令[ [ | | |& ]命令2 ...]
echo Hello | rev
! rm greeting.txt
```
们很少使用`time`关键字，但它可以方便地了解运行我们的命令需要多长时间。这个!关键字一开始有点奇怪，就像 time 关键字一样，它与连接命令没有太大关系。当我们讨论条件和测试命令是否成功时，我们将了解它的作用。第一个`command`和第二个`command2`可以是本节中的任何类型的命令。`Bash` 将为每个命令创建一个子 `shell`，并设置第一个命令的标准输出文件描述符，使其指向第二个命令的标准输入文件描述符。这两个命令将同时运行，`bash` 将等待这两个命令结束。

两个命令之间有一个|符号。这也称为“管道”符号，它告诉 bash 将第一个命令的输出连接到第二个命令的输入。或者，我们可以使用|&命令之间的符号来指示我们不仅希望第一个命令的标准输出，而且还希望将其标准错误连接到第二个命令的输入。这通常是不希望的，因为标准错误文件描述符通常用于向用户传达消息。如果我们将这些消息发送到第二个命令而不是终端显示器，我们需要确保第二个命令可以处理接收这些消息。

##### 列表

列表是其他命令的序列。本质上，脚本是一个命令列表：一个命令接一个命令。列表中的命令由控制运算符分隔，该运算符指示 bash 在执行之前的命令时要执行的操作。
```bash
command control-operator [ command2 control-operator ... ]
cd music; mplayer *.mp3
rm hello.txt || echo "Couldn't delete hello.txt." >&2
```
命令之后是控制运算符，它告诉 `bash` 如何执行该命令。最简单的控制运算符只是开始一个新行，这相当于;告诉 `bash` 仅运行该命令并等待其结束，然后再前进到列表中的下一个命令。第二个示例使用||控制运算符，它告诉 `bash` 像平常一样运行之前的命令，但在完成该命令后，仅当之前的命令失败时才移至下一个命令。如果前面的命令没有失败，`||`操作员将使 bash 跳过后面的命令。这对于在命令失败时显示错误消息很有用。我们将在后面的部分中更深入地讨论所有控制运算符。

##### 复合命令

复合命令是内部具有特殊语法的命令。它们可以做很多不同的事情，但表现为命令列表中的单个命令。最明显的例子是命令块：该块本身表现为单个大命令，但其内部是一堆“子”命令。有很多不同类型的复合命令。
```bash
if list [ ;|<newline> ] then list [ ;|<newline> ] fi
    { list ; }

if ! rm hello.txt; then echo "Couldn't delete hello.txt." >&2; exit 1; fi

rm hello.txt || { echo "Couldn't delete hello.txt." >&2; exit 1; }
```
两个示例执行相同的操作。第一个示例是复合命令，第二个示例是命令列表中的复合命令。我们`||`之前简单讨论过该运算符：除非它前面的命令失败，否则会跳过它右侧的命令。这是一个很好的例子，说明了复合命令的一个重要属性：它们的行为就像命令列表中的一个命令一样。第二个示例中的复合命令从 {开始，一直持续到下一个}，因此大括号内的所有内容都被视为单个命令，这意味着我们有一个包含两个命令的命令列表：命令rm后跟`{ ... }`复合命令。如果我们忘记了大括号，我们将得到三个命令的命令列表：`rm`命令后跟`echo`命令，然后是`exit`命令。这种差异对于操作员在成功完成`||`前面的命令后决定要做什么时非常重要。`rm`如果rm成功，`||`将跳过其后的命令，如果我们省略大括号，则该命令将只是命令`echo`。大括号将 `echo`和`exit`命令组合成一个复合命令，允许在成功`||`时跳过这两个命令`rm`。

##### Coprocesses

协进程是更多的 `bash` 语法糖：它允许您轻松地异步运行命令（无需让 bash 等待它结束，也称为“`in the background`”），并且还可以设置一些直接连接的新文件描述符插件新命令的输入和输出。您不会太频繁地使用协进程，但是当您执行高级操作时，它们非常方便。
```bash
coproc [ name ] command [ redirection ... ]

coproc auth { tail -n1 -f /var/log/auth.log; }
read latestAuth <&"${auth[0]}"
echo "Latest authentication attempt: $latestAuth"
```
该示例启动一个异步`tail`命令。当它在后台运行时，脚本的其余部分将继续。首先，脚本从名为的协进程读取一行输出`auth`（这是命令输出的第一行`tail`）。接下来，我们编写一条消息，显示从协进程读取的最新身份验证尝试。该脚本可以继续，每次从协进程管道读取时，它都会从命令中获取下一行 `tail`。

##### Functions

当您在 `bash` 中声明一个函数时，您实际上是在创建一个临时的新命令，您可以稍后在脚本中调用该命令。当您在脚本中多次重复同一任务时，函数是一种将命令列表分组到自定义名称下的好方法，以方便您使用。
```bash
name () compound-command [ redirection ]

exists() { [[ -x $(type -P "$1" 2>/dev/null) ]]; }
exists gpg || echo "Please install GPG." <&2
```

`name`首先为您的函数指定这是新命令的名称，稍后您可以通过使用该名称编写一个简单的命令来运行它。命令名称后面是括号()。有些语言使用这些括号来声明函数接受的参数：`bash` 不这样做。括号应始终为空。它们只是表示您正在声明一个函数。接下来是每次运行该函数时将执行的复合命令。要在运行函数期间更改脚本的文件描述符，您可以选择指定函数的自定义文件重定向。

> Bash 命令告诉 bash 执行特定的工作单元。这些工作单元不能再细分：bash 需要知道整个命令才能执行它。不同类型的操作有不同类型的命令。某些命令将其他命令分组或测试其结果。许多命令类型都是语法糖：它们的效果可以通过不同方式实现，但它们的存在是为了使工作变得更容易。

##### 命令的名称和运行程序
```bash
[ var=value ... ] name [ arg ... ] [ redirection ... ]

```
我将仅简要提及别名：在 bash 执行此搜索之前，它首先检查您是否已通过命令名称声明了任何别名。如果您这样做了，它将在继续之前将名称替换为别名的值。别名很少有用，仅在交互式会话中起作用，并且几乎完全被函数取代。几乎在所有情况下您都应该避免使用它们。

> 要运行命令，bash 使用命令的名称并搜索如何执行该命令。按照顺序，bash 将检查它是否具有该名称的函数或内置函数。如果失败，它将尝试将该名称作为程序运行。如果 bash 找不到运行命令的方法，它将输出一条错误消息。

##### The PATH to a program

我们的计算机上安装了各种各样的程序。不同的程序安装在不同的地方。有些程序随我们的操作系统一起提供，其他程序是由我们的发行版添加的，还有一些程序是由我们或我们的系统管理员安装的。在标准 `UNIX` 系统上，程序有几个标准化位置。某些程序将安装在 中`/bin`，其他程序将安装在 中`/usr/bin`，还有一些程序将安装在 中，`/sbin`依此类推。如果我们必须记住程序的确切位置，那将是一个真正的麻烦，特别是因为它们可能因系统而异。`PATH` 环境变量来救援。您的`PATH`变量包含一组应搜索程序的目录。
```
$ ping 127.0.0.1

    PATH=/bin:/sbin:/usr/bin:/usr/sbin
           │     │
           │     ╰──▶ /sbin/ping ?  found!
           ╰──▶ /bin/ping ?  not found.
```
每当您尝试启动一个它还不知道位置的程序时，`Bash` 都会通过查看其列出的目录来搜索此变量。假设您正在尝试启动`ping`安装在 的程序`/sbin/ping`。如果您的`PATH`设置为`/bin:/sbin:/usr/bin:/usr/sbin`，则 `bash` 将首先尝试启动`/bin/ping`，而该启动并不存在。如果失败，它将尝试`/sbin/ping`。它会找到该 `ping`程序，记录其位置以备`ping`将来您再次需要时使用，然后继续为您运行该程序。

如果您对 bash 到底在哪里找到要运行的命令名称相关程序感到好奇，您可以使用type内置函数来查找：
```bash
$ type ping
ping is /sbin/ping
$ type -a echo          # The -a switch tells type to show us all the possibilities
echo is a shell builtin # If we just run 'echo', bash will use the first possibility
echo is /bin/echo       # We have an echo built-in but also a program called echo!
```
还记得上一节中 `bash` 是如何内置一些功能的吗？其中之一是程序的功能`echo`。如果您`echo`在 `bash` 中运行该命令，甚至在 `bash` 尝试`PATH`搜索之前，它就会注意到有一个具有该名称的内置命令并使用它。`type`是可视化此查找过程的好方法。请注意，执行内置命令比启动额外程序要快得多。但是，如果您需要 的`echo`功能而不使用 `bash`，则可以使用该`echo`程序。有时您需要运行未安装在任何目录中的程序PATH。在这种情况下，您必须手动指定 `bash` 可以找到该程序的路径，而不仅仅是其名称：
```bash
$ /sbin/ping -c 1 127.0.0.1
PING 127.0.0.1 (127.0.0.1): 56 data bytes
64 bytes from 127.0.0.1: icmp_seq=0 ttl=64 time=0.075 ms

--- 127.0.0.1 ping statistics ---
1 packets transmitted, 1 packets received, 0.0% packet loss
round-trip min/avg/max/stddev = 0.075/0.075/0.075/0.000 ms
$ ./hello.txt                # Remember our hello.txt script?
Your name?                   # We use the path "." which means "our current directory
```

{% note warning %}
`Bash` 仅对`PATH`不包含`/`字符的命令名称执行搜索。带斜杠的命令名称始终被视为要执行的程序的直接路径名。
{% endnote %}

您可以向您的PATH. 常见的做法是使用 `/usr/local/bin`和  `~/bin`（其中`~`代表用户的主目录）。请记住，这`PATH`是一个环境变量：您可以像这样更新它：
```bash
$$ PATH=~/bin:/usr/local/bin:/bin:/usr/bin
_
```
这将更改当前 bash shell 中的变量。不过，一旦关闭 shell，更改就会丢失。我们将在后面的部分中更深入地介绍环境变量的工作原理以及应如何配置它们。当bash需要运行一个程序时，它使用命令名来执行搜索。Bash逐一搜索PATH 环境变量中的目录，直到找到包含具有您的命令名称的程序的目录。要运行未安装在PATH目录中的程序，请使用该程序的路径作为命令的名称。

在您的主目录中创建一个脚本，将其添加到您的PATH中，然后作为普通命令运行该脚本。
```bash
$ex                      # 您可以在这里替换您最喜欢的编辑器。
: i
#!/usr/bin/env bash
echo "Hello world."
.
: w myscript
"myscript" [New] 2L, 40C written
: q
$ chmod +x myscript
$ PATH=$PATH:~
$ myscript
Hello world.
```

##### Command arguments and quoting literals

```
[ var=value ... ] name [ arg ... ] [ redirection ... ]
```
现在您已经了解了 `bash` 如何查找并运行您的命令，让我们学习如何将指令传递给这些命令。这些指令告诉我们的指挥部到底需要做什么。我们可能运行`rm`命令来删除文件，或者`cp`命令来复制文件，我们可能运行命令`echo`来输出字符串或`read`命令来读取一行文本。但如果没有更多细节、更多上下文，这些命令通常无法做很多事情。我们需要告诉`rm`要删除什么文件，`cp`要复制什么文件以及将副本放在哪里。`echo`想要知道您想要它输出什么，并且`read`可以知道将其读取的文本行放在哪里。我们使用参数提供这种上下文。

{% note warning %}
`bash shell` 脚本中所有错误的主要部分是其作者没有正确理解命令参数的直接结果。通常归咎于对直觉的依赖而不是对规则的理解。
{% endnote %}

从命令语法中可以看出，参数位于命令名称之后。它们是由空格分隔的单词。当我们在 `bash` 上下文中说出单词时，我们指的并不是语言单词。在 `bash` 中，单词被定义为被 `shell` 视为单个单元的字符序列。单词也称为令牌。一个 `bash` 单词可以包含许多语言单词，实际上它可以包含`prose`。为了清楚起见，本指南的其余部分将在适用的地方使用术语参数，以避免术语单词的歧义。重要的是单词或参数对于 `shell` 来说是一个单元：它可以是文件名、变量名、程序名或人名。
```bash
$ rm hello.txt
$ mplayer '05 Between Angels and Insects.ogg' '07 Wake Up.ogg'
```
请注意，它们不是语言单词，而是有意义的单位。在这种情况下，它们都指的是文件名。为了分隔多个参数，我们使用空格。可以是空格和制表符中的一个或两者。通常，您将在参数之间使用一个空格。
现在出现了一个问题：我们在 后面有一个空格05，将其与 分开`Between。shell` 应该如何知道你的文件名是`05 Between Angels and Insects.ogg`不是05 我们如何告诉 `shell` 后面的空格05是字面意思，而不是作为“现在分割单词”的语法？我们的目的是让整个文件名保持“在一起”。也就是说： 其中的空格不应将其拆分为单独的参数。我们需要的是一种方法来告诉 `shell` 它应该按字面意思处理某些内容；意思，按原样使用它，忽略任何语法意义。如果我们可以将空格变成字面意思，它们将不再告诉 bash 将空格05从Between，bash 会将其用作正常的普通空格字符。

`bash`中有两种方法可以使字符变成原义字符：引用和转义。"引用是将字符包裹'在我们想要表达的文本周围的做法。逃逸是放置一个单一的实践\我们要使其字面化的字符前面的字符。上面的示例使用引号来构成整个文件名文字，但不包含文件名之间的空格。我们强烈建议您使用引号而不是转义，因为它会导致更清晰、更易读的代码。更重要的是：转义使得准确判断代码的哪些部分是文字部分、哪些部分不是文字变得更加困难。稍后在不引入错误的情况下编辑文字文本也变得更加不稳定。使用转义而不是引用，我们的示例如下所示：
```bash
$mplayer 05\ Between\ Angels\ and\ Insects.ogg 07\ Wake\ Up.ogg
```
引用是作为 `bash` 用户需要掌握的最重要的技能之一。它的重要性怎么强调都不为过。引用的好处是，虽然有时没有必要，但引用数据很少会出错。这些都是完全有效的：
```bash
$ ls -l hello.txt
-rw-r--r--  1 lhunath  staff  131 29 Apr 17:07 hello.txt
$ ls -l 'hello.txt'
-rw-r--r--  1 lhunath  staff  131 29 Apr 17:07 hello.txt
$ ls -l '05 Between Angels and Insects.ogg' '07 Wake Up.ogg'
```
你应该使用“双引号”对于任何包含扩展（例如`$variable`或`$(command)`扩展）的参数并且'单引号'对于任何其他论点。单引号确保引号中的所有内容都保持原义，而双引号仍然允许一些 `bash` 语法，例如扩展：
```bash
echo "Good morning, $USER."                              # Double quotes allow bash to expand $USER
echo 'You have won SECOND PRIZE in a beauty contest.'    # \Single quotes prevent even the $-syntax
     'Collect $10'                                       # from triggering expansion.
```
您会发现在引用方面培养实用主义意识是一个很好的做法：只要看一眼 `bash` 代码块，未加引号的参数就会立即跳出来，并且您应该感到有一种冲动，需要先解决这些问题，然后才能允许自己继续做其他事情。引用问题是 `bash` 问题中至少十分之九的核心，也是人们寻求帮助的问题的绝大多数原因。由于引用实际上非常容易，因此有纪律的引用者无需担心太多。

{% note success %}
引用的黄金法则非常简单：
如果参数中有空格或符号，则必须引用它。
如果没有，引号通常是可选的，但为了安全起见，您仍然可以引用它。

参数不需要引用的情况极为罕见，主要是在测试内部和扩展周围。不要从你的论点中删除或省略引号，以试图使某些东西在任何其他情况下发挥作用；相反，您更有可能引入一个可怕且难以检测的错误。[[${..+..}
{% endnote %}

{% note warning%}
缺少引号的危险有很多，但作为一个非常简单的示例，请考虑当您不小心在输入前面放置空格时会发生什么：
```bash
$ read -p 'Which user would you like to remove from your system? ' username
Which user would you like to remove from your system?  lhunath
$ rm -vr /home/$username
removed '/home/lhunath/somefile'
removed directory: '/home/lhunath'
removed '/home/bob/bobsfiles'
removed directory: '/home/bob'
removed '/home/victor/victorsfiles'
removed directory: '/home/victor'
removed directory: '/home'
rm: cannot remove 'lhunath': No such file or directory
```
{% endnote %}

这里发生的情况是，在输入时，因为您不小心`space`在要删除的用户名前添加了一个字符，该`rm`命令扩展为，这导致了一种可能让 `Victor` 和 `Bob` 都感到不安的情况：该命令现在首先删除整个`rm -vr /home/ lhunath`，包括其中的所有内容，随后它将删除`lhunath` 文件。如果您正确引用了该`rm`命令，则错误的输入将导致错误消息并且不会造成任何损坏：
```bash
$ rm -vr "/home/$username"
rm: cannot remove '/home/ lhunath': No such file or directory
```
为了告诉命令要做什么，我们向它传递参数。在 `bash` 中，参数是标记（`token`），也称为单词（`words`），它们之间用空格分隔。要在参数值中包含空格，您需要引用该参数或转义其中的空格。如果做不到这一点，`bash` 将在其空白处将您的参数分解为多个参数。引用参数还可以防止其中的其他符号被意外解释为 `bash` 代码，例如'`$10 USD`'（变量​​扩展）、"`*** NOTICE ***"（文件名扩展）等。


##### Managing a command's input and output using redirection

```
[ var=value ... ] name [ arg ... ] [ redirection ... ]
```
进程使用文件描述符连接到流。每个进程一般都会有三个标准文件描述符：标准输入（FD 0）、标准输出（FD 1）和标准错误（FD 2）。当 bash 启动一个程序时，它首先为该程序设置一组文件描述符。它通过查看自己的文件描述符并为新进程设置一组相同的描述符来实现这一点：我们说新进程“继承”" bash 的文件描述符。当您打开终端到新的 bash shell 时，终端将通过将 bash 的输入和输出连接到终端来设置 bash。这就是键盘中的字符最终出现在 bash 中以及 bash 消息的方式在你的终端窗口中。每次 bash 启动一个自己的程序时，它都会为该程序提供一组与其自身匹配的文件描述符。这样，bash 命令的消息也会出现在你的终端上，并且你的键盘输入也会显示在终端上（命令的输出和输入连接到您的终端）：
```              ╭──────────╮
    Keyboard ╾──╼┥0  bash  1┝╾─┬─╼ Display
                 │         2┝╾─┘
                 ╰──────────╯

$ ls -l a b                                     # Imagine we have a file called "a", but not a file called "b".
ls: b: No such file or directory                # Error messages are emitted on FD 2
-rw-r--r--  1 lhunath  staff  0 30 Apr 14:43 a  # Results are emitted on FD 1

                 ╭──────────╮
    Keyboard ╾┬─╼┥0  bash  1┝╾─┬─╼ Display
              │  │         2┝╾─┤ 
              │  ╰─────┬────╯  │
              │        ╎       │
              │  ╭─────┴────╮  │
              └─╼┥0  ls    1┝╾─┤
                 │         2┝╾─┘
                 ╰──────────╯
```
当bash启动一个`ls`进程时，它首先查看自己的文件描述符。然后，它为进程创建文件描述符ls，连接到与其自己相同的流：FD 1 和 FD 2 通向`Display`，FD 0 来自Keyboard. 因此，ls错误消息（在 FD 2 上发出）及其常规输出（在 FD 1 上发出）最终都会出现在终端显示屏上。如果我们想控制命令连接的位置，我们需要使用重定向：这是更改文件描述符的源或目标的做法。我们可以使用重定向做的一件事是将ls结果写入文件而不是终端显示：
```
                 ╭──────────╮
    Keyboard ╾──╼┥0  bash  1┝╾─┬─╼ Display
                 │         2┝╾─┘
                 ╰──────────╯

$ ls -l a b >myfiles.ls                               # We redirect FD 1 to the file "myfiles.ls"
ls: b: No such file or directory                      # Error messages are emitted on FD 2

                 ╭──────────╮
    Keyboard ╾┬─╼┥0  bash  1┝╾─┬─╼ Display
              │  │         2┝╾─┤
              │  ╰─────┬────╯  │
              │        ╎       │
              │  ╭─────┴────╮  │
              └─╼┥0  ls    1┝╾─╌─╼ myfiles.ls
                 │         2┝╾─┘
                 ╰──────────╯

$ cat myfiles.ls                                      # The cat command shows us the contents of a file
-rw-r--r--  1 lhunath  staff  0 30 Apr 14:43 a        # The result is now in myfiles.ls
```
您刚刚通过将命令的标准输出重定向到文件来执行文件重定向。重定向标准输出是使用`>`运算符完成的。将其想象为将命令输出发送到文件的箭头。这是迄今为止最常见和最有用的重定向形式。重定向的另一个常见用途是隐藏错误消息。您会注意到我们的重定向ls命令仍然显示错误消息。通常这是一件好事。但有时，我们可能会发现脚本中某些命令产生的错误消息对用户来说并不重要，应该隐藏。为此，我们可以再次使用文件重定向，其方式与重定向标准输出导致ls' 结果消失类似：
```
                 ╭──────────╮
    Keyboard ╾──╼┥0  bash  1┝╾─┬─╼ Display
                 │         2┝╾─┘
                 ╰──────────╯

$ ls -l a b >myfiles.ls 2>/dev/null                # We redirect FD 1 to the file "myfiles.ls"
and FD 2 to the file "/dev/null"

                 ╭──────────╮
    Keyboard ╾┬─╼┥0  bash  1┝╾─┬─╼ Display
              │  │         2┝╾─┘
              │  ╰─────┬────╯
              │        ╎
              │  ╭─────┴────╮
              └─╼┥0  ls    1┝╾───╼ myfiles.ls
                 │         2┝╾───╼ /dev/null
                 ╰──────────╯

$ cat myfiles.ls                                   # The cat command shows us the contents of a file
-rw-r--r--  1 lhunath  staff  0 30 Apr 14:43 a     # The result is now in myfiles.ls
$ cat /dev/null                                    # The /dev/null file is empty?
```
`>`请注意如何通过在操作员前面加上 FD 号码前缀来重定向任何 FD 。我们过去`2>`将 FD 2 重定向到 ，`/dev/null`同时>仍然将 FD 1 重定向到`myfiles.ls`。如果省略该数字，输出重定向默认为重定向 FD 1（标准输出）。我们的ls命令不再显示错误消息，并且结果已正确存储在`myfiles.ls`. 错误信息去哪儿了？我们已将其写入文件`/dev/null`。但是当我们显示该文件的内容时，我们看不到错误消息。出了什么问题吗？这个线索就在目录名称中。该文件`null`位于目录： 这是设备文件`/dev`的特殊目录。设备文件是代表我们系统中的设备的特殊文件。当我们向它们写入或读取时，我们是通过内核直接与这些设备通信。该设备是一个始终为空的特殊设备。您写入其中的任何内容都将丢失，并且无法从中读取任何内容。这使得它成为丢弃信息的非常有用的设备。我们将不需要的错误消息传输到设备，然后它就会消失。

如果我们想将终端上通常出现的所有输出保存到我​​们的`myfiles.ls`文件中该怎么办？结果和错误消息？直觉可能会建议：
```bash
$ ls -l a b >myfiles.ls 2>myfiles.ls                  # Redirect both file descriptors to myfiles.ls?

                 ╭──────────╮
    Keyboard ╾┬─╼┥0  bash  1┝╾─┬─╼ Display
              │  │         2┝╾─┘
              │  ╰─────┬────╯
              │        ╎
              │  ╭─────┴────╮
              └─╼┥0  ls    1┝╾───╼ myfiles.ls
                 │         2┝╾───╼ myfiles.ls
                 ╰──────────╯

$ cat myfiles.ls                                      # Contents may be garbled depending on how streams were flushed
-rw-r--r--  1 lhunath  stls: b: No such file or directoryaff  0 30 Apr 14:43 a
```
但你错了！为什么这是不正确的？经过检查，`myfiles.ls`似乎一切顺利，但实际上这里发生了非常危险的事情。如果幸运的话，您会发现文件的输出并不完全符合您的预期：它可能有点乱码、无序，甚至可能是正确的。问题是，您无法预测也无法保证此命令的结果。这里发生了什么？问题是两个文件描述符现在都有自己的文件流。这是有问题的，因为流的内部工作方式，这个主题超出了本指南的范围，但足以说明，当两个流合并到文件中时，结果是流的任意混合在一起。

要解决此问题，您需要在同一流上发送输出和错误字节。为此，您需要知道如何复制文件描述符：

```bash
$ ls -l a b >myfiles.ls 2>&1                    # Make FD 2 write to where FD 1 is writing

                 ╭──────────╮
    Keyboard ╾┬─╼┥0  bash  1┝╾─┬─╼ Display
              │  │         2┝╾─┘
              │  ╰─────┬────╯
              │        ╎
              │  ╭─────┴────╮
              └─╼┥0  ls    1┝╾─┬─╼ myfiles.ls
                 │         2┝╾─┘
                 ╰──────────╯

$ cat myfiles.ls
ls: b: No such file or directory
-rw-r--r--  1 lhunath  staff  0 30 Apr 14:43 a
```
复制文件描述符，也称为“复制”文件描述符，是将一个文件描述符的流连接复制到另一个文件描述符的行为。结果，两个文件描述符都连接到同一个流。我们使用该>&运算符，在其前面加上我们要更改的文件描述符，并在其后面加上我们需要“复制”其流的文件描述符。您将相当频繁地使用此运算符，并且在大多数情况下，它将像上面那样将 FD 1 复制到 FD 2。您可以将语法`2>&1` 转换为 `Make FD 2 write(>) to where FD(&) 1`。

我们现在已经看到了相当多的重定向操作，我们甚至将它们组合起来。在你疯狂之前，你需要了解一个更重要的规则：重定向是从左到右评估的，方便地与我们阅读它们的方式相同。这似乎是显而易见的，但忽视这一点导致许多前辈犯了这个错误：
```bash
$ ls -l a b 2>&1 >myfiles.ls      # Make FD 2 go to FD 1 and FD 1 go to myfiles.ls?
```
编写此代码的人可能会认为，由于 FD 2 的输出将发送到 FD 1，而 FD 1 的输出将发送到 FD `myfiles.ls`，因此错误应该最终出现在文件中。他们的推理中的逻辑错误是假设将`2>&1`FD 2 的输出发送到 FD 1。但事实并非如此。它将 FD 2 的输出发送到FD 1 连接到的流，此时该流可能是终端而不是文件，因为 FD 1 尚未重定向。上述命令的结果可能会令人沮丧，因为它看起来好像标准错误的重定向没有生效，而实际上，您只是将标准错误重定向到终端（标准输出的目标），这就是它的位置之前就已经指过了。还有很多其他重定向运算符，但它们并不都像您刚刚学到的那样有用。事实证明，人们学会用简单的英语阅读命令重定向是有用的。我现在将列举 bash 的重定向运算符。

- **文件重定向**
```
[x]>file, [x]<file

echo Hello >~/world
rm file 2>/dev/null
read line <file
```
`Make FD x write to / read from file`. 打开文件流以进行写入或读取，并连接到文件描述符x。当省略`x`时，写入时默认为FD 1（标准输出），读取时默认为FD 0（标准输入）。

- **文件描述符复制**
```
[ x ] >& y , [ x ] <& y

ping 127.0.0.1 >results 2>&1
exec 3>&1 >mylog; echo moo; exec 1>&3 3>&-
```
使 FD x写入/读取 FD y的流。FD y使用的流连接被复制到 FD x。第二个例子相当高级：要理解它，您需要知道`exec`可以用来更改 `bash` 本身的文件描述符（而不是新命令的文件描述符），并且如果您使用尚不存在的x ，`bash` 将使用该编号为您创建一个新的文件描述符（“插头”）。

- **附加文件重定向**
```
[ x ] >>文件

echo Hello >~/world
echo World >>~/world
```
将 FD x追加到文件末尾。打开文件流以追加模式写入，并连接到文件描述符x。常规文件重定向运算符`>`在打开文件时会清空文件的内容，以便文件中只有您的字节。在追加模式 ( `>>`) 中，文件的现有内容将保留，流的字节将添加到文件的末尾。

- **重定向标准输出和标准错误**
```
&>file

ping 127.0.0.1 &>results
```
将 FD 1（标准输出）和 FD 2（标准错误）都写入文件。这是一个便利运算符，它的作用与此相同，但更简洁。同样，您可以通过双击箭头来追加而不是截断：`>file 2>&1&>>file`

- **Here Documents**
```
   <<[-]delimiter
        here-document
   delimiter

   cat <<. 
   Hello world.
   Since I started learning bash, you suddenly seem so much bigger than you were before.
.
   .
```
使 FD 0（标准输入）从分隔符 之间的字符串中读取。`Here` 文档是将大块文本提供给命令输入的好方法。它们从定界符之后的行开始，并在 `bash` 遇到仅包含定界符的行时结束。重要的是要记住，您的终​​止分隔符不能缩进，因为这样它就不再只是该行上的分隔符。您可以在初始分隔符声明前加上 前缀-，这将告诉 `bash` 忽略您放在定界符前面的任何制表符。这样，您可以缩进定界文档，而不会在输入字符串中显示缩进。它还允许您使用制表符缩进终止分隔符。最后，可以将变量扩展放入此处文档的字符串中。这允许您将变量数据注入此处文档中。稍后我们将了解有关变量和扩展的更多信息，但只要说如果不需要扩展，您就需要在 的初始声明周围加上引号。'`delimiter`'。

- **Here Strings**
```
<<<string

cat <<<"Hello world.
Since I started learning bash, you suddenly seem so much bigger than you were before."
```
使 FD 0（标准输入）从字符串 中读取。

- **Closing file descriptors**
```
x>&-, x<&-

exec 3>&1 >mylog; echo moo; exec 1>&3 3>&-
```
关闭 FD x。流与文件描述符`x`断开，并且文件描述符从进程中删除。在重新创建之前它不能再次使用。当省略`x>&-`时，默认关闭标准输出并`<&-`默认关闭标准输入。您很少会使用该运算符。

- **Moving file descriptors**
```
[x]>&y-, [x]<&y-

exec 3>&1- >mylog; echo moo; exec >&3-
```
将 FD `x`替换为 FD `y`。`y`处的文件描述符被复制到`x`并关闭`y` 。实际上，它将`x`替换为`y`。它是一个方便的操作符。同样，您很少会使用此运算符。`[x]>&y y>&-`

- **Reading and writing with a file descriptor**
```
[x]<>file

exec 5<>/dev/tcp/ifconfig.me/80
echo "GET /ip HTTP/1.1
Host: ifconfig.me
" >&5
cat <&5
```
打开 FD `x`以读取和写入文件。`x`处的文件描述符通过文件流打开，可用于写入和读取字节。通常您将为此使用两个文件描述符。这是有用的极少数情况之一是使用读/写设备（例如网络套接字）设置流时。上面的示例将几行 `HTTP` 写入`ifconfig.me`端口（标准 `HTTP` 端口）的主机，然后读取从网络返回的字节，两者都使用为此设置的`80`相同文件描述符。

作为关于重定向的最后一点，我想指出，对于简单命令，重定向运算符可以出现在简单命令中的任何位置。也就是说，它们不需要出现在末尾。虽然将它们保留在命令末尾是一个好主意，如果主要是为了一致性并避免在长命令中出现意外或错过操作符，但在某些情况下，有些人习惯将重定向操作符放在其他地方。特别是，为了可读性，经常将重定向运算符放在`echo`或者命令名称后面，特别是当它们有一个序列时：`printf`
```
echo >&2 "Usage: exists name"
echo >&2 "   Check to see if the program 'name' is installed."
echo >&2
echo >&2 "RETURN"
echo >&2 "   Success if the program exists in the user's PATH and is executable.  Failure otherwise."
```
默认情况下，新命令继承 `shell` 的当前文件描述符。我们可以使用重定向来更改命令输入的来源及其输出的位置。文件重定向（例如`2>errors.log`）允许我们将文件描述符流式传输到文件。我们可以复制文件描述符（例如`2>&1`）以使它们共享一个流。还有许多其他更高级的重定向运算符。

1. 仅将最后一个命令的标准错误消息发送到名为errors.log. errors.log然后在终端上显示 的内容。
```bash
$ ls /bin/bash /bob/bash 2>errors.log
/bin/bash* $ 
ls: /bob/bash: 没有这样的文件或目录
cat errors.log
```

2. 将最后一个命令的标准输出和错误消息附加到名为errors.log. errors.log然后再次在终端上显示 的内容。
```bash
$ ls /bin/bash /bob/bash >>errors.log 2>&1
$cat errors.log 
ls: /bob/bash: 没有这样的文件或目录
ls: /bob/bash: 没有这样的文件或目录
/bin/bash*
```
3. Hello world.使用`here-string`在终端上显示该字符串。
```bash
$cat <<< 'Hello world.'
Hello world.
```
4. 修复此命令，以便将消息正确保存到文件中`log`，并随后正确关闭 FD `3：exec 3>&2 2>log; echo 'Hello!'; exec 2>&3`
```bash
$exec 3>&1 >log; echo 'Hello!'; exec 1>&3 3>&-
```