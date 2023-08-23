---
title: Git 解决分支冲突
date: 2023-07-27 12:23:01
tag: 
   - git
category:
   - Git
---
昨天我在处理项目时我尝试使用 `git pull` 命令从远程分支中拉取更改的内容，但最终报错：`"fatal: Need to specify how to reconcile divergent branches"`, 这阻碍了我的工作。我决定写一篇关于这个问题的文章，以便它也能帮助你们。

#### 解决了“reconcile divergent branches”

如果您使用 `Git`，那么在尝试从远程存储库执行 git pull 时出现此错误并不罕见。虽然大多数时候您会看到`"fatal: Need to specify how to reconcile divergent branches"`警告，但有时您会看到它是致命错误。如果错误显示为警告，那么您仍然可以从存储库中提取更改，但如果它显示为致命错误，那么您将无法继续进行。

```
$ git pull origin main
*branch dev -> FETCH_HEAD
提示：您有不同的分支，需要指定如何协调它们。
提示：您可以通过在提示之前运行以下命令之一来完成此操作
：您的下一个拉取：
提示：
提示：git config pull.rebase false＃合并
提示：git config pull.rebase true＃rebase
提示：git config pull.ff only # 仅快进
提示：
提示：您可以将“git config”替换为“git config --global”以设置默认
提示：所有存储库的首选项。您还可以传递 --rebase, --no-rebase,
在命令行上提示：或 --ff-only 以覆盖每个
提示：调用配置的默认值。fatal：需要指定如何协调不同的分支。
```
<!-- more -->

要解决这个错误，您可以有以下两种解决方案并应用适合您的解决方案。但在此之前，请使用`git --version`命令检查当前的 git 版本。发现以下解决方案运行良好`Git 2.27.0`。
```
$ git --version 
git 版本 2.35.1
```

**解决方案1：切换到合并策略**

当存在不在本地分支上的远程更改时，需要解决它们。默认的 Git 行为是合并，这将在本地分支上创建一个新的提交来解决这些更改。您可以使用`git config pull.rebase false`命令切换回默认合并策略，而不是使用变基策略。

```
$ git config pull.rebase false
```
切换回默认策略后，您可以再次尝试运行`git pull origin main`命令以从分支中提取所有更改main。这次您可以看到所有更改现在都已由`'ort'`策略合并。查看此处以了解有关 `ort` 策略的更多信息。

```
$ git pull origin main

*branch dev -> FETCH_HEAD
由“ort”策略进行合并。
application.yaml | 13 ++++++++++++ 
config.yaml | 160 ++++++++++++++++++++++++++++++++++++++++++++++ 
+++++++ +++++++++++++++++++++++++++++++++++++ 
2 个文件已更改，173 个插入(+)
创建模式 100644 application.yaml
创建模式100644 config.yaml
```

**解决方案2：切换到快进策略**

有时您的默认策略只是 `FF`。因此，要切换回此策略，您需要运行`git config --global pull.ff only`如下所示的命令

```
$ git config --global pull.ff only
```
这会将以下行添加到`$HOME/.gitconfig`：
```
[pull] 
      ff = only
```
但这里还需要注意的是，只有在不创建新提交的情况下可以“快进”时，Git 才会更新您的分支。如果这无法完成，意味着本地和远程存在分歧，则`git config --global pull.ff only`只需中止并显示错误消息。

