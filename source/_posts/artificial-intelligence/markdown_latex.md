---
title: Markdown常用LaTex数学公式
date: 2024-02-27 14:20:32
tags:
  - AI
categories:
  - 人工智能
---

#### 样式系列

##### 换行与空格

换行：`\\` ；空格：`\:`

##### 居中

用两个美元符号`$$`包围的公式，会独占一行显示：
$$y = x\beta + \varepsilon$$

##### 加粗和斜体

$$\mathit{公式}: \mathbf{y} = \mathbf{x}\beta + u$$

##### 大小

$\tiny Markdown常用LaTex数学公式\\$
$\scriptsize Markdown常用LaTex数学公式\\$
$\small Markdown常用LaTex数学公式\\$
$\normalsize Markdown常用LaTex数学公式\\$
$\large Markdown常用LaTex数学公式\\$
$\Large Markdown常用LaTex数学公式\\$
$\huge Markdown常用LaTex数学公式\\$
$\Huge Markdown常用LaTex数学公式\\$

##### 颜色

$\color{Red}{红色}$,    $\color{blue}{蓝色}$
$\color{orange}{橙色}$, $\color{Green}{绿色}$
$\color{gray}{灰色}$,   $\color{purple}{紫色}$

#### 数学公式

##### 常用数学公式
|数学公式|LaTex表达式|
|:--|:--|
|$\sqrt{ab}$|`$\sqrt{ab}$`|
|$\sqrt[n]{ab}$|`$\sqrt[n]{ab}$`|
|$\log_{a}{b}$|`$\log_{a}{b}$`|
|$\lg{ab}$|`$\lg{ab}$`|
|$a^{b}$|`$a^{b}$`|
|$a_{b}$|`$a_{b}$`|
|$x_a^b$|`$x_a^b$`|
|$\int$|`$\int$`|
|$\int_{a}^{b}$|`$\int_{a}^{b}$`|
|$\oint$|`$\oint$`|
|$\oint_a^b$|`$\oint_a^b$`|
|$\sum$|`$\sum$`|
|$\sum_a^b$|`$\sum_a^b$`|
|$\coprod$|`$\coprod$`|
|$\coprod_a^b$|`$\coprod_a^b$`|
|$\prod$|`$\prod$`|
|$\prod_a^b$|`$\prod_a^b$`|
|$\bigcap$|`$\bigcap$`|
|$\bigcap_a^b$|`$\bigcap_a^b$`|
|$\bigcup_a^b$|`$\bigcup_a^b$`|
|$\bigsqcup$|`$\bigsqcup$`|
|$\bigsqcup_a^b$|`$\bigsqcup_a^b$`|
|$\bigvee$|`$\bigvee$`|
|$\bigvee_a^b$|`$\bigvee_a^b$`|
|$\bigwedge$|`$\bigwedge$`|
|$\bigwedge_a^b$|`$\bigwedge_a^b$`|
|$\widetilde{ab}$|`$\widetilde{ab}$`|
|$\widehat{ab}$|`$\widehat{ab}$`|
|$\overleftarrow{ab}$|`$\overleftarrow{ab}$`|
|$\overrightarrow{ab}$|`$\overrightarrow{ab}$`|
|$\overbrace{ab}$|`$\overbrace{ab}$`|
|$\underbrace{ab}$|`$\underbrace{ab}$`|
|$\underline{ab}$|`$\underline{ab}$`|
|$\overline{ab}$|`$\overline{ab}$`|
|$\frac{ab}{cd}$|`$\frac{ab}{cd}$`|
|$\frac{\partial a}{\partial b}$|`$\frac{\partial a}{\partial b}$`|
|$\frac{\text{d}x}{\text{d}y}$|`$\frac{\text{d}x}{\text{d}y}$`|
|$\lim_{a \rightarrow b}$|`$\lim_{a \rightarrow b}$`|

##### 高级数学公式

|数学公式|LaTex表达式|
|:--|:--|
|$\displaystyle\sum\limits_{i=0}^n i^3$|`$\displaystyle\sum\limits_{i=0}^n i^3$`|
|$\left(\begin{array}{c}a\\ b\end{array}\right)$|`$\left(\begin{array}{c}a\\ b\end{array}\right)$`|
|$\left(\frac{a^2}{b^3}\right)$|`$\left(\frac{a^2}{b^3}\right)$`|
|$\left.\frac{a^3}{3}\right\lvert_0^1$|`$\left.\frac{a^3}{3}\right\lvert_0^1$`|
|$\begin{bmatrix}a & b \\c & d \end{bmatrix}$|`$\begin{bmatrix}a & b \\c & d \end{bmatrix}$`|
|$\begin{cases}a & x = 0\\b & x > 0\end{cases}$|`$\begin{cases}a & x = 0\\b & x > 0\end{cases}$`|
|$\sqrt{\frac{n}{n-1} S}$|$\sqrt{\frac{n}{n-1} S}$|
|$\begin{pmatrix} \alpha& \beta^{*}\\ \gamma^{*}& \delta \end{pmatrix}$|`$\begin{pmatrix} \alpha& \beta^{*}\\ \gamma^{*}& \delta \end{pmatrix}$`|
|$A\:\xleftarrow{n+\mu-1}\:B$|`$A\:\xleftarrow{n+\mu-1}\:B$`|
|$B\:\xrightarrow[T]{n\pm i-1}\:C$|`$B\:\xrightarrow[T]{n\pm i-1}\:C$`|
|$\frac{1}{k}\log_2 c(f)\;$|`$\frac{1}{k}\log_2 c(f)\;$`|
|$\iint\limits_A f(x,y)\;$|`$\iint\limits_A f(x,y)\;$`|
|$x^n + y^n = z^n$|`$x^n + y^n = z^n$`|
|$E=mc^2$|`$E=mc^2$`|
|$e^{\pi i} - 1 = 0$|`$e^{\pi i} - 1 = 0$`|
|$p(x) = 3x^6$|`$p(x) = 3x^6$`|
|$3x + y = 12$|`$3x + y = 12$`|
|$\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x$|`$\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x$`|
|$\sqrt[n]{1+x+x^2+\ldots}$|`$\sqrt[n]{1+x+x^2+\ldots}$`|
|$\binom{x}{y} = \frac{x!}{y!(x-y)!}$|`$\binom{x}{y} = \frac{x!}{y!(x-y)!}$`|
|$\frac{\frac{1}{x}+\frac{1}{y}}{y-z}$|`$\frac{\frac{1}{x}+\frac{1}{y}}{y-z}$`|
|$f(x)=\frac{P(x)}{Q(x)}$|`$f(x)=\frac{P(x)}{Q(x)}$`|
|$\frac{1+\frac{a}{b}}{1+\frac{1}{1+\frac{1}{a}}}$|`$\frac{1+\frac{a}{b}}{1+\frac{1}{1+\frac{1}{a}}}$`|
|$\sum_{\substack{0\le i\le m\\ 0\lt j\lt n}} P(i,j)$|`$\sum_{\substack{0\le i\le m\\ 0\lt j\lt n}} P(i,j)$`|
|$\lim_{x \to \infty} \exp(-x) = 0$|`$\lim_{x \to \infty} \exp(-x) = 0$`|
|$\cos (2\theta) = \cos^2 \theta - \sin^2 \theta$|`$\cos (2\theta) = \cos^2 \theta - \sin^2 \theta$`|

#### 数学符号

|运算符|公式|运算符|公式|运算符|公式|
|:--|:--|:--|:--|:--|:--|
|$\emptyset$|`$\emptyset$`|$\in$|`$\in$`|$\notin$|`$\notin$`|
|$\subset$|`$\subset$`|$\supset$|`$\supset$`|$\subseteq$|`$\subseteq$`|
|$\nsubseteq$|`$\nsubseteq$`|$\nsupseteq$|`$\nsupseteq$`|$\nsubseteqq$|`$\nsubseteqq$`|
|$\nsupseteqq$|`$\nsupseteqq$`|$\subsetneq$|`$\subsetneq$`|$\supsetneq$|`$\supsetneq$`|
|$\subsetneqq$|`$\subsetneqq$`|$\supsetneqq$|`$\supsetneqq$`|$\varsubsetneq$|`$\varsubsetneq$`|
|$\varsupsetneq$|`$\varsupsetneq$`|$\varsubsetneqq$|`$\varsubsetneqq$`|$\varsupsetneqq$|`$\varsupsetneqq$`|
|$\bigcap$|`$\bigcap$`|$\bigcup$|`$\bigcup$`|$\bigvee$|`$\bigvee$`|
|$\bigwedge$|`$\bigwedge$`|$\biguplus$|`$\biguplus$`|$\bigsqcup$|`$\bigsqcup$`|
|$\Subset$|`$\Subset$`|$\Supset$|`$\Supset$`|$\subseteqq$|`$\subseteqq$`|
|$\supseteqq$|`$\supseteqq$`|$\sqsubset$|`$\sqsubset$`|$\sqsupset$|`$\sqsupset$`|

##### 集合符号表

##### 常用符号表

##### 希腊字母表

##### 函数公式表

##### 特殊符号-箭头系列

##### 扩展更新
