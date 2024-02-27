---
title: Markdown常用LaTex数学公式
date: 2024-02-27 14:20:32
tags:
  - AI
categories:
  - 人工智能
---

众所周知，数据挖掘、机器学习以及深度学习等，在学习与运用过程中，会涉及到大量的数学公式，而公式的编辑往往比较繁琐。`LaTeX`公式有两种，一种是用在正文中的，一种是单独显示的。正文中的公式如下:
```
$...$
```
单独一行显示的时候使用如下命令：
```
$$...$$
```
其中，`$`符号中间包含的三个点表格的是`LaTex`的公式命令。
<!-- more -->
#### 样式系列

##### 换行与空格

换行：`\\` ；空格：`\:`

##### 居中

用两个美元符号`$$`包围的公式，会独占一行显示：
$$y = x\beta + \varepsilon$$

##### 加粗和斜体

$$\mathit{公式}: \mathbf{y} = \mathbf{x}\beta + u$$
{% asset_img math_13.png %}

##### 大小

$\tiny Markdown常用LaTex数学公式\\$
$\scriptsize Markdown常用LaTex数学公式\\$
$\small Markdown常用LaTex数学公式\\$
$\normalsize Markdown常用LaTex数学公式\\$
$\large Markdown常用LaTex数学公式\\$
$\Large Markdown常用LaTex数学公式\\$
$\huge Markdown常用LaTex数学公式\\$
$\Huge Markdown常用LaTex数学公式\\$

{% asset_img math_12.png %}

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

{% asset_img math_6.png %}

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

{% asset_img math_7.png %}

#### 数学符号

##### 集合符号表

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

{% asset_img math_1.png %}

##### 常用符号表

|运算符|公式|运算符|公式|运算符|公式|
|:--|:--|:--|:--|:--|:--|
|$\cdot$|`$\cdot$`|$\vdots$|`$\vdots$`|$\grave{x}$|`$\grave{x}$`|
|$.$|`$.$`|$\ddots$|`$\ddots$`|$\breve{x}$|`$\breve{x}$`|
|$*$|`$*$`|$,$|`$,$`|$\dot{x}$|`$\dot{x}$`|
|$+$|`$+$`|$!$|`$!$`|$\widehat{xxx}$|`$\widehat{xxx}$`|
|$-$|`$-$`|$;$|`$;$`|$\ddot{x}$|`$\ddot{x}$`|
|$\times$|`$\times$`|$?$|`$?$`|$\check{x}$|`$\check{x}$`|
|$\div$|`$\div$`|$\colon$|`$\colon$`|$\ddot{x}$|`$\ddot{x}$`|
|$=$|`$=$`|$\acute{x}$|`$\acute{x}$`|$\tilde{x}$|`$\tilde{x}$`|
|$\neq$|`$\neq$`|$\bar{x}$|`$\bar{x}$`|$\hat{x}$|`$\hat{x}$`|
|$\dotsm$|`$\dotsm$`|$\vec{x}$|`$\vec{x}$`|||
|$\dotso$|`$\dotso$`|$\widetilde{xxx}$|`$\widetilde{xxx}$`|$\backslash$|`$\backslash$`|
|$/$|`$/$`|||$]$|`$]$`|
|$\smallsetminus$|`$\smallsetminus$`|$\lVert$|`$\lVert$`|$\lbrace$|`$\lbrace$`|
|||$\rVert$|`$\rVert$`|$\rbrace$|`$\rbrace$`|
|$\lvert$|`$\lvert$`|$\lgroup$|`$\lgroup$`|$\langle$|`$\langle$`|
|$\rvert$|`$\rvert$`|$\rgroup$|`$\rgroup$`|$\rangle$|`$\rangle$`|
|$\rmoustache$|`$\rmoustache$`|$[$|`$[$`|$\lmoustache$|`$\lmoustache$`|
|$\lfloor$|`$\lfloor$`|$\lceil$|`$\lceil$`|$\rceil$|`$\rceil$`|
|||$\rfloor$|`$\rfloor$`|||

{% asset_img math_2.png %}

##### 希腊字母表

|字母|公式|字母|公式|字母|公式|
|:--|:--|:--|:--|:--|:--|
|$\alpha$|`$\alpha$`|$\beta$|`$\beta$`|$\chi$|`$\chi$`|
|$\delta$|`$\delta$`|$\Delta$|`$\Delta$`|$\epsilon$|`$\epsilon$`|
|$\eta$|`$\eta$`|$\Gamma$|`$\Gamma$`|$\iota$|`$\iota$`|
|$\kappa$|`$\kappa$`|$\lambda$|`$\lambda$`|$\Lambda$|`$\Lambda$`|
|$\mu$|`$\mu$`|$\nabla$|`$\nabla$`|$\nu$|`$\nu$`|
|$\omega$|`$\omega$`|$\Omega$|`$\Omega$`|$\phi$|`$\phi$`|
|$\Phi$|`$\Phi$`|$\pi$|`$\pi$`|$\Pi$|`$\Pi$`|
|$\psi$|`$\psi$`|$\Psi$|`$\Psi$`|$\rho$|`$\rho$`|
|$\sigma$|`$\sigma$`|$\Sigma$|`$\Sigma$`|$\tau$|`$\tau$`|
|$\theta$|`$\theta$`|$\Theta$|`$\Theta$`|$\upsilon$|`$\upsilon$`|
|$\varepsilon$|`$\varepsilon$`|$\varsigma$|`$\varsigma$`|$\vartheta$|`$\vartheta$`|
|$\xi$|`$\xi$`|$\zeta$|`$\zeta$`|||

{% asset_img math_3.png %}

##### 函数公式表

|函数|公式|函数|公式|函数|公式|
|:--|:--|:--|:--|:--|:--|
|$\sin$|`$\sin$`|$\sin^{-1}$|`$\sin^{-1}$`|$\inf$|`$\inf$`|
|$\cos$|`$\cos$`|$\cos^{-1}$|`$\cos^{-1}$`|$\arg$|`$\arg$`|
|$\tan$|`$\tan$`|$\tan^{-1}$|`$\tan^{-1}$`|$\det$|`$\det$`|
|$\sinh$|`$\sinh$`|$\sinh^{-1}$|`$\sinh^{-1}$`|$\dim$|`$\dim$`|
|$\cosh$|`$\cosh$`|$\cosh^{-1}$|`$\cosh^{-1}$`|$\gcd$|`$\gcd$`|
|$\tanh$|`$\tanh$`|$\tanh^{-1}$|`$\tanh^{-1}$`|$\hom$|`$\hom$`|
|$\csc$|`$\csc$`|$\exp$|`$\exp$`|$\ker$|`$\ker$`|
|$\sec$|`$\sec$`|$\lg$|`$\lg$`|$\Pr$|`$\Pr$`|
|$\cot$|`$\cot$`|$\ln$|`$\ln$`|$\sup$|`$\sup$`|
|$\coth$|`$\coth$`|$\log$|`$\log$`|$\deg$|`$\deg$`|
|$\hom$|`$\hom$`|$\log_{e}$|`$\log_{e}$`|$\injlim$|`$\injlim$`|
|$\arcsin$|`$\arcsin$`|$\log_{10}$|`$\log_{10}$`|||
|$\arccos$|`$\arccos$`|$\lim$|`$\lim$`|$\varinjlim$|`$\varinjlim$`|
|$\det$|`$\det$`|$\liminf$|`$\liminf$`|$\varprojlim$|`$\varprojlim$`|
|$\arctan$|`$\arctan$`|$\limsup$|`$\limsup$`|$\varliminf$|`$\varliminf$`|
|$\textrm{arccsc}$|`$\textrm{arccsc}$`|$\max$|`$\max$`|$\projlim$|`$\projlim$`|
|$\textrm{arcsec}$|`$\textrm{arcsec}$`|$\min$|`$\min$`|$\varlimsup$|`$\varlimsup$`|
|$\textrm{arccot}$|`$\textrm{arccot}$`|$\infty$|`$\infty$`|||

{% asset_img math_4.png %}

##### 特殊符号-箭头系列

|箭头|公式|箭头|公式|箭头|公式|
|:--|:--|:--|:--|:--|:--|
|$\uparrow$|`$\uparrow$`|$\longleftarrow$|`$\longleftarrow$`|$\downdownarrows$|`$\downdownarrows$`|
|$\downarrow$|`$\downarrow$`|$\longrightarrow$|`$\longrightarrow$`|$\upuparrows$|`$\upuparrows$`|
|$\updownarrow$|`$\updownarrow$`|$\rightarrow$|`$\rightarrow$`|$\rightharpoondown$|`$\rightharpoondown$`|
|$\Uparrow$|`$\Uparrow$`|$\leftarrow$|`$\leftarrow$`|$\downharpoonleft$|`$\downharpoonleft$`|
|$\Downarrow$|`$\Downarrow$`|$\mapsto$|`$\mapsto$`|$\rightharpoonup$|`$\rightharpoonup$`|
|$\Leftarrow$|`$\Leftarrow$`|$\nrightarrow$|`$\nrightarrow$`|$\downharpoonright$|`$\downharpoonright$`|
|$\Rightarrow$|`$\Rightarrow$`|$\nleftarrow$|`$\nleftarrow$`|$\upharpoonleft$|`$\upharpoonleft$`|
|$\Leftrightarrow$|`$\Leftrightarrow$`|$\rightrightarrows$|`$\rightrightarrows$`|$\upharpoonright$|`$\upharpoonright$`|
|$\nLeftrightarrow$|`$\nLeftrightarrow$`|$\leftleftarrows$|`$\leftleftarrows$`|$\leftharpoondown$|`$\leftharpoondown$`|
|$\nLeftarrow$|`$\nLeftarrow$`|$\rightleftarrows$|`$\rightleftarrows$`|$\leftharpoonup$|`$\leftharpoonup$`|
|$\nRightarrow$|`$\nRightarrow$`|$\leftrightarrows$|`$\leftrightarrows$`|$\hookleftarrow$|`$\hookleftarrow$`|
|$\Updownarrow$|`$\Updownarrow$`|$\curvearrowleft$|`$\curvearrowleft$`|$\hookrightarrow$|`$\hookrightarrow$`|
|$\circlearrowleft$|`$\circlearrowleft$`|$\curvearrowright$|`$\curvearrowright$`|$\rightleftharpoons$|`$\rightleftharpoons$`|
|$\circlearrowright$|`$\circlearrowright$`|$\Longleftarrow$|`$\Longleftarrow$`|$\leftrightharpoons$|`$\leftrightharpoons$`|
|$\Lleftarrow$|`$\Lleftarrow$`|$\Longrightarrow$|`$\Longrightarrow$`|$\looparrowleft$|`$\looparrowleft$`|
|$\Rrightarrow$|`$\Rrightarrow$`|$\longleftrightarrow$|`$\longleftrightarrow$`|$\looparrowright$|`$\looparrowright$`|
|$\nwarrow$|`$\nwarrow$`|$\Longleftrightarrow$|`$\Longleftrightarrow$`|$\rightsquigarrow$|`$\rightsquigarrow$`|
|$\swarrow$|`$\swarrow$`|$\longmapsto$|`$\longmapsto$`|$\Lsh$|`$\Lsh$`|
|$\searrow$|`$\searrow$`|$\rightarrowtail$|`$\rightarrowtail$`|$\Rsh$|`$\Rsh$`|
|$\nearrow$|`$\nearrow$`|$\leftarrowtail$|`$\leftarrowtail$`|$\multimap$|`$\multimap$`|
|$\twoheadleftarrow$|`$\twoheadleftarrow$`|$\twoheadrightarrow$|`$\twoheadrightarrow$`|$\leftrightsquigarrow$|`$\leftrightsquigarrow$`|
|$\leftrightarrow$|`$\leftrightarrow$`|$\nleftrightarrow$|`$\nleftrightarrow$`|||

{% asset_img math_5.png %}

#### 扩展更新

##### 二次方程求解

$$y = {ax^2+bx+c}$$

$$x={\frac{-b \pm \sqrt{b^2-4ac}}{2a}}$$ or $$x = {-b \pm \sqrt{b^2-4ac} \over 2a}$$

{% asset_img math_8.png %}

##### 矩阵系列
$$
\begin{bmatrix}
1&0&0 \\
0&1&0 \\
0&0&1
\end{bmatrix}
$$

{% asset_img math_9.png %}

##### 方程组

$$
\begin{equation}
    \left\{
    \begin{aligned}     %请使用'aligned'或'align*'
    2x + y  &= 1  \\     %加'&'指定对齐位置
    2x + 2y &= 2
    \end{aligned}
    \right.
    \end{equation}
$$

{% asset_img math_10.png %}

{% note danger %}
**注**：如果各个方程需要在某个字符处对齐（如等号对齐），只需在所有要对齐的字符前加上&符号。如果不需要公式编号，只需在宏包名称后加上`*`号。
{% endnote %}

分情况讨论方程式:

$
f(x) =
\begin{cases}
x^2 \qquad & a \gt 0   \\
e^x \qquad & a \le 0
\end{cases}
$

{% asset_img math_11.png %}

##### 编号

插入编号，使用`\tag`指令指定公式的具体编号，并使用`\labe`l指令埋下锚点。如`y=x^2 \tag{1.5a} \label{eq:test}`
引用编号，使用`\eqref`指令引用前面埋下的锚点，`\eqref{eq:test}`将显示为：