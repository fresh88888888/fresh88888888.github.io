---
title: 优化算法 (机器学习)(TensorFlow)
date: 2024-05-29 12:24:11
tags:
  - AI
categories:
  - 人工智能
mathjax:
  tex:
    tags: 'ams'
  svg:
    exFactor: 0.03
---

#### RMSProp算法

`RMSProp`算法作为将速率调度与坐标自适应学习率分离的简单修复方法。问题在于，`Adagrad`算法将梯度{% mathjax %}\mathbf{g}_t{% endmathjax %}的平方累加成状态矢量{% mathjax %}\mathbf{s}_t = \mathbf{s}_{t-1} + \mathbf{g}_t^2{% endmathjax %}。因此，由于缺乏规范化，没有约束力，{% mathjax %}\mathbf{s}_t{% endmathjax %}持续增长，几乎是在算法收敛时呈线性递增。解决此问题的一种方法是使用{% mathjax %}\mathbf{s}_t/t{% endmathjax %}。对{% mathjax %}\mathbf{g}_t{% endmathjax %}的合理分布来说，它将收敛。遗憾的是，限制行为生效可能需要很长时间，因为该流程记住了值的完整轨迹。另一种方法是按动量法中的方式使用泄漏平均值，即{% mathjax %}\mathbf{s}_t \leftarrow\gamma \mathbf{s}_{t-1} + (1-\gamma)\mathbf{g}_t^2{% endmathjax %}，其中参数{% mathjax %}\gamma > 0{% endmathjax %}。保持所有其它部分不变就产生了`RMSProp`算法。
<!-- more -->
##### 算法

让我们详细写出这些方程式。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \mathbf{s}_t \leftarrow\gamma \mathbf{s}_{t-1} + (1-\gamma)\mathbf{g}_t^2 \\ 
& \mathbf{x}_t \leftarrow\mathbf{x}_{t-1} - \frac{\eta}{\sqrt{\mathbf{s}_t} + \epsilon} \odot\mathbf{g}_t 
\end{align}
{% endmathjax %}
常数{% mathjax %}\epsilon > 0{% endmathjax %}通常设置为{% mathjax %}10^{-6}{% endmathjax %}，以确保我们不会因除以零或步长过大而受到影响。鉴于这种扩展，我们现在可以自由控制学习率{% mathjax %}\eta{% endmathjax %}，而不考虑基于每个坐标应用的缩放。就泄漏平均值而言，我们可以采用与之前在动量法中适用的相同推理。扩展{% mathjax %}\mathbf{s}_t{% endmathjax %}定义可获得：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{s}_t & = (1-\gamma)\mathbf{g}_t^2 + \gamma\mathbf{g}_t^2 \\ 
& = (1-\gamma)(\mathbf{g}_t^2 + \gamma\mathbf{g}_{t-1}^2 + \gamma^2\mathbf{g}_{t-1} + \ldots,)
\end{align}
{% endmathjax %}
##### 总结 

`RMSProp`算法与`Adagrad`算法非常相似，因为两者都使用梯度的平方来缩放系数。`RMSProp`算法与动量法都使用泄漏平均值。但是，`RMSProp`算法使用该技术来调整按系数顺序的预处理器。在实验中，学习率需要由实验者调度。系数{% mathjax %}\gamma{% endmathjax %}决定了在调整每坐标比例时历史记录的时长。

#### Adadelta算法

`Adadelta`是`AdaGrad`的另一种变体，主要区别在于前者减少了学习率适应坐标的数量。此外，广义上`Adadelta`被称为没有学习率，因为它使用变化量本身作为未来变化的校准。`Adadelta`使用两个状态变量，{% mathjax %}\mathbf{s}_t{% endmathjax %}用于存储梯度二阶导数的泄露平均值，{% mathjax %}\Delta\mathbf{x}_t{% endmathjax %}用于存储模型本身参数变化二阶导数的泄露平均值。以下是Adadelta的技术细节。鉴于参数`du jour`是{% mathjax %}\rho{% endmathjax %}，我们获得了以下泄漏更新：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{s}_t = \rho\mathbf{s}_{t-1} + (1-\rho)\mathbf{g}_t^2
{% endmathjax %}
我们使用重新缩放的梯度{% mathjax %}\mathbf{g'}_t{% endmathjax %}执行更新，即：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{x}_t = \mathbf{x}_{t-1} - \mathbf{g'}_t
{% endmathjax %}
那么，调整后的梯度{% mathjax %}\mathbf{g'}_t{% endmathjax %}是什么？我们可以按如下方式计算它：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{g'}_t = \frac{\sqrt{\Delta\mathbf{x}_{t-1} + \epsilon}}{\sqrt{\mathbf{s}_t + \epsilon}} \odot\mathbf{g}_t
{% endmathjax %}
其中{% mathjax %}\Delta\mathbf{x}_{t-1}{% endmathjax %}是重新缩放梯度的平方{% mathjax %}\mathbf{g'}_t{% endmathjax %}的泄漏平均值。我们将{% mathjax %}\Delta\mathbf{x}_0{% endmathjax %}初始化为{% mathjax %}0{% endmathjax %}，然后在每个步骤中使用{% mathjax %}\mathbf{g'}_t{% endmathjax %}更新它，即：
{% mathjax '{"conversion":{"em":14}}' %}
\Delta\mathbf{x}_t = \rho\Delta\mathbf{x}_{t-1} + (1 - \rho)\mathbf{g'}_t^2
{% endmathjax %}
和{% mathjax %}\epsilon{% endmathjax %}（例如{% mathjax %}10^{-5}{% endmathjax %}这样的小值）是为了保持数字稳定性而加入的。
##### 总结

`Adadelta`没有学习率参数。相反，它使用参数本身的变化率来调整学习率。`Adadelta`需要两个状态变量来存储梯度的二阶导数和参数的变化。`Adadelta`使用泄漏的平均值来保持对适当统计数据的运行估计。

#### Adam算法

我们学习了：随机梯度下降在解决优化问题时比梯度下降更有效；在一个小批量中使用更大的观测值集，可以通过向量化提供额外效率。这是高效的多机、多GPU和整体并行处理的关键；我们添加了一种机制，用于汇总过去梯度的历史以加速收敛。我们通过对每个坐标缩放来实现高效计算的预处理器。我们通过学习率的调整来分离每个坐标的缩放。`Adam`算法将所有这些技术汇总到一个高效的学习算法中。不出预料，作为深度学习中使用的更强大和有效的优化算法之一，它非常受欢迎。但是它并非没有问题，有时`Adam`算法可能由于方差控制不良而发散。
##### 算法

`Adam`算法的关键组成部分之一是：它使用指数加权移动平均值来估算梯度的动量和二次矩，即它使用状态变量。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
& \mathbf{v}_t \leftarrow \beta_1\mathbf{v}_t + (1 - \beta_1)\mathbf{g}_t \\ 
& \mathbf{s}_t \leftarrow \beta_2\mathbf{s}_t + (1 - \beta_2)\mathbf{g}_t^2 \\
\end{align}
{% endmathjax %}
这里{% mathjax %}\beta_1{% endmathjax %}和{% mathjax %}\beta_2{% endmathjax %}是非负加权参数。常将它们设置为{% mathjax %}\beta_1 = 0.9{% endmathjax %}和{% mathjax %}\beta_2 = 0.999{% endmathjax %}。也就是说，方差估计的移动远远慢于动量估计的移动。注意，如果我们初始化{% mathjax %}\mathbf{v}_0 = \mathbf{s}_0 = 0{% endmathjax %}，就会获得一个相当大的初始偏差。我们可以通过使用{% mathjax %}\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}{% endmathjax %}来解决这个问题。相应地，标准化状态变量由下式获得：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \;\text{and}\;\hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}
{% endmathjax %}
有了正确的估计，我们现在可以写出更新方程。首先，我们以非常类似于`RMSProp`算法的方式重新缩放梯度以获得
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{g'}_t = \frac{\eta\hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}
{% endmathjax %}
与`RMSProp`不同，我们的更新使用动量{% mathjax %}\hat{\mathbf{v}}_t{% endmathjax %}而不是梯度本身。此外，由于使用{% mathjax %}\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}{% endmathjax %}而不是进行缩放，两者会略有差异。前者在实践中效果略好一些，因此与`RMSProp`算法有所区分。通常，我们选择{% mathjax %}\epsilon = 10^{-6}{% endmathjax %}，这是为了在数值稳定性和逼真度之间取得良好的平衡。最后，我们简单更新：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g'}_t
{% endmathjax %}
回顾Adam算法，它的设计灵感很清楚：首先，动量和规模在状态变量中清晰可见，它们相当独特的定义使我们移除偏项（这可以通过稍微不同的初始化和更新条件来修正）。其次，`RMSProp`算法中两项的组合都非常简单。最后，明确的学习率{% mathjax %}\eta{% endmathjax %}使我们能够控制步长来解决收敛问题。

`Adam`算法也存在一些问题：即使在凸环境下，当{% mathjax %}\mathbf{s}_t{% endmathjax %}的二次矩估计值爆炸时，它可能无法收敛。为{% mathjax %}\mathbf{s}_t{% endmathjax %}提出了的改进更新和参数初始化。论文中建议我们重写`Adam`算法更新如下：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2)(\mathbf{g}_t^2 - \mathbf{s}_{t-1})
{% endmathjax %}
每当{% mathjax %}\mathbf{g}_t^2 {% endmathjax %}具有值很大的变量或更新很稀疏时，{% mathjax %}\mathbf{s}_t{% endmathjax %}可能会太快地“忘记”过去的值。 一个有效的解决方法是将{% mathjax %}\mathbf{g}_t^2 - \mathbf{s}_{t-1}{% endmathjax %}替换为{% mathjax %}\mathbf{g}_t^2\odot \text{sgn}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}){% endmathjax %}。这就是`Yogi`更新，现在更新的规模不再取决于偏差的量。
{% mathjax '{"conversion":{"em":14}}' %}
\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2)\mathbf{g}_t^2\odot \text{sgn}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})
{% endmathjax %}
论文中，作者还进一步建议用更大的初始批量来初始化动量，而不仅仅是初始的逐点估计。
##### 总结

`Adam`算法将许多优化算法的功能结合到了相当强大的更新规则中。`Adam`算法在`RMSProp`算法基础上创建的，还在小批量的随机梯度上使用`EWMA`。在估计动量和二次矩时，`Adam`算法使用偏差校正来调整缓慢的启动速度。对于具有显著差异的梯度，我们可能会遇到收敛性问题。我们可以通过使用更大的小批量或者切换到改进的估计值{% mathjax %}\mathbf{s}_t{% endmathjax %}来修正它们。`Yogi`提供了这样的替代方案。

#### 学习率调度器

到目前为止，我们主要关注**如何更新权重向量的优化算法**，而不是它们的更新速率。然而，调整学习率通常与实际算法同样重要，有如下几方面需要考虑：
- 首先，学习率的大小很重要。如果它太大，优化就会发散；如果它太小，训练就会需要过长时间，或者我们最终只能得到次优的结果。我们之前看到问题的条件数很重要。直观地说，这是最不敏感与最敏感方向的变化量的比率。
- 其次，衰减速率同样很重要。如果学习率持续过高，我们可能最终会在最小值附近弹跳，从而无法达到最优解。简而言之，我们希望速率衰减，但要比{% mathjax %}\mathcal{O}(t^{-\frac{1}{2}}){% endmathjax %}慢，这样能成为解决凸问题的不错选择。
- 另一个同样重要的方面是初始化。这既涉及参数最初的设置方式，又关系到它们最初的演变方式。这被戏称为**预热**（`warmup`），即我们最初开始向着解决方案迈进的速度有多快。一开始的大步可能没有好处，特别是因为最初的参数集是随机的。最初的更新方向可能也是毫无意义的。
- 最后，还有许多优化变体可以执行周期性学习率调整。

鉴于管理学习率需要很多细节，因此大多数深度学习框架都有自动应对这个问题的工具。
##### 策略

虽然我们不可能涵盖所有类型的**学习率调度器**，但我们会尝试在下面简要概述常用的策略：多项式衰减和分段常数表。 此外，余弦学习率调度在实践中的一些问题上运行效果很好。在某些问题上，最好在使用较高的学习率之前预热优化器。
###### 单因子调度器

多项式衰减的一种替代方案是乘法衰减，即{% mathjax %}\eta_{t+1}\leftarrow \eta_t\cdot\alpha{% endmathjax %}其中{% mathjax %}\alpha\in (0,1){% endmathjax %}。为了防止学习率衰减到一个合理的下界之下，更新方程经常修改为{% mathjax %}\eta_{t+1}\leftarrow \max(\eta_{\min,\eta_t\cdot \alpha}){% endmathjax %}。
```python
class FactorScheduler:
    def __init__(self, factor=1, stop_factor_lr=1e-7, base_lr=0.1):
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.base_lr = base_lr

    def __call__(self, num_update):
        self.base_lr = max(self.stop_factor_lr, self.base_lr * self.factor)
        return self.base_lr

scheduler = FactorScheduler(factor=0.9, stop_factor_lr=1e-2, base_lr=2.0)
plt.plot(tf.range(50), [scheduler(t) for t in range(50)])
```
###### 多因子调度器

训练深度网络的常见策略之一是保持学习率为一组分段的常量，并且不时地按给定的参数对学习率做乘法衰减。具体地说，给定一组降低学习率的时间点，例如{% mathjax %}s = \{5,10,20\}{% endmathjax %}，每当{% mathjax %}t\in s{% endmathjax %}时，降低{% mathjax %}\eta_{t+1}\leftarrow \eta_t\cdot\alpha{% endmathjax %}。假设每步中的值减半，我们可以按如下方式实现这一点。
```python
class MultiFactorScheduler:
    def __init__(self, step, factor, base_lr):
        self.step = step
        self.factor = factor
        self.base_lr = base_lr

    def __call__(self, epoch):
        if epoch in self.step:
            self.base_lr = self.base_lr * self.factor
            return self.base_lr
        else:
            return self.base_lr

scheduler = MultiFactorScheduler(step=[15, 30], factor=0.5, base_lr=0.5)
plt.plot(tf.range(num_epochs), [scheduler(t) for t in range(num_epochs)])
```
这种分段恒定学习率调度背后的直觉是，让优化持续进行，直到权重向量的分布达到一个驻点。此时，我们才将学习率降低，以获得更高质量的代理来达到一个良好的局部最小值。
###### 余弦调度器

余弦调度器是提出的一种启发式算法。它所依据的观点是：我们可能不想在一开始就太大地降低学习率，而且可能希望最终能用非常小的学习率来“改进”解决方案。这产生了一个类似于余弦的调度，函数形式如下所示，学习率的值在{% mathjax %}t\in [0,T]{% endmathjax %}之间。
{% mathjax '{"conversion":{"em":14}}' %}
\eta_t = \eta_T + \frac{\eta_0 - \eta_T}{2}(1 + \cos(\pi t/T))
{% endmathjax %}
这里{% mathjax %}\eta_0{% endmathjax %}是初始学习率，{% mathjax %}\eta_T{% endmathjax %}是当{% mathjax %}T{% endmathjax %}时的目标学习率。此外，对于{% mathjax %}t > T{% endmathjax %}，我们只需将值固定到{% mathjax %}\eta_T{% endmathjax %}而不再增加它。在下面的示例中，我们设置了最大更新步数{% mathjax %}T = 20{% endmathjax %}。
```python
class CosineScheduler:
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + math.cos(
                math.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr

scheduler = CosineScheduler(max_update=20, base_lr=0.3, final_lr=0.01)
plt.plot(tf.range(num_epochs), [scheduler(t) for t in range(num_epochs)])
```
##### 预热

在某些情况下，初始化参数不足以得到良好的解。这对某些高级网络设计来说尤其棘手，可能导致不稳定的优化结果。对此，一方面，我们可以选择一个足够小的学习率，从而防止一开始发散，然而这样进展太缓慢。另一方面，较高的学习率最初就会导致发散。解决这种困境的一个相当简单的解决方法是使用预热期，在此期间学习率将增加至初始最大值，然后冷却直到优化过程结束。为了简单起见，通常使用线性递增。预热可以应用于任何调度器，而不仅仅是余弦。其中，这篇论文的点睛之笔的发现：预热阶段限制了非常深的网络中参数的发散程度 。这在直觉上是有道理的：在网络中那些一开始花费最多时间取得进展的部分，随机初始化会产生巨大的发散。
##### 总结

在训练期间逐步降低学习率可以提高准确性，并且减少模型的过拟合。在实验中，每当进展趋于稳定时就降低学习率，这是很有效的。从本质上说，这可以确保我们有效地收敛到一个适当的解，也只有这样才能通过降低学习率来减小参数的固有方差。余弦调度器在某些计算机视觉问题中很受欢迎。优化之前的预热期可以防止发散。优化在深度学习中有多种用途。对于同样的训练误差而言，选择不同的优化算法和学习率调度，除了最大限度地减少训练时间，可以导致测试集上不同的泛化和过拟合量。