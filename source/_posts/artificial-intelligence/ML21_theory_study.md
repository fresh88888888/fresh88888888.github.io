---
title: 机器学习(ML)(二十一) — 强化学习探析
date: 2024-12-19 17:30:11
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

#### 泛化(Generalization)

**强化学习**(`RL`)可用于自动驾驶汽车和机器人等领域，可以在现实世界中使用**强化学习**(`RL`)算法。现实是多种多样、非平稳和开放的，为了处理各种情况，**强化学习**(`RL`)需要对环境的变化具有**鲁棒性**，并且能够在部署期间转移和适应从未见过的（但相似的）环境。**强化学习**(`RL`)中的**泛化**就是创建可以解决这些困难的方法，挑战以前的**强化学习**(`RL`)研究中的一个常见假设，即训练和测试环境是相同的。
<!-- more -->
**泛化**是指**强化学习**(`RL`)中的一类问题，而不是特定问题。第一种是单例环境，传统**强化学习**(`RL`)专注于训练和测试相同的环境；但在现实世界中，训练和测试环境会有所不同；第二种是`IID`零样本泛化环境，来自相同的分布；第三种是`OOD`零样本泛化环境，来自不同的分布。`OOD`(`Out-of-Distribution`)指的是在训练模型时未见过的数据分布。**独立同分布**(`Independent and Identically Distributed,IID`)是概率论和统计学中的一个基本概念，广泛应用于机器学习和数据分析等领域。
{% asset_img ml_1.png %}

当我们说**强化学习**中的**泛化**时，到底是什么意思呢？该领域的所有研究都是在一些任务、级别或环境中进行的，而**泛化**是通过这个集合的不同子集上训练和测试进行衡量的。**上下文马尔可夫决策过程**(`CMDP`)，它可以捕捉所有这些设置。每个任务、层级或环境都是由一个上下文决定的，例如，这个上下文可以是程序内容生成过程的随机种子、任务`ID`或用于可控仿真和渲染过程的参数向量。为了测试泛化能力，要将上下文集合分为**训练集**和**测试集**。策略在训练集上进行训练，然后在测试集上进行评估，测试集通常包含从未见过的上下文，因此也包含从未见过的级别、任务或环境。**泛化**问题由**上下文马尔可夫决策过程**(`CMDP`)和训练、测试上下文集合的选择来指定。

**监督学习**中的**泛化**是一个被广泛研究的领域，比**强化学习**中的**泛化**更成熟。在**监督学习**中，一些**预测器**在训练数据集上进行训练，模型的性能在保留的测试数据集上进行评估。通常训练和测试数据集中的数据点都是从相同的底层分布中**独立同分布**(`IID`)绘制的，**泛化性能**与**测试性能**同义，因为模型需要“**泛化**”在训练期间从未见过的输入。对于训练、测试数据{% mathjax %}D_{\text{train}},D_{\text{test}}{% endmathjax %}和**损失函数**{% mathjax %}\mathcal{L}{% endmathjax %}的模型{% mathjax %}\phi{% endmathjax %}，在**监督学习**中的**泛化**差距定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\text{GenGap}(\phi) = \mathbb{E}_{(x,y)\sim D_{\text{test}}}[\mathcal{L}(\phi,x,y)] - \mathbb{E}_{(x,y)\sim D_{\text{train}}}[\mathcal{L}(\phi,x,y)]
{% endmathjax %}
这种**差距**通常被作为**泛化**的衡量标准，与训练或测试性能无关：对于给定的训练性能指标，**差距**越小意味着模型的**泛化**能力越好。这个指标并不完美，因为在训练和测试中随机表现模型的差距为`0`。此外，如果训练和测试数据集不是`IID`绘制的，那么测试数据集可能更容易（或更难），因此差距为`0`并不意味着泛化很好。但是，它可以用于衡量跨基准的**泛化性能**，其中绝对性能无法比较，或通过降低差距而不改变测试性能（实际上是通过降低训练性能）来激励可能改善**泛化**方法的改进。这些方法可以与提高训练性能的方法相结合，以提高整体测试性能，假设这些方法不冲突。引入这个指标是为了完整性，因为它经常在文献中用于测试性能。总之，它作为测试的补充指标很有用，但不能替代。在**监督学习**中经常研究的一种与**强化学习**(`RL`)相关的**泛化**是**组合泛化**。虽然这是为**语言泛化**而设计的，但其中许多形态与**强化学习**(`RL`)相关。`5`种**组合泛化**形态为：
- **系统性**：通过系统地重新组合已知部分和规则实现概括。
- **生产力**：将预测扩展到训练数据中的长度之外的能力。
- **替代性**：通过用同义词替换组件的能力实现概括。
- **局部性**：模型组合操作是局部的还是全局的。
- **过度概括**：模型是否关注或对异常具有鲁棒性。

**强化学习**(`RL`)中的**马尔可夫决策过程**(`MDP`)。`MDP`由一个元组{% mathjax %}M = (S,A,R,T,p){% endmathjax %}组成，其中{% mathjax %}S{% endmathjax %}是状态空间；{% mathjax %}A{% endmathjax %}是动作空间；{% mathjax %}R\;:\;S\times A \times S\rightarrow \mathbb{R}{% endmathjax %}是**奖励函数**；{% mathjax %}T(s_0|s,a){% endmathjax %}是**随机的马尔可夫转换函数**；{% mathjax %}p(s_0){% endmathjax %}是**初始状态分布**。我们还考虑部分可观察的`MDP`(`POMDP`)由一个元组{% mathjax %}M = (S,A,O,R,T,\phi,p){% endmathjax %}组成，其中{% mathjax %}O{% endmathjax %}是观察空间，{% mathjax %}\phi : S\rightarrow O{% endmathjax %}是发射或观察函数。在`POMDP`中，策略观察由{% mathjax %}\phi{% endmathjax %}产生的。 `MDP`的问题是学习一个策略{% mathjax %}\pi(a|s){% endmathjax %}，该策略在给定状态下产生动作分布，使得`MDP`中策略的累积奖励最大化：
{% mathjax '{"conversion":{"em":14}}' %}
\pi^* = \text{arg}\;\underset{\pi\in \prod}{\max}\mathbb{E}_{s\sim  p(s_0)}[\mathcal{R}(s)]
{% endmathjax %}
其中{% mathjax %}\pi^*{% endmathjax %}是最优策略，{% mathjax %}\prod{% endmathjax %}是所有策略的集合，{% mathjax %}\mathcal{R}:S\rightarrow \mathbb{R}{% endmathjax %}是状态的回报，计算如下：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}(s):= \mathbb{E}_{a_t\sim\pi(a_t|s_t),s_{t+1}\sim T()s_{t+1}|s_t,a_t}\bigg[\sum\limits_{t=0}^{\infty}R(s_t,a_t,s_{t+1})|s_0 = s \bigg]
{% endmathjax %}
策略是从状态{% mathjax %}s{% endmathjax %}获得的总预期奖励。`POMDP`的目标是相同的，但策略使用观察而不是状态作为输入。如果**马尔可夫决策过程**(`MDP`)没有固定的时间范围，则该总和可能不存在。因此我们通常使用两种回报形式之一，要么假设每个回合有固定的步数（时间范围`H`），要么通过折扣因子{% mathjax %}\gamma{% endmathjax %}对预期奖励做指数折扣。请注意，这里将策略形式化为**马尔可夫策略**，以简化问题。但该策略可以将完整历史记录{% mathjax %}(s_1,a_1,r_1,\ldots,s_{t-1},a_{t-1},r_{t-1},s_t){% endmathjax %}作为输入，例如使用**循环神经网络**。我们将状态和动作空间的历史集合定义为{% mathjax %}H[S,A] = \{(s_1,a_1,r_1,\ldots,s_{t-1},a_{t-1},r_{t-1},s_t)|t\in \mathbb{N}\}{% endmathjax %}，观察空间也是如此。**非马尔可夫策略**使其具有自适应性。

谈到**零样本泛化**(`ZSG`)，我们希望有一种推理任务、环境实例或级别集合一种方式：泛化的需求源于在不同的环境实例集合上训练和测试的场景。以`OpenAI Procgen`为例：在这个基准测试套件中，每个游戏都是程序生成的集合。生成哪种级别完全由级别种子决定，标准协议是在一组固定的`200`种级别上训练策略，然后评估整个级别分布的性能。所有基准测试都共享这种结构：它们有一个级别或任务的集合，由某个种子、`ID`或参数向量指定，并且通过在级别或任务集合上的不同分布进行训练和测试来衡量**泛化**。**上下文马尔可夫决策过程**(`CMDP`)的定义：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{M} = (S',A,O,R,T,C,\phi\;:\;S'\times C\rightarrow O,p(s'|c),p(c))
{% endmathjax %}
对于{% mathjax %}A,O,R,T,\phi{% endmathjax %}与`POMDP`的定义相同。{% mathjax %}C{% endmathjax %}是上下文空间。`CMDP`是一个拥有状态空间{% mathjax %}S:=S'\times C{% endmathjax %}、初始状态分布{% mathjax %}p((s',c)) = p(c)p(s'|c){% endmathjax %}的`POMDP`，即`POMDP`{% mathjax %}(S'\times C,A,O,R,T,\phi,p(c)p(s'|c)){% endmathjax %}。因此，{% mathjax %}R{% endmathjax %}的类型为{% mathjax %}R:S'\times C\rightarrow \mathbb{R}{% endmathjax %}，并且{% mathjax %}T((s,c),a){% endmathjax %}是**转移概率分布**的形式。对于要成为`CMDP`的元组，必须对转移函数进行分解，使得上下文在一个回合内不会发生变化，如果{% mathjax %}c'\neq c{% endmathjax %}，则{% mathjax %}T((s,c),a)((s',c')) = 0{% endmathjax %}。{% mathjax %}S_0{% endmathjax %}则被称为**底层状态空间**，将{% mathjax %}p(c){% endmathjax %}称为**上下文分布**。

在这个定义中，上下文(`context`)充当种子(`seed`)、ID或参数向量的角色，这决定了级别的高低。这里的“**上下文**”指的是在模拟或算法中所使用的状态信息，它包含了所有必要的参数和设置，以便于生成可重复的结果。因此，它不应该在一集合内改变，而应该在各个集合之间改变。`CMDP`是整个任务或环境实例的集合；在`Procgen`中，每个游戏（例如`starpilot、coinrun`等）都是一个单独的`CMDP`。上下文分布{% mathjax %}p(c){% endmathjax %}用于确定级别、任务或环境实例的训练和测试集合；在`Procgen`中，这个分布在训练时固定的`200`个种子上是均匀的，在测试时所有种子上是均匀的。请注意，此定义未指定智能体是否观察到上下文：如果对于某个底层观察空间{% mathjax %}O'{% endmathjax %}和{% mathjax %}\phi((s',c)) = (\phi'(s),c){% endmathjax %}，对于某个底层观察函数{% mathjax %}\phi':S'\rightarrow O'{% endmathjax %}，则观察到上下文，否则没有。需要观察到上下文才能使`CMDP`成为`MDP`（而不是`POMDP`），{% mathjax %}\phi'{% endmathjax %}不能是恒等式，这种情况下`POMDP`不太可能是`MDP`。由于**奖励函数**、**转换函数**、**初始状态分布**和**发射函数**都将上下文作为输入，因此上下文的选择决定了除了**动作空间**之外`MDP`的一切，我们假设动作空间是固定的。给定一个上下文{% mathjax %}c^*{% endmathjax %}，`CMDP`{% mathjax %}\mathcal{M}{% endmathjax %}限制为单个上下文的`MDP`称为**上下文马尔可夫决策过程**(`CMDP`){% mathjax %}\mathcal{M}_{c^*}{% endmathjax %}。通常这是一个新的`CMDP`，如果{% mathjax %}c = c^*{% endmathjax %}，则{% mathjax %}p(c):= 1{% endmathjax %}，否则为`0`。这是一个特定的任务或环境实例，例如，`Procgen`中游戏的单个等级，由上下文的单个随机种子指定。一些`MDP`具有随机转换或奖励函数。当模拟这些`MDP`时，研究人员通常通过选择随机种子来控制这种随机性。理论上，这些随机**MDP**可以被视为上下文`MDP`，其中上下文是随机种子。不认为随机`MDP`以这种方式自动关联上下文，并假设随机种子始终是随机选择的，而不是作为上下文建模。这更接近于现实世界中随机动态的场景，这里无法控制随机性。

使用`CMDP`形式来描述我们关注的**泛化**问题类别。如前所述，泛化需求源于训练和测试环境实例之间的差异，因此我们希望指定一组训练上下文`MDP`和一组测试集。通过上下文集合来指定这些上下文`MDP`集合，因为上下文唯一地决定了`MDP`。首先，需要描述如何使用训练和测试上下文集来创建新的`CMDP`。

**定义**：对于任何`CMDP`{% mathjax %}\mathcal{M} = (S',A,O,R,T,C,\phi,p(s'|c),p(c)){% endmathjax %}，可以选择上下文集合{% mathjax %}C'\subseteq C{% endmathjax %}的一个子集，然后生成一个新的`CMDP`。
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{M}|_{C'} = (S',A,O,R,T,C',\phi\,p(s'|c),p'(c))
{% endmathjax %}
其中，若{% mathjax %}c\in C'{% endmathjax %}，则{% mathjax %}p'(c) = \frac{p(c)}{Z}{% endmathjax %}，否则为`0`，且{% mathjax %}Z{% endmathjax %}是一个重正化项{% mathjax %}Z = \sum_{c\in C'}p(c){% endmathjax %}，确保{% mathjax %}p'(c){% endmathjax %}是**概率分布**。这样，我们就可以根据上下文将整个上下文`MDP`集合拆分为更小的子集。例如，在`Procgen`中，所有种子集的任何可能子集都可用于定义具有有限级别集的不同版本的游戏。对于目标，我们使用策略的预期回报：

**定义**：对于任何`CMDP`{% mathjax %}\mathcal{M}{% endmathjax %}，定义策略的预期回报为`CMDP`：
{% mathjax '{"conversion":{"em":14}}' %}
R(\pi,\mathcal{M}) := \mathbb{E}_{c\sim p(c)}[\mathcal{R}(\pi,\mathcal{M}_c)]
{% endmathjax %}
其中{% mathjax %}\mathcal{R}{% endmathjax %}是（上下文）`MDP`中策略的预期回报，而{% mathjax %}p(c){% endmathjax %}是**上下文分布**。现在可以正式定义**零样本策略转移**(`ZSPT`)问题类。

**定义**（**零样本策略迁移**）：`ZSPT`问题由上下文集合{% mathjax %}C{% endmathjax %}的`CMDP`{% mathjax %}\mathcal{M}{% endmathjax %}的选择以及训练和测试上下文集合{% mathjax %}C_{\text{train}},C_{\text{train}}\subseteq C{% endmathjax %}的选择定义。目标是生成**非马尔可夫策略**{% mathjax %}\pi:H[O,A]\rightarrow A{% endmathjax %}，从而最大化测试`CMDP`{% mathjax %}\mathcal{M}|_{C_{\text{test}}}{% endmathjax %}中的预期回报：
{% mathjax '{"conversion":{"em":14}}' %}
J(\pi) = \mathbf{R}(\pi, \mathcal{M}|_{C_{\text{test}}})
{% endmathjax %}
该策略可以跟`CMDP`{% mathjax %}\mathcal{M}|_{C_{\text{train}}}{% endmathjax %}训练来产生，针对固定环境和回合样本{% mathjax %}N_s,N_e{% endmathjax %}。

**零样本泛化**(`ZSG`)研究通常涉及开发能够解决各种`ZSPT`问题的算法。例如，在`Procgen`中，目标是生成一种可以解决每个游戏的`ZSPT`问题的算法。希望在对关卡的训练分布（固定的`200`个关卡集）进行`2500`万步（{% mathjax %}N_s = 25,N_e = \infty{% endmathjax %}）训练后，在测试分布（即关卡的完整分布）上实现尽可能高的回报。

**定义**（`ZSPT`**可控上下文**）：可控上下文`ZSPT`问题与上面的`ZSPT`问题相同，只是学习算法可以在训练期间调整训练`CMDP`{% mathjax %}C_{\text{train}}{% endmathjax %}的上下文分布，只要它保持仅从训练上下文集合中采样的属性：如果{% mathjax %}c\in C_{\text{train}}{% endmathjax %}，则{% mathjax %}p_{\text{train}} = 0{% endmathjax %}。

请注意，这种形式定义了一类问题，每个问题都由`CMDP`、训练和测试上下文集合的选择以及上下文是否可控决定。不对上下文-`MDP`之间`CMDP`内的共享结构做出任何假设：对于任何特定问题，学习可能需要某种此类假设（隐式或显式）。评估**零样本泛化**。与监督学习一样，可以将训练和测试性能之间的差距视为**泛化**的衡量标准。将其定义为类似于监督学习中的定义，交换训练和测试之间的顺序（最大化奖励，而不是最小化损失）：
{% mathjax '{"conversion":{"em":14}}' %}
\text{GenGap}(\pi) := \mathbf{R}(\pi,\mathcal{M}|_{C_{\text{train}}}) - \mathbf{R}(\pi,\mathcal{M}|_{C_{\text{test}}})
{% endmathjax %}
在**监督学习**中，不同算法的**泛化**能力通常通过任务的最终表现来评估。当用于评估模型的任务接近（或相同）模型最终将部署的任务时，很明显，最终性能是一个很好的评估指标。然而，在**强化学习**(`RL`)中，使用的基准任务通常与想要应用这些算法的最终现实任务非常不同。此外，**强化学习算法**目前仍然很脆弱，性能可能会因超参数调整和所使用的特定任务而有很大差异。在这种情况下，可能更关心算法的**零样本泛化**潜力，方法是将泛化与训练性能分离，并使用泛化差距进行评估。例如，如果算法`A`的测试性能高于算法`B`，但`A`的泛化差距大得多，更倾向于在新的环境中使用算法`B`，这样我们就可以更好地保证部署性能不会偏离训练性能太多，并且算法可以更稳健。这就是以前的文献经常在测试性能的同时报告这一指标的原因。然而，**强化学习**中的泛化差距与**监督学习**泛化差距存在同样的问题：差距为零并不一定意味着性能良好（即随机策略的差距可​​能为`0`），如果**奖励函数**在训练和测试中不可比，那么差距的大小可能不具参考价值。将其用作提高性能的唯一指标可能不会导致`ZSG`取得进展。此外，鉴于目前的假设范围很广，不太可能存在一个单一的通用衡量标准来衡量`ZSG`的进展：在如此广泛的问题类别中，目标甚至可能相互冲突。因此，我们的建议首先是关注特定问题的**基准**，并恢复到使用特定设置中的整体性能的`SL`标准（例如视觉干扰、随机动态、稀疏奖励、硬探索）。各种**强化学习**(`RL`)算法的**泛化性能**取决于部署的环境类型，因此需要仔细分类部署时存在的类型，正确评估`ZSG`能力。

现在对**强化学习**(`RL`)中的`ZSG`进行分类。当训练和测试上下文集合不相同时，就会出现`ZSG`问题。`ZSG`问题有很多种类型，因此有许多不同风格的方法。将方法分为：增加训练和测试数据与目标之间的相似性的方法、处理训练和测试环境之间差异的方法和提升`ZSG`性能优化的方法。它们主要改变环境、损失函数或架构。进行全面的分类使我们能够看到`ZSG`研究中尚未探索的领域，
{% asset_img ml_2.png "强化学习中解决零样本泛化问题的方法分类" %}

**强化学习**(`RL`)中`ZSG`的研究仍属新兴事物，但如果想要开发适用于解决实际问题的**强化学习**方案，那么这项研究至关重要。这里提出了`ZSG`基准的分类，将分类法分为**环境**和**评估协议**，并对现有的解决`ZSG`问题的方法进行了分类。
- **零样本策略迁移**值得研究，即使在特定设置下我们也可以放宽零样本假设，因为它提供了可以构建特定领域解决方案的基础算法。
- 应该做更多的工作来超越**零样本策略迁移**，特别是在**持续强化学习**中，作为一种绕过不变最优性原则限制的方法。
- 现实世界中，必须同时考虑**样本效率**和**上下文效率**。评估方法在不同大小的训练上下文集合上的性能是一个有用的评估指标，它提供了更多信息来选择不同的方法。

**强化学习**(`RL`)是一种**顺序决策范式**，用于训练**智能体**处理复杂任务，例如机器人运动、玩视频游戏和设计硬件芯片。虽然**强化学习**(`RL`)的**智能体**在各种活动中都表现出了良好的效果，但很难将这些**智能体**的能力转移到新任务上，即使这些任务在语义上是等效的。例如，考虑一个跳跃任务，其中一个**智能体**需要从图像观察中学习，跳过一个障碍物。经过其中一些任务的训练，**深度强化学习**的**智能体**的障碍物位置各不相同，但很难成功跳过以前从未见过的位置的障碍物。先前关于**泛化**的研究通常来自**监督学习**，并围绕**增强学习**过程展开。这些方法很少利用**序列**方面的属性，例如跨时间观察的动作相似性。当**智能体**在这些状态下的最佳行为与未来状态相似时，这些状态就接近。这种**接近性**，称之为**行为相似性**，可以推广到不同任务的观察中。为了衡量不同任务中状态之间的**行为相似性**（例如，跳跃任务中不同的障碍物位置），引入了**策略相似性度量**(`PSM`)，这是一种受双模拟启发的理论驱动的**状态相似性度量**。**策略相似性度量**(`PSM`)将高相似性分配给此类行为相似的状态，将低相似性分配给不相似的状态。

为了提高泛化能力，需要学习**状态嵌入**，它对应于**神经网络**的任务状态表示，将行为相似的状态聚集在一起，同时将行为不同的状态分开。为此，提出了**对比度量嵌入**(`CME`)，利用**对比学习**的优势来学习**基于状态相似性度量的表示**。使用**策略相似性度量**(`PSM`)实例化对比嵌入，以学习策略相似性嵌入(`PSE`)。`PSE`将相似的表示分配给在这些状态和未来状态下具有相似行为的状态。通过使用`UMAP`（一种流行的高维数据可视化技术）将`PSE`和**基线方法**学习到的表示投影到`2D`点，从而对它们进行可视化。与之前的方法不同，`PSE`将行为相似的状态聚集在一起，将不相似的状态分开。此外，`PSE`将状态划分为两组：(1) 跳跃前的所有状态和 (2) 动作不影响结果的状态（跳跃后的状态）。

总体而言，这项研究通过两项贡献推动了**强化学习**中的**泛化**：**策略相似性度量**和**对比度量嵌入**。`PSE`结合了这两个方法来增强**泛化**。未来研究希望找到更好的方法来定义**行为相似性**并利用这种结构进行**表征学习**。

#### Decision Transformers 

`Decision Transformers`是一种新的**强化学习模型**。这一模型将**强化学习**(`RL`)问题抽象为条件序列建模问题。**条件序列建模**：与传统的通过**拟合值函数**或计算**策略梯度**来训练策略的方法不同，`Decision Transformers`利用**序列建模算法**（如`Transformer`），在给定期望回报、过去状态和动作的情况下生成未来的动作。这种方法是**自回归**的，意味着它会根据输入的历史数据逐步预测未来的动作。`Decision Transformers`模型会将状态、动作和回报作为输入，通过**因果自注意力机制**处理这些信息，从而生成一系列未来的动作。这种方法不仅简化了**强化学习**(`RL`)中的策略优化过程，还允许模型在没有与环境直接交互的情况下进行训练，这种方式被称为**离线强化学习**。

**离线强化学习**：在由元组{% mathjax %}(S,A,P,R){% endmathjax %}描述的**马尔可夫决策过程**(`MDP`)中进行学习。`MDP`元组由状态{% mathjax %}s\in S{% endmathjax %}、动作{% mathjax %}a\in A{% endmathjax %}、转换动态{% mathjax %}P(s_0|s,a){% endmathjax %}和奖励函数{% mathjax %}r = R(s,a){% endmathjax %}组成。分别使用{% mathjax %}s_t{% endmathjax %}、{% mathjax %}a_t{% endmathjax %}和{% mathjax %}r_t = R(s_t,a_t){% endmathjax %}表示时间步{% mathjax %}t{% endmathjax %}时的状态、动作和奖励。轨迹由一系列状态、动作和奖励组成：{% mathjax %}\tau = (s_0,a_0,r_0,s_1,a_1,r_1,\ldots,s_T,a_T,r_T){% endmathjax %}。轨迹在时间步{% mathjax %}t{% endmathjax %}的回报{% mathjax %}R_t = \sum\limits_{t'=t}^T r_{t'}{% endmathjax %}，是该时间步未来奖励的总和。**强化学习**的目标是学习一种策略，该策略可最大化`MDP`中的预期回报{% mathjax %}\mathbb{E}[\sum_{t=1}^T r_t]{% endmathjax %}。在**离线强化学习**中，无法通过环境交互获取数据，而是只能访问由任意策略的轨迹展开组成的一些固定的有限数据集。这种设置更难，因为它剥夺了**智能体**探索环境和收集额外反馈的能力。

`Transformer`：作为一种高效建模序列数据的架构。这些模型由**堆叠**的**自注意力层**和**残差连接**组成。每个自注意力层接收对应于唯一输入标记的`n`个嵌入{% mathjax %}\sum_{i=1}^n{% endmathjax %}，并输出{% mathjax %}n{% endmathjax %}个嵌入{% mathjax %}\sum_{i=1}^n{% endmathjax %}，保留输入维度。第{% mathjax %}i{% endmathjax %}个标记通过线性变换映射到键{% mathjax %}k_i{% endmathjax %}、查询 {% mathjax %}q_i{% endmathjax %}和值{% mathjax %}v_i{% endmathjax %}。**自注意力层**的第{% mathjax %}i{% endmathjax %}个输出由查询{% mathjax %}q_i{% endmathjax %}和其他键{% mathjax %}k_j{% endmathjax %}之间的归一化点积加权值{% mathjax %}v_j{% endmathjax %}得出：
{% mathjax '{"conversion":{"em":14}}' %}
z_i = \sum\limits_{j=1}^n \text{softmax}(\{\langle q_i, k_{j'}\rangle\}_{j' = 1}^n)_j\cdot v_j
{% endmathjax %}
这允许该层通过查询和键向量的**相似性**（最大化点积）隐式状态返回关联来分配“信用”。在这项工作中，使用`GPT`架构，它使用**因果自注意力掩码**修改了`Transformer`架构，以实现自回归生成，用序列中的前一个标记{% mathjax %}(j\in [1,i]){% endmathjax %}替换{% mathjax %}n{% endmathjax %}个标记上的求和/激活。

**轨迹表示**：选择**轨迹表示**的关键要求是，它应该使`Transformers`能够学习的模式，并且能够在测试时有条件地生成动作。创建奖励模型并非易事，因为希望模型根据未来的期望回报而不是过去的回报来生成动作。因此，不是直接提供奖励，而是向模型提供回报{% mathjax %}\hat{R}_t = \sum_{t' = t}^T r_{t'}{% endmathjax %}。该表示适合自回归训练和生成：
{% mathjax '{"conversion":{"em":14}}' %}
\tau = (\hat{R}_1, s_1, a_1,\hat{R}_2, s_2, a_2,\ldots,\hat{R}_T, s_T, a_T)
{% endmathjax %}
在测试时，可以指定期望的性能（例如，`1`表示成功或`0`表示失败）以及环境起始状态，作为启动生成的条件信息。在执行当前状态的生成操作后，我将目标回报减少已获得的奖励并重复，直到回合终止。**架构**：将最后{% mathjax %}K{% endmathjax %}个时间步输入到`Decision Transformer`中，总共{% mathjax %}3K{% endmathjax %}个`token`（每个模态包含：未来回报、状态或动作）。为了获得`token`嵌入，让每种模态学习一个线性层，将原始输入投射到嵌入维度，然后进行层规范化。对于具有视觉输入的环境，状态被输入到**卷积编码器**而不是线性层。此外，每个时间步的嵌入都会被学习并添加到每个`token`中，请注意，这与`Transformer`使用的位置嵌入不同，因为一个时间步对应`3`个`token`。然后，这些`token`由`GPT`模型处理，该模型通过**自回归**建模预测未来的动作`token`。**训练**：这里获得了一个离线轨迹数据集。从数据集中抽取序列长度为{% mathjax %}K{% endmathjax %}的小批量数据。与输入`token`{% mathjax %}s_t{% endmathjax %}相对应的预测头经过训练以预测出{% mathjax %}a_t{% endmathjax %}，对于离散动作使用**交叉熵损失**，对于连续动作使用**均方误差**，并且对每个时间步的损失进行平均。
```python
# R,s,a,t : return-to-go, states, actions, or timesteps
# transformer : transformer with causal masking (GPT)
# embed_s, embed_a, embed_R : linear embedding layers 
# embed_t : learned episode positional embedding
# pred_a : linear action prediction layer

def decision_transformer(R,s,a,t):
  # compute embeddingsfor tokens
  pos_embedding = embed_t(t)   # per-timestep(note: not per-token)
  s_embedding = embed_s(s) + pos_embedding
  a_embedding = embed_a(a) + pos_embedding
  R_embedding = embed_R(R) + pos_embedding

  # interleave token as (R_1,s_1,a_1,...,R_k,s_k,a_k)
  input_embeds = stack(R_embedding, s_embedding, a_embedding)

  # use transformer to get hidden state
  hidden_states = transformer(input_embeds = input_embeds)

  # select a hidden states for action prediction tokens
  a_hidden = unstack(hidden_states).actions

  # predict actions
  return pred_a(a_hidden)

# training loop
for (R,s,a,t) in dataloader:   # dims: (batch_size, K, dim)
  a_preds = decision_transformer(R,s,a,t)
  loss = mean((a_preds - a) ** 2)  # L2 loss for continuous actions
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

# evaluation loop
target_return = 1
R,s,a,t, done = [target_return], [env.reset()], [], [1], False
while not done:  # autoregressive generation/sampling
  # sample next action
  action = decision_transformer(R,s,a,t)[-1]   # for cts actions
  new_s, r, done, _ = env.step(action)

  # append new tokens to sequence 
  R = R + [R[-1] - r]   # decrement returns -to -go with reward
  s,a,t = s + [new_s], a + [action], t + [len(R)]
  R,s,a,t= R[-K:], ...  # only keep context length of K
```
`Decision Transformer`，旨在统一**语言/序列建模**和**强化学习**的思想。在离线强化学习基准测试中，表明`Decision Transformer`可以匹敌或超越专为离线强化学习设计的强大算法，并且只需对语言建模架构进行最小的修改。可以考虑更复杂的回报、状态和动作嵌入，以回报分布为条件来模拟随机设置而不是确定性回报。`Transformer`模型也可用于模拟轨迹的状态演变，可能作为基于模型的**强化学习**的替代方案，我们希望在未来的工作中探索这一点。在实际应用中，了解`Transformer`在`MDP`设置中犯的错误以及可能产生的负面后果非常重要，但这些后果尚未得到充分探索。训练模型的数据集也很重要，这可能会增加破坏性偏差，特别是当考虑研究使用更多来自可疑来源的数据来增强**强化学习**的智能体时。例如，恶意行为者的奖励设计可能会产生意想不到的动作，因为模型是通过调节期望的回报来产生动作。

#### 强化学习—语言模型

**语言模型**(`LM`)在处理文本时展现出了前所未有的能力。它们用庞大的文本语料库来训练，使它们能够编码各种类型的知识，也包括抽象的物理规律类知识，这些知识是否能让机器人等**智能体**在解决日常任务时收益呢？智能体缺乏学习知识的方法。这种限制阻碍了**智能体**适应环境（例如，修复错误的知识）或学习新技能。

**强化学习算法**通常在缺乏稠密和良好设计的**奖励函数**时表现不佳。内在激励的探索方法通过奖励**智能体**访问新颖的状态或转移来解决这一限制，但在大的环境中，这些方法的好处有限，因为大多数发现的新颖性与后续任务无关。介绍一种利用文本语料库中的背景知识来引导探索的方法。这种方法称为`ELLM`，它通过奖励**智能体**实现由语言模型根据**智能体**当前状态的描述所建议的目标。通过利用**大语言模型**的预训练，`ELLM`指导智能体朝向对人类有意义且可能有用的行为发展，而无需人类参与。在`Crafter`游戏环境和`Housekeep`机器人模拟器中评估了`ELLM`，结果表明，经过`ELLM`训练的智能体在预训练过程中对常识行为的覆盖更好，并且在一系列后续任务上表现相当或更佳。
{% asset_img ml_3.png  %}

这里考虑由元组{% mathjax %}(\mathcal{S},\mathcal{A},\mathcal{O},\Omega,\mathcal{T}, \gamma, \mathcal{R}){% endmathjax %}定义的**部分可观测马尔可夫决策过程**(`POMDP`)，其中观察值{% mathjax %}o\in \Omega{% endmathjax %}是通过{% mathjax %}\mathcal{O}(o|s,a){% endmathjax %}从状态{% mathjax %}s\in \mathcal{S}{% endmathjax %}和动作{% mathjax %}a\in \mathcal{A}{% endmathjax %}导出的。{% mathjax %}\mathcal{T}(s'|s,a){% endmathjax %}描述了环境的动态，而{% mathjax %}\mathcal{R}{% endmathjax %}和{% mathjax %}\gamma{% endmathjax %}分别是环境的**奖励函数**和**折扣因子**。
{% asset_img ml_4.png 左边：ELLM的策略参数化，右边：LLM奖励计划 %}

`ELLM`使用`GPT-3`来作为适当的探索目标，并利用`SentenceBERT`嵌入来计算目标与行为之间的相似性，从而作为一种内在奖励。M个智能体在优化内在奖励{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}的同时，或者替代外部奖励{% mathjax %}\mathcal{R}{% endmathjax %}。特别是`CB-IM`方法通过一系列目标条件奖励函数来定义{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}。具体来说，`CB-IM`方法的内在奖励 {% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}可以表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\text{int}}(o,a,o') = \mathbb{E}_{g\sim \mathcal{G}}[\mathcal{R}_{\text{int}}(o,a,o'|g)]
{% endmathjax %}
`CB-IM`智能体在优化内在奖励{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}时，期望能够在原始奖励{% mathjax %}\mathcal{R}{% endmathjax %}上表现良好，前提是内在奖励{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}更易于优化并且与{% mathjax %}\mathcal{R}{% endmathjax %}高度一致，这样最大化的行为也会最大化{% mathjax %}\mathcal{R}{% endmathjax %}。每个`CB-IM`算法必须在以上公式中定义两个要素：从中抽样的目标分布，即{% mathjax %}\mathcal{G}{% endmathjax %}；**目标条件奖励函数**{% mathjax %}\mathcal{R}_{\text{int}}(o,a,o'|g){% endmathjax %}。基于这些要素，`CB-IM`算法训练一个目标条件策略{% mathjax %}\pi(a,o|g){% endmathjax %}来最大化{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}。对于某些内在奖励函数，智能体可能会立即在原始奖励函数{% mathjax %}\mathcal{R}{% endmathjax %}下获得高奖励；而对于其他函数，则可能需要通过额外的微调来优化{% mathjax %}\mathcal{R}{% endmathjax %}。在以上公式中，目标空间{% mathjax %}\mathcal{G}{% endmathjax %}是由**目标条件奖励函数**{% mathjax %}\mathcal{R}_{\text{int}}(\cdot|g){% endmathjax %}决定的：每个选择的{% mathjax %}g{% endmathjax %}都会引发一个对应的最佳行为分布。因此，`CB-IM算`法的设计需要确保目标选择和奖励函数能够有效引导智能体朝向更一般的奖励函数{% mathjax %}\mathcal{R}{% endmathjax %}，以便在探索过程中实现有效的学习和优化。在选择目标分布{% mathjax %}\mathcal{G}{% endmathjax %}和**目标条件奖励函数**{% mathjax %}\mathcal{R}_{\text{int}}(\cdot|g){% endmathjax %}时，为了帮助智能体朝着一般奖励函数{% mathjax %}\mathcal{R}{% endmathjax %}取得进展，所针对的目标在探索过程中应满足以下三个属性：
- **多样性**：针对多样化的目标可以增加目标行为与其中一个目标相似的机会。
- **常识敏感性**：学习应集中于可行的目标（例如“砍树”比“喝树”更合理），这些目标在我们关心的目标分布中更可能出现（例如“喝水”比“走进岩浆”更合理）。
- **上下文敏感性**：学习应关注当前环境配置中可行的目标（例如，仅在视野中有树时才砍树）。

这些属性旨在确保智能体能够有效地选择和优化目标，从而在复杂环境中实现对一般**奖励函数**{% mathjax %}\mathcal{R}{% endmathjax %}的有效学习。大多数CB-IM算法手动定义与原始任务{% mathjax %}\mathcal{R}{% endmathjax %}对齐的奖励函数{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}和目标分布的支持{% mathjax %}\mathcal{G}{% endmathjax %}，但使用各种内在动机来指导目标抽样，例如新颖性、学习进展和中间难度。在“利用大型语言模型进行探索”(`ELLM`)中，建议利用基于语言的目标表示和基于语言模型的目标生成，以减轻对环境手动编码定义的需求。大型语言模型中捕获的世界知识将使得自动生成多样化、人类可理解且上下文敏感的目标成为可能。预训练的大语言模型大致分为三类：**自回归模型**、**掩码模型**和**编码器-解码器模型**。**自回归模型**（例如`GPT`）通过最大化给定所有前文的下一个单词的**对数似然**来进行训练，因此能够进行语言生成。仅**编码器模型**（例如`BERT`）则通过掩码目标进行训练，从而有效地编码句子的语义。在大文本语料库上预训练语言模型能够在多种语言理解和生成任务中实现令人印象深刻的`0-shot`或`few-shot`表现，这些任务不仅需要语言知识，还需要世界知识。`ELLM`利用**自回归语言模型**生成目标，并使用**掩码语言模型**构建目标的向量表示。当**大语言模型**生成目标时，目标分布的支持范围变得与自然语言字符串的空间一样广泛。虽然无条件查询大语言模型以获取目标可以提供多样性和常识敏感性，但上下文敏感性需要对智能体状态的了解。因此，在每个时间步，需要使用一系列智能体可用动作的提示和当前观察的文本描述，借助状态描述器{% mathjax %}C_{\text{obs}}:\Omega\rightarrow \sum^*{% endmathjax %}来获取目标，其中{% mathjax %}\sum^*{% endmathjax %}是所有字符串的集合。

**大语言模型**(`LLM`)中提取目标的两种具体策略：
- **开放式生成**，其中`LLM`输出建议目标的文本描述（例如“接下来你应该...”）。
- **封闭式生成**，其中将一个可能的目标作为问答任务提供给`LLM（`例如“智能体应该做什么？（是/否）”）。在这种情况下，只有当“是”的对数概率大于“否”时，`LLM`的目标建议才被接受。

前者更适合开放式探索，而后者更适合具有大量但可界定目标空间的环境。**目标条件奖励**，通过测量LLM生成的目标与智能体在环境中转移描述之间的语义相似性来计算给定目标{% mathjax %}g{% endmathjax %}的奖励{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}，该测量由转移描述器{% mathjax %}\mathcal{C}_{\text{transition}}:\Omega\times A \times \Omega \rightarrow \sum{% endmathjax %}完成。
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\text{int}}(o,a,o'|g) =
\begin{cases}
      \Delta(\mathcal{C}_{\text{transition}}(o,a,o'),g)\;\; & \text{if} > T \\
      0 \;\; & \text{otherwise}
\end{cases}
{% endmathjax %}
在这里，**语义相似性函数**{% mathjax %}\Delta(\cdot,\cdot){% endmathjax %}定义为来自语言模型编码器{% mathjax %}E(\cdot){% endmathjax %}对描述和目标的表示之间的**余弦相似性**：
{% mathjax '{"conversion":{"em":14}}' %}
\Delta(\mathcal{C}_{\text{transition}}(o,a,o'),g) = \frac{E(\mathcal{C}_{\text{transition}}(o,a,o'))\cdot E(g)}{E(\|\mathcal{C}_{\text{transition}}(o,a,o'))\| \|E(g)\|}
{% endmathjax %}
在实践中，我们使用预训练的`SentenceBERT`模型作为{% mathjax %}E(\cdot){% endmathjax %}。选择**余弦相似性**来衡量智能体动作与`LLM`生成之间的对齐。当转移的描述与目标描述之间的相似性足够接近{% mathjax %}(\Delta > T){% endmathjax %}其中{% mathjax %}T{% endmathjax %}是**相似性阈值超参数**）时，智能体将根据它们的相似性获得奖励。最后，由于可能会建议多个目标，通过取目标奖励的最大值来奖励智能体实现{% mathjax %}k{% endmathjax %}个建议。
{% mathjax '{"conversion":{"em":14}}' %}
\Delta^{\max} = \underset{i = 1\ldots k}{\max}\Delta\;(\mathcal{C}_{\text{transition}}(o_t,a_t,o_{t+1}),g_t^i)
{% endmathjax %}
因此，`CB-IM`方法的总体奖励函数可以重写为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\text{int}}(o,a,o') = \mathbb{E}_{\text{LLM}(g^{1\ldots k}|C_{\text{obs}}(o))}^{[\Delta^{\max}]}
{% endmathjax %}
为了施加新颖性偏差，过滤掉智能体在同一回合中已经实现的语言模型建议。这可以防止智能体重复探索相同的目标。考虑两种形式的智能体训练：**目标条件设置**，其中智能体获得建议目标列表的句子嵌入{% mathjax %}\pi(a|o,E(g^{1:k})){% endmathjax %}；**无目标设置**，其中智能体无法访问建议目标{% mathjax %}\pi(a|o){% endmathjax %}。虽然在这两种情况下{% mathjax %}\mathcal{R}_{\text{int}}{% endmathjax %}保持不变，但训练一个目标条件的智能体会引入挑战和好处：智能体可能需要时间来学习不同目标的含义并将其与奖励联系起来，但拥有一个**语言-目标条件策略**可能比仅依赖探索奖励训练的智能体更适合下游任务。本文重点关注大语言模型先验对**强化学习**探索的好处，并假设存在一个预先存在的描述。在模拟中，可以通过真实状态模拟器免费获取。在现实世界应用中，可以使用**物体检测模型**、**描述生成模型**或**动作识别模型**。另外，也可以使用具有类似语言模型组件的**多模态视觉-语言模型**。