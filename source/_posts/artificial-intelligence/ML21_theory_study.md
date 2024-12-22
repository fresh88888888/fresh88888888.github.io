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

#### 泛化

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

**定义**：（`ZSPT`可控上下文）。可控上下文`ZSPT`问题与上面的`ZSPT`问题相同，只是学习算法可以在训练期间调整训练`CMDP`{% mathjax %}C_{\text{train}}{% endmathjax %}的上下文分布，只要它保持仅从训练上下文集合中采样的属性：如果{% mathjax %}c\in C_{\text{train}}{% endmathjax %}，则{% mathjax %}p_{\text{train}} = 0{% endmathjax %}。

请注意，这种形式定义了一类问题，每个问题都由`CMDP`、训练和测试上下文集合的选择以及上下文是否可控决定。不对上下文-`MDP`之间`CMDP`内的共享结构做出任何假设：对于任何特定问题，学习可能需要某种此类假设（隐式或显式）。评估**零样本泛化**。与监督学习一样，可以将训练和测试性能之间的差距视为**泛化**的衡量标准。将其定义为类似于监督学习中的定义，交换训练和测试之间的顺序（最大化奖励，而不是最小化损失）：
{% mathjax '{"conversion":{"em":14}}' %}
\text{GenGap}(\pi) := \mathbf{R}(\pi,\mathcal{M}|_{C_{\text{train}}}) - \mathbf{R}(\pi,\mathcal{M}|_{C_{\text{test}}})
{% endmathjax %}
在**监督学习**中，不同算法的**泛化**能力通常通过任务的最终表现来评估。当用于评估模型的任务接近（或相同）模型最终将部署的任务时，很明显，最终性能是一个很好的评估指标。然而，在**强化学习**(`RL`)中，使用的基准任务通常与想要应用这些算法的最终现实任务非常不同。此外，**强化学习算法**目前仍然很脆弱，性能可能会因超参数调整和所使用的特定任务而有很大差异。在这种情况下，可能更关心算法的**零样本泛化**潜力，方法是将泛化与训练性能分离，并使用泛化差距进行评估。例如，如果算法`A`的测试性能高于算法`B`，但`A`的泛化差距大得多，更倾向于在新的环境中使用算法`B`，这样我们就可以更好地保证部署性能不会偏离训练性能太多，并且算法可以更稳健。这就是以前的文献经常在测试性能的同时报告这一指标的原因。然而，**强化学习**中的泛化差距与**监督学习**泛化差距存在同样的问题：差距为零并不一定意味着性能良好（即随机策略的差距可​​能为`0`），如果**奖励函数**在训练和测试中不可比，那么差距的大小可能不具参考价值。将其用作提高性能的唯一指标可能不会导致`ZSG`取得进展。此外，鉴于目前的假设范围很广，不太可能存在一个单一的通用衡量标准来衡量`ZSG`的进展：在如此广泛的问题类别中，目标甚至可能相互冲突。因此，我们的建议首先是关注特定问题的**基准**，并恢复到使用特定设置中的整体性能的`SL`标准（例如视觉干扰、随机动态、稀疏奖励、硬探索）。各种**强化学习**(`RL`)算法的**泛化性能**取决于部署的环境类型，因此需要仔细分类部署时存在的类型，正确评估`ZSG`能力。