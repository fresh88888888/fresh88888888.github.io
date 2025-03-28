---
title: 机器学习(ML)(二十) — 强化学习探析
date: 2024-12-11 12:30:11
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

#### ML-Agents

**强化学习**(`RL`)的挑战之一是**创建环境**。幸运的是，我们可以使用**游戏引擎**来实现它。这些引擎（例如`Unity`、`Godot`或`Unreal Engine`）是为创建视频游戏而开发的工具包。它们非常适合创建环境：它们提供物理系统、`2D/3D`渲染等。`Unity ML-Agents Toolkit`是一个`Unity`**游戏引擎**的插件，可以使用`Unity`**游戏引擎**作为**环境构建器**来训练**智能体**(`Agent`)。`Unity ML-Agents Toolkit`提供了许多出色的预制环境。
<!-- more -->
**深度强化学习**中的好奇心是什么？要理解好奇心是什么，我们首先需要了解**强化学习**(`RL`)的两个主要问题：
- **稀疏奖励问题**：即大多数奖励不包含信息，因此被设置为`0`。**强化学习**(`RL`)基于**奖励假设**，即每个目标都可以描述为奖励的最大化。因此，奖励充当**强化学习**(`RL`)**智能体**的反馈；如果没有收到任何反馈，就无法判定执行的动作合不合适。
- **奖励函数**是人为创建的；在每个环境中，都必须创建**奖励函数**。

创建一个**智能体**(`Agent`)固有的**奖励函数**，即由**智能体**(`Agent`)自身生成的**奖励函数**。**智能体**(`Agent`)将充当自学者，因为它将是学生和自己的反馈大师。这种内在奖励机制被称为好奇心，因为这种奖励会促使**智能体**(`Agent`)探索新奇/不熟悉的状态。为了实现这一点，**智能体**(`Agent`)在探索新轨迹时将获得高额奖励。这种奖励的灵感来源于人类的行为方式。人类天生就有探索环境、发现新事物的内在欲望。计算这种内在奖励的方法有很多种。经典方法是将**好奇心**计算为**智能体**(`Agent`)在给定当前状态和所采取的动作的情况下预测下一个状态的误差。

#### Actor-Critic

在基于策略的方法中，我们的目标是直接优化策略，而不使用价值函数。更准确地说，`Reinforce`是基于策略的方法的一个子类，称为**策略梯度方法**。这个子类通过使用**梯度上升**估计最优策略的权重来直接优化策略。虽然`Reinforce`效果很好。但是，由于使用了**蒙特卡洛抽样**来估计回报（使用整个回合来计算回报），因此在**策略梯度预测**中存在显著差异。**策略梯度预测**是收益增长最快的方向。换句话说，如何更新**策略权重**，以便有更高的概率采取良好收益的动作。`Actor-Critic`方法，这是一种结合价值和策略的方法的**混合架构**，它通过减少**方差**来帮助稳定训练：控制智能体(`Agent`)动作的`Actor`（基于策略的方法）；衡量所采取动作好坏的`Critic`（基于价值的方法）。

在`Reinforce`中，根据回报率的高低按比例增加轨迹中动作的概率。记作：{% mathjax %}\nabla_{\theta}J(\theta) = \sum\limits_{t=0} \nabla_{\theta}\log_{\pi_{\theta}}(a_t|s_t)R(\tau){% endmathjax %}。
- 如果回报很高，就将提高（状态，动作）组合的概率。
- 如果回报很低，就将降低（状态，动作）组合的概率。

此回报{% mathjax %}R(\tau){% endmathjax %}是使用**蒙特卡洛抽样**计算的。通过收集一条轨迹并计算折扣回报，并使用此分数来增加或减少该轨迹中采取的每个动作的概率。如果回报良好，所有动作都会通过增加其被采取的可能性而得到“**强化**”。{% mathjax %}R(\tau) = R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \ldots{% endmathjax %}，这种方法的优点是**无偏**。由于不预测回报，因此只使用获得的真实回报。鉴于环境的随机性（一个回合期间的随机事件）和策略的随机性，轨迹可能产生不同的回报，从而导致较高的**方差**。因此，相同的起始状态可能产生截然不同的回报。正因为如此，从相同状态开始的回报在各个回合之间可能会有很大差异。解决办法是通过使用大量轨迹来减少**方差**，希望任何一条轨迹引入的方差总体上都会减少，并提供对回报的“真实”预测。

利用`Actor-Critic`方法减少方差：减少强化算法的方差并更快更好地训练**智能体**(`Agent`)的办法是利用基于策略和价值的方法的组合，即`Actor-Critic`方法。要理解`Actor-Critic`，想象一下你正在玩电子游戏。你可以和一个朋友一起玩，他会给你一些反馈。你是演员，你的朋友是评论家。一开始你不知道怎么玩，所以你随机尝试一些动作。评论家观察你的动作并提供反馈。通过这些反馈，您可以更新您的策略并更好地玩该游戏。另一方面，你的朋友（评论家）也会更新来提供反馈，以便下次反馈得更好。这就是`Actor-Critic`背后的思想。学习两个函数近似：控制智能体如何采取动作的策略：{% mathjax %}\pi_{\theta}(s){% endmathjax %}；通过衡量所采取的动作有多好来协助策略更新**价值函数**：{% mathjax %}\hat{q}_w(s,a){% endmathjax %}。

`Actor-Critic`过程：正如所见，使用`Actor-Critic`方法，有两个函数近似（两个神经网络）。在每个时间步{% mathjax %}t{% endmathjax %}，从环境中获取当前状态{% mathjax %}S_t{% endmathjax %}​并将其作为输入传递给`Actor`和`Critic`。我们的策略输入状态并输出动作。
{% asset_img ml_1.png %}

评论家(`Critic`)将该动作作为输入，并使用{% mathjax %}S_t{% endmathjax %}​和{% mathjax %}A_t{% endmathjax %}​计算在该状态下采取该动作的价值：`Q`值。
{% asset_img ml_2.png %}

在环境中执行的动作{% mathjax %}A_t{% endmathjax %}​输出新的状态{% mathjax %}S_{t+1}{% endmathjax %}和奖励{% mathjax %}R_{t+1}{% endmathjax %}​。
{% asset_img ml_3.png %}

演员(`Actor`)使用`Q`值更新其策略参数。记作：{% mathjax %}\Delta\theta = \alpha\nabla_{\theta}(\log_{\pi_{\theta}}(s,a))\hat{q_w}(s,a){% endmathjax %}。由于其更新了参数，演员(`Actor`)在给定新状态{% mathjax %}S_{t+1}{% endmathjax %}​的情况下生成在{% mathjax %}A_{t+1}{% endmathjax %}时要采取的下一个动作。然后，评论家(`Critic`)会更新其价值参数。
{% asset_img ml_4.png %}

我们可以使用`Advantage`函数作为评论家(`Critic`)代替**动作值函数**来进一步提高学习的稳定性。其思想是`Advantage`函数计算某个动作相对于某个状态下其他动作的相对优势：与该状态下的平均值相比，在某个状态下采取该动作更好，它从**状态-动作对**中减去状态的平均值，记作{% mathjax %}A(s,a) = Q(s,a) - V(s){% endmathjax %}。通过这个函数计算，如果在该状态下采取这个动作，得到的额外奖励与在该状态下得到的平均奖励的差值。额外的奖励是超出该状态预期值的奖励。
- 如果{% mathjax %}A(s,a) > 0{% endmathjax %}：**梯度**就会被推向那个方向。
- 如果{% mathjax %}A(s,a) < 0{% endmathjax %}，**梯度**就会被推向相反的方向。

实现`Advantage`函数需要两个值函数{% mathjax %}-Q(s,a){% endmathjax %} 和{% mathjax %}V(s){% endmathjax %}。可以使用`TD`误差作为`Advantage`函数的良好预测值。记作{% mathjax %}Q(s,a) = r + \gamma V(s')\;,\;A(s,a) = r + \gamma V(s') - V(s){% endmathjax %}。

#### 多智能体强化学习

之前我们研究了单智能体的强化学习，但是实际我们处在一个多智能体的世界，智能体与智能体之间互动。因此，我们的目标是创建能够与多个智能体互动的智能体。多个智能体在同一个环境中共享和交互。例如，你可以想象一个仓库，其中多个机器人需要导航来装载和卸载包裹，或者一条有几辆自动驾驶汽车的道路。在这些示例中，我们有多个智能体在环境中与其他智能体进行交互。鉴于在多智能体系统中，智能体与其他智能体交互，我们可以拥有不同类型的环境：
- **合作环境**：您的代理商需要最大化共同利益的地方。例如，在仓库中，机器人必须协作才能高效地装卸包裹。
- **竞争/对抗环境**：在这种情况下，智能体希望通过最小化对手的利益来最大化自己的利益。例如，在一场网球比赛中，每个智能体都想击败另一个智能体。
- **对抗与合作混合**：就像在`SoccerTwos`（一款`2vs2`游戏）环境中一样，两个智能体是团队的一部分，他们需要相互合作并击败对手团队。

目前有`2`种方案来设计多智能体系统：**集中式**和**去中心化式**：
- **集中式**：在这个方案中，有一个收集智能体(`Agent`)经验的流程：**经验缓冲区**。通过利用经验缓冲区的经验来学习一个通用的策略。
{% asset_img ml_5.png %}
- **去中心化式**：在分散式学习中，每个智能体(`Agent`)都独立于其他智能体(`Agent`)进行训练。而不是关心其他智能体正在做什么。好处是代理之间不共享任何信息，因此可以像训练单个智能体一样设计和训练。然而这种方式的缺点是它会使环境变得不稳定，因为底层的**马尔可夫决策过程**会随着其他智能体在环境中交互而随时间而变化。这对于许多无法在非平稳环境中达到全局最优的**强化学习算法**来说是个问题。
{% asset_img ml_6.png %}

在去中心化方式中，独立对待所有智能体，而不考虑其他智能体的存在。在这种情况下，所有智能体都将其他智能体视为环境的一部分。这是一个非平稳环境条件，因此无法保证收敛。在集中式方法中，从所有智能体中学习到单一策略。以环境的当前状态作为输入，并以策略输出联合动作，该奖励是全局性的。

在对抗游戏中训练智能体非常复杂。一方面，需要找到一个训练有素的对手来与你的智能体对战。另一方面，如果找到了一个训练有素的对手，当对手太强大时，你的智能体将如何改进其策略？想象一下一个刚开始学习足球的孩子。与非常优秀的足球运动员比赛是毫无意义的，因为获胜或拿到球太难了。所以孩子会不断失败而没有时间学习好的策略。解决方法是**自我对弈**。在**自我对弈**中，智能体使用自己的先前副本（其策略）作为对手。这样，智能体将与同一级别的智能体对弈（具有挑战性但不会太难），有机会逐步改进其策略，然后随着对手变得更好而更新其策略。这是一种引导对手并逐步增加对手复杂性的方法。这与人类在竞争中学习的方式相同：开始与水平相当的对手进行训练，然后从中学习，当掌握了一些技能后，就可以与更强大的对手走得更远。

在对抗类游戏中，跟踪累积奖励并不总是一个有意义的指标：因为这个指标取决于对手的技能。使用`ELO`评级系统（以`Arpad Elo`命名），计算零和游戏中两名玩家之间的相对技能水平。**零和博弈**：一个智能体赢，另一个智能体输。这是一种数学表示，即每个参与者的效用收益或损失恰好与其他参与者的效用收益或损失相平衡。`ElO`系统是与其他玩家的输赢和打平情况得出的。这意味着玩家的评级取决于对手的评级和对手的得分结果。球员的表现被视为一个服从**正态分布**的随机变量。

#### 近端策略优化(PPO)

**近端策略优化**(`PPO`)是一种通过避免策略更新过大来提高智能体训练稳定性的架构。这样做可以保证我们的策略更新不会太大，并且训练更加稳定。训练期间较小的策略更新更有可能收敛到最优解。策略更新步调过大，很容易生成糟糕的策略，并且需要很长时间，甚至没有恢复的可能。为此，我们需要使用当前策略和上一个策略之间的比率来衡量当前策略与上一个策略相比的变化率。将这个比率限制在一个范围内{% mathjax %}[1-\epsilon,1 + \epsilon]{% endmathjax %}，这意味着我们消除了当前策略与旧策略相差太大的几率。

回顾移一下，**策略目标函数**，{% mathjax %}L^{PG}(\theta) = \mathbb{E}_t[\log\pi_{\theta}(a_t,s_t)\ast A_t]{% endmathjax %}，通过这个函数执行**梯度上升**，这将推动智能体采取更高回报的动作并避免有害动作。主要问题来自于步长：步长太小，训练过程太慢；步长太高，训练的变异性太大。它目的是利用`PPO`的目标函数来约束策略的更新，该函数使用`clip`将策略变化限制在一个小的范围内。旨在避免破坏性的大规模权重更新：
{% mathjax '{"conversion":{"em":14}}' %}
L^{\text{CLIP}}(\theta) = \mathbb{E}_t[\min(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t)]
{% endmathjax %}
其中{% mathjax %}r_t(\theta){% endmathjax %}是**比率函数**，**比率函数**的计算公式为：{% mathjax %}r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}{% endmathjax %}。它是在当前策略中在状态{% mathjax %}s_t{% endmathjax %}采取动作{% mathjax %}a_t{% endmathjax %}的概率，除以上一个策略的概率。我们可以看到，{% mathjax %}r_t(\theta){% endmathjax %}表示当前策略与旧策略之间的概率比：
- 如果{% mathjax %}r_t(\theta) > 1{% endmathjax %}，则在状态{% mathjax %}s_t{% endmathjax %}下的动作{% mathjax %}a_t{% endmathjax %}在当前策略中比旧策略中更有可能发生。
- 如果{% mathjax %}0 \leq r_t(\theta) \leq 1{% endmathjax %}，则当前策略采取该动作的可能性小于旧策略。

因此，这个概率比是预测旧策略和当前策略之间差异的简单方法。裁剪智能体的目标函数中未裁剪的部分：这个比率可以替代在**策略目标函数**中使用的对数概率。
{% mathjax '{"conversion":{"em":14}}' %}
L^{\text{CLIP}}(\theta) = \mathbb{\hat{E}}_t[\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}\hat{A}_t] = \mathbb{\hat{E}}_t[r_t(\theta)\hat{A}_t]
{% endmathjax %}
如果没有约束，当前策略中采取的动作比之前策略中采取的动作更有可能发生，这将导致显著的**策略梯度下降**，从而导致过度的策略更新。裁剪代理目标函数的裁剪部分，通过裁剪比率，确保不会有太大的策略更新，因为当前策略不能与旧策略有太大差别。为此，这里有`2`个方法：
- `TRPO`(`Trust Region Policy Optimization`)利用目标函数外的`KL`散度约束来约束策略更新，但该方法实现起来比较复杂，计算时间较长。
- `PPO`将**目标函数**中的裁剪概率比直接与其裁剪后的替代目标函数相加。

他的剪辑部分是{% mathjax %}r_t(\theta){% endmathjax %}被剪辑范围是在{% mathjax %}[1-\epsilon, 1+ \epsilon]{% endmathjax %}之间。使用裁剪替代**目标函数**，我们得到两个概率比，一个未裁剪，一个裁剪，范围介于{% mathjax %}[1-\epsilon, 1+ \epsilon]{% endmathjax %}之间，{% mathjax %}\epsilon{% endmathjax %}是一个超参数，可以帮助定义这个剪辑范围（在论文中{% mathjax %}\epsilon = 0.2{% endmathjax %}）。然后，取剪辑和未剪辑目标中的最小值，因此最终结果是未剪辑目标的下限。取剪辑和非剪辑目标中的最小值意味着将根据比率和优势情况选择剪辑或非剪辑目标。
{% asset_img ml_7.png %}

如上图所示，这里有`6`种情况：
- 情况`1`、`2`的概率比范围介于{% mathjax %}[1-\epsilon, 1+ \epsilon]{% endmathjax %}之间，在情况`1`中，具有积极优势：该动作优于该状态下所有动作的平均值。因此，我们应该激励当前策略增加在该状态下采取该动作的概率；在情况`2`中，具有负面优势：该动作比该状态下所有动作的平均值更差。因此，我们应该降低当前策略在该状态下采取该动作的概率。
- 情况`3`、`4`的概率比低于{% mathjax %}1-\epsilon{% endmathjax %}范围，如果概率比{% mathjax %}1-\epsilon{% endmathjax %}低，则在该状态下采取该动作的概率比旧策略低。在情况`3`中，优势预测为正({% mathjax %} A > 0{% endmathjax %})，那么应该增加在该状态下采取该动作的概率；在情况`4`中，优势预测为负，为了提升该状态下采取该动作的概率。由于{% mathjax %}\text{Gradient} = 0{% endmathjax %}（因为在一条水平线上），所以不用更新权重。
- 情况`5`、`6`：概率比高于{% mathjax %}1+ \epsilon{% endmathjax %}范围，如果概率比高于{% mathjax %}1+ \epsilon{% endmathjax %}，则当前策略中该状态下采取该动作的概率远高于上一个策略。在情况`5`中，优势为正，与上一个策略相比，在该状态下采取该行动的概率已经很高，由于{% mathjax %}\text{Gradient} = 0{% endmathjax %}（因为在一条水平线上），所以不用更新权重；在情况`6`中，优势为负，希望降低在该状态下采取该动作的概率。

使用未裁剪的目标部分来更新策略。当最小值是裁剪的目标部分时，这里不会更新策略权重，因为梯度将等于`0`。因此，仅会在以下情况下更新策略：
- 概率比在{% mathjax %}[1-\epsilon, 1+ \epsilon]{% endmathjax %}范围内的时候。
- 概率比超出了{% mathjax %}[1-\epsilon, 1+ \epsilon]{% endmathjax %}范围，但是概率比小于{% mathjax %}1-\epsilon{% endmathjax %}且{% mathjax %}A > 0{% endmathjax %}，或概率比大于{% mathjax %}1+ \epsilon{% endmathjax %}且{% mathjax %}A < 0{% endmathjax %}。

你可能会想，为什么当最小值为截断比时，梯度为`0`。当概率比被截断时，在这种情况下，导数将不是{% mathjax %}r_t(\theta)\ast A_t{% endmathjax %}​的导数，而是{% mathjax %}(1 - \epsilon)\ast A_t{% endmathjax %}​或{% mathjax %}(1 + \epsilon)\ast A_t{% endmathjax %}​的导数，其中两者都等于`0`。`PPO Actor-Critic`的最终裁剪替代了目标损失，它利用裁剪替代**目标函数**、**价值损失函数**和**熵奖励**的组合：
{% mathjax '{"conversion":{"em":14}}' %}
L_t^{\text{CLIP+VF+S}}(\theta) = \hat{\mathbb{E}}_t[L_t^{\text{CLIP}}(\theta) - c_1L_t^{\text{VF}}(\theta) + c_2 S[\pi_{\theta}](s_t)]
{% endmathjax %}
其中{% mathjax %}c_1,c_2{% endmathjax %}是系数，{% mathjax %}L_t^{\text{VF}}(\theta){% endmathjax %}是价值损失函数，{% mathjax %}S[\pi_{\theta}](s_t){% endmathjax %}是熵奖励函数，为了确保其充分的探索。

`Sample Factory`是最快的**强化学习**(`RL`)库之一，专注于同步和异步**策略梯度**(`PPO`)实现。`Sample Factory`具有的相关特性：
- 高度优化的算法架构，实现学习的最大吞吐量。
- 支持同步和异步训练机制。
- 支持串行（单进程）模式，方便调试。
- 在基`CPU`和`GPU`加速的环境下均实现了最优性能。
- 支持**单智能体**和**多智能体**训练，**自我对弈**，同时支持在一个或多个`GPU`上同时训练多个策略。
- 支持`PBT`(`Population-Based Training`)模型训练方法。
- 支持离散、连续、混合动作空间。
- 支持基于矢量、基于图像、字典观察空间。
- 通过解析动作/观察空间规范自动创建模型架构。支持自定义模型架构。
- 支持自定义环境导入工程。
- 详细的`WandB`和`Tensorboard`摘要、自定义指标。
- 整合了多个示例（调整参数和训练模型）并与环境集成。

`Sample Factory`工作原理，如下图所示：
{% asset_img ml_8.png %}

`Sample Factory`的工作原理是由多个执行工作者(`rollout workers`)、推理工作者(`inference workers`)和一个学习工作者(`learner worker`)进程构成。工作者进程之间通过**共享内存**进行通信，从而降低了进程之间的通信成本。执行工作者(`rollout workers`)与环境进行互动，并将观察结果发送给推理工作者(`inference workers`)。推理工作者查询策略的指定版本，并将动作发送回执行工作者(`rollout workers`)。经过k步之后，执行工作者(`rollout workers`)会将经验轨迹发送给学习工作者(`learner worker`)，以便学习工作者(`learner worker`)能够更新智能体的**策略网络**。

`Sample Factory`中的`Actor Critic`模型由`3`个组件构成：
- **编码器**(`encoder`)：处理输入观察值（图像、矢量）并将其映射到矢量，这部分可以自定义。
- **核心**(`core`)：整合一个或多个编码器的向量，可以选择在基于内存的智能体中包含单层或多层`LSTM/GRU`。
- **解码器**(`decoder`)：在计算策略和值输出之前，将附加层应用于模型**核心**(`core`)的输出。

#### 有模型的强化学习(MBRL)

**有模型的强化学习**(`MBRL`)与**无模型的强化学习**仅在学习动态模型方面有所不同，但这对于决策的制定方式有着重大的影响。**动态模型**通常对环境转移动态进行建模，{% mathjax %}s_{t+1} = f_{\theta}(s_t,a_t){% endmathjax %}，但**逆动态模型**（从状态到动作的映射）或**奖励模型**（预测奖励）都可以在这个框架中使用。`MBRL`定义：
- 有一个智能体反复尝试解决一个问题，积累状态和动作数据。
- 利用这些数据，智能体可以创建一个结构化的学习工具，即**动态模型**，用来推理真实环境。
- 通过动态模型，智能体可以通过预测接下来采取的动作。
- 通过这些动作，智能体可以收集更多数据，改进所述模型，并且改进接下来的动作。

**有模型的强化学习**(`MBRL`)采用智能体在环境交互中的框架，学习**环境模型**，然后利用该模型进行决策。具体来说，智能体在转换函数{% mathjax %}s_{t+1} = f(s_t,a_t){% endmathjax %}控制的**马尔可夫决策过程** (`MDP`)中执行，并在每一步中返回奖励{% mathjax %}r(s_t,a_t){% endmathjax %}。通过收集的数据集{% mathjax %}D:= s_i,a_i,s_{i+1},r_i{% endmathjax %}，智能体学习一个模型，{% mathjax %}s_{t+1} = f_{\theta}(s_t,a_t){% endmathjax %}是以最小化转换的负对数似然。这里采用基于样本的模型预测控制(`MPC`)，使用学习到的**动态模型**，它从一组从均匀分布抽样的动作{% mathjax %}U(a){% endmathjax %}中，优化有限、递归预测范围{% mathjax %}\tau{% endmathjax %}内的预期奖励。

#### 离线&在线强化学习

**深度强化学习**(`DRL`)是一个构建**智能体**(`Agent`)决策的框架。这些**智能体**(`Agent`)旨在通过反复试验与环境互动并获得奖励作为反馈来学习最佳策略。**智能体**(`Agent`)的目标是最大化其累积奖励，称为**回报**。因为**强化学习**(`RL`)基于**奖励假设**：所有目标都可以描述为最大化的预期累积奖励。**深度强化学习**(`DRL`)**智能体**(`Agent`)通过批量经验学习。关键是如何收集这些经验？
- 在在线强化学习中，智能体直接收集数据：它通过与环境交互来收集一批经验，然后，从这批经验（或通过**重放缓冲区**）中学习（更新其策略）。意味着要么直接在现实世界中训练智能体，要么使用模拟器。如果没有，就需要构建它，这可能非常复杂（如何在环境中反映真实世界的复杂性）且不安全。
- 在离线强化学习中，智能体仅从其他智能体或人类演示中收集数据，且不与环境进行交互。流程如下：使用一个或多个策略且人机交互生成数据集，在该数据集上运行**离线强化学习算法**来学习策略。这种方法有一个缺点：**反事实查询问题**。如果我们的智能体决定做某件事，而我们没有相关数据，该怎么办？

**马尔可夫决策过程**(`MDP`)：**马尔可夫决策过程**(`MDP`)定义为一个元组{% mathjax %}\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, d0, r, \gamma){% endmathjax %}，其中{% mathjax %}\mathcal{S}{% endmathjax %}是一组状态且{% mathjax %}s\in \mathcal{S}{% endmathjax %}，可以是离散的也可以是连续的（即**多维向量**），{% mathjax %}\mathcal{A}{% endmathjax %}是一组动作且{% mathjax %}a\in\mathcal{A}{% endmathjax %}，同样可以是离散的或连续的，{% mathjax %}T{% endmathjax %}是一个条件概率分布，形式为{% mathjax %}T(s_{t+1},|s_t,a_t){% endmathjax %}，用于描述系统的动态，{% mathjax %}d_0{% endmathjax %}是初始状态分布，形式为{% mathjax %}d_0(s_0){% endmathjax %}，{% mathjax %}r\;:\;\mathcal{S}\times \mathcal{A}\rightarrow \mathbb{R}{% endmathjax %}是奖励函数，{% mathjax %}\gamma \in (0,1]{% endmathjax %}是值为标量的**折扣因子**。

**部分可观察的马尔可夫决策过程**：**部分可观察的马尔可夫决策过程**定义为一个元组{% mathjax %}\mathcal{M} = (\mathcal{S}, \mathcal{A},\mathcal{O},T, d0, E,r, \gamma){% endmathjax %}，其中{% mathjax %}\mathcal{S}, \mathcal{A}, T, d0, r, \gamma{% endmathjax %}的定义与之前相同，{% mathjax %}\mathcal{O}{% endmathjax %}是一组观测值，其中每个观测值由{% mathjax %}o\in \mathcal{O}{% endmathjax %}给出，{% mathjax %}E{% endmathjax %}是一个**发射函数**，它定义了分布{% mathjax %}E(o_t|s_t){% endmathjax %}。**强化学习**问题的最终目标是学习一种策略，该策略定义了以状态为条件的动作的分布{% mathjax %}\pi(a_t|s_t){% endmathjax %}，或以部分观察设置中的观察为条件的动作分布{% mathjax %}\pi(a_t|o_t){% endmathjax %}。该策略还可以以观察历史为条件，{% mathjax %}\pi(a_t|o_0:t){% endmathjax %}。从这些定义中，我们可以得出**轨迹分布**。轨迹是长度为{% mathjax %}H{% endmathjax %}的状态和动作序列，由{% mathjax %}\tau = (s_0,a_0,\ldots,s_H,a_H){% endmathjax %}给出，其中{% mathjax %}H{% endmathjax %}可能是无限大。给定**马尔可夫决策过程**(`MDP`){% mathjax %}\mathcal{M}{% endmathjax %}和策略{% mathjax %}\pi{% endmathjax %}的轨迹分布{% mathjax %}p_{\pi}{% endmathjax %}由以下公式给出：
{% mathjax '{"conversion":{"em":14}}' %}
p_{pi}(\tau) = d_0(s_0)\prod\limits^{H}_{t=0}\pi(a_t|s_t)T(s_{t+1}|s_t,a_t)
{% endmathjax %}
其中包含观测值{% mathjax %}o_t{% endmathjax %}和**发射函数**{% mathjax %}E(o_t|s_t){% endmathjax %}，此定义可轻松扩展到部分观察。**强化学习**目标{% mathjax %}J(\pi){% endmathjax %}可写为该轨迹分布下的期望值：
{% mathjax '{"conversion":{"em":14}}' %}
J(\pi) = \mathbb{E}_{\tau\sim p_{\pi}(\tau)}\bigg[\sum\limits_{t=0}^{H}\gamma^t r(s_t,a_t)\bigg]
{% endmathjax %}
当{% mathjax %}H{% endmathjax %}为无穷大时，有时也可以方便地假设由{% mathjax %}\pi(a_t|s_t)T(S_{t+1}|s_t,a_t){% endmathjax %}定义的在{% mathjax %}(s_t,a_t){% endmathjax %}上的**马尔可夫链**是可以遍历的，并根据该**马尔可夫链**的预期奖励来定义目标。由于**折扣因子**的作用。在许多情况下，我们会发现引用轨迹分布{% mathjax %}p_{\pi}(\tau){% endmathjax %}的边际很方便。我们将使用{% mathjax %}d^{\pi}(s){% endmathjax %}来指代按时间步总体状态平均访问频率，并使用{% mathjax %}d^{\pi}_t(s_t){% endmathjax %}来指代时间步{% mathjax %}t{% endmathjax %}的状态访问频率。所有标准**强化学习算法**都遵循相同的学习规律：**智能体**通过使用某种行为策略与**马尔可夫决策过程**(`MDP`){% mathjax %}\mathcal{M}{% endmathjax %}进行交互，该策略可能与{% mathjax %}\pi(a|s){% endmathjax %}匹配也可能不匹配，通过观察当前状态{% mathjax %}s_t{% endmathjax %}、选择一个动作{% mathjax %}a_t{% endmathjax %}，然后观察产生的下一个状态{% mathjax %}s_{t+1}{% endmathjax %}和奖励值{% mathjax %}r_t = r(s_t,a_t){% endmathjax %}。这可能会重复多步，然后**智能体**使用观察到的转换{% mathjax %}(s_t,a_t,s_{t+1},r_t){% endmathjax %}来更新其策略。此更新也可能利用先前观察到的转换。我们将使用{% mathjax %}\mathcal{D}{% endmathjax %}来表示**智能体**可用于更新策略（“**学习**”）的**转换集**，该**转换集**可能由所有转换（迄今为止看到的）或其中的某个子集组成。

**策略梯度**：**强化学习**(`RL`)目标的最直接方法之一是直接估计其梯度。在这种情况下，我们通常假设策略由参数向量{% mathjax %}\theta{% endmathjax %}参数化，因此由{% mathjax %}\pi_{\theta}(a_t|s_t){% endmathjax %}πθ(at|st) 给出。例如，{% mathjax %}\theta{% endmathjax %}可能表示输出 (离散) 动作{% mathjax %}a_t{% endmathjax %}对数的深度网络的权重。在这种情况下，我们可以将目标相对于{% mathjax %}\theta{% endmathjax %}的梯度表示为：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla_{\theta}J(\pi_{\theta}) = \mathbb{E}_{\tau\sim p_{\pi_{\theta}}}(\tau)\bigg[\sum\limits_{t=0}^H \gamma^t\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\underbrace{\bigg( \sum\limits_{t'=t}^H \gamma^{t'-t} r(s_{t'},a_{t'} - b(s_t)) \bigg)}_{\text{return estimate}\;\hat{A}(s_t,a_t)}  \bigg]
{% endmathjax %}
其中**回报预测器**{% mathjax %}\hat{A}(s_t,a_t){% endmathjax %}本身可以作为单独的神经网络进行学习，或者可以简单地用**蒙特卡洛样本**进行预测，在这种情况下，我们只需从{% mathjax %}p_{\pi_{\theta}}(\tau){% endmathjax %}生成样本，然后对采样轨迹的时间步中的奖励进行汇总。基线{% mathjax %}b(s_t){% endmathjax %}可以预测为采样轨迹的平均奖励，或者使用**价值函数预测器**{% mathjax %}V(s_t){% endmathjax %}。可以等效地将此梯度表达式写为对{% mathjax %}d^{\pi}(s){% endmathjax %}的期望，如下所示：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla_{\theta}J(\pi_{\theta}) = \sum_{t=0}^H \mathbb{E}_{s_t\sim d^{\pi}_t(s_t),a_t\sim\pi_{\theta}(a_t,s_t)}\bigg[\gamma^t\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)\hat{A}(s_t,a_t) \bigg]
{% endmathjax %}
一种常见的修改是删除梯度前面的{% mathjax %}\gamma^{t}{% endmathjax %}项，这近似于平均奖励设置。删除该项并采用无限期公式，可以进一步将**策略梯度**重写为{% mathjax %}d^{\pi}(s){% endmathjax %}下的期望，如下所示：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla_{\theta}J(\pi_{\theta}) = \frac{1}{1 - \gamma}\mathbb{E}_{s\sim d^{\pi}(s_t),a\sim\pi_{\theta}(a,s)}\bigg[\nabla_{\theta}\log \pi_{\theta}(a|s)\hat{A}(s,a) \bigg]
{% endmathjax %}
其中常数缩放项{% mathjax %}\frac{1}{1-\gamma}{% endmathjax %}经常被忽略。这种无限期公式通常便于分析和推导**策略梯度**方法。最后可以总结出一个**蒙特卡洛策略梯度算法**，如下所示：
<embed src="algorithm.pdf" type="application/pdf" width="100%" height="200">

**近似动态规划**(`Approximate dynamic programming`)。优化**强化学习**目标的另一种方法是**观察**，如果我们能够准确估计状态或状态-动作对的**价值函数**，那么很容易接近最优策略。**价值函数**提供了预期累积奖励的预测，当给定状态{% mathjax %}s_t{% endmathjax %}，在状态值函数{% mathjax %}V^{\pi}(s_t){% endmathjax %}的情况下，或者当给定状态-动作对元组{% mathjax %}(s_t,a_t){% endmathjax %}，在状态-动作对的**价值函数**{% mathjax %}Q^{\pi}(s_t,a_t){% endmathjax %}的情况下，遵循一些策略{% mathjax %}\pi(s_t,a_t){% endmathjax %}获得奖励，可以将这些**价值函数**定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
V^{\pi}(s_t) & = \mathbb{E}_{\tau\sim p_{\pi}(\tau|s_t)}\bigg[\sum\limits_{t'=t}^H \gamma^{t' - t}r(s_t,a_t) \bigg] \\
Q^{\pi}(s_t,a_t) & = \mathbb{E}_{\tau\sim p_{\pi}}(\tau|s_t,a_t)\bigg[\sum\limits_{t'=t}^H \gamma^{t' - t}r(s_t,a_t) \bigg]
\end{align}
{% endmathjax %}
由此，可以推导出这些**价值函数**的递归定义，其形式为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
V^{\pi}(s_t) & = \mathbb{E}_{a_t\sim p_{\pi}}(\tau|a_t|s_t)\bigg[Q^{\pi}(s_t,a_t) \bigg] \\
Q^{\pi}(s_t,a_t) & = r(s_t,a_t) + \gamma\mathbb{E}_{\tau\sim T(s_{t+1}|s_t,a_t)}\bigg[V^{\pi}(s_{t+1}) \bigg]
\end{align}
{% endmathjax %}
我们可以把这两个方程结合起来，用{% mathjax %}Q^{\pi}(s_{t+1},a_{t+1}){% endmathjax %}来表示{% mathjax %}Q^{\pi}(s_t,a_t){% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
Q^{\pi}(s_t,a_t) = r(s_t,a_t) + \gamma\mathbb{E}_{\tau\sim T(s_{t+1}|s_t,a_t),a_{t+1}\sim\pi(a_{t+1}|s_{t+1})}\bigg[Q^{\pi}(s_{t+1},a_{t+1}) \bigg]
{% endmathjax %}
可以用策略{% mathjax %}\pi{% endmathjax %}的**贝尔曼算子**来表示，将其表示为{% mathjax %}\mathcal{B}^{\pi}{% endmathjax %}。例如，公式{% mathjax %}Q^{\pi}(s_t,a_t) = r(s_t,a_t) + \gamma\mathbb{E}_{\tau\sim T(s_{t+1}|s_t,a_t),a_{t+1}\sim\pi(a_{t+1}|s_{t+1})}\bigg[Q^{\pi}(s_{t+1},a_{t+1}) \bigg]{% endmathjax %}可以写成{% mathjax %}\vec{Q}^{\pi} = \mathcal{B}^{\pi}\vec{Q}^{\pi}{% endmathjax %}，其中{% mathjax %}\vec{Q^{\pi}}{% endmathjax %}表示以长度为{% mathjax %}|\mathcal{S}|\times|\mathcal{A}|{% endmathjax %}的向量表示的`Q`函数{% mathjax %}Q^{\pi}{% endmathjax %}。在继续基于这些定义推导学习算法之前，需要讨论一下**贝尔曼算子**的一些性质。这个**贝尔曼算子**有一个唯一的不动点，它对应于策略{% mathjax %}\pi(a|s){% endmathjax %}的真实`Q`函数，可以通过重复迭代{% mathjax %}\vec{Q}^{\pi}_{k+1} = \mathcal{B}^{\pi}\vec{Q}^{\pi}_k{% endmathjax %}来获得，并且可以证明{% mathjax %}\lim_{k\rightarrow \infty}\vec{Q}^{\pi}_k = \vec{Q}^{\pi}{% endmathjax %}，对此的证明来自以下观察：{% mathjax %}\mathcal{B}^{\pi}{% endmathjax %}是{% mathjax %}\mathcal{\ell}_{\infty}{% endmathjax %}范数的一个收缩。基于这些定义，我们可以推导出两种常用的基于**动态规划**的算法：`Q-Learning`和`actor-critic`方法。为了推导出`Q-Learning`，我们用`Q`函数隐式地表示策略，如{% mathjax %}\pi(a_t|s_t) = \delta(a_t = \text{arg}\;\max Q(s_t,a_t)){% endmathjax %}，并且只学习`Q`函数{% mathjax %}Q(s_t,a_t){% endmathjax %}。通过将此（隐式）策略代入上述动态规划方程，就获得了最佳`Q`函数：
{% mathjax '{"conversion":{"em":14}}' %}
Q^{*}(s_t,a_t) = r(s_t,a_t) + \gamma\mathbb{E}_{s_{t+1}\sim T(s_{t+1}|s_t,a_t)}\bigg[\underset{a_{t+1}}{\max}Q^{*}(s_{t+1},a_{t+1}) \bigg]
{% endmathjax %}
我们可以再次用向量符号将其表示为{% mathjax %}\vec{Q} = \mathcal{B}^{*}\vec{Q}{% endmathjax %}，其中{% mathjax %}\mathcal{B}^{*}{% endmathjax %}指的是**贝尔曼最优算子**。但请注意，由于以上公式右侧的最大化，该算子不是线性的。为了将该等式转换为学习算法，我们可以最小化该等式左侧和右侧相对于参数为{% mathjax %}\phi,Q_{\phi}(s_t,a_t){% endmathjax %}的参数`Q`函数预测器的参数之间的差异。此`Q-Learning`过程有许多变体，包括在每次迭代中完全最小化上述方程左侧和右侧之间差异的变体，通常称为拟合`Q`迭代，以及采用单个梯度步骤的变体，例如原始`Q-Learning`方法。**深度强化学习**中常用的变体是这两种方法的混合体，采用**重放缓冲区**并在数据收集的同时对**贝尔曼误差**目标采取梯度步骤。经典`Q-Learning`可以推导出缓冲区大小为`1`的极限情况，我们采用{% mathjax %}G = 1{% endmathjax %}梯度步骤并每次迭代收集{% mathjax %}S = 1{% endmathjax %}个过渡样本，而经典拟合`Q`迭代运行内部梯度下降阶段收敛（即{% mathjax %}G = \infty{% endmathjax %}），并使用等于采样步骤数{% mathjax %}S{% endmathjax %}的缓冲区大小。请注意，许多当前实现还采用目标网络，其中目标值{% mathjax %}r_i + \gamma\max_{a'}Q{\phi k}(s',a'){% endmathjax %}实际上使用{% mathjax %}\phi{% endmathjax %}，其中{% mathjax %}L{% endmathjax %}是滞后迭代。请注意，这些近似值违反了`Q-Learning`算法可以证明收敛的假设。然而，最近的研究表明，对应于非常大的`Q`集的高容量函数近似器通常确实倾向于使这种方法在实践中收敛，从而产生接近{% mathjax %}\vec{Q}^{\pi}{% endmathjax %}的`Q`函数。
<embed src="algorithm_1.pdf" type="application/pdf" width="100%" height="200">

`actor-critic`算法。`actor-critic`算法结合了**策略梯度**和**近似动态规划**的基本思想。此类算法同时采用**参数化策略**和**参数化值函数**，并使用值函数为**策略梯度**计算提供更好的{% mathjax %}\hat{A}(s,a){% endmathjax %}预测。`actor-critic`方法有许多不同的变体，包括直接预测{% mathjax %}V^{\pi}(s){% endmathjax %}的在线策略变体和通过参数化状态-动作对值函数{% mathjax %}Q^{\pi}_{\phi}(s,a){% endmathjax %}预测{% mathjax %}Q^{\pi}(s,a){% endmathjax %}的**离线策略变体**。这种算法的基本设计是**动态规划**和**策略梯度**思想的直接组合。与直接尝试学习最优`Q`函数的`Q-Learning`不同，`actor-critic`方法旨在**学习**与当前**参数化策略**{% mathjax %}\pi_{\theta}(a|s){% endmathjax %}相对应的`Q`函数，该函数必须遵循以下公式：
{% mathjax '{"conversion":{"em":14}}' %}
Q^{\pi}(s_t,a_t) = r(s_t,a_t) + \gamma\mathbb{E}_{s_{t+1}\sim T(s_{t+1}|s_t,a_t),a_{t+1}\sim\pi_{\theta}(a_{t+1}|s_{t+1})}\bigg[Q^{\pi}(s_{t+1},a_{t+1}) \bigg]
{% endmathjax %}
与之前一样，该公式可以用策略的**贝尔曼算子**{% mathjax %}\vec{Q}^{\pi} = \mathcal{B}\vec{Q}^{\pi}{% endmathjax %}的向量表示，其中{% mathjax %}\vec{Q}^{\pi}{% endmathjax %}表示`Q`函数{% mathjax %}Q^{\pi}{% endmathjax %}，表示长度为{% mathjax %}|\mathcal{S}|\times|\mathcal{A}|{% endmathjax %}的向量。现在可以根据这个想法实例化一个完整的算法。`Actor Critic`算法与**动态规划**中经常出现的另一类方法密切相关，称为**策略迭代**(`PI`)。**策略迭代**包括两个阶段：**策略评估**和**策略改进**。**策略评估阶段**通过求解固定点（使得{% mathjax %}Q^{\pi} = \mathcal{B}Q^{\pi}{% endmathjax %}）来计算当前策略{% mathjax %}\pi{% endmathjax %}的`Q`函数{% mathjax %}Q^{\pi}{% endmathjax %}。这可以通过线性规划或求解线性方程组来完成，或者通过梯度更新来完成。然后在**策略改进阶段**计算下一个**策略迭代**，通过选择在每个状态下最大化`Q`值的贪婪动作，使得{% mathjax %}\pi_{k+1}(a|s) = \delta(a= \text{arg}\;\underset{a}{\max}Q^{\pi k}(s,a)){% endmathjax %}，或者使用基于梯度的更新程序。在没有函数近似（例如，使用表格表示）的情况下，**策略迭代**会产生单调改进的**策略序列**，并收敛到**最佳策略**。当我们设置{% mathjax %}G_Q = \infty{% endmathjax %}和{% mathjax %}G_{\pi} = \infty{% endmathjax %}时，当缓冲区{% mathjax %}\mathcal{D}{% endmathjax %}由`MDP`的每个转换组成时，**策略迭代**可以作为`Actor Critic`算法的一个特例获得。

**有模型强化学习**：**有模型强化学习**是一个通用术语，指利用转换或动态函数{% mathjax %}T(s_{t+1}|s_t,a_t){% endmathjax %}显式预测的一类方法，该函数由参数向量{% mathjax %}\psi{% endmathjax %}参数化，将其表示为{% mathjax %}T_{\psi}(s_{t+1}|s_t,a_t){% endmathjax %}。**有模型强化学习**方法没有单一的配方。一些常用的**有模型强化学习**算法仅学习动态模型{% mathjax %}T_{\psi}(s_{t+1}|s_t,a_t){% endmathjax %}，然后在测试时利用它进行规划，通常通过**模型预测控制**(`MPC`)和各种轨迹优化方法。其他**有模型强化学习**方法除了动态模型外，还利用学习到的策略{% mathjax %}\pi_{\theta}(a_t|s_t){% endmathjax %}，并采用**时间反向传播**来优化策略，以实现预期的奖励目标。还有一组算法使用该模型生成“合成”样本，以扩充**无模型强化学习**方法可用的样本集。经典的`Dyna`算法将此方法与`Q-Learning`和通过模型从先前看到的状态进行的一步预测相结合，而最近提出的各种算法采用基于合成模型的**策略梯度**和`Actor Critic`算法。
<embed src="algorithm_2.pdf" type="application/pdf" width="100%" height="200">

**离线强化学习**问题可以定义为数据驱动的强化学习问题。最终目标是优化方程。然而，**智能体**与环境交互并使用行为策略不再具有收集转换的能力。相反，学习算法提供了一个静态转换数据集{% mathjax %}\mathcal{D}=\{s^i_t,a^i_t,s^i_{t+1},r^i_t\}{% endmathjax %}，并且必须使用该数据集学习**最佳策略**。这个公式更接近标准的监督学习问题，可以将{% mathjax %}\mathcal{D}{% endmathjax %}视为策略的训练集。本质上，**离线强化学习**要求学习算法完全从固定数据集中充分了解`MDP`{% mathjax %}\mathcal{M}{% endmathjax %}背后的动态系统，然后构建一个策略{% mathjax %}\pi(a|s){% endmathjax %}，当它用于与`MDP`交互时，该策略可获得最大的累积奖励。我们将使用{% mathjax %}\pi_{\beta}{% endmathjax %}来表示{% mathjax %}\mathcal{D}{% endmathjax %}中状态和动作的分布，假设状态-动作对元组{% mathjax %}(s,a)\in \mathcal{D}{% endmathjax %}根据{% mathjax %}s\sim d^{\pi_{\beta}}(s){% endmathjax %}进行采样，并根据行为策略对动作进行采样，使得{% mathjax %}a\sim \pi_{\beta}(a|s){% endmathjax %}。**离线策略强化学习**表示所有可以使用转换{% mathjax %}\mathcal{D}{% endmathjax %}数据集的**强化学习算法**，其中每个转换中的相应动作都是使用除当前策略{% mathjax %}\pi(a|s){% endmathjax %}之外的任何策略收集的。`Q-Learning`算法、利用`Q`函数的`Actor Critic`算法和许多**有模型强化学习算法**都是**离线策略算法**。然而，**离线​​策略算法**在学习过程中仍然经常使用额外的交互（即在线数据收集）。因此，术语“**完全离线策略**”有时用于表示不执行额外的在线数据收集。另一个常用术语是“**批量强化学习**”。因为在迭代学习算法中使用“批量”也可以指一种使用一批数据、更新模型，然后获得不同批次的方法，而不是传统的在线学习算法，后者一次使用一个样本。原则上任何**离线策略强化学习算法**都可以用作**离线强化学习算法**。例如，只需使用`Q-Learning`而无需额外的在线探索，使用{% mathjax %}\mathcal{D}{% endmathjax %}预填充数据缓冲区，即可获得简单的**离线强化学习方法**。

采用重要性抽样来预测{% mathjax %}J(\pi){% endmathjax %}，其中轨迹从{% mathjax %}\pi_{\beta}(\tau){% endmathjax %}中采样。这被称为**离线策略评估**。原则上，一旦能够评估{% mathjax %}J(\pi){% endmathjax %}，就可以选择性能最佳的策略。可以使用**重要性抽样**来推导出**离线策略轨迹**的{% mathjax %}J(\pi){% endmathjax %}无偏估计量。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
J(\pi_{\theta}) & = \mathbb{E}_{\tau\sim\pi_{\beta}(\tau)}\bigg[\frac{\pi_{\theta}(\tau)}{\pi_{\beta}(\tau)}\sum\limits_{t=0}^H \gamma^t r(s,a) \bigg] \\
& = \mathbb{E}_{\tau\sim\pi_{\beta}(\tau)}\bigg[\bigg(\prod_{t=0}^H\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\beta}(a_t|s_t)}\bigg)\sum\limits_{t=0}^H \gamma^t r(s,a) \bigg] \approx \sum\limits_{i=1}^n w^i_H \sum\limits_{t=0}^H \gamma^t r^i_t
\end{align}
{% endmathjax %}
其中{% mathjax %}w^i_t = \frac{1}{n}\prod_{t'=0}^t \frac{\pi_{\theta}(a_{t'}|s_{t'})}{\pi_{\beta}(a_{t'}|s_{t'})}{% endmathjax %}和{% mathjax %}\{(s^i_0,a^i_0,r^i_0,s^i_1,\ldots)\}^n_{i=1}{% endmathjax %}是来自{% mathjax %}\pi_{\beta}(\tau){% endmathjax %}的{% mathjax %}n{% endmathjax %}个轨迹样本。不幸的是，由于重要性权重的乘积，这种**估计量**可能具有非常高的方差（如果 {% mathjax %}H{% endmathjax %}是无穷大，则可能无界）。对**重要性权重**进行自正则化（即，将权重除以{% mathjax %}\sum\limits_{i=1}^n w^i_H{% endmathjax %}）会产生加权**重要性抽样**估计量，该估计量有偏差，但方差可以低得多，并且仍然是一个强一致性估计量。为了改进这个估计量，我们需要利用问题的统计结构。因为{% mathjax %}r_t{% endmathjax %}不依赖于{% mathjax %}s_{t'}{% endmathjax %}和{% mathjax %}a_{t'}{% endmathjax %}且{% mathjax %}t' > t{% endmathjax %}，我们可以从未来时间步中删除**重要性权重**，从而得到每个决策**重要性抽样**估计量：
{% mathjax '{"conversion":{"em":14}}' %}
J(\pi_{\theta}) = \mathbb{E}_{\tau\sim\pi_{\beta}(\tau)}\bigg[\bigg(\prod_{t=0}^H\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\beta}(a_t|s_t)}\bigg)\sum\limits_{t=0}^H \gamma^t r(s,a) \bigg] \approx \frac{1}{n}\sum\limits_{i=1}^n\sum\limits_{t=0}^H w^i_t\gamma^t r^i_t
{% endmathjax %}
与之前一样，该估计量具有高方差，通过对权重进行**归一化**来形成加权的每个决策**重要性估计量**。不幸的是，在许多实际问题中，加权的每个决策**重要性估计量**仍然具有很大的方差，无法发挥作用。如果有一个近似模型，可用于获得每个状态-动作对元组{% mathjax %}(s_t,a_t){% endmathjax %}的{% mathjax %}Q{% endmathjax %}值的近似值，将其表示为{% mathjax %}Q^{\pi}(s_t,a_t){% endmathjax %}，将其合并到此估计中。例如，可以通过预测`MDP`转移概率{% mathjax %}T(s_{t+1}|s_t,a_t){% endmathjax %}的模型，然后求解相应的`Q`函数，或通过其他近似`Q`值的方法获得这样的估计。将这些估计作为控制变量合并到**重要性采样**估计量中，以获得两者的最佳效果：
{% mathjax '{"conversion":{"em":14}}' %}
J(\pi_{\theta}) \approx \sum\limits_{i=1}^n\sum\limits_{t=0}^H \gamma^t (w^i_t(r^i_t - \hat{Q}^{\pi_{\theta}}(s_t,a_t)) - w^i_{t-1}\mathbb{E}_{a\sim\pi_{\theta}(a|s_t)}[\hat{Q}^{\pi_{\theta}}(s_t,a)])
{% endmathjax %}
这被称为**双重稳定估计量**，因为如果{% mathjax %}\pi_{\beta}{% endmathjax %}已知或模型正确，它就是无偏的。可以通过对权重进行**归一化**来形成加权版本。通过利用要评估的策略知识训练模型，以及通过优化权衡偏差和方差，形成更复杂的估计量。除了**一致性**或**无偏估计**之外，经常希望对策略的性能有保证。基于**浓度不等式**得出了**置信界限**，专门用于处理**重要性加权估计量**的高方差和可能的范围。或者，根据分布假设（例如，渐近正态性）或通过引导构建**置信界限**。此类估计量还可用于策略改进，通过搜索与估计回报有关的策略。在**离线强化学习**的应用中，希望改进**行为策略**。可以使用**重要性抽样估计量**的较低**置信界限**来搜索策略，确保满足安全约束。

**重要性抽样**还可用于直接预测**策略梯度**，而不仅仅是获得给定策略值的估计值。**策略梯度方法**旨在通过计算策略参数的梯度预测来优化{% mathjax %}J(\pi){% endmathjax %}。我们可以使用**蒙特卡洛样本**预测梯度，但这需要**在线策略轨迹**（即{% mathjax %}\tau\sim \pi_{\theta}(\tau){% endmathjax %}）。在这里，我们将这种方法扩展到离线设置。以前的工作通常集中在离线策略设置上，其中轨迹是从不同的**行为策略**{% mathjax %}\pi_{\beta}(a|s){% endmathjax %}中采样的。然而，与离线设置相比，这种方法假设可以不断从{% mathjax %}\pi_{\beta}{% endmathjax %}中采样新轨迹，而旧轨迹则被重新使用以提高效率。注意{% mathjax %}J(\pi){% endmathjax %}和**策略梯度**之间结构相似，可以将预测{% mathjax %}J(\pi){% endmathjax %}**离线策略**的技术应用于**策略梯度**。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\nabla_{\theta}J(\pi_{\theta}) & = \mathbb{E}_{\tau\sim\pi_{\beta}(\tau)}\bigg[\frac{\pi_{\theta}(\tau)}{\pi_{\beta}(\tau)}\sum\limits_{t=0}^H \gamma^t \nabla_{\theta}\log \pi_{\theta}(a_t,s_t)\hat{A}(s_t,a_t) \bigg] \\
& = \mathbb{E}_{\tau\sim\pi_{\beta}(\tau)}\bigg[\bigg(\prod_{t=0}^H\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\beta}(a_t|s_t)}\bigg)\sum\limits_{t=0}^H \gamma^t \nabla_{\theta}\log \pi_{\theta}(a_t,s_t)\hat{A}(s_t,a_t) \bigg] \\
& \approx \sum\limits_{i=1}^n w^i_H \sum\limits_{t=0}^H \gamma^t \nabla_{\theta}\log \pi_{\theta}(a_t^i,s_t^i)\hat{A}(s_t^i,a_t^i)
\end{align}
{% endmathjax %}
其中{% mathjax %}\{(s_0^i,a_0^i,r_0^i,s_1^i,\ldots)\}^n_{i=1}{% endmathjax %}是来自{% mathjax %}\pi_{\beta}(\tau){% endmathjax %}的{% mathjax %}n{% endmathjax %}个轨迹样本。类似地，我们可以对**重要性权重**进行归一化，从而得到**加权重要性抽样策略梯度估计量**，该估计量有偏差，但方差很低，并且仍然是一致的估计量。如果我们使用{% mathjax %}\hat{A}{% endmathjax %}基线的蒙特卡洛估计量（即{% mathjax %}\hat{A}(s_t^i,a_t^i) = \sum\limits_{t'=t}^H\gamma^{t'-t}r^{t'} - b(s^i_t){% endmathjax %}，其中{% mathjax %}r_t{% endmathjax %}不依赖于{% mathjax %}s_{t'}{% endmathjax %}和{% mathjax %}a_{t'}{% endmathjax %}且{% mathjax %}t' > t{% endmathjax %}），通过降低**重要性权重**，从而得到每个决策**重要性抽样策略梯度估计量**：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla_{\theta}J(\pi_{\theta}) = \sum\limits_{i=1}^n\sum\limits_{t=0}^H w^i_t\gamma^t\bigg(\sum\limits_{t'=t}^H \gamma^{t' - t}\frac{w^i_{t'}}{w^i_t}r_{t'} - b(s^i_t) \bigg) \nabla_{\theta}\log \pi_{\theta}(a_t^i,s_t^i)
{% endmathjax %}
该估计量具有高方差，通过对权重进行**归一化**来形成加权的每个决策**重要性估计量**。与策略评估的**双重稳定估计量**的发展并行，还推导出用于策略梯度的**双重稳定估计量**。不幸的是，在许多实际问题中，这些估计量的方差太高而无效。从此类估计量派生的实用离线策略算法也可以采用**正则化**，使得学习到的策略{% mathjax %}\pi_{\theta}(a|s){% endmathjax %}不会偏离**行为策略**{% mathjax %}\pi_{\beta}(a|s){% endmathjax %}太远，从而导致**重要性权重**的方差变大。这种**正则化**的例子是（未归一化的）**重要性权重**的最大值。该**正则化梯度估计量**{% mathjax %}\nabla_{\theta}\bar{J}(\pi_{\theta}){% endmathjax %} 具有以下形式：
{% mathjax '{"conversion":{"em":14}}' %}
\nabla_{\theta}\bar{J}(\pi_{\theta}) = \bigg( \sum\limits_{i=1}^n w^i_H \sum\limits_{t=0}^H \gamma^t \nabla_{\theta}\log \pi_{\theta}(a_t^i,s_t^i)\hat{A}(s_t^i,a_t^i)\bigg) + \lambda log\bigg( \sum\limits_{i=1}^n w^i_H \bigg)
{% endmathjax %}
当{% mathjax %}n\leftarrow \infty{% endmathjax %}时，则{% mathjax %}\sum\limits_{i=1}^n w^i_H \leftarrow 1{% endmathjax %}，这意味着**梯度估计器**是一致的。然而，在样本数量有限的情况下，这样的估计器会自动调整策略{% mathjax %}\pi_{\theta}{% endmathjax %}以确保至少一个样本具有较高的**重要性权重**。基于**重要性抽样**的**深度强化学习算法**通常采用基于样本的`KL`**散度正则化器**，当在策略{% mathjax %}\pi_{\theta}{% endmathjax %}上使用**熵正则化器**时，它的函数形式在数学上与此类似。

**重要性加权策略**目标要求在时间步上乘以每个动作的**重要性权重**，这会导致非常高的方差。可以通过使用**行为策略**的状态分布{% mathjax %}d^{\pi_{\beta}}(s){% endmathjax %}代替当前策略的状态分布{% mathjax %}d^{\pi}(s){% endmathjax %}来得出**近似的重要性采样梯度**。由于状态分布不匹配，这会导致梯度有偏差，但在实践中可以提供合理的学习性能。相应的目标，我们将其表示为{% mathjax %}J_{\pi_{\beta}}(\pi_{\theta}){% endmathjax %}，以强调其对**行为策略状态分布**的依赖性，由以下公式给出：
{% mathjax '{"conversion":{"em":14}}' %}
J_{\pi_{\beta}}(\pi_{\theta}) = \mathbb{E}_{s\sim d^{\pi_{\beta}}}[V^{\pi}(s)]
{% endmathjax %}
请注意，{% mathjax %}d^{\pi_{\beta}}(s){% endmathjax %}和{% mathjax %}d^{\pi_{\beta}}(s){% endmathjax %}在预测回报的状态分布（{% mathjax %}d^{\pi_{\beta}}(s){% endmathjax %}`vs`{% mathjax %}d^{\pi}(s){% endmathjax %}）有所不同，这使得{% mathjax %}J_{\pi_{\beta}}{% endmathjax %}成为{% mathjax %}J_{\pi_{\theta}}{% endmathjax %}的**有偏估计量**。在某些情况下，这可能是次优解决方案。但是，在离线情况下，可以通过从数据集{% mathjax %}\mathcal{D}{% endmathjax %}中抽样状态轻松计算状态分布下的期望，从而无需进行**重要性抽样**。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\nabla_{\theta}J(\pi_{\theta}) & = \mathbb{E}_{s\sim d^{\pi_{\beta}}(s),a\sim\pi_{\theta}(a,s)}\bigg[Q^{\pi_{\theta}(s,a)} \nabla_{\theta}\log \pi_{\theta}(a|s)+ \nabla_{\theta}Q^{\pi_{\theta}}(s,a) \bigg] \\
& \approx \mathbb{E}_{s\sim d^{\pi_{\beta}}(s),a\sim\pi_{\theta}(a,s)}\bigg[Q^{\pi_{\theta}(s,a)} \nabla_{\theta}\log \pi_{\theta}(a|s) \bigg]
\end{align}
{% endmathjax %}
在限制条件下，**近似梯度**保留了{% mathjax %}J_{\pi_{\beta}}(\pi){% endmathjax %}的局部最优值。这种**近似梯度**被广泛用于**深度强化学习算法**。为了计算**近似梯度**的估计值，需要从**离线策略轨迹估计** {% mathjax %}Q^{\pi_{\theta}}(s,a){% endmathjax %}。一些估计器使用动作样本，这需要**重要性权重**从{% mathjax %}\pi_{\beta}{% endmathjax %}样本校正{% mathjax %}\pi_{\theta}{% endmathjax %}样本。通过引入**控制变量**和**裁剪重要性权重**来控制方差，可以获得进一步的改进。如果我们想避免使用**离线策略状态分布**产生的偏差和使用每个动作**重要性权重**产生的高方差，可以尝试直接**预测状态边际重要性**。使用**状态边际重要性权重**估计器的方差小于使用每个动作**重要性权重**的乘积。然而，准确计算这个比率通常是很难。根据所用底层**贝尔曼方程**的形式，可以将它们分为两类：使用“**前向**”**贝尔曼方程**直接预测**重要性比率**的方法，以及在类似于价值函数的函数上使用“**后向**”**贝尔曼方程**的方法，然后从学习到的**价值函数**中得出**重要性比率**。基于**前向贝尔曼方程**的方法。**状态边际重要性比率**{% mathjax %}\rho^{\pi}(s){% endmathjax %}满足一种“**前向**”**贝尔曼方程**：
{% mathjax '{"conversion":{"em":14}}' %}
\forall s'\;\;\underbrace{d^{\pi_{\beta}}(s')\rho^{\pi}(s')}_{:=(d^{\pi_{\beta}}o\rho^{\pi})(s')} = \underbrace{(1 - \gamma)d_0(s') + \gamma\sum\limits_{s,a} d^{\pi_{\beta}}(s)\rho^{\pi}(s)\pi (a|s)T(s'|s,a)}_{:=(\bar{\mathcal{B}}^{\pi}o\rho^{\pi})(s')}
{% endmathjax %}
可以利用这种关系进行**时间差异更新**，预测策略下的**状态边际重要性比率**。例如，当使用**随机近似**时，`Gelada`和`Bellemare`使用以下更新规则来在线预测{% mathjax %}\rho^{\pi}(s'){% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{\rho}^{\pi}(s')\leftarrow \hat{\rho}^{\pi}(s') + \alpha\bigg[(1 - \gamma) + \gamma\frac{\pi(a|s)}{\pi_{\beta}(a|s)}\hat{\rho}^{\pi}(s) - \hat{\rho}^{\pi}(s') \bigg]
{% endmathjax %}
其中{% mathjax %}s\sim d^{\pi_{\beta}}(s),a\sim \pi_{\beta}(a|s),s'\sim T(s'|s,a){% endmathjax %}。已经使用了几种技术来稳定学习，包括使用{% mathjax %}TD(\lambda){% endmathjax %}预测和自动调整特征维度。这里请参阅`Hallak`和`Mannor (2017)`以及`Gelada`和`Bellemare (2019)`。`Gelada`和`Bellemare (2019)`还讨论了一些实用技巧，例如**软归一化**和**折扣评估**，使这些方法适应深度`Q-Learning`设置，这与**线性函数逼近**不同。`Wen`等人`(2020)`从**幂迭代**的角度看待问题，并提出了一种**变分幂方法**，将**函数逼近**和**幂迭代**结合起来预测{% mathjax %}\rho^{\pi}{% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
L(\rho,f) = \gamma\mathbb{E}_{s,a,s'\sim \mathcal{D}}\bigg[ \bigg(\rho(s)\frac{\pi(a|s)}{\pi_{\beta}(a|s)} - \rho(s')\bigg)f(s')\bigg] + (1-\gamma)\mathbb{E}_{s_0\sim d_0}[(1-\rho(s))f(s)]
{% endmathjax %}
其中{% mathjax %}L(\rho,f)=0,\forall f{% endmathjax %}当且仅当{% mathjax %}\rho = \rho^{\pi}{% endmathjax %}。可以通过最小化{% mathjax %}L(\rho,f){% endmathjax %}的最坏情况预测来学习{% mathjax %}\rho{% endmathjax %}，此方法是解决**对抗性鞍点优化**：{% mathjax %}\min_{\rho}\max_{f}L(\rho,f)^2{% endmathjax %}。最近的研究改进了这种方法，特别是消除了对{% mathjax %}\pi_{\beta}{% endmathjax %}的访问需要。一旦获得{% mathjax %}\rho^*{% endmathjax %}，就会使用该估计量来计算**离线策略梯度**。`Zhang`等人(`2020`)提出了另一种**离线策略评估方法**，该方法通过直接优化**前向贝尔曼方程**的**贝尔曼残差误差**的变体来计算状态-动作对边际的重要性比率{% mathjax %}\rho^{\pi}(s,a):= \frac{d^{\pi}(s,a)}{d^{\pi_{\beta}}(s,a)}{% endmathjax %}，该方程包含动作，如下所示:
{% mathjax '{"conversion":{"em":14}}' %}
\underbrace{d^{\pi_{\beta}}(s',a')\rho^{\pi}(s',a')}_{:=(d^{\pi_{\beta}}\circ\rho^{\pi})(s',a')} = \underbrace{(1 - \gamma)d_0(s')\pi(a'|s') + \gamma\sum\limits_{s,a} d^{\pi_{\beta}}(s,a)\rho^{\pi}(s,a)\pi(a|s)T(s'|s,a)}_{:=(\bar{\mathcal{B}}^{\pi}\circ\rho^{\pi})(s',a')}
{% endmathjax %}
可以通过**前向贝尔曼方程**的两边之间应用**散度度量**来推导，同时**限制重要性比率**{% mathjax %}\rho^{\pi}(s,a){% endmathjax %}，使其在数据集{% mathjax %}\mathcal{D}{% endmathjax %}上的期望积分为`1`，防止出现退化，如下所示：
{% mathjax '{"conversion":{"em":14}}' %}
\underset{\rho^{\pi}}{\min}\;\;D_f((\bar{\mathcal{B}}^{\pi}\circ\rho^{\pi})(s,a),(d^{\pi_{\beta}}\circ\rho^{\pi})(s,a))\;\;\;\mathbb{E}_{s,a,s'\sim\mathcal{D}}[\rho^{\pi}(s,a)]= 1
{% endmathjax %}
进一步应用了**受对偶嵌入**(`Dai et al., 2016`)启发的技巧，使目标变得易于处理，并避免抽样估计而导致的偏差。原始对偶求解器可能无法求解以上方程，通过用{% mathjax %}\frac{1}{d^{\pi_{\beta}}}{% endmathjax %}引起的范数替换f散度来修改目标。这创建了一个优化问题，该问题在线性函数近似下证明是收敛的。
{% mathjax '{"conversion":{"em":14}}' %}
\underset{\rho^{\pi}}{\min}\;\;\frac{1}{2}\|(\bar{\mathcal{B}}^{\pi}\circ\rho^{\pi})(s,a),(d^{\pi_{\beta}}\circ\rho^{\pi})(s,a)\|^2_{(d^{\pi_{\beta}})} + \frac{\lambda}{2}(\mathbb{E}_{s,a,s'\sim\mathcal{D}}[\rho^{\pi}(s,a)] - 1)^2
{% endmathjax %}
通过**凸对偶**实现**后向贝尔曼方程**的方法。由于这些方法从优化角度出发，因此它们可以发挥**凸优化**和**在线学习**的作用。`Lee`和`He (2018)`将应用于凸优化技术的工作扩展到**策略优化**和**离线策略设置**。证明了离线策略设置中的样本复杂度界限，但是，将这些结果扩展到实际的深度强化学习设置已被证明具有难度。
{% mathjax '{"conversion":{"em":14}}' %}
\rho^{\pi} = \text{arg}\;\underset{x:\mathcal{S}\times\mathcal{A}\rightarrow \mathbb{R}}{\min} \frac{1}{2}\mathbb{E}_{s,a,s'\sim\mathcal{D}}[x(s,a)^2] - \mathbb{E}_{s\sim d^{\pi}(s),a\sim\pi(a|s)}[x(s,a)]
{% endmathjax %}
这个目标需要来自在**策略状态边际分布** {% mathjax %}d^{\pi}(s){% endmathjax %}的样本。关键的思想是改变变量{% mathjax %}x(s,a) = v(s,a) - \mathbb{E}_{s'\sim T(s'|s,a),a'\sim \pi(a'|s')}[v(s',a')]{% endmathjax %}并引入变量{% mathjax %}v(s,a){% endmathjax %}来简化以上公式。为简洁起见，这里定义了一个修改的`Bel`算子{% mathjax %}\tilde{\mathcal{B}^{\pi}}v(s,a):= \mathbb{E}_{s'\sim T(s'|s,a),a'\sim \pi(a'|s')}[v(s',a')]{% endmathjax %}，没有奖励项{% mathjax %}r(s,a){% endmathjax %}的{% mathjax %}\mathcal{B}^{\pi}{% endmathjax %}表达式。
{% mathjax '{"conversion":{"em":14}}' %}
\underset{x:\mathcal{S}\times\mathcal{A}\rightarrow \mathbb{R}}{\min} \frac{1}{2}\mathbb{E}_{s,a,s'\sim\mathcal{D}}\bigg[\bigg(v(s,a) - \tilde{\mathcal{B}^{\pi}}v(s,a)\bigg)^2\bigg] - \mathbb{E}_{s_0\sim d_0^{\pi}(s_0),a\sim\pi(a|s_0)}[v(s_0,a)]
{% endmathjax %}
值得注意的是，以上公式不需要**在线策略样本**来评估。最优解表示为{% mathjax %}v^*{% endmathjax %}，我们可以使用关系{% mathjax %}\rho^{\pi}(s,a) = v^* (s,a) - \tilde{\mathcal{B}^{\pi}}v^*(s,a){% endmathjax %}获得**密度比**{% mathjax %}\rho^{\pi}{% endmathjax %}。然后可以使用**密度比**进行**离线策略评估**和改进。

{% mathjax %}f{% endmathjax %}-散度正则化的强化学习问题，带有权衡因子{% mathjax %}\alpha{% endmathjax %}，由以下公式给出：
{% mathjax '{"conversion":{"em":14}}' %}
\underset{\pi}{\max}\;\;\mathcal{E}_{s\sim d^{\pi}(s),s\sim\pi(\cdot|s)}[r(s,a)] - \alpha D_f(d^{\pi}(s,a), d^{\pi_{\beta}}(s,a))
{% endmathjax %}
通过利用如下所示的{% mathjax %}f{% endmathjax %}-散度的变分（对偶）方式：
{% mathjax '{"conversion":{"em":14}}' %}
D_f(p,q) = \underset{x:\mathcal{S}\times\mathcal{A}\rightarrow \mathbb{R}}{\max} (\mathbb{E}_{y\sim p(y)}[x(\mathbf{y})] - \mathbb{E}_{y\sim q(y)}[f^*(x(\mathbf{y}))])
{% endmathjax %}
然后将变量从{% mathjax %}x{% endmathjax %}更改为{% mathjax %}Q{% endmathjax %}，其中{% mathjax %}Q{% endmathjax %}满足{% mathjax %}Q(s,a) = \mathbb{E}_{s'\sim T(s'|s,a),a'\sim \pi(a'|s')}[r(s,a) - \alpha x(s,a) + \gamma Q(s',a')]{% endmathjax %}，这样就得到一个用于正则化**强化学习**(`RL`)目标的**鞍点优化**问题。
{% mathjax '{"conversion":{"em":14}}' %}
\underset{\pi}{\max}\;\;\underset{Q}{\min}\;\;L(Q,\pi_{\beta},\pi) := \mathbb{E}_{s_0\sim d_0(s_0),a\sim \pi(\cdot|s_0)}[Q(s_0,a)] + \alpha\mathbb{E}_{s,a\sim d^{\pi_{\beta}}(s,a)}\bigg[ f^*\bigg(\frac{r(s,a) + \gamma \mathbb{E}_{s'\sim T(s'|s,a),a'\sim \pi(a'|s')}[Q(s',a')] - Q(s,a)}{\alpha} \bigg)\bigg]
{% endmathjax %}
当{% mathjax %}f(x) = x^2,\;f^*(x) = x^2{% endmathjax %}时，可以证明，在最佳{% mathjax %}Q{% endmathjax %}函数{% mathjax %}Q^*{% endmathjax %}​​处，关于{% mathjax %}L(Q^*,\pi_{\beta},\pi){% endmathjax %}策略的导数恰好等于**正则化策略梯度问题**中的**在线策略梯度**：
{% mathjax '{"conversion":{"em":14}}' %}
\frac{\partial}{\partial \pi}L(Q^*,\pi_{\beta},\pi) = \mathbb{E}_{s\sim d^{\pi}(s),a\sim\pi(\cdot|s)}\bigg[ \tilde{Q_{\pi}}(s,a)\cdot \nabla_{\pi}\log_{\pi}(a|s) \bigg]
{% endmathjax %}
其中{% mathjax %}\tilde{Q_{\pi}}{% endmathjax %}是与正则化**强化学习**(`RL`)问题相对应的**动作值函数**。
