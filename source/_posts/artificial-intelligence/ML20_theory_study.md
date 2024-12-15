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
- 在基CPU和GPU加速的环境下均实现了最优性能。
- 支持**单智能体**和**多智能体**训练，**自我对弈**，同时支持在一个或多个GPU上同时训练多个策略。
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

