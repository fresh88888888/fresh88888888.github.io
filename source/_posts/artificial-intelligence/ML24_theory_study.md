---
title: 机器学习(ML)(二十四) — 强化学习探析
date: 2025-01-08 16:00:11
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

#### A2PO

**离线强化学习**旨在利用离线数据集来构建有效的**智能体策略**，而无需在线交互。这种方法需要在**行为策略**的支持下施加适当的**保守约束**，以应对分布外问题。然而，现有工作在处理来自多个**行为策略**的离线数据集时，常常面临**约束冲突**问题，即不同的**行为策略**可能在**状态空间**中表现出不一致的动作和不同的回报。为了解决这一问题，近期的**优势加权方法**优先考虑具有高优势值的样本进行**智能体**训练，但不可避免地忽视了**行为策略**的多样性。
<!-- more -->

**行为策略**(`Behavior Policy`)是**强化学习**中的一个重要概念，指的是**智能体**在与环境交互时实际采取的策略。**行为策略**(`Behavior Policy`)是**智能体**在特定状态下选择动作的规则或映射。它描述了**智能体**如何在环境中做出**决策**，并生成与环境交互所需的数据。
- **目标策略**(`Target Policy`)：是**智能体**希望学习和优化的策略，通常是期望达到最优性能的策略。
- **行为策略**(`Behavior Policy`)：与**目标策略**之间的区别在于，**行为策略**是实际执行的策略，而**目标策略**则是通过**行为策略**收集的数据来进行学习和优化的对象。

**行为策略**在**强化学习**中起着至关重要的作用，因为它直接影响到数据的多样性和质量。一个好的**行为策略**能够提供丰富的数据，使得**智能体**能够更有效地学习并优化其**目标策略**。

**优势感知策略优化**(`A2PO`)是一种新颖的**离线强化学习**方法，旨在处理混合质量数据集中的**策略优化**问题。该方法通过显式构建**优势感知策略约束**，帮助**智能体**在没有在线交互的情况下有效学习。特点：**优势感知**，`A2PO`利用**条件变分自动编码器**(`CVAE`)来解耦不同的**行为策略**，明确建模每个动作的优势值，从而优化**智能体**的决策过程；**混合质量数据集**，该方法特别设计用于处理来自多种**行为策略**的混合质量数据集，解决了传统方法在面对不一致动作和回报时可能出现的约束冲突问题；实验验证，在`D4RL`基准上进行的大量实验表明，`A2PO`在单一质量和混合质量数据集上均优于现有的其他**离线强化学习**方法，如`BCQ`、`TD3+BC`和`CQL`等。

`A2PO`适用于需要从静态数据集中学习的各种应用场景，如机器人控制、自动驾驶和游戏`AI`等。通过有效利用历史数据，该方法能够减少探索过程中的风险和成本，同时提升**智能体**的性能。`A2PO`为**离线强化学习**领域提供了一种新的思路，尤其是在处理复杂数据集时，其**优势感知机制**显著提高了**策略优化**的效果。

**离线强化学习**(`Offline Reinforcement Learning, ORL`)旨在从预先收集的数据集中学习有效的**控制策略**，而无需在线探索。这种方法在多个现实世界应用中取得了前所未有的成功，包括机器人控制和电网控制等。然而，**离线强化学习**面临着一个严峻的挑战，即**分布外**(`Out-Of-Distribution, OOD`)**问题**，这涉及到**学习策略**产生的数据与**行为策略**收集的数据之间的**分布偏移**。因此，直接在**在线强化学习**方法上应用会出现**外推误差**，即**对未见状态-动作对的错误估计**。为了解决这个`OOD`问题，**离线强化学习**方法尝试在数据集的分布范围内对**智能体**施加适当的保守约束，例如，通过正则化项限制**学习策略**，或对`OOD`动作的价值过高估计进行惩罚。**离线强化学习**在处理混合质量数据集时常常遇到约束冲突问题。具体而言，当训练数据来自多个具有不同回报的**行为策略**时，现有工作仍然平等对待每个样本约束，而没有考虑数据质量和多样性的差异。这种忽视导致对冲突动作施加不当约束，最终导致更差的结果。

**分布外**(`OOD`)：指的是在训练过程中，**智能体**所选择的**状态-动作对**不在其训练数据集的分布中。这意味着**智能体**在决策时可能会遇到未曾见过的情况，从而导致对这些**状态-动作对**的价值估计不准确。在**离线强化学习**中，**智能体**使用的是预先收集的静态数据集，而不是实时与环境交互。因此，**智能体**无法从环境中获取新的**状态-动作对**，这限制了其学习能力。当**智能体**尝试采取在训练数据集中没有出现过的动作时，就会出现`OOD`问题。这些未见过的**状态-动作对**可能会导致`Q`值或策略的高估，从而影响整体性能。

`A2PO`能够实现来自不同**行为策略**的**优势感知策略约束**，其中采用了定制的**条件变分自动编码器**(`CVAE`)来推断与**行为策略**相关的**多样化动作分布**，通过将优势值建模为条件变量。在`D4RL`基准上进行了大量实验，包括单一质量和混合质量数据集，结果表明，所提出的`A2PO`方法在性能上显著优于其他先进的**离线强化学习**基线，以及优势加权竞争者。

**离线强化学习**(`ORL`)可以大致分为四类：**策略约束**、**价值正则化**、**基于模型的方法**和**基于回报条件的监督学习**。
- **策略约束**：该方法对学习到的策略施加约束，以保持其接近**行为策略**。之前的研究直接在策略学习中引入了显式约束，例如**行为克隆**、**最大均值差异**或**最大似然估计**。相对而言，最近的努力主要集中在通过近似由`KL`-**散度约束**推导出的最优策略来隐式实现**策略约束**。
- **价值正则化**：该方法对**价值函数**施加约束，以缓解对**分布外**(`OOD`)动作的高估。研究者们尝试通过`Q-`**正则化项**来近似**价值函数**的下界，以实现保守的动作选择。
- **基于模型的方法**：该方法构建环境动态，以估计**状态-动作对**的不确定性，从而施加**分布外**(`OOD`)**惩罚**。
- **基于回报条件的监督学习任务**。**决策变换器**(`Decision Transformer，DT`)构建了一个基于当前状态和额外累积回报信号的**变换器策略**，并通过**监督学习**进行训练。`Yamagata`等人通过用`Q-Learning`结果重新标记回报信号，改善了**决策变换器**(`DT`)**策略**在次优样本上的拼接能力。然而，在混合质量数据集且无法访问轨迹回报信号的**离线强化学习**背景下，所有这些方法都平等对待每个样本，而未考虑数据质量，从而导致不当的正则化和进一步的次优学习结果。

**优势加权离线强化学习方法**通过加权采样来优先训练具有高优势值的离线数据集中的转换。为提高样本效率，`Peng`等人引入了一种**优势加权最大似然损失**，通过**轨迹回报**直接计算优势值。此外，还有研究使用**评论家网络**来估计优势值，以进行**优势加权策略训练**。这项技术已被纳入其他工作的流程当中，用于**智能体**策略提取。最近，**优势加权回归**(`AW`)方法在解决混合质量数据集中的约束冲突问题上也得到了很好的研究。一些研究提出将**优势加权行为克隆**作为直接**目标函数**或**显式策略约束**。此外，`LAPO`框架采用**优势加权损失**来训练**条件变分自编码器**(`CVAE`)，以生成基于状态条件的高优势动作。除了**优势加权回归**(`AW`)方法外，`Hong`等人增强了经典**离线强化学习**训练目标，通过后续回报的权重进行调整。而`Hong`等人则直接学习**最优策略密度**作为**权重函数**，以便从高性能策略中进行采样。然而，这种**优势加权回归**(`AW`)机制不可避免地减少了数据集中的多样性。相对而言，**优势感知策略优化**(`A2PO`)方法直接将**智能体**策略条件化于**状态**和**估计**的优势值，使得能够有效利用所有样本，无论其质量如何。

**优势加权回归**(`Advantage-Weighted Regression, AWR`)是一种简单且可扩展的**离线强化学习算法**，旨在利用标准的**监督学习**方法作为子程序。`AWR`的核心思想是通过两个**监督回归**步骤来训练**智能体**的策略和**价值函数**。原理：**价值函数回归**，首先，`AWR`通过**回归累计奖励**来训练一个**价值函数基线**。这一步骤的目的是建立一个对**环境奖励**的估计，使得后续的**策略更新**能够更好地反映出哪些动作是有效的；**策略回归**，接下来，`AWR`使用**加权回归**来更新策略。这里，**加权**是基于每个动作的**优势值**(`advantage`)，即**某个动作相对于当前策略的期望收益的提升程度**。通过这种方式，`AWR`能够优先选择那些表现更好的动作，从而提高**学习效率**。

`LAPO`(`Latent-Variable Advantage-Weighted Policy Optimization`)是一种针对**离线强化学习算法**，旨在有效处理异质性数据集中的**策略学习**问题。该方法通过利用潜在变量生成模型来表示高优势的**状态-动作对**，从而提高策略的学习效率和效果。原理：**潜在空间建模**，`LAPO`通过学习一个状态条件的潜在空间，生成高优势动作样本。这种方法使得**智能体**能够选择那些在训练数据中支持的动作，同时有效地解决目标任务；**优势加权**，该算法采用**优势加权策略**，通过**最大化加权对数似然**来学习高回报动作。具体来说，`LAPO`通过两步交替进行：估计每个动作的重要性权重，并根据这些权重回归数据集中的动作；`Q`**函数优化**，`LAPO`在每次迭代中优化潜在策略，以直接最大化回报。它使用标准**强化学习**方法（如`DDPG`或`TD3`）来更新潜在动作，这些动作经过**解码器**转换为**原始动作空间**。

我们将**强化学习**(`RL`)任务形式化为一个**马尔可夫决策过程**(`MDP`)，定义为一个**元组**{% mathjax %}\mathcal{M} = \langle \mathcal{S},\mathcal{A},P,r,\gamma,p_0 \rangle{% endmathjax %}，其中{% mathjax %}\mathcal{S}{% endmathjax %}表示**状态空间**，{% mathjax %}\mathcal{A}{% endmathjax %}表示动作空间，{% mathjax %}P\;:\;\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow [0,1]{% endmathjax %}表示**环境动态**，{% mathjax %}r\;:\;\mathcal{S}\times\mathcal{A}\rightarrow \mathbb{R}{% endmathjax %}表示**奖励函数**，{% mathjax %}\gamma\in(0,1]{% endmathjax %}是**折扣因子**，{% mathjax %}p_0{% endmathjax %}是**初始状态分布**。在每个时间步{% mathjax %}t{% endmathjax %}，**智能体**观察状态{% mathjax %}s\in \mathcal{S}{% endmathjax %}，并根据其策略{% mathjax %}\pi{% endmathjax %}选择一个动作{% mathjax %}a_t\in\mathcal{A}{% endmathjax %}。这个动作导致根据动态分布{% mathjax %}P{% endmathjax %}转移到下一个状态{% mathjax %}s_{t+1}{% endmathjax %}。此外，**智能体**还会收到奖**励信号**{% mathjax %}r_t{% endmathjax %}。**强化学习**的目标是学习一个**最优策略**{% mathjax %}\pi^*{% endmathjax %}，以最大化期望回报：{% mathjax %}\pi^* = \text{arg }\max_{\pi}\;\mathbb{E}_{\pi}\bigg[\sum\limits_{k=0}^{\infty} \gamma^k r_{t+k}\bigg]{% endmathjax %}。在**离线强化学习**中，**智能体**只能从离线数据集中学习，而无法与环境进行在线交互。在单一质量设置中，离线数据集{% mathjax %}\mathcal{D} = \{(s_t,a_t,r_t,s_{t+1})|t=1,\ldots,N\}{% endmathjax %}是由单一行为策略{% mathjax %}\pi_{\beta}{% endmathjax %}收集的，包含{% mathjax %}N{% endmathjax %}次转换。在混合质量设置中，离线数据集{% mathjax %}\mathcal{D} = \bigcup_i\{(s_{i,t},a_{i,t},r_{i,t},s_{i,t+1})|t=1,\ldots,N\}{% endmathjax %}是由多个**行为策略**{% mathjax %}\{\pi_{\beta_i}\}_{i=1}^M{% endmathjax %}收集的。通过**动作价值函数**{% mathjax %}Q^{\pi}(s,a) = \mathbb{E}_{\pi}[\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)|s_0 = s,a_0 = a]{% endmathjax %}来评估学习到的策略{% mathjax %}pi{% endmathjax %}。**状态价值函数**定义为{% mathjax %}V^{\pi}(s) = \mathbb{E}_{a\sim\pi}[Q^{\pi}(s,a)]{% endmathjax %}，而**优势函数**定义为{% mathjax %}A^{\pi}(s,a) = Q^{\pi}(s,a) - V^{\pi}(s){% endmathjax %}。对于连续控制，`A2PO`实现使用基于`actor-critic`框架的`TD3`算法作为其基础，以确保稳健的性能。**演员网络**{% mathjax %}\pi_{\omega}{% endmathjax %}，即学习到的策略，由参数{% mathjax %}\omega{% endmathjax %}参数化，而**评论家网络**则由参数为{% mathjax %}\theta{% endmathjax %}的`Q`-网络{% mathjax %}Q_{\theta}{% endmathjax %}和参数为{% mathjax %}\phi{% endmathjax %}的`V`-网络{% mathjax %}V_{\phi}{% endmathjax %}组成。`actor-critic`框架涉及两个步骤：**策略评估**和**策略改进**。在**策略评估**阶段，通过**时间差**(`TD`)**损失优化**`Q`-网络{% mathjax %}Q_{\theta}{% endmathjax %}。
{% asset_img ml_1.png "优势感知策略优化(Advantage-Aware Policy Optimization, A2PO)方法示意图" %}

{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}_Q(\theta) = \mathbb{E}_{(s,a,r,s')\sim\mathcal{D},a'\sim\pi_{\hat{\omega}}(s')}[Q_{\theta}(s,a) - (r(s,a) + \gamma Q_{\hat{\theta}}(s',a'))]^2
{% endmathjax %}
我们将**目标网络**的参数{% mathjax %}\hat{\theta}{% endmathjax %}和{% mathjax %}\hat{\omega}{% endmathjax %}定期更新，以保持学习的稳定性，这些参数是通过在线参数{% mathjax %}\theta{% endmathjax %}和{% mathjax %}\omega{% endmathjax %}更新的。`V`-网络{% mathjax %}V_{\phi}{% endmathjax %}也可以通过类似的**时间差**(`TD`)**损失**进行优化。在连续控制中的策略改进阶段，**演员网络**{% mathjax %}\pi_{\omega}{% endmathjax %}可以通过确定性**策略梯度损失**进行优化。
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}_{\pi}(\omega) = \mathbb{E}_{s\sim\mathcal{D}}[-Q_{\theta}(s,\pi_{\omega}(s))]
{% endmathjax %}
{% note warning %}
`A2PO`方法由两个关键步骤组成：**行为策略解耦**和**智能体策略优化**。在**行为策略解耦**阶段，使用**条件变分自编码器**(`CVAE`)解开**行为策略**，基于收集的**状态-动作对**的优势值条件下的**动作分布**。通过输入不同的优势值，新的**条件变分自编码器**(`CVAE`)允许**智能体**推断与各种**行为策略**相关的不同**动作分布**。然后，在**智能体**策略优化阶段，从优势条件导出的**动作分布**作为解耦的**行为策略**，建立一个**优势感知的策略约束**来指导**智能体**的训练。

**注意**，**离线强化学习**将在优化损失上施加保守约束，以应对**分布外**(`OOD`)问题。此外，最终学习到的策略{% mathjax %}\pi_{\omega}{% endmathjax %}的性能在很大程度上依赖于与**行为策略**{% mathjax %}\{\pi_{\beta_i}\}{% endmathjax %}相关的离线数据集{% mathjax %}\mathcal{D}{% endmathjax %}的质量。
{% endnote %}

##### 行为策略解耦

为了实现**行为策略解耦**，我们采用**条件变分自编码器**(`CVAE`)将不同具体**行为策略**{% mathjax %}\pi_{\beta_i}{% endmathjax %}的**行为分布**与基于优势的条件变量关联起来，这与之前的方法大相径庭，后者仅利用`CVAE`来近似仅基于特定状态{% mathjax %}s{% endmathjax %}的整体混合质量行为策略集{% mathjax %}\{\pi_{\beta_i}\}_{i=1}^M{% endmathjax %}。具体而言，对`CVAE`的架构进行了调整，使其具备**优势感知能力**。**编码器**{% mathjax %}q_{\varphi}(z|a,c){% endmathjax %}接收条件{% mathjax %}c{% endmathjax %}和动作{% mathjax %}a{% endmathjax %}，将它们投影到潜在表示{% mathjax %}z{% endmathjax %}中。给定特定条件{% mathjax %}c{% endmathjax %}和**编码器**输出{% mathjax %}z{% endmathjax %}，**解码器**{% mathjax %}p_{\psi}(a|z,c){% endmathjax %}捕捉条件{% mathjax %}c{% endmathjax %}与潜在表示{% mathjax %}z{% endmathjax %}之间的相关性，以重构原始动作{% mathjax %}a{% endmathjax %}。与之前的方法不同，这里不仅考虑状态{% mathjax %}s{% endmathjax %}，还考虑优势值{% mathjax %} \xi{% endmathjax %}作为`CVAE`的条件。**状态-优势条件**{% mathjax %}c{% endmathjax %}被公式化为：
{% mathjax '{"conversion":{"em":14}}' %}
c = s\| \xi
{% endmathjax %}
因此，给定当前状态{% mathjax %}s{% endmathjax %}和优势值{% mathjax %}\xi{% endmathjax %}作为**联合条件**，`CVAE`模型能够生成与优势值{% mathjax %}\xi{% endmathjax %}正相关的不同质量的相应动作{% mathjax %}a{% endmathjax %}。对于**状态-动作对**{% mathjax %}(s,a){% endmathjax %}，优势值{% mathjax %}\xi{% endmathjax %}可以通过以下公式计算：
{% mathjax '{"conversion":{"em":14}}' %}
\xi = \tanh (\underset{i=1,2}{\min}\;Q_{\theta_i}(s,a) - V_{\phi}(s))
{% endmathjax %}
其中采用两个`Q`-**网络**并使用最小操作以确保**离线强化学习**环境中的保守性。此外，我们使用了{% mathjax %}\tanh(\cdot){% endmathjax %}**函数**将优势条件归一化到{% mathjax %}(-1,1){% endmathjax %}范围内。这一操作防止了过多的异常值影响`CVAE`的性能，提高了生成的可控性。`CVAE`模型使用**状态-优势条件**{% mathjax %}c{% endmathjax %}和相应的动作{% mathjax %}a{% endmathjax %}进行训练。训练目标涉及**最大化经验下界**(`ELBO`)上采样的小批量数据的**对数似然**。
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}_{\text{CAVE}}(\varphi,\psi) = -\mathbb{E}_{\mathcal{D}}\bigg[ \mathbb{E}_{q\varphi(z|a,c)}[\log(p_{\psi}(a|z,c))] + alpha\cdot \mathbf{KL} [q_{\varphi}(z|a,c)\|p(z)]\bigg]
{% endmathjax %}
其中，{% mathjax %}\alpha{% endmathjax %}是用于平衡`KL`-**散度损失项**的系数，{% mathjax %}p(z){% endmathjax %}表示**先验分布**，设置为{% mathjax %}\mathcal{N}(0,1){% endmathjax %}。第一个对数似然项鼓励生成的动作尽可能与真实动作匹配，而第二个`KL`-**散度项**则使潜在**变量分布**与**先验分布**{% mathjax %}p(z){% endmathjax %}对齐。在每轮`CVAE`训练中，从离线数据集中抽取一小批**状态-动作对**{% mathjax %}(s,a){% endmathjax %}。这些对被输入到{% mathjax %}Q_{\theta}{% endmathjax %}和{% mathjax %}V_{\phi}{% endmathjax %}中，通过以上公式获取相应的优势条件{% mathjax %}\xi{% endmathjax %}。然后，**优势感知**的`CVAE`随后通过以上公式进行优化。结合优势条件{% mathjax %}\xi{% endmathjax %}，`CVAE`捕捉了{% mathjax %}\xi{% endmathjax %}与**行为策略**的**动作分布**之间的关系，这进一步使得`CVAE`能够基于**状态-优势条件**{% mathjax %}c{% endmathjax %}生成动作{% mathjax %}a{% endmathjax %}，使得**动作质量**与**优势条件**{% mathjax %}\xi{% endmathjax %}正相关。此外，`优势感知`的`CVAE`被用于为下一阶段的**智能体策略优化**建立一个**优势感知策略约束**。

##### 智能体策略优化

**智能体**是基于`actor-critic`框架构建的。**评论家**由两个`Q`-**网络**{% mathjax %}Q_{\theta_{i=1,2}}{% endmathjax %}和一个`V`-**网络**{% mathjax %}V_{\phi}{% endmathjax %}组成，用于**近似智能体策略**的价值。**演员**，即**优势感知策略**{% mathjax %}\pi_{\omega}(\cdot|c){% endmathjax %}，以{% mathjax %}c = s\|\xi{% endmathjax %}为输入，基于状态{% mathjax %}s{% endmathjax %}和指定的**优势条件**{% mathjax %}\xi{% endmathjax %}生成潜在表示{% mathjax %}\tilde{z}{% endmathjax %}。然后，这个潜在表示{% mathjax %}\tilde{z}{% endmathjax %}以及条件{% mathjax %}c{% endmathjax %}被输入到**解码器**{% mathjax %}p_{\psi}{% endmathjax %}中，以生成可识别的动作{% mathjax %}a_{\xi}{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
a_{\xi}\sim p_{\psi}(\cdot|\tilde{z},c),\;、text{其中}\;\tilde{z}\sim \pi_{\omega}(\cdot|c)
{% endmathjax %}
**智能体**的**优势感知策略**{% mathjax %}\pi_{\omega}{% endmathjax %}预计将生成与指定的优势输入{% mathjax %}\xi{% endmathjax %}正相关的不同质量的动作，该输入在公式{% mathjax %}\xi = \tanh (\underset{i=1,2}{\min}\;Q_{\theta_i}(s,a) - V_{\phi}(s)){% endmathjax %}中被归一化到{% mathjax %}(-1,1){% endmathjax %}范围内。因此，输出的最优动作{% mathjax %}a^*{% endmathjax %}是通过输入{% mathjax %}c^* = s\|\xi^*{% endmathjax %}获得的，其中{% mathjax %}\xi^* = 1{% endmathjax %}。需要注意的是，**评论家网络**旨在**近似最优策略**{% mathjax %}\pi_{\omega}(\cdot|c^*){% endmathjax %}的期望值。根据`actor-critic`框架，**智能体**优化包括**策略评估**和**策略改进**两个步骤。在**策略评估**步骤中，通过**最小化与最优策略**{% mathjax %}\pi_{\omega}(\cdot|c^*){% endmathjax %}的**时间差损失**来更新评论家。具体而言，对于`V`-**网络**{% mathjax %}V_{\phi}{% endmathjax %}，采用**一步贝尔曼算子**来近似在当前**智能体感知策略**下的**状态价值**，该**状态价值**以最优优势输入{% mathjax %}\xi^* = 1{% endmathjax %}为条件，如下所示：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}_{\text{TD}}(\phi) = \mathbb{E}_{(s,a,r,s')\sim \mathcal{D},\tilde{z}^*\sim \pi_{\omega}(\cdot|c*),a^*_{\xi}\sim p_{\psi}(\cdot|\tilde{z}^*,c^*)}\bigg[ r + \gamma\underset{i}{\min}\;Q_{\hat{\theta}_i}(s' - a^*_{\xi})- V_{\phi}(s)^2\bigg]
{% endmathjax %}
目标网络{% mathjax %}\hat{Q}_{\theta}{% endmathjax %}是通过软更新方式进行更新的。对于`Q`-**网络**，两个`Q`-**网络**实体{% mathjax %}Q_{\theta_i}{% endmathjax %}都是按照公式与**智能体策略**{% mathjax %}\pi_{\omega}(\cdot|c^*){% endmathjax %}进行优化。在**策略改进**阶段，**演员损失**定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{L}_{\text{AC}}(\omega) = -\lambda \cdot \mathbb{E}_{s\sim \mathcal{D},\tilde{z}^*\sim \pi_{\omega}(\cdot|c^*),a^*_{\xi}\sim p_{\psi}(\cdot|\tilde{z}^*,c^*)}\bigg[ Q_{\theta_1(s,a_{\xi}^*)}\bigg] + \mathbb{E}_{(s,a)\sim \mathcal{D},\tilde{z}\sim \pi_{\omega}(\cdot|c),a_{\xi}\sim p_{\psi}(\cdot|\tilde{z},c)}\bigg[ (a - q_{\xi})^2 \bigg] 
{% endmathjax %}
在第一项中，{% mathjax %}a^*_{\xi}{% endmathjax %}是通过固定的最大优势条件{% mathjax %}\xi^* = 1{% endmathjax %}输入生成的**最优动作**，而{% mathjax %}a_{\xi}{% endmathjax %}是通过根据公式从评论家获得的**优势条件**{% mathjax %}\xi{% endmathjax %}得到的。同时，遵循`TD3+BC`的方法，在第一项中添加了**归一化系数**{% mathjax %}\lambda = \frac{\alpha}{\frac{1}{N}\sum_{(s_i,a_i)}|Q(s_i,a_i)|}{% endmathjax %}以保持`Q`值目标和**正则化**之间的**尺度平衡**，其中{% mathjax %}\alpha{% endmathjax %}是一个超参数，用于控制**归一化**`Q`**值**的规模。第一项鼓励在条件{% mathjax %}c^*{% endmathjax %}下的**最优策略**选择产生最高期望回报的动作，这与传统**强化学习**方法中的策略改进步骤一致。第二个**行为克隆项**明确对**优势感知策略**施加约束，确保**策略选择**符合由评论家确定的**优势条件**{% mathjax %}\xi{% endmathjax %}的样本动作。因此，具有**低优势条件**{% mathjax %}\xi{% endmathjax %}的次优样本不会干扰**最优策略**{% mathjax %}\pi_{\omega}(\cdot|c^*){% endmathjax %}的优化，并对相应策略{% mathjax %}\pi_{\omega}(\cdot|c){% endmathjax %}强制施加有效约束。
{% note warning %}
**注意**，在**策略评估**和改进过程中，**解码器**是固定的。`A2PO`实现选择`TD3+BC`作为其基础框架，以确保其稳健性。
{% endnote %}
