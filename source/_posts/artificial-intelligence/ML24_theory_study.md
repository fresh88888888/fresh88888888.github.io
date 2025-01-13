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
`A2PO`方法由两个关键步骤组成：**行为策略解耦**和**智能体策略优化**。在**行为策略解耦**阶段，使用**条件变分自编码器**(`CVAE`)解开**行为策略**，基于收集的**状态-动作对**的优势值条件下的**动作分布**。通过输入不同的优势值，新的**条件变分自编码器**(`CVAE`)允许**智能体**推断与各种**行为策略**相关的不同**动作分布**。然后，在**智能体**策略优化阶段，从优势条件导出的**动作分布**作为解耦的**行为策略**，建立一个**优势感知的策略约束**来指导**智能体**的训练。
{% note warning %}
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

为了**解决混合质量离线数据集中的约束冲突问题**，这里采用了`A2PO`的方法，通过**优势感知的策略约束**。具体而言，`A2PO`利用**条件变分自编码器**(`CVAE`)有效地解耦与各种**行为策略**相关的**动作分布**。这是通过将所有训练数据的优势值建模为条件变量来实现的。因此，**优势感知**的**智能体策略优化**可以集中于最大化高优势值，同时遵循由混合质量数据集施加的**解耦分布约束**。实验结果表明，`A2PO`成功解耦了潜在的**行为策略**，并显著优于先进的**离线强化学习**竞争者。**局限性**：`A2PO`的局限性在于它在训练过程中引入了`CVAE`，这可能导致相当大的时间开销。

#### 深度强化学习—结构

**强化学习**(`RL`)在**深度神经网络**(`DNN`)强大的**函数逼近**能力的支持下，在众多应用中取得了显著成功。然而，它在应对各种现实场景方面的实用性仍然有限，这些场景具有多样化和不可预测的动态、噪声信号以及庞大的状态和**动作空间**。这一局限性源于**数据效率低、泛化能力有限、安全保障缺失**以及**缺乏可解释性**等多个因素。为克服这些挑战并改善在这些关键指标上的表现，一个有前景的方向是将关于问题的额外结构信息纳入**强化学习**的学习过程中。各种**强化学习**的子领域已提出了归纳整合的方法。将这些不同的方法整合到一个统一框架下，阐明结构在学习问题中的作用，并将这些方法分类为不同的结构整合模式。并为**强化学习**研究奠定了**设计模式**视角的基础。

**强化学习**(`RL`)在**序列决策**和**控制问题**中发挥了重要作用，例如游戏、机器人操作和优化化学反应。大多数传统**强化学习**研究集中在设计解决由任务固有动态引发的**序列决策问题**的**智能体**，例如控制套件`OpenAI Gym`中的小车倒立摆任务所遵循的微分方程。然而，当环境发生变化时，它的性能就会显著下降。此外，将**强化学习智能体**部署于还面临额外挑战，例如复杂的动态、难以处理且计算成本高昂的状态和**动作空间**，以及**噪声奖励信号**。因此，**强化学习**的研究开始划分为两种范式来解决这些问题：**泛化**，开发出能够解决更广泛问题的方法，其中**智能体**在各种任务和环境中进行训练；**可部署性**，专门针对具体现实问题而设计的方法，如特征工程、计算预算优化和安全性。**泛化**与**可部署性**的交集涵盖了需要处理任务多样性的同时又能针对特定应用进行部署。为了促进这一领域的研究，`Mannor`和`Tamar`主张采用以**设计模式**为导向的方法，将方法抽象为专门针对特定问题的模式。

**强化学习**(`RL`)的**设计模式**之路上，对**设计决策**与其适用问题属性之间关系的理解存在一些空白。尽管使用状态抽象来处理**高维空间**的决策似乎是显而易见的，但对于使用**关系神经架构**来解决某些问题的决策却并不那么显而易见。为此，理解如何将额外的**领域知识**融入**学习流程**，将为这一过程增添原则性的支撑。学习问题本身的结构，包括**状态空间**、**动作空间**、**奖励函数**或**环境动态的先验知识**，是**领域知识**的重要来源。
尽管这些方法在**强化学习**的发展历史中一直是研究主题，但在**深度强化学习**中实现这些目标的方法却散落在现代**强化学习**研究的各个子领域中。融入结构意味着利用关于可分解性的附加信息，以提高样本效率、泛化能力、可解释性和安全性。一个**强化学习**(`RL`)**智能体**可以选择适合学习者当前状态和学习目标的学习材料、活动和评估。这种场景充满了**结构特性**和**分解**，例如学习风格或学习者的隐性技能熟练度、学习项目中的**知识领域**之间的关系，以及模块化内容交付机制。虽然可以通过将问题视为一个整体来构建**马尔科夫决策过程**(`MDP`)，但这并不一定是最有效的解决方案。相反，可以通过不同方式构建问题，其中关于这种可分解性的**先验知识**可以将**归纳偏见**编码到**强化学习**(`RL`)**智能体**中。关于分解的**先验知识**还可以通过辅助方法发现，例如**大语言模型**(`LLMs`)，这些模型能够分析大量教育内容，提取关键概念、学习目标和难度水平。将附加信息融入学习流程，例如使用`LLM`生成内在**奖励**，可以提高**强化学习**(`RL`)**智能体**收敛速度，使其对问题变化具有**鲁棒性**，并帮助提高**安全性**和**可解释性**。
{% asset_img ml_2.png "强化学习框架-结构" %}

**强化学习**(`RL`)的辅助信息可用于提高样本效率、泛化性、可解释性很安全性等指标。辅助信息的另一个特定来源是**可分解性**，包括**潜在空间**(`Latent`)、**因子化**(`Factored`)、**关系型**(`Relational`)和**模块化**(`Modular`)。强化学习的结构大致分为`7`种**设计模式**：
- **抽象**(`Abstraction`)：通过简化环境或任务，将复杂问题转化为更易处理的形式。
- **增强**(`Augmentation`)：利用额外的信息或功能来丰富智能体的学习过程。
- **辅助优化**(`Auxiliary Optimization`)：引入辅助任务以促进主任务的学习。
- **辅助模型**(`Auxiliary Model`)：使用辅助模型来提供额外的信息或指导。
- **仓储**(`Warehouse`)：利用存储机制来管理和重用经验。
- **环境生成**(`Environment Generation`)：动态生成环境以适应不同的学习需求。
- **显式设计**(`Explicitly Designed`)：针对特定问题设计特定的解决方案。

在实际应用中，例如一个出租车服务的**强化学习**(`RL`)**智能体**，需要学习城市的布局、交通模式和乘客行为等信息。直接学习所有这些信息可能会让智能体感到不知所措。因此，通过将问题分解为更易处理的部分并在学习管道中融入结构，可以使问题变得更加可管理。通过将**结构假设**与**分解方法**结合，**强化学习模型**不仅可以提高效率，还能变得更加智能和适应现实世界的挑战。

**序列决策问题**通常使用**马尔可夫决策过程**(`MDP`)的概念进行形式化，可以写成一个五元组{% mathjax %}\mathcal{M} = \langle \mathcal{S},\mathcal{A},R,P,p \rangle{% endmathjax %}。在任何时间步，环境处于状态{% mathjax %}s\in \mathcal{S}{% endmathjax %}，其中{% mathjax %}p{% endmathjax %}是**初始状态分布**。**智能体**采取一个动作{% mathjax %}a\in\mathcal{A}{% endmathjax %}，使环境转移到一个新的状态{% mathjax %}s'\in \mathcal{S}{% endmathjax %}。随机转移函数控制这种转移的动态，表示为{% mathjax %}P\;:\;\mathcal{S}\times \mathcal{A}\rightarrow \Delta(\mathcal{A}){% endmathjax %}，它以状态{% mathjax %}s{% endmathjax %}和动作{% mathjax %}a{% endmathjax %}为输入，输出一个关于后续状态的**概率分布**{% mathjax %}\Delta(\cdot){% endmathjax %}，从中可以抽样得到后续状态{% mathjax %}s'{% endmathjax %}。对于每个转移，**智能体**会获得一个奖励{% mathjax %}\mathcal{R}:\mathcal{S}\times \mathcal{A}\rightarrow \mathbb{R}{% endmathjax %}，其中{% mathjax %}R\in \mathcal{R}{% endmathjax %}。序列{% mathjax %}(s,a,r,s'){% endmathjax %}被称为一次**经验**。**智能体**根据策略{% mathjax %}\pi\;:\;\mathcal{S}\rightarrow \Delta(\mathcal{A}){% endmathjax %}行动，该策略在**策略空间**{% mathjax %}\Pi{% endmathjax %}中生成给定状态下的**动作概率分布**。这是一个确定性策略的**德尔塔分布**，这意味着该策略输出一个单一的动作。使用当前策略，**智能体**可以反复生成经验，而这样的经验序列也称为轨迹{% mathjax %}(\tau){% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\tau = \{(s_t,a_t,r_t,s_{t+1})\}_{t\in [t_0,t_{T-1}]}\;\forall(s,a,r,s)\in \mathcal{S}\times \mathcal{A}\times \mathcal{R}\times \mathcal{S}
{% endmathjax %}
在**情节强化学习**(`episodic RL`)中，轨迹由在多个情节中收集的经验组成，每个情节都会重置环境。相对而言，在**持续设置**(`continual settings`)中，轨迹包含在单个情节中收集的一段时间内的经验。轨迹{% mathjax %}\tau{% endmathjax %}中的奖励可以累积成一个称为**回报**(`return`){% mathjax %}G{% endmathjax %}的**期望总和**，该回报可以为任何起始状态{% mathjax %}s{% endmathjax %}计算如下：
{% mathjax '{"conversion":{"em":14}}' %}
G(\pi,s) = \mathbb{E}_{(s_0=s,a_1,r_1,\ldots)\sim \pi}\bigg[ \sum\limits_{t=0}^{\infty} r_t \bigg]
{% endmathjax %}
为了使公式中的总和可处理，假设问题的时间范围为固定长度{% mathjax %}T{% endmathjax %}（有限时间回报），即轨迹在{% mathjax %}T{% endmathjax %}步后终止，要么通过**折扣因子**{% mathjax %}\gamma{% endmathjax %}来折扣未来的奖励（无限时间回报）。然而，折扣也可以应用于有限时间范围。解决一个**马尔可夫决策过程**(`MDP`)相当于确定策略{% mathjax %}\pi^*\in \Pi{% endmathjax %}，以最大化其轨迹的**回报期望**。这个期望可以通过（**状态-动作**）**值函数**{% mathjax %}Q\in \mathcal{Q}{% endmathjax %}来捕捉。给定一个策略{% mathjax %}\pi{% endmathjax %}，这个期望可以递归地写成：
{% mathjax '{"conversion":{"em":14}}' %}
Q^{\pi}(s,a)= \mathbb{E}_{\pi}[\sum\limits_{t=0}^T r_t|s_0 = s,a_0 = a]= \mathbb{E}_{\pi}[R(s,a) + \gamma\mathbb{E}_{a'\sim \pi(\cdot|s')}[Q^{\pi}(s',a')]]
{% endmathjax %}
因此，目标现在可以表述为寻找一个能够最大化{% mathjax %}Q^{\pi}(s,a){% endmathjax %}的**最优策略**。
{% mathjax '{"conversion":{"em":14}}' %}
\pi^*\in \underset{\pi\in \Pi}{\text{arg }\max}Q^{\pi}(s,a)\; \forall(s,a)\in \mathcal{S}\times \mathcal{A}
{% endmathjax %}
我们还考虑**部分可观测马尔可夫决策过程**(`POMDP`)，它建模了状态无法完全观察的情况。`POMDP`被定义为一个七元组{% mathjax %}\mathcal{M} = \langle \mathcal{S},\mathcal{A},\mathcal{O},R,P,\xi,p \rangle{% endmathjax %}，其中{% mathjax %}\mathcal{S},\mathcal{A},R,P,p {% endmathjax %}的定义与上述相同。**智能体**现在不是观察状态{% mathjax %}s\in \mathcal{S}{% endmathjax %}，而是可以访问通过**发射函数**{% mathjax %}\xi : \mathcal{S}\times \mathcal{A}\rightarrow \Delta(\mathcal{O}){% endmathjax %}从实际状态生成的观察{% mathjax %}o\in \mathcal{O}{% endmathjax %}。因此，观察在经验生成过程中取代了状态的角色。然而，解决`POMDP`需要维护一个额外的**信念**，因为多个{% mathjax %}(s,a){% endmathjax %}可以导致相同的{% mathjax %}o{% endmathjax %}。

**强化学习**(`RL`)算法的任务是通过模拟其**转移动态**{% mathjax %}P(s'|s,a){% endmathjax %}和**奖励函数**{% mathjax %}R(s,a){% endmathjax %}与**马尔可夫决策过程**(`MDP`)进行交互，并学习最优策略完成的。在**深度强化学习**中，策略是一个**深度神经网络**，用于生成轨迹{% mathjax %}\tau{% endmathjax %}。我通过最小化目标{% mathjax %}J{% endmathjax %}来优化策略。一个`MDP`的模型{% mathjax %}\hat{\mathcal{M}}{% endmathjax %}允许**智能体**通过模拟生成经验来规划轨迹。使用这种模型的**强化学习**方法被归类为基于模型的**强化学习**(Model-Based RL)。另一方面，如果没有这样的模型，则需要直接从经验中学习策略，这类方法则属于**无模型强化学习**(`Model-Free RL`)。
**强化学习**方法还可以根据目标{% mathjax %}J{% endmathjax %}的类型进行分类。使用**值函数**的方法，以及相应的**蒙特卡洛估计**或**时间差分**(`Temporal Difference, TD`)**误差**，用于学习策略，这类方法属于**基于值的强化学习**(`Value-Based RL`)。**时间差分方法**中的一个关键思想是**自举**(`bootstrapping`)，它使用已学习的值估计来改善前一个状态的估计。**在线策略方法**直接更新生成经验的策略，而**离线策略方法**则使用单独的策略来生成经验。基于策略的方法直接对策略进行参数化，并使用**策略梯度定理**来创建目标{% mathjax %}J{% endmathjax %}。实践中的**强化学习**方法的一个核心研究主题集中在通过迭代学习上述一个或多个量来近似全局解决方案，使用**监督学习**和**函数近似**。使用管道的概念来讨论不同的**强化学习**方法。下图展示了**强化学习**管道的结构。**管道**可以定义为一个数学元组{% mathjax %}\omega = \langle \mathcal{S},\mathcal{A},R,PQ,\pi,\hat{\mathcal{M}},J,\mathcal{E} \rangle{% endmathjax %}，其中所有定义与之前相同。为了求解**马尔可夫决策过程**(`MDP`)，管道在给定环境{% mathjax %}\mathcal{E}{% endmathjax %}的情况下运作，通过将状态{% mathjax %}s\in \mathcal{S}{% endmathjax %}作为输入并产生动作{% mathjax %}a\in \mathcal{A}{% endmathjax %}作为输出。环境根据动态{% mathjax %}P{% endmathjax %}和**奖励函数**{% mathjax %}R{% endmathjax %}运作。管道可能通过直接与环境{% mathjax %}\mathcal{E}{% endmathjax %}交互来生成经验，即从经验中学习，或通过模拟已学习的环境模型{% mathjax %}\hat{\mathcal{M}}{% endmathjax %}来生成经验。优化过程涵盖当前策略{% mathjax %}\pi{% endmathjax %}、其值函数{% mathjax %}Q{% endmathjax %}、奖励{% mathjax %}R{% endmathjax %}和学习目标{% mathjax %}J{% endmathjax %}之间的相互作用。
{% asset_img ml_3.png "强化学习管道" %}

