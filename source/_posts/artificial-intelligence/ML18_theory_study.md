---
title: 机器学习(ML)(十八) — 强化学习探析
date: 2024-11-25 10:10:11
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

#### 介绍

**强化学习**(`Reinforcement Learning, RL`)是一种**机器学习**的范式，主要关注**智能体**(`agent`)如何通过与环境的互动来学习最优策略，以最大化累积奖励。与**监督学习**和**无监督学习**不同，**强化学习**并不依赖于标注数据，而是通过**试错**(`trial and error`)的方法来**优化决策**。在**强化学习**中，主要涉及以下几个核心要素：**智能体**(`Agent`)，执行动作以影响环境的实体；**环境**(`Environment`)，**智能体**所处的外部系统，它对**智能体**的动作做出反应并提供反馈；**状态**(`State`)，描述环境在某一时刻的情况，**智能体**根据当前状态做出决策；**动作**(`Action`)，**智能体**在特定状态下可以选择的行为；**奖励**(`Reward`)，环境对**智能体**行为的反馈信号，通常是一个标量值，用于评估该行为的好坏；**策略**(`Policy`)，定义了**智能体**在特定状态下选择动作的规则，可以是确定性的也可以是随机性的；**价值函数**(`Value Function`),用于评估在某一状态下，**智能体**能够获得的长期回报期望。
<!-- more -->

**强化学习**(`RL`)的工作原理：**强化学习**(`RL`)的核心在于通过与环境的互动来学习。**智能体**在每个时间步选择一个动作，然后环境根据这个动作返回新的**状态**和**奖励**。**智能体**根据这些反馈调整其策略，以期在未来获得更高的累积奖励。这一过程通常涉及到以下几个步骤：1、**观察当前状态**；2、**选择一个动作**，依据当前策略；3、**执行该动作**，并接收新的状态和奖励；4、**更新策略**，以优化未来的决策。这种循环过程使得**智能体**能够逐渐改善其决策能力，从而达到最大化长期收益的目标。

**强化学习**(`RL`)已经在多个领域取得了显著成就，包括但不限于：
- **游戏**：如`AlphaGo、AlphaStar和OpenAI Five`等，这些系统通过**强化学习技术**在复杂游戏中击败了人类顶级选手。
- **机器人控制**：利用**强化学习**使机器人能够自主学习复杂任务，如抓取物体、行走等。
- **推荐系统**：通过用户反馈优化推荐算法，提高用户满意度。
- **金融交易**：在股票市场中应用**强化学习**进行自动化交易策略优化。

**强化学习**(`RL`)算法大致可以分为两类：
- **有模型学习**(`Model-Based Learning`)：**智能体**尝试构建**环境模型**，并利用该模型进行**规划**和**决策**。
- **无模型学习**(`Model-Free Learning`)：直接从环境交互中学习，不构建**环境模型**，常见方法包括`Q-learning`和**策略梯度**方法等。

如果假设脚下有一张五美元的钞票，这时可以弯腰捡起，或穿过街区，步行半小时捡起一张十美元的钞票。你更愿意选择哪一个？十美元比五美元多，但相比于步行半小时拿这张十美元，也许直接捡起五美元更方便。**回报**(`return captures`)的概念表明，更快获得奖励比需要花很长时间获得奖励更有价值。来看看它究竟是如何运作的？这里有一个火星探测器的例子。如果从状态`4`开始向左走，我们看到从状态`4`开始的第一步获得的奖励为`0`，从状态`3`开始的奖励为`0`，从状态`2`开始的奖励为`0`，状态`1`（最终状态）获得奖励为`100`。**回报**(`return captures`)定义为这些奖励的总和，但需要加权一个因子，称为**折扣因子**。**折扣因子**是一个略小于`1`的实数。这里选择`0.9`作为**折扣因子**。第一步的奖励加权为0，第二步的奖励是{% mathjax %}0.9 \times 0{% endmathjax %}，第三步的奖励是{% mathjax %}0.9^2 \times 0{% endmathjax %}，第三步的奖励是{% mathjax %}0.9^3 \times 100{% endmathjax %}。最终奖励的加权和为`72.9`。假设第一步获得奖励为{% mathjax %}R_1{% endmathjax %}，在第二步获得奖励为{% mathjax %}R_2{% endmathjax %}，第三步获得奖励为{% mathjax %}R_3{% endmathjax %}，那么**回报**(`return captures`)为{% mathjax %}R_1 + \gamma R_2 + \gamma^2 R_3 + \ldots{% endmathjax %}。**折扣因子**({% mathjax %}\gamma{% endmathjax %})的作用是让强化学习算法能够在做出决策时平衡当前和未来的收益。**回报**(`return captures`)将主要取决于第一个奖励，即{% mathjax %}R_1{% endmathjax %}，少一点的**回报**(`return captures`)归于第二步的奖励，即{% mathjax %}\gamma R_2{% endmathjax %}，奖励更少来自于第三步，即{% mathjax %}\gamma^2 R_3 {% endmathjax %}，由此越早获得奖励，**总回报**就越高。在许多**强化学习**算法中，**折扣因子**通常设为接近1的实数，例如`0.9、0.99、0.999`。在使用的示例中，将折扣因子设置为`0.5`。这会大大降低奖励，每经过一个额外的时间戳，获得的奖励只有前一步奖励的一半。如果{% mathjax %}\gamma = 0.5{% endmathjax %}，则上述示例中的**回报**(`return captures`)将是{% mathjax %}0\times + 0.5\times 0 + 0.5^2\times + 0.5^3\times 100 = 12.5{% endmathjax %}。在金融应用中，**折扣因子**还有一个非常自然的解释，即利率或货币的时间价值。如果你今天能得到一美元，那么它的价值可能比你未来只能得到一美元要高一点。因为即使是今天的一美元，你也可以存入银行，赚取一些利息，一年后最终会多一点钱。对于金融应用，**折扣因子**表示未来`1`美元与今天的`1`美元相比少了多少。**回报**(`return captures`)取决于采取的**动作**(`Action`)。如果机器人从状态`4`开始，**回报**(`return captures`)是`12.5`，如果它从状态`3`开始，回报将是`25`，因为它早一步到达`100`的奖励。如果从状态`2`开始，回报将是`50`。如果从状态`1`开始，那么它会获得`100`的奖励，因此没有折扣。如果从状态`1`开始，回报将是`100`，如果你从状态`6`开始，那么回报是`6.25`。现在，如果采取不同的动作，回报实际上会有所不同。总而言之，**强化学习**(`RL`)中的**回报**(`return captures`)是系统获得的奖励的总和，由**折扣因子加权**，其中远期的奖励由**折扣因子**的更高次方加权。在我们讨论的例子中，所有奖励都是`0`或正数。但如果奖励是负数，那么**折扣因子**实际上会激励系统将**负奖励**尽可能推迟到未来。以金融为例，如果你必须向某人支付`10`美元，那么这可能是`-10`的负奖励。但如果你可以将付款推迟几年，那么你实际上会更好，因为几年后的`10`美元，由于利率，实际上价值低于今天支付的`10`美元。对于具有**负奖励**的系统，它会导致算法将**奖励**尽可能推迟到未来。

**强化学习**算法如何选择动作呢？在**强化学习**中，目标是提出一个称为**策略**(`Policy`){% mathjax %}\pi{% endmathjax %}函数，它的工作原理是将状态{% mathjax %}s{% endmathjax %}作为输入并将其映射到它希望的某个动作{% mathjax %}a{% endmathjax %}。例如，如果处于状态`2`，那么它会将我们映射到左侧操作。如果处于状态`3`，策略会向左走。如果处于状态`4`，策略会向左走，如果你处于状态`5`，策略会向右走。{% mathjax %}a = \pi(s){% endmathjax %}，代表**策略函数**{% mathjax %}\pi{% endmathjax %}在状态{% mathjax %}s{% endmathjax %}下执行什么动作。**强化学习**的目标是找到一个策略{% mathjax %}\pi{% endmathjax %}，在每个状态下需要采取什么行动，能够最大化**回报**(`return captures`)。**强化学习**在应用中被称为**马尔可夫决策过程**(`MDP`)，**马尔可夫决策过程**(`MDP`)中**马尔可夫**指的是**未来只取决于当前状态，而不取决于当前状态之前的任何状态**。也就是说，在**马尔可夫决策过程**中，未来只取决于你现在的位置。
{% asset_img ml_1.png %}

#### 状态/动作值函数

在**强化学习**中，**状态/动作值函数**(`State Action Value Function`)是一个关键概念，用于评估在特定状态下采取某个动作的期望回报。它通常用符号{% mathjax %}Q(s,a){% endmathjax %}表示，其中{% mathjax %}s{% endmathjax %}代表当前状态，{% mathjax %}a{% endmathjax %}代表在该状态下采取的动作。**状态/动作值函数**{% mathjax %}Q(s,a){% endmathjax %}定义为在状态{% mathjax %}s{% endmathjax %}下采取动作{% mathjax %}a{% endmathjax %}后，按照某一**策略**(`Policy`)所能获得的期望累积奖励。具体来说，它可以表示为：{% mathjax %}Q(s,a) = \mathbb{E}[R_t|S_t = s,A_t = a]{% endmathjax %}，其中，{% mathjax %}R_t{% endmathjax %}是从时间{% mathjax %}t{% endmathjax %}开始的未来奖励的总和，{% mathjax %}\mathbb{E}{% endmathjax %}表示期望值。**状态/动作值函数**的作用：**决策支持**，通过评估不同动作在特定状态下的价值，**智能体**(`Agent`)可以选择最优动作，以最大化其长期回报；**策略改进**，在策略迭代中，**状态/动作值函数**用于更新策略，使得**智能体**(`Agent`)能够逐步学习到更优的行为方式。

假设我们有一个简单的`K-`**臂赌博机**(`k-armed bandit`)，其中有三个不同的动作{% mathjax %}a_1,a_2,a_3{% endmathjax %}，每个动作都有一个未知的奖励分布，**智能体**(`Agent`)的目标是通过选择不同的动作来最大化其获得的累积奖励。在这个例子中，**状态/动作值函数**{% mathjax %}Q(s,a){% endmathjax %}可以定义为选择动作{% mathjax %}a{% endmathjax %}时所期望的奖励，例如，假设我们在某个状态{% mathjax %}s{% endmathjax %}下选择动作{% mathjax %}a_1{% endmathjax %}，我们可以表示为：{% mathjax %}Q(s,a_1) = \mathbb{E}[R_t|S_t = s,A_t = a_1]{% endmathjax %}，其中{% mathjax %}R_t{% endmathjax %}是时间步{% mathjax %}t{% endmathjax %}获得的奖励，实际操作步骤：
- **初始化**：假设我们初始化每个动作的值函数为`0`：{% mathjax %}Q(s,a_1) = 0,Q(s,a_2) = 0,Q(s,a_3) = 0{% endmathjax %}。
- **选择动作**：**智能体**(`Agent`)根据当前的**状态/动作值函数**选择一个动作。例如，可以使用**ε-贪婪策略**（以概率`ε`随机选择一个动作，以探索新的可能性）。
- **观察奖励**：执行选定的动作后，**智能体**(`Agent`)会收到一个即时奖励。例如，如果选择了{% mathjax %}a_1{% endmathjax %}，并获得了奖励{% mathjax %}r_1{% endmathjax %}。
- **更新值函数**：使用**样本平均法**更新该动作的值函数：如果选择了{% mathjax %}a_1{% endmathjax %}，则更新公式为{% mathjax %}Q(s,a_1) = Q(s,a_1) + \frac{1}{N(a_1)}(r_1 - Q(s,a_1)){% endmathjax %}，其中{% mathjax %}N(a_1){% endmathjax %}是已选择该动作的次数。
- **重复过程**：**智能体**(`Agent`)不断重复选择、执行、观察和更新的过程，逐渐收敛到每个动作的真实期望奖励。

通过这个**K-臂赌博机**的问题示例，我们可以看到，**状态/动作值函数**不仅帮助**智能体**评估不同动作的价值，还能指导其在复杂环境中做出更好的**决策**。这种方法在许多强化学习算法中都得到了广泛应用，如`Q`学习和深度`Q`网络(`DQN`)等。**状态/动作值函数**是**强化学习**中的核心组成部分，它帮助**智能体**(`Agent`)在复杂环境中做出有效决策。通过理解和应用这一概念，**智能体**(`Agent`)能够不断优化其行为，从而实现更高的累积奖励。

#### 贝尔曼方程

**贝尔曼方程**(`Bellman Equation`)是**强化学习**和**动态规划**中的一个核心概念，它描述了在给定状态下，如何通过选择最佳动作来最大化未来的期望回报。**贝尔曼方程**为决策过程提供了一种递归关系，使得我们能够从当前状态推导出未来状态的价值。在**强化学习**中，**贝尔曼方程**通常分为两种类型：**状态值函数**(`State Value Function`)和**动作值函数**(`Action Value Function`)。
- **状态值函数**(`State Value Function`)：**状态值函数**{% mathjax %}V(s){% endmathjax %}表示在状态{% mathjax %}s{% endmathjax %}下，遵循某一策略{% mathjax %}\pi{% endmathjax %}所能获得的期望回报，其贝尔曼方程可以表示为：{% mathjax %}V(s) = \sum\limits_{a}\pi(a|s)\sum\limits_{s',r} P(s',r|s,a)[r + \gamma V(s')]{% endmathjax %}，其中{% mathjax %}\pi(a|s){% endmathjax %}表示为在状态{% mathjax %}s{% endmathjax %}下选择动作{% mathjax %}a{% endmathjax %}的概率，{% mathjax %}P(s',r|s,a){% endmathjax %}表示为在状态{% mathjax %}s{% endmathjax %}下采取动作{% mathjax %}a{% endmathjax %}后转移到状态{% mathjax %}s'{% endmathjax %}并获得奖励{% mathjax %}r{% endmathjax %}的概率，{% mathjax %}\gamma{% endmathjax %}为折扣因子，介于`0~1`之间，用于权衡未来奖励的重要性。
- **动作值函数**(`Action Value Function`)：**动作值函数**{% mathjax %}Q(s,a){% endmathjax %}表示在状态{% mathjax %}s{% endmathjax %}下采取动作{% mathjax %}a{% endmathjax %}后，遵循某一策略所能获得的期望回报。为了描述**贝尔曼方程**，我将使用以下符号。使用{% mathjax %}s{% endmathjax %}来表示当前状态。使用{% mathjax %}R(s){% endmathjax %}表示当前状态的奖励。在之前的示例中，状态`1`的奖励{% mathjax %}R(1) = 100{% endmathjax %}、状态`2`的奖励为{% mathjax %}R(2) = 0{% endmathjax %}、状态`6`的奖励是{% mathjax %}R(6) = 40{% endmathjax %}。使用{% mathjax %}a{% endmathjax %}表示当前动作，即在状态{% mathjax %}s{% endmathjax %}中采取的动作。执行动作{% mathjax %}a{% endmathjax %}后进入某个新的状态。例如，状态`4`采取左侧的动作，那么进入状态`3`。用{% mathjax %}s'{% endmathjax %}表示当前状态{% mathjax %}s{% endmathjax %}执行动作{% mathjax %}a{% endmathjax %}后进入的状态。用{% mathjax %}a'{% endmathjax %}表示状态{% mathjax %}s'{% endmathjax %}中执行的动作。贝尔曼方程可以表示为：{% mathjax %}Q(s,a) = \sum\limits_{s',r} P(s',r|s,a)[r + \gamma V(s')]{% endmathjax %}，如果使用最优策略，可以写成{% mathjax %}Q^{*}(s,a) = \sum\limits_{s',r} P(s',r|s,a)[r + \gamma\underset{a'}{\max}Q^{*}(s',a')]{% endmathjax %}。

**贝尔曼方程**(`Bellman Equation`)的重要性体现在以下几个方面：**递归结构**，它将一个复杂问题分解为更简单的子问题，使得我们可以通过**动态规划**的方法来求解；**最优性原则**，**贝尔曼方程**体现了**最优策略**的特性，即在每个决策点上选择能最大化未来回报的动作；**强化学习算法基础**，许多强化学习算法（如`Q-learning`、`SARSA`等）都是基于**贝尔曼方程**进行更新和优化的。**贝尔曼方程**是**强化学习**和**动态规划**中的一个基本工具，它为**智能体**(`Agent`)提供了一种系统的方法来评估和优化决策过程。

#### K-臂赌博机

`K-`**臂赌博机**(`K-Armed Bandit`)问题是**多臂赌博机**(`Multi-Armed Bandit`)问题的一种特例，具体指有固定数量{% mathjax %}K{% endmathjax %}的臂。每个臂都有一个未知的概率分布，用于生成随机奖励。`K-`**臂赌博机**明确规定了臂的数量为{% mathjax %}K{% endmathjax %}，例如，{% mathjax %}K = 3{% endmathjax %}表示有三个可供选择的臂。问题通常集中在如何在有限次尝试中找到最佳臂，以最大化总回报。`K-`**臂赌博机**通常假设每个臂的奖励分布是**独立且同分布**(`IID`)的。首先，我们需要了解**反馈类型**（奖励/惩罚）之间的区别，因为奖励是**代理**(`Agent`)的一种**反馈类型**，如下图所示，**代理**(`Agent`)与环境交互，在每一个时间步进行观察({% mathjax %}O_t{% endmathjax %})，并基于这些观察执行动作，这里包含了`4`种动作，分别为{% mathjax %}A_1,A_2,A_3{% endmathjax %}和{% mathjax %}A_4{% endmathjax %},假设最佳动作为{% mathjax %}a^{*} = A_2{% endmathjax %}，但**代理**(`Agent`)选择了{% mathjax %}A_4{% endmathjax %}。
{% asset_img ml_2.png "4-臂赌博机在简单环境中的代理交互示例图" %}

这里指导的反馈动作为{% mathjax %}A_4{% endmathjax %}，是错误的。而最佳动作为{% mathjax %}A_2{% endmathjax %}，这种情况在**监督学习**任务中经常发生，对于反馈的评估取决于采取的动作，这在**强化学习**任务中很常见。假设你需要反复从多个选项（动作）中进行选择。每次做选择之后，根据**平稳性概率分布**获得奖励分值（或惩罚分值）。“平稳性”是指**奖励**和**转换**的概率分布随时间保持不变。为了简化，**强化学习**(`RL`)算法依赖于**平稳性**和**无模型**方法，例如`Q-learning`。然而，这种假设在实际应用中并不总是成立，因此还有针对**非平稳环境**的算法。对于{% mathjax %}k{% endmathjax %}个有效动作的每一个都有预期的平均奖励，称作动作的价值分数，假设在时间步{% mathjax %}t{% endmathjax %}选取的动作为{% mathjax %}A_t{% endmathjax %}，这时获得的奖励分值为{% mathjax %}R_t{% endmathjax %}，定义{% mathjax %}q^{*}(a){% endmathjax %}为随机动作{% mathjax %}a{% endmathjax %}的价值分数。{% mathjax %}q^{*}(a){% endmathjax %}的意思是**代理**(`Agent`)在时间步{% mathjax %}t{% endmathjax %}时采取动作{% mathjax %}a{% endmathjax %}的预期奖励。数学上定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
q^{*}(a) = \mathbb{E}[R_t|A_t = a]
\end{align}
{% endmathjax %}
其中{% mathjax %}q^{*}(a){% endmathjax %}表示采取动作{% mathjax %}a{% endmathjax %}的预期奖励，这作为衡量`K-`**臂赌博机**动作评估的基础。如果我们知道要采取的最佳动作，那么问题就很简单了，因为我们通常都会选择最佳的动作。如果没有这些信息，那么就必须评估每个动作的价值分数，在时间步{% mathjax %}t{% endmathjax %}时{% mathjax %}Q_t(a){% endmathjax %}应该更加接近{% mathjax %}q^{*}(a){% endmathjax %}。在评估了动作价值分数之后，每个时间步至少有一个动作应该具有最高的评估值，这些动作被称为“贪婪动作”。选择“贪婪动作”会利用当前知识获得**即时奖励**，而选择“非贪婪动作”则会推动和改进评估值。通过采样平均值来评估动作价值分数，对于稳定`K-`**臂赌博机**问题非常有效，实际情况，在非平稳环境中，对于近期奖励赋予更大权重是有意义的。通常使用恒定步长参数，将样本平均方程重写为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
Q_{n+1}= Q_n + \alpha[R_n - Q_n]
\end{align}
{% endmathjax %}
其中，{% mathjax %}Q_{n+1}{% endmathjax %}表示第{% mathjax %}n{% endmathjax %}次奖励之后更新动作的奖励分值，{% mathjax %}Q_n{% endmathjax %}表示第{% mathjax %}n{% endmathjax %}次奖励之前的当前动作的奖励分值，{% mathjax %}\alpha{% endmathjax %}表示新信息覆盖旧信息程度的参数，{% mathjax %}R_n{% endmathjax %}表示在第{% mathjax %}n{% endmathjax %}步采取动作后获得奖励。此更新规则根据收到的奖励与当前评估之间的差异逐步调整动作奖励分值，并由学习率{% mathjax %}\alpha{% endmathjax %}做加权。此迭代过程允许**代理**(`Agent`)调整其价值分数来改进决策。为了评估 `K-`**臂赌博机**问题中的动作价值分数，需要使用以下方程，该方程将动作价值分数{% mathjax %}Q_t(a){% endmathjax %}表示为截至到时间步长{% mathjax %}t{% endmathjax %}时从该动作获得的奖励分数的均值。计算评估动作价值分数并根据这些评估做出选择的技术，称为**动作价值方法**。动作的真实价值对应于选择时的**平均奖励**，通过平均奖励来计算，如下式所示：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
Q_t(a) = \frac{\text{在时间步}t\text{之前采取的动作}a\text{时的奖励总和}}{\text{在时间步}t\text{之前采取的动作}a\text{的次数}}
\end{align}
{% endmathjax %}
{% mathjax %}Q_t(a){% endmathjax %}可以用数学公式来表示：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
Q_t(a) = \frac{\sum_{i=1}^{t-1}R_i\cdot\mathbf{1}_{\{A_i = a\}}}{\sum_{i=1}^{t-1}\mathbf{1}_{\{A_i = a\}}}
\end{align}
{% endmathjax %}
其中，{% mathjax %}\mathbf{1}_{\{A_i = a\}}{% endmathjax %}是指示函数(`indicator function`)，指示函数是一种在数学和统计学中广泛使用的函数，用于表示某个条件是否成立。它在概率论、统计推断和机器学习等领域中具有重要作用。指示函数(`indicator function`){% mathjax %}\mathbf{1}_{\{A_i = a\}}{% endmathjax %}用于计算在时间步{% mathjax %}t{% endmathjax %}之前采取的动作{% mathjax %}a{% endmathjax %}的次数，如：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathbf{1}_{\{A_i = a\}} = 
    \begin{cases}
      1 & \text{如果在时间步}t\text{采取动作}a \\
      0 & \text{否则}
    \end{cases}
\end{align}
{% endmathjax %}
如果上边公式的分母为`0`，{% mathjax %}Q_t(a){% endmathjax %}则使用默认值来设置。需要注意的是，当分母趋近于无穷大时，根据大数定律，{% mathjax %}Q_t(a){% endmathjax %}收敛到{% mathjax %}Q^{*}(a){% endmathjax %}。使用**样本平均法**，根据奖励样本的均值计算动作价值分数。此方法可能不是最有效的，但可以用于估算动作的起点。
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
A_t \equiv \text{arg}\;\underset{a}{\max}Q_t(a)
\end{align}
{% endmathjax %}
其中{% mathjax %}A_t{% endmathjax %}表示为在时间步{% mathjax %}t{% endmathjax %}时选取的动作{% mathjax %} {% endmathjax %}，即动作价值分数最高的{% mathjax %}Q_t(a){% endmathjax %}。**贪婪**利用当前的知识来最大化**即时奖励**。但是可能存在问题，因为它需要搜索所有动作才能知道它们的奖励。一个简单而有效的替代方法是大多数时候**贪婪**，但以很小的概率从所有动作中随机选择`ϵ`，这种方法称为`ϵ-`**贪婪方法**，它在搜索与利用之间取得平衡。这个方法确保所有动作都经过充分采样，以准确计算其真实价值，随着时间的推移，最终确定最佳的策略。这只是理论上的长期收益，并没有直接表明实际有效性。通过**抽样均值**来评估动作价值分数，对于大量的样本({% mathjax %}K > n{% endmathjax %})来说是低效的。更有效的方法是推导出{% mathjax %}Q_n{% endmathjax %}，现在，使用以下公式对**样本均值**的动作价值分数定义如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
Q_n \equiv \frac{R_1 + R_2 +\ldots+ R_{n-1}}{n-1}
\end{align}
{% endmathjax %}
这里可以递归地表达动作价值分数：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]
\end{align}
{% endmathjax %}
这个递归方程只需要对{% mathjax %}Q_n{% endmathjax %}和{% mathjax %}n{% endmathjax %}进行内存分配，对计算每个奖励的计算量极小。**动作价值分数**的一般形式为：`NewEstimate ← OldEstimate + StepSize[Target − OldEstimate]`，尽管存在噪音干扰，但目标仍会给出一个调整的首选方向。在这个示例中，目标是第{% mathjax %}n{% endmathjax %}个奖励。**样本均值**适用于平稳的**赌博机**问题。在非平稳环境中，**即时奖励**更为相关。使用恒定步长参数，该方程可以重写为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
Q_{n+1} = Q_n + \alpha[R_n - Q_n]
\end{align}
{% endmathjax %}
**动作价值分数**的初始值在学习过程中起着至关重要的作用。这些初始值会影响**代理**(`Agent`)做出的早期决策。虽然**样本均值方法**可以在每个动作被选择至少一次后减少这种初始偏差，但使用恒定步长学习率参数{% mathjax %}\alpha{% endmathjax %}的方法往往会随着时间的推移逐渐减少这种偏差。设置乐观的初始值可能会有利。通过分配更高的初始值，可以鼓励**代理**(`Agent`)尽早探索更多的动作。这是因为最初的乐观情绪使未尝试的动作看起来更有吸引力，从而促进探索，即使**代理**(`Agent`)使用的**贪婪策略**也是如此。这种方法有助于确保**代理**(`Agent`)在收敛到最终**策略**(`Policy`)之前调整完动作空间。然而，这种**策略**(`Policy`)需要仔细确定初始值，在标准做法中，初始值通常设置为0。初始值的选择应该反映出对潜在回报的合理猜测，如果管理不当，过于乐观的值可能会阻碍了**代理**(`Agent`)有效收敛。总体而言，乐观的初始值可以成为平衡**强化学习**中的搜索和利用的方法，鼓励更广泛的探索，并带来更优的长期**策略**(`Policy`)。由于与**动作价值分数**计算相关的不确定性，搜索是必不可少的。在`ϵ-greedy`方法中，非贪婪动作被无差别地探索。最好根据非贪婪动作的潜在**最优性**和**不确定性**有选择地探索非贪婪动作。基于**上置信界**(`UCB`)算法，**上置信界算法**是一种用于解决多臂赌博机问题的强化学习算法，旨在平衡探索与利用之间的权衡。`UCB`算法通过利用不确定性来选择动作，从而优化长期回报。动作的选择基于以下标准：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
A_t \equiv \text{arg}\;\underset{a}{\max}\Bigg[ Q_t(a)+ c\sqrt{\frac{\ln t}{N_t(a)}}\;\Bigg]
\end{align}
{% endmathjax %}
这里{% mathjax %}\ln t{% endmathjax %}是时间步{% mathjax %}t{% endmathjax %}的自然对数，{% mathjax %}c{% endmathjax %}控制探索，且{% mathjax %}c > 0{% endmathjax %}。{% mathjax %}N_t(a){% endmathjax %}是在时间步骤{% mathjax %}t{% endmathjax %}之前采取动作{% mathjax %}a{% endmathjax %}的次数。如果{% mathjax %}N_t(a) = 0{% endmathjax %}，则{% mathjax %}a{% endmathjax %}被视为价值分数最大化的动作。作为其价值评估的一部分，`UCB`将**不确定性**纳入其对动作价值分数上限的计算中。**置信度**由常数{% mathjax %}c{% endmathjax %}来控制。通过选择动作{% mathjax %}a{% endmathjax %}，与{% mathjax %}a{% endmathjax %}相关的不确定性会随着{% mathjax %}N_t(a){% endmathjax %}的增加而减少，而通过选择其他动作，与{% mathjax %}a{% endmathjax %}相关的不确定性会随着{% mathjax %}t{% endmathjax %}的增加而增加。使用**自然对数**，随着不确定性调整随时间而减少，最终会探索所有动作。频繁选取的动作的频率会随着时间的推移而在减少。

#### 有限马尔可夫决策过程

**马尔可夫决策过程**(`MDP`)提供了一个**顺序决策框架**，其中的动作会影响**即时奖励**以及未来结果。在**马尔可夫决策过程**(`MDP`)中，即时奖励于延迟奖励保持平衡，它的目标是是确定每个动作{% mathjax %}a{% endmathjax %}价值分数与**赌博机**问题不同，**马尔可夫决策过程**(`MDP`)的目标是计算在状态{% mathjax %}s{% endmathjax %}下采取动作{% mathjax %}a{% endmathjax %}的价值分数，换句话说就是采取最佳动作的情况下处于状态{% mathjax %}s{% endmathjax %}的价值分数。正确评估策略的长期效果需要计算这些特定状态的值，有限**马尔可夫决策过程**(`MDP`)由状态、动作和奖励组成({% mathjax %}S,A,R{% endmathjax %})。根据先前的状态和动作，将**离散概率分布**分配给随机变量{% mathjax %}R_t{% endmathjax %}和{% mathjax %}S_t{% endmathjax %}。使用随机变量{% mathjax %}R_t{% endmathjax %}和{% mathjax %}S_t{% endmathjax %}的概率，导出这些变量的方程，当一个动作的结果独立于过去的动作和状态，则被认为是**马尔可夫**。**马尔可夫特性**要求状态能够包括影响未来结果的整个过去互动的重要细节，此定义是**马尔可夫决策过程**(`MDP`)在**强化学习**中使用的基础。为了描述**马尔可夫决策过程**(`MDP`)的动态，我们使用**状态转换概率函数** {% mathjax %}p(s',r|s,a){% endmathjax %}，其定义如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
p(s',r|s,a) \equiv \text{Pr}\{S_t = s',R_t = r|S_{t-1} = s,S_{t-1} = a\}
\end{align}
{% endmathjax %}
其中函数{% mathjax %}p{% endmathjax %}被定义为**马尔可夫决策过程**(`MDP`)动态。可以从`4`个参数的**动态函数**{% mathjax %}p{% endmathjax %}中得出以下**状态转换概率**、**状态动作**和状态-动作-下一状态的预期奖励的三元组，如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
p(s',r|s,a) \equiv & \;\text{Pr}\{S_t = s',R_t = r|S_{t-1} = s,S_{t-1} = a\} = \sum\limits_{r\in R}p(s',r|s,a) \\
r(s,a) \equiv & \;\mathbb{E}\{R_t|S_{t-1} = s,A_{t-1} = a\} = \sum\limits_{r\in R}r\sum\limits_{r\in R}p(s',r|s,a) \\
r(s,a,s') \equiv & \;\mathbb{E}\{R_t|S_{t-1} = s,A_{t-1} = a,S_t = s'\} = \sum\limits_{r\in R}\frac{rp(s',r|s,a)}{p(s',r|s,a)}
\end{align}
{% endmathjax %}
动作(`action`)的概念涵盖与学习有关的任何决定，而**状态**(`state`)的概念涵盖可用于通知这些决定的任何信息。任何学习问题都可以归结为**代理**(`Agent`)与**环境**之间的三个信号：**动作**、**状态**和**奖励**。回报表示为{% mathjax %}G_t{% endmathjax %}，是从时间步{% mathjax %}t{% endmathjax %}开始获得的奖励累积总和。在时间步{% mathjax %}t{% endmathjax %}之后获得的奖励序列，它定义如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
G_t \equiv R_{t+1} + R_{t+2} +\ldots + R_{T}
\end{align}
{% endmathjax %}
其中{% mathjax %}G_t{% endmathjax %}是奖励序列的一个特殊函数。在{% mathjax %}T{% endmathjax %}处终止序列有何目的？顾名思义，**情景问题**是指**代理**与**环境**之间的交互自然地按顺序发生的问题，称为**情景**，而任务称为**情景任务**。正在进行的任务通常涉及在整个任务期间持续存在的交互，例如过程控制或机器人程序。由于**持续任务**中没有终止状态({% mathjax %}T = /infty{% endmathjax %}) ，因此**持续任务**的回报应以不同的方式定义。如果**代理**(`Agent`)持续获得奖励，则回报可能是无限的。对于**持续任务**，对于没有终止状态的持续任务，回报{% mathjax %}G_t{% endmathjax %}定义为**未来奖励**的折扣总和：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
G_t \equiv R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} +\ldots = \sum\limits_{k=0}^{\infty}\gamma^k R_{t+k+1}
\end{align}
{% endmathjax %}
其中{% mathjax %}\gamma{% endmathjax %}是**折扣因子**({% mathjax %}0 \leq \gamma \geq 1{% endmathjax %})。**折扣因子**{% mathjax %}\gamma{% endmathjax %}会影响未来奖励的当前价值。当{% mathjax %}\gamma < 1{% endmathjax %}时，**无限和**收敛到有限值。当{% mathjax %}\gamma = 0{% endmathjax %}时，**代理**(`Agent`)最大化**即时奖励**。当{% mathjax %}\gamma{% endmathjax %}接近 `1`时，未来的奖励会变得更有分量。将**回报递归**定义为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
G_t \equiv R_{t+1} + \gamma G_{t+1}
\end{align}
{% endmathjax %}
如果奖励非零且为常数，则**回报**是有限的，并且**折扣因子**{% mathjax %}\gamma < 1{% endmathjax %}。对**偶发任务**和**持续任务**使用一个公式：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
G_t \equiv \sum\limits_{k = t+1}^T\gamma^{k-t-1}R_k
\end{align}
{% endmathjax %}
如果{% mathjax %}T = \infty{% endmathjax %}或{% mathjax %}\gamma = 1{% endmathjax %}，则此公式适用于**情景任务**和**持续任务**。

#### 策略和价值函数

**价值函数**用于评估代理在特定状态（或在特定状态下采取的动作）的**预期回报**。根据所选取动作的不同，结果也会有所不同。**价值函数**与**策略**之间存在联系，而**价值函数**又与基于状态的动作相关。**价值函数**可分为以下两大类：
- **状态值函数**：{% mathjax %}v_{\pi}(s){% endmathjax %}是指策略{% mathjax %}\pi{% endmathjax %}指导下状态{% mathjax %}s{% endmathjax %}的**价值函数**，它是从状态{% mathjax %}s{% endmathjax %}开始并执行完策略{% mathjax %}\pi{% endmathjax %}之后的**预期回报**。
- **动作值函数**：在策略{% mathjax %}\pi{% endmathjax %}指导下, {% mathjax %}q_{\pi}(s,a){% endmathjax %}表示在状态{% mathjax %}s{% endmathjax %}采取动作{% mathjax %}a{% endmathjax %}的价值分数，它是从状态{% mathjax %}s{% endmathjax %}开始，采取动作{% mathjax %}a{% endmathjax %}，然后遵循策略{% mathjax %}\pi{% endmathjax %}的**预期回报**。

对于**马尔可夫决策过程**(`MDP`)来说，{% mathjax %}v{% endmathjax %}和{% mathjax %}q{% endmathjax %}定义为如下：**状态值函数**{% mathjax %}v_{\pi}(s){% endmathjax %}表示从状态{% mathjax %}s{% endmathjax %}开始并遵循策略{% mathjax %}\pi{% endmathjax %}的预期回报。它在数学上定义如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
v_{\pi}(s) \equiv \mathbb{E}_{\pi}\Bigg[\sum\limits_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t = s \Bigg],\;\text{for all s\in S}
\end{align}
{% endmathjax %}
**动作价值函数**{% mathjax %}q_{\pi}(s,a){% endmathjax %}表示从状态{% mathjax %}s{% endmathjax %}开始，采取动作{% mathjax %}a{% endmathjax %}，然后遵循策略{% mathjax %}\pi{% endmathjax %}的预期回报。其定义如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
q_{\pi}(s,a) \equiv \mathbb{E}_{\pi}[G_t|S_t = s,A_t = a] = \mathbb{E}_{\pi}\Bigg[\sum\limits_{k=0}^{\infty}\gamma^kR_{t+k+1}|S_t = s,A_t = a\Bigg]
\end{align}
{% endmathjax %}
需要注意{% mathjax %}v{% endmathjax %}和{% mathjax %}q{% endmathjax %}之间的区别，即{% mathjax %}q{% endmathjax %}取决于每个状态下采取的动态。{% mathjax %}q{% endmathjax %}有`10`个状态，每个状态有`8`个动作，因此{% mathjax %}q{% endmathjax %}需要`80`个函数，而{% mathjax %}v{% endmathjax %}只需要`10`个函数。遵循策略{% mathjax %}\pi{% endmathjax %}，如果**代理**(`Agent`)对每个状态的回报求均值，则均值收敛到{% mathjax %}v_{\pi}(s){% endmathjax %}。对每个动作的回报，则均值收敛到{% mathjax %}q_{\pi}(s,a){% endmathjax %}。在**蒙特卡罗方法**中，许多随机收益样本被均值化。这种方法不提供样本效率，需要为每个状态分别计算均值。通过使用参数较少的**参数化函数**可以改进计算。{% mathjax %}v{% endmathjax %}应以递归方式，编写如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
v_{\pi}(s) \equiv \mathbb{E}_{\pi}[G_t|s_t = s] = \mathbb{E}_{\pi}[R_{t+1} + \gammaG_{t+1}|s_t = s] = \sum\limits_a \pi(a|s)\sum\limits_{s'}\sum\limits_r p(s',r|s,a)[r + \gamma v_{\pi}(s')]
\end{align}
{% endmathjax %}
其中{% mathjax %}v_{\pi}{% endmathjax %}的**贝尔曼方程**。**贝尔曼方程**将状态的值与其潜在后继状态的值联系起来。初始状态的值等于预期的下一个状态的折扣值加上预期的奖励。

**状态值函数**{% mathjax %}v_{\pi}(s){% endmathjax %}和**动作值函数**{% mathjax %}q_{\pi}(s,a){% endmathjax %}在**强化学习**中发挥着不同的作用。在评估确定性**策略**或需要理解处于特定状态的价值时，使用**状态值函数**。在**策略评估**和**策略迭代**方法中，策略被明确定义，并且需要评估在策略下处于特定状态的性能，这时**状态值函数**非常有用。当存在许多动作时，使用**状态值函数**是有效的，因为它们只需要评估**状态值**即可降低复杂性。**动作价值函数**用于评估和比较在同一状态下发生不同动作的可能性。它们对于动作的选取很重要，例如在`Q-learnning`和`SARSA`中，**目标**是确定每种状态最合适的动作。由于**动作价值函数**考虑了不同动作的预期回报，因此它们在具有随机策略的环境中特别有用。此外，在处理**连续动作空间**时，**动作价值函数**可以提供对动作影响的更详细信息，有助于**策略**实施的微调。

考虑这样一个赌博场景：玩家从10美元开始，并面临有关下注金额的决定。此游戏说明了**强化学习**中的**状态**和**动作价值函数**。
- **状态价值函数**({% mathjax %}v_{\pi}(s){% endmathjax %})：**状态价值函数**{% mathjax %}v_{\pi}(s){% endmathjax %}量化给定策略{% mathjax %}\pi{% endmathjax %}时，状态{% mathjax %}s{% endmathjax %}的预期累积未来奖励。假设玩家有`5`美元：如果连续下注`1`美元，{% mathjax %}v_{\pi}(5) = 0.5{% endmathjax %}表示预期收益为`0.5`美元；如果持续下注`2`美元，{% mathjax %}v_{\pi}(5) = -1{% endmathjax %}表示预期损失`1`美元。
- **动作价值函数**{% mathjax %}q_{\pi}(s,a){% endmathjax %}：**动作价值函数**{% mathjax %}q_{\pi}(s,a){% endmathjax %}评估在状态{% mathjax %}s{% endmathjax %}下动作{% mathjax %}a{% endmathjax %}的预期累积未来奖励。例如：{% mathjax %}q_{\pi}(5,1) = 1{% endmathjax %}表示从`5`美元下注`1`美元可获得`1`美元的累计奖励；{% mathjax %}q_{\pi}(5,2) = -0.5{% endmathjax %}表示从`5`美元下注`2`美元，损失`0.5`美元。

这个赌博游戏场景强调了**状态**和**动作价值函数**在**强化学习**中的作用，指导动态环境中的最佳决策。

**价值函数**对策略创建偏序，允许基于**预期累积奖励**进行比较和排名。如果对于所有状态{% mathjax %}s{% endmathjax %}下的{% mathjax %}v_{\pi}(s) \geq v_{\pi_0}(s){% endmathjax %}，则策略{% mathjax %}\pi{% endmathjax %}优于或等于{% mathjax %}\pi_0{% endmathjax %}。**最优策略**优于或等于所有其他策略，用{% mathjax %}\pi^{*}{% endmathjax %}表示，共享相同的**最优状态值函数** {% mathjax %}v^{*}{% endmathjax %}：**最优状态值函数**{% mathjax %}v^{*}(s){% endmathjax %}定义为所有策略的**最大值函数**：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
v^{*}(s) \equiv \underset{\pi}{\max}v^{\pi}(s)\;\text{for all s}\in S
\end{align}
{% endmathjax %}
最优策略也具有相同的**最优动作价值函数**{% mathjax %}q^{*}{% endmathjax %}，**最优动作值函数**{% mathjax %}q^{*}(s,a){% endmathjax %}定义为所有策略的**最大动作值函数**：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
q^{*}(s,a) \equiv \mathbb{E}[R_{t+1} + \gamma v^{*}(S_{t+1})|S_t = s,A_t = a]
\end{align}
{% endmathjax %}
该方程以**即时奖励**和**折扣未来状态值**的形式表达**状态-动作对**的预期累积回报。**最优值函数**和**策略**代表**强化学习**的理想状态。然而，由于实际情况，在计算要求高的任务中很少能找到真正最优的策略。**强化学习**的**代理**(`Agent`)的目标是接近**最佳策略**。假设**环境模型**很完美，**动态规划**(`DP`)则有助于确定最佳值。`DP`和`RL`的基本思想是使用**价值函数**来组织对策略的搜索。对于**马尔可夫决策过程**(`MDP`)，环境的动态由概率{% mathjax %}p(s',r|s,a){% endmathjax %}给出。**动态规划**在特殊情况下会找到精确解，例如查找最短路径。**最优状态值函数**{% mathjax %}v^{*}(s){% endmathjax %}和**最优动作值函数**{% mathjax %}q^{*}(s,a){% endmathjax %}的**贝尔曼最优方程**如下：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
v^{*}(s) = & \underset{a}{\max}\mathbb{E}[R_{t+1} + \gamma v^{*}(S_{t+1})|S_t = s,A_t = a] = \underset{a}{\max}\sum\limits_{s',r}p(s',r|s,a)[r + \gamma v^{*}(s')] \\
q^{*}(s,a) = & \mathbb{E}[R_{t+1} + \underset{a'}{\max}q^{*}(S_{t+1},a')|S_t = s,A_t = a] = \sum\limits_{s',r}p(s',r|s,a)[r + \gamma\underset{a'}{\max}q^{*}(s',a')]
\end{align}
{% endmathjax %}
**动态规划**(`DP`)算法是通过将**贝尔曼方程**转化为更新规则从而推导出来的。