---
title: 机器学习(ML)(二十二) — 强化学习探析
date: 2024-12-26 10:00:11
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

#### RLHF

**人类反馈的强化学习**(`RLHF`)是一种结合了**人类反馈**与**强化学习**技术的**机器学习**方法，旨在提高人工智能模型的表现，尤其是在生成式人工智能（如`LLM`）中的应用。**人类反馈的强化学习**(`RLHF`)的核心思想是利用人类提供的反馈来优化**机器学习**模型，使其能够更好地满足用户需求和期望。传统的**强化学习**依赖于预定义的**奖励函数**来指导学习，而`RLHF`则将人类的主观反馈纳入其中，以便更灵活地捕捉复杂任务中的细微差别和主观性。
<!-- more -->
`RLHF`通常包括以下几个步骤：
- **预训练语言模型**：首先，使用大量标注数据对语言模型进行预训练。这一步骤通常通过**监督学习**完成，以确保模型能够生成合理的初步输出。
- **训练奖励模型**：在此阶段，生成多个可能的问答，并由人类评估这些问答的质量。人类反馈被用于训练一个**奖励模型**，该模型能够评估生成内容的好坏。
- **强化学习微调**：最后，使用训练好的**奖励模型**对**语言模型**进行**微调**，通过**强化学习算法**（如**近端策略优化**：`PPO`）进一步优化其表现，以便更好地符合人类反馈和偏好。

**人类反馈的强化学习**(`RLHF`)在多个领域展现了其重要性，尤其是在**自然语言处理**(`NLP`)和**生成式**`AI`中。通过引入**人类反馈**，`RLHF`能够：**提高生成内容的人性化程度**，使得`AI`生成的文本更符合人类的沟通习惯和情感表达；**增强适应性**，`AI`系统能够根据实时反馈调整其行为，**解决复杂任务**，在一些难以明确量化成功标准的任务中，`RLHF`提供了一种有效的方法来利用人类直观判断作为反馈。适应不断变化的用户需求和偏好。**人类反馈的强化学习**(`RLHF`)是一种前沿技术，通过将**人类直观反馈**与**强化学习**结合起来，为生成式`AI`的发展提供了新的方向。它不仅提高了`AI`系统与用户之间的互动质量，也为复杂任务提供了新的解决方案。

##### 预训练语言模型

首先，使用经典的预训练目标训练一个语言模型，对这一步模型，`OpenAI`在其第一个`RLHF`模型的`InstructGPT`中使用了较小版本的`GPT-3`；`Anthropic`使用了`100`0万 `～` `520`亿参数的`Transformer`模型进行训练；`DeepMind`使用了自家的`2800`亿参数模型`Gopher`。这里可以用额外的文本或者条件对这个`LM`进行微调，例如`OpenAI`采用 “**更可取**”(`preferable`)的人工生成文本进行了微调；而`Anthropic`采用了“**有用、诚实和无害**” 的标准在上下文线索上蒸馏了原始的`LM`。这里或许使用了昂贵的增强数据，但并不是`RLHF`必要的一步。由于`RLHF`还是一个尚待探索的领域，对于” 哪种模型” 适合作为`RLHF`的起点并没有明确的答案。
{% asset_img ml_1.png %}

##### 训练奖励模型

**奖励模型** (`RM`，也叫**偏好模型**)的训练是`RLHF`区别于旧范式的开端。这一模型接收一系列文本并返回一个标量奖励，数值上对应人的偏好。我们可以用端到端的方式用`LM`建模，或者用模块化的系统建模 (比如对输出进行排名，再将排名转换为奖励) 。这一奖励数值将对后续无缝接入现有的`RL`算法至关重要。关于模型选择方面，**奖励模型**可以是另一个经过微调的`LM`，也可以根据偏好数据从头开始训练的`LM`。例如`Anthropic`提出了一种特殊的预训练方式，即用**偏好模型预训练**(`Preference Model Pretraining，PMP`)来替换一般预训练后的微调过程。因为前者被认为对样本数据的利用率更高。但对于哪种**奖励模型**更好尚无定论。

关于训练文本方面，**奖励模型** 的提示-生成对文本是从预定义数据集中采样生成的，并用初始的`LM`给这些提示生成文本。`Anthropic`的数据主要是通过`Amazon Mechanical Turk`上的聊天工具生成的，并在`Hub`上可用，而 `OpenAI`使用了用户提交给`GPT API`的`prompt`。

关于训练奖励数值方面，这里需要人工对`LM`生成的回答进行排名，起初可能会认为应该直接对文本标注分数来训练**奖励模型**，但是由于标注者的价值观不同导致这些分数未经过校准并且充满噪声。通过排名可以比较多个模型的输出并构建更好的规范数据集。对具体的排名方式，是对不同的`LM`在相同提示下的输出进行比较，然后使用`Elo`(评分系统，是一种用于计算棋手和其他竞技游戏玩家相对技能水平的方法)系统建立一个完整的排名。这些不同的排名结果将被归一化为用于训练的**标量奖励值**。
{% asset_img ml_2.png %}

##### 强化学习微调

长期以来出于工程和算法原因，人们认为用**强化学习**训练LM是不可能的。而目前多个组织找到的可行方案是使用**策略梯度强化学习**(`Policy Gradient RL`)**算法**、**近端策略优化**(`Proximal Policy Optimization，PPO`)微调初始`LM`的部分或全部参数。因为微调整个`10B～100B+`参数的成本过高。首先将微调任务表述为**强化学习**问题。该策略是一个接受提示并返回一系列文本(或文本的概率分布)的`LM`。这个策略的**动作空间**(`action space`)是`LM`的词表对应的所有**词元** (一般在`50k`数量级)，**观察空间**(`observation space`)是输入**词元序列**，也比较大(词汇量 `x` 输入标记的数量)。**奖励函数**是奖**励模型**和**策略转变约束**(`Policy shift constraint`)的结合。`PPO`算法的奖励函数计算如下：将提示(`prompt`){% mathjax %}x{% endmathjax %}输入初始LM和当前微调的LM，分别得到了输出文本{% mathjax %}y_1,y_2{% endmathjax %}将来自当前策略的文本传递给**奖励模型**得到一个标量的奖励{% mathjax %}r_{\theta}{% endmathjax %}。将两个模型的生成文本进行比较计算差异的**惩罚项**，在来自`OpenAI、Anthropic`和`DeepMind`的多篇论文中设计为输出词分布序列之间的`Kullback–Leibler (KL) divergence`散度的缩放，即{% mathjax %}r = r_{\theta} - \lambda r_{KL}{% endmathjax %}。这一项被用于惩罚**强化学习**策略在每个训练批次中生成大幅偏离初始模型，以确保模型输出合理连贯的文本。如果去掉这一惩罚项可能导致模型在优化中生成乱码文本来愚弄奖励模型提供高奖励值。此外，`OpenAI`在`InstructGPT`上实验了在`PPO`添加新的**预训练梯度**，可以预见到**奖励函数**的公式会随着`RLHF`研究的进展而继续进化。最后根据`PPO`算法，按当前批次数据的**奖励指标**进行优化(来自`PPO`算法`on-policy`的特性)。`PPO`算法是一种**信赖域优化**(`Trust Region Optimization，TRO`)算法，它使用**梯度约束**确保更新步骤不会破坏学习过程的稳定性。`DeepMind`对`Gopher`使用了类似的奖励设置，但是使用`A2C`(`synchronous advantage actor-critic`)算法来优化**梯度**。
{% asset_img ml_3.png %}

作为一个可选项，`RLHF`可以通过迭代**奖励模型**和策略共同优化。随着策略模型更新，用户可以继续将输出和早期的输出进行合并排名。`Anthropic`在他们的论文中讨论了迭代在线`RLHF`，其中策略的迭代包含在跨模型的`Elo`**排名系统**中。这样引入策略和**奖励模型**演变的复杂动态，代表了一个复杂和开放的研究问题。收集人类偏好数据的质量和数量决定了`RLHF`系统性能的上限。`RLHF`系统需要两种人类偏好数据：**人工生成的文本**和**对模型输出的偏好标签**。除开数据方面的限制，一些有待开发的设计选项可以让`RLHF`取得长足进步。例如对`RL`**优化器**的改进方面，`PPO`是一种较旧的算法，但目前没有什么结构性原因让其他算法可以在现`有RLHF`工作中更具有优势。另外，微调`LM`策略的成本是策略生成的文本都需要在`RM`上进行评估，通过离线`RL`优化策略可以节约这些大模型`RM`的预测成本。最近，出现了新的`RL`算法如**隐式语言**`Q-Learning`(`Implicit Language Q-Learning，ILQL`) 也适用于当前`RL`的优化。在`RL`训练过程的其他核心权衡，例如探索和开发(`exploration-exploitation`) 的平衡也有待尝试和记录。

#### NLPO

大多数**语言模型**在训练时并没有直接的**人类偏好信号**，监督目标字符串仅作为代理。一个整合用户反馈的选项是采用人机协作，即用户在模型训练过程中需要为每个样本提供反馈，但这种密集监督的程度往往是不可行且低效的。自动化指标提供了一个有前景的折衷方案：如**成对学习偏好模型**、`BERTScore`、`BLEURT`等人类偏好的模型，与早期指标（如`BLEU`、`METEOR`等）相比，显著提高了与人类判断的相关性，并且评估成本较低。然而，这些函数通常不是**逐词可微分**的：与人类一样，这些指标只能对完整生成结果提供质量估计。**强化学习**(`RL`)为优化不可微分的标量目标提供了一条自然路径。最近的研究表明，通过约束基于偏好的奖励来结合流畅性概念，**强化学习**(`RL`)在将`LM`与人类偏好对齐方面取得了很好的结果，但这一研究方向的进展受到缺乏开源基准和算法实现的严重阻碍——导致人们认为**强化学习**(`RL`)是`NLP`的一个具有挑战性的范式。为了促进构建**强化学习**(`RL`)**算法**以更好地对齐语言模型(`LM`)。首先，发布了`RL4LMs`库，使得生成`HuggingFace`模型（如`GPT-2`或`T5`）能够使用多种现有的**强化学习**(`RL`)方法进行训练，例如`PPO`、`A2C`等。接下来，使用`RL4LMs`训练的模型应用于新的`GRUE`（**通用强化语言理解评估**）基准：`GRUE`是一个包含`7`个`NLP`任务的集合；与其他基准不同的是，每个任务配对**奖励函数**，而不是进行监督训练。`GRUE`保证模型在保持流畅的语言生成能力的同时，优化这些**奖励函数**。通过**强化学习**(`RL`)训练语言模型——无论是否进行任务监督预训练——以优化奖励。最后，除了现有的**强化学习**(`RL`)方法，还引入了一种新颖的**在线强化学习**(`RL`)**算法——自然语言策略优化**(`NLPO`)，该算法能够在逐词级别动态学习任务特定的约束。实验结果和人类评估表明，与其他方法相比，**自然语言策略优化**(`NLPO`)在学习偏好奖励的同时，更好地保持了语言流畅性，包含了`PPO`的能力。在使用**标量奖励反馈**进行学习时，发现**强化学习**(`RL`)可以更具：**数据效率**，优于通过**监督学习**使用额外专家示范（尽管两者结合是最佳选择）——当作为**自然语言策略优化**(`NLPO`)方法的信号时，学习到的**奖励函数**在性能上优于使用`5`倍数据训练的监督方法；**参数效率**——使得一个结合**监督**和**自然语言策略优化**(`NLPO`)训练的`2.2`亿参数模型超越一个`30`亿参数的**监督模型**。

在情感引导的续写任务中，**自然语言策略优化**(`NLPO`)旨在使**语言模型**（即**策略**）根据评论提示生成积极的情感续写。这里需要平衡两个目标：`1`、作为奖励的自动化人类偏好**智能体**（此处为**情感分类器**）；`2`、通过与未经过**显式人类反馈训练**的语言模型之间的`KL`**散度**来衡量“**自然性**”。如下图所示，**自然语言策略优化**(`NLPO`)与流行的**策略梯度**(`PPO`)的验证学习曲线比较。如果去掉**自然**`KL`**惩罚**(`naturall KL penalty`)，**强化学习**(`RL`)方法可以轻松获得高奖励，但代价是更高的**困惑度**。建议方法：`NLPO + KL`成功地在**奖励**和**自然性**之间取得了比以往研究更有效的**平衡**。
{% asset_img ml_4.png %}

**模仿学习**(`Imitation Learning, IL`)是一种**强化学习范式**，旨在通过从专家示范中进行**监督学习**来执行任务。许多与**自然语言处理**(`NLP`)相关的算法，如**调度采样**(`Schedule Sampling, SS`)、**并行调度采样**(`Parallel SS`)、`Transformer`**调度采样**、**差分调度采样**(`Differential SS`)、`LOL`(`Learning to Optimize Language Sequences`)、`TextGAIL`和`SEARNN`，都受到**DAGGER**和**SEARN**的启发。然而，这些算法在生成过程中普遍存在**偏差**和**马尔可夫决策过程**(`MDP`)问题。在**大动作空间强化学习**中，`MIXER`结合了**调度采样**和**REINFORCE**的思想。`actor-critic`算法解决了`REINFORCE`进行语言生成时的**方差**和**大动作空间问题**；`KG-A2C、TrufLL、AE-DQN`和`GALAD`通过消除和减少探索过程中的动作空间来解决类似问题。

`RL4LMs`是一个开源库，提供了用于**微调**和**评估**基于**语言模型**(`LM`)的**强化学习**(`RL`)算法的构建模块。该库是基于`HuggingFace`和`stable-baselines-3`构建。`RL4LMs`可以用于训练`HuggingFace`中的任何**解码器**或**编码器-解码器**`Transformer`模型，并支持来自`stable-baselines-3`的任何在线**强化学习算法**。此外，还提供了针对`LM`微调的在**线强化学习算法**的实现，例如`PPO、TRPO、A2C`和`NLPO`。该库是**模块化**的，用户可以插入自定义**环境**、**奖励函数**、**指标**和**算法**。在初始版本中，支持`6`种不同的`NLP`任务、`16`种评估**指标**和**奖励**，以及`4`种**强化学习算法**。

每个环境都是一个**自然语言处理**(`NLP`)任务：我们有一个监督数据集{% mathjax %}D = \{(x_i,y_i)\}^N_{i=1}{% endmathjax %}，其中包含{% mathjax %}N{% endmathjax %}个示例，其中{% mathjax %}x\in X{% endmathjax %}是语言输入，{% mathjax %}y\in Y{% endmathjax %}是目标字符串。生成可以视为一个**马尔可夫决策过程**(`MDP`){% mathjax %}h(S,A,R,P,\gamma,T){% endmathjax %}，使用有限的词汇表{% mathjax %}V{% endmathjax %}。`MDP`中的每个回合从数据集中抽取一个数据点{% mathjax %}(x,y){% endmathjax %}开始，并在当前时间步{% mathjax %}t{% endmathjax %}超过时间范围{% mathjax %}T{% endmathjax %}或生成结束句子(`EOS`)标记时结束。输入{% mathjax %}x = \{x_0,\ldots,x_m\}{% endmathjax %}是一个特定任务的提示，作为初始状态{% mathjax %}s_0 = \{x_0,\ldots,x_m\}{% endmathjax %}，其中{% mathjax %}s_0\in S{% endmathjax %}，而{% mathjax %}S{% endmathjax %}是**状态空间**，且{% mathjax %}x_m\in V{% endmathjax %}。环境中的一个动作{% mathjax %}a_t \in A{% endmathjax %}由词汇表{% mathjax %}V{% endmathjax %}中的一个标记组成。转移函数{% mathjax %}P: S\times A\rightarrow \Delta(S){% endmathjax %}确定性地将动作{% mathjax %}a_t{% endmathjax %}附加到状态{% mathjax %}s_{t-1} = (x_0,\ldots,x_m,a_0,\ldots,a_{t-1}){% endmathjax %}的末尾。这一过程持续到时间范围结束，即{% mathjax %}t\leq T{% endmathjax %}，并且获得状态{% mathjax %}s_T = (x_0,\ldots,x_m,a_0,\ldots,a_T){% endmathjax %}。在每个回合结束时，会根据状态和目标字符串的组合{% mathjax %}(s_T,y){% endmathjax %}发出奖励{% mathjax %}R: S\times A \times Y\rightarrow R_1{% endmathjax %}，例如，像`PARENT`这样的**自动化指标**。`RL4LMs`提供了一个类似`OpenAI Gym`的`API`，用于模拟这种基于`LM`的`MDP`公式。这种抽象允许快速添加新任务，并与所有已实现的算法兼容。

由于`RL4LMs`提供了一个通用接口，用于每个标记或每个序列生成奖励，因此可以快速将各种**强化学习算法**应用于多样化的文本指标作为奖励。提供了以下接口：
- `n-gram`**重叠指标**，如`ROUGE`、`BLEU`、`SacreBLEU`、`METEOR`。
- **基于模型的语义指标**，如`BertScore`和`BLEURT`，这些指标通常与人类判断具有更高的相关性。
- **特定任务指标**，如`CIDER`、`SPICE`（用于图像描述/常识生成）、`PARENT`（用于数据到文本生成）和`SummaCZS`（用于摘要的真实性）。
- **多样性/流畅性/自然性指标**，如困惑度、平均分段类型标记比率（`MSSTR`）、单词和双词的**香农熵**、不同`n-gram`的比例(`Distinct-1`、`Distinct-2`)，以及在整个生成文本中仅出现一次的`n-gram`数量。
- **基于模型的人类偏好的特定任务指标**，在`Ouyang`等人的方法中收集的人类偏好数据上训练的**分类器**。

`RL4LMs`支持通过在线`actor-critic`算法对**语言模型**进行**微调**和**从头训练**。这类算法允许训练一个参数化的**控制策略**，定义为{% mathjax %}\pi_{\theta}:S\rightarrow \Delta(A){% endmathjax %}，这是一个函数，在给定状态下选择一个动作，以最大化轨迹上的**长期折扣奖励**{% mathjax %}\mathbb{E}_{\pi}\bigg[\sum_{t=0}^T \gamma^t R(s_t,a_t)\bigg]{% endmathjax %}。基准实验专注于**微调**一个预训练的语言模型{% mathjax %}\pi_0{% endmathjax %}，并将其作为**智能体策略**的初始策略{% mathjax %}\pi_{\theta} = \pi_0{% endmathjax %}。类似地，用于估计价值函数的价值网络{% mathjax %}V_{\phi}{% endmathjax %}也从{% mathjax %}\pi_0{% endmathjax %}初始化，除了最后一层是随机初始化以输出一个标量值。与其他**深度强化学习**`actor-critic`**算法**一样，**价值函数**和`Q`**值函数**为：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
V_t^{\pi} & = \mathbb{E}_{a_t\sim \pi}\bigg[\sum\limits_{\tau = t}^T \gamma R(s_{\tau},a_{\tau, y}) \bigg] \\
Q_t^{\pi}(s_t,a_t) & = R(s_t,a_t,y) + \gamma \mathbb{E}_{s_{t+1}\sim P}[V_{t+1}^{\pi}(s_{t+1})]
\end{align}
{% endmathjax %}
优势函数的定义为：
{% mathjax '{"conversion":{"em":14}}' %}
A_t^{\pi}(s,a) = Q_t^{\pi}(s,a) = Q_t^{\pi}(s,a) - V_t^{\pi}
{% endmathjax %}
为了提高训练的稳定性，优势使用**广义优势估计**(`Generalized Advantage Estimation`)进行近似。给定输入输出对{% mathjax %}(x,y){% endmathjax %}和**智能体**的生成预测，由于环境奖励是序列级且稀疏的，按照`Wu`的方法，使用逐标记的`KL`惩罚来正则化奖励函数，以防止模型过度偏离初始化的语言模型{% mathjax %}\pi_0{% endmathjax %}。正则化后的奖励函数为：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{R}(s_t,a_t,y) = R(s_t,a_t,y) - \beta \text{KL}(\pi_{\theta}(a_t|s_t)\|\pi_{0}(a_t|s_t))
{% endmathjax %}
其中{% mathjax %}\hat{R}{% endmathjax %}是**正则化**后的`KL`奖励，{% mathjax %}y{% endmathjax %}是真实预测，{% mathjax %}\text{KL}(\pi_{\theta}(a_t|s_t)\|\pi_{0}(a_t|s_t)) = (\log_{pi_0}(s_t|s_t) - \log_{pi_{\theta}}(a_t|s_t)){% endmathjax %}，`KL`系数{% mathjax %}\beta{% endmathjax %}是动态调整的。

语言生成的**动作空间**比大多数**离散动作空间**的**强化学习**(`RL`)**算法**设计的要大几个数量级，例如，`GPT-2`和`GPT-3`的词汇大小分别为`50K`和`32K`。**动作空间**的大小是使用现有**强化学习方法**训练**语言模型**时不稳定的核心原因。为了解决这个问题，需要引入了`NLPO`（**自然语言策略优化**），该方法受到**动作消除/无效动作掩蔽**工作的启发。`NLPO`是`PPO`的一个**参数化掩蔽**扩展，学习在训练过程掩蔽上下文中不太相关的`token`。`NLPO`通过`top-p`采样实现这一点，该方法将`token`限制在其累积概率大于概率参数{% mathjax %}p{% endmathjax %}的最小可能集合中。
具体而言，`NLPO`维护一个**掩蔽策略**{% mathjax %}\pi_{\psi}{% endmathjax %}：**掩蔽策略**是当前策略{% mathjax %}\pi_{\theta}{% endmathjax %}的副本，但仅每{% mathjax %}\tau{% endmathjax %}步更新一次。通过从词汇中选择`top-p`标记来创建一个参数化的**无效掩蔽**，然后对剩余标记应用**无效掩蔽**——即在训练期间从{% mathjax %}\pi_{\theta}{% endmathjax %}抽样动作时，将它们的概率设置为`0`；这种周期性更新的策略{% mathjax %}\pi_{\psi}{% endmathjax %}受到离线`Q-Learning`算法的启发，为策略{% mathjax %}\pi_{\theta}{% endmathjax %}提供了一个额外约束，以平衡包含更多任务相关信息的好处与源自{% mathjax %}\pi_{\theta}{% endmathjax %}的`KL`**惩罚**之间的关系，以及奖励操纵的风险。`NLPO`算法的伪代码实现如下：
<embed src="algorithm_2.pdf" type="application/pdf" width="100%" height="200">

#### PSRL

在**强化学习**(`RL`)中，**智能体**面临与未知环境互动的任务，同时试图最大化随时间累积的总奖励。**强化学习**(`RL`)中的一个核心挑战是如何平衡：当采取探索性动作时，**智能体**会获得更多未知环境的知识，利用获得的知识可能会产生更高的**即时回报**。基于**汤普森采样**(`Thompson sampling`)开发**随机化探索**方法。虽然这些方法已被证明有效，但它们在很大程度上局限于情节环境。具体而言，其操作模式是在每个情节开始之前随机采样一个新策略，该策略旨在最大化在统计上合理的环境模型中的**期望回报**，并在整个情节中遵循该策略。例如，`bootstrapped DQN`维护一个**近似最优动作值函数**{% mathjax %}Q^*{% endmathjax %}的后验分布集成，并在每个第{% mathjax %}\ell{% endmathjax %}个情节之前采样一个随机元素{% mathjax %}\hat{Q}_{\ell}{% endmathjax %}。然后，执行相对于{% mathjax %}\hat{Q}_{\ell}{% endmathjax %}的**贪婪策略**。尽管在持续环境中学习是**强化学习**中的一个基本问题，但关于**随机化探索方**法的研究主要集中在**情节环境**上，只有少数例外特定于“**白板**”上下文 。首次开发了适用于强化学习(`PSRL`)的后验采样版本，该版本易于扩展以适应复杂环境中的函数逼近。称为**持续**`PSRL`，在每个时间点以概率{% mathjax %}p{% endmathjax %}重新采样新策略。这里{% mathjax %}p{% endmathjax %}是作为**智能体**设计的一部分指定的，表示**智能体**如何选择将其经验划分为试验的区间。通过这个重采样规则，自然可以执行一个最大化折扣回报的策略，**折扣因子**为{% mathjax %}\gamma = 1 - p{% endmathjax %}。在选择{% mathjax %}\gamma{% endmathjax %}的情况下，在连续重采样时间之间获得的未折扣回报构成了所用策略的{% mathjax %}\gamma{% endmathjax %}-**折扣回报的无偏估计**。这种简单的重采样方案可以轻松集成到具有**函数逼近**的**随机化探索算法**中。例如，可以通过以概率{% mathjax %}p{% endmathjax %}从当前**近似后验分布**中重新采样**动作值函数**来修改`bootstrapped DQN`，以应对持续环境。与`bootstrapped DQN`的原始版本一样，每次执行的动作都是相对于最近采样的**动作值函数**贪婪选择的。

许多理论研究将**折扣因子**{% mathjax %}\gamma{% endmathjax %}视为**环境**的一部分，例如，它们直接分析{% mathjax %}\gamma{% endmathjax %}折扣的遗憾。与此不同，评估**智能体**的表现时采用**无折扣遗憾**。因此，尽管**折扣因子**{% mathjax %}\gamma{% endmathjax %}在**智能体**设计中起着重要作用，但它并没有反映设计者的目标。分析表明，尽管以概率{% mathjax %}p = 1 - \gamma{% endmathjax %}进行重采样并以{% mathjax %}\gamma{% endmathjax %}折扣目标进行规划并不会导致每个时间步遗憾消失，但通过随着时间的推移增加{% mathjax %}\gamma{% endmathjax %}可以实现这一点。先前的工作考虑了处理持续环境的`PSRL`版本，通过直接在**无折扣遗憾**下进行规划。`Ouyang`等人提出的算法在每次满足以下两个条件之一时，从环境后验中重新采样环境：1、自上次重采样以来经过的时间超过两次最近重采样之间的间隔；2、自上次重采样以来，任何**状态-动作对**的访问次数翻倍。后者条件发挥着重要作用，但在复杂环境中操作时不可行，例如，处理不可计算的**大状态空间**并使用**神经网络近似动作值函数**的分布。特别是，如何有效跟踪访问计数并不明确，即使能做到这一点，这些计数也可能无关紧要，因为访问任何单个状态超过一次甚至可能很少。为了解决**大状态空间**问题，`Theocharous`等人考虑简单地将每对连续重采样之间的时间延长一倍。尽管生成的算法避免了维护访问计数，但他们的分析严重依赖技术假设，否则遗憾界限将随着时间线性增长。**随机探索的重采样方法**——包括固定和递减重置概率，并进行了严格分析，该分析建立了类似于`Ouyang`等人的**遗憾界限**，但其重采样标准比该文中提出的方法更简单、更可扩展。

通过与未知环境{% mathjax %}\mathcal{E} = (\mathcal{A},\mathcal{S},p){% endmathjax %}的单一交互流来学习优化性能的问题，该环境被建模为**马尔可夫决策过程**(`MDP`)。在这里，{% mathjax %}\mathcal{A}{% endmathjax %}是一个有限的**动作空间**，{% mathjax %}\mathcal{S}{% endmathjax %}是一个有限的**状态空间**，而{% mathjax %}p{% endmathjax %}是一个函数，指定在当前状态{% mathjax %}s\in \mathcal{S}{% endmathjax %}和动作{% mathjax %}a\in \mathcal{A}{% endmathjax %}下的**状态转移概率**{% mathjax %}p(s'|s,a){% endmathjax %}。到时间{% mathjax %}t{% endmathjax %}的交互构成一个历史{% mathjax %}\mathcal{H}_t = (\mathcal{S}_0,\mathcal{A}_0,\mathcal{S}_1,\mathcal{A}_1,\ldots,\mathcal{S}_t,\mathcal{A}_t){% endmathjax %}，**智能体**在观察到状态{% mathjax %}\mathcal{S}_t{% endmathjax %}后选择动作{% mathjax %}\mathcal{A}_t{% endmathjax %}。环境及所有相关随机量都在一个共同的**概率空间**{% mathjax %}(\Omega,\mathcal{F},\mathbb{P}){% endmathjax %}中定义。环境{% mathjax %}\mathcal{E}{% endmathjax %}本身是随机的，使用分布{% mathjax %}\mathbb{P}(\mathcal{E}\in \cdot){% endmathjax %}来捕捉**智能体**设计者对所有可能环境的**先验信念**。随着历史的发展，可以学习到的内容由后验分布{% mathjax %}\mathbb{P}(\mathcal{E}\in \cdot|\mathcal{H}_t){% endmathjax %}表示。假设{% mathjax %}\mathcal{A}{% endmathjax %}和{% mathjax %}\mathcal{S}{% endmathjax %}是确定且已知的，但**观察概率函数**{% mathjax %}p{% endmathjax %}是一个随机变量，**智能体**需要学习。为了简化，假设初始状态{% mathjax %}\mathcal{S}_0{% endmathjax %}是确定的，但相同的分析很容易扩展到初始状态的分布。**智能体**的偏好可以通过**奖励函数**{% mathjax %}r\;:\;\mathcal{S}\times \mathcal{A}\mapsto [0,1]{% endmathjax %}来表示。在状态{% mathjax %}\mathcal{S}_t{% endmathjax %}中选择动作{% mathjax %}\mathcal{A}_t{% endmathjax %}后，**智能体**观察到状态{% mathjax %}\mathcal{S}_{t+1}{% endmathjax %}并获得一个确定的奖励{% mathjax %}R_{t+1} = r(S_t,A_t){% endmathjax %}，该奖励限制在区间`[0,1]`内。为了简化，假设**奖励函数**{% mathjax %}r{% endmathjax %}是确定且已知的，很容易推广到**随机奖励函数**。**智能体**通过策略来指定其动作。**随机策略**{% mathjax %}\pi{% endmathjax %}可以通过**智能体**在给定情境状态{% mathjax %}\mathcal{S}_t{% endmathjax %}下对动作集合{% mathjax %}\mathcal{A}{% endmathjax %}分配的**概率质量函数**{% mathjax %}\pi(\cdot|S_t){% endmathjax %}来表示。在正式定义**智能体**的学习目标之前，扩展**智能体**状态考虑**随机策略**。这里考虑`Lu`等人(`2021`)提出的**算法状态概念**{% mathjax %}Z_t{% endmathjax %}，它捕捉了时间 {% mathjax %}t{% endmathjax %}的**算法随机性**。一个算法是一个确定性序列{% mathjax %}\{\mu_t|t = 1,2\ldots\}{% endmathjax %}的函数，每个函数将对{% mathjax %}(\mathcal{H}_t,Z_t){% endmathjax %}的映射到一个策略。在每个时间步{% mathjax %}t{% endmathjax %}，算法从**随机算法状态**{% mathjax %}Z_t{% endmathjax %}中采样，并计算策略{% mathjax %}\pi_t = \mu(\mathcal{H}_t,Z_t){% endmathjax %}。当上下文中引入了由{% mathjax %}Z_t{% endmathjax %}引起的随机性时，也可以写作{% mathjax %}\pi_t\sim \mu_t(\mathcal{H_t}){% endmathjax %}。然后，算法在时间点采样动作{% mathjax %}A_t\sim \pi_t(\cdot|S_t){% endmathjax %}。对于策略{% mathjax %}\pi{% endmathjax %}，表示从状态{% mathjax %}s{% endmathjax %}开始的平均奖励为：
{% mathjax '{"conversion":{"em":14}}' %}
\lambda_{\pi,\mathcal{E}}(s) = \underset{T\rightarrow \infty}{\text{lim inf}}\;\mathbb{E}_{\pi}\bigg[\frac{1}{T}\sum\limits_{t=0}^{T-1}R_{t+1}|\mathcal{E},S_0 = s \bigg]
{% endmathjax %}
其中期望的下标表示奖励序列是通过遵循策略{% mathjax %}\pi{% endmathjax %}实现的，而下标{% mathjax %}\mathcal{E}{% endmathjax %}则强调**平均奖励**对环境{% mathjax %}\mathcal{E}{% endmathjax %}的依赖。最优平均奖励为：
{% mathjax '{"conversion":{"em":14}}' %}
\lambda_{*,\mathcal{E}}(s) = \underset{\pi}{\text{sup}}\lambda_{\pi,\mathcal{E}}(s)\;\;\;\forall_s\in \mathcal{S}
{% endmathjax %}
**弱通信**`MDP`：如果存在一组状态，其中该组中的每个状态都可以通过某种确定性静态策略从该组中的每个其他状态访问，同时可能存在一个在每种策略下都是**瞬态**的**空状态集**，则该`MDP`被称为**弱通信**。在**弱通信**`MDP`下，最优平均奖励{% mathjax %}\lambda_{*,\mathcal{E}}(\cdot){% endmathjax %}是与**状态无关**的。因此，用符号{% mathjax %}\lambda_{*,\mathcal{E}}(s){% endmathjax %}来表示所有状态{% mathjax %}s\in \mathcal{S}{% endmathjax %}的**最优平均奖励**。对于策略{% mathjax %}\pi{% endmathjax %}，定义其在时间{% mathjax %}T{% endmathjax %}之前的遗憾为：
{% mathjax '{"conversion":{"em":14}}' %}
\text{Regret}(T,\pi) := \sum\limits_{t=0}^{T-1}(\lambda_{*,\mathcal{E}} - R_{t+1})
{% endmathjax %}
**遗憾**本身是一个随机变量，依赖于随机环境{% mathjax %}\mathcal{E}{% endmathjax %}、算法的内部**随机采样**和**随机转移**。通过**遗憾**及其**期望值**来衡量**智能体**的表现。

`Continuing PSRL`是一种针对**持续强化学习**环境的**后验采样算法**，旨在解决在无限时间范围内的平均奖励问题。它是对**传统后验采样强化学习**(`PSRL`)算法的扩展，特别适用于复杂的高维状态空间，避免了维护状态-动作访问计数的需求。主要特征包括：
- **随机重采样机制**：在每个时间步，`Continuing PSRL`以一定概率{% mathjax %}p{% endmathjax %}重新采样环境模型。这种重采样机制使得**智能体**能够根据当前历史信息从后验分布中获取新的环境模型，从而适应环境的变化。
- **无折扣奖励规划**：该算法通过简单的随机化方案决定何时重采样新模型，并且在重采样时使用的是**无折扣奖励**，以便更好地适应持续学习的场景。
- **性能保证**：`Continuing PSRL`在表格设置下建立了{% mathjax %}\tilde{O}(\tau S\sqrt{AT}){% endmathjax %}的**贝叶斯遗憾界限**，其中{% mathjax %}S{% endmathjax %}是**环境状态值**，{% mathjax %}A{% endmathjax %}是动作值，{% mathjax %}\tau{% endmathjax %}表示**奖励平均时间**。这一结果表明，该算法在多次交互中能够有效地估计**平均奖励**。

在**有限时间的马尔可夫决策过程**中，规划的时间范围是固定且已知的。规划目标通常是每个回合结束前有限时间步内的**累积奖励**。当时间范围为无限时，**智能体**的前瞻性规划变得具有挑战性。解决这一挑战的一种方法是通过维持**折扣因子**{% mathjax %}\gamma\in [0,1){% endmathjax %}来为**智能体**设定一个有效的有限规划范围。实际上，{% mathjax %}\gamma{% endmathjax %}决定了算法重新采样用于规划的独立环境模型的频率。给定这个**折扣因子**，将原始的无限时间学习问题划分为随机长度的**伪回合**。每个**伪回合**在算法重新采样并计算新策略时终止。具体而言，在时间步{% mathjax %}t = 0,1,\ldots{% endmathjax %}的开始，**智能体**采样一个二进制指示器{% mathjax %}X_t{% endmathjax %}。如果{% mathjax %}X_t = 0{% endmathjax %}，**智能体**根据当时可用的历史{% mathjax %}\mathcal{H}_t{% endmathjax %}重新采样一个新环境{% mathjax %}\mathcal{E}{% endmathjax %}，并将{% mathjax %}t{% endmathjax %}标记为**新伪回合**的开始。然后，它计算一个新的策略{% mathjax %}\pi{% endmathjax %}来在这个**伪回合**中执行，并根据策略{% mathjax %}\pi{% endmathjax %}行动。如果{% mathjax %}X_t = 1{% endmathjax %}，则继续当前**伪回合**并遵循最近计算的策略{% mathjax %}\pi{% endmathjax %}。当{% mathjax %}X_t\sim \text{Bernoulli}(\gamma){% endmathjax %}时，可以将{% mathjax %}\gamma{% endmathjax %}理解为**伪回合**在时间步{% mathjax %}t{% endmathjax %}的**存活概率**。

**折扣值**：在每个时间步，**智能体**优化一个带有上述**折扣因子**{% mathjax %}\gamma{% endmathjax %}的**折扣目标**。对于每个环境{% mathjax %}\mathcal{E}{% endmathjax %}和策略{% mathjax %}\pi{% endmathjax %}，{% mathjax %}\gamma{% endmathjax %}-`discounted`**折扣值函数**{% mathjax %}V_{\pi,\mathcal{E}}^{\gamma}\in \mathbb{R}^S{% endmathjax %}定义为：
{% mathjax '{"conversion":{"em":14}}' %}
V_{\pi,\mathcal{E}}^{\gamma} := \mathbb{E}\bigg[ \sum\limits_{h=0}^{H-1} P_{\pi}^{h} r_{\pi} | \mathcal{E}\bigg] = \mathbb{E}\bigg[ \sum\limits_{h=0}^{\infty} P_{\pi}^{h} r_{\pi} | \mathcal{E}\bigg]
{% endmathjax %}
在**马尔可夫决策过程**(`MDP`)，对于所有状态{% mathjax %}s,s'\in \mathcal{S}{% endmathjax %}和动作{% mathjax %}a\in \mathcal{A}{% endmathjax %}有{% mathjax %}P_{ss'}^{\pi} = \sum\limits_{a\in \mathcal{A}}\pi(a|s)p(s'|s,a){% endmathjax %}以及{% mathjax %}r^{\pi}_s = \sum\limits_{a\in \mathcal{A}}\pi(a|s) r_{as}{% endmathjax %}。其中，期望是基于随机回合长度{% mathjax %}\mathcal{H}{% endmathjax %}。由于伪回合在时间{% mathjax %}t{% endmathjax %}终止时，独立采样的{% mathjax %}X_t\sim \text{Bernoulli}(\gamma){% endmathjax %}取值为`0`，因此其长度{% mathjax %}\mathcal{H}{% endmathjax %}遵循参数为{% mathjax %}\gamma{% endmathjax %}的二项分布。上述第二个等式直接源于这一观察。作为最优值。此外，将每个环境{% mathjax %}\mathcal{E}{% endmathjax %}下与{% mathjax %}V_{*,\mathcal{E}}^{\gamma}{% endmathjax %}相关的最优策略表示为{% mathjax %}\pi_{\gamma}^{\mathcal{E}}{% endmathjax %}，这在分析中将非常有用。当上下文中清楚地指明了{% mathjax %} \gamma{% endmathjax %}时，可以省略{% mathjax %}\gamma{% endmathjax %}的下标以避免混淆。值得注意的是，{% mathjax %}V_{\pi,\mathcal{E}}^{\gamma}{% endmathjax %}满足**贝尔曼方程**(`Bellman Equation`)，该方程描述了**状态价值**与**即时奖励**和**后续状态价值**之间的关系。**贝尔曼方程**表示为：
{% mathjax '{"conversion":{"em":14}}' %}
V_{\pi,\mathcal{E}}^{\gamma} = r_{\pi} + \gamma P_{\pi}V_{\pi,\mathcal{E}}^{\gamma}
{% endmathjax %}
**奖励平均时间**：策略{% mathjax %}\pi{% endmathjax %}的**奖励平均时间**{% mathjax %}\tau_{\pi,\mathcal{E}}{% endmathjax %}是满足以下条件的最小值{% mathjax %}\tau\in[0,\infty){% endmathjax %}，使得
{% mathjax '{"conversion":{"em":14}}' %}
\bigg|\mathbb{E}_{\pi}\bigg[ \sum\limits_{t=0}^{T-1}R_{t+1}|\mathcal{E},S_0 = s\bigg] - T\cdot \lambda_{\pi,\mathcal{E}}(s)\bigg| \leq \tau
{% endmathjax %}
对于所有{% mathjax %}T\geq 0{% endmathjax %}且{% mathjax %}s\in \mathcal{S}{% endmathjax %}。当{% mathjax %}\pi^*{% endmathjax %}是环境{% mathjax %}\mathcal{E}{% endmathjax %}的**最优策略**时，{% mathjax %}\tau_{*,\mathcal{E}}:= \tau_{\pi^*,\mathcal{E}}{% endmathjax %}等价于`Bartlett`和`Tewari(2009)`中的**跨度**(`span`)概念。定义{% mathjax %}\Omega_{*}{% endmathjax %}为所有**弱通信**`MDP`{% mathjax %}\mathcal{E}{% endmathjax %}的集合，并对**先验分布**{% mathjax %}\mathbb{P}(\mathcal{E}\in \cdot){% endmathjax %}做出以下假设。该假设表明，我们关注的是**有界奖励平均时间**的**弱通信**`MDP`。此定义强调了在评估策略性能时，**奖励平均时间**的关键作用。**奖励平均时间**反映了在给定环境和策略下，**智能体**需要多长时间才能稳定地估计其**长期平均奖励**，从而影响其学习和决策过程。通过引入这一概念，我够更好地理解和分析在不同环境下**智能体**的表现及其**遗憾界限**。存在{% mathjax %}\tau < \infty{% endmathjax %}，使得**环境**的**先验分布**{% mathjax %}\mathbb{P}(\mathcal{E}\in \cdot){% endmathjax %}满足：
{% mathjax '{"conversion":{"em":14}}' %}
\mathbb{P}(\mathcal{E}\in \Omega_*,\tau_{*,\mathcal{E}\leq \tau}) = 1
{% endmathjax %}
对于所有{% mathjax %}\pi,s\in \mathcal{S}{% endmathjax %}和{% mathjax %}\gamma\in [0,1){% endmathjax %}。
{% mathjax '{"conversion":{"em":14}}' %}
\bigg|V^{\gamma}_{\pi,\mathcal{E}}(s) - \frac{\lambda_{\pi,\mathcal{E}(s)}}{1 - \gamma} \bigg| \leq \tau_{\pi,\mathcal{E}(s)}
{% endmathjax %}
我们再次注意到，对于**弱通信**的环境{% mathjax %}\mathcal{E}\in \Omega_*{% endmathjax %}，**最优平均奖励**是与状态无关的。
{% mathjax '{"conversion":{"em":14}}' %}
\bigg|V_{*,\mathcal{E}^{\gamma}(s)} - V_{*,\mathcal{E}^{\gamma}(s')} \bigg| \leq 2\tau_{*,\mathcal{E}} \leq 2\tau
{% endmathjax %}
对所有状态{% mathjax %}s{% endmathjax %}，{% mathjax %}s'\in \mathcal{S}{% endmathjax %}。

**折扣遗憾**：为了分析算法在{% mathjax %}T{% endmathjax %}个时间步上的表现，设定{% mathjax %}K = \text{arg }\max\{k:t_k\leq T\}{% endmathjax %}为在时间{% mathjax %}T{% endmathjax %}之前的伪回合数量。在后续分析中，采用约定{% mathjax %}t_{K+1} = T + 1{% endmathjax %}。为了获得一般{% mathjax %}T{% endmathjax %}的界限，可以始终填充其余的时间步以形成完整的**伪回合**，并且渐近速率保持不变。此外，很容易看出，对于所有{% mathjax %}\gamma \in [0,1),\mathbb{E}[K]\leq (1-\gamma)T + 1{% endmathjax %}。给定折扣因子{% mathjax %}\gamma \in [0,1){% endmathjax %}到时间{% mathjax %}T{% endmathjax %}的{% mathjax %}\gamma{% endmathjax %}**折扣遗憾**为：
{% mathjax '{"conversion":{"em":14}}' %}
\text{Regret}_{\gamma}(K,\pi) := \sum\limits_{k=1}^K \Delta_k
{% endmathjax %}
定义**伪回合**{% mathjax %}k{% endmathjax %}的**遗憾**为{% mathjax %}\Delta_k = V_{*,\mathcal{E}^{\gamma}(s_{k,1})} - V_{\pi_k,\mathcal{E}^{\gamma}(s_{k,1})}{% endmathjax %}，其中{% mathjax %}V_{*,\mathcal{E}^{\gamma}} = V_{\pi_*,\mathcal{E}^{\gamma}} = V_{\pi^{\mathcal{E}},\mathcal{E}^{\gamma}}{% endmathjax %}，且{% mathjax %}\pi_k\sim \mu_k(\mathcal{H}_{t_k}),A_t\sim \pi_k(\cdot|S_t),S_{t+1}\sim p(cdot|S_t,A_t){% endmathjax %}，以及{% mathjax %}R_t= r(S_t,A_t,S_{t+1}){% endmathjax %}对于{% mathjax %}t\in E_k{% endmathjax %}。

**经验估计**：定义算法使用的**经验转移概率**。令{% mathjax %}N_t(s,a) = \sum_{\tau = 1}^t\;\mathbb{I}\{(S_{\tau},A_{\tau}) = (s,a)\}{% endmathjax %}为在时间步{% mathjax %}t{% endmathjax %}之前，在状态{% mathjax %}s{% endmathjax %}中采样动作{% mathjax %}a{% endmathjax %}的次数。对于每一对{% mathjax %}(s,a){% endmathjax %}，如果{% mathjax %}N_{t_k}(s,a) > 0{% endmathjax %}，则**伪回合**{% mathjax %}k{% endmathjax %}之前的**经验转移概率**为：
{% mathjax '{"conversion":{"em":14}}' %}
\hat{p}_k(s'|s,a) = \sum\limits_{t=1}^{k-1}\; \sum\limits_{t\in E_k} \frac{\mathbb{I}\{(S_{t},A_{t},S_{t+1}) = (s,a,s')\}}{N_{t_k}(s,a)}
{% endmathjax %}
对于所有{% mathjax %}s'\in \mathcal{S}{% endmathjax %}。如果在伪回合{% mathjax %}k{% endmathjax %}之前，状态{% mathjax %}s{% endmathjax %}中的动作{% mathjax %}a{% endmathjax %}从未被采样过，令{% mathjax %}\hat{p}_k(s'|s,a) = 1{% endmathjax %}对于随机的{% mathjax %}s'\in \mathcal{S}{% endmathjax %}，并且{% mathjax %}\hat{p}_k(s''|s,a) = 0{% endmathjax %}对于{% mathjax %}s''\in \mathcal{S}\setminus \{s'\}{% endmathjax %}。相应的矩阵表示{% mathjax %}\hat{P}^k{% endmathjax %}也按类似方式定义。

**持续后验采样强化学习**(`Continuing Posterior Sampling for Reinforcement Learning, CPSRL`)**算法**，该算法将**后验采样强化学习**(`PSRL`)扩展到具有{% mathjax %}\gamma{% endmathjax %}折扣规划的无限时间范围设置。该算法从环境的**先验分布**开始，其中包含**动作**{% mathjax %}\mathcal{A}{% endmathjax %}和状态{% mathjax %}\mathcal{S}{% endmathjax %}。此外，算法设置一个指示器{% mathjax %} X_1 = 0{% endmathjax %}并假设一个**折扣因子**{% mathjax %}\gamma{% endmathjax %}。在每个时间步{% mathjax %}t{% endmathjax %}开始，如果指示器{% mathjax %}X_t = 0{% endmathjax %}，**持续后验采样强化学习**(`CPSRL`)从基于当时可用历史{% mathjax %}\mathcal{H}_t{% endmathjax %}的**后验分布**中抽样环境{% mathjax %}\mathcal{E}_t = (\mathcal{A},\mathcal{S},p_t){% endmathjax %}，并将{% mathjax %}t{% endmathjax %}标记为**新伪回合**的开始。然后，它计算并遵循策略{% mathjax %}\pi_t = \pi^{\mathcal{E}_t}{% endmathjax %}在时间{% mathjax %}t{% endmathjax %}的执行。否则，如果{% mathjax %}X_t = 1{% endmathjax %}，它在时间{% mathjax %}t{% endmathjax %}继续使用策略{% mathjax %}\pi_t = \pi_{t-1}{% endmathjax %}并将时间步{% mathjax %}t{% endmathjax %}添加到当前**伪回合**中。接着，{% mathjax %}X_{t+1}{% endmathjax %}从参数为{% mathjax %}\gamma{% endmathjax %}的**伯努利分布**中抽样，以便在下一个时间步使用。
<embed src="algorithm_1.pdf" type="application/pdf" width="100%" height="200">

与传统的`PSRL`相比，**持续后验采样强化学习**(`CPSRL`)仅增加了一个独立的伯**努利随机数生成器**来决定何时重新采样。尽管**持续后验采样强化学习**(`CPSRL`)并不是专门设计用于实际应用，但这种重新采样机制带来了可扩展性和通用性。例如，当环境具有极大的状态或动作空间时，例如`Atari`游戏(`Mnih et al., 2015`)，依赖于**状态-动作**访问统计的先前重新采样方法需要一个庞大的**查找表**，而**持续后验采样强化学习**(`CPSRL`)中的重新采样方法仍然可以应用，并且计算开销很小。将**后验采样强化学习**(`PSRL`)扩展到环境没有重置计划的情境下，**智能体**必须在无限时间范围内进行规划。理论上证明了**持续后验采样**(`CPSRL`)具有接近理论最优性的**遗憾上界**。值得注意，**持续后验采样强化学习**(`CPSRL`)仅依赖于一个**伯努利随机数生成器**来重新采样环境，而不是以往工作中复杂的**回合停止**方案。这种设计原则可以很容易地应用于具有大状态空间的一般环境。在表格和连续的`RiverSwim`环境中的模拟展示了该方法的有效性。此外，**持续后验采样强化学习**(`CPSRL`)还突出了**折扣因子**在**智能体**设计中的作用，因为**折扣因子**不再被视为学习目标的一部分，而主要作为**智能体**动态调整其规划时间范围的工具。因此，这项工作可能为理解**折扣因子**提供了重要的一步，而**折扣因子**在**强化学习**应用中已经广泛流行。

#### 总结

**强化学习**受到了至少`3`种教条的影响。第一种是**环境聚光灯**，这指的是倾向于关注**建模环境**而非**智能体**。第二种是将**学习**视为找到**任务解决方案**，而不是适应的过程。第三种是**奖励假设**，认为所有目标和目的都可以被视为**最大化奖励信号**。这三种教条塑造了我们对**强化学习**的理解。

在《科学革命的结构》中，托马斯·库恩区分了科学活动的两个阶段：第一个阶段被称为“**常规科学**”，库恩将其比作**解谜**；第二个阶段被称为“**革命**”阶段，涉及对科学基本价值、方法和承诺的重新构想，库恩统称为“**范式**”。**人工智能**(`AI`)的历史可以说经历了这两个阶段之间的多次波动，以及多个范式的更替。第一个阶段始于`1956`年的达特茅斯研讨会(`McCarthy et al., 2006`)，并且可以说一直持续到`1973`年`Lighthill`等人发布的报告，这一报告被认为对第一次`AI`寒冬的到来产生了重大影响(`Haenlein & Kaplan, 2019`)。在此后的几十年中，我们见证了多种方法和研究框架的兴起，如**符号**`AI`(`Newell & Simon, 1961；2007`)、**基于知识的系统**(`Buchanan et al., 1969`)和**统计学习理论**(`Vapnik & Chervonenkis, 1971；Valiant, 1984；Cortes & Vapnik, 1995`)，知道近年来，深度学习(`Krizhevsky et al., 2012；LeCun et al., 2015；Vaswani et al., 2017`)和**大语言模型**(`Brown et al., 2020；Bommasani et al., 2021；Achiam et al., 2023`)的出现标志着**人工智能**领域的重要发展。当放宽这些教条时，会得到一种将**强化学习**视为**智能体科学研究**的观点，这一愿景与**强化学习**和**人工智能**的经典教材(`Sutton & Barto, 2018；Russell & Norvig, 1995`)以及**控制论**(`Wiener, 2019`)的既定目标密切相关。作为重要的特殊情况，这些**智能体**可能与**马尔可夫决策过程**(`MDP；Bellman, 1957；Puterman, 2014`)进行了互动，寻求特定问题的解决方案，或在存在**奖励信号**的情况下进行学习，以最大化该信号，但这些并不是唯一感兴趣的情况。

##### 教条一：环境聚光灯

第一个教条为**环境聚光灯**，它指的是集体关注于**建模环境**和以**环境为中心**的概念，而非**智能体**。例如，**智能体**本质上是解决**马尔可夫决策过程**(`MDP`)的工具，而不是一个独立的、具体的模型。**人工智能科学**的本质最终是关于**智能体**的，然而，我们的思维方式，以及数学模型、分析和核心结果往往围绕解决特定问题展开，而不是围绕**智能体**本身。换句话说，我们缺乏一个规范的**智能体正式模型**。这就是第一个教条的本质。专注于环境的意思是需要搞清楚以下`2`个问题：1、在**强化学习**中，至少有一个规范的**环境数学模型**是什么？2、在**强化学习**中，至少有一个规范的**智能体数学模型**是什么？这些问题的重点在于，尽管我们在**强化学习**中对环境的建模有明确的框架和理论基础，但对于**智能体**本身的建模却缺乏相应的规范模型。这种不平衡反映了我们在研究中对**环境**的过度关注。

第一个问题（什么是规范的**环境模型**？）有一个直接的答案：**马尔可夫决策过程**(`MDP`)，或其附近的变体，如`k-`**臂老虎机**、**上下文老虎机**或**部分可观察马尔可夫决策过程**(`POMDP`)。每个模型都编码了不同版本的**决策问题**，受不同结构假设的影响——以**马尔可夫决策过程**(`MDP`)为例，通过假设存在一个可维护的信息集合（称之为**状态**），该**状态**是下一个奖励和同一信息集合的下一个分布的**充分统计量**，从而做出了**马尔可夫假设**。假设这些状态由环境定义，并且在每个时间步都可以被**智能体**直接观察，以用于学习和决策。**部分可观察马尔可夫决策过程**(`POMDP`)放宽了这一假设，而是仅向**智能体**揭示**观察结果**，而不是**状态**。通过接受`MDP`，能够引入多种算法。例如，知道每个`MDP`至少有一个确定性的、最优的、静态的**策略**，并且**动态规划**可以用来识别该**策略**(`Bellman, 1957；Blackwell, 1962；Puterman, 2014`)。此外，还探索了**马尔可夫决策过程**(`MDP`)的变体，例如**块状**`MDP`(`Du et al., 2019`)、**丰富观察**`MDP`(`Azizzadenesheli et al., 2016`)、**面向对象的**`MDP`(`Diuk et al., 2008`)、`Dec-POMDP`(`Oliehoek et al., 2016`)、**线性**`MDP`(`Todorov, 2006`)和**分解**`MDP`(`Guestrin et al., 2003`)等。这些模型各自突出了不同类型的问题或结构假设，并激发了大量启发性的研究。

第二个问题（什么是规范的**智能体模型**？）没有明确的答案。可能会倾向于用一种特定的**流行学习算法**（指在**机器学习**和**强化学习**领域中广泛应用的一类算法），例如`Q-Learning`，但这是错误的。`Q-Learning`只是可以支持**智能体**的逻辑的一种实例，但它并不是对智能体是什么的通用抽象，无法与`MDP`作为**广泛序列决策问题模型**的地位相提并论。我们缺少一个规范的**智能体模型**，甚至缺乏一个基本的概念框架。在当前阶段，这已成为一个限制，部分原因在于对环境的关注。实际上，**专注于以环境为中心的概念**（例如**动态模型**、**环境状态**、**最优策略**等）往往会掩盖**智能体**自身的重要角色。因此，在探索直接涉及**智能体**的问题时能力不足。但在这里，希望审视**以智能体为中心的范式**的兴趣，这样可以提供探索**智能体**原则所需的概念清晰性。如果没有这样的基础，难以准确定义和区分关键的**智能体家族**，如“**有模型**”和“**无模型**”**智能体**，或研究有关**智能体**与**环境边界**(`Jiang, 2019；Harutyunyan, 2020`)、**扩展心智**(`Clark & Chalmers, 1998`)、**嵌入式代理**(`Orseau & Ring, 2012`)、**具身性影响**(`Ziemke, 2013；Martin, 2022`)或**资源约束影响**(`Simon, 1955；Ortega et al., 2015；Griffiths et al., 2015；Kumar et al., 2023；Aronowitz, 2023`)等更复杂的问题。大多数**以智能体为中心**的概念通常超出了基本数学语言的范围。替代方案：关注智能体，除了问题和环境外，定义、建模和分析智能体也很重要。应该朝着**一个规范的智能体数学模型**迈进，我们应该进行基础工作，以建立**表征重要智能体**属性和家族的公理。借鉴心理学、认知科学、哲学、生物学、人工智能和博弈论等多个研究**智能体**的学科。这样做可以扩大科学研究的视野，以理解和设计**智能体**。
{% asset_img ml_5.png "环境聚光灯" %}

##### 教条二：学习作为寻找解决方案

第二个教条嵌入在对**学习概念**的处理方式中。倾向于将**学习**视为一个有限的过程，涉及对给定任务的解决方案的搜索和最终发现。例如，考虑一个**强化学习智能体**学习玩棋盘游戏的经典问题，如双陆棋或围棋。在这些情况下，通常假设一个好的**智能体**会进行大量游戏，以学习如何有效地玩游戏。最终在足够多的游戏之后，**智能体**将达到最佳玩法，并可以停止学习，因为所需的知识已经获得。换句话说，我们往往隐含地假设设计的**学习智能体**最终会找到当前任务的解决方案，此时学习可以停止。也同样出现在许多经典基准测试中，在这些测试中，**智能体**会学习直到达到目标。从某种角度看，这些**智能体**可以被理解为在**可表示函数**的空间中**搜索**，这些函数捕捉了**智能体**可用的**动作选择策略**，类似于**问题空间假设**(`Newell, 1994`)。关键是，这个空间至少包含一个**函数**——例如`MDP`的**最优策略**——其质量足以认为任务已解决。通常，希望设计能够保证收敛到终点的**学习智能体**，此时**智能体**可以停止其搜索（停止学习）。这种观点嵌入到许多目标中，并且很自然地将**马尔可夫决策过程**(`MDP`)作为**决策问题模型**来使用。众所周知，每个`MDP`至少有一个最优的**确定性策略**，并且可以通过**动态规划**或其**近似方法**进行学习或计算。替代方案：**学习作为适应**。接受**学习**也可以被视为**适应**的观点。因此，我们的关注将从**最优性**转向一种**强化学习问题**的版本，其中**智能体**不断改进，而不是专注于试图解决特定问题的**智能体**。当然，这种问题的版本已经通过**终身学习**(`Brunskill & Li, 2014；Schaul et al., 2018`)、**多任务学习**(`Brunskill & Li, 2013`)和**持续强化学习**(`Ring, 1994；1997；2005；Khetarpal et al., 2022；Anand & Precup, 2023；Abel et al., 2023b；Kumar et al., 2023`)的视角进行了探索。当我们从**最优性**转向**适应性**时，如何看待**评估**？如何准确地定义这种学习形式，并将其与其他形式区分开来？执行这种学习形式的基本算法构建块是什么？它们与今天使用的算法有何不同？标准分析工具，如**遗憾**和**样本复杂度**，仍然适用吗？这些问题都很重要。
{% asset_img ml_6.png "学习作为寻找解决方案" %}

##### 教条三：奖励假设

第三个教条是**奖励假设**，其表述为：“**我们所理解的所有目标和目的都可以被视为最大化所接收标量信号（奖励）累积和的期望值**”。首先，是要承认，这一假设并不值得被称为“**教条**”。最初提出时，**奖励假设**旨在对目标和目的的思考，类似于之前的**期望效用假设**。而且，**奖励假设**为**强化学习**的研究奠定了基础，促成了许多应用和算法的发展。然而，在继续追求**智能体设计**的过程中，认识到这一假设的细微差别。特别是，`Bowling`等人(`2023`)的最新分析，全面阐明了**奖励假设**成立所需的隐含条件。这些条件有两种形式。首先，`Bowling`等人提供了一对**解释性假设**，澄清了**奖励假设**为真或为假的含义——大致上，这可以归结为两点。第一，“**目标和目的**”可以通过对结果的偏好关系来理解。第二，如果由**价值函数**引导的**智能体**排序与由**智能体**结果上的偏好引导的排序相匹配，则**奖励函数**捕捉到这些偏好。在这种解释下，只有当**偏好关系**满足`4`个冯·诺依曼-摩根斯坦公理以及`Bowling`等人称之为{% mathjax %}\gamma{% endmathjax %}`-Temporal`无差异的第`5`个公理时，**马尔可夫奖励函数**才存在以捕捉偏好关系。这一点非常重要，因为它表明，当写下一个**马尔可夫奖励函数**以捕捉期望的目标或目的时，我们实际上是在强迫目标或目的遵循这`5`个公理。也就是说，某些抽象美德，如幸福和正义，可能被认为是不可比较的。或者，同样，两种具体经历可能是不可测量的，例如在海滩散步和吃早餐——我们如何能用相同的标准来衡量这些经历呢？`Chang`指出，两项事物可能在没有进一步参考**特定用途**或**上下文**的情况下不可比较：“一根棍子不能比一颗台球更大……它必须在某方面更大，例如质量或长度。”然而，第一个公理，即**完备性**，严格要求隐含的偏好关系在所有经验对之间分配真实偏好。因此，如果认为**奖励假设**为真，只能在拒绝**不可比较性**和**不可测量性**的情况下，将目标或目的**编码**到**奖励函数**中。值得注意的是，**完备性**特别受到`Aumann`的批评，因为它对持有偏好关系的个体提出了要求。最后，**完备性公理**并不是唯一限制可行目标和目的空间的公理；第三公理，**无关选择的独立性**，由于**阿莱悖论**拒绝了风险敏感目标。实际上，`Skalse`和`Abate`(`2023`)证明**马尔可夫奖励**无法捕捉**风险敏感**或**多标准目标**，而`Miura`(`2022`)同样证明**多维马尔可夫奖励**在表达能力上严格优于**标量**。替代方案：**认识并接受细微差别**。意识到**标量奖励**的局限性，并对描述**智能体**目标的其他语言保持开放态度。当通过**奖励信号**来表示一个目标或目的时，重要的是要认识到对可行目标和目的所施加的**隐含限制**。值得强调的是，**偏好**本身只是**表征目标**的另一种语言——可能还有其他语言，因此在思考目标追求时，采取广泛的视角是很重要的。

**强化学习**(`RL`)的长期愿景是为**智能体科学**提供一个**整体范式**。为了实现这一愿景，是时候重新审视与塑造**强化学习**的`3`种**隐含教条**的关系。这`3`种教条过于**强调环境**、**找解决方案**，以及**奖励作为描述目标的语言**。此外，应该将**智能体**视为研究的中核心对象之一。其次，必须超越仅研究为**特定任务**寻找解决方案的**智能体**，转而研究那些能够从经验中不断改进的**智能体**。最后，应该认识到将**奖励作为目标语言的局限性**，并考虑其他替代方案。**规范智能体模型**是什么？如***智能体与环境边界***、具身性、资源约束和嵌入式代理。学习的目标是什么？当找不到最优解决方案时，如何看待学习？我们如何开始评估这样的智能体，并衡量它们的学习进展？
