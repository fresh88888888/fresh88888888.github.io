---
title: 机器学习(ML)(二十三) — 强化学习探析
date: 2025-01-02 16:00:11
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

#### 递归内省

**递归内省**(`Recursive Introspection`)是一种新方法，旨在教授**语言模型智能体**（如**大语言模型**，`LLMs`）如何自我改进。该方法的核心在于使模型能够对自身行为进行**内省**、**推理**并**纠正错误**。其主要特点：**自我改进能力**，**递归内省**的目标是使**语言模型**能够在多轮交互中逐步改善其响应。这种方法强调通过反复的**反馈**和**调整**，模型能够识别并纠正先前的错误；`RISE`**方法**，该方法被称为`RISE`(`Recursive IntroSpEction`)，是一种**微调技术**，允许模型在面对复杂问题时，通过观察之前的失败尝试和额外的环境反馈来调整其**策略**；**多轮数据收集与训练**，`RISE`借鉴了**在线模仿学习**和**强化学习**的原则，提出了多轮数据收集和训练策略，以增强`LLM`在后续迭代中**递归检测**和**纠正错误**的能力。
<!-- more -->

`RISE`将单轮提示的**微调**视为解决多轮**马尔可夫决策过程**(`MDP`)，其中初始状态为提示。受**在线模仿学习**和**强化学习**的启发，提出了多轮数据收集和训练策略，以使`LLM`具备在后续迭代中**递归检测**和纠正其先前错误的能力。实验表明，`RISE`使`Llama2、Llama3`和`Mistral`模型能够通过在更多回合在数学推理任务上自我改进，在相同推理时间计算下超越了几种单轮策略。还发现`RISE`具有良好的扩展性，通常在更强大的模型上获得更大的收益。`RISE`对响应做出了改进，使其能够在不干扰单轮能力的情况下找到挑战性提示的正确解决方案。
{% asset_img ml_1.png %}

**递归内省**(`RISE`)是一种利用迭代多轮训练的方法，旨在通过**策略回放**和**奖励函数**的监督来训练模型，使其能够在多个**回合**中自我改进。在**推理阶段**，对来自不同**回合**的候选输出进行多数投票，以获得最终响应。

可以训练模型使其具备自我改进响应的能力吗？如果做到这一点，并在多样化的问题和场景中进行训练，这可能为**大语言模型**(`LLM`)引入一种通用的方法，指导其如何通过自我改进来应对困难提示，而不是仅仅监督其“应该”如何响应，因为这种方法在测试提示超出分布时可能不具备**泛化**能力。尽管一种直接的方法是生成多个顺序**回合**改进的数据，但仅仅模仿这些数据并不足以赋予模型这种能力。这主要有两个原因：首先，来自不同模型的多轮数据不会展示学习者所犯错误的改进，因此对学习者而言是无关的。其次，通常从专有模型收集的顺序多轮数据质量也不高，因为这些模型通常不擅长提出对自身错误的有意义改进。因此，需要一种不同的策略来赋予模型自我改进的能力。关键是以迭代方式监督学习者自身响应的改进，借鉴**在线模仿学习**和**强化学习**(`RL`)中的方法。这种**监督**可以是从更强大模型中**独立同分布抽样**得到的对提示的`oracle`（指**提供最优策略或环境模型的信息源**）响应，或者由学习者自身生成。**递归内省**(`RISE`)在给定提示的多个尝试中提高`LLM`的自我改进能力。在每次迭代中，**递归内省**(`RISE`)从学习者的**策略回放**中引导出更好的下一回合响应，这些响应是通过在多个修订候选中运行最佳{% mathjax %}N{% endmathjax %}（使用任务成功指标）获得的，这些候选可以是从学习者自身抽样得到的，也可以是使用更强大模型的响应。通过这种方式，能够构建回放，使学习者了解如何在自身分布下改善其响应。然后，使用**奖励加权回归**(`RWR`)目标对学习者进行**微调**，该目标能够从这些**回放**的高质量和低质量部分中学习。通过反复迭代这一过程，能够将自我改进能力灌输到`LLM`中。结果表明，通过**递归内省**(`RISE`)训练的`LLM`能够在更多提示上产生正确响应，在更具难度的提示上随着**回合**数增加而改善。尽管强大的基础和指令调优**大语言模型**(`LLMs`)在多个顺序尝试中常常未能改善其响应，**递归内省**(`RISE`)成功地赋予了类似规模的`LLM`自我改进能力，使其在每个回合后的任务表现单调增加。
{% mathjax '{"conversion":{"em":14}}' %}
\underset{\pi_{\theta}}{\max}\;\sum\limits_{i=1}^L \mathbb{E}_{x,y^\sim \mathcal{D},\hat{y}_i\sim \pi_{\theta}(\cdot|[x,\hat{y}_{1:i-1},p_{1:i-1}])}[\mathbb{I}(\hat{y}_i == y^*)]
{% endmathjax %}
具体而言，给定一个数据集{% mathjax %}\mathcal{D} = \{(x_i,y_i^*)\}_{i=1}^N{% endmathjax %}，其中包含问题 {% mathjax %}x_i{% endmathjax %}和对应的`oracle`响应{% mathjax %}y_i{% endmathjax %}，目标是获得一个`LLM`{% mathjax %}\pi_{\theta}(\cdot|[x,\hat{y}_{1:t},p_{1:t}]){% endmathjax %}，该模型在给定问题{% mathjax %}x{% endmathjax %}、之前模型对该问题的尝试{% mathjax %}\hat{y}_{1:t}{% endmathjax %}，以及**辅助指令**{% mathjax %}p_{1:t}{% endmathjax %}（例如，查找错误并改进响应的指令；或来自环境的额外编译器反馈）时，尽可能正确地解决给定问题。为此，将这一**目标编码**为优化的以下学习目标：与标准的**监督微调**不同，后者训练模型{% mathjax %}\pi{% endmathjax %}在给定{% mathjax %}x{% endmathjax %}的情况下产生单一响应{% mathjax %}\hat{y}{% endmathjax %}，训练{% mathjax %}\pi{% endmathjax %}对其自身先前的响应历史{% mathjax %}\hat{y}_{1:i-1}{% endmathjax %}作出反应。通过将**单回合问题**转换为**多回合马尔可夫决策过程**(`MDP`)。需要注意，基于提示的方法(`Self-Refine`)仍然可以被视为训练{% mathjax %}\pi{% endmathjax %}优化{% mathjax %}\pi(y^*|x){% endmathjax %}，但仅在允许调节提示{% mathjax %}\pi{% endmathjax %}来优化以上公式，由于参数{% mathjax %}\theta{% endmathjax %}不变，这样做并不能有效地完全优化该目标。

**递归内省**(`RISE`)方法，首先将问题转换为**多回合马尔可夫决策过程**，然后收集数据，最后在这个**多回合马尔可夫决策过程**中运行**离线奖励加权监督学习**。
{% asset_img ml_2.png %}

将**单回合问题**转换为**多回合马尔可夫决策过程**，**状态**由**提示**、**先验的历史**和来自环境的**可选反馈**组成。**动作**是基于迄今为止多轮交互状态生成的`LLM`响应。**数据收集**：通过将当前模型展开{% mathjax %}k-1{% endmathjax %}次，然后生成改进版本的响应来收集数据，该响应可以通过以下方式获得：(1)**自我蒸馏**：从当前模型中采样多个响应，并使用最佳的响应；(2)**蒸馏**：通过查询更强大的模型获得`oracle`响应。

**思维链**(`Chain-of-Thought, CoT`)是一种提升**大语言模型**(`LLM`)在复杂**推理任务**上的技术。它的核心理念是**模拟人类的推理过程**，通过逐步推导出一系列中间步骤或子目标，从而最终得出正确答案。其特点：
- **逐步推理**：`CoT`技术要求模型在生成最终答案之前，先产生一系列中间推理步骤。这些步骤构成了一个“**思维链**”，帮助模型更清晰地理解问题并找到解决方案。
- **可解释性**：由于`CoT`提供了推理过程的可见性，用户可以更容易理解模型的决策过程，从而提高模型的可解释性。
- **逻辑推理能力**：`CoT`能够帮助模型进行复杂的逻辑推理，特别是在需要综合多个事实或信息片段的问题上。
- **上下文利用**：在`CoT`中，模型可以利用上下文信息，通过逐步推理来解决问题，而不是仅仅依赖于直接的答案。

构建**递归内省**(`RISE`)方法的第一步是将**单回合数据集**的提示和`oracle`**响应**构建为**多回合马尔可夫决策过程**。给定一个数据集{% mathjax %}\mathcal{D} = \{(x_i,y_i)\}{% endmathjax %}，其中包含提示{% mathjax %}x_i{% endmathjax %}和相应的`oracle`响应{% mathjax %}y_i^*{% endmathjax %}（例如，**数学问题**及其**自然语言响应**），将从{% mathjax %}\mathcal{D}{% endmathjax %}构建一个诱导的`MDP`{% mathjax %}\mathcal{M}{% endmathjax %}，然后在这个`MDP`中学习**策略**。该`MDP`中的初始状态是一个提示{% mathjax %}x_i \in \mathcal{D}{% endmathjax %}。将基础模型的输出响应表示为动作{% mathjax %}a{% endmathjax %}。给定状态{% mathjax %}s{% endmathjax %}，下一个状态可以通过将表示状态{% mathjax %}s{% endmathjax %}的标记与模型提出的动作{% mathjax %}a{% endmathjax %}以及一个额外的固定提示{% mathjax %}f{% endmathjax %}连接起来获得。**奖励函数**是一个**稀疏**的**二元指标**，用于指示在给定状态{% mathjax %}s{% endmathjax %}下答案的正确性，定义为：{% mathjax %}s,r([x_i,\ldots],a) = 1{% endmathjax %}当且仅当{% mathjax %}a = y_i^*{% endmathjax %}，并由**答案检查函数**获得。这种从数据集{% mathjax %}\mathcal{D}{% endmathjax %}到`MDP`{% mathjax %}\mathcal{M}{% endmathjax %}的构造如下所示：
{% mathjax '{"conversion":{"em":14}}' %}
\begin{align}
\mathcal{D} & = \{(x_i,y_i^*)\}\;\rightarrow \;\mathcal{M}\;:\;p(s_0) = \text{Unif}(x_1,x_2,\ldots,x_N) \\
P(s'|s,a) & = \delta(s' = \text{concat}[s,a,f]) \\
r(s,a) & = 1\;(a = y^*_i\;\text{if }x_i\in s)
\end{align}
{% endmathjax %}
在`MDP`构建完成后，下一步是训练模型在**回放**过程中自我改进。可以采用一种**离线学习**的方法，具体描述如下：
- **步骤一**：**自我改进**的数据收集，为了确保这个**多回合马尔可夫决策过程**(`MDP`)的回放数据对教授模型如何**自我改进**是有用的，它必须满足几个条件：(1)必须展示学习者可能犯的错误，并展示如何在下一次尝试中改进这些错误；(2)数据必须展示与给定问题和上下文中先前尝试相关的响应；(3)必须不包含在后续**回合**中退化的**回放**。在给定回合{% mathjax %}k{% endmathjax %}中，对于给定问题{% mathjax %}x_i{% endmathjax %}，展开当前模型{% mathjax %}\pi_{\theta_k}(\cdot|\cdot){% endmathjax %}尝试生成多个顺序，记作{% mathjax %}y^i_t\sim \pi_{\theta_k}(\cdot|s^i_t){% endmathjax %}。在有外部输入（例如，**编译器反馈**）的情况下，观察到一个可变长度的**自然语言**外部输入{% mathjax %}f^i_t{% endmathjax %}（例如，在数学问题中，要求模型**自我纠正**）。还观察到一个**标量奖励值**{% mathjax %}r(s^i_t,y^i_t){% endmathjax %}，简称为{% mathjax %}r^i_t{% endmathjax %}，将这个模型**回放**的数据集记作{% mathjax %}\mathcal{D}_{\text{on-policy}}:= \{(s^i_t,y^i_t,f^i_t)^T_{t=1}\}{% endmathjax %}。对于每个时间步，构建响应{% mathjax %}y^i_t{% endmathjax %}，记作{% mathjax %}\tilde{y}^i_t{% endmathjax %}。与这个改进响应相关的**奖励分数**为{% mathjax %}r(s^i_t,\tilde{y}^i_t){% endmathjax %}，或简称为{% mathjax %}\tilde{r}^i_t{% endmathjax %}。为了获得响应{% mathjax %}y^i_t{% endmathjax %}的改进版本，可以采用几种策略。最直接的方法是查找一个更强大的模型，根据提示{% mathjax %}x^i{% endmathjax %}、先前响应{% mathjax %}y^i_t{% endmathjax %}和外部反馈{% mathjax %}f^i_t{% endmathjax %}（可选）提供正确的响应。将其称为**蒸馏变体**，因为它使用强大的“**教师**”**模型**来指导**自我改进**（请注意，这与经典的**知识蒸馏**概念不同）。
{% mathjax '{"conversion":{"em":14}}' %}
\tilde{\mathcal{D}}_{\text{on-policy + distill}} := \bigg\{\{(s^i_t,\tilde{y}_t^i,f^i_t,\tilde{r}^i_t)\}_{t=1}^T \bigg\}_{i=1}^{|\mathcal{D}|}
{% endmathjax %}
第二种变体，旨在减轻对**教师模型**的依赖，通过从学习者自身多次采样来构建改进响应。将这种方法称为**自我蒸馏变体**。具体而言，对于数据集中每个状态{% mathjax %}s^i_t\in \mathcal{D}_{\text{on-policy}}{% endmathjax %}，从模型中采样{% mathjax %}N{% endmathjax %}个响应{% mathjax %}\tilde{y}^i_t[0],\tilde{y}^i_t[1],\ldots,\tilde{y}^i_t[N]\sim \pi_{\theta}(\cdot|s^i_t){% endmathjax %}，并使用这{% mathjax %}N{% endmathjax %}个候选响应中最好的一个（根据**奖励值**{% mathjax %}\tilde{r}^i_t[0],\tilde{r}^i_t[1],\ldots,\tilde{r}^i_t[N]{% endmathjax %}来衡量）来重新标记改进轨迹中下一步{% mathjax %}t+1{% endmathjax %}的模型响应。形式上，设{% mathjax %}\tilde{y}^i_t[m] = \text{arg }\max_{j\in[N]}r(s_i,\tilde{y}^i_t[j]){% endmathjax %}，那么在步骤{% mathjax %}t+1{% endmathjax %}中将数据集{% mathjax %}\mathcal{D}_{\text{on-policy}}{% endmathjax %}中的响应标记为**改进响应**及其相关的**奖励值**{% mathjax %}\tilde{r}^i_t[m]{% endmathjax %}：
{% mathjax '{"conversion":{"em":14}}' %}
\tilde{\mathcal{D}}_{\text{on-policy + self-distillation}} := \bigg\{\{(s^i_{t+1},\tilde{y}_t^i[m],f^i_{t+1},\tilde{r}^i_t[m])\}_{t=0}^{T-1} \bigg\}_{i=1}^{|\mathcal{D}|}
{% endmathjax %}
- **步骤二**：**策略改进**，通过上上面的数据构建方案，现在可以在这些数据集上**训练模型**。一般来说，可以使用任何**离线强化学习**(`RL`)方法在这些数据上进行训练，也可以使用**加权监督学习**的方法。执行**加权监督回归**，其中权重由数据集{% mathjax %}\tilde{\mathcal{D}}{% endmathjax %}中**奖励值**的**指数变换**给出。
{% mathjax '{"conversion":{"em":14}}' %}
\text{Reward-Weighted RL:  }\underset{\theta}{\max}\;\mathbb{E}_{x^i\sim \tilde{\mathcal{D}}}\bigg[ \sum\limits_{t=1}^T \log\pi_{\theta}(\tilde{y}^i_t|s^i_t)\cdot \exp(r^i_t/\tau)\bigg]
{% endmathjax %}
**温度参数**{% mathjax %}\tau{% endmathjax %}用于进一步扩展或缩小良好和不良动作之间的差异。在初步实验中，发现以上公式会导致偏向于提高高奖励响应的**对数似然**，优先更新那些奖励已经很高的简单问题。为了解决这个问题，对以上公式进行了轻微修改，使得指数化的奖励围绕在给定提示的所有尝试中平均的均值进行**中心化**，这类似于**优势加权回归**(`advantage-weighted regression, AWR`)。使用**优势**代替**奖励**有助于避免在简单问题上出现“**富者愈富**”的现象。

**优势加权回归**(`Advantage-Weighted Regression, AWR`)是一种**强化学习**中的**策略优化**方法，旨在通过利用**优势函数**来改进学习过程。它的核心思想是通过对**回报**进行**加权**，从而增强学习信号的质量，并有效利用历史数据，即使在非互动环境中也能发挥作用。

**在部署时推理**：`RISE`可以在推理时使用两种模式运行。最直接的方式是通过**多回合回放运行**由`RISE`训练的策略{% mathjax %}\pi_{\theta}(\cdot|cdot){% endmathjax %}，在这种模式下，模型根据过去的**上下文**（即多回合`MDP`中的状态）采样新的响应。这个过去的上下文包括与响应{% mathjax %}y^{\text{test}}_i{% endmathjax %}相关的外部反馈{% mathjax %}p^{\text{test}}_i{% endmathjax %}，并且一旦当前响应根据环境的**答案验证函数**被判断为正确，**回放**就会终止。换句话说，在奖励等于`oracle`**响应**的奖励时终止**回放**：{% mathjax %}r(x,y^{\text{test}}_i) = r(x,y^*){% endmathjax %}。这个协议在每个**回合**后调用**奖励函数**进行查询。由于执行了多个**奖励函数**查询，我们将这种方法称为“有`oracle`”。`RISE`还可以在一种模式下运行，该模式避免在**回放**过程中查询**结果检查器**或**奖励函数**。在这种情况下，通过强制模型重试来运行完整长度的**回放**，而忽略响应的正确性。然后，利用基于**多数投票**的**自一致机制**来决定每个**回合**结束时的候选响应。具体而言，在每个回合{% mathjax %}j{% endmathjax %}结束时，通过对前几个**回合**的所有响应候选进行**多数投票**来确定响应{% mathjax %}(\text{maj}\big(y_1^{\text{test}},y_2^{\text{test}},\ldots,y_j^{\text{test}}\big)){% endmathjax %}，包括第{% mathjax %}j{% endmathjax %}回合。称之为“无`oracle`”。大多数评估使用的是**无**`oracle`的方法。在迭代{% mathjax %}k{% endmathjax %}中，由于智能体能够从{% mathjax %}j{% endmathjax %}改进其响应到{% mathjax %}j+1{% endmathjax %}（当{% mathjax %}j\leq k{% endmathjax %}时），为了避免测试时分布偏移，在这两种模式下，当**回合**数{% mathjax %}j{% endmathjax %}大于迭代数{% mathjax %}k{% endmathjax %}时，使用大小为{% mathjax %}k{% endmathjax %}的**滑动窗口**来存储最近的对话历史。
{% asset_img ml_3.png %}

如上图所示，`RISE`**推理**时有两种查询模型的方式：(1)有`oracle`（左侧）：每当模型改进其响应时，它可以检查其答案与环境的匹配，并在找到正确答案后提前终止；(2)无`oracle`（右侧）：要求模型顺序修正其自身的响应{% mathjax %}j{% endmathjax %}次，并对来自不同**回合**的所有候选输出进行**多数投票**，以获得最终响应。如果回合数{% mathjax %}j{% endmathjax %}大于迭代数{% mathjax %}k{% endmathjax %}，则**智能体**仅保留最近的历史记录，限制为{% mathjax %}k{% endmathjax %}次交互，以避免测试时**分布偏移**。

#### 多智能体模仿学习

**多智能体模仿学习**(`Multi-Agent Imitation Learning, MAIL`)是一种研究如何通过模仿**专家**的行为来训练多个**智能体**的学习方法。该方法的核心思想是利用**专家**在特定环境中的示范，帮助学习者协调和优化一组**智能体**的行为。主要特点：**模仿学习基础**，**多智能体模仿学习**(`MAIL`)基于**模仿学习**的原理，旨在让**智能体**通过观察专家的行为来学习如何在复杂环境中做出**决策**，而无需依赖明确的**奖励信号**；**行为匹配**，传统的**模仿学习方法**通常将问题简化为在**专家**示范的支持范围内匹配**专家**的行为。这种方法在**非战略性智能体**中能够有效地减少**学习者**与**专家**之间的价值差距，但对于具**有战略性的智能体**则可能不够**鲁棒**；**遗憾差距**，在**多智能体**环境中，由于**智能体**之间可能存在**战略性偏离**，因此需要引入新的目标，例如“**遗憾差距**”，以更好地处理可能的**策略偏离**。这种方法考虑了**智能体**在面对不同状态时可能采取的不同策略，从而提高了系统的**稳定性**和**鲁棒性**。**多智能体模仿学习**(`MAIL`)的方法可以包括**行为克隆**(`Behavior Cloning`)和**逆强化学习**(`Inverse Reinforcement Learning`)。**行为克隆**直接模仿**专家**的动作，而逆**强化学习**则试图推断出**专家**行为背后的**奖励函数**。处理**战略性偏离**是**多智能体模仿学习**(`MAIL`)面临的一大挑战。未来的研究可能会集中在如何设计更有效的算法，以应对**智能体**之间的相互影响和复杂交互。未来**多智能体模仿学习**(`MAIL`)也需要适应动态变化的环境，以实现更高效的**协作**和**决策**能力。

**遗憾差距**(`Regret Gap`)是**多智能体模仿学习**(`MAIL`)中的一个重要概念，旨在解决**智能体**之间的**策略协调**问题。与传统的**价值差**(`Value Gap`)不同，**遗憾差距**明确考虑了**智能体**可能的**策略偏离**，提供了一种新的**目标函数**来提高系统的**鲁棒性**。**遗憾差距**定义：**遗憾差距**是指在**多智能体**系统中，由于**智能体**的**策略偏离**而导致的性能损失。它关注的是在给定**状态**下，**智能体**在选择不同**策略**时可能产生的**后悔值**。**价值差**通常用于衡量**学习者**与**专家**之间的**行为匹配程度**，而**遗憾差距**则更关注在面对**战略性偏离**时的表现。研究表明，即使实现了**价值等价**，**遗憾差距**仍然可能非常大，这意味着在**多智能体模仿学习**(`MAIL`)中实现**后悔等价**比实现**价值等价**更为复杂。为了**最小化遗憾差距**，研究者提出了两种有效的方法：`MALICE`(**在专家覆盖假设下**)和`BLADES`(**在可查询专家的情况下**)。这些方法通过将问题规约到**无后悔在线凸优化**中，从而有效地处理**策略偏离**的问题。**遗憾差距**作为**多智能体模仿学习**(`MAIL`)中的新目标，为处理**智能体**之间的**策略协调**和**鲁棒性**问题提供了新的视角。通过深入研究**价值差**与**后悔差**之间的关系，以及提出有效的算法来**最小化遗憾差距**，可以显著提升**多智能体系统**在复杂环境中的表现和稳定性。

**无后悔在线凸优化**(`No-Regret Online Convex Optimization`)是一种在**线学习框架**，旨在通过优化算法在动态环境中有效地处理**凸优化问题**。该方法的核心思想是设计能够在面对对手或环境变化时，保**证学习者**的**决策**不会产生显著的**后悔值**。特点：在线学习，**无后悔在线凸优化**关注的是在每一轮决策中，**学习者**根据当前信息选择一个**决策**，并在后续观察到损失。**学习者**的目标是通过不断更新**策略**来**最小化累积损失**；**凸优化**，该方法假设**损失函数**是凸的，这意味着任何**局部最优解**都是**全局最优解**。**凸函数**具有良好的数学性质，使得优化过程更为稳定和可预测；**后悔度**，**后悔度**是指在**在线学习**过程中，**学习者**所遭受的**实际损失**与**最优静态策略**所能获得的损失之间的差距。**无后悔在线凸优化**算法旨在确保随着时间的推移，**学习者**的**后悔度**逐渐减小。其算法包括：**在线梯度下降**(`Online Gradient Descent, OGD`)`OGD`是**无后悔算法**中最经典的算法之一。它通过在每个时间步使用当前**损失函数**的**梯度**来更新模型，从而实现对**损失函数**的有效优化；**在线牛顿步**(`Online Newton Step, ONS`)`ONS`是另一种用于处理**在线凸优化问题**的方法，特别适用于**强凸函数**。它通过利用**二阶信息**来加速收敛，并能够达到更优的**后悔界限**。

**后悔界限**(`Regret Bound`)是一个重要的概念，尤其在**在线学习**和**决策理论**中。它用于描述在特定**决策**过程中，**学习者**与**最优策略**之间的性能差距，具体体现在**后悔值**的**上限**。特点：**后悔界限**是指在一系列**决策**中，**学习者**所遭受的**实际损失**与**最优静态策略**所能获得的损失之间的**最大差距**。它衡量的是**学习者**在面对动态环境时，未能选择**最优策略**所造成的潜在损失；**后悔值**，**后悔值**是指**学习者**在每个时间步骤中，基于其选择的动作与所有可能动作的最佳结果之间的差异。**后悔界限**则提供了一个数学上可量化的**上限**，表示随着时间推移，**学习者**的**后悔值**不会无限增长；**无后悔算法**，在**无后悔在线凸优化**中，算法设计的目标是确保**后悔界限**随着时间的推移而收敛到`0`。这意味着，通过足够多的学习和调整，**学习者**的**决策**将逐渐接近**最优决策**。

**后悔界限**在**在线学习算法**中起着核心作用，特别是在需要**实时决策**和适应环境变化的场景，如金融市场、广告投放和推荐系统中。在**多智能体**环境中，如果所有**智能体**都采用**无后悔学习算法**，那么它们的联合行为将渐进地收敛于一组**无后悔点**。**后悔界限**帮助分析和优化**智能体**之间的协调与合作。在数学上，**后悔界限**通常通过不等式来表示，例如：
{% mathjax '{"conversion":{"em":14}}' %}
R(T) \leq B
{% endmathjax %}
其中{% mathjax %}R(T){% endmathjax %}是在{% mathjax %}T{% endmathjax %}次决策中的**总后悔值**，而{% mathjax %}B{% endmathjax %}是一个常数或函数，表示**后悔界限**。通过设计有效的算法，研究者可以控制这个**界限**，使其保持在可接受范围内。

**策略偏离**(`Strategic Deviation`)是**多智能体系统**和**博弈论**中的一个重要概念，指的是**智能体**在决策过程中选择与**预定策略**不同的行动。这种偏离可能是出于自我利益的考虑，尤其是在存在多个**智能体**相互影响的环境中。特点：**自利行为**，**智能体**可能会因为追求个人利益而偏离团队策略。例如，在合作博弈中，某个智能体可能选择背叛而不是合作，以期获得更高的短期收益；**对系统的影响**，**策略偏离**可能导致整体系统性能下降，因为个别**智能体**的**自利行为**可能破坏团队的协调与合作。例如，在**多智能体强化学习**中，如果某个智能体偏离了团队策略，可能会导致整个团队无法达到最优解；**博弈论中的均衡**，**博弈论背景**，**纳什均衡**是一种状态，其中没有参与者能够通过单方面改变自己的策略来获得更好的收益。**策略偏离**意味着某个**智能体**试图打破这种均衡，从而可能导致新的均衡状态或不稳定性。**策略偏离**是**多智能体系统**中的一个关键问题，它影响着系统的稳定性和效率。通过合理设计**奖励机制**、使用**后悔最小化算法**以及**强化学习**中的合作机制，可以有效应对**策略偏离**带来的挑战，提高**多智能体系统**的整体性能。假设开发一个路由应用程序，为一组用户提供个性化的路线推荐（{% mathjax %}\sigma{% endmathjax %}），这些用户具有联合策略{% mathjax %}\pi{% endmathjax %}（例如，`Google Maps`中提供的路由策略）。与**模仿学习**(`IL`)中的常规假设一样，假设访问来自**专家**{% mathjax %}\sigma_E{% endmathjax %}的示范（例如，历史版本）。两种类型的用户（**智能体**）：非战略用户，他们盲目跟随路由应用程序的推荐；以及战略用户，他们在有激励时会偏离推荐（例如，向繁忙的司机推荐了一条较长的路线）。用{% mathjax %}J_i(\pi_{\sigma}){% endmathjax %}表示中介学习到的策略{% mathjax %}\sigma{% endmathjax %}在第{% mathjax %}i{% endmathjax %}个**智能体**的价值。
- **案例一**：**无战略智能体**。在所有**智能体**都完全服从的理想情况下，将**多智能体模仿学习**(`MAIL`)问题视为**联合策略**上的**单智能体模仿学习**(`SAIL`)问题。
{% mathjax '{"conversion":{"em":14}}' %}
\underset{i\in [m]}{\max}J_i(\pi_{\sigma_E}) - J_i(\pi_{\sigma})
{% endmathjax %}
将**价值差距**降低到`0`，只要所有**智能体**盲目遵循建议，学会了一种策略，其表现至少与专家的策略相当。在路由应用程序中，意味着如果没有司机偏离先前的行为，所有司机的满意度至少与应用程序的先前版本相同。
- **案例二**，**战略智能体**，对于任何**多智能体模仿学习**(`MAIL`)问题，如果**智能体**具有代理权，需要考虑到智能体可能会偏离建议，如果从他们的主观角度来看这样做似乎是有利的。将智能体{% mathjax %}i{% endmathjax %}的偏差（即策略修改）类表示为{% mathjax %}\Phi_i{% endmathjax %}。定义由中介的策略引起的**遗憾**为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\Phi}(\sigma):= \underset{i\in[m]}{\max}\;\underset{\phi_i\in \Phi_i}{\max}(J_i(\pi_{\sigma,\phi_i})- J_i(\pi_{\sigma}))
{% endmathjax %}
其中，{% mathjax %}\phi_i{% endmathjax %}是智能体{% mathjax %}i{% endmathjax %}的**战略偏差**，而{% mathjax %}\pi_{\sigma},\phi_i{% endmathjax %}是由除{% mathjax %}i{% endmathjax %}之外的所有智能体遵循{% mathjax %}\sigma{% endmathjax %}的建议所诱导的**联合智能体策略**。直观上，**遗憾**捕捉了任何**智能体**在群体中偏离中介建议的最大动机。然后，比较**专家**和**学习者**策略之间的这一指标，以得出**遗憾差距**。
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\Phi}(\sigma) - \mathcal{R}_{\sigma}(\sigma_E) 
{% endmathjax %}
将**遗憾差距**降低到`0`（即实现**后悔等价**）意味着，即使**智能体**可以自由偏离，学习到的策略在群体中任意**智能体**的角度来看，至少与专家的策略同样优秀。但所有**智能体**在选择替代路线时的动机不会比在应用程序的历史版本下更大。较小的**价值差距**通常并不意味着较小的**遗憾差距**。考虑在所有服从的情况下**学习者**策略与偏离的第{% mathjax %}i{% endmathjax %}个**智能体**之间的性能差异({% mathjax %}J_i(\pi_{\sigma}){% endmathjax %})和({% mathjax %}J_i(\pi_{\sigma},\phi_i){% endmathjax %})。我们可以将这个量分解为以下内容：
{% mathjax '{"conversion":{"em":14}}' %}
J_i(\pi_{\sigma},\phi_i) - J_i(\pi_{\sigma}) = \underbrace{(J_i(\pi_{\sigma},\phi_i) - J_i(\pi_{\sigma_E},\phi_i))}_{(\text{I: value gap under }\phi_i)} + \underbrace{(J_i(\pi_{\sigma_E},\phi_i) - J_i(\pi_{\sigma_E}))}_{(\text{II: expert regret }\phi_i)} + \underbrace{(J_i(\pi_{\sigma_E}) - J_i(\pi_{\sigma}))}_{(\text{III: SAIL value gap})}
{% endmathjax %}
其中{% mathjax %}\pi_{\sigma_E},\phi_i{% endmathjax %}表示在专家建议和偏差{% mathjax %}\phi_i{% endmathjax %}下的**智能体**联合行为。第三项是标准的**单智能体价值差距**（即在假设没有**智能体**偏离的情况下的性能差异）。第二项是专家在偏差{% mathjax %}\phi_i{% endmathjax %}下的遗憾（即无法控制的量）。**遗憾差距**与**价值差距**目标之间的差异可以归结为第一项：{% mathjax %}J_i(\pi_{\sigma},\phi_i) - J_i(\pi_{\sigma_E},\phi_i){% endmathjax %}。请注意，由于偏差{% mathjax %}\phi_i{% endmathjax %}引起的状态分布变化，最小化第三项并不能保证第一项的结果。在**多智能体模仿学习**(`MAIL`)中，**后悔**是困难的，因为它需要知道**专家**在面对任意**智能体**偏差时会做什么。
- **马尔可夫博弈**中**多智能体模仿学习**(`MAIL`)的**遗憾差距**。与**单智能体模仿学习**中的标准目标——**价值差距**不同，**遗憾差距**捕捉了群体中的**智能体**可能会偏离中介建议的事实。从**价值差距**到**遗憾差距**的转变反映了**单智能体模仿学习**和**多智能体模仿学习**问题之间的根本区别。
- **遗憾差距**与**价值差距**之间的关系。在**完全奖励**和**偏差函数**类别的假设下，**后悔等价**意味着**价值等价**。然而，**价值等价**对**遗憾差距**几乎没有保证，从而确立了将**单智能体模仿学习**算法应用于**多智能体模仿学习**问题的基本局限性。
- 提供了一对在特定假设下**最小化遗憾差距**的高效算法。虽然在一般情况下实现**后悔等价**是困难的，因为它依赖于反事实专家建议，但推导出一对高效的方法来**最小化遗憾差距**，这些方法在不同假设下运行：`MALICE`（在覆盖假设下运行）和`BLADES`（需要访问可查询的专家）。证明这两个算法可以提供关于**遗憾差距**的{% mathjax %}\mathcal{O}(H){% endmathjax %}**界限**，其中{% mathjax %}H{% endmathjax %}是**时间范围**，与**单智能体模仿学习**中已知的最强结果相匹配。
{% asset_img ml_4.png "table 1: 各种多智能体模仿学习(MAIL)方法的遗憾差距" %}

**单智能体模仿学习**(`SAIL`)的理论大多集中在**单智能体**设置上。离线方法如**行为克隆**(`BC`)将模仿问题简化为纯粹的**监督学习**。忽视**专家**和**学习者**策略之间**状态分布**的**协变量偏移**可能导致累积错误。为此，**交互式模仿学习**方法如**逆强化学习**(`IRL`)允许**学习者**在训练过程中观察其行为的后果，从而防止累积错误。由于需要反复解决一个困难的**强化学习**问题，这些方法可能相当低效。替代方法包括**交互式查询专家**，以获取**学习者**诱导**状态分布**上的**动作标签**(`DAgger`)，或者在假设演示完全覆盖的情况下，使用**重要性加权**来纠正**协变量偏移**(`ALICE`)。在相同假设下运行，`BLADES`和`MALICE`算法可以看作是**遗憾差距**与**价值差距**的类比。

**多智能体模仿学习**(`MAIL`)的**遗憾差距**概念最早是在`Waugh`等人的工作中提出的，尽管他们的研究仅限于标准形式博弈(`NFG`)，而我们关注的是更一般的**马尔可夫博弈**(MG)。`Fu`等人简要考虑了**马尔可夫博弈**中的**遗憾差距**，但并未探讨其性质或提供有效最小化的算法。大多数经验性的**多智能体模仿学习**(`MAIL`)工作基于**价值差距**，而我们则退一步思考，首先要问的是**多智能体模仿学习**(`MAIL`)的正确目标是什么。

**逆游戏理论**(`Inverse Game Theory`)是**博弈论**的一个分支，主要关注如何根据观察到的**智能体**行为推导出**效用函数**或**策略**，而不是通过**示范学习**来协调行为。这种理论的核心在于理解和重建参与者在博弈中所采取的策略背后的动机和效用。特点：**目标导向**，**逆游戏理论**的主要目标是恢复一组**效用函数**，这些函数能够合理化观察到的**智能体**行为。这与传统的**博弈论**不同，后者通常关注如何在给定的**效用函数**下优化**策略**；**应用场景**，**逆游戏理论**常用于分析**多智能体系统**中的决策过程，尤其是在信息不完全或不对称的情况下。它可以帮助研究者理解复杂环境中各个**智能体**之间的互动；**方法论**，**逆游戏理论**通常涉及从**智能体**行为中推断出潜在的效用结构。研究者可能会使用统计和**机器学习**方法来分析数据，以识别和恢复这些**效用函数**；**与逆强化学习的关系**，**逆游戏理论**与**逆强化学习**有交集，后者同样关注从专家行为中推导出**奖励信号**。在**逆强化学习**中，目标是找到一个**奖励函数**，使得专家的行为是最优的，而在**逆游戏理论**中，则是找到合理化**智能体**行为的**效用函数**。

**效用函数**(`Utility Function`)是**博弈论**和**经济学**中的一个重要概念，用于量化参与者在决策过程中所获得的效用或满意度。定义：**效用函数**通常表示为{% mathjax %}U:X\rightarrow \mathbb{R}{% endmathjax %}，其中{% mathjax %}X{% endmathjax %}是消费品或策略的集合，{% mathjax %}\mathbb{R}{% endmathjax %}是实数集。该函数将每个可能的结果映射到一个实数值，表示该结果所带来的**效用**。在**博弈论**中，**效用函数**用于描述每个参与者在不同策略组合下的收益。例如，在一个**策略型博弈**中，每个参与者的效用不仅取决于自己的策略选择，还取决于其他参与者的策略。因此，参与者会根据自己的**效用函数**来选择最优策略，以最大化自身的收益。效用函数的类型分为2种：**直接效用函数**，只依赖于消费束（商品数量向量），例如{% mathjax %}U(X){% endmathjax %}；**间接效用函数**，依赖于商品价格和消费者预算约束，例如{% mathjax %}V(P,m){% endmathjax %}。**效用函数**是理解经济行为和决策过程的重要工具。它不仅帮助研究者分析消费者行为，还在**博弈论**中为参与者提供了评估和选择策略的方法。通过建立适当的**效用函数**，可以更好地理解和预测个体在复杂环境中的行为。

首先{% mathjax %}\Delta(X){% endmathjax %}表示集合{% mathjax %}X{% endmathjax %}上的**概率分布空间**。用{% mathjax %}\mathcal{L}{% endmathjax %}表示每个算法优化的**损失函数**，这可以被视为对**总变差距离**(`Total Variation Distance, TV`)的一个**凸上界**。当**损失函数**恰好是**总变差距离**时，则用{% mathjax %}\mathcal{L}_{TV}{% endmathjax %}来表示。**总变差距离**(`TV`)是一种用于量化两个**概率分布**之间差异的统计测量。它被定义为两个不同分布在同一事件上分配的概率之间的最大差异。

**马尔可夫博弈**(`MG`)，用{% mathjax %}MG(H,\mathcal{S},\mathcal{A},\mathcal{T},\{r_i\}_{i=1}^m,p_0){% endmathjax %}来表示一个包含{% mathjax %}m{% endmathjax %}个**智能体**的**马尔可夫博弈**(`MG`)。这里{% mathjax %}H{% endmathjax %}是时间范围，{% mathjax %}\mathcal{S}{% endmathjax %}是**状态空间**，{% mathjax %}\mathcal{A} = \mathcal{A_1}\times\ldots\times\mathcal{A_m}{% endmathjax %}是所有**智能体**的**联合动作空间**。用{% mathjax %}\mathcal{T}:\mathcal{S}\times \mathcal{A}\rightarrow \Delta(\mathcal{S}){% endmathjax %}来表示**转移函数**。此外，**智能体**{% mathjax %}i\in [m]{% endmathjax %}的**奖励（效用）函数**表示为{% mathjax %}r_i:\mathcal{S}\times\mathcal{A}\rightarrow [-1,1]{% endmathjax %}。最后，用{% mathjax %}p_0{% endmathjax %}表示初始**状态分布**，从中抽样初始状态{% mathjax %}s_0\sim p_0{% endmathjax %}。

**学习协调**，不考虑在**马尔可夫博弈**中学习单个**智能体**策略的问题，而是从**中介**的角度出发，给每个**智能体**提供建议，以帮助他们协调行动（例如，一个智能手机地图应用为一组用户提供方向）。在每个时间步，**中介**会给每个智能体{% mathjax %}i{% endmathjax %}在当前状态{% mathjax %}s{% endmathjax %}下一个私有的动作建议{% mathjax %}a_i{% endmathjax %}。关键是，没有**智能体**会观察到中介提供给其他**智能体**的建议。可以将中介表示为一个**马尔可夫联合策略**{% mathjax %}\sigma\in\Sigma{% endmathjax %}，其中{% mathjax %}\sigma : \mathcal{S}\rightarrow \Delta (\mathcal{A}){% endmathjax %}。用{% mathjax %}\sigma(\vec{a}|s){% endmathjax %}表示在状态{% mathjax %}s{% endmathjax %}下推荐联合动作{% mathjax %}a{% endmathjax %}的概率。用{% mathjax %}\pi:\mathcal{S}\rightarrow \Delta(\mathcal{A}){% endmathjax %}表示**智能体**根据**中介策略**所采取的**联合策略**。当**智能体**完全遵循中介的建议时，**联合策略**表示为{% mathjax %}\pi_{\sigma}{% endmathjax %}。

轨迹{% mathjax %}\xi\sim \pi = \{s_h,\vec{a}_h\}_{h=1,\ldots,H}{% endmathjax %}指的是从{% mathjax %}s_0\sim p_0{% endmathjax %}开始生成的一系列**状态-动作对**，通过反复从策略{% mathjax %} \pi{% endmathjax %}和**转移函数**{% mathjax %}\mathcal{T}{% endmathjax %}中抽样联合动作{% mathjax %}\vec{a}_h{% endmathjax %}和下一个状态{% mathjax %}s_{h+1}{% endmathjax %}，进行{% mathjax %}H-1{% endmathjax %}次时间步。设{% mathjax %}d_h^{\pi}{% endmathjax %}表示在时间步长{% mathjax %}h{% endmathjax %}下按照策略{% mathjax %}\pi{% endmathjax %}的状态访问分布，并且让 {% mathjax %}d^{\pi} = \frac{1}{H}\sum_{h=1}^H d_h^{\pi}{% endmathjax %}为平均状态分布。让{% mathjax %}p_h^{\pi}(s_h,\vec{a}_h){% endmathjax %}表示占用测度——即在时间步长{% mathjax %}h{% endmathjax %}到达状态{% mathjax %}s{% endmathjax %}并采取动作{% mathjax %}\vec{a}{% endmathjax %}的概率。根据定义，{% mathjax %}\forall_h,\sum_{s,\vec{a}}p_h^{\pi}(s,\vec{a}) = 1{% endmathjax %}。让{% mathjax %}p^{\pi}(s,\vec{a}) = \frac{1}{H}\sum_{h=1}^H p_h^{\pi}(,\vec{a}){% endmathjax %}为**平均占用测度**。

用{% mathjax %}V_{i,h}^{\pi}{% endmathjax %}表示**智能体**{% mathjax %}i{% endmathjax %}在此策略下从时间步长{% mathjax %}h{% endmathjax %}开始的**期望累计奖励**，即{% mathjax %}V_{i,h}^{\pi}(s) = \mathbb{E}_{\xi\sim\pi}[\sum_{t=h}^H r_i(s_t,\vec{a}_t)|s_h = s]{% endmathjax %}，将智能体{% mathjax %}i{% endmathjax %}的`Q`**值函数**定义为{% mathjax %}V_{i,h}^{\pi}(s,\vec{a}) = \mathbb{E}_{\xi\sim\pi}[\sum_{t=h}^H r_i(s_t,\vec{a}_t)|s_h = s,\vec{a}_h = \vec{a}]{% endmathjax %}。定义**智能体**{% mathjax %}i{% endmathjax %}的优势为其在选定动作上的`Q`-值与状态的`V`-值之间的差，即{% mathjax %}A_{i,h}^{\pi}(s,\vec{a}) = Q_{i,h}^{\pi}(s,\vec{a}) - V_{i,h}^{\pi}(s){% endmathjax %}，还定义从**智能体**{% mathjax %}i{% endmathjax %}的角度来看策略{% mathjax %}\pi{% endmathjax %}的表现为{% mathjax %}J_{i}(\pi) = \mathbb{E}_{s_0\sim p_0}[\mathbb{E}_{\xi\sim\pi}[\sum_{t=1}^H r_i(s_t,\vec{a}_t)|s = s_0]]{% endmathjax %}，注意，表现是**占用测度**与**智能体奖励函数**之间的内积，即{% mathjax %}J_i(\pi) = H\sum_{s,\vec{a}}p^{\pi}(s,\vec{a})r_i(s,\vec{a}){% endmathjax %}。

通过引入**相关均衡**(`CE`)的概念。首先，将第{% mathjax %}i{% endmathjax %}个**智能体**的策略偏差定义为映射{% mathjax %}\phi_i : \mathcal{S}\times \mathcal{A_i}\rightarrow \mathcal{A}_i{% endmathjax %}。直观上，**策略偏差**捕捉了**智能体**如何响应当前世界状态和中介的建议——他们可以选择服从（此时{% mathjax %}\phi_i(s,a) = a{% endmathjax %}）或偏离（此时{% mathjax %}\phi_i(a,a) \neq  a{% endmathjax %}）。令{% mathjax %}\phi_i{% endmathjax %}为智能体{% mathjax %}i{% endmathjax %}的偏差集合，它是所有可能偏差的子集。用{% mathjax %}\Phi:=\{\Phi_i\}_{i=1}^m{% endmathjax %}来表示所有**智能体**的偏差。假设对于所有的{% mathjax %}i{% endmathjax %}，恒等映射{% mathjax %}\phi_i(s,a) = a{% endmathjax %}属于{% mathjax %}\Phi_i{% endmathjax %}。用{% mathjax %}\pi_{\sigma,\phi_i}{% endmathjax %}来表示由中介策略{% mathjax %}\sigma{% endmathjax %}被偏差{% mathjax %}\phi_i{% endmathjax %}覆盖所诱导的**联合智能体策略**，即{% mathjax %}(\phi_i\circ\pi_{\sigma,i})\odot \pi_{\sigma,-i}{% endmathjax %}。

设{% mathjax %}\sigma\in \Sigma{% endmathjax %}为**马尔可夫博弈**中的**中介策略**，并且{% mathjax %}\Phi_i,i\in [m]{% endmathjax %}为每个**智能体**的偏差类。然后，定义中介策略{% mathjax %}\sigma{% endmathjax %}的后悔（遗憾）为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\Phi}(\sigma):= \underset{i\in [m]}{\max}\;\underset{\phi_i\in \Phi_i}{\max}(J_i(\pi_{\sigma,\phi_i}) - J_i(\pi_{\sigma}))
{% endmathjax %}
一个**中介策略**{% mathjax %}\sigma{% endmathjax %}诱导一个{% mathjax %}\epsilon{% endmathjax %}-**近似相关均衡**(`CE`)，如果：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\Phi}(\sigma)\leq \epsilon
{% endmathjax %}
后悔（遗憾）捕捉了任何**智能体**通过偏离中介建议所能获得的最大效用。**相关均衡**(`CE`)是一个诱导的**联合策略**，在这个策略下，没有**智能体**有很大的动机去偏离。

**遗憾差距与价值差距之间的关系**：
- **价值差距**(`value gap`)：定义专家策略{% mathjax %}\sigma_E{% endmathjax %}与学习者策略{% mathjax %}\sigma\in \Sigma{% endmathjax %}之间的**价值差距**为：
{% mathjax '{"conversion":{"em":14}}' %}
\underset{i\in [m]}{\max}(J_i(\pi_{\sigma_E}) - J_i(\pi_{\sigma}))
{% endmathjax %}
- **遗憾差距**(`regret gap`)：定义专家策略{% mathjax %}\sigma_E{% endmathjax %}与学习者策略{% mathjax %}\sigma\in \Sigma{% endmathjax %}之间的**遗憾差距**为：
{% mathjax '{"conversion":{"em":14}}' %}
\mathcal{R}_{\Phi}(\sigma) - \mathcal{R}_{\Phi}(\sigma_E) = \underset{i\in [m]}{\max}\;\underset{\phi_i\in \Phi_i}{\max}(J_i(\pi_{\sigma,\phi_i}) - J_i(\pi_{\sigma})) - \underset{k\in [m]}{\max}\;\underset{\phi_k\in \Phi_k}{\max}(J_k(\pi_{\sigma_E,\phi_k}) - J_k(\pi_{\sigma_E}))
{% endmathjax %}
{% asset_img ml_5.png "遗憾等价意味着价值等价，但反之不成立" %}
当学习者的策略在**价值/遗憾**等价时，则**价值/遗憾差距**为`0`。如上图所示，**多智能体模仿学习**(`MAIL`)中**价值**和**遗憾差距**之间的关系，用{% mathjax %}J_i(\pi_{\sigma},f){% endmathjax %}和{% mathjax %}\mathcal{R}_{\Phi}(\sigma,f){% endmathjax %}来表示策略{% mathjax %}\sigma{% endmathjax %}在**奖励函数**{% mathjax %}f{% endmathjax %}下的价值/遗憾。

如果**奖励函数类**和**偏差类**都是完整的，那么**遗憾等价性**等同于**价值等价性**。当**奖励函数类**是完整的时，则{% mathjax %}\mathcal{F} = \{\mathcal{S}\times \mathcal{A}\rightarrow [-1,1]\}{% endmathjax %}（即**所有状态-动作指标的凸组合**）；而当**偏差类**是完整的时，则对于每个智能体{% mathjax %}i{% endmathjax %}，有{% mathjax %}\Phi_i = \{\mathcal{S}\times \mathcal{A}\rightarrow \mathcal{A}_i\}{% endmathjax %}（即**所有可能的偏差**）。

**定理一**：如果**奖励函数类**{% mathjax %}\mathcal{F}{% endmathjax %}和**偏差类**{% mathjax %}\Phi{% endmathjax %}是完整的，并且满足**遗憾等价性**（即{% mathjax %}\text{sup}_{d\in\mathcal{F}}(\mathcal{R}_{\Phi}(\sigma,f)-\mathcal{R}_{\Phi}(\sigma_E,f)) = 0{% endmathjax %}），那么**价值等价性**也得以满足：{% mathjax %}\text{sup}_{d\in\mathcal{F}}\max_{i\in [m]}(J_i(\pi_{\sigma_E},f) - J_i(\pi_{\sigma},f)) = 0{% endmathjax %}。

**定理二**：存在一个**马尔可夫博弈**(`MG`)、一个专家策略{% mathjax %}\sigma_E{% endmathjax %}和一个训练过的策略{% mathjax %}\sigma{% endmathjax %}，使得**真实奖励函数**{% mathjax %}r{% endmathjax %}满足**遗憾等价性**，即{% mathjax %}\mathcal{R}_{\Phi}(\sigma,r) - \mathcal{R}_{\Phi}(\sigma_E,r) = 0{% endmathjax %}，而**价值差距**为{% mathjax %}\max_{i\in [m]}(J_i(\pi_{\sigma_E},r) - J_i(\pi_{\sigma},r)) \neq 0{% endmathjax %}。

综合这些结果，当**奖励函数**/**偏差类**足够表达时，**遗憾等价性**比**价值等价性**更强。**价值等价性**并不代表着**低遗憾差距**！在最坏的情况下，**价值等价性**无法提供任何有意义的**遗憾差距**保证。这揭示了`SAIL`与`MAIL`之间的一个关键区别。

**定理三**：存在一个**马尔可夫博弈**(`MG`)、一个专家策略{% mathjax %}\sigma_E{% endmathjax %}和一个学习者策略{% mathjax %}\sigma{% endmathjax %}，使得即使策略{% mathjax %}\pi_{\sigma}{% endmathjax %}的**占用测度**与{% mathjax %}\pi_{\sigma_E}{% endmathjax %}完全匹配，即对所有状态和动作组合有{% mathjax %}\forall(s,\vec{a}),p^{\pi_{\sigma}}(s,\vec{a}) = p^{\pi_{\sigma_E}}(s,\vec{a}){% endmathjax %}（即在所有奖励下具有**价值等价性**），**遗憾差距**却满足{% mathjax %}\mathcal{R}_{\Phi}(\sigma) - \mathcal{R}_{\Phi}(\sigma_E) \geq \Omega(H){% endmathjax %}。
如下图所示，**专家**和**学习者策略**仅访问下路径中的状态{% mathjax %}s_2,s_4,\ldots,s_{2H - 2}{% endmathjax %}。训练过的策略通过在访问的状态{% mathjax %}s_2,s_4,\ldots,s_{2H - 2}{% endmathjax %}中采取相同的动作，完美匹配了专家的**占用测度**。然而，专家演示缺乏对状态{% mathjax %}s_1{% endmathjax %}的覆盖，因为通过执行{% mathjax %}\pi_E{% endmathjax %}无法到达该状态。当**智能体**`1`偏离原始策略时，这一遗漏变得至关重要，使得状态{% mathjax %}s_1{% endmathjax %}在高概率下无法到达。因此，训练过的策略在状态{% mathjax %}s_1{% endmathjax %}的表现可能很差，而专家在**真实奖励函数**下却能表现出色。这一例子突显了**价值等价性**与**遗憾等价性**之间的关键区别：前者仅依赖于策略实际访问的状态，而后者则依赖于学习者在未访问状态下对**智能体**偏离所做出的**反事实推荐**。

正如定理三所示，即使学习者能够从专家演示中获得关于均衡路径的**无限样本**，学习者仍可能对专家在未被访问（但可由偏离的**智能体**的**联合策略**到达）的状态下的行为一无所知。因此，从信息理论的角度来看，学习者无法在不知道专家在这些状态下会如何行动的情况下**最小化遗憾差距**。这展示了**最小化遗憾差距**的根本困难，因此，在`MAIL`中，**遗憾**是“困难”的。因此，需要一种新的`MAIL`算法范式来**最小化遗憾差距**。
{% asset_img ml_6.png "一个捕捉“遗憾是困难的”原因的马尔可夫博弈示例" %}

在这里，{% mathjax %}\sigma_E(a_1 a_1|s_0) = 1{% endmathjax %}。注意，当所有智能体都遵循{% mathjax %}\sigma_E{% endmathjax %}时，且状态{% mathjax %}s_1{% endmathjax %}是未被访问的，但在偏离策略{% mathjax %}\phi_1{% endmathjax %}下，其访问概率为`1`（{% mathjax %}\phi_1(s_0,a_1) = \phi_1(s_1,a_1) = a_2{% endmathjax %}）。这意味着，除非知道专家{% mathjax %}\sigma_E{% endmathjax %}在状态{% mathjax %}s_1{% endmathjax %}下会如何进行**反事实推荐**，否则无法**最小化遗憾差距**。

**定理四**：如果专家策略{% mathjax %}\sigma_E{% endmathjax %}诱导了一个{% mathjax %}\delta{% endmathjax %}-近似的**相关均衡**(`CE`)，并且学习者策略{% mathjax %}\sigma{% endmathjax %}满足{% mathjax %}\mathcal{R}_{\Phi}(\sigma) - \mathcal{R}_{\Phi}(\sigma_E)\leq \delta_2{% endmathjax %}，那么{% mathjax %}\sigma{% endmathjax %}诱导一个{% mathjax %}\delta_1 + \delta_2{% endmathjax %}-近似的**相关均衡**(`CE`)。

然后，通过与定理三结合，可以得出**低价值差距**并不意味着学习者正在执行**相关均衡**。

**推论**：存在一个**马尔可夫博弈**(`MG`)、一个专家策略{% mathjax %}\sigam_E{% endmathjax %}和一个学习者策略{% mathjax %}\sigam{% endmathjax %}，使得{% mathjax %}\sigma_E{% endmathjax %}诱导了一个{% mathjax %}\delta_1{% endmathjax %}-近似的**相关均衡**(`CE`)，并且{% mathjax %}\sigam{% endmathjax %}满足{% mathjax %}\max_{i\in [m]}(J_i(\pi_{\sigma_E}) - J_i(\pi_{\sigma})) = \dalta_2{% endmathjax %}，则{% mathjax %}\sigma{% endmathjax %}诱导一个{% mathjax %}\Omega(H){% endmathjax %}-近似的**相关均衡**(`CE`)。综合这些结果表明，如果希望在智能体中诱导出一个**相关均衡**(`CE`)，那么**遗憾差距**是一个更合适的目标。

尽管已经表明**价值差距**在某种意义上是一个“**较弱**”的目标，但在许多现实场景中，**智能体**可能是**非战略性**的。在这些场景中，**最小化价值差距**可以是一个合理的学习目标。**单智能体逆向强化学习**算法的**多智能体**推广可以有效地**最小化价值差距**——因此，在**多智能体**学习中，价值是“简单的”。**行为克隆**(`BC`)和**逆向强化学习**是两个旨在**最小化价值差距**的**单智能体模仿学习算法**。通过在**联合策略**上运行这些算法，我们可以将**行为克隆**(`BC`)和**逆强化学习** 应用于**多智能体**设置，称之为**联合行为克隆**(`J-BC`)和**联合逆向强化学习**(`J-IRL`)。这样做会导致与**单智能体**设置相同的**价值差距界限**。

**定理五**：如果`J-BC`返回一个策略{% mathjax %}\sigma{% endmathjax %}，使得{% mathjax %}\mathbb{E}_{s\sim d_{\pi_{\sigma_E}}}[\mathcal{L}(\sigma_E(s),\sigma(s))]\leq \epsilon{% endmathjax %}，那么**价值差距**{% mathjax %}\max_{i\in [m]}(J_i(\pi_{\sigma_E}) - J_i(\pi_{\sigma})) \leq \mathcal{O}(\epsilon H^2){% endmathjax %}。

**定理六**：如果`J-IRL`输出一个策略{% mathjax %}\sigma{% endmathjax %}，其此刻的匹配误差满足：
{% mathjax '{"conversion":{"em":14}}' %}
\underset{f\in \mathcal{F}}{\text{sup}}\;\mathbb{E}_{\pi_{\sigma_E}}\bigg[\sum\limits_{h=1}^H f(s_h,\vec{a}_h) \bigg] - \mathbb{E}_{\pi_{\sigma}}\bigg[\sum\limits_{h=1}^H f(s_h,\vec{a}_h) \bigg] \leq \epsilon H
{% endmathjax %}
那么**价值差距**为：{% mathjax %}\max_{i\in [m]}(J_i(\pi_{\sigma_E}) - J_i(\pi_{\sigma})) \leq \mathcal{O}(\epsilon H){% endmathjax %}。满足上述任一定理的条件可以通过归约到**无遗憾在线学习**以高效地实现。
