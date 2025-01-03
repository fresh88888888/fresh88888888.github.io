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
\underset{\pi_{theta}}{\max}\;\sum\limits_{i=1}^L \mathbb{E}_{x,y^\sim \mathcal{D},\hat{y}_i\sim \pi_{\theta}(\cdot|[x,\hat{y}_{1:i-1},p_{1:i-1}])}[\mathbb{I}(\hat{y}_i == y^*)]
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