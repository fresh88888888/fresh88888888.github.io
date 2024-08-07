---
title: 基于人类反馈的强化学习(RLHF) — 推导（深度学习）
date: 2024-06-28 12:20:11
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

基于人类反馈的强化学习(`Reinforcement Learning Human Feedback, RLHF`)是一种结合强化学习技术和人类反馈来训练人工智能(`AI`)模型的方法。`RLHF`是一种机器学习方法，通过人类反馈来优化`AI`模型的行为，使其更符合人类的期望和偏好。这种方法特别适用于自然语言处理(`NLP`)任务，如对话系统、文本生成和摘要生成等。`RLHF`的训练过程通常分为三个主要阶段：
<!-- more -->
- **预训练语言模型**：首先，使用传统的预训练目标对语言模型进行预训练。这一步通常使用大量的文本数据来训练模型，使其具备基本的语言理解和生成能力。例如，`OpenAI`的`InstructGPT`就是在一个较小版本的`GPT-3`模型上进行预训练的。
- **训练奖励模型**：接下来，收集人类反馈数据并训练一个奖励模型。奖励模型的作用是预测人类对模型生成文本的评分。具体步骤如下：人类评估者对模型生成的文本进行评分或排序；使用这些评分数据训练一个监督学习模型，使其能够预测给定文本的评分。
- **使用强化学习微调语言模型**：最后，使用奖励模型对语言模型进行微调。通过强化学习算法（如近端策略优化`PPO`），模型根据奖励模型的评分来优化其生成的文本，使其更符合人类偏好。

在实践中为单词序列分配概率，例如：`“Shanghai is a city in”`，如下图所示：
{% asset_img d_1.png %}

这里简化为：一个`token`是一个单词，在大多数语言模型中实际上并不是这样的，但用于解释它很有帮助。给定特定提示的情况下，下一个`token`是`china`、`Beijing`、`Cat`、`Pizza`的概率是`85%、10%、2.5%、...`，语言模型给了我们这些概率。我们如何使用语言模型来生成文本呢？首先我们会给出一个提示，如：`“Where is Shanghai?”`，我们把它提交给语言模型，语言模型会给出下一个单词或`token`的概率列表，假设我们选择概率分数最高的`token`，假设它是`“Shanghai”`，然后我们将其(`token`)放回到提示中：`“Where is Shanghai? Shanghai”`，再次交给语言模型，然后语言模型再次给出一个单词或`token`的概率列表，我们选择相关性最重要的一个。然后将其放回到提示后边，然后再次提交给语言模型等，直到完成了句子标记的结尾。这种情况下，这是一个特殊的`token`。
{% asset_img d_2.png %}

#### 强化学习

强化学习(`Reinforcement Learning, RL`)是机器学习的一个重要分支,主要关注如何让智能体(`agent`)在与环境的交互中学习最优策略。智能体通过试错的方式,在环境中采取行动,获得奖励或惩罚,从而学习如何最大化长期累积奖励。举个简单的例子：
|`example_1`|`example_2`|
|:---|:---|
|`Agent`：猫|`Agent`：语言模型|
|状态：猫在网格中的位置`(x,y)`|状态：提示（输入的`tokens`）|
|行为：在每个位置，猫可以移动到`4`个方向连接的单元格之一，如果移动无效，则单元格将不会移动并保持在同一位置。每次猫移动时，都会产生新的状态和奖励。|行为：哪个`token`被选为下一个 `token`|
|奖励模式：<br> 1.移至另一个空单元格将导致奖励`0`。<br> 2.移向扫帚将导致奖励`-1`。<br> 3.移向浴缸将导致奖励`-10`，猫会晕倒（剧集结束）。猫会再次重生在初始位置。<br> 4.移向肉将导致奖励`+100`|奖励模式：语言模型应该因产生“好的反应”而获得奖励，而不应该因产生“坏的反应”而获得任何奖励。|
|策略：策略规定代理如何在其所处的状态下选择要执行的操作：{% mathjax %}a_t\sim \pi(\cdot | s_t){% endmathjax %}|策略：对于语言模型来说，策略就是语言模型本身！因为它根据代理的当前状态模拟动作空间的概率：{% mathjax %}a_t\sim \pi(\cdot | s_t){% endmathjax %}|
|{% asset_img d_3.png %}|{% asset_img d_4.png %}|

`RL`中的目标是选择一种策略，当代理按照该策略采取行动时，该策略可以最大化预期回报。

#### 奖励模型架构

当我们将一串`token`作为语言模型（通常是`Transformer`模型）的输入时，它会生成一个隐藏状态列表，每个隐藏状态对应一个输入`token`，这是一个“捕获”其之前所有`token`信息的嵌入。然后，隐藏状态通过线性层转换为逻辑，然后使用`softmax`函数转换为概率。要生成响应的奖励，我们只需使用响应的最后一个`token`的隐藏状态，将其发送到线性层（只有一个输出特征），并将其用作与输入相关的奖励值。
{% asset_img r_1.png %}

#### 奖励模型损失

为了使用强化学习(`RL`)优化语言模型的行为，我们需要一个评分模型，该模型为语言模型生成的每个响应提供数值。现在我们有了一个数据集，可以根据查询（提示）定义我们喜欢哪个答案，我们可以构建一个神经网络，为每个响应提供数值分数。
{% asset_img r_2.png %}

{% mathjax '{"conversion":{"em":14}}' %}
Loss = -\log\sigma(r(x,y_w) - r(x,y_l))
{% endmathjax %}
在这里我们分析一下这个损失函数：有两种可能性：
- 当{% mathjax %}r(x,y_w) > r(x,y_l){% endmathjax %}时，`Sigmoid`返回一个大于`0.5`的值，损失将返回一个很小的负值（顺序正确的情况下损失值将会很小）。
- 当{% mathjax %}r(x,y_w) < r(x,y_l){% endmathjax %}时，`Sigmoid`返回一个小于`0.5`的值，损失将返回一个很大的负值（顺序错误的情况下损失值将会很大）。

这种损失将导致模型对“好”的答案给予高奖励，对“坏”的答案给予低奖励，因为这是模型最小化损失的唯一方法。在`HuggingFace`中，我们可以使用`RewardTrainer`和`AutoModelForSequenceClassification`来训练自定义奖励模型，`AutoModelForSequenceClassification`是一个顶部带有特殊线性层的语言模型。我们只需要求语言模型输出与最后一个`token`相对应的隐藏状态，将其发送到计算奖励的线性层，然后利用上边的损失函数来训练语言模型。
{% asset_img r_3.png %}

如上所述，强化学习的目标是选择一种策略，当代理按照该策略行事时，该策略可以最大化预期回报。更正式地方式为：
{% mathjax '{"conversion":{"em":14}}' %}
\pi^* = \underset{\pi}{\text{arg max}}J(\pi)
{% endmathjax %}
策略的预期回报是所有可能**轨迹**的预期回报。
{% mathjax '{"conversion":{"em":14}}' %}
J(\pi)\int_{\tau} P(\tau|\pi)R{\tau} = \underset{\tau\sim\pi}{E} [R(\tau)]
{% endmathjax %}
轨迹是从初始状态开始的一连串动作、状态：
{% mathjax '{"conversion":{"em":14}}' %}
\tau = (s_0,a_0,s_1,a_1,\ldots)
{% endmathjax %}
我们将下一个状态建模为随机的（假设猫喝醉了，并不总是能正确移动）：
{% mathjax '{"conversion":{"em":14}}' %}
s_{t+1}\sim P(\cdot|s_t,a_t)
{% endmathjax %}
因此，我们可以定义轨迹的概率如下：
{% mathjax '{"conversion":{"em":14}}' %}
P(\tau|\pi) = \rho_0 (s_0)\prod_{t=0}^{T-1} P(s_{t+1}|s_t,a_t)\pi(a_t|s_t)
{% endmathjax %}
我们将始终提供折扣奖励（我们更喜欢即时奖励而不是未来奖励）：
{% mathjax '{"conversion":{"em":14}}' %}
R(\tau) \sum_{t=0}^{\infty}\mathcal{r}^t r_t
{% endmathjax %}

#### 语言模型中的轨迹

在处理语言模型时，我们希望对语言模型进行微调，以便它以最大化的获得奖励的方式选择下一个`token`。
{% mathjax '{"conversion":{"em":14}}' %}
\pi^* = \underset{\pi}{\text{arg max}}J(\pi)
{% endmathjax %}
语言模型的轨迹是什么？它是一串提示词（状态）及其下一个`token`（动作）。
{% mathjax '{"conversion":{"em":14}}' %}
\tau = (s_0,a_0,s_1,a_1,\ldots)
{% endmathjax %}
{% asset_img r_4.png %}

我们可以看到，当使用语言模型来生成问题的答案（或者一般根据提示生成文本）时，我们可以看到一系列状态和动作，它们定义了一条轨迹。

#### 策略梯度优化

假设我们有一个策略{% mathjax %}\pi_{\theta}{% endmathjax %}，由参数{% mathjax %}\theta{% endmathjax %}参数化。我们希望更改策略的参数，以便在使用该策略时最大化预期回报。也就是说，我们希望最大化以下内容：
{% mathjax '{"conversion":{"em":14}}' %}
J(\pi_{\theta}) = \underset{\tau\sim\pi_{\theta}}{E} [R(\tau)]
{% endmathjax %}
当我们拥有深度神经网络时，我们的目标是迭代地改变网络的参数，从而最小化损失函数：这是随机梯度下降的典型用例。在我们的例子中，我们想要最大化一个函数，因此我们可以使用随机梯度上升：
{% mathjax '{"conversion":{"em":14}}' %}
\theta_{k+1} = \theta_k + \alpha\nabla_{\theta}J(\pi_{\theta})|_{\theta_k}
{% endmathjax %}
策略的梯度称为**策略梯度**，用这种方法优化策略的算法称为**策略梯度算法**。问题在于，为了计算梯度，我们需要对所有可能的轨迹进行评估，除非我们的状态空间非常小，否则这在计算上是难以解决的。让我们尝试推导一个可以在合理时间内计算的策略梯度公式。
{% asset_img r_5.png %}

这是一个期望，意味着我们可以通过收集一组轨迹`D`，用样本均值来近似它。
{% mathjax '{"conversion":{"em":14}}' %}
\hat{g} = \frac{1}{|\mathcal{D}|}\sum_{\tau\in \mathcal{D}}\sum{t=0}^T \nabla_{\theta}\log \pi_{\theta} (a_t|s_t) R(\tau)
{% endmathjax %}
我们首先找到预期奖励关于参数{% mathjax %}\theta{% endmathjax %}的梯度公式，然后使用样本均值对其进行近似。在实践中，这被称为`REINFORCE`算法：
- 创建一个定义了策略的神经网络（输入`Agent`的当前状态并输出动作空间上的概率）。
- 使用网络对轨迹及其相应的奖励进行采样（例如，我们可以运行每个轨迹`100`步或直到猫晕倒）。
- 使用样本计算梯度。
- 运行随机梯度上升以更新策略/网络的参数。
- 返回到第2步。

{% mathjax '{"conversion":{"em":14}}' %}
\hat{g} = \frac{1}{|\mathcal{D}|}\sum_{\tau\in \mathcal{D}}\sum{t=0}^T \nabla_{\theta}\log \pi_{\theta} (a_t|s_t) R(\tau)
{% endmathjax %}

{% mathjax '{"conversion":{"em":14}}' %}
\theta_{k+1} = \theta_k + \alpha\nabla_{\theta}J(\pi_{\theta})|_{\theta_k}
{% endmathjax %}
还记得我们为奖励模型构建的偏好数据集吗？我们可以使用数据集中的问题并要求我们的模型生成答案。然后，我们计算生成的答案的奖励并根据策略的近似梯度训练模型，如`REINFORCE`算法中所描述。由于文本生成过程会产生一系列状态（提示词）和动作（下一个`token`），因此我们获得了一组轨迹！

#### 计算对数概率

我们如何计算语言模型轨迹的对数概率？假设我们的语言模型针对给定的问题生成了以下答案。让我们看看如何利用生成的答案来计算单个（状态，动作）对的对数概率。
{% asset_img r_6.png %}

#### 计算每条轨迹的奖励

我们也可以为所有”(状态，动作)对“生成奖励！这是因为奖励模型通常是一个语言模型，顶部有一个线性层。
{% asset_img r_7.png %}

#### 梯度策略优化问题

第一个问题：”梯度估计的高方差问题“。虽然我们的梯度近似是无偏的，这保证了长期的准确性，但高方差问题可能会导致短期的不稳定性和收敛困难。解决这个问题对于提高学习算法的效率和稳定性至关重要。梯度估计器通过将每个动作的对数概率梯度与整个轨迹的累积奖励相乘，来近似策略梯度。虽然它是无偏的，但可能存在高方差问题，这促使了许多后续的改进方法的发展。
{% asset_img r_8.png %}

可以证明，从奖励中减去基线，仍然会得到梯度的无偏估计量。使用值函数{% mathjax %}V^{\pi}(s){% endmathjax %}作为基线是一种有效的方差减少技术，它通过提供每个状态的预期回报作为参考点，帮助算法更好地评估动作的相对优劣。这种方法结合了策略梯度和值函数近似，形成了现代强化学习中强大的`Actor-Critic`框架。{% mathjax %}V^{\pi}(s){% endmathjax %}表示在状态{% mathjax %}s{% endmathjax %}下，按照策略{% mathjax %}\pi{% endmathjax %}行动，预期能获得的未来累积奖励。我们在语言模型（策略{% mathjax %}\pi_{\theta}{% endmathjax %}）之上添加了一个额外的线性层，用于估计特定时间步骤的状态值。请注意，语言模型（策略{% mathjax %}\pi_{\theta}{% endmathjax %}）已经有一个线性层，用于将隐藏状态转换为逻辑。下面显示的是添加到模型中的另一个层。

通过引入`Q`函数和`V`函数,我们可以构建更精确的梯度估计器,从而减少方差,提高学习效率和稳定性。这是现代强化学习算法的核心技术之一。`Q`函数(`Q-value function`): {% mathjax %}Q^{\pi}(s,a){% endmathjax %}表示在状态{% mathjax %}s{% endmathjax %}下采取动作{% mathjax %}a{% endmathjax %},然后遵循策略{% mathjax %}\pi{% endmathjax %}的预期累积奖励。`V`函数(`Value function`): {% mathjax %}V^{\pi}(s){% endmathjax %}表示在状态{% mathjax %}s{% endmathjax %}下遵循策略{% mathjax %}\pi{% endmathjax %}的预期累积奖励。
{% asset_img r_9.png %}

还记得我们说过也可以使用基线来减少方差吗？那么让我们使用另一个函数（称为`V`函数）作为基线，进一步减少方差。
{% asset_img r_10.png %}

**优势函数**通过提供一个相对的评价标准,帮助算法更好地理解动作的相对价值,从而减少梯度估计的方差,使强化学习过程更加稳定和高效。它是现代强化学习算法中减少方差的核心技术之一。{% mathjax %}A(s,a) = Q(s,a) - V(s){% endmathjax %}，其中{% mathjax %}Q(s,a){% endmathjax %}是动作值函数,{% mathjax %}V(s){% endmathjax %}是状态值函数。
{% asset_img r_11.png %}

优势函数告诉我们，在状态{% mathjax %}s{% endmathjax %}中选择特定动作{% mathjax %}a{% endmathjax %}比在相同状态{% mathjax %}s{% endmathjax %}中随机选择一个动作所获得的平均期望值要好多少。在这个状态下，选择“向下”动作而不是随机选择一个动作会给代理带来更多奖励。这意味着“向下”动作比这个状态下的平均动作要好。

**估计优势项**是强化学习中减少方差、提高学习效率的关键技术。在`RLHF`等高级应用中,它帮助语言模型更好地理解和优化人类偏好。我们可以用多种方式来估计优势项，如下：
{% asset_img r_12.png %}

如果我们停止得太早，我们将得到非常高的偏差（因为我们正在近似价值函数，并且只使用来自我们轨迹的一个“真实”奖励）。如果我们在很多项之后停止，我们将得到很高的方差。为了解决这个偏差方差问题，我们可以对项进行加权求和以获得广义优势估计：
{% asset_img r_13.png %}

这是一个**递归公式**，其中最后一项等于第一个展开式，倒数第二项等于第二个展开式（以{% mathjax %}\lambda{% endmathjax %}加权）等等。

对于语言模型，此结果会告诉策略（语言模型）增加选择下一个标记的可能性，前提是提示词（状态）预期会产生“高于平均水平”的奖励。这意味着语言模型将选择更有可能产生符合其奖励模型（与我们的训练数据集更一致）的未来`token`的`token`。
{% asset_img r_14.png %}

假设此操作（“上海”一词）将产生好的答案，因此奖励较高。这将训练模型在看到相同提示时更频繁地选择“上海”一词；假设此操作（“巧克力”一词）将产生坏的答案，因此奖励较低。这将训练模型在看到相同提示时较少选择“巧克力”一词。
{% asset_img r_15.png %}

第二个问题：参数更新与轨迹采样的关系。每次我们更新神经网络的参数时，实际上是在改变策略。这意味着之前采样的轨迹可能不再准确反映新策略下的行为。因此，为了获得准确的期望估计，我们需要在每次参数更新后重新采样轨迹。
{% asset_img r_16.png %}

{% mathjax '{"conversion":{"em":14}}' %}
\theta_{k+1} = \theta_k + \alpha\nabla_{\theta}J(\pi_{\theta})|_{\theta_k}
{% endmathjax %}

#### 重要性采样

**重要性采样**(`Importance Sampling`)是一种在强化学习中广泛使用的技术，特别是在离线策略学习中。它的主要目的是解决样本分布不匹配的问题。重要性采样允许我们使用一个策略（行为策略）生成的数据来评估另一个策略（目标策略）的期望值。这在离线策略学习中特别有用，因为我们可以使用旧策略收集的数据来评估和改进新策略。重要性抽样允许使用从另一个分布`Y`中获取的样本来评估分布`X`的期望值。
{% asset_img r_17.png %}

#### 离线策略学习

**离线策略学习**(`Off-policy Learning`)是强化学习中的一种重要方法。离线策略学习允许智能体从一个策略（行为策略）生成的数据中学习另一个策略（目标策略）。换句话说，用于选择动作的策略与正在学习和改进的策略是不同的。行为策略(`Behavior Policy`)：用于与环境交互和生成数据的策略。目标策略(`Target Policy`)：我们希望学习和优化的策略。`Q-learning`是一个经典的离线策略学习算法。它学习最优动作值函数，而不考虑当前正在遵循的策略。
{% asset_img r_18.png %}

这里没有拷贝两份策略。我们采样初始轨迹，将其保存在内存中。然后我们使用保存的轨迹来更新模型。

#### PPO损失

{% asset_img r_19.png %}

#### 奖励黑客（Reward hacking）

如果我们应用上述“`PPO`”，语言模型可能只会学习输出奖励模型想要看到的内容，以最大化其回报。我们当然希望语言模型获得良好的奖励，但同时我们希望语言模型输出的内容仍然看起来像它所训练的训练数据那样。出于这个原因，对于模型生成的每个奖励，我们都会通过优化策略生成的逻辑与语言模型的冻结版本之间的`KL`散度来惩罚奖励。计算对数概率（使用线性层和`softmax`），如果对数概率差异太大，则惩罚分配的奖励（使用`KL`散度），再次计算对数概率（使用线性层和`softmax`）。

{% asset_img r_20.png `Reward hacking 流程` %}

#### 实现

```python
import torch
import wandb
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl.core import LengthSampler
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    # Build a dataset to be used for the training.
    # It is a series of prompts (each with different length chosen randomly)
    # We will use it to generate the responses and compute the rewards.
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load the IMDB dataset
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    # Only choose reviews with more than 200 tokens
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        # From each review just keep the first `input_size` tokens, this represents the prompt used to generate the response
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

if __name__ == '__main__':
    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=1.41e-5,
        log_with="wandb",
    )

    wandb.init()
    dataset = build_dataset(config)

    # This is the model we are going to fine-tune with PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    # This is the reference model (frozen) for the KL divergence
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug

    # This is the reward model: a "positive" (e.g. a positive review) response will be given a high reward, 
    # a "negative" response will be given a low reward
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

    # Print some examples of sentiments generated by the reward model
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": 16}
    text = "this movie was really bad!!"
    print(sentiment_pipe(text, **sent_kwargs))

    text = "this movie was really good!!"
    print(sentiment_pipe(text, **sent_kwargs)) 
    
    # [{'label': 'NEGATIVE', 'score': -2.335047960281372}, {'label': 'POSITIVE', 'score': 2.557039737701416}]

    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)

    # The configuration to generate responses (trajectories)
    response_generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }

    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]

        #### Phase 1: Get trajectories from the offline policy
        # In this case we are only generating the responses, but not computing the log probabilities, which will be computed internally by the PPOTrainer.
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            response_generation_kwargs["max_new_tokens"] = gen_len # Number of tokens to generate (chosen randomly)
            response = ppo_trainer.generate(query, **response_generation_kwargs) # It returns the (query + response) tokens
            response_tensors.append(response.squeeze()[-gen_len:]) # Only take the tokens corresponding to the generated response (remove the prompt/query from the beginning)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Phase 1: Compute rewards
        # Join the query (prompt) + response (generated tokens)
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # Compute the reward for each of the texts (query + response)
        # shape: A list of dictionaries with two keys: POSITIVE and NEGATIVE. We are interested in the POSITIVE score. This will be our reward.
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs) 
        
        # [{'label': 'NEGATIVE', 'score': -2.335047960281372}, {'label': 'POSITIVE', 'score': 2.557039737701416}]

        # The reward for each text is the score (logit) corresponding to the POSITIVE class. 
        # shape: A list of scalars, one for each generated response. 
        # It means we assign the reward to the whole response (not to each token).
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        #### Phase 1 + Phase 2: calculate the logprobs and then run the PPO update
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)

    model.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
    tokenizer.save_pretrained("gpt2-imdb-pos-v2", push_to_hub=False)
```