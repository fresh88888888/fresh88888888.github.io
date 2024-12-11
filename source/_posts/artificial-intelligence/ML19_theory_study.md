---
title: 机器学习(ML)(十九) — 强化学习探析
date: 2024-12-02 17:30:11
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

**强化学习**(`RL`)背后的想法是**智能体**(`Agent`)通过与**环境**(`Environment`)交互（通过反复试验），并从环境中接收**奖励**(`Rewards`)作为执行**动作**(`Action`)的反馈来学习。从环境的互动中学习，源自于经验。这就是人类与动物通过互动进行学习的方式，**强化学习**(`RL`)是一个**解决控制任务**（也称**决策问题**）的框架，通过构建**智能体**(`Agent`)，通过反复试验与环境交互从环境中学习并获得奖励（正面或负面）作为独特反馈。**强化学习**(`RL`)只是一种从行动中学习的计算方法。
<!-- more -->

**任务**是**强化学习**(`RL`)问题的一个实例，这里有两种类型的任务：**情景式任务**和**持续式任务**。
- **情景式任务**：这种情况下，会有一个起点和终点（称为**终端状态**），这将创建一个情节：状态、动作、奖励和新状态的列表。
- **持续式任务**：这些任务会永远持续下去（没有终止状态），在这种情况下，**智能体**(`Agent`)必须学习如何选择最佳**动作**(`Action`)，并同时与环境进行交互，例如，一个执行自动股票交易的代理。对于此任务，没有起点和终点。

**探索**：是通过尝试随机动作来探索环境以便找到有关环境的更多信息。**利用权衡**：是利用已知信息来最大化回报。**强化学习**(`RL`)**智能体**(`Agent`)的目标是最大化预期累积奖励。我们需要平衡对环境的**探索程度**和**环境的利用程度**。因此，我们必须定义一个有助于处理这种权衡的规则。例如，选择餐厅的问题，**利用权衡**：你每天都去同一家你知道不错的餐厅，但却有可能错过另一家更好的餐厅。**探索**：尝试从未去过的餐厅，可能会有不愉快的经历，但也有可能获得美妙的体验。**策略**{% mathjax %}\pi{% endmathjax %}可以视为**智能体**(`Agent`)的大脑，它是一种函数，可以告知在指定状态下采取什么动作。
{% asset_img ml_1.png %}

**策略**{% mathjax %}\pi{% endmathjax %}就是我们要学习的函数，目标就是找到最优的**策略**{% mathjax %}\pi^{*}{% endmathjax %}，也就是当**智能体**(`Agent`)按照这个策略行动时，能够**最大化预期回报**的策略。通过训练找到{% mathjax %}\pi^{*}{% endmathjax %}。有2种方法通过训练**智能体**(`Agent`)来找到最佳策略{% mathjax %}\pi^{*}{% endmathjax %}：基于策略的和基于价值的方法。
- **基于策略的方法**：在**基于策略的方法**中，学习策略函数。此函数定义从每个状态到最佳动作的映射。或者定义该状态下可能动作集的**概率分布**。这里的策略分为：**确定性策略**，给定状态下的策略将始终返回相同的动作，记作{% mathjax %}a = \pi(s){% endmathjax %}。**随机性策略**：输出**动作**的概率分布，记作{% mathjax %}\pi(a|s) = P[A|s]{% endmathjax %}。
- **基于价值的方法**：在**基于价值的方法**中，不是学习策略函数，而是学习将状态映射到它的预期值的**价值函数**，记作{% mathjax %}v_{\pi}(s) = \mathbb{E}_{\pi}[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots | S_t = s]{% endmathjax %}，这里可以看到**价值函数**为每一个可能得状态定义了值。

任何**强化学习**(`RL`)**智能体**(`Agent`)的目标都是**最大化预期累积奖励**（也称为**预期回报**），因为**强化学习**(`RL`)基于奖励假设，即所有目标都可以描述为**预期累积奖励**的最大化。**强化学习**(`RL`)过程是一个**循环**，输出状态、动作、奖励和下一个状态的序列。为了计算**预期累积奖励**（**预期回报**），我们会对奖励进行折扣：更早出现的奖励（在游戏开始时）更有可能发生，因为它们比长期未来奖励更可预测。要解决**强化学习**(`RL`)问题，您需要找到一个最佳策略。策略是**智能体**的“大脑”，它将告诉我们在给定状态下应采取什么动作。**最佳策略**是**智能体**(`Agent`)最大化预期回报的行动的策略。**深度强化学习**引入了**深度神经网络**来估计要采取的**动作**（基于策略）或估计**状态**的值（基于价值），来解决强化学习问题。

|概念|描述|
|---|---|
|**智能体**(`Agent`)|智能体通过反复试验并根据周围环境的奖励（正面或负面）来学习做出决策。|
|**环境**(`Environment`)|环境是一个模拟的世界，智能体可以通过与其交互来学习。|
|**观察**(`Observe`)|环境/世界状态的部分描述。|
|**状态**(`State`)|对世界状态的完整描述。|
|**动作**(`Action`)|离散动作：有限数量的动作，例如左、右、上、下；连续动作：动作的无限可能性；例如，在自动驾驶汽车的情况下，驾驶场景中发生动作的可能性是无限的。|
|**奖励**(`Reward`)|强化学习中的基本要素。判断智能体采取的动作的好坏。强化学习算法专注于最大化累积奖励。强化学习问题可以表述为（累积）回报的最大化。|
|**折扣**(`Discounting`)|刚刚获得的奖励比长期奖励更可预测，因此更有可能发生。|
|**任务**(`Task`)|任务分为：情景式，有起点和终点；连续式，有起点，没有终点。|
|**探索**(`Exploration`)|通过尝试随机动作来探索环境并从环境中获取反馈/回报/奖励。|
|**利用**(`Exploitation Trade-Off`)|它平衡了对环境的探索程度和环境的利用程度。|
|**离线策略算法**|训练和推理时使用不同的策略|
|**在线策略算法**|训练和推理过程中使用相同的策略|
|**策略**(`Policy`)|它被称为智能体的大脑。在给定状态下要采取什么动作。当智能体按照该策略执行时，最大化预期回报的策略。它是通过训练来学习的。以{% mathjax %}1 - \epsilon{% endmathjax %}的概率选择预期奖励最高的动作。选择一个具有{% mathjax %}\epsilon{% endmathjax %}概率的随机动作。随着时间的推移，{% mathjax %}\epsilon{% endmathjax %}通常会减少，将焦点转移到探索上。|
|{% mathjax %}\epsilon -{% endmathjax %}**贪婪策略**|**强化学习**中常见策略是平衡探索和利用。|
|**贪婪策略**|根据当前对环境的了解，始终选择预计会带来最高回报的动作（仅限探索），总是选择预期回报最高的行动。不包括任何探索。在不确定性或最佳行动未知的环境中可能会不利。|
|**基于策略的方法**|在这个方法中，策略是直接学习的。将每个状态映射到该状态下最佳对应的动作。或者该状态下可能动作集合的概率分布。策略通常用神经网络进行训练，选择在给定状态下采取什么动作。在这种情况下，神经网络输出智能体应该采取的动作，而不是**价值函数**。根据环境获得的经验，神经网络将重新调整并提供更好的动作。|
|**基于价值的方法**|在这个方法中，不需要训练策略，而是训练一个**价值函数**，将每个状态映射到该状态的预期值。**价值函数**经过训练，输出状态或**状态-动作对**的值。但是，这个值并没有定义智能体应该采取什么动作。相反，需要根据**价值函数**的输出指定智能体的动作。例如，我们可以决定采用一种策略来采取始终能带来最大回报的动作（**贪婪策略**）。总之，该策略是一种**贪婪策略**，它使用**价值函数**的值来决定要采取的动作。|
|**蒙特卡洛**(`MC`)**学习策略**|在回合结束时进行学习。使用**蒙特卡洛**，等到回合结束，然后根据完整的回合更新**价值函数**（或策略函数）。|
|**时间差分**(`TD`)**学习策略**|每一步都进行学习。通过**时间差分学习**，在每一步更新**价值函数**（或策略函数）。|
|**深度**`Q-Learning`|一种基于价值的**深度强化学习算法**，使用深度神经网络(**卷积神经网络**)来近似给定状态下动作的`Q`值。深度`Q-Learning`的目标是通过学习动作值来找到**最大化预期累积奖励**的最佳策略。|
|**策略梯度**|基于策略的方法的子集，其目标是使用**梯度上升**最大化、参数化策略的性能。**策略梯度**的目标是通过调整策略来控制动作的**概率分布**，方便更频繁地采样好的动作（最大化回报）。|
|**蒙特卡洛强化**|一种**策略梯度算法**，使用整个回合的预测回报来更新策略参数。|
`Gymnasium`是一个为所有单智能体**强化学习**环境提供`API`的框架，其中包括常见环境的实现：`cartpole`（游戏）、`pendulum`（游戏）、`mountain-car`（山地车）、`mujoco`（物理引擎模拟器）、`atari`（游戏）等。`Gymnasium`包括其四个主要功能：`make()`、`Env.reset()`、`Env.step()`和`Env.render()`。`Gymnasium`的核心是`Env`，一个`Python`类，代表**强化学习**理论中的**马尔可夫决策过程**(`MDP`)（注意：这不是完美的重构，缺少`MDP`的几个组成部分）。该类为用户提供了生成初始状态、根据操作转换/移动到新状态以及可视化环境的能力。除了`Env`之外，还提供`Wrapper`来帮助增强/修改环境，特别是智能体观察、奖励和采取的动作。
```python
import gymnasium as gym

# 首先创建一个RL环境，被称作`LunarLander-v2`
env = gym.make('LunarLander-v2')

# 接下来，重置这个环境
observation, info = env.reset()

for _ in range(20):
    # 采取一个随机的动作
    action = env.action_space.simple()
    print('action taken: ',action)
    
    # 在这个环境中执行这个动作，并获取下一个状态、奖励、terminated、truncated、info
    observation, reward, terminated, truncated, info = env.step(action)

    # 如果游戏被terminated或truncated，则重置这个环境。
    if terminated or truncated :
        observation, info = env.reset()
        print("environment is reset")

env.close()
```
我们将训练一个**智能体**(`Agent`)，即**月球着陆器**，使其正确地着陆在月球上。**智能体**(`Agent`)需要学习调整其速度和位置（水平、垂直和角度），从而实现正确着陆。我们看到，通过观察空间形状`(8,)`，观察到是一个大小为`8`的向量，其中每个值包含有关着陆器的不同信息：水平坐标(`x`)、垂直坐标(`y`)、水平速度(`x`)、垂直速度(`y`)、角度、角速度、左腿接触点是否已接触地面（布尔值）、右腿接触点是否已接触地面（布尔值）。动作空间（**智能体**(`Agent`)可以采取的一组可能的动作）是离散的，有`4`个动作可用：动作`0`-不做任何事，动作`1`-启动左方向的引擎，动作`2`-启动主发动机，动作`3`-启动右方向引擎。
```python
env = gym.make("LunarLander-v2")
env.reset()

print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation

# Observation Space Shape (8,)
#Sample observation [-11.904714    12.34132      1.8366828   -1.7705393   -1.5868014    4.7483463    0.08337952   0.5845598 ]

print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action

# Action Space Shape 4
# Action Space Sample 0
```
**奖励函数**（在每个时间步给予奖励的函数），每一步之后都会获得奖励。一个回合的总奖励是该回合中所有步的奖励总和。 对于每一步，奖励：随着着陆器距离着陆台越来越近或越来越远，其变化幅度会越来越大或越来越小；着陆器移动得越慢/越快，其增加/减少量就越大；着陆器倾斜越大（角度不水平），其衰减就越小；着陆器倾斜越大（角度不水平），其衰减就越小；每条腿接触地面一次，增加`10`分；每帧侧发动机启动时减少`0.03`分；主发动机启动每帧减少`0.3`分。该回合将因坠毁或着陆分别获得`-100`或`+100`分的额外奖励。如果某一回合得分达到`200`分，则该回合则被视为解决方案。我们创建了一个由`16`个环境组成的**矢量化环境**（将多个独立环境堆叠成一个环境的方法）。
```python
# Create the environment
env = make_vec_env('LunarLander-v2', n_envs=16)
```
通过控制左、右和主方向引擎，能够将月球着陆器正确地着陆到着陆台。为此，我们将使用**深度强化学习库**：`Stable Baselines3 (SB3)`。`SB3`是`PyTorch`实现的**深度强化学习算法库**。为了解决这个问题，我们将使用 SB3的`PPO`算法。`PPO`（又名**近端策略优化**）。`PPO`是**基于价值**的**强化学习方法**（学习一个**动作价值函数**，在给定状态和动作的情况下采取的最有价值的动作）和**基于策略的强化学习方法**（学习一种策略，为我们提供行动的概率分布）。`Stable-Baselines3`设置：
- 创建环境；
- 定义模型并实例化该模型(`model = PPO("MlpPolicy")`)；
- 使用`model.learn`训练**智能体**(`Agent`)，并定义训练时间步数。

```python
# Define a PPO MlpPolicy architecture
model = model = PPO(policy = 'MlpPolicy', env = env, n_steps = 1024, batch_size = 64, n_epochs = 4,gamma = 0.999,
    gae_lambda = 0.98, ent_coef = 0.01, verbose=1)
```
接下来训练**智能体**(`Agent`)包含`1,000,000`个时间步。
```python
# Train it for 1,000,000 timesteps
model.learn(total_timesteps=1000000)
# Save the model
model.save("ppo-LunarLander-v2")
```
```bash
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 94.4     |
|    ep_rew_mean     | -159     |
| time/              |          |
|    fps             | 2584     |
|    iterations      | 1        |
|    time_elapsed    | 6        |
|    total_timesteps | 16384    |
---------------------------------
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 511         |
|    ep_rew_mean          | -4.69       |
| time/                   |             |
|    fps                  | 1262        |
|    iterations           | 17          |
|    time_elapsed         | 220         |
|    total_timesteps      | 278528      |
| train/                  |             |
|    approx_kl            | 0.006326129 |
|    clip_fraction        | 0.0303      |
|    clip_range           | 0.2         |
|    entropy_loss         | -1.22       |
|    explained_variance   | 0.578       |
|    learning_rate        | 0.0003      |
|    loss                 | 82.3        |
|    n_updates            | 64          |
|    policy_gradient_loss | -0.00175    |
|    value_loss           | 213         |
-----------------------------------------
......
```
**智能体**(`Agent`)评估：将环境包装在监视器中。当评估**智能体**(`Agent`)时，不应该使用训练环境，而是创建一个评估环境。
```python
# 创建一个新的评估环境
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))

# Evaluate the model with 10 evaluation episodes and deterministic=True
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
```
接下来介绍另外一个例子“小狗`Huggy`捡棍子”，它是`Thomas Simonini`根据`Puppo The Corgi`创建的环境，使用的库是`MLAgents`。首先需要下载`MLAgents`代码库：
```bash
# Clone the repository (can take 3min)
git clone --depth 1 https://github.com/Unity-Technologies/ml-agents

# 进入代码库内，并进行安装
cd ml-agents
pip3 install -e ./ml-agents-envs
pip3 install -e ./ml-agents
```
下载并解压环境`Huggy`文件，需要将解压的文件放在`./trained-envs-executables/linux/`文件夹下。
```bash
# 创建./trained-envs-executables/linux/ 文件夹。
!mkdir ./trained-envs-executables
!mkdir ./trained-envs-executables/linux

# 下载和解压Huggy.zip压缩包文件，并确保解压的文件有足够的访问权限。
wget "https://github.com/huggingface/Huggy/raw/main/Huggy.zip" -O ./trained-envs-executables/linux/Huggy.zip

unzip -d ./trained-envs-executables/linux/ ./trained-envs-executables/linux/Huggy.zip

chmod -R 755 ./trained-envs-executables/linux/Huggy
```
{% asset_img ml_2.png %}
`Huggy`的腿是由关节马达驱动的。为了完成目标，`Huggy`需要学会正确地旋转每条腿的关节马达，这样它才能移动。**奖励函数**的初衷是为了让`Huggy`完成目标："取回棍子"。**强化学习**的核心之一是**奖励假设**：目标可以描述为预期累积奖励的最大化。在这里，目标是`Huggy`朝棍子走去并捡起棍子，但不要旋转太多。因此**奖励函数**必须转化这个目标。奖励函数：
- **定位奖励**：这里奖励它接近目标。
- **时间惩罚**：每次动作都给予固定**时间惩罚**，以迫使它尽快到达棍子所在的位置。
- **旋转惩罚**：如果`Huggy`旋转太多或者转得太快，就对它进行惩罚。
- **达到目标奖励**：奖励`Huggy`完成目标。

<video controls autoplay><source src="https://huggingface.co/datasets/huggingface-deep-rl-course/course-images/resolve/main/en/notebooks/unit-bonus1/huggy.mp4" type="video/mp4">
</video>

在用`ML-Agents`训练**智能体**(`Agent`)之前，你需要创建一个`/content/ml-agents/config/ppo/Huggy.yaml`来保存训练的超参数。
```yaml
behaviors:
  Huggy:
    trainer_type: ppo
    hyperparameters:
      batch_size: 2048
      buffer_size: 20480
      learning_rate: 0.0003
      beta: 0.005
      epsilon: 0.2
      lambd: 0.95
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize: true
      hidden_units: 512
      num_layers: 3
      vis_encode_type: simple
    reward_signals:
      extrinsic:
        gamma: 0.995
        strength: 1.0
    checkpoint_interval: 200000
    keep_checkpoints: 15
    max_steps: 2e6
    time_horizon: 1000
    summary_freq: 50000
```
`checkpoint_interval`：每个检查点之间收集的训练时间步数。`keep_checkpoints`：要保留的模型检查点的最大数量。为了训练**智能体**(`Agent`)，只需要启动`mlagents-learn`并选择包含环境的可执行文件。
```bash
mlagents-learn "./config/ppo/Huggy.yaml" --env="./trained-envs-executables/linux/Huggy" --run-id="Huggy" --no-graphics
```
使用`ML Agents`运行训练脚本。这里定义了四个参数：
- `mlagents-learn <config>`：超参数配置文件所在的路径。
- `--env`：环境可执行文件所在的位置。
- `--run-id`: 为训练运行`ID`创建的名称。
- `--no-graphics`：在训练期间不启动可视化。
- `--resume`：训练模型，发生中断时使用`--resume`标志继续训练。

#### Q-Learning

在**强化学习**(`RL`)中，构建了一个可以做出智能决策的**智能体**(`Agent`)。例如，一个学习玩视频游戏的**智能体**(`Agent`)。或者一个通过决策买入以及何时卖出股票来学习最大化其收益的交易智能体**智能体**(`Agent`)。
{% asset_img ml_3.png %}

为了做出明智的决策，**智能体**(`Agent`)将通过与**环境**互动从环境中学习，并获得奖励（正面或负面）作为反馈。其目标是**最大化其预期累积奖励** 。**智能体**(`Agent`)的决策过程称为策略{% mathjax %}\pi{% endmathjax %}：给定一个状态，策略将输出一个动作或动作的概率分布。也就是说，给定对环境的观察，策略将提供**智能体**(`Agent`)应该采取的一个动作（或每个动作的多个概率）。这里的目标是找到一个**最优策略**{% mathjax %}\pi^{*}{% endmathjax %}，也就是能够带来最佳预期累积回报的策略。目前有两种解决**强化学习**(`RL`)问题的方法：
- **基于策略的方法**：直接训练策略来学习在给定状态（或该状态下动作的概率分布）下采取什么动作（{% mathjax %}\text{State}\rightarrow \pi(\text{State}) \rightarrow \text{Action}{% endmathjax %}）。策略以状态作为输入，并输出在该状态下采取的动作（**确定性策略**：在给定状态下输出一个动作的策略，与输出动作概率分布的随机策略相反）。我们不需要手动定义策略的动作，而是通过训练来定义它。
- **基于价值的方法**：训练一个**价值函数**来了解哪种状态更有价值，并使用该**价值函数** 采取该状态的动作，通过训练输出状态或状态-动作对的值的**价值函数**。给定这个**价值函数**，策略将采取动作。由于策略未经训练/学习，所以需要指定其动作。例如，如果想要一个策略，在给定**价值函数**的情况下，采取始终带来最大回报的动作，这时需要创建一个**贪婪策略**。无论你使用什么方法解决问题，你都会有一个策略。在**基于价值的方法**的任务，你不需要训练策略：策略只是一个简单的预先指定的函数（例如，**贪婪策略**），它使用**价值函数**给出的值来选择其动作。

在**基于价值的方法**中，需要学习一个**价值函数**，将状态映射到该状态的预期值。状态的预期值是**智能体**(`Agent`)从该状态开始并按照当前策略执行时获得的**预期折扣回报**。在**基于策略**的训练中，通过直接训练策略来找到最佳策略（表示为{% mathjax %}\pi^{*}{% endmathjax %}）。在**基于价值**的训练中，找到一个**最佳价值函数**（表示为{% mathjax %}Q^{*}{% endmathjax %}或{% mathjax %}V^{*}{% endmathjax %}）就会得到最佳策略，记作{% mathjax %}\pi^{*}(s) = \text{arg}\;\underset{a}{\max} Q^{*}(s,a){% endmathjax %}。

现在有`2`种基于价值的函数：
- **状态值函数**：我们将策略{% mathjax %}\pi{% endmathjax %}下的状态值函数写成如下形式：{% mathjax %}V_{\pi}(s) = \mathbb{E}_{\pi}[G_t|S_t = s]{% endmathjax %}。对于每个状态，如果**智能体**(`Agent`)从该状态开始，然后永远遵循该策略，则状态值函数输出预期的回报。
- **动作值函数**： 在动作价值函数中，对于每个**状态-动作对**，如果**智能体**(`Agent`)从该状态开始，采取该动作，然后永远遵循策略，则**动作值函数**输出预期的回报。写成如下形式：{% mathjax %}Q_{\pi}(s,a) = \mathbb{E}_{\pi}[G_t|S_t = s, A_t = a]{% endmathjax %}

**状态值函数**与**动作值函数**的区别是：对于**状态值函数**，需要计算状态{% mathjax %}S_t{% endmathjax %}的值；对于**动作值函数**，需要计算**状态-动作对**的值({% mathjax %}S_t, A_t{% endmathjax %})在该状态下采取该动作的值。无论我们选择哪种价值函数（**状态值**或**动作值函数**），返回的值都是预期的回报。为了计算**状态**或**状态-动作对**的每个值，我们需要将**智能体**(`Agent`)从该状态开始时可以获得的所有奖励相加。使用**贝尔曼方程**简化了状态值或者状态-动作对值的计算。**贝尔曼方程**是一个递归方程，其工作原理如下：不必从头开始计算每个状态的回报，而是可以将任何状态的值视为：{% mathjax %}\text{即时奖励}R_{t+1} + \gamma V_{\pi}(S_{t+1}){% endmathjax %}，**贝尔曼方程**可以写成如下形式：{% mathjax %}V_{\pi}(s_t) = \mathbb{E}_{\pi}[R_{t+1} + \gamma V_{\pi}(S_{t+1})|S_t = s]{% endmathjax %}。**贝尔曼方程**的思想不是将每个值计算为预期回报的总和，而是将值计算为**即时奖励 + 后续状态的折扣值的总和**。

**强化学习**(`RL`)的**智能体**(`Agent`)通过与环境交互来学习。其理念是根据经验和获得的的奖励，**智能体**(`Agent`)将更新其**价值函数**或**策略**。**蒙特卡洛**和**时间差分学习**是训练**价值函数**或**策略函数**的两种不同策略，它们都使用经验来解决**强化学习**(`RL`)问题。**蒙特卡洛**在学习之前使用了整个经验。而**时间差分学习**可以在每一步{% mathjax %}(S_t,A_t,S_{t+1},R_{t+1}){% endmathjax %}进行学习。

**蒙特卡洛**等待一个回合结束，计算{% mathjax %}G_t{% endmathjax %}（回报）并用于更新目标{% mathjax %}V(S_t){% endmathjax %}。因此，在更新价值函数之前，需要完成一次完整的交互。可以记作{% mathjax %}V(S_t)\leftarrow V(S_t) + \alpha [G_t - V(S_t)]{% endmathjax %}，其中{% mathjax %}\alpha{% endmathjax %}是学习率，{% mathjax %}V(S_t){% endmathjax %}是状态{% mathjax %}t{% endmathjax %}的更新值。代理使用策略采取动作。例如，使用`Epsilon Greedy`策略，即在探索（随机动作）和利用之间交替的策略，得到了奖励和下一个状态。在这一回合结尾有一个包含(`State, Actions, Rewards, Next States`)的多元组的列表，**智能体**(`Agent`)将所有奖励回报{% mathjax %}G_t{% endmathjax %}求和，然后根据公式{% mathjax %}V(S_t)\leftarrow V(S_t) + \alpha [G_t - V(S_t)]{% endmathjax %}更新状态值{% mathjax %}V(S_t){% endmathjax %}，执行多个回合之后，**智能体**(`Agent`)学习的越来越好。

**时间差分学习**的核心思想是利用当前的估计值来更新未来的估计值。它通过比较当前状态的价值与下一个状态的价值之间的差异（即“时间差”）来进行更新。这种方法允许智能体在每一步都进行增量学习，而不必等待整个回合结束。因为没有经历整个过程，所以没有{% mathjax %}G_t{% endmathjax %}，那么用{% mathjax %}R_{t+1} + \gamma V(S_{t+1}){% endmathjax %}来估算{% mathjax %}G_t{% endmathjax %}，**时间差分学习**更新的状态值是基于{% mathjax %}V(S_{t+1}){% endmathjax %}估算，而不是完整的预期回报{% mathjax %}G_t{% endmathjax %}。可以记作{% mathjax %}V(S_t)\leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]{% endmathjax %}

通过**蒙特卡洛方法**学习，可以从一个完整的回合中更新**价值函数**，使用该事件的实际准确的折扣回报。使用**时间差分学习**学习，可以一步一步更新**价值函数**，并替换了{% mathjax %}G_t{% endmathjax %}。

`Q-Learning`是**基于离线策略价值的方法**，它使用**时间差分学习**(`TD`)方法来训练其**动作值函数**：**基于价值的方法**，通过训练价值或**动作值函数**间接地找到最优策略，该函数将告诉我们每个**状态**或每个**状态-动作对**的值；**时间差分学习**方法，在每一步而不是在回合结束时更新其**动作值函数**。`Q-Learning`是我们用来训练`Q`函数的算法，`Q`函数是一个**动作值函数**，它决定在该状态下采取动作的值。给定一个状态和动作，`Q`函数输出一个状态-动作值（也称为`Q`值）。**价值**和**奖励**的区别：状态或状态-动作对的值（**价值**）是**智能体**(`Agent`)从该状态（或状态-动作对）开始并根据其策略执行时获得的**预期累积奖励**；**奖励**是某个状态下执行某个动作后从环境中获得的**反馈**。`Q`函数由`Q`表编码，该表中的每个单元格对应一个**状态-动作对值**。给定一个**状态**和**动作**，`Q`函数将在其`Q`表中搜索以输出该值。
{% asset_img ml_4.png %}

`Q-Learning`是一种强化学习算法。训练一个`Q`函数（动作值函数），其内部是一个包含所有**状态-动作对**值的`Q`表。给定一个状态和动作，`Q`函数将在其`Q`表中搜索相应的值。当训练完成后，有一个最佳的`Q`函数，意味着有一个最佳的`Q`表。如果有一个最佳`Q`函数，那么就有一个**最佳策略**。它可以表示为：{% mathjax %}\pi^{*}(s) = \text{arg}\;\underset{a}{\max}Q^{*}(s,a){% endmathjax %}。起初`Q`表的内容毫无意义，因为它将每个状态-动作对初始化为`0`。随着**智能体**(`Agent`)探索环境并更新`Q`表，它将越来越接近最优策略的近似值。如下图所示，这是`Q-Learning`伪代码。
{% asset_img ml_5.png %}

- 步骤一：初始化{% mathjax %}Q{% endmathjax %}表，需要为{% mathjax %}Q{% endmathjax %}表中的每个**状态-动作对**做初始化为`0`。
- 步骤二：使用`epsilon-greedy`策略选取一个动作。**贪婪策略**是一种处理探索/利用的策略。初始时{% mathjax %}\epsilon = 0{% endmathjax %}，在训练开始时，探索的概率会很大（{% mathjax %}\epsilon{% endmathjax %}值较大），这时候选择探索的策略（随机选取动作）；但随着训练的进行，{% mathjax %}Q{% endmathjax %}表内容的值越来越接近，{% mathjax %}\epsilon {% endmathjax %}值逐渐降低，这时候选择利用策略（**智能体**选取具有最高**状态-动作对**值的动作）。
- 步骤三：执行动作{% mathjax %}A_t{% endmathjax %}，获得奖励{% mathjax %}R_{t+1}{% endmathjax %}和下一个状态{% mathjax %}S_{t+1}{% endmathjax %}。
- 步骤四：更新{% mathjax %}Q(S_t,A_t){% endmathjax %}表。

{% note warning %}
请记住，在**时间差分学习**(`TD`)学习中，在交互的一个回合之后更新**策略**或**价值函数**（取决于选择的**强化学习**方法）。
{% endnote %}

为了实现**时间差分学习**(`TD`)的目标，这里使用了**即时奖励**加上下一个状态的折扣值，记作{% mathjax %}R_{t+1} + \gamma V(S_{t+1}){% endmathjax %}，通过找到在下一个状态下最大化当前{% mathjax %}Q{% endmathjax %}函数的动作来获得（称之为**引导**）。
{% mathjax '{"conversion":{"em":14}}' %}
V(S_t)\leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
{% endmathjax %}
则{% mathjax %}Q(S_t,A_t){% endmathjax %}函数的更新公式如下：
{% mathjax '{"conversion":{"em":14}}' %}
Q(S_t,A_t)\leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma \underset{a}{\max}Q(S_{t+1},a) - Q(S_t,A_t)]
{% endmathjax %}
为了更新{% mathjax %}Q(S_t,A_t){% endmathjax %}值，则需要知道{% mathjax %}S_t,A_t,R_{t+1},S_{t+1}{% endmathjax %}和{% mathjax %}R_{t+1} + \gamma \underset{a}{\max}Q(S_{t+1},a){% endmathjax %}（**时间差分学习**的目标）。获得奖励{% mathjax %}R_{t+1}{% endmathjax %}之后，执行后一个动作{% mathjax %}A_t{% endmathjax %}。为了获得下一个状态的最佳**状态-动作对**值，使用**贪婪策略**来选择下一个最佳动作。请注意，这不是`epsilon-greedy`策略，它将始终采取具有最高**状态-动作对**值的动作。当完成此{% mathjax %}Q{% endmathjax %}值的更新后，从新的状态开始，并再次使用`epsilon-greedy`策略来选取动作。这也是`Q-Learning`是离线策略算法的原因。**离线策略**，使用不同的策略进行推理和训练；**在线策略**，使用相同的策略进行推理和训练。

#### 深度Q-Learning

`Q-Learning`是一种用于训练`Q`函数的算法，`Q`函数是一种**动作值函数**，它决定了该状态下采取动作的值，`Q`来自于该状态下该动作的“**质量**”(`the Quality`)。但存在一个问题，由于`Q-Learning`是一种表格方法，如果**状态和动作空间**不够小，则无法用数组和表格表示，也就是说，`Q`表无法进行扩展。
{% asset_img ml_6.png %}

深度`Q`网络(`DQN`)的网络架构：由3部分组成，分别是**神经网络**，`DQN`使用**深度神经网络**（通常是**卷积神经网络**）作为**函数逼近器**，当前状态作为输入，所有可能动作的`Q`值向量为输出。通过这种方式，`DQN`能够处理高维输入；**经验回放**，`DQN`引入了**经验回放机制**，将智能体与环境交互过程中获得的经验存储在一个**回放缓冲区**中。每次更新时，从这个缓冲区中随机抽取一批经验进行训练。这种方法可以打破样本之间的相关性，提高学习效率和稳定性；**目标网络**，`DQN`使用两个神经网络：一个是**主网络**（用于选取动作），另一个是**目标网络**（用于计算目标`Q`值）。**目标网络**的参数定期更新，以保持相对稳定。这有助于减少训练过程中的波动性。

深度`Q-Learning`算法使用深度神经网络（**卷积神经网络**）来近似某一状态下每个动作的`Q`值（**价值函数**预测）。与`Q-Learning`算法不同之处在于，在训练阶段，不会直接更新**状态-动作对**的`Q`值；而是创建一个**损失函数**，用于比较预测Q值与目标Q值，并使用**梯度下降**来更新深度Q网络(`DQN`)的权重，为了更好的接近目标Q值。这里`Q-Target`记作{% mathjax %}R_{t+1} + \gamma \underset{a}{\max}Q(S_{t+1},a){% endmathjax %}，`Q-Loss`记作为：{% mathjax %}R_{t+1} + \gamma \underset{a}{\max}Q(S_{t+1},a) - Q(S_t,A_t){% endmathjax %}。深度Q-Learning算法有2个阶段：**采样**，执行动作并将观察到的经验元组存储在**回放缓冲区**中；**训练**，随机选择一小批元组，并使用**梯度下降**更新从该批元组中学习。深度`Q-Learning`训练可能存在不稳定性，主要是结合了非线性`Q`值函数（**神经网络**）和**引导程序**（使用现有预测而不是实际的、完整的回报来更新目标）。为了实现稳定训练，有`3`种不同的解决方案：
- 经验回放，使经验利用变得更加高效。
- 固定`Q-Target`以稳定训练。
- 双重深度`Q-Learning`，解决`Q`值预测过高的问题。

深度`Q-Learning`的**经验回放**有两个作用：1、**在训练过程中更有效地利用经验**。通常，在**在线强化学习**中，**智能体**(`Agent`)与**环境**交互，获得经验（状态、动作、奖励和下一个状态），从中学习（更新神经网络），然后丢弃它们。这种方式效率不高。**经验回放**有助于更有效地利用经验。使用**回放缓冲区**来保存经验，并在训练期间重复使用的经验样本，这使得**智能体**(`Agent`)可以从相同的经验中多次学习；2、避免忘记之前的经验（又称**灾难性遗忘**），从而减少经验之间的相关性。**灾难性遗忘**：如果将连续的**经验样本**提供给神经网络，那么就会遇到一个问题，当它获得新的经验时，就会忘记旧的经验。解决方案：创建一个**重放缓冲区**，在与环境交互时存储**经验元组**，然后对一小批元组进行采样。这样可以防止网络只学习它之前做过的事情。**经验回放**还有其他好处。通过随机抽样经验，可以消除观察序列中的相关性，并避免动作值发生剧烈震荡或发散。如下图所示，在`Deep Q-Learning`伪代码中：
{% asset_img ml_7.png %}

初始化一个容量为`N`的**重放内存缓冲区**`D`（`N`是定义的超参数）。然后，将经验存储在**重放内存缓冲区**中，并在训练阶段抽取一批经验来提供给深度`Q`网络。使用具有固定参数的单独网络来预测`TD`目标，每隔`C`步从深度`Q`网络复制参数以更新**目标网络**。

**双重深度**`Q`**网络**：又称**双重深度**`Q`**学习神经网络**，由`Hado van Hasselt`提出 。此方法可解决`Q`值预测过高的问题。在计算`TD`目标时，如何确定下一个状态的最佳动作是`Q`值最高的动作？解决方案是：当计算`Q`目标时，使用两个网络将动作选取与目标`Q`值分离。使用`DQN`网络来选择下一个状态的最佳动作（具有最高`Q`值的动作）。使用**目标网络**来计算在下一个状态下采取该动作的目标`Q`值。

#### 策略梯度

**强化学习**(`RL`)的主要目标是找到最优策略{% mathjax %}\pi^{*}{% endmathjax %}，使其**最大化预期累积奖励**。因为**强化学习**(`RL`)是基于“**奖励假设**”：所有目标都可以描述为**预期累计奖励的最大化**。例如，在一场足球比赛中，您将训练两个**智能体**(`Agent`)，目标是赢得比赛。我们可以在**强化学习**(`RL`)中将此目标描述为最大化对方球门的进球数（当球越过球门线时），并最小化自己球门的进球数。基于价值的方法：通过最优价值函数来实现最优策略{% mathjax %}\pi^{*}{% endmathjax %}。目标是**最小化预测值**和**目标值**之间的**损失**以近似真实的**动作值函数**。基于策略的方法：这个方法是将策略参数化，例如，使用神经网络{% mathjax %}\pi_{\theta}{% endmathjax %}，该策略将输出动作的**概率分布**（**随机策略**），记作{% mathjax %}\pi_{\theta}(s) = \mathbb{P}[A|s;\theta]{% endmathjax %}。它的目标就是利用**梯度**上升来最大参数化策略的性能。基于策略的方法，可以直接优化策略{% mathjax %}\pi_{\theta}{% endmathjax %}输出动作的概率分布{% mathjax %}\pi_{\theta}(a|s){% endmathjax %}，​​从而实现最佳累积收益。为此，这里定义一个目标函数{% mathjax %}J(\theta){% endmathjax %}，即**预期累积奖励**。这里只需要找到最大化该**目标函数**的{% mathjax %}\theta{% endmathjax %}值。策略梯度方法：是基于策略的方法的一个子集，优化大多数时候都是基于策略的，因为对于每次更新，只需要使用最新版本的{% mathjax %}\pi_{\theta}(a|s){% endmathjax %}。基于**策略的方法**和**策略梯度方法**之间的区别：
- **在基于策略的方法中**，直接搜索最优策略。通过使用`hill climbing`、**模拟退火**或**进化策略**等技术最大化**目标函数**的局部近似值来优化参数{% mathjax %}\theta{% endmathjax %}。
- **在基于策略梯度方法中**，由于它是基于策略的方法的子集，直接搜索最优策略。但优化参数{% mathjax %}\theta{% endmathjax %}直接通过对**目标函数**的性能进行梯度上升{% mathjax %}J(\theta){% endmathjax %}。

**策略梯度方法**优点：**集成简单**，可以直接预测策略，而无需存储额外的数据；**策略梯度方法**可以学习随机策略，而**价值函数**则不能；**策略梯度方法**在高维动作空间和连续动作空间中更有效；**策略梯度方法**具有更好的收敛特性。**策略梯度方法**缺点：通常，**策略梯度方法**会收敛到局部最大值而不是全局最大值。**策略梯度**逐渐变慢，训练可能需要更长的时间（效率低下）。**策略梯度**可能具有较高的方差。

**策略梯度**的目标是通过调整策略来控制动作的概率分布，以便将来更频繁地采样好的动作（最大化回报）。每次**智能体**(`Agent`)与环境交互时，都会调整参数，以便将来更有可能采样好的动作。但是要如何利用预期回报来优化权重呢？随机策略记作{% mathjax %}\pi{% endmathjax %}，参数记作{% mathjax %}\theta{% endmathjax %}，策略{% mathjax %}\pi{% endmathjax %}是给定一个状态，输出动作的**概率分布**。{% mathjax %}\pi_{\theta}(a_t|s_t){% endmathjax %}是智能体根据当下的策略，从状态{% mathjax %}s_t{% endmathjax %}中选取动作{% mathjax %}a_t{% endmathjax %}的概率。但如何判定该策略是否有效？这里定义一个{% mathjax %}J(\theta){% endmathjax %}的**目标函数**。**目标函数**提供了给定轨迹的**智能体**(`Agent`)的性能，并输出**预期累积奖励**。记作{% mathjax %}J(\theta) = \mathbb{E}_{\tau\sim\pi}[R(\tau)]\;,\;R(\tau) = r_{t+1} + \gamma r_{t+2} + \gamma^{2} r_{t+3} + \gamma^{3} r_{t+4} + \ldots{% endmathjax %}。{% mathjax %}J(\theta) = \sum\limits_{\tau}P(\tau;\theta)R(\tau){% endmathjax %}预期回报也称为**预期累计奖励**，是加权平均值（其中权重由回报{% mathjax %}P(\tau;\theta){% endmathjax %}给出，并包含回报{% mathjax %}R(\tau){% endmathjax %}取得的所有值）。
- {% mathjax %}R(\tau){% endmathjax %}：从任意轨迹获得的回报。要获取此量并用它来计算预期回报，需要将其乘以每个可能轨迹的概率。
- {% mathjax %}P(\tau;\theta){% endmathjax %}：每个可能轨迹的概率{% mathjax %}\tau{% endmathjax %}（该概率取决于{% mathjax %}\theta{% endmathjax %}，因为它定义了用于选择轨迹动作的策略，而该策略会对所访问的状态产生影响）{% mathjax %}J(\theta) = \sum\limits_{\tau}P(\tau;\theta)R(\tau)\;,\;P(\tau;\theta) = [\underset{t=0}{\prod}P(s_{t+1}|s_t,a_t)\pi_{\theta}(a_t|s_t)]{% endmathjax %}。
- {% mathjax %}J(\theta){% endmathjax %}：预期回报，通过对所有轨迹求和，给定{% mathjax %}\theta{% endmathjax %}乘以该轨迹的回报，得出采取该轨迹的概率。我们的目标是通过找到输出最佳动作概率分布的 {% mathjax %}\theta{% endmathjax %}来最大化预期累积奖励：{% mathjax %}\underset{a}{\max}J(\theta) = \mathbb{E}_{\tau\sim\pi_{\theta}}[R(\tau)]{% endmathjax %}。

**策略梯度**是一个优化问题：想要找到最大化目标函数{% mathjax %}J(\theta){% endmathjax %}的{% mathjax %}\theta{% endmathjax %}值，需要使用**梯度上升**。它是**梯度下降**的**逆函数**，因为它给出了最陡峭的增长方向{% mathjax %}J(\theta){% endmathjax %}。**梯度上升**的更新步骤是{% mathjax %}\theta\leftarrow \theta + \alpha \ast \nabla_{\theta}J(\theta){% endmathjax %}，通过反复更新，使{% mathjax %}\theta{% endmathjax %}收敛到{% mathjax %}J(\theta){% endmathjax %}最大化。但是计算{% mathjax %}J(\theta){% endmathjax %}的导数存在`2`个问题：
- 无法计算**目标函数**的**真实梯度**，因为它需要计算每条可能轨迹的概率，这在计算上代价非常大。所以使用基于样本的预测值（收集一些轨迹）来计算**梯度预测值**。
- 为了区分这个**目标函数**，需要区分**状态分布**，称为**马尔可夫决策过程动力学**，这与环境有关。它给出了环境进入下一个状态的概率，考虑到当前状态和**智能体**(`Agent`)采取的动作。问题是无法区分**状态分布**。

解决办法是使用**策略梯度定理**，它将目标函数重新表述为**可微函数**，而不涉及状态分布的**微​​分**。记作{% mathjax %}\nabla_{\theta}J(\theta) = \mathbb{E}_{\pi\theta}[\nabla_{\theta}\log \pi_{\theta}(a_t|s_t)R(\tau)]{% endmathjax %}。**强化算法**，也称为**蒙特卡洛策略梯度算法**，是一种使用整个事件的预测回报来更新策略参数的**策略梯度算法**{% mathjax %}\theta{% endmathjax %}，在循环中：
- 使用策略{% mathjax %}\pi_{\theta}{% endmathjax %}来收集一个回合的{% mathjax %}\tau{% endmathjax %}。
- 使用回合来预测梯度{% mathjax %}\nabla_{\theta}J(\theta) \approx \hat{g} = \sum\limits_{t=0} \nabla_{\theta}\log_{\pi_{\theta}}(a_t|s_t)R(\tau){% endmathjax %}。
- 更新策略的权重{% mathjax %}\theta \leftarrow \theta + \alpha\hat{g}{% endmathjax %}。

{% mathjax %}\nabla_{\theta}\log_{\pi_{\theta}}(a_t|s_t){% endmathjax %}是从状态{% mathjax %}s_t{% endmathjax %}中选取动作{% mathjax %}a_t{% endmathjax %}的对数概率增加最快的方向。如果想增加/减少在状态{% mathjax %}s_t{% endmathjax %}下选取动作{% mathjax %}a_t{% endmathjax %}的对数概率的情况下，该如何改变策略权重？{% mathjax %}R(\tau){% endmathjax %}是奖励函数，如果回报很高，它将提高（状态，动作）组合的概率；否则，它将降低（状态，动作）组合的概率。我们还可以收集多个回合（轨迹）来预测梯度：{% mathjax %}\nabla_{\theta}J(\theta) \approx \hat{g} = \frac{1}{m}\sum\limits_{i=1}\sum\limits_{t=0} \nabla_{\theta}\log_{\pi_{\theta}}(a_t^{(i)}|s_t^{(i)})R(\tau^{(i)}){% endmathjax %}。

