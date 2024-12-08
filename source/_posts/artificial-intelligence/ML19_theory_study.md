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
|**利用权衡**(`Exploitation Trade-Off`)|它平衡了对环境的探索程度和环境的利用程度。|
|**政策**(`Policy`)|它被称为智能体的大脑。在给定状态下要采取什么动作。当智能体按照该策略执行时，最大化预期回报的策略。它是通过训练来学习的。|
|**基于策略的方法**|在这个方法中，策略是直接学习的。将每个状态映射到该状态下最佳对应的动作。或者该状态下可能动作集合的概率分布。|
|**基于价值的方法**|在这个方法中，不需要训练策略，而是训练一个价值函数，将每个状态映射到该状态的预期值。|

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
{% end note%}

为了实现**时间差分学习**(`TD`)的目标，这里使用了**即时奖励**加上下一个状态的折扣值，记作{% mathjax %}R_{t+1} + \gamma V(S_{t+1}){% endmathjax %}，通过找到在下一个状态下最大化当前{% mathjax %}Q{% endmathjax %}函数的动作来获得（称之为**引导**）。
{% mathjax '{"conversion":{"em":14}}' %}
V(S_t)\leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
{% endmathjax %}
则{% mathjax %}Q(S_t,A_t){% endmathjax %}函数的更新公式如下：
{% mathjax '{"conversion":{"em":14}}' %}
Q(S_t,A_t)\leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma \text{arg}\underset{a}{\max}Q(S_{t+1},a) - Q(S_t,A_t)]
{% endmathjax %}
为了更新{% mathjax %}Q(S_t,A_t){% endmathjax %}值，则需要知道{% mathjax %}S_t,A_t,R_{t+1},S_{t+1}{% endmathjax %}和{% mathjax %}R_{t+1} + \gamma \text{arg}\underset{a}{\max}Q(S_{t+1},a){% endmathjax %}（**时间差分学习**(`TD`)的目标）。获得奖励{% mathjax %}R_{t+1}{% endmathjax %}之后，执行后一个动作{% mathjax %}A_t{% endmathjax %}。为了获得下一个状态的最佳**状态-动作对**值，使用**贪婪策略**来选择下一个最佳动作。请注意，这不是`epsilon-greedy`策略，它将始终采取具有最高**状态-动作对**值的动作。当完成此{% mathjax %}Q{% endmathjax %}值的更新后，从新的状态开始，并再次使用`epsilon-greedy`策略来选取动作。这也是`Q-Learning`是离线策略算法的原因。**离线策略**，使用不同的策略进行推理和训练；**在线策略**，使用相同的策略进行推理和训练。

