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
