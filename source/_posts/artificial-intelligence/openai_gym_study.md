---
title: OpenAI Gym（PyTorch）
date: 2024-02-21 19:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### OpenAI Gym

`OpenAI Gym`就像一个智能游戏平台，帮你把复杂环境规则打包成一个易用接口。它就像这样：
- ——> `reset()`：一键重启游戏，返回新场景给`AI`看
- ——> `step(action)`：`AI`选择动作，执行后，`Gym`会反馈：
    - 新鲜出炉的观测结果（就是新的游戏画面）
    - 动作带来的奖励分
    - 游戏是否过关
    - 还有额外的小贴士（`info`）

简而言之，`Gym`简化了环境交互，让你专心训练`AI`算法，无需研究背后的实现。
<!-- more -->

```python
import torch
import gym

# 创建一个OpenAI Gym环境实例
env = gym.make('CartPole-v1')

class RLInterface:
    def __init__(self):
        # 将环境的观测空间转化为Pytorch 张量所需的形状
        self.observation_space = env.observation_space.shape
        
        # 将动作空间转换为Pytorch张量所需的形状
        self.action_space = env.action_space
    
    def reset(self):
        # 重置环境并获取新的观测
        obs = env.reset()
        # 转换观测为Pytorch张量
        return torch.tensor(obs, dtype=torch.float32)
    
    def step(self, action):
        # 动作需要从PyTor 张量转回原始形式
        action = int(action.item())
        
        # 在环境中执行动作
        obs, reward, done, info = env.step(action=action)
        
        # 将观测、奖励转化为适合Pytorc使用的形式
        obs_tensor = torch.tensor(obs, dtype= torch.float32)
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        done_tensor = torch.tensor(done, dtype=torch.bool)
        
        return obs_tensor, reward_tensor, done_tensor, info

# 实例化接口
rl_env = RLInterface()

# 开始循环训练
for episode in range(num_episodes):
    obs = rl_env.reset()
    
    while True:
        # 假设Agent是一个策略网络，选择一个动作
        action = agent.choose_action(obs)
        
        # 执行动作并获取环境的反馈
        next_obs, reward, done, _ = rl_env.step(action=action)
        
        # 更新agent（根据强化学习算法的具体逻辑）
        agent.update(obs, action, reward, next_obs, done)
        
        obs = next_obs
        if done:
            break

# 训练结束后关闭环境
env.close()
```