---
title: 指数移动平均（PyTorch）
date: 2024-02-21 21:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 指数移动平均

指数移动平均（`EMA`）就像一个会偏爱新信息的智能计算器，帮你更准确地算出数据变化的趋势。不同于简单的移动平均（`SMA`），它给最近的数据点更大话语权。每个时间节点上，新的`EMA`值是由当前数`+`以前EMA值按神秘权重阿尔法相加得出，这个权重一般在`0`到`1`之间浮动，越靠近现在，数据点的影响力越大。而在深度学习领域，`EMA`变成超能力助手，专门帮模型“记账”，追踪参数的平均值。训练时它能打造一个平滑版的权重集合，这对某些高级训练战术特管用，比如测试阶段用“影子”模型来稳稳预测。
<!-- more -->

```python
import torch

# 定义一个简单的类来管理指数移动平均
class ExponentMovingAverage:
    def __init__(self, model, decay_rate = 0.99):
        '''
        初始化指数移动平均类
        '''
        self.model = model 
        self.decay_rate = decay_rate
        # 创建一个字典来存储EMA版本的权重和偏置
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.require_grad and name in self.shadow:
                new_avrage = (1 - self.decay_rate) * param.data + self.decay_rate * self.show[name]
                self.shadow[name].copy_(new_avrage)
    
    def apply_shadow(self):
        '''
        将EMA计算出的权重应用原模型
        '''
        for name, param in self.model.name_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])
        
# 模型实例化
model = SomePyTorchModel()

# 创建EMA对象
ema = ExponentMovingAverage(model, decay_rate=0.9999)

# 在每次训练步骤后更新EMA
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        # 训练模型
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        # 更新EMA权重
        ema.update()
        
# 在训练结束后，可以将EMA权重应用回原始模型以进行推断或评估
ema.apply_shadow()
```