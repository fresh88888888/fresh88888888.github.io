---
title: 基于LSTM实现一个股票预测模型（PyTorch）
date: 2024-02-22 8:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 基于LSTM实现一个股票预测模型

这只是一个简单的例子，并未考虑很多实际问题，，例如特征选择、模型复杂度调整、超参数优化等。在真实环境中，股票价格受到多种因素影响，包括但不限于市场情绪、公司基本面、政策法规等，但纯依赖于历史价格进行预测可能效果有限。此代码并未包含模型验证和预测部分，实际应用中还需要对模型在测试集上的表现进行评估，并用模型对未来数据进行预测。股票预测模型通常是基于机器学习或深度学习技术，通过历史数据（如：开盘价、收盘价、最高价、最低价、交易量等）来预测未来的股价走势。这是一个非常基础的基于LSTM（长短期记忆网络）的股票预测模型实现。
<!-- more -->

```python
import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# 定义一个自定义的数据集类，用于读取和处理股票数据
class StockDataset(Dataset):
    def __init__(self, p_data, look_back= 10):
        self.data = p_data
        self.look_back = look_back
        
    def __len__(self):
        return len(self.data) - self.look_back
    
    def __getitem__(self, idx):
        # 将过去look_back天的数据作为输入，下一天的价格作为输出
        x = self.data[idx:idx+ self.look_back]
        y = self.data[idx + self.look_back]
        # 预测的目标值
        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# 加载并预处理股票数量
df = pd.read_csv('stock_data.csv')

# 只读取收盘价作为特征
data = df['close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
data = scaler.fit_transform(data)

# 划分训练集和测试集
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# 创建数据加载器
dataset = StockDataset(train_data)
dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=True)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=100, output_dim=1, n_layers = 2):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=output_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size0(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers, x.size0(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        
        # 我们只取最后一个时间步的隐藏状态用于预测
        return out
    
model = LSTMModel()

# 定义损失函数即优化器
criterion = nn.MSELoss()

# 使用均方误差作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch [{epoch + 1/100}], Loss: {loss.item():.4f}')

# 测试模型（部分代码未给出，需要根据实际情况设计）
```