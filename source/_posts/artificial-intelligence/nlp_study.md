---
title: NLP（PyTorch）
date: 2024-02-22 20:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### NLP

自然语言处理（`NLP`）：就是轻松教会计算机理解人类的语言，像阅读、回答问题、识别语音这些日常操作，都离不开他。这其中，有几大关键算法功不可没：
- 词袋模型：像数豆子一样统计每个词出现的次数，虽然不考虑顺序，却能帮计算机快速分类文本情绪。
- 循环神经网络（`RNN`）：专门对付一连串信息，比如做翻译和预测下一句，只是对长句子力不从心。
- 长短时记忆网络（`LSTM`）：升级版的`RNN`，解决了记忆差的问题，擅长学习语言中的长距离关联。
- 支持向量机（`SVM`）：经典分类器，在区分不同类别文本上有一手。
- 生成对抗网络（`GAN`）：一边生成文本一边鉴别真伪。用于生成逼真的对话内容
- 注意力机制：让模型懂得抓重点，像机器翻译和摘要提炼，就用上了这一招。
- 预训练语言模型：`BRET、GPT`等明星模型先在海量文本里“修炼”，然后应用于各种场景。
- `Transformer`：自注意力机制的核心，编码解码一手抓，翻译和生成任务轻松搞定。
<!-- more -->

```python
import torch 
from torch import nn, optim

# 假设我们一些预先处理好的电影评论和标签，reviews= ['这部电影很棒','我不喜欢这部电影'] labels= [1,0,...,1] # 1表示正面评论，0表示负面评论
# 将文本转化为数字，这个过程被称为词嵌入(word embedding)，这里我们使用一个简单的方法，将每个词映射到一个唯一的整数。
word_to_index = {'很棒':0, '不': 1, '喜欢':2, '电影': 3, '这部': 4}
index_to_word = {0: '很棒', 1: '不', 2: '喜欢', 3: '电影', 4: '这部'}

# 定义一个简单的神经网络模型
class SentimentClassifier(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        embedded = torch.embedding(text)
        # 将文本转化为词向量
        hidden = torch.relu(self.fc1(embedded))  # 通过一个隐藏层
        
        output = self.fc2(hidden)                # 输出层
        return output
    
# 初始化模型
vocabulary_size = len(word_to_index)
embedding_dim = 10
hidden_dim = 50
output_dim = 1
model = SentimentClassifier(vocabulary_size=vocabulary_size, embedding_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=output_dim)

#定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epoch = 10
for epoch in range(num_epoch):
    for review, label in zip(reviews, labels):
        # 将评论转化为索引
        review_indics = [word_to_index for word in review]
        
        # 小黄建一个批次，这里我们只有一个样本
        review_tensor = torch.tensor(review_indics, dtype=torch.long).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
        
        # 前向传播
        output = model(review_tensor)
        loss = criterion(output, label_tensor)
        
        #反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch [{epoch + 1}/{num_epoch}], Loss: {loss.item()}')
    
# 使用模型进行预测
def predict(review):
    review_indics = [word_to_index[word] for word in review]
    review_tensor = torch.tensor(review_indics, dtype=torch.long).unsqueeze(0)
    output = model(review_tensor)
    prediction = torch.sigmoid(output) >= 0.5
    
    return '正面' if prediction.item() == 1 else '负面'

# 测试模型
test_review = '这部电影很棒'
print(predict(test_review))
```