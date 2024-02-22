---
title: CNN与MLP的区别（PyTorch）
date: 2024-02-22 16:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### CNN与MLP的区别

`CNN`与`MLP`的区别：
- `CNN`（卷积神经网络）：就像一个厉害的画家，它通过扫描局部特征（比如边缘、纹理）来识别图片。特备擅长看图说话，由“局部观察员”卷积层找到关键线索，接着用池化层浓缩信息，最后全连接层整合所有线索判断画中内容类别。`CNN`的强项在于处理图像和视频这类局部细节丰富的数据。
- `MLP`（多层感知机）：则仿佛是一位音乐大师，专注于倾听旋律与节奏的全局特征，从而识别音乐风格。`MLP`在处理文本和声音这类整体模式重要的数据时独具匠心。它的结构就像层层堆叠的乐团，每层神经元都跟前一层相连，共同构建复杂映射关系。总结一下，`CNN`是图像处理专家，专攻局部特征；`MLP`则是文本和音频的理解高手，聚焦全局特征。
<!-- more -->

`CNN`模型：
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定义超参数
input_size = 784       # 输入图像的维度
hidden_size = 500      # 隐藏层神经元数量
num_classes = 10       # 输出类别的数量（0~9）
num_epochs = 5         # 训练轮数
batch_sise = 100       # 批处理大小
learning_rate = 0.001  # 学习率


# 数据预处理: 将图像转化为张量，并归一化到[0，1]区间
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

tarin_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_sise, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_sise, shuffle=True)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forwar(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 实例化模型、损失函数和优化器
model = CNN(input_size, hidden_size, num_classes)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# SGD优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tarin_loader):
        # 前向传播，得到预测输出
        outputs = model(images)
        # 计算损失值
        loss = criterion(outputs, labels)
        # 清空梯度缓存（因为PyTorch会累积梯度）
        optimizer.zero_grad()
        # 反向传播，计算梯度值
        loss.backward()
        # 根据梯度执行权重（执行优化步骤）
        optimizer.step()
        if (i + 1) % 100 == 0:
            # 每100个Batch打印一次损失值和准确率
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(tarin_loader)}], Loss: {loss.item():.4f}')
            

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

print('Training finshied.')
```

`MLP`模型：
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forwar(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据预处理
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype= torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# 构建词汇表
MAX_VOCAB_SIZE = 64
TEXT.build_vocab(train_data, MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

# 数据加载器
batch_size = 64
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = batch_size,
    device = device,
)

# 初始化模型、损失函数和优化器
model = MLP(input_dim=TEXT.vocab.vectors.shape[1], hidden_dim=512, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for texts, labels in enumerate(train_iterator):
        # 清空梯度缓存（因为PyTorch会累积梯度）
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs.squeeze(), labels.float())
        # 反向传播，计算梯度值
        loss.backward()
        # 根据梯度执行权重（执行优化步骤）
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for texts, labels in test_iterator:
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
```