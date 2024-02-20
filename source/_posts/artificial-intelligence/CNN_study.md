---
title: CNN神经网络（PyTorch）
date: 2024-02-19 10:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 卷积神经网络（CNN）用于图像识别

需要根据实际情况填充假设的参数（如图片的通道数、大小、类别数等），并且在真实应用中还需要进行模型训练，验证以及保存加载登步骤。同时，在使用前需要导入必要的库，并准备相应的训练和测试数据集。
<!-- more -->

```python
import torch
import torch.nn as nn
import torch.optim as optim

class ImageClassifier(nn.Module):
    def __init__(self,num_classes):
        super(ImageClassifier, self).__init__()
        
        # 卷积层块
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 全连接层用于分类
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 32 * 32)
        x = self.fc_layers(x)
        return x

# 实例化模型
model = ImageClassifier(num_classes=4)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 查看模型摘要信息
print(model)
```
结果输出为：
```bash
ImageClassifier(
  (conv_layers): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (4): ReLU()
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (fc_layers): Sequential(
    (0): Linear(in_features=1024, out_features=64, bias=True)
    (1): ReLU()
    (2): Dropout(p=0.5, inplace=False)
    (3): Linear(in_features=64, out_features=4, bias=True)
  )
)
```

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
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                  1, num_epochs, i + 1, len(tarin_loader), loss.item()))

print('Training finshied.')
```