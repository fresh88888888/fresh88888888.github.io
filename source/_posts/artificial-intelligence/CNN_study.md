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
        
        # 卷积层快
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