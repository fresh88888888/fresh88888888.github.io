---
title: Transformer（PyTorch）
date: 2024-02-22 09:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### 多模态融合

多模态融合是指将来自不同感知渠道（如图像、文本、语音、视频等）的数据信息整合起来，共同进行分析和理解的过程，传统的多模态融合方法可能无法有效地捕获跨模态之间的依赖关系，而`Transformer`通过自注意力机制可以灵活地捕捉到不同模态特征间的相关性，并且能够根据不同模态输入的重要性动态调整权重，实现高效的信息交互与融合。简答来说，就是将不同类型的数据“混合搅拌”，从而得到更丰富、更深入的信息。比如，把图片和文字放到一起，让机器自己找联系。在这个过程中，不同模态的数据就像是不同口味的食材，还能快速的烹饪出美味。它通过学习和转换不同模态的数据，让多模态融合变得更加简单、高效。未来，随着技术的进步，多模态融合将在更多领域大放异彩。
<!-- more -->

```python
    import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

visual_input_size = 10
textual_input_size = 20
# 假设我们有两个模态，每个模态有各自的特征提取器提取出固定长度的向量表示
class ModelFeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModelFeatureExtractor, self).__init__()
        
        self.model_net = nn.Linear(input_size, output_size)
        
    def forward(self, model_input):
        return self.model_net(model_input)
    
# 定义多模态融合模块，这里使用了共享的Transformer编码器层
class MultiModelFusionTransformer(nn.Module):
    def __init__(self, model_size, fusion_dim, num_heads, num_layers, dropout):
        super(MultiModelFusionTransformer, self).__init__()
        
        # 分别为两个创建特征提取器
        self.visual_extractor = ModelFeatureExtractor(visual_input_size, fusion_dim)
        self.textual_extractor = ModelFeatureExtractor(textual_input_size, fusion_dim)

        # 创建一个共享的Transformer编码器核心
        fused_representation = self.trnsformer_encoder(combined_features)
        
        return fused_representation

if __name__ == '__main__':
    # 假设我们有一些预处理后的视觉和文本数据
    visual_data = torch.randn((batch_size, visual_seq_length, visual_feature_size))
    textual_data = torch.randn(
        (batch_size, textual_seq_length, textual_feature_size))

    model = MultiModelFusionTransformer(
        model_size=(visual_feature_size, textual_feature_size),
        fusion_dim = 256,    # 融合的维度大小
        num_heads = 8,       # 自注意力头数
        num_layers = 2,      # 编码器层数
        dropout=0.1          # 正则化dropout比例
    )
    
    # 计算融合后的特征
    fused_representations = model(visual_data, textual_data)
```
#### Transformer介绍

而`Transformer`就像是一个超级智能的读者，它能同时看到整本书的所有词句，并且知道哪些词句之间最重要、最相关。它通过“自注意力机制”来实现这一点，这种机制让模型可以聚焦于文本的不同部分，并考虑整体上下文信息，而不是顺序地一次处理一个词。简单来说，`Transformer`能够更高效并行地处理整个句子和段落，决定每个词在生成最终结果时的重要程度。由于这个特性，`Transformer`在翻译、问答系统、文本分类等各种自然语言处理任务中表现出色，并且是`BERT、GPT`等众多NLP模型的基础架构。

```python
import torch
import torch.nn as nn


# 定义一个自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 对输入进行线性变化已创建查询键和值向量
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.ke_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # 将嵌入维度分成多个头的大小
        self.num_heads = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        
        #初始化 归一化系数
        self.scaling = self.head_dim ** -0.5
    
    def forward(self, x):
        query = self.query_proj(x)
        key = self.ke_proj(x)
        value = self.value_proj(x)
        
        # 进行矩阵乘法和分隔
        query = query.reshape(-1, x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(-1, x.shape[1], self.num_heads,
                              self.head_dim).transpose(1, 2)
        value = value.reshape(-1, x.shape[1], self.num_heads,
                              self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重加权求和得到上下文向量
        context = torch.matmul(attention_weights, value).transpose(1, 2).reshape(-1, x.shape[1], self.embed_dim)
        
        return context
    
# 定义一个简单的Transformer编解码器
class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleTransformerEncoderLayer, self).__init__()
        
        self.self_attntion = SelfAttention(embed_dim, num_heads)
        
        #在自注意力后还有一个前馈神经网络（FFN）
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
        #注意力后的残差层和归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, inputs):
        #自注意力部分
        attn_output = self.self_attntion(inputs)
        out1 = self.norm1(inputs + attn_output)
        
        # 前馈神经网络部分
        ffn_output = self.feed_forward_network(out1)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2

# 创建一个实例并应用到一些随机数据上
embed_dim = 512
num_heads = 8
model = SimpleTransformerEncoderLayer(embed_dim, num_heads)
# 假设我们有10个样本，每个样本包含20个特征的512维嵌入向量
inputs = torch.randn(10, 20, embed_dim)
output = model(inputs)
```

#### Transformer的编/解码器

`Transformer`模型的编码器和解码器是用来处理序列数据（比如自然语言文本）的核心组件。
- 编码器：想象一下你正在阅读一本外文书，你需要把它翻译成你的母语。编码器就像是一个非常聪妈且能同时理解整段话含义的读者。它会逐个读取输入句子中的每个单词，并生成一个“上下文感知”的向量表示。这个过程不是顺序但不执行的，而是让每个单词都能关注到句子中其他所有单词，从而获得整个句子的全局信息。具体来说，编码器通过多头自注意力机制实现这一点，即每个单词都计算与它单词之间的权重关系，然后综合这些关系形成自己的表示。
- 解码器：接收到编码器输出的整个源语言句子的山下文向量后，解码器的任务就像一个能够根据上下预测下一个单词的翻译者。但它不能看到完成的待翻译目标句子，而是在生成每个目标词时及参考已生成的部分和源语言的上下文信息。解码器也有一个多头自注意力层，它不仅考虑了已生成的目标词汇，还通过所谓的“编码器-解码器注意力”机制获取源语言的信息。这样，解码器可以逐步生成翻译后的目标句子。

```python
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn= MultiHeadAttention(d_model, n_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self_fc = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self_dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # 输入src是经过位置编码后的词嵌入序列
        attn_output = self.self_attntion(src, src, src, attn_mask=atten_mask)
        attn_output = self.norm1(src + attn_output)
        out1 = self.norm1(src + attn_output)   # 残差连接归一化
        
        fc_out = self.fc(out1)
        fc_out = self.dropout2(fc_out)
        out2 = self.norm2(out1 + fc_out)       # 再次残差连接归一化
        
        return out2

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn= MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        # 注意力后的残差层和归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self_fc = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory, tgt_mask=None):
        # tgt 是待解码的目标序列词嵌入memory是编码器的输出，包含源语言的上下文信息
        self_attn_output, _ = self.self_attntion(tgt, tgt, tgt, attn_mask=tgt_mask)
        self_attn_output = self.dropout1(self_attn_output)
        out1 = self.norm1(tgt + self_attn_output)   # 目标自注意力部分的残差连接和归一化

        enc_dec_attn_output, _ = self.enc_dec_attn(out1, memory, attn_mask = memory_mask)
        enc_dec_attn_output = self.dropout2(enc_dec_attn_output)
        out2 = self.norm2(out1 + enc_dec_attn_output)  # 编码器-解码器注意力部分的残差连接和归一化
        
        fc_out = self.fc(out2)
        fc_out = self.dropout3(fc_out)
        out3 = self.norm3(out2 + fc_out)  # 全连接层后的残差连接和归一化

        return out3
```

#### 自注意力机制

自注意力机制，你可以把它理解为一种“内部关注”机制。在处理复杂的信息时，比如机器翻译、文本生成等任务，模型需要理解并权衡输入信息中的不同部分。想象一下，你在阅读一句话或一段文章时，会根据当前的语境，对某些词语或短语给予更多关注，这些往往对理解整体意思至关重要。自注意力机制就是让机器模仿这种“关注”能力。具体来说，在神经网络模型中，每个输入元素（如一个单词）不仅可以获得自身的编码信息，还可以通过自注意力机制了解到其他所有输入元素的信息，并根据它们的重要性分配不同的权重，从而更好地理解和处理整个序列信息。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init(self, embed_size):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.liner = nn.Linear(embed_size, embed_size)
        self.gmma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs):
        # 计算query、key、value
        query = self.liner(inputs);
        key = inputs
        value = inputs
        
        # 计算query和Key之间的相关性得分
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = self.gmma + scores    # 添加gmma用于缩放，使其落在合适的范围内
        
        # 对相关性得分应用softmax函数，得到Attention权重
        attention_weights = self.softmax(scores)
        
        # 使用attention权重和value计算输出
        output = torch.bmm(value, attention_weights.transpose(1, 2))
        
        return output  # 返回注意力机制的输出结果
```

#### 多头注意力

想象一下有个超级助手，它不只有一双眼睛，而是有好几双，每双眼睛都负责从不同视角解读一段话。在AI界，Transformer就像这个超级助手，用“多头注意力”来处理文本。简单来说，就是模型不再单一视角分析句子，而是分成多个小分队（即“头”），每个小分队专门研究句子的不同方面，比如一个关注词与词之间的语法关系，另一个则关注于词语背后的意思。这样一来，不仅能够多样化的理解输入信息，还因为这些小分队可以并行工作，大大提高了计算速度和学习效率，让模型更能把握住复杂的语言结构。

#### 微调常用方法
- 冻结参数：相当于只换家具，房屋框架不变，也就是只更新最后几层参数，让模型快速适应新任务，同时留住通用智慧
- 精选训练层：有选择性地翻新部分楼层（比如顶层房间）针对性更新这些层的参数，确保模型贴合特定任务需求
- 差异化学习率：不同楼层施工进度各异，顶层动作快一些（大学习率）。底层稳扎稳打（小学习率）。这样既能稳住大局，又能迅速学会新技能。

#### 为什么大模型采用decoder-only结构

如果我们要训练一个机器翻译模型，解码器-独占结构只需要输入原文，然后输出目标语言的译文。整个过程就是一个解码的过程，非常直接。相比之下，其他结构可能需要更多的训练数据和计算资源。而且解码器-独占结构只需要对输入的句子一次处理，就能得到完整的译文。这样在处理大量数据时，它的效率会更高。总的来说，解码器-独占结构之所以被广泛应用于大模型中，是因为它简单、高效，而且能够处理多种语言任务。

```python
import torch
import torch.nn as nn

# 假设我们有一个词表，大小(vocab_size)
vocab_size = 10000

# 定义Decoder隐藏层维度
hidden_size = 512

# 定义自注意力机制所需要的参数
num_attention_heads = 8
attention_dropout = 0.1

#Decoder类定义
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # 自注意力子层
        self.self_attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=attention_dropout)
        
        # 前馈神经网络子层
        self.feedforward_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        # 层归一化和残差连接
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout用于减少过拟合
        self.dropout = nn.Dropout(0.1)
        
        def forward(self, input_tokens):
            # 自注意力步骤
            query, key, value = input_tokens, input_tokens, input_tokens
            attention_output, _= self.self_attention(query, key, value, mask)
            attention_output, _ =self.dropout(attention_output)
            out1 = self.norm1(attention_output + input_tokens)       # 残差连接
            
            # 前馈神经网络处理
            ff_output = self.feedforward_network(out1)
            out2 = self.norm2(ff_output + out1)

            return out2

# 创建一个decoder实例
decoder = Decoder()

# 假设我们有随机初始化的输入
input_seq = torch.randn(32, 64, hidden_size)

# 运行Decoder
output_seq = decoder(input_seq)

```

#### 深度可分离卷积

深度可分离卷积就是那个让手机里的美颜相机更快更省电的秘密武器，传统的卷积神经网络就像个大胃王，每个滤镜（卷积核）都要跟照片上每一点颜色仔仔细细地“过招”，虽然效果好但太费劲了。深度可分离卷积把繁重任务一分为二：先是“挑通道专家”出场，一个滤镜只和一种颜色单独打交道，大大减轻计算负担；接着是“融合大师”接手，用小巧的标准卷积给各个通道调调亮度、加加魔法调料，最后生成美美的输出结果。
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义深度可分离卷积层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channles, out_channles, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 创建深度卷积层，每个卷积核与输入特征图的一个通道进行计算
        self.depthwise_conv = nn.Conv2d(in_channles, in_channles, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channles)
        
        # 创建逐点卷积层，对深度卷积的输出进行独立的缩放和偏移
        self.pointwise_conv = nn.Conv2d(in_channles, out_channles, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # 首先进行蛇毒卷积
        x_depthwise = self.depthwise_conv(x)
        # 然后进行逐点卷积
        x_pointwise = self.pointwise_conv(x_depthwise)
        
        return x_pointwise
    
# 创建一个简单的深度可分离卷积网络
class SimpleSeparableNet(nn.Module):
    def __init__(self, input_channles=3, out_channles=64, kernel_size=3, stride=1, padding=1):
        super(SimpleSeparableNet, self).__init__()

        # 首先应用深度可分离卷积
        self.ds_conv = DepthwiseSeparableConv(input_channles, out_channles, kernel_size = kernel_size, stride=stride, padding=padding)
        
        # 然后接一个ReLU激活函数
        self.relu = nn.ReLU
        
        # 最后接一个池化层，这里用的是2x2的最大池化
        self.pool = nn.MaxPool2d(kernel_size= 2, stride=2)
        
    def forward(self, x):
        # 先通过深度可分离卷积
        x = self.ds_conv(x)
        # 然后通过ReLu激活函数
        x = self.relu(x)
        # 最后通过池化层
        x = self.pool(x)
        
        return x

#创建深度可分离卷积网络
model = SimpleSeparableNet()

# 创建一个随机张量作为输入
input_tensor = torch.randn(1,3,32,32)

# 通过网络前向传播输入张量
output_tensor = model(input_tensor)
```

#### 多模态融合学习

多模态融合学习，就是让计算机向我们一样，用多种感官理解世界。不再是单一的图片或文字，而是文字、图片、声音、视频等多种数据一起上。`AI`通过学习这些不同感官输入的关联，能够更全面、更准确的理解事物。比如，`AI`可以一边分析图片，一边阅读描述，更准确地识别图片里的物体；或者结合视频图像和音频，理解视频内容，这种技术能大大提升机器学习性能，让`AI`更好地理解我们的语言、情感和环境，交互体验更智能、更自然。实现多模态融合学习只需几步：
- 准备多种类型的数据，比如图片和文本
- 设计一个能处理多种数据的模型架构
- 用多模态数据训练模型，定义损失函数，优化算法
- 用新数据评估模型性能
- 应用模型，比如自动标记图片物体，理解视频场景

`Pytorch`提供了丰富的内置功能和扩展库，让多模态模型的构建、训练和部署变得轻松简单。
<!-- more -->

```python
import torch
import torchvision
import pandas as pd
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn, optim

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 定义图像数据集
image_data = datasets.ImageFolder('/path/to/image_data', transform=transform)
image_loader = DataLoader(image_data, batch_size=64, shuffle=True)

# 定义文本数据集（这里假设文本数据已经是一个Pandas DataFrame, 列名为'text'和'label'）
text_data= pd.read_csv('/path/to/text_data.csv')
text_loader = DataLoader(text_data, batch_size=64, shuffle=True)

# 定义一个多模态模型


class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        
        # 定义文本模型
        self.tex_model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 定义图像模型
        self.image_model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, text_data, image_data):
        # 文本模型前向传播
        text_output = self.tex_model(text_data)
        
        # 图像模型前向传播
        image_output = self.image_model(image_data)
        
        # 输出拼接
        output = torch.cat((text_data, image_data), 1)
        
        return output
    
# 实例化模型、损失函数和优化器
model = MultimodalModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):
    for text_batch, image_batch, labels in zip(text_loader, image_loader , labels):
        # 前向传播
        outputs = model(text_batch, image_batch)
        
        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch[{epoch + 1}/10], Loss: {loss.item():.4f}')
```

# Transformer的核心

首先，我们需要需要理解Transformer模型的核心构造。该模型主要包括两部分：自注意力机制和位置编码。这两部分的设计，是的模型可以理解和处理输入数据中的长距离依赖关系，这在传统模型中是一个挑战。
- 自注意力机制：让每个输入的位置都能够考虑全局的信息，从而更好滴理解上下文信息。这意味着，当我们要识别一个词的含义时，它不仅仅考虑前面的词，还会考虑到后面的词，甚至是远离他的词，这样使得整体的信息被全面捕获。
- 位置编码：则解决了Transformer模型中的一个问题，由于没有明确的层级结构，它如何理解词语顺序和位置信息？通过给每个位置一个独特的编码，模型可以知道每个词语在句子中的位置，从而更准确地处理和理解输入数据。

在各个领域中，这种强大的理解和处理能力使得Transformer模型表现得非常出色。例如，在自然语言处理任务中，他可以更好地理解和生成语言；在语音识别中，它可以更好地捕捉语音的节奏和音高；在图像识别中，他可以更好地理解图像中的场景和物体。所以，简单来说，Transformer模型之所以先进，是通过自注意力和位置编码的设计，实现了对输入数据的强大理解和处理能力，从而在各个领域中都表现出了卓越的性能。

```python
class TransformerModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, num_classes, nhead, nhid, nlayers, dropout):
        super(TransformerModel, self).__init__()
        
        # 嵌入层，将词汇表中的词转换为固定维度的向量
        self.encoder = nn.Embedding(vocabulary_size, embedding_dim=embedding_dim)
        # Transformer结构， 包含自注意力机制和位置编码
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=nhead, num_encoder_layers =nlayers, num_encoder_layers=nlayers, dropout=dropout)
        # 输出层，将Transformer的输出转化为类别概率分布
        self.decoder = nn.Linear(embedding_dim, num_classes)
        
        # 初始化权重参数
        self.init_weights()
    
    def init_weights(self):
        '''
        初始化权重参数
        '''
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src):
        '''
        前向传播过程
        '''
        # 将输入数据嵌入到固定维度的向量中
        embedded = self.encoder(src)
        
        # 通过Transformer结构进行自注意力和位置编码处理
        output = self.transformer(embedded)
        
        # 将Transformer的输出转化为类别概率分布
        output = self.decoder(output)
        
        return output   # [batch_size, num_classes]
```