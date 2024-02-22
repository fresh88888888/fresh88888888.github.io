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