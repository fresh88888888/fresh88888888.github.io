---
title: GPT与传统模型的区别（PyTorch）
date: 2024-02-21 15:34:32
tags:
  - AI
categories:
  - 人工智能
---

#### GPT与传统模型的区别

在`GPT`之前，已经有多年的自然语言处理（`NLP`）研究和应用，但是传统的模型和方法往往面临一些挑战，例如大量标注数据的依赖、模型的可解释性较差，难以处理复杂的语义关系等。说人话就是得益于预训练无监督技术的发展。`GPT`获得无监督学习和零杨本学习能力主要依赖于预训练（`Pre-training`）技术。预训练技术一种通过对大量无标注文本数据（即无监督数据）进行学习，从而得到一种通用的语言表示能力的方法。在`GPT`中，预训练是预测给定前文（`context`）的下一个词（`target`）实现的，这是一个自回归任务（`Autoregressive task`）。通过在大量无标注文本数据上进行预训练，`GPT`可以学习到语言的语法、语义和上下文信息，从而具备了强大的语言生成和理解能力。

`GPT`的两样本学习能力来自于其预训练的通用性和微调（`Fine-tuning`）的能力。在微调阶段，可以将`GPT`用于具体的任务，例如文本分类、命名实体识别等，这通常需要在有标注的数据集上进行。通过微调，`GPT`可以在特定的任务上获得更好的性能，而不需要从头开始训练模型。这种微调过程类似于“迁移学习”，使得`GPT`可以在不同任务之间共享知识和经验，从而实现零样本学习。然而，以前没有这种技术的主要原因在于数据规模和计算资源的限制。无监督学习需要大量的无标注数据，而零样本学习需要模型具有强大的泛化能力。在`GPT`之前，由于数据规模较小和计算资源有限，很难训练出具有这样能力的模型。随着大数据和云计算的发展，现在可以训练出更大规模和更加强大的模型，这使得无监督学习和零样本学习成为可能。因此，`GPT`的出现标志着自然语言处理领域的一个重大突破。

```python
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class SimpleGPT(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(SimpleGPT, self).__init__()
        
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def forward(self, prompt):
        # 将输入文本转化为模型可以理解的形式
        inputs = self.tokenizer(prompt, return_tensor = 'pt', padding = True, truncation=True)
        inputs = inputs.to(self.device)
        
        # 获取输出
        outputs = self.model(**inputs)
        hidden_states = outputs[0]     #  隐藏层状态，包含文本表示和自回归结果
        
        # 从最后一个token生成一个词（假设我们的任务是根据给定的提示生成写一个词）
        last_token_hidden = hidden_states[:, -1]
        
        # 取最后一个token的隐藏状态
        probs = self.model.get_logits(last_token_hidden)
        
        # 获取生成下一个词的概率分布
        next_word = self.tokenizer.decode([torch.argmax(probs).item()])
        # 根据最大概率选择下一个词并解码
        
        return next_word

```