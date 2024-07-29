---
title: 模型(LLM)基准-探析
date: 2024-07-28 17:25:11
tags:
  - AI
categories:
  - 人工智能
mathjax:
  tex:
    tags: 'ams'
  svg:
    exFactor: 0.03
---

#### 介绍

`LLM`基准如何运作？从本质上讲，`LLM`基准测试遵循一个相当简单的原则：给模型一个任务，看看它表现如何，然后测量评估结果。但是，在评估可靠性方面存在一些细微差别。运行基准测试有以下几种方法：
- **零样本**：模型在没有任何先前示例或提示的情况下接受任务。这展示了其理解和适应新情况的原始能力。
- **少量样本**：在要求`LLM`解决类似任务之前，会先给其一些如何完成任务的示例。这揭示了其从少量数据中学习的能力。
- **微调**：在这种情况下，`LLM`专门针对与基准任务相关的数据进行训练，目的是最大限度地提高其在该特定领域的熟练程度。如果微调有效，它将展示模型在任务中的最佳性能。
<!-- more -->

区分一个`LLM`的表现优于另一个`LLM`是纯粹出于运气还是实际技能差异非常重要。因此，坚持严格的**统计完整性**很重要。鉴于此，在将模型与竞争对手的性能进行基准测试时，必须明确模型是针对**特定任务**部署在**零样本**、**少量样本**还是**微调**能力下。可以使用哪些指标来比较LLM的表现？
- **准确性**(`Accuracy`)：许多基准的基石，这只是大语言模型(`LLM`)完全正确答案的百分比。
- **BLEU分数**(`BLEU Score`)：衡量`LLM`生成的文本与人工编写的参考文献的匹配程度。这对于翻译和创意写作等任务很重要。
- **困惑度**(`Perplexity`)：`LLM`面对任务时表现出的惊讶或困惑程度。困惑度越低，理解能力越好。
- **人工评估**(`Human Evaluation`)：基准非常有用，但有时细微的任务需要专家对`LLM`输出的**质量、相关性**或**连贯性**进行判断。

#### LLM基准

语言很复杂，这意味着需要进行各种测试才能确定大语言模型(`LLM`)的真正能力。以下是一些最常见的基准，用于评估大语言模型(`LLM`)在人工智能的普遍应用中的表现，以及它们的工作原理和用途。
<table>
<caption> `LLM`基准(`Benchmarks`)</caption>
<tr><th>`Category`</th><th>`Benchmark`</th><th>`Descrption`</th></tr>
<tr>
    <td rowspan="3">通用</td>
    <td>`MMLU Chat(0-shot, CoT)`</td>
    <td></td>
</tr>
<tr>
    <td>`MMLU PRO(5-shot, CoT)`</td>
    <td></td>
</tr>
<tr>
    <td>`IFEval`</td>
    <td></td>
</tr>
<tr>
    <td rowspan="2">`Code`</td>
    <td>`HumanEval(0-shot)`</td>
    <td></td>
</tr>
<tr>
    <td>`MBPP EvalPlus(base) (0-shot)`</td>
    <td></td>
</tr>

<tr>
    <td rowspan="2">`Math`</td>
    <td>`GSM8K(8-shot, CoT)`</td>
    <td></td>
</tr>
<tr>
    <td>`MATH(0-sho, CoT)`</td>
    <td></td>
</tr>

<tr>
    <td rowspan="2">推理</td>
    <td>`ARC Challenge(0-shot)`</td>
    <td></td>
</tr>
<tr>
    <td>`GPQA(0-shot, CoT)`</td>
    <td></td>
</tr>

<tr>
    <td rowspan="2">使用工具</td>
    <td>`BFCL`</td>
    <td></td>
</tr>
<tr>
    <td>Nexus(0-shot)</td>
    <td></td>
</tr>

<tr>
    <td rowspan="3">长上下文</td>
    <td>`ZeroSCROLLS/QuALITY`</td>
    <td></td>
</tr>
<tr>
    <td>`InfiniteBench/En.MC`</td>
    <td></td>
</tr>
<tr>
    <td>`NIH/Multi-needle`</td>
    <td></td>
</tr>

<tr>
    <td>多语言</td>
    <td>`Multilingual MGSM(0-shot)`</td>
    <td></td>
</tr>
</table>