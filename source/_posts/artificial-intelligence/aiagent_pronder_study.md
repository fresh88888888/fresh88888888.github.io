---
title: AI Agent — 思考
date: 2024-04-07 11:45:11
tags:
  - AI
categories:
  - 人工智能
---

`AI`的发展方向我们认为主要有两个：一个是有趣、亲和、更像人的`AI`；另一个是有用、严谨、更像工具的`AI`。`AI`更像工具还是更像人呢？其实有很多争议。就技术的发展阶段来说：目前很长一段时间应该是”更像工具的`AI`“，未来的某个时间节点会朝着”更像人的`AI`“方向发展。
<!-- more -->

`AI`除了有趣和有用这个维度之外，还有另外一个维度，就是**快思考和慢思考**。**所谓快思考就是不需要过脑子的基础视觉、听觉等感知能力和说话等表达能力**，像`ChatGPT`、`stable diffusion`这种一问一答、解决特定问题的`AI`可以认为是一种工具向的快思考。你不问它问题的时候，它不会主动去找你。而 `Character AI`、`Inflection Pi`和`Talkie`（星野）这些`AI Agent`产品都是模拟一个人或者动漫游戏角色的对话，但这些对话不涉及复杂任务的解决，也没有长期记忆，因此只能用来闲聊。**而慢思考就是有状态的复杂思考，也就是说如何区规划和解决一个复杂的问题，先做什么、后做什么**。比如`MetaGPT`写代码是模拟一个软件开发团队的分工合作，`AutoGPT`是把一个复杂任务拆分成很多个阶段来一步步完成，虽然这些系统在实用中还有很多问题，但已经是一个具备慢思考能力的雏形了。遗憾的是，现有产品中几乎没有兼具**慢思考和类人属性**的`AI Agent`。有趣的是，科幻电影里面的`AI`其实大部分是兼具**慢思考和类人属性**的。因此这就是目前`AI Agent`和人类梦想之间的差距。

因此我们在做的事情跟`Sam Altman`说的正好相反，我们希望让`AI`更像人，同时又具备慢思考的能力，最终演进成一个数字生命。什么是`AGI`呢？我觉得它又需要有趣，又需要有用。有趣的方面，就是它需要能够有自主思考的能力、有自己的个性和感情。而有用的方面，就是`AI`能够解决工作、生活中的问题。现在的`AI`要么是只有趣但没用，要么是只有用但是不有趣，不像人。

我认为未来真正有价值的`AI`，是由算力及资源决定的。有单个资源和有限算力的`AIOS`（`AI`操作系统）和多个资源和接近无限算力的分布式`AIOS`。AIOS可以帮助个人/家庭解决很多生活中、部分工作中的问题，比传统操作系统做的又快又好，用户体验更棒，同时它有记忆、有感情、有意识。分布式`AIOS`更像是具有AI能力的超级计算机，它的算力和资源接近于无限。它能解决单个AIOS无法解决的复杂工作任务。面向的客户群体不一样，一个面向个人/家庭、一个面向企业/政府。它们的唯一区别是算力和资源的规模不同。

那么我们首先来看一看如何去构建一个真正有趣的`AI`。有趣的`AI`我认为就像一个有趣的人，可以分为好看的皮囊和有趣的灵魂这两个方面。**好看的皮囊就是它能够听得懂语音，看得懂文本、图片和视频，有这样一个视频、语音的形象，能够跟人实时交互**。**有趣的灵魂就是它需要像人一样能够去独立思考，有长期记忆，有自己的个性**。

##### 好看的皮囊：多模态理解能力

说到好看的皮囊，很多人认为只要有一个`3D`的形象能够在这儿摇头晃脑地展示就行了。但是我认为更关键的一部分是`AI`能够去看到，并且理解周围的世界，就是他的视觉理解能力是很关键的，不管是机器人还是可穿戴设备，还是手机上的摄像头。我认为，多模态大模型有三条路：
- 第一条是用多模态数据端到端预训练的模型。
- 第二条是输入的图片、语音、视频分别通过不同的`encoder`去做编码，编码结果经过`projection layer`映射到`token`，输入给`Transformer`大模型。大模型的输出`token`经过`projection layer`，分别映射到图片、语音、视频的解码器，这样就可以生成图片、语音、视频了。
- 第三条是连`projection layer`都不要了，直接用文本去粘接`encoder`、`decoder`和文本大模型，不需要做任何训练。例如语音部分就是先做语音识别，把语音转换成文字输入给大模型，然后再把大模型的输出送给语音合成模型生成音频。不要小看这种听起来很土的方案，在语音领域，目前这种方案还是最靠谱的，现有的多模态大模型在识别和合成人类说话语音方面都不太行。

##### 好看的皮囊：多模态生成能力

如果真的打算把语音作为一个用户体验的重大加分项，基于开源自研语音模型不仅是必要的，也是可行的。我们知道图片生成现在已经比较成熟，视频生成会是`2024`年一个非常重要的方向。有几条典型的技术路线，比如`Live2D`，`3D`模型，`DeepFake`，`Image Animation`和`Video Diffusion`。目前`AI`很难自动生成`Live2D`和`3D`模型，这还需要基础模型的进步。因此`AI` 能做的事就是在输出中插入动作提示，让3D模型一边说话一边做指定的动作。

`DeepFake、Image Animation`和`Video Diffusion`则是通用视频生成`3`条不同的技术路线。
- **DeepFake 是录制一个真人视频，随后利用 AI 把视频中的人脸换成指定的人脸照片**。这种方法其实也是基于上一代深度学习的方法，它从`2016`年开始就存在了。
- `Image Animation`，比如说最近比较火的阿里通义千问的`Animate Anyone`或者字节的`Magic Animate`，它实际上是给定一张照片，随后根据这张照片生成一系列的对应视频。然而，这个技术相比于`DeepFake`的缺点是它可能目前还达不到实时视频生成，而且视频生成的成本相比`DeepFake`要高一些。但是`Image Animation`可以生成大模型指定的任意动作，甚至可以把图片背景填充进去。当然，不管是`DeepFake`还是`Image Animation`生成的视频，都不是完全准确，有时候可能发生穿帮的情况。
- `Video Diffusion`我认为是一个更为终极的技术路线。虽然这条路线现在还不够成熟，比如像`Runway ML` 的`Gen2`，以及`PIKA Labs`都在探索这一领域。我们认为，可能未来基于 `Transformer`的方式端到端的生成视频是一个终极的解决方案，可以解决人和物体的运动以及背景生成的问题。我认为视频生成的关键是要对世界有一个很好的建模和理解。现在我们的很多生成模型，比如`Runway ML`的`Gen2`，在对物理世界的建模方面实际上存在很大的缺陷。许多物体的物理规律和其物理属性并不能被正确地表达出来，因此它生成的视频的一致性也较差，稍微长一点的视频就会出现问题。同时，即使是非常短的视频，也只能生成一些简单的运动，而对于复杂的运动，是没办法正确建模的。此外，成本也是一个大问题，现在`Video Diffusion`的成本是所有这些技术中最高的。因此，我认为`Video Diffusion`是`2024`年一个非常重要的方向。我相信，只有当`Video Diffusion`在效果足够好的同时，成本也大幅降低，每个`AI`的数字分身才真的能拥有自己的视频形象。

除了基于`prompt`的方式之外，在构建人物个性方面我们还有一种更好的方法，就是基于微调的`agent`。我们用来做微调的语料可以大致分为对话性语料和事实性语料两类。对话性语料包括像 `Twitter`、聊天记录等，往往是第一人称的，主要是用来微调人物的个性和说话的风格。而事实性语料包括`Wikipedia`上关于他的页面、关于他的新闻以及博客等，往往是第三人称的，这些可能更多的是关于这个人物事实性的记忆。这里就有一个矛盾，就是如果只用对话性语料去训练，他可能只能学到该人的说话风格和思维方式，但学不到关于他的很多事实性记忆。但如果只用事实性语料训练，又会导致其说话风格像是写文章的人的风格，而不是本人的说话风格。

那么如何平衡这两者呢？我们采用了一个两步训练的方法。第一步，我们先用对话性语料去微调他的个性和说话风格。第二步，再去把事实性语料进行数据清洗后，基于各种角度提问，生成这个人物第一人称口吻的回答，这叫做数据增强。用这种数据增强之后生成的回答，再去微调人物的事实记忆。也就是说，所有用来微调事实记忆的语料都已经以第一人称的口吻组织成了问题和回答对。这样也解决了微调领域的另一个问题，即事实性语料往往是长篇文章，而长篇文章不能直接用来做微调，只能用来做预训练。

##### 有趣的灵魂：慢思考与记忆

要解决这些问题需要一个系统的解决方案，关键就是一个慢思考。我们开头就讲过，慢思考是神经科学的一个概念，区别于基础的感知、理解、生成这些快思考能力。**我们前面提到 “好看的皮囊” 里面这些多模态的能力，可以认为是快思考。而 “有趣的灵魂” 更多需要慢思考**。

我们可以思考一下，人类是如何感觉到时间流逝的？有一种说法认为，时间流逝感源自工作记忆的消逝。另一种说法认为，时间流逝感源自思考的速度。我认为这两种说法都是对的。这也是大模型思考的两个本质问题：记忆（`memory`）和自主思考（`autonomy`）。人的工作记忆只能记住 7 项左右的原始数据，其余数据都是整理后储存，再进行匹配和提取。今天的大模型`attention`是线性的，上下文不管多长，都是线性扫描，这不仅效率低下，也难以提取逻辑深度较深的信息。

**人类的思考是基于语言的**。《人类简史》认为语言的发明是人类区别于动物最明显的标志，因为只有基于复杂的语言才可能进行复杂的思考。我们在大脑中没有说出来的话，就像大模型的 `Chain-of-Thought`（思维链），是思考的中间结果。大模型需要`token`来思考，而`token`就像是大模型的时间。

其中的第一个问题就是长期记忆。其实我们应该庆幸大模型帮我们解决了短期记忆的问题。上一代的模型，比如基于 BERT 的那些模型，很难理解上下文之间的关联。当时一个指代问题就很难解决，搞不清楚 “他” 说的是谁，“这个” 指的是哪个东西。表现出来就是，前面几个回合告诉`AI`的东西，后面几个回合就忘了。基于`Transformer`的大模型是首个根本上解决上下文之间语义关联的技术，可以说是解决了短期记忆的问题。但`Transformer`的记忆是用`attention`实现的，受限于上下文长度。超出上下文的历史只能丢掉。那么超出上下文的长期记忆怎么解决？学界有两条路线，一条是长上下文，就是把上下文支持到`100K`甚至无限大。另一条是`RAG`和信息压缩，就是把输入的信息总结整理之后再压缩存储，需要的时候只提取相关的记忆。

在当前技术条件下，长期记忆我认为关键是个信息压缩的问题。我们不追求在几十万字的输入中大海捞针，像人类一样的记忆可能就足够了。目前大模型的记忆就是聊天记录，而人类记忆显然不是用聊天记录的方式工作的。大家正常聊天的时候不会不停地在那儿翻聊天记录，而且人也记不住聊过的每一个字。一个人真正的记忆应该是他对周围环境的感知，不仅包括别人说的话、他说的话，还包括他当时想了什么。而聊天记录里面的信息是零散的，不包含人自己的理解和思考。比如别人说了一段话我可能被激怒可能不被激怒，但人是会把当时是否被激怒了这个心情记忆下来的。如果不做记忆，每次都根据原始聊天记录去推断当时的心情，那可能每次推出来的都不一样，就可能发生前后不一致的问题。

长期记忆实际上有很多的东西可以做。记忆可以分为事实性的记忆和程序性的记忆。事实性记忆比如我们第一次是什么时候见面的，程序性记忆比如个性以及说话风格。前面讲到人物角色微调的时候也提到了对话性语料和事实性语料，对应的就是这里的程序记忆和事实记忆。事实性记忆里面也有多种方案，比如总结、`RAG`和长上下文。总结就是信息压缩。最简单的总结方法是文本总结，也就是把聊天记录用一小段话总结一下。**更好的方法是用指令的方式去访问外部存储**。还有一种方法是在模型层面上用`embedding`做总结，比如`LongGPT`这个工作，目前主要是学术界在研究，实用性没有 `MemGPT`和文本总结强。

大家最熟悉的事实性记忆方案可能是`RAG（Retrieval Augmented Generation）`了。`RAG`就是搜索相关的信息片段，再把搜索结果放到大模型的上下文里，让大模型基于搜索结果回答问题。很多人说`RAG`就等于向量数据库，但我认为`RAG`背后一定是一整套信息检索系统，`RAG`一定不是向量数据库这么简单。因为大规模语料库仅仅使用向量数据库的匹配准确率是非常低的。向量数据库比较适合语义匹配，传统的`BM25`之类基于关键词的检索比较适合细节匹配。而且不同信息片段的重要程度不同，需要有个搜索结果排序的能力。

长上下文前面已经提到了，可能是一种终极方案。如果长上下文结合持久化`KV Cache、KV Cache`的压缩技术和一些`attention`的优化技术，可以做到足够便宜，那么只要把所有对话的历史和`AI` 当时的思考和心情记录下来，就可以实现一个记忆力比人还好的`AI Agent`。但是有趣的`AI Agent`记忆力如果太好，比如能清楚的记得一年前的早上吃了什么，会不会显得不太正常，这就是需要产品设计方面思考了。

这三种技术也不是互斥的，它们是互相补充的。比如说总结和`RAG`就是可以结合在一起的，我们可以分门别类的做总结，对每一次聊天做总结，一年下来这些总结也会有很多内容，需要`RAG`的方法提取有关的总结，作为大模型的上下文。

##### 有用的 AI

**有用的AI其实更多是一个大模型基础能力的问题，比如复杂任务的规划和分解、遵循复杂指令、自主使用工具以及减少幻觉等等，并不能通过一个外部的系统简单解决**。比如`GPT-4`的幻觉就比 `GPT-3.5`少很多。区分哪些问题是模型基础能力问题，哪些问题是可以通过一套外部系统来解决的，也是很需要智慧的。其实有一篇很著名的文章叫做`The Bitter Lesson`，它讲的是凡是能够用算力的增长解决的问题，最后发现充分利用更大的算力可能就是一个终极的解决方案。`Scaling law`是`OpenAI`最重要的发现，但是很多人对`Scaling law`还是缺少足够的信仰和敬畏之心。

要搞清楚大模型适合做什么，我们需要先想清楚一点：有用`AI`的竞争对手不是机器，而是人。工业革命里面的机器是取代人的体力劳动，计算机是取代人的简单重复脑力劳动，而大模型则是用来取代人更复杂一些的脑力劳动。所有大模型能做的事情，人理论上都能做，只是效率和成本的问题。因此，要让`AI`有用，就要搞清楚大模型到底哪里比人强，扬长避短，拓展人类能力的边界。

比如，**大模型阅读理解长文本的能力是远远比人强的**。给它一本几十万字的小说或者文档，它几十秒就能读完，而且能回答出`90%`以上的细节问题。这个大海捞针的能力就比人强很多。那么让大模型做资料总结、调研分析之类的任务，那就是在拓展人类能力的边界。`Google`是最强的上一代互联网公司，它也是利用了计算机信息检索的能力远比人强这个能力。

再如，**大模型的知识面是远比人广阔的**。现在不可能有任何人的知识面比`GPT-4`还广，因此`ChatGPT`已经证明，通用的`chatbot`是大模型一个很好的应用。生活中的常见问题和各个领域的简单问题，问大模型比问人更靠谱，这也是在拓展人类能力的边界。很多创意性工作需要多个领域的知识交叉碰撞，这也是大模型适合做的事情，真人因为知识面的局限，很难碰撞出这么多火花来。但有些人非要把大模型局限在一个狭窄的专业领域里，说大模型的能力不如领域专家，因此认为大模型不实用，那就是没有用好大模型。

在严肃的商业场景下，我们更多希望用大模型辅助人，而不是代替人。也就是说人是最终的守门员。比如说大模型阅读理解长文本的能力比人强，但我们也不应该把它做的总结直接拿去作为商业决策，而要让人`review`一下，由人做最终的决定。

这里边有两个原因，第一个是准确性问题；另外一个方面，大模型的能力目前只是达到一个入门级的水平，达不到专家级。我们可以把大模型当成一个干活非常快但不太靠谱的初级员工。我们可以让大模型做一些初级的工作，比如写一些基础的`CRUD`代码，比人写得还快。但是你让他去设计系统架构，去做研究解决技术前沿问题，那是不靠谱的。我们在公司里也不会让初级员工去做这些事情。有了大模型之后，相当于有了大量又便宜干活又快的初级员工。怎么把这些初级员工用好，是一个管理问题。

当前有用的`AI Agent`大致可以分成两类：个人助理和商业智能。记忆是有趣和有用`AI`都必须具备的公共能力。情感是有趣`AI`需要的。而解决复杂任务和使用工具更多是有用`AI`所需的能力。

幻觉是大模型的基础问题，更大的模型幻觉相对会较少，幻觉的消除根本上还是要靠`scaling law`，靠基础模型的进步。但也有一些工程方法减少现有模型的幻觉。这里介绍两种典型的方法：**事实性校验和多次生成**。

**事实性校验**（`Factual Checking`）就是首先用大模型生成回答，然后用 RAG 的方法，用搜索引擎、向量数据库、倒排索引或者知识图谱找出与回答内容匹配的原始语料，然后将回答内容和原始语料送进大模型，让大模型判断回答与原始语料是否相符。事实性校验方法有两个问题：首先，幻觉有多种种类，事实性校验只能发现编造事实类的幻觉，但不能发现答非所问类的幻觉。比如我问中国的首都是哪里，它回答中国是一个有悠久历史的大国，用事实性校验也挑不出毛病，但这并没有正确回答问题。其次，原始语料的内容不一定就是事实，互联网上有大量不准确的信息。

**多次生成**是`SelfCheckGPT`这篇论文提出的，它的思想也很简单，就是多次生成同一问题的回答，然后把这些回答都放进大模型里，让大模型从中挑出最一致的那个。多次生成方法可以解决偶发的幻觉问题，但不能解决系统性偏差。例如让`GPT-3.5 Turbo`讲讲 “林黛玉倒拔垂杨柳” 的故事，几乎每次都会编一个类似的出来，而没有发现这个事件在历史上就不存在，这种幻觉就是多次生成很难消除的。

##### AI Agent：路在何方

我认为长期来看有用的价值更高，短期来看有趣的价值更高。这就是我们为什么在商业模式上选择有趣的`AI`，同时持续探索有用的`AI`。因为比如说语音闲聊，一块钱一个小时已经很不容易了，`Character AI`可能有上千万的用户，但是它每个月实际收入只有上千万美金，大多数是不付费的。但是如果一些在线教育、甚至是更专业领域的比如心理咨询、法律咨询等等它可能收入更高，但是这里边更关键的问题是需要质量和品牌才能产生一个更高的附加价值。更长远来看，我们的终极目标是`AGI`，那么`AGI`一定更多是有用的，可以扩展人类能力的边界，让人类做到之前做不到的事情。

