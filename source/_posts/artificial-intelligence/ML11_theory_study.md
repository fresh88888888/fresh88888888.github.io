---
title: 机器学习(ML)(十一) — 推荐系统探析
date: 2024-10-16 15:50:11
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

**推荐系统**(`Recommendation system`)的商业影响和实际使用案例数量甚至远远超过学术界的关注程度。每次你访问京东`app`、淘宝`app`、美团`app`等或腾讯视频等电影流媒体网站，或者访问提供短视频(抖音、快手)应用时，此类引用都会向你推荐他们认为你可能想买的东西、他们认为你可能想看的电影或他们认为你可能想尝试的餐馆。对于许多公司来说，很大一部分销售额是由他们的**推荐系统**(`Recommendation system`)推动的。因此，对于许多公司来说，**推荐系统**(`Recommendation system`)带来的经济效益或价值非常大。因此，我们很有必要深入了解一下什么是**推荐系统**(`Recommendation system`)。
<!-- more -->

我将使用预测电影评分的应用作为示例。假设您经营一家大型电影流媒体网站，您的用户使用一到五颗星对电影进行评分。因此，在典型的**推荐系统**(`Recommendation system`)中，您有一组用户，这里有四个用户`Alice、Bob Carol`和`Dave`。用户编号为`1、2、3、4`。以及一组电影《爱在最后》、《浪漫永恒》、《可爱的小狗》、《不停歇的汽车追逐》和《剑与空手道》。用户所做的就是将这些电影评为一到五颗星。假设`Alice`给《爱在最后》评了五颗星，给《浪漫永恒》评了五颗星。也许她还没有看过《可爱的小狗》，所以没有对这部电影进行评分。则通过问号来表示，她认为《不停歇的汽车追逐》和《剑与空手道》应该得零颗星等。在**推荐系统**(`Recommendation system`)中，你有一定数量的用户和一定数量的项目。在这种情况下，**项目**是您想要推荐给用户的电影。尽管在这个例子中使用的是电影，但同样的逻辑或同样的东西也适用于任何东西，从产品或网站到餐馆，甚至推荐哪些媒体的文章、要展示的社交媒体文章，对用户感到更有趣的东西。这里使用的符号是{% mathjax %}n_u{% endmathjax %}来表示用户数量。所以在这个例子中，{% mathjax %}n_u = 4{% endmathjax %}，因为你有四个用户{% mathjax %}n_m{% endmathjax %}表示电影数量或实际上是项目数量。所以在这个例子中，{% mathjax %}n_m = 5{% endmathjax %}，因为我们有五部电影。如果用户{% mathjax %}j{% endmathjax %}对电影{% mathjax %}i{% endmathjax %}进行了评分，将设置{% mathjax %}r(i,j) = 1{% endmathjax %}。假设`Dallas Alice`对电影`1`进行了评分，但尚未对电影`3`进行评分，因此{% mathjax %}r(1,1) = 1{% endmathjax %}，因为她对电影`1`进行了评分，但{% mathjax %}r(3,1) = 0{% endmathjax %}，因为她尚未对电影`3`进行评分。最后使用{% mathjax %}y^{(i,j)}{% endmathjax %}表示用户{% mathjax %}j{% endmathjax %}对电影{% mathjax %}i{% endmathjax %}给出的评分。例如，此处的评分将是用户`2`对电影`3`的评分等于{% mathjax %}y^{(3,2)} = 4{% endmathjax %}。请注意，并非每个用户都会对每部电影进行评分，因此系统需要知道哪些用户对哪些电影进行了评分。这就是为什么用户{% mathjax %}j{% endmathjax %}对电影{% mathjax %}i{% endmathjax %}进行了评分，将定义为{% mathjax %}r(i,j) = 1{% endmathjax %}，如果用户{% mathjax %}j{% endmathjax %}尚未对电影{% mathjax %}i{% endmathjax %}进行评分，则{% mathjax %}r(i,j) = 0{% endmathjax %}。使用此**推荐系统**(`Recommendation system`)框架，解决问题的一种方法是查看用户尚未评分的电影。并尝试预测用户对这些电影的评分，因为这样我们就可以尝试向用户推荐他们更有可能评为五星的电影。
