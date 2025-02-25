---
title: 机器学习(ML)(十六) — 推荐系统探析
date: 2024-11-11 16:10:11
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

#### 重排 - 多样性算法(DPP)

**行列式点过程**(`determinantal point process, DPP`)是一种**概率模型**，最早引入于**量子物理学**中，用于描述**费米子系统**的分布。`DPP`的核心思想是能够有效地从一个全集中抽取出具有高相关性和**多样性**的子集，广泛应用于推荐系统、机器学习等领域，`DPP`是目前推荐系统重排多样性公认最好的多样性算法。
<!-- more -->

在`2`维空间中，**超平行体**就是平行四边形。如下图所示，向量{% mathjax %}\vec{v}_1,\vec{v}_2{% endmathjax %}是平行四边形的两条边，它们唯一确定了这个平行四边形，平行四边形中的点都可以表示为：{% mathjax %}x = \alpha_1\vec{v}_1 + \alpha_2\vec{v}_2{% endmathjax %}，其中{% mathjax %}\alpha_1{% endmathjax %}和{% mathjax %}\alpha_2{% endmathjax %}是系数，取值范围是`[0,1]`，举例这里有一个平行四边形中的点{% mathjax %}x{% endmathjax %}，记作{% mathjax %}x = \frac{1}{2}\vec{v}_1 + \frac{1}{2}\vec{v}_2{% endmathjax %}，这个点落在平行四边形的中心；还有一个点刚好落在品行四边形的边界上，记作{% mathjax %}x = \vec{v}_1 + \vec{v}_2{% endmathjax %}，也就是说系数{% mathjax %}\alpha_1,\alpha_2{% endmathjax %}的值都是`1`。
{% asset_img ml_1.png %}

3维空间的超平行体为**平行六面体**。如下图所示，向量{% mathjax %}\vec{v}_1,\vec{v}_2,\vec{v}_3{% endmathjax %}是平行六面体的`3`条边，它们唯一确定了一个平行六面体，平行六面体中的点都可以表示为：{% mathjax %}x = \alpha_1\vec{v}_1 + \alpha_2\vec{v}_2 + \alpha_3\vec{v}_3{% endmathjax %}，其中系数{% mathjax %}\alpha_1,\alpha_2,\alpha_3{% endmathjax %}的取值范围是`[0,1]`。
{% asset_img ml_2.png %}

**超平行体**：一组向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}可以确定一个{% mathjax %}k{% endmathjax %}维超平形体：{% mathjax %}\mathcal{P}(\vec{v}_1,\ldots,\vec{v}_k) = \{\alpha_1\vec{v}_1 + \ldots + \alpha_k\vec{v}_k| 0\leq \alpha_1,\ldots,\alpha_k\leq 1\}{% endmathjax %}。超平行体的边{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}都是{% mathjax %}d{% endmathjax %}维向量，{% mathjax %}k{% endmathjax %}是向量的数量，{% mathjax %}d{% endmathjax %}是向量的维度，要求{% mathjax %}k\leq d{% endmathjax %}，比如{% mathjax %}d = 3{% endmathjax %}空间中有{% mathjax %}k=2{% endmathjax %}维平行四边形。如果想让超平行体有意义，那么{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}就必须线性独立，如果{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}线性相关（某个向量可以表示为其余向量的加权和），则体积{% mathjax %}\text{vol}(\mathcal{P}) = 0{% endmathjax %}（例：有{% mathjax %}k=3{% endmathjax %}个向量，落在一个平面上，则平行六面体的体积为`0`）。

**平行四边形的面积**：{% mathjax %}\text{面积} = ||\text{底}||_2 \times ||\text{高}||_2{% endmathjax %}，如下图所示，向量{% mathjax %}\vec{v}_1{% endmathjax %}为底，计算高{% mathjax %}\vec{q}_2{% endmathjax %}，这两个向量必须正交，两个向量长度的乘积就是平行四边形的面积。假如给定向量{% mathjax %}\vec{v}_1,\vec{v}_2{% endmathjax %}，把{% mathjax %}\vec{v}_1{% endmathjax %}作为底，该如何计算高{% mathjax %}\vec{q}_2{% endmathjax %}？可以计算向量{% mathjax %}\vec{v}_2{% endmathjax %}在{% mathjax %}\vec{v}_1{% endmathjax %}方向上的投影：{% mathjax %}\text{Proj}_{\vec{v}_1}(\vec{v}_2) = \frac{\vec{v}_1^{\mathsf{T}}\vec{v}_2}{||\vec{v}_1||^2_2}\cdot \vec{v}_1{% endmathjax %}。用{% mathjax %}\vec{v}_2{% endmathjax %}减去刚才得到的投影，就得到{% mathjax %}\vec{q}_2 = \vec{v}_2 - \text{Proj}_{\vec{v}_1}(\vec{v}_2){% endmathjax %}。计算的结果满足于：底{% mathjax %}\vec{v}_1{% endmathjax %}与高{% mathjax %}\vec{q}_2{% endmathjax %}正交。同样可以以{% mathjax %}\vec{v}_2{% endmathjax %}为底，计算{% mathjax %}\vec{q}_1{% endmathjax %}，首先计算向量{% mathjax %}\vec{v}_1{% endmathjax %}在{% mathjax %}\vec{v}_2{% endmathjax %}方向上的投影{% mathjax %}\text{Proj}_{\vec{v}_2}(\vec{v}_1) = \frac{\vec{v}_2^{\mathsf{T}}\vec{v}_1}{||\vec{v}_2||^2_2}\cdot \vec{v}_2{% endmathjax %}。计算出向量与{% mathjax %}\vec{v}_2{% endmathjax %}的方向相同，用{% mathjax %}\vec{v}_1{% endmathjax %}减去刚才得到的投影，就得到{% mathjax %}\vec{q}_1 = \vec{v}_1 - \text{Proj}_{\vec{v}_2}(\vec{v}_1){% endmathjax %}。计算的结果满足于：底{% mathjax %}\vec{v}_2{% endmathjax %}与高{% mathjax %}\vec{q}_1{% endmathjax %}正交。不管以谁为底，计算出的平行四边形的面积都是相同的，
{% asset_img ml_3.png %}

**平行六面体的体积**：{% mathjax %}\text{体积} = \text{底面积} \times ||\text{高}||_2{% endmathjax %}，以{% mathjax %}\vec{v}_1,\vec{v}_2{% endmathjax %}为边，组成平行四边形，这个平行四边形{% mathjax %}\mathcal{P}(\vec{v}_1,\vec{v}_2){% endmathjax %}是平行六面体{% mathjax %}\mathcal{P}(\vec{v}_1,\vec{v}_2,\vec{v}_3){% endmathjax %}的底，向量{% mathjax %}q_3{% endmathjax %}平行六面体的高，高{% mathjax %}q_3{% endmathjax %}垂直于{% mathjax %}\mathcal{P}(\vec{v}_1,\vec{v}_2){% endmathjax %}，用底面积乘以{% mathjax %}q_3{% endmathjax %}的长度，就得到平行六面体的体积。假设固定向量{% mathjax %}\vec{v}_1,\vec{v}_2,\vec{v}_3{% endmathjax %}的长度，那么平行六面体的体积何时最大化、最小化？设{% mathjax %}\vec{v}_1,\vec{v}_2,\vec{v}_3{% endmathjax %}都是**单位向量**，也就是说它们的二范数都等于`1`，当`3`个向量正交时，平行六面体是正方体，此时的体积最大，{% mathjax %}\text{vol} = 1{% endmathjax %}。如果3个向量线性相关，也就是说某个向量可以表示为其他向量的加权和，那么此时的体积就最小，{% mathjax %}\text{vol} = 0{% endmathjax %}。
{% asset_img ml_4.png %}

这里给定{% mathjax %}k{% endmathjax %}个物品，把它们表征为单位向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}，它们的二范数都等于`1`，之前说过向量最好用`CLIP`学习出的图文内容表征，这些向量都是{% mathjax %}d{% endmathjax %}维的，必须{% mathjax %}d\geq k{% endmathjax %}，可以用超平行体的体积来衡量物品的多样性，体积介于`0~1`之间。如果向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}两两正交（多样性好），此时的体积最大，{% mathjax %}\text{vol} = 1{% endmathjax %}；如果向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}线性相关（多样性差），此时的体积最小，{% mathjax %}\text{vol} = 0{% endmathjax %}。给定{% mathjax %}k{% endmathjax %}个物品单位向量的表征{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}，把它们作为矩阵{% mathjax %}\mathbf{V} = \in \mathbb{R}^{d\times k}{% endmathjax %}的列，设{% mathjax %}d\geq k{% endmathjax %}，行列式与体积满足：{% mathjax %}\text{det}(\mathbf{V}^{\mathsf{T}}\mathbf{V}) = \text{vol}(\mathcal{P}(\vec{v}_1,\ldots,\vec{v}_k))^2{% endmathjax %}。因此可以用行列式{% mathjax %}\text{det}(\mathbf{V}^{\mathsf{T}}\mathbf{V}){% endmathjax %}衡量向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}的**多样性**。多样性越好，体积和行列式都会越大。矩阵{% mathjax %}\mathbf{V}{% endmathjax %}如下图所示：
{% asset_img ml_5.png %}

**精排**给{% mathjax %}n{% endmathjax %}个候选物品打分，把融合之后的分数记为：{% mathjax %}\text{reward}_1,\ldots,\text{reward}_n{% endmathjax %}，它们表示物品的价值。每个物品还有一个向量表征，通常是基于图文内容计算出的向量，把这{% mathjax %}n{% endmathjax %}个向量记作{% mathjax %}v_1,\ldots,v_n\in \mathbb{R}^d{% endmathjax %}，在精排的后处理阶段，也就是重排阶段，需要从{% mathjax %}n{% endmathjax %}个物品中选出{% mathjax %}k{% endmathjax %}个，组成集合{% mathjax %}\mathcal{S}{% endmathjax %}，做选择要考虑两个因素：一个是物品价值大，分数之和{% mathjax %}\sum_{j\in\mathcal{S}}\text{reward}_j{% endmathjax %}越大越好。第二个是物品的多样性好，把集合{% mathjax %}\mathcal{S}{% endmathjax %}中的{% mathjax %}k{% endmathjax %}个向量组成的超平行体{% mathjax %}\mathcal{P}(\mathcal{S}){% endmathjax %}的体积越大越好，体积越大就说明多样性越好。集合{% mathjax %}\mathcal{S}{% endmathjax %}中的{% mathjax %}k{% endmathjax %}个向量作为列，组成矩阵{% mathjax %}\mathbf{V}_{\mathcal{S}}\in \mathbb{R}^{d\times k}{% endmathjax %}，矩阵{% mathjax %}\mathbf{V}_{\mathcal{S}}{% endmathjax %}的形状是{% mathjax %}d\times k{% endmathjax %}，以这{% mathjax %}k{% endmathjax %}个向量为边，组成超平行体{% mathjax %}\mathcal{P}(\mathcal{S}){% endmathjax %}，超平行体的体积记作{% mathjax %}\text{vol}(\mathcal{P}(\mathcal{S})){% endmathjax %}，它可以衡量集合{% mathjax %}mathcal{S}{% endmathjax %}中物品的**多样性**。多样性越好，则体积越大。如果物品的数量{% mathjax %}k\leq d{% endmathjax %}，则行列式与体积满足：{% mathjax %}\text{det}(\mathbf{V}^{\mathsf{T}}_{\mathcal{S}}\mathbf{V}_{\mathcal{S}}) = \text{vol}(\mathcal{P}(\mathcal{S}))^2{% endmathjax %}。这说明行列式和体积是等价的，因此可以用行列式衡量向量的多样性多样性越好，则行列式也就越大。

**行列式点过程**(`determinantal point process, DPP`)是一种传统的统计机器学习方法：{% mathjax %}\underset{\mathcal{S}:|\mathcal{S}| = k}{\text{argmax}}\;\log\text{det}(\mathbf{V}^{\mathsf{T}}_{\mathcal{S}}\mathbf{V}_{\mathcal{S}}){% endmathjax %}。它可以度量集合{% mathjax %}\mathcal{S}{% endmathjax %}中向量的多样性，`DPP`要求从{% mathjax %}n{% endmathjax %}个候选物品中选出{% mathjax %}k{% endmathjax %}个，组成集合{% mathjax %}\mathcal{S}{% endmathjax %}，使得行列式的对数最大化，`Hulu`的论文[`Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity`](https://arxiv.org/pdf/1709.05135)将`DPP`应用在**推荐系统**，记作：{% mathjax %}\underset{\mathcal{S}:|\mathcal{S}| = k}{\text{argmax}}\ \theta\cdot (\sum_{j\in\mathcal{S}}\text{reward}_j) + (1 -\theta)\cdot\text{det}(\mathbf{V}^{\mathsf{T}}_{\mathcal{S}}\mathbf{V}_{\mathcal{S}}){% endmathjax %}。从{% mathjax %}n{% endmathjax %}个物品选出{% mathjax %}k{% endmathjax %}个，组成集合{% mathjax %}\mathcal{S}{% endmathjax %}，这一项{% mathjax %}\mathbf{V}^{\mathsf{T}}_{\mathcal{S}}\mathbf{V}_{\mathcal{S}}{% endmathjax %}，用行列式的对数度量集合{% mathjax %}\mathcal{S}{% endmathjax %}的多样性，把这一项记作矩阵{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}，形状为{% mathjax %}k\times k{% endmathjax %}，设矩阵{% mathjax %}\mathbf{A}{% endmathjax %}为{% mathjax %}n\times n{% endmathjax %}的矩阵，它的第{% mathjax %}i{% endmathjax %}行第{% mathjax %}j{% endmathjax %}列的元素为{% mathjax %}a_{ij} = v_i^{\mathsf{T}}v_j{% endmathjax %}（内积），给定向量{% mathjax %}v_1,\ldots,v_n\in \mathbb{R}^d{% endmathjax %}，它们的大小都是{% mathjax %}d{% endmathjax %}维，计算{% mathjax %}\mathbf{A}{% endmathjax %}的时间复杂度为{% mathjax %}\mathcal{O}(n^2d){% endmathjax %}，上面定义了{% mathjax %}\mathbf{A}_{\mathcal{S}} = \mathbf{V}^{\mathsf{T}}_{\mathcal{S}}\mathbf{V}_{\mathcal{S}}{% endmathjax %}，{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}是{% mathjax %}\mathbf{A}{% endmathjax %}的一个{% mathjax %}k\times k{% endmathjax %}的子矩阵。如果物品{% mathjax %}i,j\in \mathcal{S}{% endmathjax %}，{% mathjax %}a_{ij}{% endmathjax %}是{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}的一个元素。`DPP`的公式可以等价于写成：{% mathjax %}\underset{\mathcal{S}:|\mathcal{S}| = k}{\text{argmax}}\ \theta\cdot (\sum_{j\in\mathcal{S}}\text{reward}_j) + (1 -\theta)\cdot\log\text{det}(\mathbf{A}_{\mathcal{S}}){% endmathjax %}。`DPP`是个组合优化问题，要求从集合{% mathjax %}\{1,\ldots,n\}{% endmathjax %}中选出一个大小为{% mathjax %}k{% endmathjax %}的子集{% mathjax %}\mathcal{S}{% endmathjax %}，精确求解`DPP`是不可能的，因为`DPP`是一个`NP-hard`问题，用集合{% mathjax %}\mathcal{S}{% endmathjax %}表示已选中的物品，用集合{% mathjax %}\mathcal{R}{% endmathjax %}表示未选中的物品，通常会用贪心算法近似求解：{% mathjax %}\underset{i\in\mathcal{R}}{\text{argmax}}\ \theta\cdot\text{reward}_i + (1 -\theta)\cdot\log\text{det}(\mathbf{A}_{\mathcal{S}\cup\{i\}}){% endmathjax %}。{% mathjax %}\text{reward}_i {% endmathjax %}是物品{% mathjax %}i{% endmathjax %}的价值，{% mathjax %}\log\text{det}(\mathbf{A}_{\mathcal{S}\cup\{i\}}){% endmathjax %}是行列式的对数，{% mathjax %}\mathbf{A}_{\mathcal{S}\cup\{i\}}{% endmathjax %}是{% mathjax %}\mathbf{A}{% endmathjax %}的一个子矩阵，这个子矩阵比{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}多了一行和一列。每一轮选择一个物品{% mathjax %}i{% endmathjax %}，这个物品既要有较高的价值，也不能跟已经选中的物品相似。

求解`DPP`的方法：
- **贪心算法求解**：{% mathjax %}\underset{i\in\mathcal{R}}{\text{argmax}}\ \theta\cdot\text{reward}_i + (1 -\theta)\cdot\log\text{det}(\mathbf{A}_{\mathcal{S}\cup\{i\}}){% endmathjax %}，从集合{% mathjax %}\mathcal{R}{% endmathjax %}中选出一个物品，对于集合{% mathjax %}\mathcal{R}{% endmathjax %}中所有的物品{% mathjax %}i{% endmathjax %}需要计算这样的行列式({% mathjax %}\log\text{det}(\mathbf{A}_{\mathcal{S}\cup\{i\}}){% endmathjax %})，算法的难点就在于计算行列式，简单粗暴计算行列式，则计算量非常大。对于单个物品{% mathjax %}i{% endmathjax %}，计算{% mathjax %}\log\text{det}(\mathbf{A}_{\mathcal{S}\cup\{i\}}){% endmathjax %}的时间复杂度为{% mathjax %}\mathcal{O}(|\mathcal{S}^3|){% endmathjax %}。想要求解上边的公式，需要对集合{% mathjax %}\mathcal{R}{% endmathjax %}中所有物品{% mathjax %}i{% endmathjax %}求行列式，因此计算行列式的时间复杂度为{% mathjax %}\mathcal{O}(|\mathcal{S}^3|\cdot |\mathcal{R}|){% endmathjax %}，我们想要从{% mathjax %}n{% endmathjax %}个物品中选出{% mathjax %}k{% endmathjax %}个物品，所以要重复求解上面的公式{% mathjax %}k{% endmathjax %}次，如果暴力计算行列式，那么总的时间复杂度为：{% mathjax %}\mathcal{O}(|\mathcal{S}^3|\cdot |\mathcal{R}|\cdot k) = \mathcal{O}(nk^4){% endmathjax %}，暴力算法的总时间复杂度为：{% mathjax %}\mathcal{O}(dn^2 + nk^4){% endmathjax %}。{% mathjax %}dn^2{% endmathjax %}是计算矩阵{% mathjax %}\mathbf{A}{% endmathjax %}的时间，{% mathjax %}nk^4{% endmathjax %}是计算行列式的时间。{% mathjax %}n{% endmathjax %}的量级是几百，{% mathjax %}k,d{% endmathjax %}的量级都是几十。系统留给算法的时间也就是`10`毫秒左右，这个算法他慢了。
- `Hulu`**的快速算法**：`Hulu`的论文设计了一种数值算法，仅需{% mathjax %}\mathcal{O}(dn^2 + nk^2){% endmathjax %}的时间从{% mathjax %}n{% endmathjax %}个物品中选出{% mathjax %}k{% endmathjax %}个物品。给定向量{% mathjax %}v_1,\ldots,v_n\in \mathbb{R}^d{% endmathjax %}，需要{% mathjax %}\mathcal{O}(dn^2){% endmathjax %}计算矩阵{% mathjax %}\mathbf{A}{% endmathjax %}，因为算法运行过程中，需要不断用到{% mathjax %}\mathbf{A}{% endmathjax %}的子矩阵，还需要{% mathjax %}\mathcal{O}(nk^2){% endmathjax %}的时间计算所有行列式（利用了`Cholesky`分解）。`Hulu`的算法比暴力算法快。矩阵的`Cholesky`分解：给定一个对称的矩阵{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}，可以把它分解为{% mathjax %}\mathbf{A}_{\mathcal{S}} = \mathbf{L}\mathbf{L}^{\mathsf{T}}{% endmathjax %}，其中的{% mathjax %}\mathbf{L}{% endmathjax %}是一个下三角矩阵（对角线以上的元素全都等于`0`），有了`Cholesky`分解就可以计算矩阵{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}的行列式，下三角矩阵{% mathjax %}\mathbf{L}{% endmathjax %}的行列式{% mathjax %}\text{det}(\mathbf{L}){% endmathjax %}等于{% mathjax %}\mathbf{L}{% endmathjax %}对角线元素乘积。矩阵{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}的行列式等于{% mathjax %}\text{det}(\mathbf{A}_{\mathcal{S}}) = \text{det}(\mathbf{L})^2 = \prod_i l_{ii}^2{% endmathjax %}。基本思想是这样的：已知矩阵{% mathjax %}\mathbf{A}_{\mathcal{S}} = \mathbf{L}\mathbf{L}^{\mathsf{T}}{% endmathjax %}的`Cholesky`分解，给{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}添加一行或者一列不需要重新计算`Cholesky`分解，否则代价很大，给{% mathjax %}\mathbf{A}_{\mathcal{S}}{% endmathjax %}添加一行或者一列，`Cholesky`分解的变化非常小，有办法快速算出变化的地方，这样就不用完整算一遍`Cholesky`分解，那么就可以快速算出所有{% mathjax %}\mathbf{A}_{\mathcal{S}\cup\{i\}}{% endmathjax %}的`Cholesky`分解，因此就可以快速算出所有{% mathjax %}\mathbf{A}_{\mathcal{S}\cup\{i\}}{% endmathjax %}的行列式，这样时间复杂度就很低。

#### 物品冷启动 - 评价指标

什么是**物品冷启动**？例如小红书上用户新发布的笔记、`B`站或`YouTube`上用户新上传的视频、今日头条上作者新发布的文章，这些都属于**物品冷启动**。这里只考虑`UGC`(`User-Generated Content`)的冷启，内容都是用户自己上传的。与`UGC`相对应的`PGC`(`Plant-Generated Content`)，主要内容是像腾讯视频、爱奇艺平台采购的，`UGC`比`PGC`的冷启更难，这是因为用户自己上传的内容良莠不齐，而且量很大，很难用人工去评判，很难让运营人员去做流量调控。

**新物品冷启动**：新物品缺少与用户的交互，很难根据用户的行为做推荐，这会导致推荐的难度大、效果差。如果用正常的推荐链路，新物品很难得到曝光。即使得到曝光，效果也不好，消费指标也会很差。特殊对待新物品的另一个原因是促进发布，大量的实验表明，扶持新发布、低曝光的物品，可以增强作者的发布意愿。出现首次曝光的交互越快，有利于作者的积极性，新物品获得的曝光越多，也有利于作者的积极性。

**优化冷启的目标**：
- **精准推荐**：把新物品推荐给合适的用户，不引起用户的反感。
- **激励发布**：流量向新物品低曝光倾斜，激励作者发布，丰富内容池。
- **挖掘高潜**：通过初期小流量的试探，找到高质量的物品，给与流量倾斜。

**冷启动评价指标**：作者侧指标：发布渗透率(`penetration rate`)、人均发布量；用户侧指标：新物品的点击率、交互率、消费时长、日活、月活；内容侧指标：高热物品占比。{% mathjax %}\text{发布渗透率} = \text{当日发布人数} / \text{日活人数}{% endmathjax %}，{% mathjax %}\text{人均发布量} = \text{当日发布物品数} / \text{日活人数}{% endmathjax %}，发布渗透率(`penetration rate`)、人均发布量这两个指标反映出了作者的发布积极性。**物品冷启动的优化**包括：**优化全链路**（包括召回和排序）；**流量调控**（流量怎样在新物品、老物品中分配）。

#### 物品冷启动 - 简单召回通道

物品召回的依据包括：自带文字、图片、地点，算法或人工标注的标签，用户点击、点赞等信息，但是新物品缺少用户点击、点赞等信息。用户对物品的点击、点赞等统计数据可以反映出物品的质量，以及什么样的用户会喜欢这个物品，这对精准推荐的帮助会很大。可惜新物品没有这些信息，而且`ItemCF、UserCF`需要知道之类的召回通道需要知道物品跟哪些用户有些交互。如果一个物品还没有跟用户交互，就走不了`ItemCF`这种召回通道，物品冷启动缺少的另外一个关键信息是物品`ID Embedding`，召回和排序模型都有`Embedding`层，把每个物品`ID`映射到一个向量，这个向量是从用户跟物品的交互行为中学习出来的，物品冷启的时候这个向量是刚刚初始化的，还没有用反向传播更新，也就是说，新物品的`ID Embedding`啥都不是，反映不出物品的特点，缺少这个特征会是召回和排序很不准。

**冷启动改造双塔模型**：物品ID是物品塔中最重要的特征，每个物品都有一个`ID Embedding`向量，需要从用户和物品的交互中学习，可是新物品还没有跟用户交互过，所以它的`ID Embedding`向量还没有背学好，如果用双塔模型直接做新物品的召回，效果会不太好。介绍一下，改造`ID Embedding`的`2`种方案：
- **方案一**：新物品使用`default embedding`，也就是说，物品塔做`ID Embedding`的时候，让所有新物品共享一个`ID`，而不是用自己真正的`ID`。新物品发布之后，到下次模型训练的时候，新物品才具有自己`ID Embedding`向量。
- **方案二**：利用相似物品的`ID Embedding`向量，当新物品发布之后，查找`Top-K`内容最相似（图片、文字）的高爆物品，把{% mathjax %}k{% endmathjax %}个高爆物品的`ID Embedding`向量取平均，作为新物品的`ID Embedding`。之所以选择高曝光的物品，通常是因为高曝光的物品的`ID Embedding`通常学的比较好。

在实践中，通常会用多个向量召回池，用多个召回池可以让新物品有更多的曝光机会，所有这些召回池会共享一个双塔模型，那么多个召回池不增加训练模型的代价。**基于类目的召回通道**：一般推荐系统会维护一份从**类目->物品**的索引，索引上的key是类目，比如美食、旅游、摄影等。每个类目后边是一个物品的列表，按发布时间倒排，最新发布的物品排在最前面。系统要维护这样一个类目索引，系统用类目索引做召回：用户画像-> 类目-> 物品列表。取回物品列表上前{% mathjax %}k{% endmathjax %}个物品；**基于关键词召回**：原理跟基于类目的召回通道是一样的，系统需要维护一个关键词索引：**关键词-> 物品列表**(按时间倒排)，给用户做推荐的时候，根据用户画像上的**关键词**做召回。跟类目召回唯一的区别是用关键词代替类目，这两种召回通道都有很明显的缺点：1、只对刚刚发布的新物品有效，基于类目的索引、关键词索引都是根据发布时间倒排，发布的新物品排在最前面，并取回某类目或关键词下最新的{% mathjax %}k{% endmathjax %}个物品。发布之后时间比较长的物品，就再也没有机会被召回了；2、弱个性化、不够精准。

#### 物品冷启动 - 聚类召回

**聚类召回**的思想：如果用户喜欢一个物品，那么他会喜欢内容相似的物品。事先训练一个神经网络，基于物品的类目和图文内容，把物品映射到向量，向量的相似度就是物品内容的相似度。对物品向量做**聚类**，划分为`1000`个`cluster`，记录每个`cluster`的中心方向（`k-means`聚类，用余弦相似度）。**聚类召回通道**有一个索引，当新物品发布的时候，新物品就会上**聚类索引**，当一个物品发布之后，用神经网络把它映射到特征向量，然后把这个向量跟`1000`个向量（对应`1000`个`cluster`）做比较，找到最相似的向量，作为新物品的`cluster`。到这一步，新物品绑定了一个`cluster`，把新物品的`ID`添加到聚合索引上，聚合索引是：`cluster`-> 物品`ID`列表（按发布时间倒排），最新的物品排在最前面。有了索引，就可以在线上做召回，当用户发起推荐请求，系统就会根据的他的ID，找到他的`last-n`交互的物品列表，把这些物品作为种子物品，去召回相似的物品。用神经网络把每个种子物品映射到向量，然后跟`1000`个中心向量作比较。寻找最相似的`cluster`（知道了用户对哪些`cluster`感兴趣），最后从每个`cluster`的物品列表中，取回最新的{% mathjax %}m{% endmathjax %}个物品。这样最多取回{% mathjax %}m\times n{% endmathjax %}个新物品。{% mathjax %}n{% endmathjax %}的意思是保留{% mathjax %}n{% endmathjax %}条用户行为记录，包括点赞、收藏、转发等行为，把这{% mathjax %}n{% endmathjax %}个物品作为种子物品，{% mathjax %}m{% endmathjax %}的意思是每个种子物品召回{% mathjax %}n{% endmathjax %}个新物品。所以一共召回{% mathjax %}m\times n{% endmathjax %}个物品。

**聚类召回**需要调用一个**神经网络**，把物品图文内容映射到一个向量，如果两个物品的内容比较相似，那么两个物品的向量具有较大的**余弦相似度**，训练方式与三塔模型相似，把正样本物品、种子物品、负样本物品输入3个神经网络，神经网络包含`CNN `+ `BERT` + 全连接层，这`3`个神经网络的参数是相同的，神经网络分别输出`3`个向量，记作{% mathjax %}b^{+},a,b^{-}{% endmathjax %}，分别对应证样本物品、种子物品、负样本物品。计算向量{% mathjax %}b^{+}{% endmathjax %}和向量{% mathjax %}a{% endmathjax %}的余弦相似度，{% mathjax %}\cos(a,b^{+}){% endmathjax %}表示正样本物品与种子物品的内容相似度，数值越大越好；再计算向量{% mathjax %}b^{-}{% endmathjax %}和向量{% mathjax %}a{% endmathjax %}的余弦相似度，{% mathjax %}\cos(a,b^{-}){% endmathjax %}表示负样本物品与种子物品的内容相似度，数值越小越好。做训练的目标：鼓励{% mathjax %}\cos(a,b^{+}) > \cos(a,b^{-}){% endmathjax %}，这是`Triplet Hinge Loss`：{% mathjax %}L(b^{+},a,b^{-}) = \max\{0,\cos(a,b^{-}) + m - \cos(a,b^{+})\}{% endmathjax %}，最小化`Triplet Hinge Loss`会鼓励种子物品与正样本物品的相似度尽量大，种子物品与负样本物品的相似度尽量小。这是`Triplet Logistics Loss`：{% mathjax %}L(b^{+},a,b^{-}) = \log(1 + \exp(\cos(a,b^{-})  - \cos(a,b^{+}))){% endmathjax %}，可以最小化损失函数来学习神经网络参数，每一条训练数据都是一个三元组，包含<种子物品,正样本物品,负样本物品>。**正样本的选取**：最直接方法就是人工标注二元组的相似度，这样就得到了正样本，但是人工标注的代价大。也可以用算法自动选取正样本，而且代价比较小。筛选条件：只用高曝光物品作为二元组（因为有充足的用户交互信息），算法选的正样本会比较准。2个物品有相同的二级类目，可以过滤掉完全不相似的物品。最后`ItemCF`的物品相似度选择正样本。**负样本物品选取**：从全体物品中随机选出满足条件的就可以了。**聚类召回**模型的训练，如下图所示：
{% asset_img ml_6.png %}

#### 物品冷启动 - Look-Alike 召回

`Look-Alike`是互联网广告中常用的一种方法，这种方法也可以应用在推荐系统中，特别是召回低曝光笔记。假设有一个广告主向精准的给`100`万受众用户投放广告，假设广告主是特斯拉，他们知道特斯拉`Model 3`的典型用户是这样的：年龄在`25~35`岁，都是年轻人、受过良好教育，学历至少是本科、特斯拉车主大多关注科技数码，而且普遍喜欢苹果电子科技产品。把符合全部条件的用户圈出来，重点在这个人群中投放广告。满足所有条件的受众被称为种子用户，这样的用户数量不会很多，可能有几万人。但是潜在的符合条件的用户会很多，但是缺少他们的部分信息，很难找到他们。比方说多数的用户不填写自己的学历和年龄，广告主想给`100`万用户投放广告，现在才圈出几万人，那么如何发现潜在的`100`万用户呢？这就用到了`Look-Alike`人群扩散，寻找跟种子用户相似的用户，把找到的用户称为`Look-Alike`用户。通过这种方式可以利用几万个种子用户找到几十万个`Look-Alike`用户。那么如何定义两个用户的相似度？例如`UserCF`，以两个用户的共同兴趣点来衡量两个用户的相似度，若两个用户同时对相同的物品感兴趣，说明两个用户的相似度比较大。还有一种方法，使用两个用户的`ID Embedding`向量，如果两个用户相似，那么两个用户`ID Embedding`向量夹角的余弦值就比较大。如果用户有点击、点赞、收藏和转发的行为，说明用户对物品可能感兴趣。把有交互（点击、点赞、收藏和转发）的用户做为新物品的种子用户，如果一个用户跟种子用户相似，那么他也可能对这个新物品感兴趣，可以把新物品推荐给他。这种方法就叫做`Look-Alike`人群扩散。根据种子用户找出兴趣相似的更多用户，把新物品从种子用户扩散到`Look-Alike`用户。

`Look-Alike`**召回**：当发布一个新物品，系统会把新物品推荐给很多用户，少数用户会对物品感兴趣，会点击、点赞、收藏和转发。把这些用户称作种子用户，系统对新物品的推荐通常不太准，有交互行为的用户数量比较少，一旦有交互行为，就要充分利用好它，让推荐变得更准。取回每个种子用户的`Embedding`向量，可以复用双塔模型取到的用户向量，然后取这些用户向量的均值，得到另一个向量，把得到的这个向量作为新物品的表征。这个向量是要做**近线更新**的。近线的意思是不用实时更新，能做到分钟级的更新就可以了。

#### 物品冷启动 - 流量调控

为什么要给新物品流量倾斜呢？复制新物品主要有两个目的：1、促进发布，增大内容池；2、挖掘出优质物品。**流量调控**：流量怎样在新、老物品之间分配。**流量调控**技术包括：
- 在推荐结果中强插新物品。
- 对新物品的排序分数做**提权**(`boost`)，干涉粗排、重排环节，给新物品提权，缺点：曝光量对提权系数很敏感，很难精确控制曝光量，容易过度曝光和不充分曝光。
- 通过提权，对新物品做保量，对一个新物品，不论物品的质量高低，保证24小时获得`100`次曝光，在原有提权系数的基础上，再乘以额外的提权系数，动态提权保量：{% mathjax %}\text{提权系数} = f(\frac{\text{发布时间}}{\text{目标时间}},\frac{\text{已有曝光}}{\text{目标曝光}}) = f(0.5,0.2){% endmathjax %}，缺点：保量的成功率远低于`100%`。
- 差异化保量，不同的物品有不同的保量目标，根据物品的内容质量、作者质量来决定保量目标。

#### 物品冷启动 - AB测试

我们要做**新物品冷启动**，既要看作者侧指标，也要看**用户侧指标**和**发布侧指标**（发布渗透率、人均发布量），这些指标可以反映出作者的发布意愿。如果冷启动做得好，可以激励作者，让渗透率和发布量增长。用`AB`测试考察**发布侧指标**是困难的。`AB`测试还要考察用户侧指标（新物品的点击率、交互率），如果冷启动的推荐做的越精准，用户对推荐的新物品越感兴趣，那么新物品的点击率和交互率也就越高。除此之外，还要看大盘的消费指标（消费时长、日活、月活）。标准的`AB`测试，通常只做用户侧指标，实验比较好做，而冷启的`AB`测试要测试很多指标。推荐系统标准的`AB`测试，把用户随机分成两组：实验组和对照组，给实验组用户做推荐，从全量的物品池中选出最合适的物品，当实验组用户发起推荐请求之后，会用新的策略。给对照组用户做推荐，也是从全量的物品池中选取合适的物品，如果一个用户是对照组的，给他做推荐的时候，用旧的策略，在实验的过程中对比两组用户的用户侧指标的差异。比如考察用户消费推荐内容的时长，发现实验组比对照组高了`1%`，冷启的`AB`测试，要测两类指标，一类是用户侧的消费指标，另一类是作者侧的发布指标。先来看用户侧实验，比如考察策略对新物品点击率的影响、用户消费时长的影响，但这种`AB`测试的设计有不足之处，这样做的缺点：这样实验组的用户可以看到更多的新物品，消费指标变差。对照组的用户看到的新物品更少，消费指标会变好。这就导致两组观测到的差异很大，但推全之后，但消费指标并没有太大的的跌幅。作者侧的实验：方案一是将新物品分成实验组和对照组，实验组使用新策略，对照组使用旧策略。老物品使用的是自然分发，不受新旧策略的影响。从全量的老物品中选出用户最喜欢的推荐给用户，实验组和对照组的新物品有机会触达全体用户。缺点是新物品之间会抢流量，新、老物品之间会抢流量。方案二是用户侧被分成了两组：实验组和对照组，实验组的用户只能看到实验组的新物品，对照组的用户只能看到对照组的新物品。这样做的目的是避免两组之间抢流量，因此方案二比方案一更可信。方案二的缺点是物品池减少了，用户可能看不到最合适的物品，影响大盘指标。做**新物品冷启动**至少有两个目标：**激励作者发布**和**增进用户满意度**。

#### 评价指标优化 - 概述

对于常见的推荐系统来说**日活用户数**(`DAU`)和**留存**是最核心的指标。目前工业界常用`LT7`和`LT30`来衡量留存，假设某个用户今天登录了`APP`，未来`7`天({% mathjax %}t_0\sim t_6{% endmathjax %})中有`4`天登录了`APP`，那么该用户今天({% mathjax %}t_0{% endmathjax %})的`LT7`等于`4`。显然有{% mathjax %}1\leq \text{LT7}\leq 7,\;1\leq \text{LT30}\leq 30{% endmathjax %}，像抖音或小红书这样的信息流推荐系统，最重要的目标就是提升`LT`，`LT`的增长通常意味着用户体验的提升（除非`LT`增长且`DAU`下降）。还有用户使用的时长、总阅读数（即总点击数）、总曝光数。这些指标的重要性低于**日活用户数**(`DAU`)和**留存**，用户使用时长增长，`LT`也会增长；用户使用时长增长，阅读数、曝光数可能会下降。

#### 评价指标优化 - 召回

**推荐系统**有几十条召回通道，它们的召回总量是固定的(例如`5000`)，总量越大，指标越好，粗排计算量也越大。双塔模型和`Item-To-Item`是最重要的两类召回模型，占据召回的大部分配额。还有很多小众的模型，它们占据的配额都比较少，在召回总量不变的前提下，添加某些召回模型可以提升核心指标。可以有很多内容池，同一个模型可以用于多个内容池，得到多个召回通道。

改进召回的模型包括**双塔模型**、`Item-To-Item`、**其它模型**：
- **双塔模型**：1、优化正、负样本，简单正样本：有点击的（用户，物品）二元组；简单负样本：随机组合的（用户，物品）二元组；困难负样本：排序靠后的（用户，物品）二元组。2、改进神经网络结构，双塔模型有用户塔和物品塔两个神经网络（全连接网络），它们分别把用户特征、物品特征作为输入，各自输出一个向量作为用户、物品的表征。可以用更高级的神经网络代替全连接网络。在用户塔中使用用户行为序列(`last-n`)也可以让双塔模型的效果更好。标准的双塔模型也叫但向量模型，两个塔各输出一个向量，根据向量相似度做分类，能够让模型区分正、负样本。这里使用多向量代替单向量模型，物品塔跟单向量模型没有区别，物品塔只输出一个向量作为物品的表征，这里的用户塔输出很多个向量，用户塔输出的每一个向量都和物品塔输出的向量形状相同，可以计算它们的内积或者余弦相似度，用这两个向量的相似度来最为一个目标（点击率、点赞率、收藏率、转发率）的预估。如果要预估`10`个目标，那么用户塔要输出`10`个向量，但是物品塔只输出一个向量，用户塔的`10`个向量分别跟物品塔的一个向量去计算相似度，作为对`10`个目标的预估。3、改进模型的训练方法，训练双塔模型最基本的方法是二分类，让模型学会区分正样本和负样本，训练的方法可以进一步改进，结合二分类、`batch`内负采样（需要做纠偏）。使用自监督学习，让冷门物品的`embedding`学得更好。
- `Item-To-Item`**模型**：是一大类模型的总称，基于相似物品做召回，常见的用法是`U2I2I`(`user-> item -> item`)，假设用户{% mathjax %}u{% endmathjax %}喜欢物品{% mathjax %}i_1{% endmathjax %}(用户历史上交互过的物品)，寻找{% mathjax %}i_1{% endmathjax %}的相似物品{% mathjax %}i_2{% endmathjax %}，即`I2I`。最后将{% mathjax %}i_2{% endmathjax %}推荐给用户{% mathjax %}u{% endmathjax %}。计算物品的相似度有两种方法：1、基于用户兴趣的相似度来计算（ItemCF及其变体）；2、基于物品的向量表征，计算向量的相似度（双塔模型、图神经网络计算物品的向量特征）。
- **其它模型**：`U2UI`(`user -> user -> item`)模型，已知用户{% mathjax %}u_1{% endmathjax %}和{% mathjax %}u_2{% endmathjax %}相似，且{% mathjax %}u_2{% endmathjax %}喜欢物品{% mathjax %}i{% endmathjax %}，那么给用户{% mathjax %}u_1{% endmathjax %}推荐物品{% mathjax %}i{% endmathjax %}。`U2A2I`(`user -> author -> item`)模型，已知用户{% mathjax %}u{% endmathjax %}喜欢作者{% mathjax %}a{% endmathjax %}，且{% mathjax %}a{% endmathjax %}发布物品{% mathjax %}i{% endmathjax %}，那么给用户{% mathjax %}u{% endmathjax %}推荐物品{% mathjax %}i{% endmathjax %}。`U2A2A2I`(`user -> author -> author -> item`)模型，已知用户{% mathjax %}u{% endmathjax %}喜欢作者{% mathjax %}a_1{% endmathjax %}，且{% mathjax %}a_1{% endmathjax %}与{% mathjax %}a_2{% endmathjax %}相似，{% mathjax %}a_2{% endmathjax %}发布物品{% mathjax %}i{% endmathjax %}，那么给用户{% mathjax %}u{% endmathjax %}推荐物品{% mathjax %}i{% endmathjax %}。还有一些小众的召回模型，比如`PDN`、`Deep Retrieval`、`SINE`、`M2GRL`等模型。

在召回总量不变的前提下，仔细调整各召回通道的配额，可以提高核心指标（可以让各用户群体用不同的配额）。

#### 评价指标优化 - 排序

**推荐系统**的**精排模型**，如下图所示，最下面是模型的输入，分为**离散特征**和**连续特征**，把离散特征输入到**神经网络**，神经网络用`embedding`层把离散特征映射成数值向量，把得到的数值向量全都拼接起来得到一个几千维的向量，再经过几个全连接层得到上面的绿色向量，大小是几百维。连续特征输入到另一个全连接网络，全连接网络的输出是上面蓝色的向量。把绿色和蓝色的向量做`concatenation`，作为更上层网络的输入，下面这两个神经网络叫做**基座**，把原始特征映射到数值向量，绿色和蓝色的向量做`concatenation`之后，同时输入到多个全连接网络，这些全连接网络通常只有`2`层，这些神将网络的输出都是介于`0~1`之间的数值，作为各种目标的预估（不如预估点击率、预估点赞率、预估收藏率、预估转发率）。精排模型的基座和上面的多目标预估都有可以改进的点。
- **基座**：基座的输入包括离散特征和连续特征，基座的输出是一个向量，也就是绿色和蓝色的向量做`concatenation`，这个向量作为多目标预估的输入。基座改进方法：1、基座加宽加深、计算量更大，预测更准确。2、做自动的特征交叉，比如`bilinear`、`LHUC`等。3、特征工程，比如添加统计特征、多模态内容特征。
- **多目标预估**：基于基座输出的向量，同时预估点击率等多个目标。多目标预估改进：1、增加新的预估目标，并把预估结果加入融合公式。2、使用`MMoE、PLE`等结构可能有效，也可能无效。3、纠正`position bias`可能有效，也可能无效。
{% asset_img ml_7.png "精排模型结构" %}

**粗排模型**的打分量比精排大`10`倍，因此粗排模型必须计算够快。**粗排模型**可以使用**多向量双塔模型**，同事预估点击率等多个目标，也可以使用更复杂一些的模型，比如三塔模型效果好，但工程实现难度较大。除了改进粗排的模型结构，还可以粗精排一致性建模来提升指标。原理是蒸馏精排模型训练粗排模型，让粗排和精排的分数趋于一致。1、`pointwise`蒸馏，设{% mathjax %}y{% endmathjax %}是用户真实的行为，设{% mathjax %}p{% endmathjax %}是精排的预估。用{% mathjax %}\frac{y + p}{2}{% endmathjax %}作为粗排拟合的目标。2、`pairwise`、`listwise`蒸馏，给定{% mathjax %}k{% endmathjax %}个候选物品，按照精排预估做排序，做`learning to rank(LTR)`训练粗排模型，让粗排拟合物品的序（而非值），粗精排一致性建模可以提升核心指标。

**改进用户行为序列建模**：1、增加序列长度，这样让预测更准确，但是会增加计算成本和推理时间，具体难点还是在于工程架构。2、筛选的方法，目的是降低序列长度，比如用类目、物品向量表征聚类。例如离线用多模态神经网络提取物品内容特征，将物品表征维向量；离线将物品向量聚为`1000`类，每个物品有一个聚类序号。聚类通常用层次聚类，在线上做排序的时候，用户行为序列中有{% mathjax %}1\,000\,000{% endmathjax %}个物品，某个候选物品的聚类序号是`70`，对{% mathjax %}1\,000\,000{% endmathjax %}个物品做筛选，只保留聚类序号为`70`的物品，这样做筛选，只有几千个物品被保留了下来。线上同时有好几种方法在筛选，取筛选结果的并集。可能还需要作进一步筛选，让物品数量再降低一个数量级，然后再输入到**注意力层**。3、对用户行为序列中的物品，使用`ID`以外的一些特征。

现在使用用户行为序列建模都是沿着`SIM`的方向在发展，让原始的序列尽量长，然后做筛选降低序列长度，最后将筛选结果输入`DIN`，对物品向量做加权平均。

线上有{% mathjax %}m{% endmathjax %}个模型，其中`1`个是`holdout`，`1`个是推全的模型，{% mathjax %}m-2{% endmathjax %}个测试的新模型。每套在线学习的机器成本都很大，因此{% mathjax %}m{% endmathjax %}数量很小，制约模型迭代开发的效率。在线学习对指标的提升会巨大，但会制约模型开发迭代的效率。

**老汤模型**的产生？用每天新产生的数据对模型做`1 epoch`训练。久而久之，老模型训练的非常好，很难被超过。对模型做改进之后，重新训练，很难追上老模型。这样会产生一些问题：如何快速判断新模型结构是否优于老模型？如何更快追平、超过线上的老模型？对于新、老模型结构，都随机初始化模型全连接层，`Embedding`层可以是随机初始化的，也可以是复用老模型训练好的参数。这样处理全连接层和`Embedding`层，新、老模型的区别只是模型结构而已，老模型并没有训练更久的优势，新、老模型可以公平对比。用{% mathjax %}n = 10{% endmathjax %}天的数据同时训练新老模型，做完训练之后，如果新模型显著优于老模型，则新模型很可能更优。尽可能多地复用老模型训练好的`Embedding`层，避免随机初始化`Embedding`层（`Embedding`层是对用户、物品特点的“记忆”，比全连接层学得慢）。用老模型做`teacher`，蒸馏新模型（用户真实行为{% mathjax %}y{% endmathjax %}，老模型的预测为{% mathjax %}p{% endmathjax %}，用{% mathjax %}\frac{y+p}{2}{% endmathjax %}作为训练新模型的目标。）。

#### 评价指标优化 - 多样性

提升**推荐系统**的多样性包括：提升**排序的多样性**、**召回的多样性**、**兴趣探索**。排序的多样性分为精排、粗排多样性。
- **精排阶段**：要结合兴趣分数和多样性分数对物品{% mathjax %}i{% endmathjax %}排序，物品{% mathjax %}i{% endmathjax %}的兴趣分数，记作{% mathjax %}s_i{% endmathjax %}，它是对点击率等多个目标的融合。物品{% mathjax %}i{% endmathjax %}的多样性分数，记作{% mathjax %}d_i{% endmathjax %}，即物品{% mathjax %}i{% endmathjax %}与已经选中物品的差异。把兴趣分数{% mathjax %}s_i{% endmathjax %}与多样性分数{% mathjax %}d_i{% endmathjax %}相加，根据这个加和({% mathjax %}s_i + d_i{% endmathjax %})对物品做排序。这个排序决定了用户最终看到的结果，**推荐系统**通常使用`MMR`、`DPP`等方法计算多样性分数，计算多样性的时候，精排使用滑动窗口，粗排不使用滑动窗口。精排决定最终的曝光，曝光页面上邻近的物品相似度应该小，所以计算精排多样性需要使用滑动窗口；粗排要考虑整体的多样性，而非一个滑动窗口内的多样性，所以粗排不使用滑动窗口。除了多样性分数，精排还使用打散策略来增加多样性，以类目为例，当前选中的物品{% mathjax %}i{% endmathjax %}，之后的`5`个位置不允许跟{% mathjax %}i{% endmathjax %}的二级类目相同。
- **粗排阶段**：粗排给`5000`个物品打分，选出`500`个物品送入精排，提升粗排、精排多样性都可以提升**推荐系统**核心指标，先根据粗排模型预估的分数{% mathjax %}s_i{% endmathjax %}，对`5000`个物品排序，选出分数最高的`200`个物品送入精排，然后对每个物品{% mathjax %}i{% endmathjax %}计算兴趣分数{% mathjax %}s_i{% endmathjax %}和多样性分数{% mathjax %}d_i{% endmathjax %}，表示物品{% mathjax %}i{% endmathjax %}与`200`个物品的差异，差异越大，多样性分数{% mathjax %}d_i{% endmathjax %}就越大，结合兴趣分数和多样性分数，对剩余`4800`个物品做排序，选出分数最高的`300`个物品进入精排。这`300`个物品即是用户感兴趣的，也是与已经选中的`200`个物品有较大的差异，保证了多样性。
- **召回阶段**：**双塔模型**中，用户塔将用户特征作为输入，输出用户的向量表征；然后做`ANN`检索，召回向量相似度高的物品。在线上做召回的时候（在计算出用户向量之后，在做`ANN`检索之前），往用户向量中添加随机噪声。用户的兴趣越窄，就越需要提升多样性，添加的噪声也就越强。往用户向量中添加噪声，会让召回变得不准，单添加噪声提升了多样线，并提升了**推荐系统**的核心指标。用户最近交互过的{% mathjax %}n{% endmathjax %}个物品（**用户行为序列**）是用户塔的输入，保留最近的{% mathjax %}r{% endmathjax %}个物品({% mathjax %}r\ll n{% endmathjax %})，从剩余的{% mathjax %}n - r{% endmathjax %}个物品随机抽样{% mathjax %}t{% endmathjax %}个物品({% mathjax %}t\ll n{% endmathjax %})。可以均匀抽样，也可以用非均匀抽样让类目平衡。将得到的{% mathjax %}r + t{% endmathjax %}个物品作为**用户行为序列**，而不是用全部{% mathjax %}n{% endmathjax %}个物品。抽样**用户行为序列**，注入了随机性，召回结果更加多样化；这样{% mathjax %}n{% endmathjax %}可以非常大，能够捕捉到用户很久之前的兴趣。
- **兴趣探索**：保留少部分的流量(流量的`1%`)给非个性化的推荐。

#### 评价指标优化 - 特殊人群

为什么要特殊对待特殊人群？因为新用户、低活用户的行为很少，个性化推荐不准确；新用户、低活用户容易流失，要想办法促使他们留存。特殊用户的行为不同于主流用户，基于全体用户行为训练出的模型在特殊用户人群上有偏。针对特殊人群涨指标的方法包括：
- **构造特殊内容池**，用于特殊用户人群的召回。构造内容池的方法：1、根据物品获得的交互次数、交互率选择优质物品；2、做因果推断，判断物品对人群留存率的贡献，根据贡献值选择物品。通常使用双塔模型从特殊内容池中做召回。
- **使用特殊的排序策略**，保护特殊用户。**排序策略**包括：1、排除低质量物品；2、差异化融分公式。
- **使用特殊的排序模型**，消除模型预估的偏差。差异化的排序模型包括：1、大模型 + 小模型；2、融合多个`experts`，类似于`MMoE`；3、大模型预估之后，使用小模型做校准。

#### 评价指标优化 - 交互行为

**交互行为**包括：点击、点赞、收藏、转发、关注、评论...，主要是将模型预估的交互率进行排序，模型将交互行为当做预估的目标，输出介于`0~1`之间的数值。将预估的点击率、交互率做融合作为排序的依据。对于一位用户，他关注的作者越多，平台对他的吸引力也就越强。把用户的留存率（{% mathjax %}r{% endmathjax %}）与关注的坐着数量（{% mathjax %}f{% endmathjax %}）正相关。如果某用户的{% mathjax %}f{% endmathjax %}比较小，则推荐系统要促使该用户关注更多作者。如何利用关注关系提升用户留存？1、用排序策略提升关注量，对于用户{% mathjax %}u{% endmathjax %}，模型预估候选物品{% mathjax %}i{% endmathjax %}/的关注率为{% mathjax %}p_i{% endmathjax %}，设用户{% mathjax %}u{% endmathjax %}已经关注了{% mathjax %}f{% endmathjax %}个作者。这里定义一个单调递减函数{% mathjax %}w(f){% endmathjax %}，用户已经关注的作者越多，则{% mathjax %}w(f){% endmathjax %}越小。在排序的融分公式中添加{% mathjax %}w(f)\cdot p_i{% endmathjax %}（如果{% mathjax %}f{% endmathjax %}小且{% mathjax %}p_i{% endmathjax %}，则{% mathjax %}w(f)\cdot p_i{% endmathjax %}给物品{% mathjax %}i{% endmathjax %}带来很大的加分）。另一种方法是构造促关注内容池和召回通道，这个内容池中物品的关注率高，如果用户点击进入这些物品，有较大概率关注作者。只有用户关注的作者数{% mathjax %}f{% endmathjax %}较小，才对用户使用该内容池。做召回时，召回配额可以是固定的，也可以与{% mathjax %}f{% endmathjax %}相关。用户关注作者越少，就从这个内容池召回越多的物品。

`UGC`平台将作者发布量、发布率作为核心指标，希望作者多发布。作者发布的物品被平台推送给用户，会产生点赞、评论、关注等交互，交互可以提升作者发布的积极性。作者的粉丝数量越少，则每增加一个粉丝对发布积极性的提升越大。用排序策略帮助低粉新作者涨粉。某作者{% mathjax %}a{% endmathjax %}的粉丝数（被关注数）为{% mathjax %}f_a{% endmathjax %}，作者{% mathjax %}a{% endmathjax %}发布的物品{% mathjax %}i{% endmathjax %}被推荐给用户{% mathjax %}u{% endmathjax %}，模型预估关注率为{% mathjax %}p_{ui}{% endmathjax %}，关注率{% mathjax %}p_{ui}{% endmathjax %}越高，用户对物品{% mathjax %}i{% endmathjax %}的兴趣也就越大。这里定义单调递减函数{% mathjax %}w(f_a){% endmathjax %}作为权重，作者{% mathjax %}a{% endmathjax %}的粉丝数越多，则{% mathjax %}w(f_a){% endmathjax %}越小。在排序公式中添加{% mathjax %}w(f_a)\cdot p_{ui}{% endmathjax %}，帮助低粉作者涨粉。

**显示关注关系**：用户{% mathjax %}u{% endmathjax %}关注了作者{% mathjax %}a{% endmathjax %}，将{% mathjax %}a{% endmathjax %}发布的物品推荐给{% mathjax %}u{% endmathjax %}，跟其它召回通道相比，关注关系召回的物品有更高的点击率、交互率。**隐式关注关系**也可以用于召回，用户{% mathjax %}u{% endmathjax %}喜欢看作者{% mathjax %}a{% endmathjax %}发布的物品，但是用户{% mathjax %}u{% endmathjax %}没有关注作者{% mathjax %}a{% endmathjax %}，隐式关注的作者数量要远大于显示关注。因此推荐系统应当挖掘隐式关注关系，给每个用户找到隐式关注的作者。利用**隐式关注关系**做`U2A2I`召回，可以提升**推荐系统**的核心指标。

- **关注**：留存价值（让新用户关注更多作者，提升新用户留存）；发布价值（帮助新作者获得更多粉丝，提升作者的发布积极性）；利用隐式关注关系做召回。
- **转发**：判断哪些用户是站外`KOL`，利用他们转发的价值，吸引站外的流量。
- **评论**：发布价值（促使新物品获得评论，提升作者发布积极性）；留存价值（给喜欢讨论的用户创造更多留评论的机会）；鼓励高质量评论的用户多留评论。

