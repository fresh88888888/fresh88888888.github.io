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

在2维空间中，**超平行体**就是平行四边形。如下图所示，向量{% mathjax %}\vec{v}_1,\vec{v}_2{% endmathjax %}是平行四边形的两条边，它们唯一确定了这个平行四边形，平行四边形中的点都可以表示为：{% mathjax %}x = \alpha_1\vec{v}_1 + \alpha_2\vec{v}_2{% endmathjax %}，其中{% mathjax %}\alpha_1{% endmathjax %}和{% mathjax %}\alpha_2{% endmathjax %}是系数，取值范围是`[0,1]`，举例这里有一个平行四边形中的点{% mathjax %}x{% endmathjax %}，记作{% mathjax %}x = \frac{1}{2}\vec{v}_1 + \frac{1}{2}\vec{v}_2{% endmathjax %}，这个点落在平行四边形的中心；还有一个点刚好落在品行四边形的边界上，记作{% mathjax %}x = \vec{v}_1 + \vec{v}_2{% endmathjax %}，也就是说系数{% mathjax %}\alpha_1,\alpha_2{% endmathjax %}的值都是`1`。
{% asset_img ml_1.png %}

3维空间的超平行体为**平行六面体**。如下图所示，向量{% mathjax %}\vec{v}_1,\vec{v}_2,\vec{v}_3{% endmathjax %}是平行六面体的`3`条边，它们唯一确定了一个平行六面体，平行六面体中的点都可以表示为：{% mathjax %}x = \alpha_1\vec{v}_1 + \alpha_2\vec{v}_2 + \alpha_3\vec{v}_3{% endmathjax %}，其中系数{% mathjax %}\alpha_1,\alpha_2,\alpha_3{% endmathjax %}的取值范围是`[0,1]`。
{% asset_img ml_2.png %}

**超平行体**：一组向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}可以确定一个{% mathjax %}k{% endmathjax %}维超平形体：{% mathjax %}\mathcal{P}(\vec{v}_1,\ldots,\vec{v}_k) = \{\alpha_1\vec{v}_1 + \ldots + \alpha_k\vec{v}_k| 0\leq \alpha_1,\ldots,\alpha_k\leq 1\}{% endmathjax %}。超平行体的边{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}都是{% mathjax %}d{% endmathjax %}维向量，{% mathjax %}k{% endmathjax %}是向量的数量，{% mathjax %}d{% endmathjax %}是向量的维度，要求{% mathjax %}k\leq d{% endmathjax %}，比如{% mathjax %}d = 3{% endmathjax %}空间中有{% mathjax %}k=2{% endmathjax %}维平行四边形。如果想让超平行体有意义，那么{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}就必须线性独立，如果{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}线性相关（某个向量可以表示为其余向量的加权和），则体积{% mathjax %}\text{vol}(\mathcal{P}) = 0{% endmathjax %}（例：有{% mathjax %}k=3{% endmathjax %}个向量，落在一个平面上，则平行六面体的体积为`0`）。

**平行四边形的面积**：{% mathjax %}\text{面积} = ||\text{底}||_2 \times ||\text{高}||_2{% endmathjax %}，如下图所示，向量{% mathjax %}\vec{v}_1{% endmathjax %}为底，计算高{% mathjax %}\vec{q}_2{% endmathjax %}，这两个向量必须正交，两个向量长度的乘积就是平行四边形的面积。假如给定向量{% mathjax %}\vec{v}_1,\vec{v}_2{% endmathjax %}，把{% mathjax %}\vec{v}_1{% endmathjax %}作为底，该如何计算高{% mathjax %}\vec{q}_2{% endmathjax %}？可以计算向量{% mathjax %}\vec{v}_2{% endmathjax %}在{% mathjax %}\vec{v}_1{% endmathjax %}方向上的投影：{% mathjax %}\text{Proj}_{\vec{v}_1}(\vec{v}_2) = \frac{\vec{v}_1^{\mathsf{T}}\vec{v}_2}{||\vec{v}_1||^2_2}\cdot \vec{v}_1{% endmathjax %}。用{% mathjax %}\vec{v}_2{% endmathjax %}减去刚才得到的投影，就得到{% mathjax %}\vec{q}_2 = \vec{v}_2 - \text{Proj}_{\vec{v}_1}(\vec{v}_2){% endmathjax %}。计算的结果满足于：底{% mathjax %}\vec{v}_1{% endmathjax %}与高{% mathjax %}\vec{q}_2{% endmathjax %}正交。同样可以以{% mathjax %}\vec{v}_2{% endmathjax %}为底，计算{% mathjax %}\vec{q}_1{% endmathjax %}，首先计算向量{% mathjax %}\vec{v}_1{% endmathjax %}在{% mathjax %}\vec{v}_2{% endmathjax %}方向上的投影{% mathjax %}\text{Proj}_{\vec{v}_2}(\vec{v}_1) = \frac{\vec{v}_2^{\mathsf{T}}\vec{v}_1}{||\vec{v}_2||^2_2}\cdot \vec{v}_2{% endmathjax %}。计算出向量与{% mathjax %}\vec{v}_2{% endmathjax %}的方向相同，用{% mathjax %}\vec{v}_1{% endmathjax %}减去刚才得到的投影，就得到{% mathjax %}\vec{q}_1 = \vec{v}_1 - \text{Proj}_{\vec{v}_2}(\vec{v}_1){% endmathjax %}。计算的结果满足于：底{% mathjax %}\vec{v}_2{% endmathjax %}与高{% mathjax %}\vec{q}_1{% endmathjax %}正交。不管以谁为底，计算出的平行四边形的面积都是相同的，
{% asset_img ml_3.png %}

**平行六面体的体积**：{% mathjax %}\text{体积} = \text{底面积} \times ||\text{高}||_2{% endmathjax %}，以{% mathjax %}\vec{v}_1,\vec{v}_2{% endmathjax %}为边，组成平行四边形，这个平行四边形{% mathjax %}\mathcal{P}(\vec{v}_1,\vec{v}_2){% endmathjax %}是平行六面体{% mathjax %}\mathcal{P}(\vec{v}_1,\vec{v}_2,\vec{v}_3){% endmathjax %}的底，向量{% mathjax %}q_3{% endmathjax %}平行六面体的高，高{% mathjax %}q_3{% endmathjax %}垂直于{% mathjax %}\mathcal{P}(\vec{v}_1,\vec{v}_2){% endmathjax %}，用底面积乘以{% mathjax %}q_3{% endmathjax %}的长度，就得到平行六面体的体积。假设固定向量{% mathjax %}\vec{v}_1,\vec{v}_2,\vec{v}_3{% endmathjax %}的长度，那么平行六面体的体积何时最大化、最小化？设{% mathjax %}\vec{v}_1,\vec{v}_2,\vec{v}_3{% endmathjax %}都是**单位向量**，也就是说它们的二范数都等于`1`，当`3`个向量正交时，平行六面体是正方体，此时的体积最大，{% mathjax %}\text{vol} = 1{% endmathjax %}。如果3个向量线性相关，也就是说某个向量可以表示为其他向量的加权和，那么此时的体积就最小，{% mathjax %}\text{vol} = 0{% endmathjax %}。
{% asset_img ml_4.png %}

这里给定{% mathjax %}k{% endmathjax %}个物品，把它们表征为单位向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}，它们的二范数都等于`1`，之前说过向量最好用`CLIP`学习出的图文内容表征，这些向量都是{% mathjax %}d{% endmathjax %}维的，必须{% mathjax %}d\req k{% endmathjax %}，可以用超平行体的体积来衡量物品的多样性，体积介于`0~1`之间。如果向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}两两正交（多样性好），此时的体积最大，{% mathjax %}\text{vol} = 1{% endmathjax %}；如果向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}线性相关（多样性差），此时的体积最小，{% mathjax %}\text{vol} = 0{% endmathjax %}。给定{% mathjax %}k{% endmathjax %}个物品单位向量的表征{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k\in \mathbb{R}^d{% endmathjax %}，把它们作为矩阵{% mathjax %}\mathbf{V} = \in \mathbb{R}^{d\times k}{% endmathjax %}的列，设{% mathjax %}d\req k{% endmathjax %}，行列式与体积满足：{% mathjax %}\text{det}(\mathbf{V}^{\mathsf{T}}\mathbf{V}) = \text{vol}(\mathcal{P}(\vec{v}_1,\ldots,\vec{v}_k))^2{% endmathjax %}。因此可以用行列式{% mathjax %}\text{det}(\mathbf{V}^{\mathsf{T}}\mathbf{V}){% endmathjax %}衡量向量{% mathjax %}\vec{v}_1,\ldots,\vec{v}_k{% endmathjax %}的**多样性**。多样性越好，体积和行列式都会越大。矩阵{% mathjax %}\mathbf{V}{% endmathjax %}如下图所示：
{% asset_img ml_5.png %}
