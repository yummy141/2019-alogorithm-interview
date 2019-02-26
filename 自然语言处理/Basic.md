
# 词向量
> CSDN/[词向量：对word2vec的理解](https://blog.csdn.net/sinat_33741547/article/details/78759289)

## 词向量分类
- one-hot
  - 无法表征相似性
- distributed representation
  - 用较短的向量来表示

## 如何获取词向量
- 潜在语义分析LSA(latent semantic analysis)
- 隐含狄利克雷分布LDA(latent dirichlet analysis)
- 神经网络

## 神经概率语言模型相对n-gram有什么区别？
1. 由于词语都已经变成向量，词语之间的相似性可以通过词向量表示
2. 基于词向量的模型自带平滑，softmax使得概率不为0

## 神经概率语言模型和CBOW（Continuous Bag-of-Words Model）模型
1. 投影层：前者是拼接，后者是累加
2. 隐藏层：前者有，后者没有
3. 输出层：前者是线性结构，后者是树形结构

