## EM聚类
- 包括`HMM`（隐马尔科夫链）、`GMM`（高斯混合模型）
- 步骤
  - E（Expecation）是通过初始化的参数来估计隐含变量
  - M（Maximization）是通过隐含变量反推来优化参数
  - 最后通过EM步骤的迭代得到模型参数

## sklearn中的GMM聚类
- `gmm = GaussianMixture(n_components=1, covariance_type='full',max_iter=100)`
  - `n_components`
    - 代表高斯混合模型的个数，也就是我们要聚类的个数
  - `covariance_type=`
    - 协方差类型
    - `full`代表完全协方差，也就是元素都不为0
    - `tied`代表相同的完全协方差
    - `diag`代表对角协方差，对角不为0，其余为0
    - `spherical`代表球面协方差，非对角为0，对角完全相同
  - `max_iter`代表最大迭代次数
    - 默认为100

## 聚类效果的评价指标`Calinski-Harabaz`
```Python
from sklearn.metrics import calinski_harabaz_score
## 分数越高代表效果越好，即相同类之间的差异小，不同类之间的差异大
print(calinski_harabaz_score(data, prediction))
```