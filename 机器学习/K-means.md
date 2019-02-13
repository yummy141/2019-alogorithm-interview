## K-means
### 三大问题
- 如何确定中心点？
- 如何将其他点划分到K类中？
- 如何区分K-Means与KNN？
    - K-Means中的K是K类，而KNN中的是K个邻居
  
### sklean中的聚类方法
- `from sklearn.cluster import KMeans`
- `KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')`
- `n_clusters`
  - k值
- `max_iter`
  - 最大迭代值
- `n_init`
  - 初始化中心点的运算次数，如果K值比较大，`n_init`也应该越大

```Python
# coding: utf-8
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
# 输入数据
data = pd.read_csv('data.csv', encoding='gbk')
train_x = data[["2019 年国际排名 ","2018 世界杯 ","2015 亚洲杯 "]]
df = pd.DataFrame(train_x)
kmeans = KMeans(n_clusters=3)
# 规范化到 [0,1] 空间
min_max_scaler=preprocessing.MinMaxScaler()
train_x=min_max_scaler.fit_transform(train_x)
# kmeans 算法
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
# 合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:u'聚类'},axis=1,inplace=True)
print(result)

```
