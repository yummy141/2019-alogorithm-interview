## KNN
- 通过计算待分类数据点与其它近邻数据点的距离，统计最近的K个邻居的分类情况，来决定这个数据点的分类情况。
- KNN既可以做分类器，也可以做回归
    - 分类器 `from sklearn.neighbors import KNeighborsClassifier`
    - 回归 `from sklearn.neighbors import KNeighborsRegressor`

## 分类器的创建
- `KNeighborsClassifier(n_neighbors=5, weights='uniform',algorithm='auto',leaf_size=30)`
- `n_neighbors`
    - KNN中的K值，默认为5
- `weights`
    - `uniform`代表所有邻居的权重相同
    - `distance`代表权重是距离的倒数，与距离成反比
- `algorithm`
    - `auto`自动选择
    - `kd_tree` KD树，适用于维度少的情况，一般维数不超过20
    - `ball_tree` 球树，适用于维度大的情况
    - `brute` 线性扫描，效率很低
- `leaf_size`
    - KD树或球树的叶子数

## 优缺点
- 可用于非线性分类
- 计算量大，样本不平衡时准确性差

