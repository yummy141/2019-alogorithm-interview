## SVM
- SVM既可以做回归，也可以做分类器
- 回归时可使用`SVR`或者`LinearSVR`
- 分类时可以使用`SVC`或者`LinearSVC`
    - `LinearSVC`相当于使用了线性核函数的`SVC`,但是效率要更高

## SVM分类器的创建
- `svm.SVC(kernel='rbf', C=1.0, gamma='auto')`
- kernel函数
    - `linear`线性核函数
        - 只能处理线性可分的数据，运算快
    - `poly`多项式核函数
        - 处理非线性可分数据，但参数比高斯多，计算量大
    - `rbf`高斯核函数（默认）
        - 也是处理非线性可分数据，参数相对较少
    - `sigmoid`
        - 使用的是多层神经网络
- 参数C
    - C越大容错率越低，泛化能力越差
- 参数gamma
    - 代表核函数的系数，默认为样本特征数的倒数

## sklearn.svm.SVR
```Python
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, 
                           param_grid, cv=5, 
                           scoring='neg_mean_squared_error', 
                           verbose=2, 
                           n_jobs=4)
grid_search.fit(housing_prepared, housing_labels)
```