# 机器学习八个步骤
1. 问题框架化，视野宏观化
2. 获取数据
- pd.read_csv() 返回DataFrame
- @proerty
    - .head()
    - .info(), 看数据类型
    - .describe(), 看数字属性摘要
    - .hist() 画直方图，快速了解数据, 我们希望数据呈钟形分布，所以会resize，同时剔除掉一些极端值
        > housing.hist(bins=50, figsize=(20,15))
    - .value_counts() 看每一列里的分类数据
- 创建并修改训练、验证、测试集
    - from sklearn.model_selection import train_test_split
        > train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    - from sklearn.model_selection import StratifiedShuffleSplit
        ```Python
        # 分层随机拆分，相对就要复杂一些 
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)  
        for train_index, test_index in split.split(housing, housing["income_cat"]):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index] 
        ```
    - 删除某个属性
        > set_.drop("income_cat", axis=1, inplace=True)
3. 探索数据以获得 深层次见解
- @property
    - .corr(),计算每对属性之间的标准相关性系数（也称为Pearson’s），并且可以继续.sort_values(ascending=False)排列相关性大小
        - 或者pandas.plotting的 scatter_matrix函数
            ```Python
            attributes = ["median_house_value",
              "median_income", 
              "total_rooms",
              "housing_median_age"]
            scatter_matrix(housing[attributes], figsize=(12, 8))
             ```
- 组合特征    
4. 准备数据以更好地将基础数据模式提供给机器学习算法
- 注意编写函数以方便服用，同时数据集需要copy
    ```Python
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    ```
- 数据清洗
    - .dropna(subset=["..."]) 删除相应区域
    - .drop() 删除整个属性
    - .fillna(,inplace=True) 填补相应的值，需要计算一个值
        - 补缺失值方法(from sklearn.preprocessing import Imputer)
            -imputer 需要fit_transform
- 处理文本和分类属性
    - from sklearn.preprocessing import LabelEncoder
    - from sklearn.preprocessing import OneHotEncoder
    - from sklearn.preprocessing import LabelBinarizer
        - 同样需要fit_transform(), 注意，返回的是array
- 特征缩放
    - from sklearn.preprocessing import StandardScaler
5. 探索不同的模型并列出最优模型
6. 微调模型并将它们组合成一个很好的解决方案
7. 展示您的解决方案
8. 运行，监控和维护您的系统 

# array 转换成 dataframe
```Python   
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))
```

# 定制转换器
虽然Scikit-Learn提供了许多有用的transformers，但我们需要编写我们自己的transformer来执行诸如自定义清理操作或特定组合属性等任务。 我们希望自定义的transformers与Scikit-Learn的功能（例如管道）无缝协作，并且由于Scikit-Learn依赖于duck typing（不是继承），所以你需要的只是创建一个类并实现三个函数：
* fit（）返回self（）
* transform（）
* fit_transform（）

最后一个只需要添加**TransformerMixin**作为基类即可实现。 此外，如果将**BaseEstimator**添加为基类（并避免构造函数中的* args和** kargs），我们可以获得两个额外的方法（get_params（）和set_params（）），这些方法对自动调整超参数很有用。 例如，这是一个小型transformer类，它添加了我们前面讨论过的组合属性：

```Python
from sklearn.base import BaseEstimator, TransformerMixin

# column index-列索引
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

# Pipeline
```Python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```

```Python
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```
