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
    > housing = strat_train_set.drop("median_house_value", axis=1)
    > housing_labels = strat_train_set["median_house_value"].copy()
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
5. 探索不同的模型并列出最优模型
6. 微调模型并将它们组合成一个很好的解决方案
7. 展示您的解决方案
8. 运行，监控和维护您的系统 

# array 转换成 dataframe
```Python   
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index = list(housing.index.values))
```