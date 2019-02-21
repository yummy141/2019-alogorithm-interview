## Logistic Regression

### 什么叫回归？
相关性分析，用一个数学模型直接描绘出X到y的映射关系。

线性回归即为：y=ax+b

而由于y的取值是零散的，对数几率回归的公式难以直接写出，所以必须用模型表示y=0和y=1的概率。

### 定义
对数几率回归属于判别模型，通过计算不同分类的概率，比较概率值来判断类别。
方法：通过`log odds(对数几率)`，将线性函数`wx+b`转换为概率
目标：
1. 拟合决策边界，比如`wx+b>0 或 wx+b<0`的分类
2. 建立决策边界和概率的联系

**logit函数（log odds对数几率）**    
$logit = ln（\frac{p}{1-p}）$, 假如$Q=\frac{e^{wx+b}}{1+e^{wx+b}}$, 则$logit=wx+b$

## 用于预测样本类别的假设函数为sigmoid函数
$h_{\theta}(x)=sigmoid(\theta*x)=\frac{1}{1+e^{-\theta*x}}=\frac{e^{\theta*x}}{e^{\theta*x}+1}$,$\theta*x=0$是分类的临界点

## softmax是sigmoid函数的推广
令二分类中的$w= \Delta w=w_1-w_0$, $sigmoid(\Delta w*x) = \frac{e^{(w_1-w_0)*x}}{1+e^{(w_1-w_0)*x}}$

等式两边同除以$e^{-w_0*x}$, $sigmoid(\Delta w*x) = \frac{e^{w_1*x}}{e^{w_1*x}+e^{w_0*x}}$

则softmax $P(y=j|x)=\frac{e^{x^T*w_j}}{\Sigma e^{x^T*w_k}}$