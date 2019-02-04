目录
---
<!-- TOC -->

- [变量](#变量)
- [占位符](#占位符)
- [操作](#操作)
- [会话](#会话)
    - [会话中估算张量的另外两种方法](#会话中估算张量的另外两种方法)
- [Saver使用](#saver使用)
- [优化器](#优化器)
- [Tensorboard](#tensorboard)
- [线性回归示例](#线性回归示例)

<!-- /TOC -->

## 变量
- `tf.Variable`, 返回值是变量（特殊张量）
- 与张量不同
    - 张量的生命周期通常都随依赖的计算完成而结束， 内存也随即释放
    - 变量则常驻内存，在每一步训练时不断更新
- `tf.Variable(<initial-value>, name=<optional-name>)`
- 可以使用`assigin`或`assign_xxx`重新给变量赋值
- `get_variable`与`Variable`类似但这个方法可以在`retrain`时让程序快速找到这个值

## 占位符
- 占位符操作表示图外输入的数据
- `x = tf.placeholder(tf.int16, shape=(), name="x")`

## 操作
- 每个节点都对应一个操作，而节点包括存储（有状态的变量）、计算、数据（占位符）节点，因此操作其实就是节点的抽象
- 操作的输入输出都是张量或者操作

## 会话
- 会话提供了估算张量和执行操作的运行环境
```Python
# 1. 创建会话
sess = tf.Session(target=..., graph=..., config=...)
# 2. 估算张量或执行操作
sess.run(...)
# 3. 关闭会话
sess.close()
```
### 会话中估算张量的另外两种方法
- `Tensor.eval`和`Operation.run`，本质上和`sess.run()`等价
```Python
with tf.Session() as sess:
    # Operation.run()
    tf.gloabl_variables_initializer().run()
    # Tensor.eval()
    fetch = y.eval(feed_dict={x: 3.0})
```

## Saver使用
```Python
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')
# 指定需要保存和恢复的变量
saver = tf.train.Saver({'v1': v1, 'v2': v2})
saver = tf.train.Saver([v1, v2])
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})
# 保存变量的方法
tf.train.saver.save(sess, 'my-model', global_step=0) # ==> filename: 'my-model-0'
# 恢复的方法
saver.restore(sess, './summary/test.ckpt-0')
```

## 优化器
- 两种方法
```Python
# 1. 计算梯度
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
grads_and_vars = optimizer.compute_gradients(loss, var_list, ...)
# 2. 处理梯度
clip_grads_and_vars = [(tf.clip_by_value(grad, -1.0, 1.0), var) fro grad, var in grads_and_vars]
# 3. 应用梯度
train_op = optimizer.apply_gradients(clip_gradients_and_vars)
```

```Python
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.Variable(0, name='global_step`, trainable=False)
train_op = optimizer.minimize(loss, global_step=global_step)
```

## Tensorboard
- 为什么要使用`tf.name_scope()`?
    - 让可视化图更有层次，更加清晰
- 而可视化数据需要创建会话（`Session`），然后用户使用`FileWriter`实例将数据写入`Event file`,最后利用`TensorBoard`读取数据
- 以上操作由`tf.summary`来实现，功能是获取和输出模型的序列化数据
    - 包含三个类`FileWriter`,`Summary`,`Event`



## 线性回归示例
```Python
import pandas as pd
import numpy as np
import tensorflow as tf

def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


df = normalize_feature(pd.read_csv('data1.csv',
                                   names=['square', 'bedrooms', 'price']))

# ones是n行1列的数据框，表示x0恒为1
ones = pd.DataFrame({'ones': np.ones(len(df))})
# 根据列合并数据
df = pd.concat([ones, df], axis=1)  

X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)

# 学习率 alpha, 训练全量数据集的轮数epoch
alpha = 0.01 
epoch = 500  

with tf.name_scope('input'):
    # 输入 X，形状[47, 3]
    X = tf.placeholder(tf.float32, X_data.shape, name='X')
    # 输出 y，形状[47, 1]
    y = tf.placeholder(tf.float32, y_data.shape, name='y')

with tf.name_scope('hypothesis'):
    # 权重变量 W，形状[3,1]
    W = tf.get_variable("weights",
                        (X_data.shape[1], 1),
                        initializer=tf.constant_initializer())
    # 假设函数 h(x) = w0*x0+w1*x1+w2*x2, 其中x0恒为1
    # 推理值 y_pred  形状[47,1]
    y_pred = tf.matmul(X, W, name='y_pred')

with tf.name_scope('loss'):
    # 损失函数采用最小二乘法，y_pred - y 是形如[47, 1]的向量。
    # tf.matmul(a,b,transpose_a=True) 表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
    # 损失函数操作 loss
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)
with tf.name_scope('train'):
    # 随机梯度下降优化器 opt
    train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 创建FileWriter实例，并传入当前会话加载的数据流图
    writer = tf.summary.FileWriter('./summary/linear-regression-1', sess.graph)
    # 开始训练模型
    # 因为训练集较小，所以每轮都使用全量数据训练
    for e in range(1, epoch + 1):
        sess.run(train_op, feed_dict={X: X_data, y: y_data})
        if e % 10 == 0:
            loss, w = sess.run([loss_op, W], feed_dict={X: X_data, y: y_data})
            log_str = "Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[0]))

# 关闭FileWriter的输出流
writer.close()            


```