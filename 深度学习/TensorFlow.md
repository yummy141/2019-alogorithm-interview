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

<!-- /TOC -->

## 变量
- `tf.Variable`, 返回值是变量（特殊张量）
- 与张量不同
    - 张量的生命周期通常都随依赖的计算完成而结束， 内存也随即释放
    - 变量则常驻内存，在每一步训练时不断更新
- `tf.Variable(<initial-value>, name=<optional-name>)`
- 可以使用`assigin`或`assign_xxx`重新给变量赋值

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

# 优化器
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