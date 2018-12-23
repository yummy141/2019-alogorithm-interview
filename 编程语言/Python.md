## [可变对象和不可变对象](http://www.cnblogs.com/congbo/archive/2012/11/20/2777031.html)
python中，对象分为可变(mutable)和不可变(immutable)两种类型。

元组（tuple)、数值型（number)、字符串(string)均为不可变对象，而字典型(dictionary)和列表型(list)的对象是可变对象。


##  Python paritial（）函数
其实就是函数调用的时候，有多个参数，其中又有已知的参数，这样就可以重新定义一个函数
```Python
from functools import partial

my_batch_norm_layer = partial(tf.layers.batch_normalization,
                              training=training, momentum=0.9)

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = my_batch_norm_layer(hidden1)
```