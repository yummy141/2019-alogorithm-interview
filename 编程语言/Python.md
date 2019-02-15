## [可变对象和不可变对象](http://www.cnblogs.com/congbo/archive/2012/11/20/2777031.html)
python中，对象分为可变(mutable)和不可变(immutable)两种类型。

元组（tuple)、数值型（number)、字符串(string)均为不可变对象，而字典型(dictionary)和列表型(list)的对象是可变对象。

## Python 中 （&，|）和（and，or）之间的区别
- example: a，b做运算
  - 如果a,b是数值变量
    - `& |`表示位运算
    - `and or` 根据非0判断
      - 注意 `1 and 2` 返回`2`
  - 如果a,b是逻辑变量
    - 则一致
- 值得提及的是在DataFrame的切片过程，要注意逻辑变量的使用，
  - 需要求得满足多个逻辑条件的数据时，要使用`&` 和`|`，在某些条件下用`and or`会报错`‘ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all()`

##  Python paritial（）函数
其实就是函数调用的时候，有多个参数，其中又有已知的参数，这样就可以重新定义一个函数
```Python
from functools import partial

my_batch_norm_layer = partial(tf.layers.batch_normalization,
                              training=training, momentum=0.9)

hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
bn1 = my_batch_norm_layer(hidden1)
```
## Python reduce()函数
```Python
from functools import reduce # Python3
reduce(function, iterable[, initializer]) #function 有两个参数

fac = lambda n:reduce(lambda x,y:x*y,range(1,n+1)) # 连乘
add = reduce(lambda x, y: x+y, [1,2,3,4,5]) # 连加
```
## 如何得到一个对象的方法
dir(对象) 

## 巧用range，划分任务
```Python
        jobs = range(n_strucs)
        jobs = [jobs[_i::COMM.size] for _i in range(COMM.size)] 
```

## Python库
- codecs 
```Python
# 相比自带的open函数 读取写入进行自我转码
 with codecs.open(FLAGS.input_file, encoding='utf-8') as f: 
        text = f.read()
```

- easydict 
可以用属性做字典
```Python
__C = edict()
__C.seed = 0
```

