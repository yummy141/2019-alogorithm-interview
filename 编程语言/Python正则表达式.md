
- > 正则表达式参考：AstrlWind博客/[Python正则表达式指南](http://www.cnblogs.com/huxi/archive/2010/07/04/1771073.html)
- Question：字符串替换。比如，protectedDiv(ave, ave_roll_std_10)这个字符串， 我要把ave替换成data["ave"]， 但是后面一个表达式会被替换为data["ave"]_roll_std_10,  有什么精确匹配的方案吗？
```Python
# 正则表达式
for i in X_train.columns:
    temp_str=str(i)
    expression = re.sub(f'(?<![_\d]){temp_str}(?![_\d])', f'data["{temp_str}"]', expression)

# 先把错误匹配替换为更错误的匹配，再把更不对的替换回来
for i in X_train.columns:
    temp_str = str(i)
    expression = expression.replace("_"+temp_str, "qazwsx" )
    expression = expression.replace(temp_str+"_", "edcrfv" )
    expression = expression.replace(temp_str+"0", "tgbyhn" )
    expression = expression.replace(temp_str+"9", "ujmik" )
    expression = expression.replace(temp_str+"1", "olp;" )

    expression = expression.replace(temp_str, f'data["{temp_str}"]')
    
    expression = expression.replace("olp;", temp_str+"1" )
    expression = expression.replace("ujmik", temp_str+"9")
    expression = expression.replace("tgbyhn", temp_str+"0" )
    expression = expression.replace("edcrfv", temp_str+"_" )
    expression = expression.replace("qazwsx", "_"+temp_str)
```

- 匹配<\a><\a> 和<\b><\b>里的内容
  - > segmentfault/[python正则表达式有多个条件](https://segmentfault.com/q/1010000005689554)
```
<test>
 <a>111</a>
 <c>123</c>
</test>
<test>
 <b>222</b>
 <c>123</c>
</test>
```
```Python
import re
pat = re.compile(r'<(a|b)>(.*?)</\1>', re.M)
for m in pat.finditer:
    print(m.group(2))
```