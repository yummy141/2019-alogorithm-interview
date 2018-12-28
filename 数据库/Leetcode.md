## 组合两个表 
sql之left join、right join、inner join的区别
left join(左联接) 返回包括左表中的所有记录和右表中联结字段相等的记录 
right join(右联接) 返回包括右表中的所有记录和左表中联结字段相等的记录
inner join(等值连接) 只返回两个表中联结字段相等的行
full join 全连接
``` sql
SELECT a.FirstName, a.LastName, b.City, b.State FROM Person AS a LEFT JOIN Address AS b ON a.PersonID=b.PersonID
```
