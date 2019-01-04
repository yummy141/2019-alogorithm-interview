## 组合两个表 
sql之left join、right join、inner join的区别  
left join(左联接) 返回包括左表中的所有记录和右表中联结字段相等的记录   
right join(右联接) 返回包括右表中的所有记录和左表中联结字段相等的记录  
inner join(等值连接) 只返回两个表中联结字段相等的行  
full join 全连接  
``` sql
SELECT a.FirstName, a.LastName, b.City, b.State FROM Person AS a LEFT JOIN Address AS b ON a.PersonID=b.PersonID
```


## 第二高的薪水
``` sql
select max(Salary) as SecondHighestSalary from Employee where Salary < (select max(Salary) from Employee)

select IFNULL((select DISTINCT Salary from Employee order by Salary DESC limit 1,1), null) as SecondHighestSalary
```

### 第N高的薪水
``` sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  declare m int;
  SET m = N-1;
  RETURN (
      # Write your MySQL query statement below.
      select distinct salary from Employee order by salary desc limit m,1
  );
END
```

## sql语句是否区分大小写
一、在windows系统中不区分大小写
二、在Linux和Unix系统中字段名、数据库名和表名要区分大小写