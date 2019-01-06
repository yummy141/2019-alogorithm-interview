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


## 分数排名
```sql
select Score, (select count(distinct score) from Scores where score >= s.Score) as Rank from Scores s order by Score desc; 
```

## 连续出现的数字
```sql
select distinct l1.Num as ConsecutiveNums from Logs l1, Logs l2, Logs l3 
where l1.Num = l2.Num and l2.Num = l3.Num and l1.Id = l2.Id-1 and l2.Id = l3.Id-1
```

## 超过经理收入的员工
```sql
select e1.Name as Employee from Employee e1, Employee e2  where e2.ID = e1.ManagerId and e2.Salary < e1.Salary 
```