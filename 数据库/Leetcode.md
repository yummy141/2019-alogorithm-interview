<!-- TOC -->

- [语法](#语法)
    - [sql语句是否区分大小写](#sql语句是否区分大小写)
    - [sql中的单引号](#sql中的单引号)
    - [sql Alias](#sql-alias)
- [组合两个表](#组合两个表)
- [第二高的薪水](#第二高的薪水)
    - [第N高的薪水](#第n高的薪水)
- [分数排名](#分数排名)
- [连续出现的数字](#连续出现的数字)
- [超过经理收入的员工](#超过经理收入的员工)
- [查找重复的电子邮箱](#查找重复的电子邮箱)
- [从不订购的客户](#从不订购的客户)
- [部门中工资最高的员工](#部门中工资最高的员工)
- [部门工资前三高的员工](#部门工资前三高的员工)

<!-- /TOC -->
## 语法
### sql语句是否区分大小写
`SQL`对大小写不敏感
一、在windows系统中不区分大小写
二、在Linux和Unix系统中字段名、数据库名和表名要区分大小写

### sql中的单引号
`SQL`使用单引号来环绕文本值（大部分数据库系统也接受双引号）。如果是数值，请不要使用引号。

### sql Alias
```sql
-- 表的 SQL Alias 语法
SELECT column_name(s)
FROM table_name
AS alias_name

-- 列的 SQL Alias 语法
SELECT column_name AS alias_name
FROM table_name
```

## 组合两个表 

**描述**
```
表1: Person

+-------------+---------+
| 列名         | 类型     |
+-------------+---------+
| PersonId    | int     |
| FirstName   | varchar |
| LastName    | varchar |
+-------------+---------+
PersonId 是上表主键
表2: Address

+-------------+---------+
| 列名         | 类型    |
+-------------+---------+
| AddressId   | int     |
| PersonId    | int     |
| City        | varchar |
| State       | varchar |
+-------------+---------+
AddressId 是上表主键
 
编写一个 SQL 查询，满足条件：无论 person 是否有地址信息，都需要基于上述两表提供 person 的以下信息：

FirstName, LastName, City, State
```
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

## 分数排名
- 对于`Score`一个分数，新开一个表，找出表中有多少个大于或等于该分数的不同的分数，然后按降序排列即可
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

## 查找重复的电子邮箱

**思路**
- 统计条件查询，要么使用`group by`后`where`, 要么使用`groupby`的`having`
- 
**描述**
```
编写一个 SQL 查询，查找 Person 表中所有重复的电子邮箱。

示例：

+----+---------+
| Id | Email   |
+----+---------+
| 1  | a@b.com |
| 2  | c@d.com |
| 3  | a@b.com |
+----+---------+
根据以上输入，你的查询应返回以下结果：

+---------+
| Email   |
+---------+
| a@b.com |
+---------+
```
```sql
select Email from
(
    select Email, count(Email) as num
    from Person 
    group by Email
)as statistic
where num > 1
;

-- or
select Email from Person group by Email having count(Email) > 1
```

## 从不订购的客户
- 
**描述**
```
某网站包含两个表，Customers 表和 Orders 表。编写一个 SQL 查询，找出所有从不订购任何东西的客户。

Customers 表：

+----+-------+
| Id | Name  |
+----+-------+
| 1  | Joe   |
| 2  | Henry |
| 3  | Sam   |
| 4  | Max   |
+----+-------+
Orders 表：

+----+------------+
| Id | CustomerId |
+----+------------+
| 1  | 3          |
| 2  | 1          |
+----+------------+
例如给定上述表格，你的查询应返回：

+-----------+
| Customers |
+-----------+
| Henry     |
| Max       |
+-----------+
```

```sql
select Name as Customers
from Customers 
where Customers.id not in
(
    select CustomerId from Orders
);

select Customers.Name as Customers
from Customers
left join Orders on Orders.CustomerId = Customers.Id
where Orders.CustomerId is NULL;

```

## 部门中工资最高的员工

**描述**
```
Employee 表包含所有员工信息，每个员工有其对应的 Id, salary 和 department Id。

+----+-------+--------+--------------+
| Id | Name  | Salary | DepartmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 70000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
+----+-------+--------+--------------+
Department 表包含公司所有部门的信息。

+----+----------+
| Id | Name     |
+----+----------+
| 1  | IT       |
| 2  | Sales    |
+----+----------+
编写一个 SQL 查询，找出每个部门工资最高的员工。例如，根据上述给定的表格，Max 在 IT 部门有最高工资，Henry 在 Sales 部门有最高工资。

+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| Sales      | Henry    | 80000  |
+------------+----------+--------+
```
```sql
select d.name as Department, e.name as Employee, e.salary from Department d, Employee e where d.Id = e.DepartmentId and e.Salary = (select max(Salary) from Employee where DepartmentId = d.Id);
```

## 部门工资前三高的员工
- 再建一个表，有点类似于嵌套循环的意思
```sql
# Write your MySQL query statement below
select d.Name as Department, e.Name as Employee, e.Salary as Salary from Employee e, Department d 
where e.DepartmentId = d.Id
and (select count(distinct e2.salary) from Employee e2 
where e2.DepartmentId = e.DepartmentId and e2.Salary > e.Salary) < 3
order by e.DepartmentId, e.Salary DESC
```