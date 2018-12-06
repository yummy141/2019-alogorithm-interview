# [初始化列表和构造函数有什么区别](https://blog.csdn.net/theprinceofelf/article/details/20057359)
类的初始化操作包含四个部分
1. 初始化列表只能初始化非静态数据成员
2. 构造函数体
    - 只可修改能被修改的静态成员
3. 类外初始化
    - 所有类静态数据成员
        -除了static const int， 既可再这里初始化，也能在成员声明处初始化
4. 类中声明时直接赋值
    - static const int

简化版：
1. 所有static成员变量在类外初始化（不管它是const，是引用，还是没默认构造函数的对象）
    - 静态成员属于类作用域，但不属于类对象，和普通的static变量一样，程序一运行就分配内存并初始化，生命周期和程序一致。所以，在类的构造函数里初始化static变量显然是不合理的。静态成员其实和全局变量地位是一样的，只不过编译器把它的使用限制在类作用域内（不是类对象，它不属于类对象成员），要在类的定义外（不是类作用域外）[初始化](http://www.cnblogs.com/zhoug2020/archive/2012/08/31/2665451.html)。
2. 普通成员变量，是const, 是引用，是没默认构造函数的，必须在初始化列表初始化
```C++
class A {
    const int x; // 或者 int &x
public:
    A() {
        this->x = 1; /* Error! */
    }
};
```
3. 普通成员变量，需要复杂运算的初始化变量，应该在构造函数内初始化，否则尽量在初始化列表中初始化。


# [explicit的作用](https://blog.csdn.net/qq_35524916/article/details/58178072)
C++中构造函数有两个作用：
1. 顾名思义的构造作用
2. 类型转换操作符
explicit用来制止类型转换操作，或者叫隐式转换（等号赋值时的转换）

# [尽量不要使用using namespace std](https://blog.csdn.net/dj0379/article/details/11565387)
可以在cpp中使用，不在头文件中使用。因为头文件中被cpp使用，就默认using namespace std了，容易导致问题。
解决方案：
1. 使用作用域来限制
```c++
void temp()  
{  
  using namespace std;  
  string test = "fooBar";  
}  
```
2. 使用typedef或者using
```c++
typedef std::map<std::string, long> ClientNameToZip;  
ClientNameToZip clientLocations;  
using ClientNameToZip = std::map<std::string, long>  //注意C++鼓励使用using来别名
```
3. using std::string

# [为什么要有静态成员函数？](https://blog.csdn.net/qq_37375427/article/details/78808900)
静态成员函数为所有对象共有，可以直接通过类名、对象名访问，没有this指针？
注意的时，静态成员函数只能访问静态变量。

# [capacity 和 size的区别](https://bbs.csdn.net/topics/390343778)
size是真实大小，capacity是预申请的内存。