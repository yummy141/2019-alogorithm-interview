## [初始化列表和构造函数有什么区别](https://blog.csdn.net/theprinceofelf/article/details/20057359)
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

## 什么是动态联编
在C++中由于虚函数导致的多态性，一个类函数的调用并不是在编译时刻被确定的（编写代码的时候并不能确定被调用的是基类的函数还是哪个派生类的函数，所以被成为“虚”函数），也就导致了静态联编不知道该把函数调用与哪一个子类对象的函数功能实现绑定一起，而是在运行时刻根据具体调用函数的对象类型来确定。
c++中每个类都会维护一个虚函数表，如果子类中重写了某个虚函数，则表中相应的指针会被替代。
```c++
class base {
public:
    int v1;
    virtual void f1() {}
    virtual void f2() {}
}

class sub : public base {
    int v2;
    virtual void f1() {}
}
```

## [explicit的作用](https://blog.csdn.net/qq_35524916/article/details/58178072)
C++中构造函数有两个作用：
1. 顾名思义的构造作用
2. 类型转换操作符
explicit用来制止类型转换操作，或者叫隐式转换（等号赋值时的转换）

## [尽量不要使用using namespace std](https://blog.csdn.net/dj0379/article/details/11565387)
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

## [为什么要有静态成员函数？](https://blog.csdn.net/qq_37375427/article/details/78808900)
静态成员函数为所有对象共有，可以直接通过类名、对象名访问，没有this指针
注意的时，静态成员函数只能访问静态变量。

## [capacity 和 size的区别](https://bbs.csdn.net/topics/390343778)
size是真实大小，capacity是预申请的内存。

## vector的emplace_back和push_back
emplace_back优势主要在传递类或者结构体之类会比较高效（C++11）

## [左值和右值](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese/blob/master/C-%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80/Cpp-C-%E5%B7%A6%E5%80%BC%E4%B8%8E%E5%8F%B3%E5%80%BC.md#move-%E4%B8%8E-forward)

## [C++中的堆栈分配](https://blog.csdn.net/sinat_36246371/article/details/55223790)
一般而言，C/C++程序占用的内存分为以下几个部分：

1、栈区（stack）— 由编译器自动分配释放 ，存放函数的参数值，局部变量的值等。其操作方式类似于数据结构中的栈。

2、堆区（heap） — 一般由程序员分配释放， 若程序员不释放，例如malloc、free，程序结束时可能由OS回收 。注意它与数据结构中的堆是两回事，分配方式倒是类似于链表。

3、全局区（静态区）（static）—，全局变量和静态变量的存储是放在一块的，初始化的全局变量和静态变量在一块区域， 未初始化的全局变量和未初始化的静态变量在相邻的另一块区域，程序结束后有系统释放。

4、文字常量区 —常量字符串就是放在这里的。 程序结束后由系统释放。

5、程序代码区—存放函数体的二进制代码。


## 文件流
istrem 基类
ostrem 基类
ifsteram 读
ofstream 写
fstream 读写
文件类型分为两种：文本文件和二进制文件

stringstream是字符串流。它将流与存储在内存中的string对象绑定[起来](http://www.cnblogs.com/propheteia/archive/2012/07/12/2588225.html)。
stringstream无法直接用不同的分隔符（如“，”），使用[getline函数](https://stackoverflow.com/questions/49708987/stringstream-delimeter)


## new int(10) 和 new int[10]
new int(10)表示创建了一个int指针，并初始化它的实例的值为10。
new int[10]表示创建一个10个大小的int指针数组。


## c++ 11 lambda表达式
[capture list] (params list) -> return type {function body}
[capture list] (params list) {function body}
[capture list] {function body}
``` c++ 
auto x = [](int a){cout << a << endl;}{123};
auto f = [x]{cout << x << endl;};x=321;f();

a=123
auto f=[=]{cout<< a << endl;}f();

auto x = [a]()mutable {cout << ++a << endl;};  //加入mutable可改动外部变量

```