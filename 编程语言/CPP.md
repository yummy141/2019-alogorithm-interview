目录
---
<!-- TOC -->

- [C++和C相比最大的特点](#c和c相比最大的特点)
- [初始化列表和构造函数有什么区别](#初始化列表和构造函数有什么区别)
- [c++中的多态](#c中的多态)
    - [编译时多态](#编译时多态)
    - [运行时多态](#运行时多态)
    - [虚函数（virtual）](#虚函数virtual)
    - [纯虚函数(pure virtual)](#纯虚函数pure-virtual)
    - [什么是动态联编](#什么是动态联编)
    - [多重继承](#多重继承)
- [static的作用](#static的作用)
    - [为什么要有静态成员函数？](#为什么要有静态成员函数)
- [explicit的作用](#explicit的作用)
- [c++强制类型转换（cast）](#c强制类型转换cast)
- [尽量不要使用using namespace std](#尽量不要使用using-namespace-std)
- [size_t](#size_t)
- [capacity 和 size的区别](#capacity-和-size的区别)
- [vector的emplace_back和push_back](#vector的emplace_back和push_back)
- [左值和右值](#左值和右值)
- [C++中的堆栈分配](#c中的堆栈分配)
- [文件流](#文件流)
- [new int(10) 和 new int[10]](#new-int10-和-new-int10)
- [c++ 11 lambda表达式](#c-11-lambda表达式)
- [const和指针](#const和指针)
- [智能指针](#智能指针)
- [new和malloc的区别](#new和malloc的区别)
- [引用和指针的区别](#引用和指针的区别)
- [单例模式](#单例模式)
- [size_t 和 ssize_t](#size_t-和-ssize_t)
- [(*fp)()，如果fp是函数指针，那么这个语句将调用对应的函数](#fp如果fp是函数指针那么这个语句将调用对应的函数)
- [C++11 标准引入了一个新特性："=delete"函数。程序员只需在函数声明后上“=delete;”，就可将该函数禁用](#c11-标准引入了一个新特性delete函数程序员只需在函数声明后上delete就可将该函数禁用)
- [锁](#锁)

<!-- /TOC -->
面试C++程序员的时候一般都是3板斧，先是基础问答，然后一顿虚函数、虚函数表、纯虚函数、抽象类、虚函数和析构函数、虚函数和构造函数。接着拷贝构造函数、操作符重载、下面是STL，最后是智能指针。
## C++和C相比最大的特点
1）面向对象：封装，继承，多态。
2）引入引用代替指针。
3）const /inline/template替代宏常量。
4）namespace解决重名的问题。
5）STL提供高效的数据结构和算法


## 初始化列表和构造函数有什么区别
> CSDN/[初始化列表和构造函数有什么区别](https://blog.csdn.net/theprinceofelf/article/details/20057359)

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



## c++中的多态
### 编译时多态
- 运算符重载和重载函数
- c++中，名字查找发生在类型检查之前
### 运行时多态
- 引用或指针的静态类型和动态类型不同是c++支持多态的根本所在[c++ primer P537]
- 静态类型是编译已知的
- 动态类型是变量或表达式在内存中的对象类型，只有运行时才能确定
> 注意：通过对象进行调用的虚或非虚函数都在编译时绑定，对象的静态或动态类型永远是一致的。  
> 注意：能使用哪些成员函数仍然是由静态类型决定的

### 虚函数（virtual）
> [csdn](https://blog.csdn.net/worldwindjp/article/details/18909079)
- 基类希望派生类进行覆盖的函数称为虚函数
- 通过virtual使得该函数在运行时执行动态绑定
父类类型的指针指向子类的实例，执行的时候会执行之类中定义的函数

构造函数可以是虚函数吗？
     答案：不能，每个对象的虚函数表指针是在构造函数中初始化的，因为构造函数没执行完，所以虚函数表指针还没初始化好，构造函数的虚函数不起作用。

析构函数可以是虚函数吗？
     答案： 可以，如果有子类的话，析构函数必须是虚函数。否则析构子类类型的指针时，析构函数有可能不会被调用到。
     编译器总是根据类型来调用类成员函数。但是一个派生类的指针可以安全地转化为一个基类的指针。这样删除一个基类的指针的时候，C++不管这个指针指向一个基类对象还是一个派生类的对象，调用的都是基类的析构函数而不是派生类的。如果你依赖于派生类的析构函数的代码来释放资源，而没有重载析构函数，那么会有资源泄漏。

虚函数表是针对类还是针对对象的?
     答案：虚函数表是针对类的，一个类的所有对象的虚函数表都一样。

虚继承和虚基类？
     答案：虚继承是为了解决多重继承出现菱形继承时出现的问题。例如：类B、C分别继承了类A。类D多重继承类B和C的时候，类A中的数据就会在类D中存在多份。通过声明继承关系的时候加上virtual关键字可以实现虚继承。

### 纯虚函数(pure virtual)
- 接口，定义可以用=0
- 含有纯虚函数的类是抽象基类，不能直接创建一个抽象基类的对象。

### 什么是动态联编
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
### 多重继承
> CSDN/[C++多态虚函数表详解(多重继承、多继承情况)](https://blog.csdn.net/qq_36359022/article/details/81870219)  

多重继承下一个类可以由多个虚函数表


## static的作用
1. 修饰普通变量，修改变量的存储区域和生命周期，使变量存储在静态区，在 main 函数运行前就分配了空间，如果有初始值就用初始值初始化它，如果没有初始值系统用默认值初始化它。
2. 修饰普通函数，表明函数的作用范围，仅在定义该函数的文件内才能使用。在多人开发项目时，为了防止与他人命令函数重名，可以将函数定位为 static。
3. 修饰成员变量，修饰成员变量使所有的对象只保存一个该变量，而且不需要生成对象就可以访问该成员。
4. 修饰成员函数，修饰成员函数使得不需要生成对象就可以访问该函数，但是在 static 函数内不能访问非静态成员。

### 为什么要有静态成员函数？
> CSDN/[为什么要有静态成员函数？](https://blog.csdn.net/qq_37375427/article/details/78808900)

静态成员函数为所有对象共有，可以直接通过类名、对象名访问，没有this指针
注意的时，静态成员函数只能访问静态变量。

## explicit的作用

> CSDN/[explicit](https://blog.csdn.net/qq_35524916/article/details/58178072)   

- C++中构造函数有两个作用：
  1. 构造作用
  2. 隐式类型转换操作符
- explicit用来制止类型转换操作，或者叫隐式转换（等号赋值时的转换）

> CSDN/[explicit](https://blog.csdn.net/csdn_tym/article/details/79044651)  
当涉及到用户自定义的类时，可以使用explicit将一个类的实例强制转换为另一个类的实例, 如：`static explicit operator A(B f)`


## c++强制类型转换（cast）
> cnblogs/[c++强制类型转换（cast）](https://www.cnblogs.com/chenyangchun/p/6795923.html)
- static_cast
  - 相当于传统的C语言中的强制转换
- dynamic_cast
  - 用于类层次间的上行和下行转换，以及类之间的交叉转换
- const_cast
  - 用于修改const或volatile
- reinterpret_cast
  - 用在任意指针（或引用）类型之间的转换；以及指针与足够大的整数类型之间的转换；从整数类型（包括枚举类型）到指针类型，无视大小。 
  - 用来辅助哈希函数 
  - static_cast 和 reinterpret_cast 操作符修改了操作数类型. 它们不是互逆的; static_cast 在编译时使用类型信息执行转换, 在转换执行必要的检测(诸如指针越界计算, 类型检查). 其操作数相对是安全的. 另一方面, reinterpret_cast 仅仅是重新解释了给出的对象的比特模型而没有进行二进制转换, 

## 尽量不要使用using namespace std
> CSDN/[尽量不要使用using namespace std](https://blog.csdn.net/dj0379/article/details/11565387)

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


## size_t
size_t是unsigned int的别名

## capacity 和 size的区别
> CSDN/[capacity 和 size的区别](https://bbs.csdn.net/topics/390343778)

size是真实大小，capacity是预申请的内存。
capacity是容量，是可存放字符的个数。
size是大小，是当前已存放字符的个数。
capacity >= size， 具体capcacity大多少，具体的stl库实现决定。

## vector的emplace_back和push_back
emplace_back优势主要在传递类或者结构体之类会比较高效（C++11）

## 左值和右值
> [github](https://github.com/imhuay/Algorithm_Interview_Notes-Chinese/blob/master/C-%E7%BC%96%E7%A8%8B%E8%AF%AD%E8%A8%80/Cpp-C-%E5%B7%A6%E5%80%BC%E4%B8%8E%E5%8F%B3%E5%80%BC.md#move-%E4%B8%8E-forward)

## C++中的堆栈分配
> CSDN/[C++中的堆栈分配](https://blog.csdn.net/sinat_36246371/article/details/55223790)

一般而言，C/C++程序占用的内存分为以下几个部分：

1、栈区（stack）— 由编译器自动分配释放 ，存放函数的参数值，局部变量的值等。其操作方式类似于数据结构中的栈。

2、堆区（heap） — 一般由程序员分配释放， 若程序员不释放，例如malloc、free，程序结束时可能由OS回收 。注意它与数据结构中的堆是两回事，分配方式倒是类似于链表。

3、全局区（静态区）（static）—，全局变量和静态变量的存储是放在一块的，初始化的全局变量和静态变量在一块区域， 未初始化的全局变量和未初始化的静态变量在相邻的另一块区域，程序结束后有系统释放。

4、文字常量区 —常量字符串就是放在这里的。 程序结束后由系统释放。

5、程序代码区—存放函数体的二进制代码。


区别和联系：
1.申请方式    
堆是由程序员自己申请并指明大小，在c中malloc函数 如p1 = (char *)malloc(10);    栈由系统自动分配，如声明在函数中一个局部变量 int b; 系统自动在栈中为b开辟空间
2.申请后系统的响应    
栈：只要栈的剩余空间大于所申请空间，系统将为程序提供内存，否则将报异常提示栈溢出。   
堆：首先应该知道操作系统有一个记录空闲内存地址的链表，当系统收到程序的申请时，会 遍历该链表，寻找第一个空间大于所申请空间的堆结点，然后将该结点从空闲结点链表中删除，并将该结点的空间分配给程序，另外，对于大多数系统，会在这块内 存空间中的首地址处记录本次分配的大小，这样，代码中的delete语句才能正确的释放本内存空间。另外，由于找到的堆结点的大小不一定正好等于申请的大 小，系统会自动的将多余的那部分重新放入空闲链表中。
3.申请大小的限制    栈：在Windows下,栈是向低地址扩展的数据结 构，是一块连续的内存的区域。这句话的意思是栈顶的地址和栈的最大容量是系统预先规定好的，在WINDOWS下，栈的大小是2M（也有的说是1M，总之是 一个编译时就确定的常数），如果申请的空间超过栈的剩余空间时，将提示overflow。因此，能从栈获得的空间较小。    堆：堆是向高地址扩展的数据结构，是不连续的内存区域。这是由于系统是用链表来存储的空闲内存地址的，自然是不连续的，而链表的遍历方向是由低地址向高地址。堆的大小受限于计算机系统中有效的虚拟内存。由此可见，堆获得的空间比较灵活，也比较大。
4.申请效率的比较：    栈由系统自动分配，速度较快。但程序员是无法控制的。 堆是由new分配的内存，一般速度比较慢，而且容易产生内存碎片,不过用起来最方便


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

## const和指针
const 和指针
const 也可以和指针变量一起使用，这样可以限制指针变量本身，也可以限制指针指向的数据。const 和指针一起使用会有几种不同的顺序，如下所示：

const int *p1;
int const *p2;
int * const p3;
在最后一种情况下，指针是只读的，也就是 p3 本身的值不能被修改；在前面两种情况下，指针所指向的数据是只读的，也就是 p1、p2 本身的值可以修改（指向不同的数据），但它们指向的数据不能被修改。

## 智能指针
> [csdn](https://blog.csdn.net/worldwindjp/article/details/18843087)
1. 从较浅的层面看，智能指针是利用了一种叫做RAII（资源获取即初始化）的技术对普通的指针进行封装，这使得智能指针实质是一个对象，行为表现的却像一个指针。
2. 智能指针的作用是防止忘记调用delete释放内存和程序异常的进入catch块忘记释放内存。另外指针的释放时机也是非常有考究的，多次释放同一个指针会造成程序崩溃，这些都可以通过智能指针来解决。
3. 智能指针还有一个作用是把值语义转换成引用语义。C++和Java有一处最大的区别在于语义不同，在Java里面下列代码：
　　Animal a = new Animal();

　　Animal b = a;

     你当然知道，这里其实只生成了一个对象，a和b仅仅是把持对象的引用而已。但在C++中不是这样，

     Animal a;

     Animal b = a;

     这里却是就是生成了两个对象。

shared_ptr实现
```c++
template<typename T>
class SharedPtr{
private:
    T *ptr;
    int *use_count;
public:

    SharedPtr(const SharedPtr<T> &orig): ptr(orig.ptr), use_count( &(++*orig.use_count) ){
        cout << "copyt constructtor : " << *ptr << " refCount = " << *use_count << endl;
    }

    SharedPtr(T *p): ptr(p), use_count(new int(1)) {
        cout << "create object : " << *ptr << " refCount = " << *use_count << endl;
    }

    SharedPtr<T>& operator=(const SharedPtr<T> &rhs){
        if(&rhs != this){
            ++*rhs.use_count;
            if(--*use_count==0){
                cout << "in function operator = . delete" << *ptr << endl;
                delete ptr;
                delete use_count;
            }

            ptr = rhs.ptr;
            use_count = rhs.use_count;
            cout << "in function operator = ." << *ptr <<" refCount = " << *use_count << endl;
            return *this;
        }
        return *this;
    }

    T operator*(){
        if(use_count == 0)
            return nullptr;
        
        return *ptr;
    }

    T* operator->(){
        if(use_count == 0)
            return nullptr;
        return ptr;
    }

    ~SharedPtr(){
        if(ptr && --*use_count == 0){
            cout << *ptr << " refCount = 0. delete the ptr" << endl;
            delete ptr;
            delete use_count;
        }
    }

    int getCount()
    {
        return *use_count;
    }

};


int main(){
    SharedPtr<string> pstr(new string(" first object "));
    SharedPtr<string> pstr2(pstr);
    SharedPtr<string> pstr3(new string(" second object "));
    //  SharedPtr<string> pstr4;

    // pstr4 = pstr2;
    return 0;
}
```

## new和malloc的区别
> CSDN/[new和malloc的区别](https://www.cnblogs.com/QG-whz/p/5140930.html#_label1_0)
1. new和delete可以调用构造和析构函数
2. new不需要显示指定申请内存大小，而malloc需要
3. malloc返回无类型指针，需要强制转换，而new返回对象类型指针
4. 申请失败，new会抛出异常，而malloc不会
5. new，delete可以被重载，而malloc不可以

## 引用和指针的区别
1. 引用不可以为空但指针可以为空
2. 引用不可以改变指向，指针可以
3. 引用的大小是所指变量的大小，而指针是指针本身大小
4. 引用比指针更安全

## 单例模式
> CSDN/[单例模式](https://www.cnblogs.com/dupengcheng/p/7205527.html?tdsourcetag=s_pctim_aiomsg)  

类只有一个instance的模式
```c++
// 示例
template<typename T>
class CSingleton
{
private:
    CSingleton()
    {
    }
    static CSingleton *p;
public:
    static CSingleton* getInstance()
    {
        // static CSingleton *p = new CSingleton();
        return p;
    }
};
template<typename T>
CSingleton<T>* CSingleton<T>::p = new CSingleton<T>(); 

int main() 
{ 

   auto t = CSingleton<int>::getInstance();
	return 0; 
} 
```

## size_t 和 ssize_t
ssize_t是有符号整型，在32位机器上等同与int，在64位机器上等同与long int;  
而size_t是无符号整型

## (*fp)()，如果fp是函数指针，那么这个语句将调用对应的函数  
> CSDN/[详解C/C++函数指针声明 ( *( void(*)())0)();](http://www.cnblogs.com/yaowen/p/4797354.html)

## C++11 标准引入了一个新特性："=delete"函数。程序员只需在函数声明后上“=delete;”，就可将该函数禁用

## 锁
[csdn](https://www.nowcoder.com/questionTerminal/554355eea5aa44d697a3a4bc99795207?page=7&onlyReference=false)