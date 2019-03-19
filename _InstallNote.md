## jupyter notebook相关
1. 列出所有的kernel `jupyter kernelspec list`
2. 改变notebook主题
```
pip install --upgrade jupyterthemes
jt -t oceans16 -f fira -fs 13 -cellw 90% -ofs 11 -dfs 11 -T
```
3. 将系统自带的kernel显式地放入jupyter中
```
jupyter kernelspec list
python3 -m ipykernel install --user --name=<>
```
4. 去除warning
```Python
import warnings
warnings.filterwarnings('ignore')
```

## pip相关
1. 更新包 `pip install <package> --upgrade`
2. 列出已经安装的包 `pip list installed`


## tensorboard
tensorboard --logdir=/path/to/log-directory --host=localhost
打开网页输入localhost:6006

## 新建环境后安装（未测试）
1. 新建环境 `virtualenv --system=site-packages - p python3.6 ./py3`
2. 安装包 `pip3 install tensorflow==1.12 jupyter matplotlib numpy pandas seaborn`

## windows上c++配置
下载`MinGW`，然后在环境变量中添加`D:\MinGW\mingw32\bin`, 即可使用g++
`sublimetext`中可以直接使用

## VScode配置c++
> [知乎]/(https://www.zhihu.com/question/30315894)

- C/C++（就是有些教程里的cpptools）
- C/C++ Clang Command Adapter：提供静态检测（Lint），很重要
- Code Runner：右键即可编译运行单文件，很方便

## 简单编译
> [csdn](https://blog.csdn.net/yc461515457/article/details/50907393)

clang++ precal.cpp -o precal.exe -Wall -g -Og -static-libgcc -fcolor-diagnostics --target=x86_64-w64-mingw -std=c++17

## Makefile
- `$@`代表目标文件
- `$^`代表所有的依赖文件
- `$<`代表第一个依赖文件

## visual studio
> 批量注释: Ctrl+K,Ctrl+C   
> 取消注释: Ctrl+K,Ctrl+U