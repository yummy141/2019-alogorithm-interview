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

## pip相关
1. 更新包 `pip install <package> --upgrade`
2. 列出已经安装的包 `pip list installed`


## tensorboard
tensorboard --logdir=/path/to/log-directory --host=localhost
打开网页输入localhost:6006

## 新建环境后安装（未测试）
1. 新建环境 `virtualenv --system=site-packages - p python3.6 ./py3`
2. 安装包 `pip3 install tensorflow==1.12 jupyter matplotlib numpy pandas seaborn`