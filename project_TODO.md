## web server认知
> [c++实现HTTP](https://blog.csdn.net/querw/article/details/6593328)
- CHTTPServer
  - 内存实现
    - 端口模型？
    - 线程
    - 监控端口事件
    - 维持URL和服务器上真实文件的对应关系
    - 响应
- CHTTPRequest
  - 对客户端请求的包装
  - 请求是一个字符串
- CHTTPResponse
  - 对服务器响应的包装
- CHTTPContent对象
  - 代表了客户端所请求的资源(URL)
  - 它可能是一个文件,也可能就是一段服务器即时生成的HTML/TEXT文本,比如某个目录的文件列表,或者是一个出错信息,如HTTP404文件未找到的提示.