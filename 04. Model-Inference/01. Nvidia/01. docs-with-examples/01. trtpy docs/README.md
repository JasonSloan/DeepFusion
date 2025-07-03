**trtpy常用命令**

```python
重要！！！key:    nonotest（试验key)    ******(正式key)（正式key请联系微信"手写AI"）
注意trtpy只支持3.10以下的python

安装
pip install trtpy -U
配置快捷键指令(这样以后就不用一直python -m了)
echo alias trtpy=\"python -m trtpy\" >> ~/.bashrc
快捷键生效
source ~/.bashrc
查看当前trtpy信息
trtpy info
设置key
trtpy set-key ******(正式key)（正式key请联系微信"手写AI"）
查看当前课程清单
trtpy list-templ
下载环境
trtpy get-env
拉取一个模板
trtpy get-templ cpp-trt-mnist
进入模板文件夹
cd cpp-trt-mnist
运行模板
make run

查看当前课程代码系列
trtpy list-series
获取课程系列
trtpy get-series cuda-driver-api   # cuda驱动API系列
trtpy get-series cuda-runtime-api   # cuda runtime API系列
trtpy get-series tensorrt-basic  # tensorrt基础系列
trtpy get-series tensorrt-integrate  # tensorrt案例系列
进入系列目录
cd cuda-driver-api
运行当前章节
make run
查看当前章节
trtpy series-detail
查看下一章节
trtpy change-proj next
查看特定章节
trtpy change-proj 1.2
查看上一章节
trtpy change-proj prev
运行当前章节
make run
安装第三方库
trtpy get-cpp-pkg opencv4.2
```

