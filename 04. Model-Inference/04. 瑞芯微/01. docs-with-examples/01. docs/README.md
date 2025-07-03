# 一. 参考文档

**RKNN操作步骤参考:**

**[官方github](https://github.com/rockchip-linux/rknn-toolkit2)提供的RKNPU2_SDK中的v1.6.0(或者更高的release版本)下的Quick_Start文档**

# 二. ssh的安装

```bash
 apt-get remove openssh*
 apt install openssh-client
 apt install openssh-server
```

# 三. 查看NPU和GPU和RGA使用率

```bash
cat /sys/kernel/debug/rknpu/load
cat /sys/class/devfreq/fb000000.gpu/load
cat /sys/kernel/debug/rkrga/load
```

# 四. CPU, GPU, NPU定频

[链接](https://zhuanlan.zhihu.com/p/678024036?utm_campaign=shareopn&utm_medium=social&utm_psn=1753489687187283969&utm_source=wechat_session)

# 五. 模型编译

**编译环境搭建: **

创建一个新的conda环境, 指定好python版本, 按照本文件夹下的requirements对应的python版本安装, 建议指定使用阿里源, 否则安装会出现 No matching distribution found for tf-estimator-nightly==2.8.0.dev2021122109
**模型编译:**

使用该文件夹下的convert.py

**注意:**

在YOLOv5中, 如果是想使用fp16类型在RK平台上推理的话, 将模型正常导出即可;

如果是想使用Int8量化后在RK平台上推理, 那么模型导出前需要去掉乘以anchor那些操作, 因为如果乘以anchor那些操作加进去, 模型的输出值的范围就是[0, 640], 但是模型的置信度以及类别概率的值域范围仍然为[0, 1]之间, 但是xywh的值域范围已经到[0, 640]之间了, 如果做量化, 那么就会把[0, 640]之间的数映射到[-128, 127]之间, 但是置信度以及类别概率又是集中于范围边界的[0, 1]之间, 他们量化后集中在[-128, 127]的一个bin中, 这样Int8模型在推理出结果后, 想再反量化回来的时候, xywh是可以反量化回来的, 但是置信度以及类别概率是反量化不回来了

在导出前, 需要修改的代码如下:

![](assets/export.jpg)

# 六. 模型推理

见cpp文件夹: 

本文件夹下YOLOv5的推理实现了: 

**FP16和Int8类型+单batch和多batch+生产者消费者模型+RAII+接口+异步返回**

# 七. 其他注意事项

前处理的letterbox和cvtCOLOR是使用瑞芯微自己开发的RGA做的, 不是opencv做的, RGA比opencv的好处在于RGA可以大大节省cpu的使用率, 但是RGA在高频使用时会出错(并发越多, 越容易出错, 在image_utils.c中的imfill的API出错), 但是实际上出错也不会影响推理, 结果也是对的, 如果报错信息多, 可以注释掉imfill, 直接令ret_rga = -1。但是如果影响到了结果, 那么解决办法就是sleep 1-5ms

使用rga处理图像时,要保证原图的宽必须是16的倍数,否则会出错, 高可以不是16的倍数

在使用多batch推理时, 需要将batch-size设置为3的倍数(RKNN只支持固定batch-size的多batch推理, 不支持动态batch推理)