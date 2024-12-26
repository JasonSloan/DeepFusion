# 一. valgrind

[官方文档](https://valgrind.org/docs/manual/QuickStart.html)

注意使用valgrind时, 需要增加编译选项, 否则不会显示内存泄漏的行号

```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
```

## 1. 安装

```bash
apt-get update && apt-get install valgrind
```

## 2. 使用命令

```bash
valgrind --tool=memcheck --leak-check=yes ./workspace/mainproject
# --tool=<name>: valgrind除了可以检查内存泄漏还能分析性能等, 我们只使用memcheck检查内存泄漏
# --leak-check=no|summary|full：指定是否对 内存泄露 给出详细信息
# 更多参考: valgrind --help
```

## 3. 内存泄漏类型

* definitely lost：

  意味着你的程序正在泄漏内存——修复这些泄漏！

* indirectly lost：

  意味着你的程序在基于指针的结构中泄漏了内存。（例如，如果二叉树的根节点“definitely lost”，则所有子节点都将“indirectly lost”。）如果你修复了“definitely lost”的泄漏，“indirectly lost”的泄漏应该会消失

* possibly lost：

  表示您的程序正在泄漏内存，除非您对指针做了一些不寻常的事情，这可能会导致它们指向已分配块的中间。如果您不想看到这些报告，请使用 --show-possibly-lost=no。但有时候第三方库也会存在possibly lost, 比如opencv的cv::resize, 这种可忽略

* still reachable：

  意味着你的程序可能没问题, 它没有释放一些本该释放的内存。这很常见，而且通常是合理的。如果你不想看到这些报告，请不要使用 --show-reachable=yes。但有时候第三方库也会存在possibly lost, 比如opencv的cv::resize, 这种可忽略

* suppressed：

  表示泄漏错误已被抑制。默认抑制文件中有一些抑制。您可以忽略抑制的错误。

## 4. 报告分析

**提示一：“Conditional jump or move depends on uninitialised value(s)”**

可能是某些变量未初始化！！

**提示二："Invalid write of size 4"**

访问越界！！

**提示三：“Source and destination overlap in memcpy()”**

程序试图将数据从一个位置复制到另一个位置，要读取的范围与要写入的范围相交。使用memcpy在重叠区域之间传输数据可能会破坏结果；memmove是在这种情况下的正确函数。

**提示四：“Invalid free()”**

程序多次尝试释放非堆地址或释放同一块内存。



