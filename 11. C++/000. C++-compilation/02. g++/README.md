# 一. g++基本使用

```bash
# 参考视频：https://www.bilibili.com/video/BV1fy4y1b7TC/?p=8&spm_id_from=333.1007.top_right_bar_window_history.content.click&vd_source=2fa3840975cc19817a9a15ddf8a1a81b
位于哔哩哔哩：收藏：g++、Cmake

# GCC默认头文件搜索路径
echo | gcc -v -x c -E -
/usr/lib/gcc/x86_64-linux-gnu/7/include
/usr/local/include
/usr/lib/gcc/x86_64-linux-gnu/7/include-fixed
/usr/include/x86_64-linux-gnu
/usr/include


# 生成二进制文件
# g++ -c test.cpp  # 生成后缀为.o的二进制文件

# 代码优化
g++ test.cpp -O2 -o a.out  # O2代表对代码进行优化，如果代码中有循环，那么会优化循环

# 代码执行时间测试
time ./a,out    # 使用time可以查看代码执行时间

# 动态库/静态库的链接
g++ -L/home/mylibfolder -lmytest test.cpp   # 如果有一个库的名字叫libmytest.so，它在位置/home/mylibfolder文件夹中，那么通过-L指定链接路径，通过 -l指定库名（不包含lib和.so)，就可以将test.cpp链接到libmytest.so库中

# 头文件搜索路径指定
g++ -I/myinclude test.cpp  # 指定头文件的搜索目录为myinclude

# 设置编译标准
g++ -std=c++11     # -std=c++11设置编译标准

# 设置宏
g++ -DDEBUG main.cpp    # 如果代码中有DEBUG宏，那么就可以通过-D指定宏来执行代码中DEBUG中的代码 

# 输出编译报警信息
g++ main.cpp -Wall     # 如果代码中有比如未使用的变量，那么就会报警 


# g++生成静态库
第一步：先生成.o的二进制文件
g++ swap.cpp -c     
第二步：生成.a的静态库
ar rs libswap.a swap.o

# g++生成动态库（假设swap.cpp在src目录下，当前代码执行目录也在src下，swap.h头文件在与src同级的include下）
g++ swap.cpp -I../include -fPIC -shared -o libswap.so  # -I为指定头文件的包含目录在include中，-o为指定输出名，其余为固定参数，此时生成的动态库在src目录下

# 将main文件链接到动态库（上一步g++生成的动态库，此时main.cpp与src和include在同一层目录下,且都在study_C++目录下）
g++ main.cpp -Iinclude -Lsrc -lswap -o main.out
# 执行main.out
export LD_LIBRARY_PATH=~/study_C++/src:$LD_LIBRARY_PATH   # 给搜索路径添加~/study_C++/src，不指定直接执行会报错
./main.out
```

## 二. 多版本g++切换

**步骤一: 安装编译器**

```bash
sudo apt update
sudo apt install build-essential
sudo apt -y install g++-8 g++-9 g++-10
```

**步骤二：使用update-alternatives工具创建多个g++编译器选项**

```c++
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 8
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
```

**步骤三：使用update-alternatives工具改变默认编译器**

```bash
sudo update-alternatives --config g++
There are 3 choices for the alternative g++ (providing /usr/bin/g++).

  Selection    Path            Priority   Status
------------------------------------------------------------
* 0            /usr/bin/g++-9   9         auto mode
  1            /usr/bin/g++-10  10         manual mode
  2            /usr/bin/g++-8   8         manual mode
  3            /usr/bin/g++-9   9         manual mode

Press  to keep the current choice[*], or type selection number:
```

**步骤四：查看版本**

```bash
gcc --version
```



