```python
# 如果想快速使用Makefile，就用trtpy get-series tensorrt-integrate拉取代码到同级目录下，然后使用其中的Makefile


# 参考视频：https://www.bilibili.com/video/BV1EM41177s1/?p=9&spm_id_from=pageDriver&vd_source=2fa3840975cc19817a9a15ddf8a1a81b
位于哔哩哔哩：收藏：g++、Cmake、Makefile
参考文档：在E-->AI-->deepblue-->code-->Makefile

0、使用debug
libs := $(shell find /usr/lib -name lib*)    
debug : 
    @echo $(libs)
.PHONY : debug

1，基本格式
targets : prerequisties
[Tab键]command
    
2，常用符号
%    :代表任意值，例如%.cpp代表所有以cpp结尾的文件
@    :隐藏执行的命令的输出
=    :赋值运算符（引用传递，在使用变量的时候展开）
:=   :赋值运算符（值传递，赋值时立即展开）
?=   :默认赋值运算符。如果该变量已经定义，则不进行任何操作;如果该变量尚未定义，则求值并分配
+=   :相当于python中的字符串相加
\    :续行符

3，伪目标
如果不加.PHONY，那么make会将其视为某个要生成的文件，而不是make命令的一个参数
clean:
    @rm -rf objs
.PHONY : clean      # 如果不加.PHONY，那么make clean会让make以为要生成clean这个文件而不是执行下面的rm -rf命令

4，变量的定义
cpp := src/main.cpp 
obj := objs/main.o

5，变量的引用
可以用 () 或 {}

6，预定义变量
$@ : 目标(target)的完整名称
$< : 第一个依赖文件（prerequisties）的名称
$^ : 所有的依赖文件（prerequisties），以空格分开，不包含重复的依赖文件
cpp := src/main.cpp 
obj := objs/main.o

$(obj) : ${cpp}
    @g++ -c $< -o $@
    @echo $^

compile : $(obj)
.PHONY : compile

7，常用函数
函数调用，很像变量的使用，也是以 “$” 来标识的
$(fn, arguments) or ${fn, arguments}
（1）shell
    $(shell <command> <arguments>)
    # shell 指令，src 文件夹下找到 .cpp 文件
    cpp_srcs := $(shell find src -name "*.cpp") 
    # shell 指令, 获取计算机架构
    HOST_ARCH := $(shell uname -m)
（2）subst
    $(subst <from>,<to>,<text>)
    字符串替换函数,把字串 <text> 中的 <from> 字符串替换成 <to>
    cpp_srcs := $(shell find src -name "*.cpp")
    cpp_objs := $(subst src/,objs/,$(cpp_objs))   # 将路径中的src/替换成objs/
（3）patsubst
    $(patsubst <pattern>,<replacement>,<text>)
    模式字符串替换函数，通配符 %，表示任意长度的字串，从 text 中取出 patttern， 替换成 replacement
    cpp_srcs := $(shell find src -name "*.cpp")     #shell指令，src文件夹下找到.cpp文件
    cpp_objs := $(patsubst %.cpp,%.o,$(cpp_srcs))     #cpp_srcs变量下cpp文件替换成 .o文件
（4）foreach
    $(foreach <var>,<list>,<text>)
    循环函数，把字串<list>中的元素逐一取出来，执行<text>包含的表达式，返回<text>所返回的每个字符串所组成的整个字符串（以空格分隔）
    library_paths := /datav/shared/100_du/03.08/lean/protobuf-3.11.4/lib \
                 /usr/local/cuda-10.1/lib64

    library_paths := $(foreach item,$(library_paths),-L$(item))  # 这样省的g++的时候，逐个为每个库路径添加-L了
    I_flag := $(include_paths:%=-I%)
（5）dir
    $(dir <names...>)
    取目录函数,从文件名序列<names>中取出目录部分。目录部分是指最后一个反斜杠（“/”）之前的部分。如果没有反斜杠，那么返回“./”
    $(dir src/foo.c hacks)    # 返回值是“src/ ./”
（6）notdir
    $(notdir <names...>)
    去掉变量中的文件夹路径，只保留文件名和后缀
    libs   := $(notdir $(shell find /usr/lib -name lib*))     # 这里找/usr/lib下所有以lib开头的文件和文件夹，再使用notdir去掉路径，只保留文件的名字
（7）filter
    $(filter <names...>)
    以一定的规则过滤
    libs    := $(notdir $(shell find /usr/lib -name lib*))
    a_libs  := $(filter %.a,$(libs))        # 只返回以.a结尾的文件    
    so_libs := $(filter %.so,$(libs))       # 值返回以.so结尾的文件
（8）basename
    $(basename <names...>)
    去掉整个路径中的文件名后缀
    libs    := $(notdir $(shell find /usr/lib -name lib*))
    a_libs  := $(subst lib,,$(basename $(filter %.a,$(libs))))    # 这样就可以获得所有的以.a结尾的静态库的库名
    so_libs := $(subst lib,,$(basename $(filter %.so,$(libs))))    # 这样就可以获得所有的以.so结尾的动态库的库名
    
8，编译选项和链接选项
编译选项：
-m64: 指定编译为 64 位应用程序
-std=: 指定编译标准，例如：-std=c++11、-std=c++14
-g: 包含调试信息
-w: 不显示警告
-O: 优化等级，通常使用：-O3
-I: 加在头文件路径前
fPIC: (Position-Independent Code), 产生的没有绝对地址，全部使用相对地址，代码可以被加载到内存的任意位置，且可以正确的执行。这正是共享库所要求的，共享库被加载时，在内存的位置不是固定的
链接选项：
-l: 加在库名前面
-L: 加在库路径前面
-Wl,<选项>: 将逗号分隔的 <选项> 传递给链接器
-rpath=: "运行" 的时候，去找的目录。运行的时候，要找 .so 文件，会从这个选项里指定的地方去找

9，编译带头文件的程序
cpp_srcs := $(shell find src -name *.cpp)
cpp_objs := $(patsubst src/%.cpp,objs/%.o,$(cpp_srcs))

# 你的头文件所在文件夹路径（建议绝对路径）
include_paths := 这里填写头文件所在的路径
I_flag        := $(include_paths:%=-I%)    # 这里相当于把"-I头文件路径"赋值给I_flag了


objs/%.o : src/%.cpp
    @mkdir -p $(dir $@)
    @g++ -c $^ -o $@ $(I_flag)        # 把头文件所在路径包含进去了

workspace/exec : $(cpp_objs)
    @mkdir -p $(dir $@)
    @g++ $^ -o $@                     # 链接成可执行文件

run : workspace/exec            
    @./$<                             # ./执行可执行文件

debug :
    @echo $(I_flag)

clean :
    @rm -rf objs

.PHONY : debug run

9，编译静态库
编译静态库的过程：
    源文件[.c/cpp] -> Object文件[.o]
    g++ -c [.c/cpp][.c/cpp]... -o [.o][.o]... -I[.h/hpp] -g
    Object文件[.o] -> 静态库文件[lib库名.a]
    ar -r [lib库名.a] [.o][.o]...
    main 文件[.c/cpp] -> Object 文件[.o]
    g++ -c [main.c/cpp] -o [.o] -I[.h/hpp] 
    链接 main 的 Object 文件与静态库文件 [lib库名.a]
    g++ [main.o] -o [可执行文件] -l[库名] -L[库路径]
    
10，编译动态库
编译动态库的过程：
    编译 .c 文件 源文件[.c/cpp] -> Object文件[.o]
    g++ -c [.c/cpp][.c/cpp]... -o [.o][.o]... -I[.h/hpp] -g -fpic
    Object文件[.o] -> 动态库文件[lib库名.so]
    g++ -shared [.o][.o]... -o [lib库名.so] 
    main文件[.c/cpp] -> Object文件[.o]
    g++ -c [main.c/cpp] -o [.o] -I[.h/hpp] -g
    链接 main 的 Object 文件与动态库文件[lib库名.so]
    g++ [main.o] -o [可执行文件] -l[库名] -L[库路径] -Wl,-rpath=[库路径]

11，生成动态库的例子
目录结构：
    project-----------Makefile
                 |
                 |
                  -------src------add.cpp
                 |             |
                 |             |
                 |              ---main.cpp
                 |
                 |
                  -------include------add.hpp

Makefile文件内容：
# 查找src目录下的所有cpp文件，得到所有cpp文件的路径
cpp_srcs := $(shell find src/ -name '*.cpp')
# 将所有的cpp文件路径替换成.o文件路径（第一步编译过程用）
cpp_objs := $(patsubst src/%.cpp, objs/%.o, $(cpp_srcs))
# 去掉objs/main.o的文件，得到链接成动态库的所需文件
so_objs := $(filter-out objs/main.o, $(cpp_objs))

# 头文件搜索路径
include_dirs := include
# 库文件搜索路径
library_dirs := lib
# 搜索的库的名字
linking_libs := func
# 将头文件搜索路径前加-I
I_options := $(include_dirs:%=-I%)
# 将库文件搜索路径前加-l
l_options := $(linking_libs:%=-l%)
# 将库名前加-L
L_options := $(library_dirs:%=-L%)
# 设置运行时库的搜索路径
r_options := $(library_dirs:%=-Wl,-rpath=%)             # -Wl,-rpath=lib

# 编译选项
compile_options := -g -O3 -w -fPIC $(I_options)
# 链接选项
linking_options := $(l_options) $(L_options) $(r_options)

# 设置.o文件各自依赖于.cpp文件
objs/%.o : src/%.cpp
    @mkdir -p $(dir $@)                                                             
    @g++ -c $^ -o $@ $(compile_options)

# 编译 make compile
compile : $(cpp_objs)

# 动态库依赖于除main.o以外的所有.o文件
lib/libfunc.so: $(so_objs)
    @mkdir -p $(dir $@)
    @g++ -shared $^ -o $@

# 链接 make dynamic
dynamic : lib/libfunc.so


# 可执行文件pro依赖的文件
workspace/pro : objs/main.o compile dynamic
    @mkdir -p $(dir $@)
    @g++ $< -o $@  $(linking_options)

# make run 生成可执行文件
run: workspace/pro
    @./$<

# make clean 删除三个文件夹
clean : 
    @rm -rf objs lib workspace

# make debug打印这几个变量的值
debug:
    @echo $(cpp_srcs)
    @echo $(cpp_objs)
    @echo $(so_objs)
    @echo $(r_options)

# 指定这几个为伪目标，并真正的生成这些文件
.PHONY: compile dynamic debug run clean
```

