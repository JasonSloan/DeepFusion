```python
一、使用CMake构建C++工程：
手动编写CMakeLists.txt
执行命令cmake PATH 生成Makefile(PATH是顶层CMakeLists.txt所在目录)
执行make进行编译
上面的内容也就是：
mkdir build
cd build
cmake ..    # 编译上级目录的CmakeLists.txt，生成Makefile和其他文件
make

一、CMake常用语法
语法：指令(参数1， 参数2...)
# 对CMake版本最低要求
cmake_minimum_required(VERSION 2.8.3)  # 对CMake版本最低要求2.8.3
# 定义工程名称
project(HELLOWORLD)   # 定义工程名称为HELLOWORLD
# 显示定义变量
set(SRC sayhello.cpp hello.cpp)   # 定义src变量，她得值为sayhello.cpp hello.cpp
# 向工程添加多个特定的头文件搜索路径
include_directories(/usr/include/myincludefolder ./include)   # 向工程添加两个头文件搜索路径
# 向工程添加多个特定的库文件搜索路径
link_directories(/usr/lib/mylibfolder ./lib)  # 向工程添加两个库搜索路径
# 生成库文件
add_library(hello SHARED ${SRC})  # 对上面SRC变量代表的2个文件生成动态库（SHARED）库文件，库文件名为hello
# 添加编译参数
add_compile_options(-Wall -std=c++11 -O2)   # 添加编译参数：-Wall输出警告信息，-std=c++11标准是c++11，-O2对代码做优化
# 生成可执行文件
add_executable(main main.cpp)    # 将main.cpp生成可执行文件main
# 为target添加需要链接的共享库
target_link_libraries(main hello)   # 将上面生成的hello动态库链接到可执行文件main中
# 向当前工程中添加存放源文件的子目录
add_subdirectory(src)    # 添加src子目录，src中需要有一个CMakeLists.txt
# 发现一个目录下所有的源代码文件并将列表存储在一个变量中，这个指令临时被用来自动构建源文件列表
aux_source_directory(. SRC)  # 定义SRC变量，其值为当前目录下所有的源代码文件
add_executable(main ${SRC})  # 编译SRC变量所代表的源代码文件，生成main可执行文件

二、CMake常用变量（想改变常用变量的值都使用set来改变）
CMAKE_C_FLAGS    # gcc编译选项
CMAKE_CXX_FLAGS  # g++编译选项
set(CMAKE_CXX_FLAGS “${CMAKE_CXX_FLAGS} -std=c++11") # 在CMAKE_CXX_FLAGS编译选项后追加-std=c++11
CMAKE_BUILD_TYPE   # 编译类型（Debug Release)
set(CMAKE_BUILD_TYPE Debug)     # 设定编译类型为debug
set(CMAKE_BUILD_TYPE Release)   # 设定编译类型为release
CMAKE_C_COMPILER     # 指定C编译器
CMAKE_CXX_COMPILER   # 指定C++编译器
EXECUTABLE_OUTPUT_PATH     # 可执行文件输出的存放路径
LIBRARY_OUTPUT_PATH        # 库文件输出的存放路径
```

