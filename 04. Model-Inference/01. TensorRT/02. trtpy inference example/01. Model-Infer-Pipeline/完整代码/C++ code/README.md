**crop_image：**

crop_image.py：将workspace下的文件夹的图片resize到指定尺寸并保存

**gen_and_modify_onnx：**

gen_onnx.py：将pytorch模型转成onnx模型

inference.py：pytorch模型的standalone的推理代码

model.py：pytorch模型的standalone的模型构建代码

modify_onnx.py：将前处理（后处理）部分添加到整体onnx网络结构中的代码

**include：**

inference.hpp：C++推理代码中的头文件（各个章节的头文件和源文件只要替换掉对应的就能直接运行）

**lib：**

libinference.so：将inference.cpp和其他依赖文件一起变异成的动态库

**onnx：**

**：onnx有关的查询的文件，放置在那里，没什么用

**src：**

onnx/**：onnx源文件，一般无需修改变动

onnx/-tensorrt/**：onnx-tensorrt源文件，一般无需修改变动

NanoLog/**：NanoLog源码，一般无需改动

inference.cpp：C++推理代码中的源文件（各个章节的头文件和源文件只要替换掉对应的就能直接运行）

main.cpp：测试inference.cpp的入口文件

**src2：**

onnx/**：onnx源文件，一般无需修改变动

onnx/-tensorrt/**：onnx-tensorrt源文件，一般无需修改变动

main.cpp：将onnx模型编译成tensorrt模型的文件（因为在编写过程中需要经常重新编译onnx模型，所以只需要将src文件夹改名为src1，src2改名为src，然后make run -j6就能运行编译）

**tmp:**

main.cpp：如果想编译成可执行文件，需要把main.cpp放入src文件夹

Makefile(executable)：如果想生成可执行文件并执行，需要把该文件替换掉外面的Makefile

**workspace：**

make run -j6的工作目录，生成的tensorrt模型以及，使用模型推理的结果产生的地方

**CMakeLists.txt：**

将inference.cpp编译成动态库的文件

**easy_cmake.sh**

使cmake编译更简单的脚本

**Makefile：**

make（make run -j6 / make all /  make clean / make debug）编译的文件









