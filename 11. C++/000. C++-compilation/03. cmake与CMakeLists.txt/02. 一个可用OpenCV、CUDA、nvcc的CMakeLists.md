

```python
# 与普通的编译相比区别在于:
#   1. 遍历使用nvcc将.cu文件逐个编译成.cu.o文件
#   2. 将所有的.cu.o文件编译成静态库libnvtmp.a
#   3. 将静态库链接到普通编译出的可执行文件
#   4. 删除静态库
# ------------------------------------------------
# 工程名称
project(pro)
# cmake最低版本要求
cmake_minimum_required(VERSION 3.5)
# 如果是想debug代码的话，必须加上这一行
set(CMAKE_BUILD_TYPE Debug)
# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
# 设置编译选项
set(CPP_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64 -fPIC -fopenmp -pthread -fpermissive)
# 设置cpp文件和cu文件的编译选项, g++版本要求11以下
set(NVCC_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64)

# 设置OpenCV_HOME以及要链接的OpenCV库名
set(OpenCV_HOME /root/software/miniconda3/lib/python3.9/site-packages/trtpy/cpp-packages/opencv4.2)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_video opencv_videoio)

# 设置CUDA_HOME以及要链接的CUDA的库名
set(CUDA_HOME /root/software/miniconda3/lib/python3.9/site-packages/trtpy/trt8cuda115cudnn8)
set(CUDA_LIBS cudart cudnn nvinfer nvinfer_plugin)

# 设置nvcc路径
set(NVCC_PATH ${CUDA_HOME}/bin)
# 设置cu文件编译出的静态库的名字
set(nvcc_target_lib nvtmp)       # nvtmp也就是静态库libnvtmp.a中间的名字
set(nvcc_target_lib_full_name libnvtmp.a)

# 头文件寻找路径
include_directories(${OpenCV_HOME}/include ${CUDA_HOME}/include/cuda ${CUDA_HOME}/include/tensorRT ${CUDA_HOME}/include/protobuf)
# 给cmake指定include_directories并不会让nvcc也去这里找, 所以还要再指定一遍
set(NVCC_INCLUDE_DIRS -I${OpenCV_HOME}/include -I${CUDA_HOME}/include/cuda -I${CUDA_HOME}/include/tensorRT -I${CUDA_HOME}/include/protobuf)
set(NVCC_CXX_FLAGS ${NVCC_CXX_FLAGS} ${NVCC_INCLUDE_DIRS})

# 库文件寻找路径
# 指定库文件寻找路径包括${CMAKE_BINARY_DIR}是因为nvcc编译生成的临时静态库libnvtmp.a会产生在这里
link_directories(${OpenCV_HOME}/lib ${CUDA_HOME}/lib64 ${CMAKE_BINARY_DIR}) # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake ..命令的路径(也就是build文件夹)

# 添加源文件
file(GLOB_RECURSE SRC "src/*.cpp")
file(GLOB_RECURSE cu_files "src/*.cu")

# 设置可执行文件
add_executable(${PROJECT_NAME} ${SRC})
# 设置编译选项
target_compile_options(${PROJECT_NAME} PRIVATE ${CPP_CXX_FLAGS})
# 链接一些库
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBS} ${nvcc_target_lib})
# 设置可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

# 添加pre-build和post-build，使用dlcc并指定相应的编译选项编译出obj文件并在之后删除这些obj文件
foreach (cu_file ${cu_files})
    set(tmp ${cu_file}.o)
    # CMAKE_SOURCE_DIR: cmake中的预置变量, 存储最顶层的CMakeLists.txt文件的目录的路径
    # CMAKE_BINARY_DIR: cmake中的预置变量, 存储执行cmake命令的路径
    # 该行命令实现的是将tmp变量对应的string中的CMAKE_SOURCE_DIR的路径替换为CMAKE_BINARY_DIR
    string(REPLACE ${CMAKE_SOURCE_DIR} ${CMAKE_BINARY_DIR} cu_obj "${tmp}")
    string(REGEX MATCH "/.*/" cu_dir ${cu_obj}) 
    # message("tmp: " ${tmp})             # /path/to/your/kernelfunction/dir/kernelfunction.cu.o
    # message("cu_obj: " ${cu_obj})       # /path/to/your/build/dir/kernelfunction.cu.o
    # message("cu_dir: " ${cu_dir})       # /path/to/your/build/dir
    set(cu_objs ${cu_objs} ${cu_obj})
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND mkdir -p ${cu_dir})
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ${NVCC_PATH}/nvcc  ${NVCC_CXX_FLAGS} -o ${cu_obj} -c ${cu_file})
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND rm ${cu_obj})
endforeach()

# 将nvcc编译出来的obj文件打包成临时静态库，编译出可执行程序后删除
# ar是编译静态库的命令
add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
                   COMMAND ar cqs ${nvcc_target_lib_full_name} ${cu_objs})
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
                   COMMAND rm ${nvcc_target_lib_full_name})

# 打印调试信息
# message("NVCC_CXX_FLAGS: ${NVCC_CXX_FLAGS}")
```



