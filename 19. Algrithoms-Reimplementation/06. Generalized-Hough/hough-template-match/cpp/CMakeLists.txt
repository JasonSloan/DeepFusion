# cmake最低版本要求
cmake_minimum_required(VERSION 2.8)
# 如果是想debug代码的话，必须加上这一行
set(CMAKE_BUILD_TYPE Debug)
# 工程名称
project(mainproject)
# 寻找OpenCV库
find_package(OpenCV REQUIRED)
# 包含头文件
include_directories(${OpenCV_INCLUDE_DIRS} include/)
# 添加可执行文件
aux_source_directory(src SRC)
add_executable(${PROJECT_NAME} ${SRC})
# 设置编译选项
set(CMAKE_CXX_STANDARD 11)
# 链接 OpenCV 库
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
# 设置可执行文件的输出路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
