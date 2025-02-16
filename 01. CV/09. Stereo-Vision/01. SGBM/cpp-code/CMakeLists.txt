cmake_minimum_required(VERSION 3.10)
project(pro)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

# 是否启用代码中的计时功能
if(WITH_CLOCKING STREQUAL "ON")
    target_compile_definitions(${PROJECT_NAME} PRIVATE WITH_CLOCKING)
endif()

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread -fpermissive -Wall -Wextra")

# 设置OpenCV_HOME以及要链接的OpenCV库名
find_package(OpenCV REQUIRED)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs opencv_calib3d)

# 设置SPDLOG_HOME(纯头文件格式)
set(SPDLOG_HOME include/spdlog-build)

# 头文件寻找路径
include_directories(include ${OpenCV_HOME}/include/opencv4 ${SPDLOG_HOME})

# 库文件寻找路径
link_directories(${OpenCV_HOME}/lib)

# 添加源文件
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRC)

# 设置输出库文件
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/workspace)