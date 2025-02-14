cmake_minimum_required(VERSION 3.0)
project(mainproject)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -pthread")
set(CPP_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64 -fPIC -fopenmp -pthread -fpermissive)
set(NVCC_CXX_FLAGS -std=c++${CMAKE_CXX_STANDARD} -w -g -O0 -m64)

# 设置OpenCV_HOME以及要链接的OpenCV库名
set(OpenCV_HOME opencv-cuda-build)
set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_imgcodecs)

# 设置SPDLOG_HOME(纯头文件格式)
set(SPDLOG_HOME include/spdlog)

# 设置CUDA_HOME以及要链接的CUDA的库名
set(CUDA_HOME /usr/local/cuda-12)
set(CUDA_LIBS cudart cudnn nvinfer nvinfer_plugin nvonnxparser)

# 头文件寻找路径
include_directories(include ${OpenCV_HOME}/include/opencv4 ${CUDA_HOME}/include ${SPDLOG_HOME})

# 库文件寻找路径
link_directories(${OpenCV_HOME}/lib ${CUDA_HOME}/lib64)

# 添加源文件
file(GLOB_RECURSE SRC "src/*.cpp")

# 设置输出库文件
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${CUDA_LIBS})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/workspace)

