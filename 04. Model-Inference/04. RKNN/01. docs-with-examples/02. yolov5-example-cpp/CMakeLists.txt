cmake_minimum_required(VERSION 3.14)
project(yolov5)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_STANDARD 17)
# 为了检查内存泄漏
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=leak")
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)

file(GLOB SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/rknn-utils/*")
file(GLOB INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include/*" "${CMAKE_CURRENT_SOURCE_DIR}/src/rknn-utils" "${CMAKE_CURRENT_SOURCE_DIR}/main-utils")

include_directories(${INCLUDE} ${OpenCV_INCLUDE_DIRS})
link_directories(lib)
add_executable(${PROJECT_NAME} ${SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} rknnrt rga libturbojpeg.a Threads::Threads spdlog::spdlog)