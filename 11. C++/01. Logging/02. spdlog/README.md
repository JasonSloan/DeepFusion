[spdlog官网](https://github.com/gabime/spdlog/tree/v2.x)

注意spdlog在与多线程使用的时候, 需要创建一个全局变量, 以控制每个线程的日志名字都不能一样, 否则会出现错误

安装(建议使用源码安装)

```bash
$ git clone https://github.com/gabime/spdlog.git
$ cd spdlog && mkdir build && cd build
$ cmake .. && make -j4 && make install
```

```bash
Debian: sudo apt install libspdlog-dev
Homebrew: brew install spdlog
MacPorts: sudo port install spdlog
FreeBSD: pkg install spdlog
Fedora: dnf install spdlog
Gentoo: emerge dev-libs/spdlog
Arch Linux: pacman -S spdlog
openSUSE: sudo zypper in spdlog-devel
vcpkg: vcpkg install spdlog
conan: spdlog/[>=1.4.1]
conda: conda install -c conda-forge spdlog
build2: depends: spdlog ^1.8.2
```



一些基本使用:

```c++
#include "spdlog/spdlog.h"
#include "spdlog/sinks/basic_file_sink.h"

int main() {   
    // 只打印日志内容, 不写入文件
    spdlog::info("Welcome to spdlog!");
    spdlog::error("Some error message with arg: {}", 1);
    
    spdlog::warn("Easy padding in numbers like {:08d}", 12);
    spdlog::critical("Support for int: {0:d};  hex: {0:x};  oct: {0:o}; bin: {0:b}", 42);
    spdlog::info("Support for floats {:03.2f}", 1.23456);
    spdlog::info("Positional args are {1} {0}..", "too", "supported");
    spdlog::info("{:<30}", "left aligned");
    
    spdlog::set_level(spdlog::level::debug); // Set global log level to debug
    spdlog::debug("This message should be displayed..");    

    // 把日志内容写入文件(不会在控制台打印)
    auto logger = spdlog::basic_logger_mt("basic_logger", "logs/basic-log.txt");
    logger->info("Welcome to spdlog!");
    logger->error("Some error message with arg: {}", 1);
    // 更多用法参考官网: https://github.com/gabime/spdlog/tree/v2.x

}
```

CMakeLists.txt

```bash
cmake_minimum_required(VERSION 3.14)
project(spdlog_example CXX)

if(NOT TARGET spdlog)
    # Stand-alone build
    find_package(spdlog REQUIRED)
endif()

add_executable(example src/main.cpp)
target_link_libraries(example PRIVATE spdlog::spdlog $<$<BOOL:${MINGW}>:ws2_32>)
```

