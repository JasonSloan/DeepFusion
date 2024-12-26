```bash
# C++代码调试过程中出现断点无效, 断点错位问题

# 在CMakeLists.txt中设置为Debug模式
set(CMAKE_BUILD_TYPE "Debug")
# 注释掉代码优化项, 如果开启代码优化项, 那么代码结构机会被改变, 因此会出现错位
# set(CMAKE_CXX_FLAGS "-O3")
```

