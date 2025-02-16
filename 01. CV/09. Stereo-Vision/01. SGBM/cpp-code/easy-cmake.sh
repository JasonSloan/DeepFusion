#! /bin/bash
# 是否启用代码中的计时
WITH_CLOCKING=$1   
mkdir -p cpp-SGBM/build 
mkdir -p cpp-SGBM/workspace
cd cpp-SGBM/build
cmake .. -D WITH_CLOCKING=${WITH_CLOCKING}
make -j6
cd ../../ && ./cpp-SGBM/workspace/pro