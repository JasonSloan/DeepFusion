#! /bin/bash
cd build
cmake .. 
make -j32 
cd ..
./workspace/mainproject