#! /bin/bash
cd build
cmake .. 
make
cd ..
cp build/yolov5 workspace/
./workspace/yolov5