使用cuda批量做yolo的前处理, 包括: resize+copyMakeBoder+BGR2RGB+/255.0, 全部使用CUDA做

代码见本文件夹下

其中.cu文件需要使用nvcc编译, .cpp文件需要使用g++编译, 如何编写CMakeLists.txt见"08. C++\000. C++编译相关\03. cmake与CMakeLists.txt\02. 一个可用OpenCV、CUDA、nvcc的CMakeLists.md"