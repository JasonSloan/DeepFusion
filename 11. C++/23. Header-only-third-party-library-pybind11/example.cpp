#include <cstring>
#include "opencv2/opencv.hpp"
#include "pybind11.hpp"

using namespace std;
using namespace cv;
namespace py = pybind11;

int main(){
	// todo: 未运行成功, 
	// 参考本仓库中的"03. 推理与优化加速\01. TensorRT\01. trtpy docs\任务50_使用pybind11使python可以调用C++代码"
	// 也许pybind11无法在main函数中运行?
    Mat img = imread("dog.jpg");
    auto numel = img.rows * img.cols * img.channels();
    py::array imageArray;
    memcpy((uchar*)(imageArray.data(0)), img.data, numel);
};