#include <iostream>
#include "opencv2/opencv.hpp"

#include "GeneralHoughTransform.hpp"

using namespace std;
using namespace cv;

int main() {
	int i = 1;
	Mat tpl = cv::imread("../images/tpl.jpg");
	Mat src = imread("../images/target.jpg");
	GeneralHoughTransform ght(tpl);								//构建模板；

 	Size s(src.size().width / 4, src.size().height / 4);		//把待检测图片压缩为原图的1/4
 	resize(src, src, s, 0, 0, cv::INTER_AREA);

 	// imshow("debug - image", src);

 	ght.accumulate(src);//执行检测
	// Mat res;
	// resize(tpl, tpl, Size(tpl.cols / 2, tpl.rows / 2));

	return 0;
}


