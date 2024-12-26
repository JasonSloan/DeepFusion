#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <math.h>
#include <iostream>
#include <iostream>

#include "util.hpp"

using namespace cv;
using namespace std;

int rad2SliceIndex(double angle, int nSlices) {
	double a = (angle > 0) ? (fmodf(angle, 2 * PI)) : (fmodf(angle + 2 * PI, 2 * PI));
	return floor(a / (2 * PI / (nSlices + 0.00000001)));
}

Mat gradientY(const Mat &src) {
	return gradientX(src.t()).t();
}

Mat gradientX(const Mat &src) {
	Mat dst(src.rows,src.cols,CV_32F);
	for (int y=0 ; y<src.rows ; y++) {
		const uchar *srcRow = src.ptr<uchar>(y);
		float *dstRow = dst.ptr<float>(y);
		dstRow[0] = srcRow[1] - srcRow[0];
		for(int x=1 ; x<src.cols-1 ; x++)
			dstRow[x] = srcRow[x+1] - srcRow[x-1];
		dstRow[src.cols-1] = srcRow[src.cols-1] - srcRow[src.cols-2];
	}
	return dst;
}

float gradientDirection(const Mat& src, int x, int y) {
	int gx,gy;
	if(x==0)				gx = src.at<uchar>(y,x+1) - src.at<uchar>(y,x);
	else if(x==src.cols-1)	gx = src.at<uchar>(y,x) - src.at<uchar>(y,x-1);
	else					gx = src.at<uchar>(y,x+1) - src.at<uchar>(y,x-1);
	if(y==0)				gy = src.at<uchar>(y+1,x) - src.at<uchar>(y,x);
	else if(y==src.rows-1)	gy = src.at<uchar>(y,x) - src.at<uchar>(y-1,x);
	if(y==0)				gy = src.at<uchar>(y+1,x) - src.at<uchar>(y-1,x);
	return atan2(gx,gy);
}

//对于图片计算梯度方向
Mat gradientDirection(const Mat& src) {
	Mat dst(src.size(), CV_64F);
//	Mat gradX = gradientX(src);
//	Mat gradY = gradientY(src);
	Mat gradX(src.size(), CV_64F);
	Sobel(src, gradX, CV_64F, 1, 0, 5);		//1 for x dim, kernel_size=5
	Mat gradY(src.size(), CV_64F);
	Sobel(src, gradY, CV_64F, 0, 1, 5);		//1 for y dim, kernel_size=5
	double t;
	for(int y = 0 ; y < gradX.rows ; y++)
		for(int x = 0 ; x < gradX.cols ; x++) {
			t = atan2(gradY.at<double>(y, x), gradX.at<double>(y, x));
			//dst.at<double>(y,x) = (t == 180) ? 0 : t;  
			dst.at<double>(y, x) =  t;
		}
	return dst;
}

void invertIntensities(const Mat& src, Mat& dst) {
	for(int i=0 ; i<src.rows ; i++)
		for(int j=0 ; j<src.cols ; j++)
			dst.at<uchar>(i,j) = 255 - src.at<uchar>(i,j);
}

float fastsqrt(float val) {
	int tmp = *(int *)&val;
	tmp -= 1<<23;
	tmp = tmp >> 1;
	tmp += 1<<29;
	return *(float *)&tmp;
}


