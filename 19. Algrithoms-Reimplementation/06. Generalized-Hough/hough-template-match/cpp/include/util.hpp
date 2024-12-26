/*
 * util.hpp
 *
 *  Created on: 15 f�vr. 2014
 *      Author: J�r�my
 */

#ifndef UTIL_HPP_
#define UTIL_HPP_

const double PI = 4.0*atan(1.0);

cv::Mat gradientY(const cv::Mat &src);
cv::Mat gradientX(const cv::Mat &src);
cv::Mat gradientDirection(const cv::Mat& src);
void invertIntensities(const cv::Mat& src, cv::Mat& dst);
float gradientDirection(const cv::Mat& src, int x, int y);
float fastsqrt(float val);
int rad2SliceIndex(double angle, int nSlices);

#endif /* UTIL_HPP_ */
