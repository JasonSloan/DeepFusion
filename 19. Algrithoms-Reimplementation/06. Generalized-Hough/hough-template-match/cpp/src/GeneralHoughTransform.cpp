#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream> // For debugging
#include "opencv2/opencv.hpp"
#include "GeneralHoughTransform.hpp"
#include "util.hpp"

using namespace cv;
using namespace std;

GeneralHoughTransform::GeneralHoughTransform(const Mat& templateImage) {
	/* Parameters to set */

	/* Computed attributes */
	m_nRotations = (m_maxRotationAngle - m_minRotationAngle) / m_deltaRotationAngle + 1;		//待遍历的角度区间的个数
	// m_nSlices = 64;																				
	m_nScales = (m_maxScaleRatio - m_minScaleRatio) / m_deltaScaleRatio + 1;					//待遍历的缩放区间的个数

	setTemplate(templateImage);
}

void GeneralHoughTransform::setTemplate(const Mat& templateImage) {
	templateImage.copyTo(m_templateImage);
	int new_width = templateImage.cols / 4;
	int new_height = templateImage.rows / 4;
 	resize(m_templateImage, m_templateImage, Size(new_width, new_height));						// resize一下可以大大减少计算量
	findOrigin();
	cvtColor(m_templateImage, m_grayTemplateImage, COLOR_BGR2RGBA);
	m_grayTemplateImage.convertTo(m_grayTemplateImage, CV_8UC1);
	Mat m_blur = Mat(m_grayTemplateImage.size(), CV_8UC1);										// 这里直接转COLOR_BGR2GRAY不是更好吗
	blur(m_grayTemplateImage, m_blur, Size(3, 3));
	Canny(m_blur, m_template, m_cannyThreshold1, m_cannyThreshold2);
	createRTable();
}
//找参考点
void GeneralHoughTransform::findOrigin() {
	m_origin = Vec2f(m_templateImage.cols / 2, m_templateImage.rows / 2); 		// By default, the origin is at the center
	/*
	for(int j=0 ; j<m_templateImage.rows ; j++) {
		Vec3b* data= (Vec3b*)(m_templateImage.data + m_templateImage.step.p[0]*j);
		for(int i=0 ; i<m_templateImage.cols ; i++)
			if(data[i]==Vec3b(0,0,255)) { // If it's a red pixel...
				m_origin = Vec2f(i,j); // ...then it's the template's origin
			}
	}
	*/
}

void GeneralHoughTransform::createRTable() {
	int iSlice;
	double phi;

	Mat direction = gradientDirection(m_template);
	// imshow("debug - template", m_template);
	// imshow("debug - positive directions", direction);

	m_nSlices = (m_maxRotationAngle - m_minRotationAngle) / m_deltaRotationAngle;	//角度划分的bins
	m_RTable.clear();
	m_RTable.resize(m_nSlices);														//m_nSlices待遍历的角度
	for(int y = 0 ; y < m_template.rows ; y++) {
		uchar *templateRow = m_template.ptr<uchar>(y);
		double *directionRow = direction.ptr<double>(y);
		for(int x = 0 ; x < m_template.cols ; x++) {
			if(templateRow[x] == 255) {												//如果是边缘点的话
				phi = directionRow[x]; 												// gradient direction in radians in [-PI;PI]
				iSlice = rad2SliceIndex(phi, m_nSlices);							// 梯度夹角落在哪个bin,哪个bin就存储r向量
				m_RTable[iSlice].push_back(Vec2f(m_origin[0] - x, m_origin[1] - y));// x_center - x, y_center - y
			}
		}
	}
}

vector< vector<Vec2f> > GeneralHoughTransform::scaleRTable(const vector< vector<Vec2f> >& RTable, double ratio) {
	vector< vector<Vec2f> > RTableScaled(RTable.size());
	for(vector< vector<Vec2f> >::size_type iSlice=0 ; iSlice<RTable.size() ; iSlice++) {
		for(vector<Vec2f>::size_type ir=0 ; ir<RTable[iSlice].size() ; ir++) {
			RTableScaled[iSlice].push_back(Vec2f(ratio*RTable[iSlice][ir][0], ratio*RTable[iSlice][ir][1]));
		}
	}
	return RTableScaled;
}

vector< vector<Vec2f>> GeneralHoughTransform::rotateRTable(const vector<vector<Vec2f>>& RTable, double angle) {
	vector<vector<Vec2f>> RTableRotated(RTable.size());
	double c = cos(angle);
	double s = sin(angle);
	int iSliceRotated;
	for(vector<vector<Vec2f>>::size_type iSlice = 0 ; iSlice < RTable.size() ; iSlice++) {
		iSliceRotated = rad2SliceIndex(iSlice * m_deltaRotationAngle + angle, m_nSlices);
		for(vector<Vec2f>::size_type ir=0 ; ir<RTable[iSlice].size(); ir++) {
			RTableRotated[iSliceRotated].push_back(Vec2f(c*RTable[iSlice][ir][0] - s*RTable[iSlice][ir][1], s*RTable[iSlice][ir][0] + c*RTable[iSlice][ir][1]));
		}
	}
	return RTableRotated;
}

void GeneralHoughTransform::showRTable(vector< vector<Vec2f> > RTable) {
	int N(0);
	cout << "--------" << endl;
	for(vector< vector<Vec2f> >::size_type i=0 ; i<RTable.size() ; i++) {
		for(vector<Vec2f>::size_type j=0 ; j<RTable[i].size() ; j++) {
			cout << RTable[i][j];
			N++;
		}
		cout << endl;
	}
	cout << N << " elements" << endl;
}

void GeneralHoughTransform::accumulate(const Mat& image) {
	/* Image preprocessing */
	Mat grayImage(image.size(), CV_8UC1), gray(image.size(), CV_8UC1), edges(image.size(), CV_8UC1);
	cvtColor(image, gray, COLOR_BGR2GRAY);
	// gray.convertTo(gray, CV_8UC1);
	blur(gray, gray, Size(3, 3));
	Canny(gray, edges, m_cannyThreshold1, m_cannyThreshold2);		
	Mat direction = gradientDirection(edges);

	/* Debug */
	imwrite("src-edges.jpg", edges);
	// imwrite("src-edges-gradient-direction.jpg", direction);
	//waitKey(0);

	/* Accum size setting */
	int X = image.cols;
	int Y = image.rows;
	int S = ceil((m_maxScaleRatio - m_minScaleRatio) / m_deltaScaleRatio) + 1;			 // Scale Slices Number
	int R = ceil((m_maxRotationAngle - m_minRotationAngle) / m_deltaRotationAngle) + 1;  // Rotation Slices Number

	/* Usefull variablaration */
	vector<vector<Mat>> accum(R, vector<Mat>(S, Mat::zeros(Size(X, Y), CV_64F)));		// R*S*X*Y的四维数组
	Mat totalAccum = Mat::zeros(Size(X, Y), CV_32S);
	int iSlice(0), iScaleSlice(0), iRotationSlice(0), ix(0), iy(0);
	double max(0.0), phi(0.0);
	vector<vector<Vec2f>> RTableRotated(m_RTable.size()), RTableScaled(m_RTable.size());// 二维数组, 每个元素记录的是一个Vec2f数据类型的点对
	Mat showAccum(Size(X, Y), CV_8UC1);
	vector<GHTPoint> points;
	GHTPoint point;
	max = 0;
	/* Main loop */
	for(double angle = m_minRotationAngle ; angle <= m_maxRotationAngle + 0.0001 ; angle += m_deltaRotationAngle) { 	// For each rotation (0.0001 double comparison)
		iRotationSlice = round((angle - m_minRotationAngle) / m_deltaRotationAngle);
		cout << "Rotation Angle\t: " << angle / PI * 180 << "°" << endl;
		RTableRotated = rotateRTable(m_RTable, angle);									// 根据旋转角度，旋转RTable
		for(double ratio=m_minScaleRatio ; ratio<=m_maxScaleRatio+0.0001 ; ratio+=m_deltaScaleRatio) { // For each scaling (0.0001 double comparison)
 			iScaleSlice = round((ratio - m_minScaleRatio) / m_deltaScaleRatio);
			cout << "|- Scale Ratio\t: " << ratio*100 << "%" << endl;
			RTableScaled = scaleRTable(RTableRotated,ratio);							// 根据缩放比例，缩放RTable
			accum[iRotationSlice][iScaleSlice] = Mat::zeros(Size(X,Y), CV_64F);
			
			for(int y=0 ; y<image.rows ; y++) {
				for(int x=0 ; x<image.cols ; x++) {
					phi = direction.at<double>(y, x);
					if(phi != 0.0) {
						iSlice = rad2SliceIndex(phi, m_nSlices);
						for(vector<Vec2f>::size_type ir=0 ; ir<RTableScaled[iSlice].size() ; ir++) { // For each r related to this angle-slice
							ix = x + round(RTableScaled[iSlice][ir][0]);	// We compute x+r, the supposed template origin position
							iy = y + round(RTableScaled[iSlice][ir][1]);
							if(ix>=0 && ix<image.cols && iy>=0 && iy<image.rows) { // If it's between the image boundaries
								totalAccum.at<int>(iy,ix)++;
								if(++accum[iRotationSlice][iScaleSlice].at<double>(iy,ix) > max) { // Icrement the accum
									max = accum[iRotationSlice][iScaleSlice].at<double>(iy,ix);
									point.phi = angle;
									point.s = ratio;
									point.y.y = iy;
									point.y.x = ix;
									point.hits = accum[iRotationSlice][iScaleSlice].at<double>(iy,ix);
								}
								/* To see the step-by-step accumulation uncomment these lines */
								// normalize(accum[iRotationSlice][iScaleSlice], showAccum, 0, 255, NORM_MINMAX, CV_8UC1);
								// imshow("debug - subaccum", showAccum);	waitKey(1);
							}
						}
					}
				}
			}
			/* Pushing back the best point for each transformation (uncomment line 159 : "max = 0") */
			points.push_back(point);
			/* Transformation accumulation visualisation */
			normalize(accum[iRotationSlice][iScaleSlice], showAccum, 0, 255, NORM_MINMAX, CV_8UC1); // To see each transformation accumulation (uncomment line 159 : "max = 0")
			// normalize(totalAccum, showAccum, 0, 255, NORM_MINMAX, CV_8UC1); // To see the cumulated accumulation (comment line 159 : "max = 0")
			imwrite("accum.jpg", showAccum);	//waitKey(1);
			// blur(accum[iRotationSlice][iScaleSlice], accum[iRotationSlice][iScaleSlice], Size(3,3)); // To harmonize the local maxima
		}
	}
	/* Pushing back the best point for cumulated transformations (comment line 159 : "max = 0") */
	// points.push_back(point);

	/* Drawing templates on best points */
	Mat out(image.size(), image.type());
	image.copyTo(out);
	//for(vector<GHTPoint>::size_type i=0 ; i<points.size() ; i++) {
	cv::circle(out, point.y, 3, cv::Scalar(0, 0, 255));
	drawTemplate(out, point);

		//int msk = 1;
	//}
	imwrite("output.jpg", out);
}

vector<GHTPoint> GeneralHoughTransform::findTemplates(vector< vector< Mat > >& accum, int threshold) {
	vector<GHTPoint> newPoints;

	//TODO

	return newPoints;
}

void GeneralHoughTransform::drawTemplate(Mat& image, GHTPoint params) {
	cout << params.y << " avec un rapport de grandeur de " << params.s << " et une rotation de " << params.phi/PI*180 << "° et avec " << params.hits << " !" << endl;
	double c = cos(params.phi);
	double s = sin(params.phi);
	int x(0), y(0), relx(0), rely(0);
	for(vector<vector<Vec2f>>::size_type iSlice = 0 ; iSlice<m_RTable.size() ; iSlice++)
		for(vector<Vec2f>::size_type ir=0 ; ir<m_RTable[iSlice].size() ; ir++) {
			relx = params.s * (c*m_RTable[iSlice][ir][0] - s*m_RTable[iSlice][ir][1]); // X-Coordinate of the template's point after transformation (relative to the origin)
			rely = params.s * (s*m_RTable[iSlice][ir][0] + c*m_RTable[iSlice][ir][1]); // Y-Coordinate of the template's point after transformation (relative to the origin)
			x = params.y.x + relx; // X-Coordinate of the template's point in the image
			y = params.y.y + rely; // Y-Coordinate of the template's point in the image
			if(x>=0 && x<image.cols && y>=0 && y<image.rows)
				image.at<Vec3b>(y,x) = Vec3b(0,255,0); // Put the pixel in green
		}
}
