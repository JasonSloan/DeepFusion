#ifndef GENERALHOUGHTRANSFORM_HPP_
#define GENERALHOUGHTRANSFORM_HPP_

struct GHTPoint {
	double phi;
	double s;
	cv::Point y;
	double hits;
};

class GeneralHoughTransform {

public:
	std::vector<std::vector<cv::Vec2f> > m_RTable;
	cv::Vec2f m_origin;
	cv::Mat m_templateImage;
	cv::Mat m_grayTemplateImage;
	cv::Mat m_template;

	int m_cannyThreshold1=150;			// canny边缘检测的弱阈值
	int m_cannyThreshold2=200;			// canny边缘检测的强阈值
	double m_minPositivesDistance;
	double m_deltaScaleRatio = 0.1;		// 缩放比例的搜索间隔
	double m_minScaleRatio = 0.8;		// 最小缩放到模版图片的多少倍
	double m_maxScaleRatio = 1.2;		// 最大缩放到模版图片的多少倍
	double m_deltaRotationAngle = 3.1415 / 32;			// 旋转角度的搜索间隔
	double m_minRotationAngle = 0.8 * 3.1415;			// 最小旋转角度(弧度)
	double m_maxRotationAngle = 1.2 * 3.1415;			// 最大旋转角度(弧度)
	int m_nScales;		// 尺度缩放会分为多少个间隔
	int m_nRotations;	// 旋转角度会分为多少个间隔
	int m_nSlices;		// 总共需要遍历多少个角度
private:
	void createRTable();
	void findOrigin();
	std::vector< std::vector<cv::Vec2f> > scaleRTable(const std::vector< std::vector<cv::Vec2f> >& RTable, double ratio);
	std::vector< std::vector<cv::Vec2f> > rotateRTable(const std::vector< std::vector<cv::Vec2f> >& RTable, double angle);
	void showRTable(std::vector< std::vector<cv::Vec2f> > RTable);

public:
	GeneralHoughTransform(const cv::Mat& templateImage);
	void accumulate(const cv::Mat& image);
	void drawTemplate(cv::Mat& image, GHTPoint params);
	std::vector<GHTPoint> findTemplates(std::vector< std::vector< cv::Mat > >& accum, int threshold);
	void setTemplate(const cv::Mat& templateImage);

};

#endif /* GENERALHOUGHTRANSFORM_HPP_ */
