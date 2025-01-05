#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

std::string labels_txt_file = "D:/python/pytorch_openvino_demo/imagenet_classes.txt";
std::vector<std::string> readClassNames();
std::vector<std::string> readClassNames()
{
	std::vector<std::string> classNames;

	std::ifstream fp(labels_txt_file);
	if (!fp.is_open())
	{
		printf("could not open file...\n");
		exit(-1);
	}
	std::string name;
	while (!fp.eof())
	{
		std::getline(fp, name);
		if (name.length())
			classNames.push_back(name);
	}
	fp.close();
	return classNames;
}

int main(int argc, char** argv) {
	std::vector<std::string> labels = readClassNames();
	cv::Mat image = cv::imread("D:/images/car_1.bmp");
	int ih = image.rows;
	int iw = image.cols;

	// 创建IE插件, 查询支持硬件设备
	ov::Core core;
	vector<string> availableDevices = core.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
		std::cout << availableDevices[i] << std::endl;
	}

	ov::CompiledModel model = core.compile_model("D:/projects/resnet18.xml", "AUTO");
	// create infer request
	auto request = model.create_infer_request();

	// set input image
	ov::Tensor input_tensor = request.get_input_tensor();
	ov::Shape input_shape = input_tensor.get_shape();
	size_t h = input_shape[2];
	size_t w = input_shape[3];
	size_t ch = input_shape[1];
	std::cout << "NCHW:" << input_shape[0] << "x" << input_shape[1] << "x" << h << "x" << w << std::endl;
	cv::Mat rgb, blob;
	cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
	cv::resize(rgb, blob, cv::Size(w, h));
	blob.convertTo(blob, CV_32F);
	blob = blob / 255.0;
	cv::subtract(blob, cv::Scalar(0.485, 0.456, 0.406), blob);
	cv::divide(blob, cv::Scalar(0.229, 0.224, 0.225), blob);

	// put image data into tensor
	// HWC => NCHW
	int image_size = w * h;
	float* data = input_tensor.data<float>();
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t c = 0; c < ch; c++) {
				data[image_size*c + row * w + col] = blob.at<Vec3f>(row, col)[c];
			}
		}
	}

	// inference
	request.infer();

	// output
	ov::Tensor output = request.get_output_tensor();
	size_t num = output.get_shape()[0];
	size_t cnum = output.get_shape()[1];
	std::cout << num << "x" << cnum << std::endl;
	cv::Mat prob(num, cnum, CV_32F, (float*)output.data());

	cv::Point maxL, minL;
	double maxv, minv;
	cv::minMaxLoc(prob, &minv, &maxv, &minL, &maxL);
	int max_index = maxL.x;
	std::cout << "label id: " << max_index << std::endl;
	cv::putText(image, labels[max_index], cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2, 8);
	cv::imshow("输入图像", image);
	cv::waitKey(0);
	return 0;
}
