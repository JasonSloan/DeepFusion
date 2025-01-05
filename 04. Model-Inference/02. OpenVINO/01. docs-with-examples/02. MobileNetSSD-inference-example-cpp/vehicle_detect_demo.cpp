#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

class SharedTensorAllocator final : public ov::AllocatorImpl {
public:
	SharedTensorAllocator(const cv::Mat& img) : img(img) {}

	~SharedTensorAllocator() = default;

	void* allocate(const size_t bytes, const size_t) override {
		return bytes <= img.rows * img.step[0] ? img.data : nullptr;
	}

	void deallocate(void* handle, const size_t bytes, const size_t) override {}

	bool is_equal(const AllocatorImpl& other) const override {
		auto other_tensor_allocator = dynamic_cast<const SharedTensorAllocator*>(&other);
		return other_tensor_allocator != nullptr && other_tensor_allocator == this;
	}

private:
	const cv::Mat img;
};

int main(int argc, char** argv) {
	cv::Mat image = cv::imread("D:/bird_test/car_1.bmp");
	int ih = image.rows;
	int iw = image.cols;

	// 创建IE插件, 查询支持硬件设备
	ov::Core core;
	vector<string> availableDevices = core.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
		std::cout << availableDevices[i] << std::endl;
	}
	std::string model_xml = "D:/projects/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml";
	auto model = core.read_model(model_xml);

	// setting input data format and layout
	ov::preprocess::PrePostProcessor ppp(model);
	ov::preprocess::InputInfo& inputInfo = ppp.input();
	inputInfo.tensor().set_element_type(ov::element::u8);
	inputInfo.tensor().set_layout({ "NHWC" });

	// apply changes and get compiled model
	model = ppp.build();
	ov::CompiledModel cmodel = core.compile_model(model, "CPU");

	// create infer request
	auto request = cmodel.create_infer_request();

	// set input image
	ov::Shape input_shape = request.get_input_tensor().get_shape();
	size_t h = input_shape[1];
	size_t w = input_shape[2];
	size_t ch = input_shape[3];
	std::cout << "NHWC:" << input_shape[0] << "x" << input_shape[1] << "x" << h << "x" << w << std::endl;
	cv::Mat blob;
	cv::resize(image, blob, cv::Size(w, h));

	// fill data into tensor
	auto allocator = std::make_shared<SharedTensorAllocator>(blob);
	auto input_tensor = ov::Tensor(ov::element::u8, ov::Shape{ 1, h, w, ch }, ov::Allocator(allocator));
	request.set_input_tensor(input_tensor);

	// inference
	request.infer();

	// output[1, 1, N,7]  7: image_id, class, conf, x_min, y_min, x_max, y_max
	ov::Tensor output = request.get_output_tensor();
	size_t num = output.get_shape()[2];
	size_t cnum = output.get_shape()[3];
	std::cout << num << "x" << cnum << std::endl;
	cv::Mat prob(num, cnum, CV_32F, (float*)output.data());
	for (int i = 0; i < num; i++) {
		float conf = prob.at<float>(i, 2);
		if (conf > 0.75) {
			int x_min = static_cast<int>(prob.at<float>(i, 3)*iw);
			int y_min = static_cast<int>(prob.at<float>(i, 4)*ih);
			int x_max = static_cast<int>(prob.at<float>(i, 5)*iw);
			int y_max = static_cast<int>(prob.at<float>(i, 6)*ih);
			cv::rectangle(image, cv::Rect(x_min, y_min, x_max - x_min, y_max - y_min), Scalar(0, 0, 255), 2, 8, 0);
		}
	}
	cv::imshow("OpenVINO2022 - 车辆与车牌检测", image);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}
