#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;
using namespace std;

static const char* const items[] = {
				"0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
				"<Anhui>", "<Beijing>", "<Chongqing>", "<Fujian>",
				"<Gansu>", "<Guangdong>", "<Guangxi>", "<Guizhou>",
				"<Hainan>", "<Hebei>", "<Heilongjiang>", "<Henan>",
				"<HongKong>", "<Hubei>", "<Hunan>", "<InnerMongolia>",
				"<Jiangsu>", "<Jiangxi>", "<Jilin>", "<Liaoning>",
				"<Macau>", "<Ningxia>", "<Qinghai>", "<Shaanxi>",
				"<Shandong>", "<Shanghai>", "<Shanxi>", "<Sichuan>",
				"<Tianjin>", "<Tibet>", "<Xinjiang>", "<Yunnan>",
				"<Zhejiang>", "<police>",
				"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
				"K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
				"U", "V", "W", "X", "Y", "Z"
};

// 固定代码,不用管
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

// 识别模型接受两个输入,所以要获取模型这两个输入的名字
std::string m_LprInputName;
std::string m_LprInputSeqName;
ov::InferRequest license_request;
void fetch_plate_text(cv::Mat &img, cv::Mat &temp_roi, cv::Point &txt_loc);
int main(int argc, char** argv) {
	cv::Mat image = cv::imread("D:/bird_test/car_test.png");
	int ih = image.rows;
	int iw = image.cols;

	// 创建IE插件, 查询支持硬件设备
	ov::Core core;
	vector<string> availableDevices = core.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		printf("supported device name : %s \n", availableDevices[i].c_str());
		std::cout << availableDevices[i] << std::endl;
	}
	// model为检测模型
	std::string model_xml = "D:/projects/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml";
	auto model = core.read_model(model_xml);

	// 设置检测模型的输入(非必要)
	ov::preprocess::PrePostProcessor ppp(model);
	ov::preprocess::InputInfo& inputInfo = ppp.input();
	inputInfo.tensor().set_element_type(ov::element::u8);
	inputInfo.tensor().set_layout({ "NHWC" });

	// apply changes and get compiled model
	model = ppp.build();
	ov::CompiledModel cmodel = core.compile_model(model, "CPU");

	// create infer request
	auto request = cmodel.create_infer_request();

	// license_model为识别模型
	std::string license_xml = "D:/projects/intel/license-plate-recognition-barrier-0001/FP32/license-plate-recognition-barrier-0001.xml";
	auto license_model = core.read_model(license_xml);
	ov::OutputVector inputs = license_model->inputs();

	// 获得识别模型的两个输入的名字,这两个输入第一种是4维度的NCHW,第二个是2维度的(不知道是个啥)
	for (auto input : inputs) {
		if (input.get_shape().size() == 4) {
			m_LprInputName = input.get_any_name();
		}
		if (input.get_shape().size() == 2)
			m_LprInputSeqName = input.get_any_name();
	}
	
	// 设置识别模型的输入
	ov::preprocess::PrePostProcessor p2p(license_model);
	ov::preprocess::InputInfo& license_inputInfo = p2p.input(m_LprInputName);
	license_inputInfo.tensor().set_element_type(ov::element::u8);
	license_inputInfo.tensor().set_layout({ "NCHW" });

	// apply changes and get compiled model
	license_model = p2p.build();
	ov::CompiledModel clicense_model = core.compile_model(license_model, "CPU");

	// create infer request
	license_request = clicense_model.create_infer_request();

	// 检测模型前处理
	ov::Shape input_shape = request.get_input_tensor().get_shape();
	size_t h = input_shape[1];
	size_t w = input_shape[2];
	size_t ch = input_shape[3];
	std::cout << "NHWC:" << input_shape[0] << "x" << input_shape[1] << "x" << h << "x" << w << std::endl;
	cv::Mat blob;
	cv::resize(image, blob, cv::Size(w, h));

	// 将图像数据放入到模型的输入中国
	auto allocator = std::make_shared<SharedTensorAllocator>(blob);
	auto input_tensor = ov::Tensor(ov::element::u8, ov::Shape{ 1, h, w, ch }, ov::Allocator(allocator));
	request.set_input_tensor(input_tensor);

	// inference
	request.infer();

	// output
	ov::Tensor output = request.get_output_tensor();
	size_t num = output.get_shape()[2];
	size_t cnum = output.get_shape()[3];
	std::cout << num << "x" << cnum << std::endl;
	// [N, 7], 7: image_id, class, conf, x_min, y_min, x_max, y_max
	cv::Mat prob(num, cnum, CV_32F, (float*)output.data());
	int padding = 5;
	for (int i = 0; i < num; i++) {
		float conf = prob.at<float>(i, 2);
		int label_id = prob.at<float>(i, 1);
		if (conf > 0.75) {
			int x_min = static_cast<int>(prob.at<float>(i, 3)*iw);
			int y_min = static_cast<int>(prob.at<float>(i, 4)*ih);
			int x_max = static_cast<int>(prob.at<float>(i, 5)*iw);
			int y_max = static_cast<int>(prob.at<float>(i, 6)*ih);
			cv::Rect box(x_min, y_min, x_max - x_min, y_max - y_min);
			// label_id为2的对应的类比是车牌, 为1对应的类别是车
			if (label_id == 2) {
				// 取车牌ROI时每个边多向外侧扩展5个像素
				cv::Rect plate_roi;
				plate_roi.x = box.x - padding;
				plate_roi.y = box.y - padding;
				plate_roi.width = box.width + 2 * padding;
				plate_roi.height = box.height + 2 * padding;
				cv::Mat temp_roi = image(plate_roi);
				cv::Point txt_loc(box.x, box.y);
				fetch_plate_text(image, temp_roi, txt_loc);
				cv::rectangle(image, box, Scalar(0, 255, 255), 2, 8, 0);
			}
			else {
				cv::rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
			}
		}
	}
	cv::imshow("OpenVINO2022 - 车辆与车牌检测", image);
	cv::waitKey(0);
	cv::destroyAllWindows();
	return 0;
}

void fetch_plate_text(cv::Mat &frame, cv::Mat &temp_roi, cv::Point &txt_loc) {
	// 将车牌ROI区域的图像数据填入识别模型的输入中
	ov::Shape input_shape = license_request.get_tensor(m_LprInputName).get_shape();
	size_t ch = input_shape[1];
	size_t h = input_shape[2];
	size_t w = input_shape[3];
	cv::Mat blob;
	cv::resize(temp_roi, blob, cv::Size(w, h));
	int image_size = w * h;
	auto input_tensor = license_request.get_tensor(m_LprInputName);
	uchar* rp_data = input_tensor.data<uchar>();
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t c = 0; c < ch; c++) {
				rp_data[image_size*c + row * w + col] = blob.at<Vec3b>(row, col)[c];
			}
		}
	}

	ov::Tensor inputSeqTensor = license_request.get_tensor(m_LprInputSeqName);
	float* data = inputSeqTensor.data<float>();
	std::fill(data, data + inputSeqTensor.get_shape()[0], 1.0f);

	// call infer
	license_request.infer();

	// post process
	std::string result;
	result.reserve(14u + 6u);  // the longest province name + 6 plate signs
	ov::Tensor lprOutputTensor = license_request.get_output_tensor();
	const auto out_data = lprOutputTensor.data<float>();
	for (int i = 0; i < 88; i++) {
		int32_t val = int32_t(out_data[i]);
		if (val == -1) {
			break;
		}
		result += items[val];
	}
	cv::putText(frame, result.c_str(), cv::Point(txt_loc.x - 50, txt_loc.y - 10), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 255, 0), 2, 8);
}
