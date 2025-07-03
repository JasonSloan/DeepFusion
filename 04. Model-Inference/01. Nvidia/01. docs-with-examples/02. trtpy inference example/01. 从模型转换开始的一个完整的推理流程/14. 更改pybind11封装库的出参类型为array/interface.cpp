#include "pybind11.hpp"
#include "inference.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
namespace py = pybind11;


class SRInfer { 
public:
	SRInfer(string engine){ instance_ = create_infer(engine); }

	bool valid(){ return instance_ != nullptr; }

	Result forward(const py::array& imageArray, const string& id){
		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");
		cv::Mat cvimage(imageArray.shape(0), imageArray.shape(1), CV_8UC3, (unsigned char*)imageArray.data(0));
		Result result = instance_->forward(cvimage, id);
		return result;
}

private:
	shared_ptr<InferInterface> instance_;
}; 

PYBIND11_MODULE(sr, m){
	py::class_<Result>(m, "Result")
        .def_readonly("output_array", &Result::output_array)									// 返回的图像数据
        .def_readonly("code", &Result::code)													// 返回的状态码
		.def_readonly("id", &Result::id);														

    py::class_<SRInfer>(m, "sr")
		.def(py::init<string>(), py::arg("engine"))												// 构造函数
		.def_property_readonly("valid", &SRInfer::valid, "Infer is valid")						// 判断是否构造成功
		.def("forward", &SRInfer::forward, py::arg("imageArray"), py::arg("id"));;				// 前向推理
}
 