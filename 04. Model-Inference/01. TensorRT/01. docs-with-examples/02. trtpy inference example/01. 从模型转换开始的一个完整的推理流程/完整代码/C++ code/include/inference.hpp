#ifndef INFERENCE_HPP
#define INFERENCE_HPP
#include <memory>
#include <vector>
#include <string>
#include "pybind11.hpp"
#include "opencv2/opencv.hpp"

struct Result{
    pybind11::array_t<uint8_t> output_array;
    int code;
    std::string id;
};

class InferInterface{
public:
    virtual Result forward(cv::Mat& cvimage, const std::string& id) = 0;
};

std::shared_ptr<InferInterface> create_infer(const std::string &file);

#endif //INFERENCE_HPP
