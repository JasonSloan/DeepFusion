#ifndef INFERENCE_HPP
#define INFERENCE_HPP
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

class InferInterface
{
public:
    virtual cv::Mat forward(const std::string& imagePath) = 0;
};
std::shared_ptr<InferInterface> create_infer(const std::string &file);

#endif //INFERENCE_HPP
