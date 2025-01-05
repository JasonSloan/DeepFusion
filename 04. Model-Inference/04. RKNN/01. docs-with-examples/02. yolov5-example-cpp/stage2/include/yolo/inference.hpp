#pragma once

#include <string>
#include <vector>
#include <memory>
#include <future>

#include "opencv2/opencv.hpp"

struct Result{
	std::vector<std::vector<int>> boxes;
	std::vector<int> labels;
	std::vector<float> scores;
};

class InferInterface{
public:
	virtual std::shared_future<Result> forward(cv::Mat input_image, bool log) = 0;
};

std::shared_ptr<InferInterface> create_infer(std::string modelPath, int batch_size, bool modelLog);

std::shared_ptr<InferInterface> create_infer(std::string modelPath);