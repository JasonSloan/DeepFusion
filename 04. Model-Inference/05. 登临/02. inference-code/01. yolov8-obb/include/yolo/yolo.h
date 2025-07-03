#pragma once

#include "yolo/pubDef.h"

class InferInterface{
public:
    virtual std::shared_future<std::vector<Result>> forward(Input* inputs, int& n_images, float conf_thre, bool inferLog=false) = 0;
	virtual std::vector<std::vector<float>> get_records() = 0;
};

std::shared_ptr<InferInterface> create_infer(std::string &file, bool modelLog, bool multi_label=true);

