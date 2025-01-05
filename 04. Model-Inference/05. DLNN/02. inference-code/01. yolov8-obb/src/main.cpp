#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <future>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "tqdm.hpp"
#include "utils.hpp"
#include "yolo/yolo.h"
#include "yolo/model-utils.h"

using namespace std;
using namespace cv;


void test_model(string &modelPath, 
				int batch_size,
				float conf_thre,
				string input_dir,
				int n_iters,
				bool modelLog,
				bool inferLog,
				bool save_results,
				bool multi_label){

	Input i_imgs[1][1];	// ! 必须显示指定i_imgs这两个维度(需要手动计算一下)

	// create infer
	shared_ptr<InferInterface> inferController;
	inferController = create_infer(modelPath, modelLog, multi_label);
	if (inferController == nullptr) return;	

	//prepare data
	int imgcount;
	vector<vector<Mat>> m_imgs;
	vector<vector<string>> save_paths;
	vector<vector<string>> unique_ids;
	collect_data(input_dir, batch_size, imgcount, m_imgs, save_paths, unique_ids);
	int n_batch = imgcount / batch_size;
	transfer_data(m_imgs, i_imgs, unique_ids);	// transfer data from cv::Mat to Input struct

	// infer
	spdlog::info("----> Start infering......");
	for (int n = 0; n < n_iters; ++n){
		for (int i : tq::trange(n_batch)){																				// i is batch idx, imgs[i] is number i batch 
			shared_future<vector<Result>> fut = inferController->forward(i_imgs[i], batch_size, conf_thre, inferLog);		// true for log or not
			vector<Result> results = fut.get();																		// asynchronous get result
			if (save_results) draw_rectangles(results, m_imgs[i], save_paths[i]);										// draw rectangles and save image
		}
	}
	printf("\n");
	spdlog::info("----> Infer successfully!");

	// print elapsed time
	auto records = inferController->get_records();
	auto avg_preprocess_time = mean(records[0]) / float(batch_size);
	auto avg_infer_time = mean(records[1]) / float(batch_size);
	auto avg_postprocess_time = mean(records[2]) / float(batch_size);
	auto avg_total_time = avg_preprocess_time + avg_infer_time + avg_postprocess_time;
	spdlog::info("----> Average time cost for {} loops with batch size {}:\n preprocess {} ms, infer {} ms, postprocess {} ms, total {} ms", 
		records[0].size(), batch_size, avg_preprocess_time, avg_infer_time, avg_postprocess_time, avg_total_time);

	std::this_thread::sleep_for(std::chrono::seconds(100));
}

int main(){
	string modelPath_onnx = "weights/onnx/v8/yolov8s-obb.onnx";
	string modelPath_engine = "weights/engine/v8/v8-fp32-official-obb.engine";
	int batch_size = 1;
	int max_batch_size = 16;	
	float conf_thre = 0.25;

	string input_dir = "inputs/images";
	int n_iters = 1; // !注意，如果use_callback为true且niters>1，可能会出现问题，因为只new出了一批图片作为输入，在循环的时候，如果超过队列长度，可能会将某些部分的图片数据释放
	
	bool modelLog = true;
	bool inferLog = false;
	bool save_results = true;
	bool multi_label = false;
	bool compileModel = false; // compile the rlym model to engine
	
	if (compileModel)
		compile_model(modelPath_onnx, modelPath_engine, max_batch_size);

	test_model(
		modelPath_engine, batch_size, conf_thre, input_dir, 
		n_iters, modelLog, inferLog, save_results, multi_label
	);
    return 0;
};