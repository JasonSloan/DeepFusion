#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>

#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "main_utils.hpp"
#include "tqdm.hpp"
#include "inference.hpp"

using namespace std;
using namespace cv;


int main() {
	// RKNN只支持固定batch-size的多batch, 不支持动态batch, batch-size一定要和模型输入保持一致
	int niter = 1;
	int batch_size = 3;
	string model_path = "weights/last-relu-3output-i8-bs3.rknn";
	string path = "inputs/images";
	// string path = "inputs/videos/ch02_20230715000000_part0.mp4";
	int imgcount;
	vector<vector<Mat>> imgs;
	vector<vector<string>> save_paths;
	collect_data(path, batch_size, imgcount, imgs, save_paths);

	// create model
	auto inferController = create_infer(model_path, batch_size, false);
	if (inferController == nullptr) {
		spdlog::error("create_infer failed\n");
		return -1;
	}

	// infer
	spdlog::info("----> Start to infer...");
	for (int n = 0; n < niter; ++n){
		for (int i : tq::trange(imgs.size())){													// i is batch idx, imgs[i] is number i batch 
			shared_future<vector<Result>> fut = inferController->forward(imgs[i], false);		// true for log or not
			vector<Result> results = fut.get();													// asynchronous get result
			// draw_rectangles(results, imgs[i], save_paths[i]);									// draw rectangles and save image
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
	spdlog::info("----> Average time cost: preprocess {} ms, infer {} ms, postprocess {} ms, total {} ms", 
		avg_preprocess_time, avg_infer_time, avg_postprocess_time, avg_total_time);

    return 0;
}