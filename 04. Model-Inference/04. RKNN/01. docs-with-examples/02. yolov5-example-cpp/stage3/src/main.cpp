#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "main_utils.hpp"
#include "tqdm.hpp"
#include "inference.hpp"

using namespace std;
using namespace cv;

extern vector<vector<float>> records;

void collect_data(string& path, int& batch_size, int& imagecount, vector<vector<Mat>>& imgs, vector<vector<string>>& save_paths){
	bool is_video = ends_with(path, ".mp4");

	// prepare and infer
	int total = 0;
	int broken = 0;
	int batch_idx = 0;
	cout << "----> Start to read and collect images/video from path '" << path << "' ..." << endl;
	if (!is_video){
		vector<string> files;
		bool success = listdir(path, files);
		// push back imgs into a vector
		int num_batch = ceil(float(files.size()) / float(batch_size));
		imgs.resize(num_batch);
		save_paths.resize(num_batch);
		for (int i : tq::trange(num_batch * batch_size)){
			cv::Mat img = cv::imread(files[i], IMREAD_COLOR);
			if (img.empty()) {
				printf("Unable to read image %s\n", files[i].c_str());
				broken++;
				continue;
			}
			if ((i != 0) && (i % batch_size == 0))		// if read successfully then compute the batch_idx
				batch_idx++;
			imgs[batch_idx].push_back(img);
			std::string filename = getFileName(files[i]);
			string save_path = "outputs/" + filename;
			save_paths[batch_idx].push_back(save_path);
			total++;
		}
	} else {
		cv::VideoCapture cap(path);
		if (!cap.isOpened()) {
			printf("Unable to open video %s", path.c_str());
			return;
		}
		cv::Mat frame;
		int frame_count = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
		int num_batch = ceil(float(frame_count) / float(batch_size));
		imgs.resize(num_batch);
		save_paths.resize(num_batch);
		for (int i : tq::trange(num_batch * batch_size)) {
			cap >> frame;
			if (frame.empty()) {
				printf("Unable to read frame %d of video %s\n", total, path.c_str());
				broken++;
				continue;
			}
			if ((i != 0) && (i % batch_size == 0))		
				batch_idx++;
			imgs[batch_idx].push_back(frame);
			string save_path = "outputs/frame_" + std::to_string(total) + ".jpg";
			save_paths[batch_idx].push_back(save_path);
			total++;
		}
		cap.release();
	}
	imagecount = total - broken;
	if (imagecount % batch_size != 0) {
		imagecount = imagecount - imagecount % batch_size;
		imgs.pop_back();			// pop the last batch
		save_paths.pop_back();
	}
	printf("\nRead video/images of path '%s' successfully, total: %d, broken: %d, reserved: %d\n", path.c_str(), total, broken, imagecount);
}

int main() {
	// RKNN只支持固定batch-size的多batch, 不支持动态batch, batch-size一定要和模型输入保持一致
	int niter = 1;
	int batch_size = 1;
	string model_path = "weights/last-silu-1output-fp-bs3.rknn";
	string path = "inputs/images";
	// string path = "inputs/videos/ch02_20230715000000_part0.mp4";
	int imgcount;
	vector<vector<Mat>> imgs;
	vector<vector<string>> save_paths;
	collect_data(path, batch_size, imgcount, imgs, save_paths);

	// create model
	auto inferController = create_infer(model_path, batch_size, false);

	// infer
	printf("----> Start to infer...\n");
	for (int n = 0; n < niter; ++n){
		for (int i : tq::trange(imgs.size())){													// i is batch idx, imgs[i] is number i batch 
			shared_future<vector<Result>> fut = inferController->forward(imgs[i], false);		// true for log or not
			vector<Result> results = fut.get();													// asynchronous get result
			// draw_rectangles(results, imgs[i], save_paths[i]);								// draw rectangles and save image
		}
	}
	printf("----> Infer successfully!\n");

	// print elapsed time
	auto records = inferController->get_records();
	auto avg_preprocess_time = mean(records[0]) / float(batch_size);
	auto avg_infer_time = mean(records[1]) / float(batch_size);
	auto avg_postprocess_time = mean(records[2]) / float(batch_size);
	auto avg_total_time = avg_preprocess_time + avg_infer_time + avg_postprocess_time;
	printf("----> Average time cost:\n preprocess %.2f ms, infer %.2f ms, postprocess %.2f ms, total %.2f ms\n", 
		avg_preprocess_time, avg_infer_time, avg_postprocess_time, avg_total_time);

    return 0;
}