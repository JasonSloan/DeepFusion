#include <stdio.h>
#include <string>
#include <vector>
#include <chrono>

#include "opencv2/opencv.hpp"
#include "inference.hpp"
#include "main_utils.hpp"
#include "tqdm.hpp"

using namespace std;
using namespace cv;

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
    // some settings
    string model_path = "weights/last-relu-3output-i8-bs1.rknn";
    string image_path = "inputs/images/0002_1_000002.jpg";
    string save_path = "outputs/0002_1_000002.jpg";

    bool log = true;
    int niters = 10000;
	int imgcount;
	vector<vector<Mat>> imgs;
	vector<vector<string>> save_paths;
    int batch_size = 3;
    string images_path = "inputs/images";
	collect_data(images_path, batch_size, imgcount, imgs, save_paths);

    // read image
    Mat img = imread(image_path, IMREAD_COLOR);
    Mat im0 = img.clone();                  // for visulization
    vector<Mat> im0s;
    im0s.push_back(im0);

    // create model
    auto inferController = create_infer(model_path);
    auto total_start = std::chrono::high_resolution_clock::now();
    vector<Result> results;
    shared_future<Result> fu_result;
    for (int i = 0; i < niters; ++i){
        fu_result = inferController->forward(img, log);
        Result result = fu_result.get();
        results.push_back(result);
    }
    // draw_rectangles(results, im0s, save_paths);
	auto total_end = std::chrono::high_resolution_clock::now();
	auto avg_total_durations = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000. / niters;
	// printf("\nTotal time consuming per image:\ntotal: %.2f ms,\tpreprocess: %.2f ms,\tinfer %.2f ms,\tpostprocess %.2f ms\n", 
    //         avg_total_durations, avg_pre_duration, avg_infer_duration, avg_post_duration);
    return 0;
}