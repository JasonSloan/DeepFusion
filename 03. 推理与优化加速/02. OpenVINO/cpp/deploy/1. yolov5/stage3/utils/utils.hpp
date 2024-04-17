#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>     // opendir和readdir包含在这里
#include <sys/stat.h>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "tqdm.hpp"
#include "inference.h"

using namespace std;
using namespace cv;

const vector<Scalar> COLORS = {
	{255, 0, 0},		// head
	{0, 255, 0},		// helmet
	{0, 0, 255},		// person
	{0, 255, 255}		// lookout
};

const string LABELS[] = {"head", "helmet", "person", "lookout"};

void print_avaliable_devices() {
	ov::Core core;
	vector<string> availableDevices = core.get_available_devices();
	for (int i = 0; i < availableDevices.size(); i++) {
		spdlog::info("supported device name : {}", availableDevices[i]);
	}
}

int listdir(string& input,  vector<string>& files_vector) {
    DIR* pDir = opendir(input.c_str());
    if (!pDir) {
        cerr << "Error opening directory: " << strerror(errno) << endl;
        return -1;
    }
    struct dirent* ptr;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());
	return 0;
}

bool ends_with(const std::string &str, const std::string &ending) {
    if (str.length() >= ending.length()) {
        return str.compare(str.length() - ending.length(), ending.length(), ending) == 0;
    }
    return false;
}

string getFileName(const std::string& file_path, bool with_ext=true){
	int index = file_path.find_last_of('/');
	if (index < 0)
		index = file_path.find_last_of('\\');
    std::string tmp = file_path.substr(index + 1);
    if (with_ext)
        return tmp;
    std::string img_name = tmp.substr(0, tmp.find_last_of('.'));
    return img_name;
}

void draw_rectangles(vector<Result>& results, vector<Mat>& im0s, vector<string>& save_paths){
	for (int i = 0; i < results.size(); ++i) {
		Result result = results[i];
		Mat& im0 = im0s[i];
		for (int j = 0; j < result.boxes.size(); j++) {
			cv::rectangle(
				im0, 
				cv::Point(result.boxes[j][0], result.boxes[j][1]), 
				cv::Point(result.boxes[j][2], result.boxes[j][3]), 
				COLORS[result.labels[j]], 
				5, 8, 0
				);
			// cv::putText(im0, LABELS[result.labels[i]], cv::Point(result.boxes[i][0], result.boxes[i][1]), cv::FONT_HERSHEY_SIMPLEX, 1.4, COLORS[result.labels[i]], 2);
			cv::imwrite(save_paths[i], im0);
		}
	}
}

void collect_data(string& path, int& batch_size, int& imagecount, vector<vector<Mat>>& imgs, vector<vector<string>>& save_paths, vector<vector<string>>& unique_ids){
	bool is_video = ends_with(path, ".mp4");

	// prepare and infer
	int total = 0;
	int broken = 0;
	int batch_idx = 0;
	spdlog::info("----> Start to read and collect images/video from path '{}' ...", path);
	if (!is_video){
		vector<string> files;
		bool success = listdir(path, files);
		// push back imgs into a vector
		int num_batch = ceil(float(files.size()) / float(batch_size));
		imgs.resize(num_batch);
		save_paths.resize(num_batch);
		unique_ids.resize(num_batch);
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
			string filename = getFileName(files[i], true);
			string save_path = "outputs/" + filename;
			save_paths[batch_idx].push_back(save_path);
			string unique_id = getFileName(files[i], false);
			unique_ids[batch_idx].push_back(unique_id);
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
	printf("\n");
	spdlog::info("Read video/images of path '{}' successfully, total: {}, broken: {}, reserved: {}", path.c_str(), total, broken, imagecount);
}

template <int M, int N>
void transfer_data(vector<vector<cv::Mat>>& m_imgs, Input (&i_imgs)[M][N], vector<vector<string>>& unique_ids){
	int n_batch = m_imgs.size();
	int batch_size = m_imgs[0].size();
	for (int n = 0; n < n_batch; ++n){
		for (int i = 0; i < batch_size; ++i) {
			cv::Mat img = m_imgs[n][i];
			int height = img.rows;
			int width = img.cols;
			Input img_one;
			img_one.unique_id = unique_ids[n][i];
			img_one.height = height;
			img_one.width = width;
			img_one.data = img.data;
			i_imgs[n][i] = img_one;
		}
	}
}

float mean(vector<float> x){
    float sum = 0;
    for (int i = 0; i < x.size(); ++i){
        sum += x[i];
    }
    return sum / x.size();
}

#endif