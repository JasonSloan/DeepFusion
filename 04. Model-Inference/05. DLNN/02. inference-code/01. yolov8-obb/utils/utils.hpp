#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <ctime>
#include <stdio.h>
#include <unistd.h>
#include <dirent.h>     // opendir和readdir包含在这里
#include <sys/stat.h>
#include <fstream>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "tqdm.hpp"
#include "yolo/yolo.h"
#include "yolo/model-utils.h"

using namespace std;
using namespace cv;


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


void draw_rectangles(
	vector<Result>& results, vector<Mat>& im0s, 
	vector<string>& save_paths
){
	for (int i = 0; i < results.size(); ++i) {
		Result result = results[i];
		Mat& im0 = im0s[i];
		for (int j = 0; j < result.rboxes.size(); j++) {
			float confidence = result.rboxes[j].score;
			std::vector<cv::Point> points;
			int x1 = result.rboxes[j].x1;
			int y1 = result.rboxes[j].y1;
			int x2 = result.rboxes[j].x2;
			int y2 = result.rboxes[j].y2;
			int x3 = result.rboxes[j].x3;
			int y3 = result.rboxes[j].y3;
			int x4 = result.rboxes[j].x4;
			int y4 = result.rboxes[j].y4;
			points.push_back(cv::Point(x1, y1));
			points.push_back(cv::Point(x2, y2));
			points.push_back(cv::Point(x3, y3));
			points.push_back(cv::Point(x4, y4));
			cv::polylines(
				im0, points, true, cv::Scalar(0, 255, 0), 2
			);
		}
		cv::imwrite(save_paths[i], im0);
	}
}

void collect_data(
	string& path, int& batch_size, int& imagecount, 
	vector<vector<Mat>>& imgs, vector<vector<string>>& save_paths, 
	vector<vector<string>>& unique_ids
){
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

long getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(epoch).count();
}

template <int M, int N>
void transfer_data(
	vector<vector<cv::Mat>>& m_imgs, Input (&i_imgs)[M][N], 
	vector<vector<string>>& unique_ids
){
	int n_batch = m_imgs.size();
	int batch_size = m_imgs[0].size();
	for (int n = 0; n < n_batch; ++n){
		for (int i = 0; i < batch_size; ++i) {
			cv::Mat img = m_imgs[n][i];
			int height = img.rows;
			int width = img.cols;
			int numel = height * width * 3;
			Input img_one;
			img_one.height = height;
			img_one.width = width;
			img_one.data = new unsigned char[numel];
			memcpy(img_one.data, img.data, numel);
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

std::string SplitString(const std::string &str, const std::string &delim){
	std::string out_str;
    std::string::size_type pos1, pos2;
    pos2 = str.find(delim);
    pos1 = 0;
    while (std::string::npos != pos2)
    {
        pos1 = pos2 + delim.size();
        pos2 = str.find(delim, pos1);
    }
    if (pos1 != str.length())
        out_str = str.substr(pos1);

	return out_str;
}

void removeSubstring(std::string& str, const std::string& substr) {
    size_t pos = str.find(substr);
    if (pos != std::string::npos) {
        str.erase(pos, substr.length());
    }
}





