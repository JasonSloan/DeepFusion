#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <dirent.h>     // opendir和readdir包含在这里
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>

#include "opencv2/opencv.hpp"
#include "inference.hpp"

using namespace std;
using namespace cv;

const vector<Scalar> COLORS = {
	{255, 0, 0},
	{0, 255, 0},
	{0, 0, 255},
	{0, 255, 255}
};

const string LABELS[] = {"head", "helmet", "person", "lookout"};

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

float mean(vector<float> x){
    float sum = 0;
    for (int i = 0; i < x.size(); ++i){
        sum += x[i];
    }
    return sum / x.size();
}

class TIMER{
    private:
        struct timeval start_time, stop_time;
        double __get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }
        char indent[40];

    public:
        TIMER(){}
        ~TIMER(){}

        void indent_set(char* s){
            strcpy(indent, s);
        }
        void indent_set(const char* s){
            strcpy(indent, s);
        }

        void tik(){
            gettimeofday(&start_time, NULL);
        }

        void tok(){
            gettimeofday(&stop_time, NULL);
        }

        void print_time(char* str, int count=1){
            printf("%s", indent);
            printf("%s use: %f ms\n", str, get_time() / (float)count);
        }
        void print_time(const char* str, int count=1){
            printf("%s", indent);
            printf("%s use: %f ms\n", str, get_time() / (float)count);
        }

        float get_time(){
            return (__get_us(stop_time) - __get_us(start_time))/1000;
        }
};