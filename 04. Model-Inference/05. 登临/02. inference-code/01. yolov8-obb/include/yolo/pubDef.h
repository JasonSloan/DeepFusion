#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <cstring>

struct Input{
	unsigned char* data;
	int height;
	int width;
};

struct RBox{
	int x1{-1};
	int y1{-1};
	int x2{-1};
	int y2{-1};
	int x3{-1};
	int y3{-1};
	int x4{-1};
	int y4{-1};
    int label = -1;
	int track_id = -1;
	float score = -1.;
};

struct Result{
	std::vector<RBox> rboxes;
};


