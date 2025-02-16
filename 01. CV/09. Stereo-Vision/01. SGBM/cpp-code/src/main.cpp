#include <stdio.h>
#include <string>
#include <fstream>
#include "json.hpp"
#include "estimator.h"


using namespace std;
using json=nlohmann::json;

int main(){
    bool use_video = true;
    int minDisparity = 1;
    int numDisparities = 64;
    int blockSize = 5;
    int P1 = -1;
    int P2 = -1;
    int disp12MaxDiff = -1;
    int preFilterCap = 31;
    int uniquenessRatio = 10;
    int speckleWindowSize = 100;
    int speckleRange = 1;
    int mode = 1;
    bool dynamic_tune = false;
    string config_path = "cpp-SGBM/src/configs.json";
    string source = "stereo-video/car.avi";
    auto estimator = createSGBMEstimator(
        use_video,
        minDisparity,
        numDisparities,
        blockSize,
        P1,
        P2,
        disp12MaxDiff,
        preFilterCap,
        uniquenessRatio,
        speckleWindowSize,
        speckleRange,
        mode,
        dynamic_tune,
        config_path
    );
    estimator->estimate(source);
    estimator->compute_time_cost();
    return 0;
}