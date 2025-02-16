#include <string>
#include <future>


class SGBMEstimatorInterface{
public:
    virtual void estimate(std::string& source) = 0;
    virtual void compute_time_cost() = 0;
};

std::shared_ptr<SGBMEstimatorInterface> createSGBMEstimator(
    bool use_video=false,
    int minDisparity=1,
    int numDisparities=64,
    int blockSize=5,
    int P1=-1,
    int P2=-1,
    int disp12MaxDiff=-1,
    int preFilterCap=31,
    int uniquenessRatio=10,
    int speckleWindowSize=100,
    int speckleRange=1,
    int mode=1, // cv::StereoSGBM::MODE_HH
    bool dynamic_tune=false,
    std::string config_path=""
);