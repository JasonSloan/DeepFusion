#include <stdio.h>
#include <string>
#include <vector>
#include <fstream>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include "json.hpp"
#include "spdlog/spdlog.h"
#include "opencv2/opencv.hpp"
#include "estimator.h"

using namespace std;
using namespace cv;
using json=nlohmann::json;


class SGBMEstimator: public SGBMEstimatorInterface{
public:
    SGBMEstimator(
        bool use_video,
        int minDisparity,
        int numDisparities,
        int blockSize,
        int P1,
        int P2,
        int disp12MaxDiff,
        int preFilterCap,
        int uniquenessRatio,
        int speckleWindowSize,
        int speckleRange,
        int mode, // cv::StereoSGBM::MODE_HH
        bool dynamic_tune,
        string config_path
    ): use_video_(use_video), minDisparity_(minDisparity), 
        numDisparities_(numDisparities), blockSize_(blockSize), 
        P1_(P1), P2_(P2), disp12MaxDiff_(disp12MaxDiff), 
        preFilterCap_(preFilterCap), uniquenessRatio_(uniquenessRatio), 
        speckleWindowSize_(speckleWindowSize), speckleRange_(speckleRange), 
        mode_(mode), dynamic_tune_(dynamic_tune){
        is_bar_set_ = false;
        if (!use_video_) list_available_cameras();
        init_sgbm();
        load_config(config_path);
    }

    ~SGBMEstimator()=default;

    void list_available_cameras(int max_range = 10, bool verbose = true, bool show = true) {
        std::vector<int> available_devices;
        for (int camera_idx = 0; camera_idx < max_range; ++camera_idx) {
            cv::VideoCapture cap(camera_idx);
            if (cap.isOpened()) {
                available_devices.push_back(camera_idx);
                if (show) {
                    cv::Mat frame;
                    bool ret = cap.read(frame);
                    while (ret) {
                        cv::putText(
                            frame,
                            "camera " + std::to_string(camera_idx),
                            cv::Point(0, 40),
                            cv::FONT_HERSHEY_SIMPLEX,
                            1,
                            cv::Scalar(0, 255, 0),
                            2
                        );
                        cv::imshow("camera " + std::to_string(camera_idx), frame);
                        if (cv::waitKey(1) == 'q') {
                            cv::destroyAllWindows();
                            break;
                        }
                        ret = cap.read(frame);
                    }
                }
                cap.release();
            }
        }

        if (verbose) {
            spdlog::info("Available camera devices are: ");
            for (int device : available_devices) 
                spdlog::info("{} ", device);
        }
    }

    void init_sgbm(){
        init_windows();
        create_stereo_sgbm();
    }

    void load_config(const std::string& config_path) {
        std::ifstream i(config_path);
        if (!i.is_open()) {
            spdlog::error("Error: Cannot open config file!");
            return;
        }

        json j;
        i >> j;
        auto left_camera_matrix = j["left"]["camera_matrix"].get<vector<vector<double>>>();
        auto left_distortion = j["left"]["distortion"].get<vector<double>>(); 
        auto right_camera_matrix = j["right"]["camera_matrix"].get<vector<vector<double>>>();
        auto right_distortion = j["right"]["distortion"].get<vector<double>>(); 
        auto R = j["R"].get<vector<vector<double>>>();
        auto T = j["T"].get<vector<double>>(); 
        
        // 因为使用json读出来的数据不连续, 所以要将数据拷贝到cv::Mat中以保证数据连续
        left_camera_matrix_ = cv::Mat(left_camera_matrix.size(), left_camera_matrix[0].size(), CV_64F);
        for (int r = 0; r < left_camera_matrix.size(); ++r)
            for (int c = 0; c < left_camera_matrix[0].size(); ++c)
                left_camera_matrix_.at<double>(r, c) = left_camera_matrix[r][c];

            right_camera_matrix_ = cv::Mat(right_camera_matrix.size(), right_camera_matrix[0].size(), CV_64F);
            for (int r = 0; r < right_camera_matrix.size(); ++r)
                for (int c = 0; c < right_camera_matrix[0].size(); ++c)
                    right_camera_matrix_.at<double>(r, c) = right_camera_matrix[r][c];

            R_ = cv::Mat(R.size(), R[0].size(), CV_64F);
            for (int r = 0; r < R.size(); ++r)
                for (int c = 0; c < R[0].size(); ++c)
                    R_.at<double>(r, c) = R[r][c];

            // 这里必须克隆一下, 虽然这里是一维数据, 数据已经是连续的, 但是由于构造的时候直接传的是一个外来指针, 为了防止外来指针意外被释放, 这里clone一下
            left_distortion_ = cv::Mat(5, 1, CV_64F, left_distortion.data()).clone();
            right_distortion_ = cv::Mat(5, 1, CV_64F, right_distortion.data()).clone();
            T_ = cv::Mat(3, 1, CV_64F, T.data()).clone();               

        spdlog::info("Camera config loaded successfully!");
    }

    void init_windows() {
        left_frame_win_name_ = "left frame";  // 左相机的原图
        right_frame_win_name_ = "right frame";  // 右相机的原图
        disp_color_win_name_ = "disparity color";  // 渲染出的深度图的彩色图
        cv::namedWindow(left_frame_win_name_, cv::WINDOW_AUTOSIZE);
        cv::namedWindow(disp_color_win_name_, cv::WINDOW_AUTOSIZE);
    }

    void create_stereo_sgbm() {
        int img_channels = 3;
        if (dynamic_tune_) {
            if (!is_bar_set_) {
                // 创建滑动条
                cv::createTrackbar("numDisparities", disp_color_win_name_, nullptr, 160, nullptr);
                cv::setTrackbarPos("numDisparities", disp_color_win_name_, 16);
                cv::createTrackbar("blockSize", disp_color_win_name_, nullptr, 11, nullptr);
                cv::setTrackbarPos("blockSize", disp_color_win_name_, 3);
                is_bar_set_ = true;
            }

            // 获取滑动条的值
            numDisparities_ = cv::getTrackbarPos("numDisparities", disp_color_win_name_);
            numDisparities_ = (numDisparities_ / 16) * 16;  // 必须被16整除

            blockSize_ = cv::getTrackbarPos("blockSize", disp_color_win_name_);
            blockSize_ = (blockSize_ % 2 == 1) ? blockSize_ : blockSize_ + 1;  // 必须是奇数
        }

        // 设置 P1 和 P2 的默认值
        if (P1_ == -1) P1_ = 8 * img_channels * blockSize_ * blockSize_;
        if (P2_ == -1) P2_ = 32 * img_channels * blockSize_ * blockSize_;

        // 创建 StereoSGBM 对象
        stereo_ = cv::StereoSGBM::create(
            minDisparity_,
            numDisparities_,
            blockSize_,
            P1_,
            P2_,
            disp12MaxDiff_,
            preFilterCap_,
            uniquenessRatio_,
            speckleWindowSize_,
            speckleRange_,
            mode_
        );
    }

    void create_mapping_table(
        vector<cv::VideoCapture>& caps, Size& size, 
        Mat& left_map1, Mat& left_map2, Mat& right_map1, Mat& right_map2,
        Mat& R1, Mat& R2, Mat& P1, Mat& P2, Mat& Q
    ) {
        // Check if using two separate USB cameras
        bool two_usbs = caps.size() == 2;
        int width = two_usbs ? static_cast<int>(caps[0].get(cv::CAP_PROP_FRAME_WIDTH))
                            : static_cast<int>(caps[0].get(cv::CAP_PROP_FRAME_WIDTH) / 2);
        int height = static_cast<int>(caps[0].get(cv::CAP_PROP_FRAME_HEIGHT));
        size = cv::Size(width, height);

        // Stereo Rectification
        cv::Rect validPixROI1, validPixROI2;
        cv::stereoRectify(left_camera_matrix_, left_distortion_,
                        right_camera_matrix_, right_distortion_,
                        size, R_, T_, R1, R2, P1, P2, Q,
                        cv::CALIB_ZERO_DISPARITY, 0, size, &validPixROI1, &validPixROI2);

        // Compute rectification maps
        cv::initUndistortRectifyMap(left_camera_matrix_, left_distortion_, R1, P1, size, CV_16SC2, left_map1, left_map2);
        cv::initUndistortRectifyMap(right_camera_matrix_, right_distortion_, R2, P2, size, CV_16SC2, right_map1, right_map2);

        // Check if the mapping tables were created successfully
        if (left_map1.empty() || left_map2.empty() || right_map1.empty() || right_map2.empty()) {
            spdlog::error("Error: Failed to create stereo rectification mapping tables!");
            return;
        }
    }

    static void onmouse_pick_points(int event, int x, int y, int flags, void* param) {
        if (event == cv::EVENT_LBUTTONDOWN) {
            cv::Mat* threeD = static_cast<cv::Mat*>(param); // 将 param 转换为 cv::Mat 指针
            if (threeD->empty()) {
                spdlog::error("3D data is empty!");
                return;
            }

            // 确保坐标在图像范围内
            if (x >= 0 && x < threeD->cols && y >= 0 && y < threeD->rows) {
                // 获取 3D 点
                cv::Vec3f point = threeD->at<cv::Vec3f>(y, x);

                // 打印像素坐标
                spdlog::info("Pixel coordinates: x = {}, y = {}", x, y);

                // 打印世界坐标（单位：米）
                spdlog::info("World coordinates: x = {:.2f}m, y = {:.2f}m, z = {:.2f}m",
                            point[0] / 1000.0f, point[1] / 1000.0f, point[2] / 1000.0f);

                // 计算距离
                float distance = std::sqrt(point[0] * point[0] + point[1] * point[1] + point[2] * point[2]);
                distance /= 1000.0f; // 转换为米
                spdlog::info("Distance: {:.2f}m\n", distance);
            } else {
                spdlog::error("Clicked coordinates are out of range!");
            }
        }
    }

    int count_files(const std::string& input) {
        vector<string> files_vector;
        DIR* pDir = opendir(input.c_str());
        struct dirent* ptr;
        while ((ptr = readdir(pDir)) != nullptr) 
            if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) 
                files_vector.push_back(input + "/" + ptr->d_name);
        closedir(pDir);

        return files_vector.size();
    }

    void save_rectified(const cv::Mat& imgL, const cv::Mat& imgR, 
                        const cv::Mat& imgL_rectified, const cv::Mat& imgR_rectified, 
                        const std::string& save_dir = "rectified-images") {
        if (access(save_dir.c_str(), 0) != F_OK)
            mkdir(save_dir.c_str(), S_IRWXU);

        // Concatenate images horizontally
        cv::Mat cat_img_org, cat_img_rectified;
        cv::hconcat(imgL, imgR, cat_img_org);
        cv::hconcat(imgL_rectified, imgR_rectified, cat_img_rectified);

        // Get image dimensions
        int height = cat_img_org.rows;
        int width = cat_img_org.cols;

        // Draw horizontal green lines every 40 pixels
        for (int i = 0; i < height; i += 40) {
            cv::line(cat_img_org, cv::Point(0, i), cv::Point(width, i), cv::Scalar(0, 255, 0), 2);
            cv::line(cat_img_rectified, cv::Point(0, i), cv::Point(width, i), cv::Scalar(0, 255, 0), 2);
        }

        // Generate file names based on existing files
        int file_count = count_files(save_dir);
        std::string save_path_org = save_dir + "/" + std::to_string(file_count + 1) + "-org.jpg";
        std::string save_path_rectified = save_dir + "/" + std::to_string(file_count + 1) + "-rectified.jpg";

        // Save images
        cv::imwrite(save_path_org, cat_img_org);
        cv::imwrite(save_path_rectified, cat_img_rectified);

        spdlog::info("Saved rectified images to {}", save_dir);
    }

    virtual void estimate(std::string& source) override{
        std::vector<cv::VideoCapture> caps;

        // 处理输入源
        if (source.find(',') != std::string::npos) {
            // 如果输入是逗号分隔的列表
            size_t pos = 0;
            size_t prev = 0;
            while ((pos = source.find(',', prev)) != std::string::npos) {
                cv::VideoCapture cap(std::stoi(source.substr(prev, pos - prev)));
                if (!cap.isOpened()) {
                    spdlog::error("Error: Cannot open camera {}", source.substr(prev, pos - prev));
                    return;
                }
                caps.push_back(cap);
                prev = pos + 1;
            }
        } else {
            // 如果输入是单个源
            cv::VideoCapture cap(source);
            if (!cap.isOpened()) {
                spdlog::error("Error: Cannot open camera {}", source);
                return;
            }
            caps.push_back(cap);
        }

        // 创建映射表
        Size size;
        Mat left_map1, left_map2, right_map1, right_map2, R1, R2, P1, P2, Q;
        create_mapping_table(caps, size, left_map1, left_map2, right_map1, right_map2, R1, R2, P1, P2, Q);

        // 提示信息
        bool two_usbs = caps.size() == 2;
        if (two_usbs) 
            spdlog::info("\033[33mYou are using two-usb stereo camera, remember to put the left camera index at front when passing the 'source' parameter.\033[0m");

        int width = size.width;
        while (true) {
            std::vector<cv::Mat> frames;
            for (auto& cap : caps) {
                cv::Mat frame;
                cap >> frame;
                frames.push_back(frame);
            }

            // 拼接左右图像
            cv::Mat frame;
            cv::hconcat(frames, frame);

            // 分割左右图像
            cv::Mat imgL = frame(cv::Rect(0, 0, width, frame.rows));
            cv::Mat imgR = frame(cv::Rect(width, 0, width, frame.rows));

            // 图像校正
            // ! 从现象上看,这里有问题,摁s保存校正后的图像,校正后的图像是全黑的
            cv::Mat imgL_rectified, imgR_rectified;
            cv::remap(imgL, imgL_rectified, left_map1, left_map2, cv::INTER_LINEAR);
            cv::remap(imgR, imgR_rectified, right_map1, right_map2, cv::INTER_LINEAR);

            // 计算视差
            cv::Mat disparity;
            stereo_->compute(imgL_rectified, imgR_rectified, disparity);

            // 动态调整参数
            if (dynamic_tune_) create_stereo_sgbm();

            // 渲染彩色深度图
            cv::Mat disp_normed, disp_color;
            cv::normalize(disparity, disp_normed, 0, 255, cv::NORM_MINMAX, CV_8U);
            cv::applyColorMap(disp_normed, disp_color, cv::COLORMAP_JET);

            // 计算三维坐标
            cv::Mat threeD;
            cv::reprojectImageTo3D(disparity, threeD, Q, true);
            threeD *= 16;  // 乘以16得到真实距离

            // 设置鼠标回调
            cv::setMouseCallback(disp_color_win_name_, onmouse_pick_points, &threeD);

            // 显示图像
            cv::imshow(left_frame_win_name_, imgL);
            cv::imshow(right_frame_win_name_, imgR);
            cv::imshow(disp_color_win_name_, disp_color);

            // 处理键盘输入
            char key = cv::waitKey(1);
            if (key == 's') save_rectified(imgL, imgR, imgL_rectified, imgR_rectified);
            if (key == 'q') break;
        }

        // 释放资源
        for (auto& cap : caps) cap.release();

        cv::destroyAllWindows();
    };

private:
    bool is_bar_set_{false};
    bool use_video_;
    int minDisparity_;
    int numDisparities_;
    int blockSize_;
    int P1_;
    int P2_;
    int disp12MaxDiff_;
    int preFilterCap_;
    int uniquenessRatio_;
    int speckleWindowSize_;
    int speckleRange_;
    int mode_; // cv::StereoSGBM::MODE_HH
    bool dynamic_tune_;
    string left_frame_win_name_{"left frame"};  // 左相机的原图
    string right_frame_win_name_{"right frame"};  // 右相机的原图
    string disp_color_win_name_{"disparity color"};  // 渲染出的深度图的彩色图
    cv::Mat left_camera_matrix_;
    cv::Mat left_distortion_;
    cv::Mat right_camera_matrix_;
    cv::Mat right_distortion_;
    cv::Mat R_;
    cv::Mat T_;
    shared_ptr<cv::StereoSGBM> stereo_;
};


std::shared_ptr<SGBMEstimatorInterface> createSGBMEstimator(
    bool use_video,
    int minDisparity,
    int numDisparities,
    int blockSize,
    int P1,
    int P2,
    int disp12MaxDiff,
    int preFilterCap,
    int uniquenessRatio,
    int speckleWindowSize,
    int speckleRange,
    int mode, 
    bool dynamic_tune,
    std::string config_path
){
    return std::make_shared<SGBMEstimator>(
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
}
