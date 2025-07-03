#include <inference.hpp>
#include <opencv2/opencv.hpp>

int main(){
    // // 第三次封装RAII接口封装前的调用代码
    std::string modelPath = "engine.trtmodel";
    std::string imagePath = "OST_009_croped.png";
    cv::Mat outputImage;
    ESRGAN esrgan(modelPath);
    outputImage = esrgan.infer(imagePath);
    return 0;
}