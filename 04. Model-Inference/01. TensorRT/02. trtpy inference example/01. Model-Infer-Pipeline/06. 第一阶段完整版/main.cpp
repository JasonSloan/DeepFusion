#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "inference.hpp"

int main(){
    // // 动态库调用使用这段，需要指定../workspace/
    // std::string modelPath = "../workspace/engine.trtmodel";
    // auto infer = create_infer(modelPath); 
    // if (infer == nullptr){           // 调用者只需要判断指针是否为空即可
    //     printf("load model failed.\n");
    //     return -1;
    // }
    // std::string imagePath = "../workspace/bytes_image.bin";
    // int height = 50;
    // int width = 80;    
    // std::vector<uint8_t> output_image_bytes;
    // output_image_bytes = infer->forward(imagePath, width, height); // infer只能调用forward，也只能看见forward

    // make run 调用使用这段
    std::string modelPath = "engine.trtmodel";
    auto infer = create_infer(modelPath); 
    if (infer == nullptr){                      // 调用者只需要判断指针是否为空即可
        printf("load model failed.\n");
        return -1;
    }
    std::string imagePath = "OST_009_croped.jpg";
    cv::Mat input_image = cv::imread(imagePath, cv::IMREAD_COLOR);
    std::vector<uint8_t> input_image_bytes;
    cv::imencode(".jpg", input_image, input_image_bytes);
    std::vector<uint8_t> output_image_bytes;
    output_image_bytes = infer->forward(input_image_bytes);    // infer只能调用forward，也只能看见forward
    cv::Mat recovered_image = cv::imdecode(output_image_bytes, cv::IMREAD_COLOR);
    cv::imwrite("output_image.jpg", recovered_image);
    return 0;
}