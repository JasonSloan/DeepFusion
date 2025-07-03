#include <stdio.h>
#include <chrono>
#include <thread>
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
    // todo: 应该全部转成fp16的，为什么图像大了推理速度会变得很慢，等在3090上对比一下wangxinyu的tensorrtx
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
    for(int i = 0; i <= 100000; ++i){
        output_image_bytes = infer->forward(input_image_bytes);    // infer只能调用forward，也只能看见forward
        // std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if(i == 100000){
            cv::Mat recovered_image = cv::imdecode(output_image_bytes, cv::IMREAD_COLOR);
            cv::imwrite("output_image.jpg", recovered_image);
        }
    }
    printf("Done!\n");
    // output_image_bytes = infer->forward(input_image_bytes);    // infer只能调用forward，也只能看见forward
    return 0;
}