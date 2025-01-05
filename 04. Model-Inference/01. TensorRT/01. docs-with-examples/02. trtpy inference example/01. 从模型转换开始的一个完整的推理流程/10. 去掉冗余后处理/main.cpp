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
    // std::string imagePath = "../workspace/OST_009_croped.jpg";
    // cv::Mat input_image = cv::imread(imagePath, cv::IMREAD_COLOR);
    // std::vector<uint8_t> input_image_bytes;
    // cv::imencode(".jpg", input_image, input_image_bytes);
    // std::vector<uint8_t> output_image_bytes;
    // output_image_bytes = infer->forward(input_image_bytes); 


    // // make run 调用使用这段
    // // todo: 应该全部转成fp16的，为什么图像大了推理速度会变得很慢，等在3090上对比一下wangxinyu的tensorrtx
    std::string modelPath = "engine.trtmodel";
    auto infer = create_infer(modelPath); 
    if (infer == nullptr){                      // 调用者只需要判断指针是否为空即可
        printf("load model failed.\n");
        return -1;
    }
    std::string imagePath = "OST_009_croped.jpg";
    cv::Mat input_image = cv::imread(imagePath, cv::IMREAD_COLOR);
    int input_height = input_image.rows;
    int input_width = input_image.cols;
    int output_height = input_height*4;
    int output_width = input_width*4;
    std::vector<uint8_t> input_image_vector;
    cv::imencode(".jpg", input_image, input_image_vector);
    std::vector<uint8_t> output_image_vector;
    for(int i = 0; i <= 10; ++i){
        // auto pre_start_time = std::chrono::high_resolution_clock::now();
        output_image_vector = infer->forward(input_image_vector);    // infer只能调用forward，也只能看见forward
        // auto pre_end_time = std::chrono::high_resolution_clock::now();
        // auto pre_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pre_end_time - pre_start_time);
        // printf("Inference time consuming: %ld ms\n", pre_duration.count());
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if(!output_image_vector.empty() && i == 10){
            cv::Mat recovered_image(output_height, output_width, CV_8UC3, output_image_vector.data());
            cv::imwrite("output_image.png", recovered_image);
        }
    }
    printf("Done!\n");
    // // output_image_bytes = infer->forward(input_image_bytes);    // infer只能调用forward，也只能看见forward
    return 0;
}