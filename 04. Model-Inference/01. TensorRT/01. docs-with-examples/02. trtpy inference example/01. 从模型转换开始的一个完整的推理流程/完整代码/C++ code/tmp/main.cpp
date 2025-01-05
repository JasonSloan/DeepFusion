#include <stdio.h>
#include <chrono>
#include <thread>
#include "opencv2/opencv.hpp"
#include "inference.hpp"

int main(){
    // make run 调用使用这段
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
    for (int i = 0; i < 100; i++){
        Result result = infer->forward(input_image, "1");    // infer只能调用forward，也只能看见forward
    }

    // if(result.code != 0){
    //     printf("infer failed.\n");
    //     return -1;
    // }
    // cv::Mat recovered_image(output_height, output_width, CV_8UC3, result.output_vector.data());
    // cv::imwrite("output_image.png", recovered_image);

    printf("Done!\n");
    return 0;
}