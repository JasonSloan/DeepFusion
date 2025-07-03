#include <stdio.h>
#include "opencv2/opencv.hpp"
#include "inference.hpp"

int main(){
    std::string modelPath = "engine.trtmodel";
    auto infer = create_infer(modelPath); 
    if (infer == nullptr){           // 调用者只需要判断指针是否为空即可
        printf("load model failed.\n");
        return -1;
    }
    std::string imagePath = "OST_009_croped.jpg"; 
    cv::Mat output_image;
    output_image = infer->forward(imagePath); // infer只能调用forward，也只能看见forward
    cv::imwrite("output_image.jpg", output_image);
    return 0;
}