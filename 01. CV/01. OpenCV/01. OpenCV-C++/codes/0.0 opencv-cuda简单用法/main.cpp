#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <iostream>

void processImageCUDA(const cv::Mat& inputImage, cv::cuda::GpuMat& outputTensor) {
    // Upload the image to the GPU
    cv::cuda::GpuMat gpuImage;
    gpuImage.upload(inputImage);

    // Step 1: Resize the image
    cv::cuda::GpuMat resizedImage;
    cv::cuda::resize(gpuImage, resizedImage, cv::Size(224, 224)); // Example target size

    // Step 2: Add a border
    cv::cuda::GpuMat borderedImage;
    int top = 10, bottom = 10, left = 10, right = 10; // Example border size
    cv::cuda::copyMakeBorder(resizedImage, borderedImage, top, bottom, left, right, 
                             cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0)); // Border color black

    // Step 3: Normalize pixel values
    cv::cuda::GpuMat normalizedImage;
    borderedImage.convertTo(normalizedImage, CV_32F, 1.0 / 255); // Normalize to range [0, 1]

    // Step 4: Convert HWC to CHW
    std::vector<cv::cuda::GpuMat> chwChannels(3); // Assuming 3 channels (e.g., RGB)
    cv::cuda::split(normalizedImage, chwChannels);

    // Concatenate the channels into a single CHW tensor
    cv::cuda::GpuMat chwImage;
    cv::cuda::vconcat(chwChannels, chwImage);

    // Output the result
    outputTensor = chwImage;
}

// 需要连接的库opencv_cudaarithm opencv_cudaimgproc opencv_cudawarping
int main() {
    // Load an image
    cv::Mat inputImage = cv::imread("example.jpg"); // Replace with your image path
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image.\n";
        return -1;
    }

    cv::cuda::GpuMat outputTensor;
    processImageCUDA(inputImage, outputTensor);

    // Download the processed tensor for verification/debugging (optional)
    cv::Mat hostOutput;
    outputTensor.download(hostOutput);
    std::cout << "Processed image size: " << hostOutput.size << std::endl;

    return 0;
}