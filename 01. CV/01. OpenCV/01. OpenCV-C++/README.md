## 编译安装OpenCV

### 方式一：

```bash
	如果是ubuntu系统，可以直接使用本仓库中release的opencv4.2, 支持对视频的处理。将opencv4.2拷贝到目标机器后, 使用ldd查看lib中的库是否都可正常链接到
```
### 方式二：

**编译普通opencv**

        安装cmake:  apt-get install cmake

        安装依赖: apt-get install build-essential libgtk2.0-dev libjpeg.dev libtiff5.dev libswscale-dev
        使用ffmpeg做后端: sudo apt-get install ffmpeg
        使用gstreamer做后端: sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-libav gstreamer1.0-plugins-bad gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly
    
        下载OpenCV4.6.0: 链接：https://pan.baidu.com/s/1OuFZNub5VGMsoF8oOSNxjQ  提取码：5tgz
    
        解压进入opencv4.6.0文件夹,创建build文件夹, 进入build文件夹, 执行cmake, 执行make, 执行make install
    
        如果想将编译后头文件和库文件安装到指定目录, 在cmake编译时需要指定-DCMAKE_INSTALL_PREFIX=/path/to/custom/folder, 然后编译结束后make install就可以安装在指定目录中了
        使用ffmpeg做后端: -D WITH_FFMPEG=ON
        使用gstreamer做后端需指定参数-D WITH_GSTREAMER=ON
    
        下载IPPICV过慢问题解决方法: 手动下载压缩包: https://gitee.com/c1h2/ippicv_2020_lnx_intel64_20191018_general/blob/master/ippicv_2020_lnx_intel64_20191018_general.tgz
        修改opencv-x.x/3rdparty/ippicv/ippicv.cmake文件, 修改方式如下:

![](assets/1.jpg)

```bash
如果出现以下错误:By not providing "FindOpenCV.cmake" in CMAKE_MODULE_PATH this project has
asked CMake to find a package configuration file provided by "OpenCV", but
CMake did not find one.
 则需要在CMakeLists.txt中增加一行"set(OpenCV_DIR /path/to/opencv-4.6.0)", 指定opencv源码的根目录

    在/etc/ld.so.conf.d下创建文件OpenCV.conf, 写入"/usr/local/lib", 保存退出执行sudo ldconfig, 使opencv的动态库可以被系统链接到

    编写main.cpp, 验证opencv是否好用
```

### 方式三：

**编译opencv-cuda**

注意安装opencv-cuda之前需安装cuda环境，参考我的另一个[仓库](https://github.com/JasonSloan/yolo-tensorrt)

```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git

cd opencv
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/your/installation/dir \
      -D OPENCV_EXTRA_MODULES_PATH=/path/to/opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D BUILD_opencv_cudev=ON ..
make -j$(nproc)
sudo make install
```

一些简单的opencv-cuda用法：

```C++
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
```



