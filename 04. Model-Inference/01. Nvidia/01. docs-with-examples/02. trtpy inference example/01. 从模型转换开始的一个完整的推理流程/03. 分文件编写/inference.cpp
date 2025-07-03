#include <iostream>   
#include <stdio.h>
#include <fstream>
#include <vector>
#include <math.h>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <onnx-tensorrt/NvOnnxParser.h>
#include <opencv2/opencv.hpp>
#include <inference.hpp>


using namespace std;

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}


// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

vector<unsigned char> load_file(const string& file){
    ifstream in(file, ios::in | ios::binary);
    if (!in.is_open())
        return {};

    in.seekg(0, ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0){
        in.seekg(0, ios::beg);
        data.resize(length);

        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}


ESRGAN::ESRGAN(const std::string& modelPath) {
    // Initialize TensorRT and load the model from 'modelPath'
    scale_factor = 4;
    input_batch = 1;
    engine_data = load_file(modelPath);
    runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (engine == nullptr) {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
}

cv::Mat ESRGAN::infer(const std::string& imagePath) {
    auto image = cv::imread(imagePath);
    if(image.empty()){
        return cv::Mat();
    }
    cudaStream_t stream = nullptr;
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext());
    int input_channel = image.channels();
    int input_height = image.rows;
    int input_width = image.cols;
    int input_numel = input_batch * input_channel * input_height * input_width;
    float* input_data_host = nullptr;
    float* input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    int image_area = image.cols * image.rows;
    unsigned char* pimage = image.data;
    float* phost_b = input_data_host + image_area * 0;
    float* phost_g = input_data_host + image_area * 1;
    float* phost_r = input_data_host + image_area * 2;
    for(int i = 0; i < image_area; ++i, pimage += 3){
        // 注意这里的顺序rgb调换了
        *phost_r++ = pimage[0] / 255.0f ;
        *phost_g++ = pimage[1] / 255.0f;
        *phost_b++ = pimage[2] / 255.0f;
    }

    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    int output_batch = input_batch;
    int output_channel = input_channel;
    int output_height = input_height * scale_factor;
    int output_width = input_width * scale_factor;
    int output_numel = output_height * output_width * output_channel * output_batch;
    float* output_data_device = nullptr;
    float output_data_host[output_numel * sizeof(float)];         
    checkRuntime(cudaMalloc(&output_data_device, output_numel * sizeof(float)));

    // 明确当前推理时，使用的数据输入大小
    auto input_dims = execution_context->getBindingDimensions(0);
    // input_dims.d[0] = input_batch;

    // for(int i=0;i<4;++i){
    //     printf("第%d个维度:%d\n", i, input_dims.d[i]);
    // }

    // 设置当前推理时，input大小
    execution_context->setBindingDimensions(0, nvinfer1::Dims4(input_batch, input_channel, input_height, input_width));
    float* bindings[] = {input_data_device, output_data_device};

    // printf("开始推理！\n");
    bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
    // printf("推理成功！\n");
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, output_numel * sizeof(float), cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    uchar output_uchar[output_numel];
    for (int i = 0; i < output_numel; ++i){
        output_uchar[i] = static_cast<uchar>(output_data_host[i]);
    }
    cv::Mat output_image(output_height, output_width, CV_8UC3, output_uchar);

    // // 计算执行时间
    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    // printf("代码执行时间: %lld ms\n", duration.count());
    
    
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
    checkRuntime(cudaStreamDestroy(stream));
    return output_image;
}