// tensorRT include
// 编译用的头文件
#include <NvInfer.h>

// onnx解析器的头文件
#include <onnx-tensorrt/NvOnnxParser.h>

// 推理用的运行时头文件
#include <NvInferRuntime.h>

// cuda include
#include <cuda_runtime.h>

// system include
#include <stdio.h>
#include <math.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <functional>
#include <unistd.h>
#include <chrono>

#include <opencv2/opencv.hpp>

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

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger : public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kINFO){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
} logger;

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

bool exists(const string& path){

#ifdef _WIN32
    return ::PathFileExistsA(path.c_str());
#else
    return access(path.c_str(), R_OK) == 0;
#endif
}


///////////////////////////////////////////////////////////////////////////////////////////////////////

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


bool build_model(){

    if(exists("engine.trtmodel")){
        printf("engine.trtmodel has exists.\n");
        return true;
    }

    TRTLogger logger;

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    if (builder->platformHasFastFp16()){
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
        printf("Model Precision has been set to FP16.\n");
    }

    // createNetworkV2(1)表示采用显性batch size，新版tensorRT(>=7.0)时，不建议采用0非显性batch size
    // 因此贯穿以后，请都采用createNetworkV2(1)而非createNetworkV2(0)或者createNetwork
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if(!parser->parseFromFile("new.onnx", 1)){
        printf("Failed to parse new.onnx\n");

        // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
        return false;
    }
    
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 如果模型有多个输入，则必须多个profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    
    // 配置动态宽高
    // input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, 3, 64, 64));         // todo: 最小尺寸是多少
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, 3, 512, 512));       // todo: 最常用尺寸是多少
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(1, 3, 720, 1080));     // todo: 最大尺寸是多少

    config->addOptimizationProfile(profile);

    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if(engine == nullptr){
        printf("Build engine failed.\n");
        return false;
    }

    // 将模型序列化，并储存为文件
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen("engine.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    // 卸载顺序按照构建顺序倒序
    printf("Build Done.\n");
    return true;
}


// void inference(){
    
//     TRTLogger logger;
//     auto engine_data = load_file("engine.trtmodel");
//     auto runtime   = make_nvshared(nvinfer1::createInferRuntime(logger));
//     auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
//     if(engine == nullptr){
//         printf("Deserialize cuda engine failed.\n");
//         runtime->destroy();
//         return;
//     }

//     cudaStream_t stream = nullptr;
//     checkRuntime(cudaStreamCreate(&stream));
//     auto execution_context = make_nvshared(engine->createExecutionContext());

//     auto image = cv::imread("OST_009_croped.png");

//     int scale_factor = 4;
//     int input_batch = 1;
//     int input_channel = image.channels();
//     int input_height = image.rows;
//     int input_width = image.cols;
//     int input_numel = input_batch * input_channel * input_height * input_width;
//     float* input_data_host = nullptr;
//     float* input_data_device = nullptr;
//     checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
//     checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));
    
//     // // 计算执行时间的开始时间
//     // auto start_time = std::chrono::high_resolution_clock::now();
//     // 此段代码实现了HWC->CHW BGR->RGB /255的转换
//     int image_area = image.cols * image.rows;
//     unsigned char* pimage = image.data;
//     float* phost_b = input_data_host + image_area * 0;
//     float* phost_g = input_data_host + image_area * 1;
//     float* phost_r = input_data_host + image_area * 2;
//     for(int i = 0; i < image_area; ++i, pimage += 3){
//         // 注意这里的顺序rgb调换了
//         *phost_r++ = pimage[0] / 255.0f ;
//         *phost_g++ = pimage[1] / 255.0f;
//         *phost_b++ = pimage[2] / 255.0f;
//     }


//     checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));
    
//     // 假如是超分到16倍
//     int output_batch = input_batch;
//     int output_channel = input_channel;
//     int output_height = input_height * scale_factor;
//     int output_width = input_width * scale_factor;
//     int output_numel = output_height * output_width * output_channel * output_batch;
//     float* output_data_device = nullptr;
//     float output_data_host[output_numel * sizeof(float)];         
//     checkRuntime(cudaMalloc(&output_data_device, output_numel * sizeof(float)));
    
    
//     // 明确当前推理时，使用的数据输入大小
//     auto input_dims = execution_context->getBindingDimensions(0);
//     // input_dims.d[0] = input_batch;

//     // for(int i=0;i<4;++i){
//     //     printf("第%d个维度:%d\n", i, input_dims.d[i]);
//     // }

//     // 设置当前推理时，input大小
//     execution_context->setBindingDimensions(0, nvinfer1::Dims4(input_batch, input_channel, input_height, input_width));
//     float* bindings[] = {input_data_device, output_data_device};

//     // printf("开始推理！\n");
//     bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);
//     // printf("推理成功！\n");
//     checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, output_numel * sizeof(float), cudaMemcpyDeviceToHost, stream));
//     checkRuntime(cudaStreamSynchronize(stream));

//     uchar output_uchar[output_numel];
//     for (int i = 0; i < output_numel; ++i){
//         output_uchar[i] = static_cast<uchar>(output_data_host[i]);
//     }
//     cv::Mat output_image(output_height, output_width, CV_8UC3, output_uchar);

//     // // 计算执行时间
//     // auto end_time = std::chrono::high_resolution_clock::now();
//     // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
//     // printf("代码执行时间: %lld ms\n", duration.count());
    
//     cv::imwrite("output_image.png", output_image);
    
//     checkRuntime(cudaFreeHost(input_data_host));
//     checkRuntime(cudaFree(input_data_device));
//     checkRuntime(cudaFree(output_data_device));
//     checkRuntime(cudaStreamDestroy(stream));
// }

int main(){
    if(!build_model()){
        return -1;
    }
    // inference();
    return 0;
}