#include <iostream>   
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>
#include <math.h>
#include <functional>
#include <unistd.h>
#include <thread>                                                           // 线程
#include <queue>                                                            // 队列
#include <mutex>                                                            // 线程锁
#include <chrono>                                                           // 时间库
#include <memory>                                                           // 智能指针
#include <future>                                                           // future和promise都在这个库里，实现线程间数据传输
#include <condition_variable>                                              // 线程通信库
#include "cuda_runtime.h"
#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "onnx-tensorrt/NvOnnxParser.h"
#include "opencv2/opencv.hpp"
#include "inference.hpp"

using namespace std;

struct Job{
    shared_ptr<promise<std::vector<uint8_t>>> pro;                         //为了实现线程间数据的传输，需要定义一个promise，由智能指针托管
    std::vector<uint8_t> input_image;
};

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
};

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
};

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
};

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
};

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
};

class InferImpl : public InferInterface{                       // 继承虚基类，从而实现load_model和destroy的隐藏
public:
    virtual ~InferImpl(){
        stop();
        printf("析构实例完成！\n");
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();                                  // 通知worker给break掉        
        }
        if(worker_thread_.joinable())                          // 子线程加入     
            worker_thread_.join();
    }

    bool startup(const string& file){
        modelPath = file;
        running_ = true; // 启动后，运行状态设置为true
        promise<bool> pro;
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));
        return pro.get_future().get();			
    }

    void worker(promise<bool>& pro){
        // 加载模型
        engine_data = load_file(modelPath);
        runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
        engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size()));
        if(engine == nullptr){
            // failed
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            runtime->destroy();
            printf("Load model failed: %s\n", modelPath.c_str());
            return;
        }
        // load success
        pro.set_value(true);                                                    // 将start_up中pro.get_future().get()的值设置为true
        printf("开始执行worker了\n");
        execution_context = make_nvshared(engine->createExecutionContext());
        vector<Job> fetched_jobs;
        while(running_){
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();});        // 一直等着，cv_.wait(lock, predicate):如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号
                if(!running_) break;                                            // 如果实例被析构了，那么就结束该线程
                fetched_jobs.emplace_back(std::move(jobs_.front()));            // 往里面fetched_jobs里塞东西  
                jobs_.pop();                                                    // 从jobs_任务队列中将当前要推理的job给pop出来 
                for(auto& job : fetched_jobs){                                  // 遍历要推理的job         // todo:如果需要这里可以改成多batch推理，但是要求宽高要一样
                    auto start_time = std::chrono::high_resolution_clock::now();
                    inference(job);                                             // 调用inference执行推理
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
                    printf("总耗时: %lld ms\n", duration.count());
                }
                fetched_jobs.clear();
            }
        printf("Infer worker done.\n");
        }
    }


    // 实际上forward函数是生产者
    virtual std::vector<uint8_t> forward(std::vector<uint8_t>& input_image_bytes) override{        
        Job job;
        job.pro.reset(new promise<std::vector<uint8_t>>());
        job.input_image = input_image_bytes;

        shared_future<std::vector<uint8_t>> fut = job.pro->get_future();        // get_future()并不会等待数据返回，get_future().get()才会
        {
            lock_guard<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut.get();                                                       // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }

    void inference(Job& job){
            auto ppre_start_time = std::chrono::high_resolution_clock::now();
            /*----------------------------前前处理计时开始---------------------------------*/
            checkRuntime(cudaStreamCreate(&stream)); 
            cv::Mat image = cv::imdecode(job.input_image, cv::IMREAD_COLOR);
            int input_channel = image.channels();   
            int input_height = image.rows;
            int input_width = image.cols;
            int input_numel = input_batch * input_channel * input_height * input_width;
            float* input_data_host = nullptr;
            float* input_data_device = nullptr;
            checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
            checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));
            /*----------------------------前前处理计时结束---------------------------------*/
            auto ppre_end_time = std::chrono::high_resolution_clock::now();
            auto ppre_duration = std::chrono::duration_cast<std::chrono::milliseconds>(ppre_end_time - ppre_start_time);
            printf("前前处理执行时间: %lld ms\n", ppre_duration.count());

            auto pre_start_time = std::chrono::high_resolution_clock::now();
            /*----------------------------前处理+推理计时开始---------------------------------*/
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

            bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);                    // todo : 加个推理成功与否的判断

            checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, output_numel * sizeof(float), cudaMemcpyDeviceToHost, stream));
            checkRuntime(cudaStreamSynchronize(stream));
            /*----------------------------前处理+推理计时结束---------------------------------*/
            auto pre_end_time = std::chrono::high_resolution_clock::now();
            auto pre_duration = std::chrono::duration_cast<std::chrono::milliseconds>(pre_end_time - pre_start_time);
            printf("推理执行时间: %lld ms\n", pre_duration.count());  
            // printf("推理成功！\n");
            
            auto post_start_time = std::chrono::high_resolution_clock::now();
            /*----------------------------后处理计时开始---------------------------------*/
            uchar output_uchar[output_numel];
            for (int i = 0; i < output_numel; ++i){
                output_uchar[i] = static_cast<uchar>(output_data_host[i]);
            }
            cv::Mat output_image(output_height, output_width, CV_8UC3, output_uchar);
            
            checkRuntime(cudaFreeHost(input_data_host));
            checkRuntime(cudaFree(input_data_device));
            checkRuntime(cudaFree(output_data_device));  
            checkRuntime(cudaStreamDestroy(stream));

            std::vector<uint8_t> output_image_bytes;
            cv::imencode(image_type, output_image, output_image_bytes);
            /*----------------------------后处理计时结束---------------------------------*/
            auto post_end_time = std::chrono::high_resolution_clock::now();
            auto post_duration = std::chrono::duration_cast<std::chrono::milliseconds>(post_end_time - post_start_time);
            printf("后处理执行时间: %lld ms\n", post_duration.count()); 
            job.pro->set_value(output_image_bytes);
    }

private:
    // 可调数据
    string modelPath;                                           // 模型路径
    int scale_factor{4};                                        // 图像超分倍数，需要与训练的模型匹配                // todo: 这样写是否行？直接再成员变量里初始化
    int input_batch{1};                                         // 每次推理batch_size=1
    string image_type{".jpg"};                                   // 返回的图像类型为jpg的
    // 多线程有关
    atomic<bool> running_{false};                               // 如果InferImpl类析构，那么开启的子线程也要break
    thread worker_thread_;
    queue<Job> jobs_;                                           // 任务队列
    mutex lock_;                                                // 负责任务队列线程安全的锁
    condition_variable cv_;                                     // 线程通信函数
    // 模型初始化有关           
    TRTLogger logger;                                           // 日志               
    std::vector<unsigned char> engine_data;                     // 存放engine.trtmodel的数据
    std::shared_ptr<nvinfer1::IRuntime> runtime;
    std::shared_ptr<nvinfer1::ICudaEngine> engine;              // 模型加载实际最终要得到的就是engine
    cudaStream_t stream{nullptr};
    std::shared_ptr<nvinfer1::IExecutionContext> execution_context;
    
};

shared_ptr<InferInterface> create_infer(const string &file){                    // 返回的指针向虚基类转化
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(file)){
        instance.reset();                                                       // 如果模型加载失败，instance要reset成空指针
    }
    return instance;
};