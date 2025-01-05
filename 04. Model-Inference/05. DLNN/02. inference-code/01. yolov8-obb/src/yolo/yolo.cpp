#include <stdio.h>
#include <fstream> 
#include <map>
#include <string>
#include <vector>
#include <math.h>
#include <functional>                                   // std::ref()需要用这个库
#include <unistd.h>
#include <thread>                                       // 线程
#include <queue>                                        // 队列
#include <mutex>                                        // 线程锁
#include <chrono>                                       // 时间库
#include <memory>                                       // 智能指针
#include <future>                                       // future和promise都在这个库里，实现线程间数据传输
#include <condition_variable>                           // 线程通信库
#include <filesystem> 
#include <unistd.h>
#include <dirent.h>                                     // opendir和readdir包含在这里
#include <sys/stat.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <dlnne/dlnne.h>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "spdlog/sinks/basic_file_sink.h"               // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "yolo/yolo.h"
#include "yolo/model-utils.h"

using namespace std;
using namespace cv;
using namespace dl::nne;
using time_point = chrono::high_resolution_clock;
template <typename Rep, typename Period>
float micros_cast(const std::chrono::duration<Rep, Period>& d) {return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(d).count()) / 1000.;};

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

template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){p->Destroy();});
}

struct Job{
    shared_ptr<promise<vector<Result>>> pro;        //为了实现线程间数据的传输，需要定义一个promise，由智能指针托管, pro负责将消费者消费完的数据传递给生产者
    vector<Mat> input_images;                       // 输入图像, 多batch 
    bool inferLog{false};                                  // 是否打印日志
};

void preprocess_kernel_invoker(
    int src_width, int src_height, int src_line_size,
    int dst_width, int dst_height, int dst_line_size,
    uint8_t* src_device, uint8_t* intermediate_device, 
    float* dst_device, uint8_t fill_value, int dst_img_area, size_t offset
);

void postprocess_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float conf_thre_, 
    float nms_thre_, float* invert_affine_matrix, float* parray, int max_objects, 
    int NUM_BOX_ELEMENT
);


class InferImpl : public InferInterface{                                        
public:
    virtual ~InferImpl(){
        stop();
        spdlog::warn("Destruct instance done!");
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();                                                   // 通知worker给break掉        
        }
        if(worker_thread_.joinable())                                           // 子线程加入     
            worker_thread_.join();
    }

    bool startup(
        const string& file, 
        bool modelLog=false,
        bool multi_label=true
    ){
        
        multi_label_ = multi_label;
        model_path_ = file;
        modelLog_ = modelLog;
        running_ = true;                                                        // 启动后，运行状态设置为true
        string modelName = getFileName(model_path_, true);
        vector<string> splits = splitString(modelName, "-");
        if (splits[0] == "v8")
            is_v5_ = false;
        promise<bool> pro;
        CUdevice _device = 0;
        shared_ptr<cudaDeviceProp> properties(new cudaDeviceProp);
        cudaGetDeviceProperties(properties.get(), _device); 
        auto device_name = properties->name; auto global_memory = properties->totalGlobalMem / (1 << 30);
        if (modelLog_) spdlog::info("Device {} name is {}, total memory {}GB", _device, device_name, global_memory);
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }

    void worker(promise<bool>& pro){
        // init log file: auto increase from 0.txt to 1.txt
        string log_name = get_logfile_name(logs_dir);
        string log_path = logs_dir + "/" + log_name + ".txt";
        try {
            logger_ = spdlog::basic_logger_mt(log_name, log_path);
            logger_->set_level(spdlog::level::warn);
        } catch (const spdlog::spdlog_ex &ex) {
            spdlog::error("Log initialization failed: {}", ex.what());
            pro.set_value(false);
            return;
        }
        // 加载模型
        auto deserializedData = load_file(model_path_);
        try{
            engine_ = make_nvshared(Deserialize(deserializedData.data(), deserializedData.size()));
        }catch (const std::exception& e) {
            // failed
            running_ = false;
            logger_->error("Load model failed from path: {}!", model_path_);
            spdlog::error("Load model failed from path: {}!", model_path_);
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            return;
        }
        context_ = make_nvshared(engine_->CreateExecutionContext());
        checkRuntime(cudaStreamCreate(&stream_));
        max_batch_size_ = engine_->GetMaxBatchSize();
        if (modelLog_) spdlog::info("Supported Max batch size is {}", max_batch_size_);
        if(context_ == nullptr){
            running_ = false;
            logger_->error("Create context failed!");
            spdlog::error("Create context failed!");
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            return;
        }
        // load success
        pro.set_value(true);  
        if (modelLog_) spdlog::info("Model loaded successfully from {}", model_path_);
        vector<Job> fetched_jobs;
        while(running_){
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();});        // 一直等着，cv_.wait(lock, predicate):如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号
                if(!running_) break;                                            // 如果实例被析构了，那么就结束该线程
                Job job_one = std::move(jobs_.front());
                jobs_.pop();                                                    // 从jobs_任务队列中将当前要推理的job给pop出来 
                l.unlock();                                                     // 注意这里要解锁, 否则调用inference等inference执行完再解锁又变同步了
                inference(job_one);                                             // 调用inference执行推理
            }
        }
    }

    // forward函数是生产者, 异步返回, 在main.cpp中获取结果
    virtual std::shared_future<std::vector<Result>> forward(Input* inputs, int& n_images, float conf_thre, bool inferLog=false) override{  
        Job job;
        vector<string> unique_ids;                              
        for (int i = 0; i < n_images; ++i){
            int numel = inputs[i].height * inputs[i].width * 3;
            cv::Mat image_one(inputs[i].height, inputs[i].width, CV_8UC3);
            memcpy(image_one.data, inputs[i].data, numel);
            // string save_path = "images-received/" + inputs[i].unique_id + ".jpg";
            // cv::cvtColor(image_one, image_one, cv::COLOR_RGB2BGR);
            // cv::imwrite(save_path, image_one);
            job.input_images.push_back(image_one);
        }            

        conf_thre_ = conf_thre;
        job.pro.reset(new promise<vector<Result>>());
        job.inferLog = inferLog;

        shared_future<vector<Result>> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            unique_lock<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut;                                                             // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }

    void malloc_data(
        int curr_batch_size,
        float*& input_data_host,
        float*& input_data_device,
        float*& output_data_host,
        float*& output_data_device
    ){
        // Malloc input data
        // int nbindings = engine_->GetNbBindings();
        // for (int i = 0; i < nbindings; ++i){
        //     auto binding_name = engine_->GetBindingName(i);
        //     bool is_input = engine_->BindingIsInput(i);
        // }
        auto input_dims = engine_->GetBindingDimensions(0);     // 注意: 这里默认为1个输入, 可以使用上面注释的代码获得输入的名字和索引
        input_dims.d[0] = curr_batch_size;                     // attach current batch size to the input_dims
        int input_channels = input_dims.d[1];
        int input_height = input_dims.d[2];
        int input_width = input_dims.d[3];
        int input_numel = curr_batch_size * input_channels * input_height * input_width;  // todo: dynamic batch        
        checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
        checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

        // Malloc output data 
        auto output_dims = engine_->GetBindingDimensions(1);    // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        output_dims.d[0] = curr_batch_size;
        int output_batch = output_dims.d[0];
        int output_numbox = output_dims.d[1];
        int output_numprob = output_dims.d[2];
        int output_numel = curr_batch_size * output_numbox * output_numprob;
        checkRuntime(cudaMallocHost(&output_data_host, output_numel * sizeof(float)));
        checkRuntime(cudaMalloc(&output_data_device, output_numel * sizeof(float)));

        if (inferLog_) {
            spdlog::info("Model input shape: {} x {} x {} x {}", curr_batch_size, input_channels, input_height, input_width);
            spdlog::info("Model max output shape: {} x {} x {}", curr_batch_size, output_numbox, output_numprob);
        }
    }

    void free_data(
        float*& input_data_host,
        float*& input_data_device,
        float*& output_data_host,
        float*& output_data_device
    ){
        cudaFree(output_data_device);
        cudaFreeHost(output_data_host);
        cudaFree(input_data_device);
        cudaFreeHost(input_data_host);
    }
    
    void preprocess_cpu(
        float*& input_data_host,
        float*& input_data_device,
        vector<Mat>& batched_imgs, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors, 
        int& curr_batch_size
    ){
        auto input_dims = engine_->GetBindingDimensions(0);         // 注意: 这里默认为1个输入, 可以使用上面注释的代码获得输入的名字和索引
        input_dims.d[0] = curr_batch_size;                         // attach current batch size to the input_dims
        int input_channels = input_dims.d[1];
        int input_height = input_dims.d[2];
        int input_width = input_dims.d[3];
        int input_numel = curr_batch_size * input_channels * input_height * input_width;

        // Resize and pad
        for (int i = 0; i < batched_imgs.size(); ++i){
            Mat& img = batched_imgs[i];
            int img_height = img.rows;
            int img_width = img.cols;
            int img_channels = img.channels();

            float scale_factor = min(static_cast<float>(input_width) / static_cast<float>(img.cols),
                            static_cast<float>(input_height) / static_cast<float>(img.rows));
            int img_new_w_unpad = img.cols * scale_factor;
            int img_new_h_unpad = img.rows * scale_factor;
            int pad_wl = round((input_width - img_new_w_unpad - 0.01) / 2);		                   
            int pad_wr = round((input_width - img_new_w_unpad + 0.01) / 2);
            int pad_ht = round((input_height - img_new_h_unpad - 0.01) / 2);
            int pad_hb = round((input_height - img_new_h_unpad + 0.01) / 2);
            cv::resize(img, img, cv::Size(img_new_w_unpad, img_new_h_unpad));
            cv::copyMakeBorder(img, img, pad_ht, pad_hb, pad_wl, pad_wr, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            batched_scale_factors.push_back(scale_factor);
            vector<int> pad_w = {pad_wl, pad_wr};
            vector<int> pad_h = {pad_ht, pad_hb};
            batched_pad_w.push_back(pad_w);
            batched_pad_h.push_back(pad_h);
        }

        // HWC-->CHW & /255. & BGR-->RGB & transfer data to input_data_host
        float* i_input_data_host;
        size_t img_area = input_height * input_width;
        for (int i = 0; i < batched_imgs.size(); ++i){
            i_input_data_host = input_data_host + img_area * 3 * i;
            unsigned char* pimage = batched_imgs[i].data;
            float* phost_r = i_input_data_host + img_area * 0;
            float* phost_g = i_input_data_host + img_area * 1;
            float* phost_b = i_input_data_host + img_area * 2;
            for(int j = 0; j < img_area; ++j, pimage += 3){
                *phost_r++ = pimage[2] / 255.0f ;
                *phost_g++ = pimage[1] / 255.0f;
                *phost_b++ = pimage[0] / 255.0f;
            }
        }

        // Copy data from host to device
        checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream_));
    }

    void preprocess_gpu_invoker(
        int src_width, int src_height,
        int dst_width, int dst_height,
        uint8_t* src_host, float* dst_device, size_t offset
    ){
        int src_line_size = src_width * 3;
        int dst_line_size = dst_width * 3;
        int dst_img_area = dst_width * dst_height;
        size_t src_size = src_width * src_height * 3 * sizeof(uint8_t);
        size_t intermediate_size = dst_width * dst_height * 3 * sizeof(uint8_t);
        size_t dst_size = dst_width * dst_height * 3 * sizeof(float);
        
        uint8_t* src_device;
        uint8_t* intermediate_device;
        uint8_t fill_value = 114;
        checkRuntime(cudaMalloc(&src_device, src_size));
        checkRuntime(cudaMalloc(&intermediate_device, intermediate_size));
        checkRuntime(cudaMemcpy(src_device, src_host, src_size, cudaMemcpyHostToDevice));
    
        preprocess_kernel_invoker(
            src_width, src_height, src_line_size,
            dst_width, dst_height, dst_line_size,
            src_device, intermediate_device, 
            dst_device, fill_value, dst_img_area, offset
        );   

        checkRuntime(cudaPeekAtLastError());
        checkRuntime(cudaFree(intermediate_device));
        checkRuntime(cudaFree(src_device));
    }

    void preprocess_gpu(
        float*& input_data_host,
        float*& input_data_device,
        vector<Mat>& batched_imgs, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors, 
        int& curr_batch_size
    ){

        auto input_dims = engine_->GetBindingDimensions(0);         // 注意: 这里默认为1个输入, 可以使用上面注释的代码获得输入的名字和索引
        input_dims.d[0] = curr_batch_size;                          // attach current batch size to the input_dims
        int input_channels = input_dims.d[1];
        int input_height = input_dims.d[2];
        int input_width = input_dims.d[3];
        int input_numel = curr_batch_size * input_channels * input_height * input_width;    

        // Resize and pad and transpose and normalize
        for (int i = 0; i < batched_imgs.size(); ++i){
            Mat& img = batched_imgs[i];
            int img_height = img.rows;
            int img_width = img.cols;
            int img_channels = img.channels();

            float scale_factor = min(static_cast<float>(input_width) / static_cast<float>(img.cols),
                            static_cast<float>(input_height) / static_cast<float>(img.rows));
            int img_new_w_unpad = img.cols * scale_factor;
            int img_new_h_unpad = img.rows * scale_factor;
            int pad_wl = round((input_width - img_new_w_unpad - 0.01) / 2);		                   
            int pad_wr = round((input_width - img_new_w_unpad + 0.01) / 2);
            int pad_ht = round((input_height - img_new_h_unpad - 0.01) / 2);
            int pad_hb = round((input_height - img_new_h_unpad + 0.01) / 2);
            batched_scale_factors.push_back(scale_factor);
            vector<int> pad_w = {pad_wl, pad_wr};
            vector<int> pad_h = {pad_ht, pad_hb};
            batched_pad_w.push_back(pad_w);
            batched_pad_h.push_back(pad_h);

            size_t offset = i * input_height * input_width * input_channels;

            preprocess_gpu_invoker(                 // todo: dynamic batch 
                img_width, img_height,
                input_width, input_height,
                img.data, input_data_device, offset
            );    
        }
    }

    bool do_infer(int curr_batch_size, void** buffers){
        bool success = context_->Enqueue(curr_batch_size, buffers, stream_, nullptr);
        return success;
    }

    void clip_boxes(
        float& box_left, 
        float& box_right, 
        float& box_top, 
        float& box_bottom, 
        vector<int>& img_org_shape
    ){
        auto clip_value = [](float value, float min_value, float max_value) {
            return (value < min_value) ? min_value : (value > max_value) ? max_value : value;
        };
        int org_height = img_org_shape[0];
        int org_width = img_org_shape[1];
        box_left = clip_value(box_left, 0, org_width);
        box_right = clip_value(box_right, 0, org_width);
        box_top = clip_value(box_top, 0, org_height);
        box_bottom = clip_value(box_bottom, 0, org_height);
    }

    void convariance_matrix(float w, float h, float r, float& a, float& b, float& c){
        float a_val = w * w / 12.0f;
        float b_val = h * h / 12.0f;
        float cos_r = cosf(r); 
        float sin_r = sinf(r);

        a = a_val * cos_r * cos_r + b_val * sin_r * sin_r;
        b = a_val * sin_r * sin_r + b_val * cos_r * cos_r;
        c = (a_val - b_val) * sin_r * cos_r;
    }

    float box_probiou(
        float cx1, float cy1, float w1, float h1, float r1,
        float cx2, float cy2, float w2, float h2, float r2,
        float eps = 1e-7
    ){
        // Calculate the prob iou between oriented bounding boxes, https://arxiv.org/pdf/2106.06072v1.pdf.
        float a1, b1, c1, a2, b2, c2;
        convariance_matrix(w1, h1, r1, a1, b1, c1);
        convariance_matrix(w2, h2, r2, a2, b2, c2);

        float t1 = ((a1 + a2) * powf(cy1 - cy2, 2) + (b1 + b2) * powf(cx1 - cx2, 2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
        float t2 = ((c1 + c2) * (cx2 - cx1) * (cy1 - cy2)) / ((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2) + eps);
        float t3 = logf(((a1 + a2) * (b1 + b2) - powf(c1 + c2, 2)) / (4 * sqrtf(fmaxf(a1 * b1 - c1 * c1, 0.0f)) * sqrtf(fmaxf(a2 * b2 - c2 * c2, 0.0f)) + eps) + eps); 
        float bd = 0.25f * t1 + 0.5f * t2 + 0.5f * t3;
        bd = fmaxf(fminf(bd, 100.0f), eps);
        float hd = sqrtf(1.0f - expf(-bd) + eps);
        return 1 - hd;    
    }

    RBox make_rbox(
        float cx, float cy,
        float w,  float h, 
        float label, float score, float t
    ){
        RBox rbox;
        regularize_rbox(w, h, t);
        xywhr2xyxyxyxy(rbox, cx, cy, w, h, t);
        rbox.label = int(label);
        rbox.score = score;
        return rbox;
    }

    void regularize_rbox(
        float& w, float& h, float& t
    ){
        float w_ = w > h ? w : h;
        float h_ = w > h ? h : w;
        float t_ = w > h ? t : t + M_PI / 2.;
        t_ = std::fmod(t_, M_PI);
        w = w_;
        h = h_;
        t = t_;
    }

    void xywhr2xyxyxyxy(
        RBox& rbox, float cx, float cy,
        float w, float h, float t
    ){
        auto float2int = [] (float x) {return static_cast<int>(round(x));};
        float cos_value = std::cos(t);   // 弧度制
        float sin_value = std::sin(t);
        float vec11 = w / 2. * cos_value;
        float vec12 = w / 2. * sin_value;
        float vec21 = -h / 2. * sin_value;
        float vec22 = h / 2. * cos_value;
        rbox.x1 = float2int(cx + vec11 + vec21);
        rbox.y1 = float2int(cy + vec12 + vec22);
        rbox.x2 = float2int(cx + vec11 - vec21);
        rbox.y2 = float2int(cy + vec12 - vec22);
        rbox.x3 = float2int(cx - vec11 - vec21);
        rbox.y3 = float2int(cy - vec12 - vec22);
        rbox.x4 = float2int(cx - vec11 + vec21);
        rbox.y4 = float2int(cy - vec12 + vec22);
    }

    void postprocess_cpu(
        int curr_batch_size,
        float* output_data_host,
        float* output_data_device,
        int& output_shape_size,
        vector<Result>& results, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<vector<int>>& batched_imgs_org_shape
    ){
        for (int i_img = 0; i_img < curr_batch_size; i_img++){
            vector<vector<float>> bboxes;                                   
            decode_boxes_1output_v8(
                output_data_host, i_img, bboxes, 
                batched_pad_w, batched_pad_h, 
                batched_scale_factors, batched_imgs_org_shape[i_img]
            );
            if (inferLog_) spdlog::info("Decoded bboxes.size = {}", bboxes.size());

            // nms非极大抑制
            // 通过比较索引为5(confidence)的值来将bboxes所有的框排序
            std::sort(bboxes.begin(), bboxes.end(), [](vector<float> &a, vector<float> &b)
                    { return a[5] > b[5]; });
            std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags

            for (int i = 0; i < bboxes.size(); ++i){
                if (remove_flags[i])
                    continue;                                       // 如果已经被标记为需要移除，则continue

                auto &ibox = bboxes[i];                             // 获得第i个box
                RBox rbox = make_rbox(ibox[0], ibox[1], ibox[2], ibox[3], ibox[4], ibox[5], ibox[6]);
                results[i_img].rboxes.emplace_back(rbox);

                for (int j = i + 1; j < bboxes.size(); ++j){        // 遍历剩余框，与box_result中的框做iou
                    if (remove_flags[j])
                        continue;                                   // 如果已经被标记为需要移除，则continue

                    auto &jbox = bboxes[j];                         // 获得第j个box
                    if (ibox[4] == jbox[4]){ 
                        // class matched
                        float iou = box_probiou(
                            ibox[0], ibox[1], ibox[2], ibox[3], ibox[6],
                            jbox[0], jbox[1], jbox[2], jbox[3], jbox[6]
                        );
                        if (iou >= nms_thre_)       // iou值大于阈值，将该框标记为需要remove
                            remove_flags[j] = true;
                    }
                }
            }
            if (inferLog_) spdlog::info("box_result.size = {}", results[i_img].rboxes.size());
        }
    }

    void synchronize(int curr_batch_size, float*& output_data_host, float*& output_data_device){
        auto output_dims = engine_->GetBindingDimensions(1);        // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_numel = curr_batch_size * output_dims.d[1] * output_dims.d[2];
        checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));
    }

    void decode_boxes_1output_v8(
        float* output_data_host,
        int& i_img, 
        vector<vector<float>>& bboxes, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<int>& img_org_shape
    ){
        auto output_dims = engine_->GetBindingDimensions(1);        // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_numbox = output_dims.d[1];
        int output_numprob = output_dims.d[2];
        int num_classes = output_numprob - 5;
        size_t offset_per_image = output_numbox * output_numprob;
        multi_label_ = multi_label_ && (num_classes > 1);
        int label; float prob; float confidence;
        vector<int> labels; vector<float> confidences;
        // yolov8 output: xywh + classes + r
        for (int i = 0; i < output_numbox; ++i) {
            float *ptr = output_data_host + offset_per_image * i_img + i * output_numprob;  // 每次偏移output_numprob

            float *pclass = ptr + 4;                                                        // 获得类别开始的地址
            label = max_element(pclass, pclass + num_classes) - pclass;                     // 获得概率最大的类别
            prob = pclass[label];                                                           // 获得类别概率最大的概率值
            confidence = prob;                                                              // 计算后验概率
            if (confidence < conf_thre_)
                continue;
            if (multi_label_)
                while (confidence >= conf_thre_){
                    labels.push_back(label);
                    confidences.push_back(confidence);
                    *(pclass + label) = 0.;
                    label = max_element(pclass, pclass + num_classes) - pclass;
                    prob = pclass[label];
                    confidence = prob;
                }                   

            // xywhr
            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];
            float theta = ptr[4+num_classes];

            // the box cords on the origional image
            float image_base_cx = (cx - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                     
            float image_base_cy = (cy - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                        
            float image_base_w  = width / batched_scale_factors[i_img];                                        
            float image_base_h  = height / batched_scale_factors[i_img];                      
            // todo: clip_bboxes
            if (multi_label_){
                for (int j = 0; j < labels.size(); ++j)
                    bboxes.push_back({image_base_cx, image_base_cy, image_base_w, image_base_h, (float)labels[j], confidences[j], theta});
                labels.clear();
                confidences.clear();
            }else{
                bboxes.push_back({image_base_cx, image_base_cy, image_base_w, image_base_h, (float)label, confidence, theta});  // 放进bboxes中
            }
        }
    }

    void inference(Job& job){
        // todo: 如果硬解码的话, 可以尝试使用cudagraph对硬解码+前处理+推理建图
        // if (InferImpl::warmuped_)
        //     this_thread::sleep_for(chrono::seconds(100));

        Result dummy;
        int curr_batch_size = job.input_images.size();
        std::vector<Result> results(curr_batch_size, dummy);

        vector<Mat> batched_imgs;
        for (int i = 0; i < job.input_images.size(); ++i){
            batched_imgs.push_back(job.input_images[i]);
        }

        inferLog_ = job.inferLog;
        vector<vector<int>> batched_imgs_org_shape;
        for (int i = 0; i < curr_batch_size; ++i){
            int height = batched_imgs[i].rows;
            int width = batched_imgs[i].cols;
            vector<int> i_shape = {height, width};
            batched_imgs_org_shape.push_back(i_shape);
        }

        // initialize time
        auto start = time_point::now();
        auto stop = time_point::now();

        // prepare data                                     
        float* input_data_host = nullptr;
        float* input_data_device = nullptr;
        float* output_data_host = nullptr;
        float* output_data_device = nullptr;
        malloc_data(curr_batch_size, input_data_host, input_data_device, output_data_host, output_data_device);
        float* buffers[] = {input_data_device, output_data_device};       

        // preprocess 
        start = time_point::now();
        vector<float> batched_scale_factors;
        vector<vector<int>> batched_pad_w, batched_pad_h;
        preprocess_cpu(input_data_host, input_data_device, batched_imgs, batched_pad_w, batched_pad_h, batched_scale_factors, curr_batch_size);
        // preprocess_gpu(input_data_host, input_data_device, batched_imgs, batched_pad_w, batched_pad_h, batched_scale_factors, curr_batch_size);
        // cudaDeviceSynchronize();         // !注意: 如果要对使用gpu做前处理计时, 需要加这一句
        stop = time_point::now();
        InferImpl::records[0].push_back(micros_cast(stop - start));

        // infer
        start = time_point::now();
        bool success = do_infer(curr_batch_size, (void**)buffers);
        if (!success){
            for (int i = 0; i < curr_batch_size; ++i){
                spdlog::error("Model infer failed!");
            }
            job.pro->set_value(results);        // dummy results
            return;
        }
        synchronize(curr_batch_size, output_data_host, output_data_device);
        stop = time_point::now();
        InferImpl::records[1].push_back(micros_cast(stop - start));

        // postprocess
        start = time_point::now();
        auto output_dims = engine_->GetBindingDimensions(1);    // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_shape_size = output_dims.nbDims;       
        postprocess_cpu(curr_batch_size, output_data_host, output_data_device, output_shape_size, results, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape);
        stop = time_point::now();
        InferImpl::records[2].push_back(micros_cast(stop - start));

        free_data(input_data_host, input_data_device, output_data_host, output_data_device);
        // set results to future
        job.pro->set_value(results);
        
    }
    
    virtual vector<vector<float>> get_records() override{       // 计时相关, 可删
        return InferImpl::records;
    }

private:
    // 可调数据
    string model_path_;                                           // 模型路径
    int max_batch_size_;
    bool multi_label_;
    bool is_v5_{true};                                                // 是使用yolov5模型还是yolov8模型
    float conf_thre_{0.25};
    float nms_thre_{0.45};
    // 多线程有关
    atomic<bool> running_{false};                               // 如果InferImpl类析构，那么开启的子线程也要break
    thread worker_thread_;
    queue<Job> jobs_;                                           // 任务队列
    mutex lock_;                                                // 负责任务队列线程安全的锁
    condition_variable cv_;                                     // 线程通信函数
    // 模型初始化有关           
    cudaStream_t stream_;
    shared_ptr<Engine> engine_;
    shared_ptr<ExecutionContext> context_;
    //日志相关
    bool modelLog_;                                             // 模型加载时是否打印日志在控制台
    bool inferLog_;                                             // 模型推理时是否打印日志在控制台
    std::shared_ptr<spdlog::logger> logger_;                    // logger_负责日志文件, 记录一些错误日志
    string logs_dir{"infer-logs"};                              // 日志文件存放的文件夹   
    // 计时相关
    static vector<vector<float>> records;                       // 计时相关: 静态成员变量声明
};

// 在类体外初始化这几个静态变量                       
vector<vector<float>> InferImpl::records(3);                    // 计时相关: 静态成员变量定义, 长度为3

shared_ptr<InferInterface> create_infer(std::string &file, bool modelLog, bool multi_label){
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(file, modelLog, multi_label)){
        instance.reset();                                                     
    }
    return instance;
};
