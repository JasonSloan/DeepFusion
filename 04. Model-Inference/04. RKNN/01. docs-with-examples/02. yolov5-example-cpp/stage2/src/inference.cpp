#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>   
#include <stdio.h>
#include <string>
#include <vector>
#include <math.h>
#include <cmath>
#include <algorithm>                                                        // max_element要用到
#include <functional>                                                       // std::ref()需要用这个库
#include <unistd.h>
#include <thread>                                                           // 线程
#include <queue>                                                            // 队列
#include <mutex>                                                            // 线程锁
#include <chrono>                                                           // 时间库
#include <memory>                                                           // 智能指针
#include <future>                                                           // future和promise都在这个库里，实现线程间数据传输
#include <condition_variable>                                               // 线程通信库

#include "common.h"
#include "rknn_api.h"
#include "image_utils.h"
#include "RgaUtils.h"
#include "im2d.hpp"
#include "opencv2/opencv.hpp"
#include "inference.hpp"


using namespace std;
using namespace cv;

const int anchors[3][6] = {{10, 13, 16, 30, 33, 23},
                           {30, 61, 62, 45, 59, 119},
                           {116, 90, 156, 198, 373, 326}};

struct Job{
    shared_ptr<promise<Result>> pro;        //为了实现线程间数据的传输，需要定义一个promise，由智能指针托管, pro负责将消费者消费完的数据传递给生产者
    Mat input_images;                       // 输入图像, 多batch                  
    bool log;                                       // 是否打印日志
};

typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

int read_data_from_file(const char *path, char **out_data){
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

static void dump_tensor_attr(rknn_tensor_attr *attr){
    printf("index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

inline static int32_t __clip(float val, float min, float max){
    float f = val <= min ? min : (val >= max ? max : val);
    return f;
}

static int8_t qnt_f32_to_affine(float f32, int32_t zp, float scale){
    float dst_val = (f32 / scale) + zp;
    int8_t res = (int8_t)__clip(dst_val, -128, 127);
    return res;
}

static float deqnt_affine_to_f32(int8_t qnt, int32_t zp, float scale) { return ((float)qnt - (float)zp) * scale; }

void release_model(rknn_app_context_t *app_ctx){
    if (app_ctx->rknn_ctx != 0){
        rknn_destroy(app_ctx->rknn_ctx);
        app_ctx->rknn_ctx = 0;
    }
    if (app_ctx->input_attrs != NULL){
        free(app_ctx->input_attrs);
        app_ctx->input_attrs = NULL;
    }
    if (app_ctx->output_attrs != NULL){
        free(app_ctx->output_attrs);
        app_ctx->output_attrs = NULL;
    }
}

class InferImpl : public InferInterface{            // 继承虚基类，从而实现load_model和destroy的隐藏
public:
    virtual ~InferImpl(){
        stop();
        printf("Destruct instance done!\n");
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();                                                   // 通知worker给break掉        
        }
        release_model(&rknn_app_ctx);                                           // 释放一些内存
        if(worker_thread_.joinable())                                           // 子线程加入     
            worker_thread_.join();
    }

    bool startup(string file){
        modelPath = file;
        running_ = true;                                                        // 启动后，运行状态设置为true
        promise<bool> pro;
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }

    int init_model(string model_path) {
        // init some variables
        int ret;
        char* model;
        int model_len = 0;
        rknn_context ctx = 0;
        rknn_app_context_t* app_ctx = &rknn_app_ctx;
        memset(app_ctx, 0, sizeof(rknn_app_context_t));

        // Load RKNN Model
        model_len = read_data_from_file(modelPath.c_str(), &model);
        if (model == NULL){
            printf("load_model fail!\n");
            return -1;
        }

        // Init rknn model
        ret = rknn_init(&ctx, model, model_len, 0, NULL);
        free(model);                            // 上一步已经将数据赋值给ctx, 所以这一步就将model中的数据释放  
        if (ret < 0){
            printf("rknn_init fail! ret=%d\n", ret);
            return -1;
        }

        // Get Model Input Output Number
        rknn_input_output_num io_num;
        ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
        if (ret != 0){
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

        // Get Model Input Info
        rknn_tensor_attr input_attrs[io_num.n_input];
        printf("input tensors:\n");
        memset(input_attrs, 0, sizeof(input_attrs));
        input_attrs[0].index = 0;                           // there is only one input, so here I just specify the input index to 0
        ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
        if (ret != 0) {
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        dump_tensor_attr(&(input_attrs[0]));

        // Get Model Output Info
        printf("output tensors:\n");
        rknn_tensor_attr output_attrs[io_num.n_output];
        memset(output_attrs, 0, sizeof(output_attrs));
        for (int i = 0; i < io_num.n_output; i++) {         // there might be 3 outputs for Int8 inference or 1 output for FP16 inference
            output_attrs[i].index = i;              
            ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
            if (ret != RKNN_SUCC){
                printf("rknn_query fail! ret=%d\n", ret);
                return -1;
            }
            dump_tensor_attr(&(output_attrs[i]));
        }

        // set to context
        app_ctx->rknn_ctx = ctx;

        // set quant or not 
        if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16){
            app_ctx->is_quant = true;
        }else{
            app_ctx->is_quant = false;
            output_numbox = output_attrs[0].dims[1];            // these two variables only use in 1 output with fp16 infer type
            output_numprob = output_attrs[0].dims[2];           // these two variables only use in 1 output with fp16 infer type
        }

        // init input_attrs and output_attrs and set them to app_ctx
        app_ctx->io_num = io_num;
        app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));       
        memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
        app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));     
        memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

        /*  my onnx model is NCHW format input, 
            but when i use input_attrs to retrieve the input format, 
            it tells me that it is NHWC format which is weird,
            but the code goes right, so i just leave it there  */
        if (input_attrs[0].fmt == RKNN_TENSOR_NCHW){
            printf("model is NCHW input fmt\n");
            app_ctx->model_channel = input_attrs[0].dims[1];
            app_ctx->model_height = input_attrs[0].dims[2];
            app_ctx->model_width = input_attrs[0].dims[3];
        }else{
            printf("model is NHWC input fmt\n");
            app_ctx->model_height = input_attrs[0].dims[1];
            app_ctx->model_width = input_attrs[0].dims[2];
            app_ctx->model_channel = input_attrs[0].dims[3];
        }
        printf("model input height=%d, width=%d, channel=%d\n",app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
        return 0;
    }

    void worker(promise<bool>& pro){
        int ret = init_model(modelPath);
        if (ret != 0) {
            pro.set_value(false);
        } else {
            pro.set_value(true);
        }
        vector<Job> fetched_jobs;
        while(running_){
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();});        // 一直等着，cv_.wait(lock, predicate):如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号
                if(!running_) break;                                            // 如果实例被析构了，那么就结束该线程
                fetched_jobs.emplace_back(std::move(jobs_.front()));            // 往里面fetched_jobs里塞东西 , std::move将对象的所有权从a转移到b // todo, 多batch这里需要结合batch-size
                jobs_.pop();                                                    // 从jobs_任务队列中将当前要推理的job给pop出来                                
                l.unlock(); 
				for(auto& job : fetched_jobs){                                  // 遍历要推理的job                          
                    inference(job);                                             // 调用inference执行推理
                }
                fetched_jobs.clear();
            }
        }
    }

    int cvtColor_rga(image_buffer_t& img_t, 
                     int src_format=RK_FORMAT_BGR_888, 
                     int dst_format=RK_FORMAT_RGB_888, 
                     bool log=false){
        auto release_handle = [] (rga_buffer_handle_t src_handle, rga_buffer_handle_t dst_handle) {
            if (src_handle) releasebuffer_handle(src_handle);
            if (dst_handle) releasebuffer_handle(dst_handle);
        };

        int width = img_t.width;
        int height = img_t.height;
        int src_buf_size, dst_buf_size;
        rga_buffer_t src_img, dst_img;
        rga_buffer_handle_t src_handle, dst_handle;

        memset(&src_img, 0, sizeof(src_img));
        memset(&dst_img, 0, sizeof(dst_img));

        src_buf_size = width * height * get_bpp_from_format(src_format);
        dst_buf_size = width * height * get_bpp_from_format(dst_format);

        shared_ptr<uchar> src_buf(new uchar[src_buf_size], default_delete<uchar[]>());
        shared_ptr<uchar> dst_buf(new uchar[src_buf_size], default_delete<uchar[]>());

        memcpy(src_buf.get(), img_t.virt_addr, src_buf_size);
        memset(dst_buf.get(), 0x80, dst_buf_size);

        src_handle = importbuffer_virtualaddr(src_buf.get(), src_buf_size);
        dst_handle = importbuffer_virtualaddr(dst_buf.get(), dst_buf_size);
        if (src_handle == 0 || dst_handle == 0) {
            printf("importbuffer failed!\n");
            release_handle(src_handle, dst_handle);
            return -1;
        }

        src_img = wrapbuffer_handle(src_handle, width, height, src_format);
        dst_img = wrapbuffer_handle(dst_handle, width, height, dst_format);

        int ret = imcheck(src_img, dst_img, {}, {});
        if (IM_STATUS_NOERROR != ret) {
            printf("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
            release_handle(src_handle, dst_handle);
            return -1;
        }

        ret = imcvtcolor(src_img, dst_img, src_format, dst_format);
        if (ret != IM_STATUS_SUCCESS) {
            printf("Convert COLOR failed, %s\n", imStrError((IM_STATUS)ret));
            release_handle(src_handle, dst_handle);
            return -1;
        }

        memcpy(img_t.virt_addr, dst_buf.get(), dst_buf_size);
        release_handle(src_handle, dst_handle);
        return 0;
    }

    int letterbox(Mat& im0, 
                  int& src_w, 
                  int& src_h, 
                  int& src_ch,  
                  image_buffer_t& src_img, 
                  image_buffer_t& dst_img, 
                  letterbox_t& letter_box,
                  rknn_app_context_t* app_ctx){
        // prepare src data
        memset(&src_img, 0, sizeof(image_buffer_t));
        image_format_t format = IMAGE_FORMAT_RGB888;
        int src_size = src_w * src_h * src_ch;
        src_img.width = src_w;
        src_img.height = src_h;
        src_img.format = format;
        src_img.virt_addr = im0.data;
        src_img.size = src_size;

        // prepare dst data
        memset(&dst_img, 0, sizeof(image_buffer_t));
        int dst_w = app_ctx->model_width;
        int dst_h = app_ctx->model_height;
        int dst_ch = app_ctx->model_channel;
        int dst_size = dst_w * dst_h * dst_ch;
        dst_img.width = dst_w;
        dst_img.height = dst_h;
        dst_img.format = format;
        shared_ptr<uchar> d_virt_addr(new uchar[dst_size], default_delete<uchar[]>());
        dst_img.virt_addr = d_virt_addr.get();
        dst_img.size = dst_size;
        
        // letterbox
        int bg_color = 114;
        memset(&letter_box, 0, sizeof(letterbox_t));
        int ret = convert_image_with_letterbox(&src_img, &dst_img, &letter_box, bg_color);      
        if (ret < 0) {
            printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        }
    }

    int preprocess_rga(Mat& im0, 
                       rknn_input* inputs,  
                       letterbox_t& letter_box, 
                       rknn_app_context_t* app_ctx,
                       bool log=false){
        int ret;
        int width = im0.cols;
        int height = im0.rows;
        int channels = im0.channels();
        
        // letterbox
        image_buffer_t src_img;
        image_buffer_t dst_img;
        ret = letterbox(im0, width, height, channels, src_img, dst_img, letter_box, app_ctx);
        if (ret != 0){
            return -1;
        }
        
        // convert color from BGR to RGB
        // ! sleep for about 1-5 ms, otherwise, the rga will go wrong, it is weird, but this is the only solution
        this_thread::sleep_for(std::chrono::milliseconds(2));
        ret = cvtColor_rga(dst_img, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888, log);
        if (ret != 0){
            return -1;
        }

        // set inputs
        memset(inputs, 0, sizeof(inputs));
        inputs[0].index = 0;
        inputs[0].type = RKNN_TENSOR_UINT8;       // cv::Mat data is uint8 type
        inputs[0].fmt = RKNN_TENSOR_NHWC;
        inputs[0].size = app_ctx->model_width * app_ctx->model_height * app_ctx->model_channel;
        inputs[0].buf = dst_img.virt_addr;
        ret = rknn_inputs_set(app_ctx->rknn_ctx, app_ctx->io_num.n_input, inputs);
        if (ret < 0) {
            printf("rknn_inputs_set fail! ret=%d\n", ret);
            return -1;
        }
    }

    void do_infer(rknn_app_context_t *app_ctx){
        int ret = rknn_run(app_ctx->rknn_ctx, nullptr);
        if (ret < 0) {
            printf("rknn_run fail! ret=%d\n", ret);
        }
    }
    
    void postprocess_fp(rknn_app_context_t *app_ctx, 
                    rknn_output* outputs,
                    letterbox_t* letter_box, 
                    float confidence_threshold, 
                    float nms_threshold,
                    Result& result, 
                    int& output_numbox,
                    int& output_numprob,
                    const bool log){
        // set & get ouput
        outputs[0].index = 0;
        outputs[0].want_float = (!app_ctx->is_quant);
        int ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
        if (ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
        }  

        // buf : xywh + conf + cls_prob on 640 * 384 image
        // decode and filter boxes by conf_thre
        float* output_data_host = (float*)outputs[0].buf;                   // fetch index 0 because there is only one output   
        vector<vector<float>> bboxes;                                       // 初始化变量bboxes:[[x1, y1, x2, y2, conf, label], [x1, y1, x2, y2, conf, label]...]
        int num_classes = output_numprob - 5;
        for (int i = 0; i < output_numbox; ++i) {
            float *ptr = output_data_host + i * output_numprob;             // 每次偏移output_numprob
            float objness = ptr[4];                                         // 获得置信度
            if (objness < confidence_threshold)
                continue;

            float *pclass = ptr + 5;                                        // 获得类别开始的地址
            int label = max_element(pclass, pclass + num_classes) - pclass; // 获得概率最大的类别
            float prob = pclass[label];                                     // 获得类别概率最大的概率值
            float confidence = prob * objness;                              // 计算后验概率
            if (confidence < confidence_threshold)
                continue;

            // xywh
            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            // xyxy
            float left = cx - width * 0.5;
            float top = cy - height * 0.5;
            float right = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // the box cords on the origional image
            float image_base_left = (left - letter_box->x_pad) / letter_box->scale;                                                                     // x1
            float image_base_right = (right - letter_box->x_pad) / letter_box->scale;                                                                   // x2
            float image_base_top = (top - letter_box->y_pad) / letter_box->scale;                                                                       // y1
            float image_base_bottom = (bottom - letter_box->y_pad) / letter_box->scale;                                                                 // y2
            bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence}); // 放进bboxes中
        }
        if (log) printf("decoded bboxes.size = %d\n", bboxes.size());

        // nms非极大抑制
        // 通过比较索引为5(confidence)的值来将bboxes所有的框排序
        std::sort(bboxes.begin(), bboxes.end(), [](vector<float> &a, vector<float> &b)
                { return a[5] > b[5]; });
        std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags
        // 定义一个lambda的iou函数
        auto iou = [](const vector<float> &a, const vector<float> &b)
        {
            float cross_left = std::max(a[0], b[0]);
            float cross_top = std::max(a[1], b[1]);
            float cross_right = std::min(a[2], b[2]);
            float cross_bottom = std::min(a[3], b[3]);

            float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
            float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
            if (cross_area == 0 || union_area == 0)
                return 0.0f;
            return cross_area / union_area;
        };

        for (int i = 0; i < bboxes.size(); ++i){
            if (remove_flags[i])
                continue;                                   // 如果已经被标记为需要移除，则continue

            auto &ibox = bboxes[i];                         // 获得第i个box
            auto float2int = [] (float x) {return static_cast<int>(round(x));};
            vector<int> _ibox = {float2int(ibox[0]), float2int(ibox[1]), float2int(ibox[2]), float2int(ibox[3])};
            result.boxes.emplace_back(_ibox);               // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
            result.labels.emplace_back(int(ibox[4]));
            result.scores.emplace_back(ibox[5]);
            for (int j = i + 1; j < bboxes.size(); ++j){    // 遍历剩余框，与box_result中的框做iou
                if (remove_flags[j])
                    continue;                               // 如果已经被标记为需要移除，则continue

                auto &jbox = bboxes[j];                     // 获得第j个box
                if (ibox[4] == jbox[4]){ 
                    // class matched
                    if (iou(ibox, jbox) >= nms_threshold)   // iou值大于阈值，将该框标记为需要remove
                        remove_flags[j] = true;
                }
            }
        }
        if (log) printf("box_result.size = %d\n", result.boxes.size());
    }

    void postprocess_i8(rknn_app_context_t *app_ctx, 
                    rknn_output* outputs,
                    letterbox_t* letter_box, 
                    float confidence_threshold, 
                    float nms_threshold,
                    Result& result,
                    const bool log){
        // set & get ouput
        int n_output = app_ctx->io_num.n_output;
        for (int i = 0; i < n_output; i++){
            outputs[i].index = i;
            outputs[i].want_float = (!app_ctx->is_quant);
        }
        int ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
        if (ret < 0) {
            printf("rknn_outputs_get fail! ret=%d\n", ret);
        }

        // set and get some arrtributes
        int stride = 0;
        int grid_h = 0;
        int grid_w = 0;
        float zp = 0.;
        float scale = 0.;
        std::vector<vector<float>> bboxes;
        int model_in_w = app_ctx->model_width;
        int model_in_h = app_ctx->model_height;

        // decode boxes
        // iter every output
        for (int k = 0; k < n_output; ++k){
            int8_t* output = (int8_t *)outputs[k].buf;
            zp = app_ctx->output_attrs[k].zp;                                       // offset 
            scale = app_ctx->output_attrs[k].scale;                                 // scale
            int anchor_size = 3;                                                    // todo: make it can be changed
            int prob_box_size = app_ctx->output_attrs[k].dims[1] / anchor_size;     // prob_box_size: nc + 5
            int nc = prob_box_size - 5;
            grid_h = app_ctx->output_attrs[k].dims[2];
            grid_w = app_ctx->output_attrs[k].dims[3];
            stride = model_in_h / grid_h;
            int grid_len = grid_h * grid_w;
            int8_t thres_i8 = qnt_f32_to_affine(confidence_threshold, zp, scale);
            // iter every anchor
            for (int a = 0; a < anchor_size; a++){    
                // iter every h grid                        
                for (int i = 0; i < grid_h; i++){
                    // iter every w grid
                    for (int j = 0; j < grid_w; j++){
                        // 想象成一个魔方, j是w维度, i是h维度, a是深度
                        int8_t box_confidence = output[(prob_box_size * a + 4) * grid_len + i * grid_w + j];
                        if (box_confidence >= thres_i8){
                            int offset = (prob_box_size * a) * grid_len + i * grid_w + j;
                            int8_t *in_ptr = output + offset;
                            float box_x = (deqnt_affine_to_f32(*in_ptr, zp, scale)) * 2.0 - 0.5;
                            float box_y = (deqnt_affine_to_f32(in_ptr[grid_len], zp, scale)) * 2.0 - 0.5;
                            float box_w = (deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale)) * 2.0;
                            float box_h = (deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale)) * 2.0;
                            box_x = (box_x + j) * (float)stride;
                            box_y = (box_y + i) * (float)stride;
                            box_w = box_w * box_w * (float)anchors[k][a * 2];
                            box_h = box_h * box_h * (float)anchors[k][a * 2 + 1];
                            float box_x1 = box_x - (box_w / 2.0);
                            float box_y1 = box_y - (box_h / 2.0);
                            float box_x2 = box_x + (box_w / 2.0);
                            float box_y2 = box_y + (box_h / 2.0);

                            int8_t maxClassProbs = in_ptr[5 * grid_len];
                            int maxClassId = 0;
                            for (int t = 1; t < nc; ++t){
                                int8_t prob = in_ptr[(5 + t) * grid_len];
                                if (prob > maxClassProbs){
                                    maxClassId = t;
                                    maxClassProbs = prob;
                                }
                            }
                            if (maxClassProbs > thres_i8){
                                float confidence = deqnt_affine_to_f32(maxClassProbs, zp, scale) * deqnt_affine_to_f32(box_confidence, zp, scale);
                                float image_base_left = (box_x1 - letter_box->x_pad) / letter_box->scale;                                                                     // x1
                                float image_base_right = (box_x2 - letter_box->x_pad) / letter_box->scale;                                                                   // x2
                                float image_base_top = (box_y1 - letter_box->y_pad) / letter_box->scale;                                                                       // y1
                                float image_base_bottom = (box_y2 - letter_box->y_pad) / letter_box->scale;  
                                bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)maxClassId, confidence});
                            }
                        }
                    }
                }
            }
        }
        if (log) printf("decoded bboxes.size = %d\n", bboxes.size());

        // nms非极大抑制
        // 通过比较索引为5(confidence)的值来将bboxes所有的框排序
        std::sort(bboxes.begin(), bboxes.end(), [](vector<float> &a, vector<float> &b)
                { return a[5] > b[5]; });
        std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags
        // 定义一个lambda的iou函数
        auto iou = [](const vector<float> &a, const vector<float> &b)
        {
            float cross_left = std::max(a[0], b[0]);
            float cross_top = std::max(a[1], b[1]);
            float cross_right = std::min(a[2], b[2]);
            float cross_bottom = std::min(a[3], b[3]);

            float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
            float union_area = std::max(0.0f, a[2] - a[0]) * std::max(0.0f, a[3] - a[1]) + std::max(0.0f, b[2] - b[0]) * std::max(0.0f, b[3] - b[1]) - cross_area;
            if (cross_area == 0 || union_area == 0)
                return 0.0f;
            return cross_area / union_area;
        };

        for (int i = 0; i < bboxes.size(); ++i){
            if (remove_flags[i])
                continue;                                   // 如果已经被标记为需要移除，则continue

            auto &ibox = bboxes[i];                         // 获得第i个box
            auto float2int = [] (float x) {return static_cast<int>(round(x));};
            vector<int> _ibox = {float2int(ibox[0]), float2int(ibox[1]), float2int(ibox[2]), float2int(ibox[3])};
            result.boxes.emplace_back(_ibox);               // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
            result.labels.emplace_back(int(ibox[4]));
            result.scores.emplace_back(ibox[5]);
            for (int j = i + 1; j < bboxes.size(); ++j){    // 遍历剩余框，与box_result中的框做iou
                if (remove_flags[j])
                    continue;                               // 如果已经被标记为需要移除，则continue

                auto &jbox = bboxes[j];                     // 获得第j个box
                if (ibox[4] == jbox[4]){ 
                    // class matched
                    if (iou(ibox, jbox) >= nms_threshold)   // iou值大于阈值，将该框标记为需要remove
                        remove_flags[j] = true;
                }
            }
        }
        if (log) printf("box_result.size = %d\n", result.boxes.size());
    }

    // 实际上forward函数是生产者, 异步返回, 在main.cpp中获取结果
    virtual std::shared_future<Result> forward(cv::Mat input_image, bool log) override{  
    // virtual std::vector<Result> forward(std::vector<cv::Mat> input_images, 
	// 	                           bool log=false) override{  
        Job job;
        job.pro.reset(new promise<Result>());
        job.input_images = input_image;
        job.log = log;

        shared_future<Result> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            lock_guard<mutex> l(lock_);
            jobs_.emplace(std::move(job));                              // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                               // 通知worker线程开始工作了
        return fut;
        // return fut.get();                                            // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }

    void print_consuming_time(std::vector<std::chrono::microseconds>& durations){
        printf("Consuming time: %.2f ms preprocess, %.2f ms infer, %.2f ms postprocess\n", 
                durations[0].count() / 1000.0, durations[1].count() / 1000.0, durations[2].count() / 1000.0);
        durations.clear();
    }
    
    void inference(Job& job){
        // fetch data
        Mat img = job.input_images;
        bool log = job.log;

        // init some variables
        rknn_input inputs[rknn_app_ctx.io_num.n_input];
        rknn_output outputs[rknn_app_ctx.io_num.n_output];
        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));
        letterbox_t letter_box;                                     // todo: if the height or width of the image is odd, the image after padding may lack of one pixel

        // initialize time
        std::vector<std::chrono::microseconds> durations;
        auto start = std::chrono::high_resolution_clock::now();
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        // preprocess 
        start = std::chrono::high_resolution_clock::now();
        vector<vector<int>> batched_pad_w, batched_pad_h;
        vector<float> batched_scale_factors;
        // use rga instead of opencv to deal with image preprocess, rga can save a lot of cpu usage compared with opencv
        // https://zhuanlan.zhihu.com/p/665203639?utm_campaign=shareopn&utm_medium=social&utm_psn=1753495067145564160&utm_source=wechat_session
        preprocess_rga(img, inputs, letter_box, &rknn_app_ctx, log);    
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        durations.push_back(duration);

        // infer
        start = std::chrono::high_resolution_clock::now();
        do_infer(&rknn_app_ctx);
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        durations.push_back(duration);

        // postprocess
        start = std::chrono::high_resolution_clock::now();
        Result result;
        if (rknn_app_ctx.is_quant) {
            postprocess_i8(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, result, log);
        } else {
            postprocess_fp(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, result, output_numbox, output_numprob, log);
        }
        stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        durations.push_back(duration);
        
        // release some pointers
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);        // 1 for num output

        // print time consuming
        if (log) print_consuming_time(durations);

        // return results
        job.pro->set_value(result);
    }

private:
    // 可调数据
    string modelPath;                                           // 模型路径
    size_t batch_size;
    float box_conf_threshold{0.5};
    float nms_threshold{0.4};
    // 多线程有关
    atomic<bool> running_{false};                               // 如果InferImpl类析构，那么开启的子线程也要break
    thread worker_thread_;
    queue<Job> jobs_;                                           // 任务队列
    mutex lock_;                                                // 负责任务队列线程安全的锁
    condition_variable cv_;                                     // 线程通信函数
    size_t pool_size{1};
    // 模型初始化有关           
    rknn_app_context_t rknn_app_ctx;
    int output_numbox;                                          // the second dim of the output tensor, eg: [1, 151200, 9]
    int output_numprob;                                         // the third dim of the output tensor, eg: [1, 151200, 9]
};

std::shared_ptr<InferInterface> create_infer(std::string modelPath){                         // 返回的指针向纯虚类转化
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(modelPath)){
        instance.reset();                                                               // 如果模型加载失败，instance要reset成空指针
    }
    return instance;
};