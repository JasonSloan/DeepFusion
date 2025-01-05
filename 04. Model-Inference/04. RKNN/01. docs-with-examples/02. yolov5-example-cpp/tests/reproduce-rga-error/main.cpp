#include <iostream>
#include <stdio.h> 

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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
#include "dma_alloc.h"
#include "opencv2/opencv.hpp"
#include "inference.hpp"

using namespace std;
using namespace cv;


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


int init_model(string modelPath, rknn_app_context_t& rknn_app_ctx) {
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
    memset(input_attrs, 0, sizeof(input_attrs));
    input_attrs[0].index = 0;                           // there is only one input, so here I just specify the input index to 0
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }

    
    // Get Model Output Info
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {         // there might be 3 outputs for Int8 inference or 1 output for FP16 inference
        output_attrs[i].index = i;              
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC){
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }      
    }

    // allocate memory and set to context
    app_ctx->rknn_ctx = ctx;

    // set quant or not 
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16){
        app_ctx->is_quant = true;
    }else{
        app_ctx->is_quant = false;
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
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }else{
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    return 0;
}

int letterbox_rga(image_buffer_t& src_img, 
                image_buffer_t& dst_img, 
                letterbox_t& letter_box,
                rknn_app_context_t* app_ctx){
    // prepare dst data
    memset(&dst_img, 0, sizeof(image_buffer_t));
    int dst_w = 640;
    int dst_h = 384;
    int dst_ch = 3;
    int dst_size = dst_w * dst_h * dst_ch;
    dst_img.width = dst_w;
    dst_img.height = dst_h;
    dst_img.format = IMAGE_FORMAT_RGB888;
    shared_ptr<unsigned char> d_virt_addr(new unsigned char[dst_size], default_delete<unsigned char[]>());
    dst_img.virt_addr = d_virt_addr.get();
    dst_img.size = dst_size;
    
    // letterbox
    int bg_color = 114;
    memset(&letter_box, 0, sizeof(letterbox_t));
    int ret = convert_image_with_letterbox(&src_img, &dst_img, &letter_box, bg_color);      
    if (ret < 0) {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
        return -1;
    }
    return 0;
}

int cvtColor_rga_old(int& i_head,
                size_t& i_size,
                image_buffer_t& img_t, 
                int src_format=RK_FORMAT_BGR_888, 
                int dst_format=RK_FORMAT_RGB_888, 
                bool log=false){
    auto release_handle = [] (rga_buffer_handle_t src_handle, rga_buffer_handle_t dst_handle) {
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
    };

    int ret;
    int width = img_t.width;
    int height = img_t.height;
    int src_buf_size, dst_buf_size;
    rga_buffer_t src_img, dst_img;
    rga_buffer_handle_t src_handle, dst_handle;

    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    src_buf_size = width * height * get_bpp_from_format(src_format);
    dst_buf_size = width * height * get_bpp_from_format(dst_format);
    assert(src_buf_size == img_t.size), "Assertion failed! src_buf_size != img_t.size";
    i_size = dst_buf_size;

    shared_ptr<unsigned char> src_buf(new unsigned char[src_buf_size], default_delete<unsigned char[]>());
    shared_ptr<unsigned char> dst_buf(new unsigned char[src_buf_size], default_delete<unsigned char[]>());

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

    ret = imcheck(src_img, dst_img, {}, {});
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

int cvtColor_rga_new(int& i_head,
                size_t& i_size,
                image_buffer_t& img_t, 
                int src_format=RK_FORMAT_BGR_888, 
                int dst_format=RK_FORMAT_RGB_888, 
                bool log=false){
    auto release_handle = [] (rga_buffer_handle_t src_handle, rga_buffer_handle_t dst_handle) {
        if (src_handle) releasebuffer_handle(src_handle);
        if (dst_handle) releasebuffer_handle(dst_handle);
    };

    int ret;
    int src_dma_fd;
    int dst_dma_fd;
    int width = img_t.width;
    int height = img_t.height;
    int src_buf_size, dst_buf_size;
    rga_buffer_t src_img, dst_img;
    rga_buffer_handle_t src_handle, dst_handle;

    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    src_buf_size = width * height * get_bpp_from_format(src_format);
    dst_buf_size = width * height * get_bpp_from_format(dst_format);
    assert(src_buf_size == img_t.size), "Assertion failed! src_buf_size != img_t.size";
    i_size = dst_buf_size;

    unsigned char* src_buf;
    unsigned char* dst_buf;
    ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, src_buf_size, &src_dma_fd, (void **)&src_buf);
    if (ret < 0) {
        printf("alloc dma32_heap buffer failed!\n");
        return -1;
    }
    ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, dst_buf_size, &dst_dma_fd, (void **)&dst_buf);
    if (ret < 0) {
        printf("alloc dma32_heap buffer failed!\n");
        return -1;
    }

    memcpy(src_buf, img_t.virt_addr, src_buf_size);
    memset(dst_buf, 0x80, dst_buf_size);

    src_handle = importbuffer_fd(src_dma_fd, src_buf_size);
    if (src_handle == 0) {
        printf("import dma_fd error!\n");
        ret = -1;
    }
    dst_handle = importbuffer_fd(dst_dma_fd, dst_buf_size);
    if (dst_handle == 0) {
        printf("import dma_fd error!\n");
        ret = -1;
    }

    src_img = wrapbuffer_handle(src_handle, width, height, src_format);
    dst_img = wrapbuffer_handle(dst_handle, width, height, dst_format);

    ret = imcheck(src_img, dst_img, {}, {});
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

    memcpy(img_t.virt_addr, dst_buf, dst_buf_size);
    release_handle(src_handle, dst_handle);
    dma_buf_free(dst_buf_size, &dst_dma_fd, dst_buf);
    dma_buf_free(src_buf_size, &src_dma_fd, src_buf);
    return 0;
}


int main() {
    string path = "inputs/images/0002_1_000002.jpg";
    string modelPath = "weights/last-relu-3output-i8-bs1.rknn";
    rknn_app_context_t rknn_app_ctx;
    init_model(modelPath, rknn_app_ctx);
    Mat img = imread(path, 1);
    image_buffer_t src_img;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    rknn_app_context_t* app_ctx = &rknn_app_ctx;
    memset(&src_img, 0, sizeof(image_buffer_t));
    memset(&dst_img, 0, sizeof(image_buffer_t));
    memset(&letter_box, 0, sizeof(letterbox_t));
    src_img.width = img.cols;
    src_img.height = img.rows;
    src_img.format = IMAGE_FORMAT_RGB888;
    src_img.virt_addr = img.data;
    src_img.size = img.cols * img.rows * 3;
    int i_head;
    size_t i_size;
    int niter = 1000;
    for (int i = 0; i < niter; ++i){
        letterbox_rga(src_img, dst_img, letter_box, app_ctx);
        // cvtColor_rga_old(i_head, i_size, src_img, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888, false);
        // cvtColor_rga_new(i_head, i_size, src_img, RK_FORMAT_BGR_888, RK_FORMAT_RGB_888, false);
        
    }
    Mat result(dst_img.height, dst_img.width, CV_8UC3, dst_img.virt_addr);
    imwrite("result.jpg", result);
    // Mat result(src_img.height, src_img.width, CV_8UC3);
    // size_t size_result = src_img.height * src_img.width * 3;
    // memcpy(result.data, src_img.virt_addr, size_result);
    // imwrite("result.jpg", result);
    return 0;
}