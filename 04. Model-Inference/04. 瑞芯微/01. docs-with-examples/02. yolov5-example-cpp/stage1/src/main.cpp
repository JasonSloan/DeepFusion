#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>                // max_element要用到
#include <chrono>

#include "rknn_api.h"
#include "common.h"
#include "image_utils.h"
// #include "yolov5.h"

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

#define OBJ_NUMB_MAX_SIZE 128
const int anchor[3][6] = {{10, 13, 16, 30, 33, 23},
                          {30, 61, 62, 45, 59, 119},
                          {116, 90, 156, 198, 373, 326}};

const vector<Scalar> COLORS = {
	{255, 0, 0},
	{0, 255, 0},
	{0, 0, 255},
	{0, 255, 255}
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


typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

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


int init_yolov5_model(const char *model_path, rknn_app_context_t *app_ctx, int& output_numbox, int& output_numprob, const bool log) {
    int ret;
    int model_len = 0;
    char *model;
    rknn_context ctx = 0;

    // Load RKNN Model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL){
        printf("load_model fail!\n");
        return -1;
    }

    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);          // 上一步已经将数据赋值给ctx, 所以这一步就将model中的数据释放  
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
    if (log) printf("model input num: %d, output num: %d\n", io_num.n_input, io_num.n_output);

    // Get Model Input Info
    
    rknn_tensor_attr input_attrs[io_num.n_input];
    if (log) printf("input tensors:\n");
    memset(input_attrs, 0, sizeof(input_attrs));
    input_attrs[0].index = 0;
    ret = rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[0]), sizeof(rknn_tensor_attr));
    if (ret != 0) {
        printf("rknn_query fail! ret=%d\n", ret);
        return -1;
    }
    if (log) dump_tensor_attr(&(input_attrs[0]));

    // Get Model Output Info
    if (log) printf("output tensors:\n");
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC){
            printf("rknn_query fail! ret=%d\n", ret);
            return -1;
        }
        if (log) dump_tensor_attr(&(output_attrs[i]));
    }

    // Set to context
    app_ctx->rknn_ctx = ctx;

    // set quant or not 
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16){
        app_ctx->is_quant = true;
    }else{
        app_ctx->is_quant = false;
        output_numbox = output_attrs[0].dims[1];            // these two variables only use in 1 output with fp16 infer type
        output_numprob = output_attrs[0].dims[2];           // these two variables only use in 1 output with fp16 infer type
    }

    app_ctx->io_num = io_num;
    app_ctx->input_attrs = (rknn_tensor_attr *)malloc(io_num.n_input * sizeof(rknn_tensor_attr));       
    memcpy(app_ctx->input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    app_ctx->output_attrs = (rknn_tensor_attr *)malloc(io_num.n_output * sizeof(rknn_tensor_attr));     
    memcpy(app_ctx->output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));

    /*  my onnx model is NCHW format input, 
        but when i use input_attrs to retrieve the input format, 
        it tells me that it is NHWC format which is weird,
        it is weird, but the code goes right, so i just leave it there  */
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW){
        if (log) printf("model is NCHW input fmt\n");
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    }else{
        if (log) printf("model is NHWC input fmt\n");
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }
    if (log) printf("model input height=%d, width=%d, channel=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);

    return 0;
}


void preprocess_rga(Mat& im0, rknn_input* inputs, rknn_app_context_t* app_ctx, letterbox_t& letter_box){
    // prepare src data
    image_buffer_t src_img;
    memset(&src_img, 0, sizeof(image_buffer_t));
    int src_w = im0.cols;
    int src_h = im0.rows;
    int src_ch = im0.channels();
    image_format_t format = IMAGE_FORMAT_RGB888;
    int src_size = src_w * src_h * src_ch;
    src_img.width = src_w;
    src_img.height = src_h;
    src_img.format = format;
    src_img.virt_addr = im0.data;
    src_img.size = src_size;

    // prepare dst data
    image_buffer_t dst_img;
    memset(&dst_img, 0, sizeof(image_buffer_t));
    int dst_w = app_ctx->model_width;
    int dst_h = app_ctx->model_height;
    int dst_ch = app_ctx->model_channel;
    int dst_size = dst_w * dst_h * dst_ch;
    dst_img.width = dst_w;
    dst_img.height = dst_h;
    dst_img.format = format;
    dst_img.virt_addr = new unsigned char[dst_size];         // ! never forget to free the pointer
    dst_img.size = dst_size;
    
    // letterbox
    int bg_color = 114;
    memset(&letter_box, 0, sizeof(letterbox_t));
    int ret = convert_image_with_letterbox(&src_img, &dst_img, &letter_box, bg_color);      
    if (ret < 0) {
        printf("convert_image_with_letterbox fail! ret=%d\n", ret);
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
                vector<vector<float>>& box_result, 
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
    box_result.reserve(bboxes.size());             // 给box_result保留至少bboxes.size()个存储数据的空间
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
        box_result.emplace_back(ibox);                  // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
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
    if (log) printf("box_result.size = %d\n", box_result.size());
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

void postprocess_i8(rknn_app_context_t *app_ctx, 
                rknn_output* outputs,
                letterbox_t* letter_box, 
                float confidence_threshold, 
                float nms_threshold,
                vector<vector<float>>& box_result,
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
                        box_w = box_w * box_w * (float)anchor[k][a * 2];
                        box_h = box_h * box_h * (float)anchor[k][a * 2 + 1];
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
    box_result.reserve(bboxes.size());             // 给box_result保留至少bboxes.size()个存储数据的空间
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
        box_result.emplace_back(ibox);                  // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
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
    if (log) printf("box_result.size = %d\n", box_result.size());
}


void draw_rectangles(vector<vector<float>>& boxes, Mat& im0, const string& save_path){
    for (int j = 0; j < boxes.size(); j++) {
        cv::rectangle(
            im0, 
            cv::Point(boxes[j][0], boxes[j][1]), 
            cv::Point(boxes[j][2], boxes[j][3]), 
            COLORS[boxes[j][4]], 
            5, 8, 0
            );
        // cv::putText(im0, LABELS[result.labels[i]], cv::Point(result.boxes[i][0], result.boxes[i][1]), cv::FONT_HERSHEY_SIMPLEX, 1.4, COLORS[result.labels[i]], 2);
        cv::imwrite(save_path, im0);
    }
}

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

float mean(vector<float> x){
    float sum = 0;
    for (int i = 0; i < x.size(); ++i){
        sum += x[i];
    }
    return sum / x.size();
}

int main() {
    // some settings
    const char* model_path = "weights/last-relu-3output-i8-bs1.rknn";
    const char* image_path = "inputs/images/035018_ch13_202307150712290350.jpg";
    const string save_path = "outputs/035018_ch13_202307150712290350.jpg";
    const float box_conf_threshold = 0.5;
    const float nms_threshold = 0.4;
    bool log = false;

    // read image
    Mat img = imread(image_path, IMREAD_COLOR);
    Mat im0 = img.clone();                  // for visulization
    cvtColor(img, img, COLOR_BGR2RGB);      // the input image should be RGB format 

    // init model
    int ret;
    int output_numbox;                      // the second dim of the output tensor, eg: [1, 151200, 9]
    int output_numprob;                     // the third dim of the output tensor, eg: [1, 151200, 9]
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));   
    ret = init_yolov5_model(model_path, &rknn_app_ctx, output_numbox, output_numprob, log);
    if (ret != 0) {
        printf("init_yolov5_model fail! ret=%d model_path=%s\n", ret, model_path);
        return -1;
    }
    int niters = 10;
    auto total_start = std::chrono::high_resolution_clock::now();
    std::vector<float> pre_durations;
    std::vector<float> infer_durations;
    std::vector<float> post_durations;
    rknn_input inputs[rknn_app_ctx.io_num.n_input];
    rknn_output outputs[rknn_app_ctx.io_num.n_output];
    vector<vector<float>> boxes;
    for (int i = 0; i < niters; ++i){
        // preprocess 
        auto pre_start = std::chrono::high_resolution_clock::now();
        letterbox_t letter_box;                                     // todo: if the height or width of the image is odd, the image after padding may lack of one pixel
        memset(inputs, 0, sizeof(inputs));
        memset(outputs, 0, sizeof(outputs));
        // use rga instead of opencv to deal with image preprocess, rga can save some cpu usage compared with opencv
        preprocess_rga(img, inputs, &rknn_app_ctx, letter_box);     
        auto pre_end = std::chrono::high_resolution_clock::now();
        auto pre_duration = std::chrono::duration_cast<std::chrono::microseconds>(pre_end - pre_start).count() / 1000.;
        pre_durations.push_back(pre_duration);

        // infer
        auto infer_start = std::chrono::high_resolution_clock::now();
        do_infer(&rknn_app_ctx);
        auto infer_end = std::chrono::high_resolution_clock::now();
        auto infer_duration = std::chrono::duration_cast<std::chrono::microseconds>(infer_end - infer_start).count() / 1000.;
        infer_durations.push_back(infer_duration);

        // postprocess (boxes: x1, y1, x2, y2, cls_idx, confidence)
        auto post_start = std::chrono::high_resolution_clock::now();
        if (!boxes.empty())
            boxes.clear();
        if (rknn_app_ctx.is_quant) {
            postprocess_i8(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, boxes, log);
        } else {
            postprocess_fp(&rknn_app_ctx, outputs, &letter_box, box_conf_threshold, nms_threshold, boxes, output_numbox, output_numprob, log);
        }
        auto post_end = std::chrono::high_resolution_clock::now();
        auto post_duration = std::chrono::duration_cast<std::chrono::microseconds>(post_end - post_start).count() / 1000.;
        post_durations.push_back(post_duration);

        // // visualization
        draw_rectangles(boxes, im0, save_path);

        // release some pointers
        rknn_outputs_release(rknn_app_ctx.rknn_ctx, rknn_app_ctx.io_num.n_output, outputs);        // 1 for num output
        if (inputs[0].buf != NULL)
            free(inputs[0].buf);
    }
	auto total_end = std::chrono::high_resolution_clock::now();
	auto avg_total_durations = std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count() / 1000. / niters;
    float avg_pre_duration = mean(pre_durations);
    float avg_infer_duration = mean(infer_durations);
    float avg_post_duration = mean(post_durations);
	printf("\nTotal time consuming per image:\ntotal: %.2f ms,\tpreprocess: %.2f ms,\tinfer %.2f ms,\tpostprocess %.2f ms\n", 
            avg_total_durations, avg_pre_duration, avg_infer_duration, avg_post_duration);

    release_model(&rknn_app_ctx);

    return 0;
}