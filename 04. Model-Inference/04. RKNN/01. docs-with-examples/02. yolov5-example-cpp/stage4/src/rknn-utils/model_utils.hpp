#pragma once

#include <chrono>
#include <fstream>
#include <stdio.h>
#include <dirent.h>     // opendir和readdir包含在这里
#include <sys/stat.h>

#include "rknn_api.h"

using namespace std;

using time_point = chrono::high_resolution_clock;
template <typename Rep, typename Period>
float micros_cast(const std::chrono::duration<Rep, Period>& d) {
    return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(d).count()) / 1000.;
}

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

string get_log_file_name(string& log_dir) {
    if (access(log_dir.c_str(), 0) != F_OK)
        mkdir(log_dir.c_str(), S_IRWXU);
    DIR* pDir = opendir(log_dir.c_str());
    struct dirent* ptr;
    vector<string> files_vector;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());
    int max_num = 0;
    if (files_vector.size() != 0) {
        for (auto &file : files_vector) {
            string num_str = file.substr(0, file.find("."));
            int num = std::stoi(num_str);
            if (num > max_num)
                max_num = num;
        }
        max_num += 1;
    }

	return std::to_string(max_num);
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