#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef unsigned char uint8_t;

__global__ void data_cast_kernel(float *src, uint8_t *dst, int dst_width, int dst_height, int src_line_size, int dst_line_size){
    int dx = blockDim.x * blockIdx.x + threadIdx.x;         // 计算线程在全局的索引
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dst_width || dy >= dst_height) return;        // 超出图像范围的线程直接返回 
    else{
        float* v_src = src + dy * src_line_size + dx * 3;
        uint8_t* v_dst = dst + dy * dst_line_size + dx * 3;
        v_dst[0] = static_cast<uint8_t>(v_src[0]);          // 将float类型的数据转换为uint8_t类型
        v_dst[1] = static_cast<uint8_t>(v_src[1]);
        v_dst[2] = static_cast<uint8_t>(v_src[2]);
    }
};

void data_cast(float *src, uint8_t *dst, int dst_width, int dst_height, int src_line_size, int dst_line_size){
    dim3 block_size(32, 32); // block_size最大只能为1024；32*32=1024
    dim3 grid_size(ceil(dst_width / 32), ceil(dst_height / 32));
    data_cast_kernel<<<grid_size, block_size, 0, nullptr>>>(src, dst, dst_width, dst_height, src_line_size, dst_line_size);
};