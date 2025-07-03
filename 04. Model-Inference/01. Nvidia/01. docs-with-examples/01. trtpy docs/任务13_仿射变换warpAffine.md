**本篇文章是帮助对仿射变换理解的，真正不需要这么使用，真正使用只需要计算放射变换矩阵，然后调用OpenCV来做**

# 一. 前置知识

## （一）双线性插值（Bilinear)

### 	1. 单线性插值

​		离得近的占比大，离得远的占比小

​					![img](pics/单线性插值.png)

![img](pics/单线性插值公式.png)

###2. 双线性插值

​		已知Q~11~	、Q~12~、Q~21~、Q~22~四个点的值，求点P的值：

​							![img](pics/双线性插值.png)

​		先通过Q~11~、Q~21~ 单线性插值求R~1~ 的值、通过Q~12~、Q~22~ 单线性插值求R~2~ 的值：

​											![img](pics/双线性插值第一步.png)

​		再通过R~1~ 、R~2~ 单线性插值求P的值：

​				![img](pics/双线性插值第二步.png)

### 	3. 例子

​						![img](pics/双线性插值例子.png)

# 二. warpAffine

## （一）概念

​	![](pics/仿射变换示例.png)

​	warpAffine（仿射变换）：**仿射变换**是指在几何中，對一个向量空间进行一次线性变换并接上一个平移，变换为另一个向量空间。

​	[参考jupyter notebook](https://github.com/duanmushuangquan/cudatrtpyCode/blob/master/mycudaCodeSpace/RuntimeAPI/warpaffine/warpaffine.ipynb)

​	[视频讲解](7.46 JVl:/ 双线性插值在计算机视觉中的应用# 在家上课 # 抖音教育 # 讲解 # 打卡学习 # 学习   https://v.douyin.com/iJTMpKQ/ 复制此链接，打开Dou音搜索，直接观看视频！)

​			![](pics/仿射变换.png)

​	**原图坐标--> 仿射变换 --> 目标图坐标**：已有**原图大小 **、**目标图大小**，可以求得**仿射变换矩阵** ，给任意原图坐标通过仿射变换矩阵可以得到原图在目标图上对应的坐标。

​	**原图 --> 仿射变换 --> 目标图**：已有**原图大小 **、**目标图大小**，可以求得**仿射变换矩阵** ，如果再有**原图数据**，则可以对原图进行仿射变换得到一张**目标图**。

​	**目标图坐标 --> 仿射变换逆矩阵 --> 目标图在原图对应的坐标** ：已知**目标图坐标**、**仿射变换逆矩阵**可以求得**目标图在原图中** 对应的坐标。如果想求取对应的像素值，由于**目标图在原图中** 对应的坐标可能不为整数，可以通过双线性插值求取对应的像素值。

## （二）代码

###1. 编写main函数

```C++
#include <cuda_runtime.h>     // cudaMalloc函数所在位置
#include <opencv2/opencv.hpp> // 图像处理函数
#include <stdio.h>            // printf函数所在位置

using namespace cv; // 使用opencv的命名空间

#define min(a, b) ((a) < (b) ? (a) : (b))                                    // 简单的宏定义
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__) // 检查每一个cuda-runtime的api的执行成功情况

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

int main()
{
    Mat image = imread("yq.jpg");                                   // 读取图像
    Mat output = warpAffine_to_center_align(image, Size(640, 640)); // 执行仿射变换得到目标图像，Size是opencv下的一个函数，center_align是居中对齐的意思
    imwrite("output.jpg", output);                                  // 存储
    printf("Done!");
    return 0;
}
```

### 2. 编写warpAffine_to_center_align函数

![](pics/warpAffine_to_center_align.png)

```C++
Mat warpAffine_to_center_align(const Mat& image, const Size& size){
    Mat output(size, CV_8UC3);   // 创建一个输出矩阵，用来接从gpu计算完拷贝回来的数据
    size_t src_size = image.rows * image.cols * 3;  // 原图占用内存大小
    size_t dst_size = size.height * size.width * 3; // 目标图占用内存大小
    uint8_t* psrc_device = nullptr;   // 初始化在gpu中原图指针
    uint8_t* pdst_device = nullptr;   // 初始化在gpu中目标图指针

    checkRuntime(cudaMalloc(&psrc_device, src_size));  // 在gpu中分配内存，这里为什么不是src_size*sizeof(uint8)，是因为sizeof(uint8)=1
    checkRuntime(cudaMalloc(&pdst_device, dst_size));// 在gpu中分配内存
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 将数据从CPU搬运到GPU上
    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114 
        );  // 在gpu上执行仿射变换

    checkRuntime(cudaPeekAtLastError());   // 检查gpu上执行完是否有错误
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将数据从GPU搬运回到CPU上

    checkRuntime(cudaFree(pdst_device));  // 释放内存
    checkRuntime(cudaFree(psrc_device));
    return output;
}
```

### 3.在cu文件中编写warp_affine_bilinear函数

```C++
void warp_affine_bilinear(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value)
{
    dim3 block_size(32, 32);                                     // block_size最大只能为1024；32*32=1024
    dim3 grid_size(ceil(dst_width / 32), ceil(dst_height / 32)); // 计算grid_size

    AffineMatrix affine;
    affine.compute(Size(src_width, src_height), Size(dst_width, dst_height));

    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>( // 调用核函数，0代表共享内存，nullptr代表默认流
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine);
}
```

### 4. 在cu文件中编写核函数warp_affine_bilinear_kernel

```C++
__global__ void warp_affine_bilinear_kernel(
    uint8_t* src, int src_line_size, int src_width, int src_height, 
    uint8_t* dst, int dst_line_size, int dst_width, int dst_height, 
	uint8_t fill_value, AffineMatrix matrix
){
    /* 已知目标图坐标（即当前线程在全局的索引）-->通过仿射变换矩阵获得目标图坐标在原图对应的坐标p（坐标p不为整数）-->
        -->坐标p向上向下取整获得周围四个点的坐标-->双线性插值得到原图对应的坐标p（也就是目标图坐标）对应的像素值
     */
    // 计算线程在全局的索引，也就是目标图当前坐标（这里我不管别的，我就指定当前线程处理目标图的对应像素值）。注释：图1
    int dx = blockDim.x * blockIdx.x + threadIdx.x; 
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    // 如果线程索引值超出目标图的宽和高了，则停止
    if (dx >= dst_width || dy >= dst_height)  return;

    // 初始化目标图上一个点三个通道的像素值
    float c0 = fill_value, c1 = fill_value, c2 = fill_value;
    // 初始化目标图做逆仿射变换后对应在原图的坐标
    float src_x = 0; float src_y = 0;
    // 做逆仿射变换，也就是已知目标图的坐标(当前线程的坐标)，反推对应在原图的坐标
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);

    // 如果超过了图像的边界，注释：图2（解释这里为什么是<-1不是小于0，为什么是>=width不是>width）
    if(src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height){
        // out of range
        // src_x < -1时，其高位high_x < 0，超出范围
        // src_x >= -1时，其高位high_x >= 0，存在取值
        printf("超过图像边界！");
    }else{
        // 计算目标图当前坐标映射回原图对应坐标的相邻的四个坐标点
        int y_low = floorf(src_y); 
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        // 初始化双线性插值的四个角的值都为{114， 114， 114}（RGB三通道）
        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        // 计算目标图当前坐标映射回原图对应坐标与相邻的四个坐标点的距离
        float ly    = src_y - y_low;        // 与上（注意y坐标轴指向是向下的）
        float lx    = src_x - x_low;        // 与左
        float hy    = 1 - ly;               // 与下
        float hx    = 1 - lx;               // 与右
        // w1:右下角的面积；w2:左下角的面积；w3:右上角的面积；w4:左上角的面积；
        float w1    = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
        // 先初始化四个角的值都为{114, 114, 114, 114}
        uint8_t* v1 = const_values;         // 左上
        uint8_t* v2 = const_values;         // 右上
        uint8_t* v3 = const_values;         // 左下
        uint8_t* v4 = const_values;         // 右下

        // 此时y方向没有越下界
        if(y_low >= 0){
            // 此时x方向没有越下界
            if (x_low >= 0)
                // 因为x、y都没越下界，所以才能取到左上角的像素值
                // 取得左上角的像素值（src地址偏移y_low * src_line_size + x_low * 3个位置）
                v1 = src + y_low * src_line_size + x_low * 3;
            // 此时x方向没有越上界
            if (x_high < src_width)
                // 因为y没越下界且x没越上界，所以才能取到右上角的像素值
                v2 = src + y_low * src_line_size + x_high * 3;
        }
        // 此时y方向没有越上界
        if(y_high < src_height){
            // 此时x方向没有越下界
            if (x_low >= 0)
                // 因为y没越上界且x没越下界，所以才能取到左下角的像素值
                v3 = src + y_high * src_line_size + x_low * 3;
            // 此时x方向没有越上界
            if (x_high < src_width)
                // 因为x、y都没越上界，所以才能取到右下的像素值
                v4 = src + y_high * src_line_size + x_high * 3;
        }
        
        // RGB三通道插值
        // 插值 = 右下面积 * 左上角的像素值 + 左下面积 * 右上角的像素值 + 右上面积 * 左下像素值 + 左上面积 * 右下像素值
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    // 将目标图像的坐标(目标图像的坐标等同于线程的索引)转换为目标图像的地址
    uint8_t* pdst = dst + dy * dst_line_size + dx * 3;
    // 为该地址下的偏移012三个位置赋予坐标值
    pdst[0] = c0; pdst[1] = c1; pdst[2] = c2;
}
```

注释图:1：参考（任务09-12_cuda_runtime_api.md）中的内存模型

![注解图1](pics/线程索引的计算.png)

注释图2：

![](pics/取值范围.png)

注释图3：

![](pics/坐标转地址.png)

###5. 在cu文件中编写 AffineMatrix结构体

```C++
// 计算仿射变换矩阵
// 计算的矩阵是居中缩放
struct AffineMatrix{
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr5UdL/
     */

    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    // 这里其实是求解imat的逆矩阵，由于这个3x3矩阵的第三行是确定的0, 0, 1，因此可以简写如下
    void invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }

    void compute(const Size& from, const Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        // 这里取min的理由是
        // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
        // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
        // **
        float scale = min(scale_x, scale_y); // 缩放比例辅助视频讲解 https://v.douyin.com/NhrH8Gm/
        /**
        这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
        scale, 0, -scale * from.width * 0.5 + to.width * 0.5
        0, scale, -scale * from.height * 0.5 + to.height * 0.5
        
        这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
        例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
        S = [
        scale,     0,      0
        0,     scale,      0
        0,         0,      1
        ]
        
        P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
        P = [
        1,        0,      -scale * from.width * 0.5
        0,        1,      -scale * from.height * 0.5
        0,        0,                1
        ]

        T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的左上角
        T = [
        1,        0,      to.width * 0.5,
        0,        1,      to.height * 0.5,
        0,        0,            1
        ]

        通过将3个矩阵顺序乘起来，即可得到下面的表达式：
        M = [
        scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
        0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
        0,        0,                     1
        ]
        去掉第三行就得到opencv需要的输入2x3矩阵
        **/

        /* 
            + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
            参考：https://www.iteye.com/blog/handspeaker-1545126
        */
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = 
            -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;

        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = 
            -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){

    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}
```

### 6.  cu文件中的其他

```C++
typedef unsigned char uint8_t; // 所有的uint8_t都是unsigned char类型

struct Size
{ // Size方便后面使用Size类型
    int width = 0;
    int height = 0;
    Size() = default;  // 默认构造函数，什么都不传
    Size(int w, int h) // 也可以传参数构造
        : width(w), height(h){};
};
```

## （三）代码整合

### 1. main.py


```C++

#include <cuda_runtime.h>     // cudaMalloc函数所在位置
#include <opencv2/opencv.hpp> // 图像处理函数
#include <stdio.h>            // printf函数所在位置

using namespace cv; // 使用opencv的命名空间

#define min(a, b) ((a) < (b) ? (a) : (b))                                    // 简单的宏定义
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__) // 检查每一个cuda-runtime的api的执行成功情况

bool __check_cuda_runtime(cudaError_t code, const char *op, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
        return false;
    }
    return true;
}

void warp_affine_bilinear(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value);

Mat warpAffine_to_center_align(const Mat &image, const Size &size)
{

    Mat output(size, CV_8UC3);                      // 创建一个输出矩阵，用来接从gpu计算完拷贝回来的数据
    size_t src_size = image.rows * image.cols * 3;  // 原图占用内存大小
    size_t dst_size = size.height * size.width * 3; // 目标图占用内存大小
    uint8_t *psrc_device = nullptr;                 // 初始化在gpu中原图指针
    uint8_t *pdst_device = nullptr;                 // 初始化在gpu中目标图指针

    checkRuntime(cudaMalloc(&psrc_device, src_size));                                    // 在gpu中分配内存，这里为什么不是src_size*sizeof(uint8)，是因为sizeof(uint8)=1
    checkRuntime(cudaMalloc(&pdst_device, dst_size));                                    // 在gpu中分配内存
    checkRuntime(cudaMemcpy(psrc_device, image.data, src_size, cudaMemcpyHostToDevice)); // 将数据从CPU搬运到GPU上
    warp_affine_bilinear(
        psrc_device, image.cols * 3, image.cols, image.rows,
        pdst_device, size.width * 3, size.width, size.height,
        114); // 在gpu上执行仿射变换

    checkRuntime(cudaPeekAtLastError());                                                  // 检查gpu上执行完是否有错误
    checkRuntime(cudaMemcpy(output.data, pdst_device, dst_size, cudaMemcpyDeviceToHost)); // 将数据从GPU搬运回到PU上

    checkRuntime(cudaFree(pdst_device)); // 释放内存
    checkRuntime(cudaFree(psrc_device));
    return output;
}

int main()
{
    Mat image = imread("yq.jpg");                                   // 读取图像
    Mat output = warpAffine_to_center_align(image, Size(640, 640)); // 执行仿射变换得到目标图像，Size是opencv下的一个函数，center_align是居中对齐的意思
    imwrite("output.jpg", output);                                  // 存储
    printf("Done!");
    return 0;
}
```

### 2. affine.cu

```C++
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>

typedef unsigned char uint8_t; // 下面所有的uint8_t都是unsigned char类型

struct Size
{ // Size方便后面使用Size类型
    int width = 0;
    int height = 0;
    Size() = default;  // 默认构造函数，什么都不传
    Size(int w, int h) // 也可以传参数构造
        : width(w), height(h){};
};

// 计算仿射变换矩阵
// 计算的矩阵是居中缩放
struct AffineMatrix{
    /* 
    建议先阅读代码，若有疑问，可点击抖音短视频进行辅助讲解(建议1.5倍速观看)
        - https://v.douyin.com/Nhr5UdL/
     */

    float i2d[6];       // image to dst(network), 2x3 matrix
    float d2i[6];       // dst to image, 2x3 matrix

    // 这里其实是求解imat的逆矩阵，由于这个3x3矩阵的第三行是确定的0, 0, 1，因此可以简写如下
    void invertAffineTransform(float imat[6], float omat[6]){
        float i00 = imat[0];  float i01 = imat[1];  float i02 = imat[2];
        float i10 = imat[3];  float i11 = imat[4];  float i12 = imat[5];

        // 计算行列式
        float D = i00 * i11 - i01 * i10;
        D = D != 0 ? 1.0 / D : 0;

        // 计算剩余的伴随矩阵除以行列式
        float A11 = i11 * D;
        float A22 = i00 * D;
        float A12 = -i01 * D;
        float A21 = -i10 * D;
        float b1 = -A11 * i02 - A12 * i12;
        float b2 = -A21 * i02 - A22 * i12;
        omat[0] = A11;  omat[1] = A12;  omat[2] = b1;
        omat[3] = A21;  omat[4] = A22;  omat[5] = b2;
    }

    void compute(const Size& from, const Size& to){
        float scale_x = to.width / (float)from.width;
        float scale_y = to.height / (float)from.height;

        // 这里取min的理由是
        // 1. M矩阵是 from * M = to的方式进行映射，因此scale的分母一定是from
        // 2. 取最小，即根据宽高比，算出最小的比例，如果取最大，则势必有一部分超出图像范围而被裁剪掉，这不是我们要的
        // **
        float scale = min(scale_x, scale_y); // 缩放比例辅助视频讲解 https://v.douyin.com/NhrH8Gm/
        /**
        这里的仿射变换矩阵实质上是2x3的矩阵，具体实现是
        scale, 0, -scale * from.width * 0.5 + to.width * 0.5
        0, scale, -scale * from.height * 0.5 + to.height * 0.5
        
        这里可以想象成，是经历过缩放、平移、平移三次变换后的组合，M = TPS
        例如第一个S矩阵，定义为把输入的from图像，等比缩放scale倍，到to尺度下
        S = [
        scale,     0,      0
        0,     scale,      0
        0,         0,      1
        ]
        
        P矩阵定义为第一次平移变换矩阵，将图像的原点，从左上角，移动到缩放(scale)后图像的中心上
        P = [
        1,        0,      -scale * from.width * 0.5
        0,        1,      -scale * from.height * 0.5
        0,        0,                1
        ]

        T矩阵定义为第二次平移变换矩阵，将图像从原点移动到目标（to）图的左上角
        T = [
        1,        0,      to.width * 0.5,
        0,        1,      to.height * 0.5,
        0,        0,            1
        ]

        通过将3个矩阵顺序乘起来，即可得到下面的表达式：
        M = [
        scale,    0,     -scale * from.width * 0.5 + to.width * 0.5
        0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
        0,        0,                     1
        ]
        去掉第三行就得到opencv需要的输入2x3矩阵
        **/

        /* 
            + scale * 0.5 - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
            参考：https://www.iteye.com/blog/handspeaker-1545126
        */
        i2d[0] = scale;  i2d[1] = 0;  i2d[2] = 
            -scale * from.width  * 0.5  + to.width * 0.5 + scale * 0.5 - 0.5;

        i2d[3] = 0;  i2d[4] = scale;  i2d[5] = 
            -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;

        invertAffineTransform(i2d, d2i);
    }
};

__device__ void affine_project(float* matrix, int x, int y, float* proj_x, float* proj_y){

    // matrix
    // m0, m1, m2
    // m3, m4, m5
    *proj_x = matrix[0] * x + matrix[1] * y + matrix[2];
    *proj_y = matrix[3] * x + matrix[4] * y + matrix[5];
}

__global__ void warp_affine_bilinear_kernel(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value, AffineMatrix matrix)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x; // 计算线程在全局的索引，注释：图1
    int dy = blockDim.y * blockIdx.y + threadIdx.y;

    if (dx >= dst_width || dy >= dst_height)
        return; // 如果线程索引值超出目标图的宽和高了，则停止

    float c0 = fill_value, c1 = fill_value, c2 = fill_value; // 初始化目标图上一点三个通道的像素值
    float src_x = 0;                                         // 初始化逆仿射变换后原图的坐标
    float src_y = 0;                                         // 初始化逆仿射变换后原图的坐标
    affine_project(matrix.d2i, dx, dy, &src_x, &src_y);      // 做仿射变换的逆变换，也就是已知目标图的坐标(当前线程的坐标)，反推原图坐标

    if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
    {
        // 如果超过了图像的边界，注释：图2（解释这里为什么是<-1不是小于0，为什么是>=width不是>width）
        printf("超过图像边界！");
    }
    else
    {
        int y_low = floorf(src_y); // 计算映射回原图中的点的相邻的四个坐标点
        int x_low = floorf(src_x);
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        float ly = src_y - y_low; // 计算映射回原图中的点距离四个坐标点的距离
        float lx = src_x - x_low;
        float hy = 1 - ly;
        float hx = 1 - lx;
        float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; // w1:右下角的面积；w2:左下角的面积；w3:右上角的面积；w4:左下角的面积；

        uint8_t const_values[] = {fill_value, fill_value, fill_value};
        uint8_t *v1 = const_values; // 初始化双线性插值的四个角的值都为{114， 114， 114}（RGB三通道）
        uint8_t *v2 = const_values;
        uint8_t *v3 = const_values;
        uint8_t *v4 = const_values;

        if (y_low >= 0) // 此时y方向没有越下界
        {
            if (x_low >= 0) // 此时x方向没有越下界
                /*计算左上角点在原图上的地址（坐标转换成地址），因为取左上角的点的值需要用到x_low坐标和y_low坐标，
                只有这俩坐标都大于0才能在原图上取值，否则取的值是越界de。这两个值为什么不用判断超过右边界和下边界呢，
                是因为他们是low，不可能超过右边界和下边界，因为，在38行的if里已经将其过滤掉了。注释：图3*/
                /*这里x_low为什么要乘3，是因为在内存中，每个x的位置都存储RGB三个值*/
                v1 = src + y_low * src_line_size + x_low * 3;
            if (x_high < src_width)
                // 取右上角点的地址，同理
                v2 = src + y_low * src_line_size + x_high * 3;
        }

        if (y_high < src_height)
        {
            if (x_low > 0)
                // 取左下角角点的地址，同理
                v3 = src + y_high * src_line_size + x_low * 3;
            if (x_high < src_width)
                // 取右下角角点的地址，同理
                v4 = src + y_high * src_line_size + x_high * 3;
        }

        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f); // R通道插值
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f); // G通道插值
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f); // B通道插值

        uint8_t *pdst = dst + dy * dst_line_size + dx * 3; // 将目标图像的坐标(目标图像的坐标等同于线程的索引)转换为目标图像的地址
        pdst[0] = c0;
        pdst[1] = c1;
        pdst[2] = c2; //  为目标图像的三通道赋值
    }
}

void warp_affine_bilinear(
    uint8_t *src, int src_line_size, int src_width, int src_height,
    uint8_t *dst, int dst_line_size, int dst_width, int dst_height,
    uint8_t fill_value)
{
    dim3 block_size(32, 32);                                     // block_size最大只能为1024；32*32=1024
    dim3 grid_size(ceil(dst_width / 32), ceil(dst_height / 32)); // 因为这里计算grid_size是使用dst_width和dst_height算的，在调用核函数的时候，指定了grid_size，所以是线程分配是为dst分配的

    AffineMatrix affine;
    affine.compute(Size(src_width, src_height), Size(dst_width, dst_height));

    warp_affine_bilinear_kernel<<<grid_size, block_size, 0, nullptr>>>( // 调用核函数，0代表共享内存，nullptr代表流（非异步）
        src, src_line_size, src_width, src_height,
        dst, dst_line_size, dst_width, dst_height,
        fill_value, affine);
}

```







​		

