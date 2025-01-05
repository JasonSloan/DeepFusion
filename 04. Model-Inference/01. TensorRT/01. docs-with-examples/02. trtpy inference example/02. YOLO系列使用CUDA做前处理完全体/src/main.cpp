#include <stdio.h>
#include <string>
#include <vector>
#include <dirent.h>     // opendir和readdir包含在这里
#include <sys/stat.h>

#include "cuda_runtime.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

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


void preprocess_kernel_invoker(
        int src_width, int src_height, int src_line_size,
        int dst_width, int dst_height, int dst_line_size,
        uint8_t* src_device, uint8_t* intermediate_device, 
        float* dst_device, uint8_t fill_value, int dst_img_area, size_t offset
);

int listdir(string& input,  vector<string>& files_vector) {
    DIR* pDir = opendir(input.c_str());
    if (!pDir) {
        cerr << "Error opening directory: " << strerror(errno) << endl;
        return -1;
    }
    struct dirent* ptr;
    while ((ptr = readdir(pDir)) != nullptr) {
        if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0) {
            files_vector.push_back(input + "/" + ptr->d_name);
        }
    }
    closedir(pDir);
    std::sort(files_vector.begin(), files_vector.end());
	return 0;
}

void preprocess_gpu(
	int n_imgs, vector<string> imgsPath,
	int dst_width, int dst_height, int dst_channels,
	float*& dst_host
){
	int dst_line_size = dst_width * dst_channels;
	int dst_img_area = dst_height * dst_width;
	float* dst_device;
	int dst_numel = n_imgs * dst_height * dst_width * dst_channels * sizeof(float);
	checkRuntime(cudaMalloc(&dst_device, dst_numel));
	checkRuntime(cudaMallocHost(&dst_host, dst_numel));

	for (int i = 0; i < n_imgs; ++i){
		Mat img = cv::imread(imgsPath[i]);
		int src_width = img.cols;
		int src_height = img.rows;
		int src_channels = img.channels();
		int src_line_size = src_width * src_channels;
		
		uint8_t* src_device;
		uint8_t* intermediate_device;
		uint8_t fill_value = 114;
		size_t offset = i * dst_height * dst_width * dst_channels; 

		int src_numel = src_height * src_width * src_channels * sizeof(uint8_t);
		int intermediate_numel = dst_height * dst_width * dst_channels * sizeof(uint8_t);
		checkRuntime(cudaMalloc(&src_device, src_numel));
		checkRuntime(cudaMalloc(&intermediate_device, intermediate_numel));
		checkRuntime(cudaMemcpy(src_device, img.data, src_numel, cudaMemcpyHostToDevice));

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
	checkRuntime(cudaMemcpy(dst_host, dst_device, dst_numel, cudaMemcpyDeviceToHost));
	checkRuntime(cudaFree(dst_device));
}

void preprocess_cpu(
	int n_imgs, vector<string> imgsPath,
	int dst_width, int dst_height, int dst_channels,
	float* dst_host
){
	// Resize and pad
	for (int i = 0; i < n_imgs; ++i){
		Mat img = cv::imread(imgsPath[i]);
		int img_height = img.rows;
		int img_width = img.cols;
		int img_channels = img.channels();

		float scale_factor = min(static_cast<float>(dst_width) / static_cast<float>(img.cols),
						static_cast<float>(dst_height) / static_cast<float>(img.rows));
		int img_new_w_unpad = img.cols * scale_factor;
		int img_new_h_unpad = img.rows * scale_factor;
		int pad_wl = round((dst_width - img_new_w_unpad - 0.01) / 2);		                   
		int pad_wr = round((dst_width - img_new_w_unpad + 0.01) / 2);
		int pad_ht = round((dst_height - img_new_h_unpad - 0.01) / 2);
		int pad_hb = round((dst_height - img_new_h_unpad + 0.01) / 2);
		cv::resize(img, img, cv::Size(img_new_w_unpad, img_new_h_unpad));
		cv::copyMakeBorder(img, img, pad_ht, pad_hb, pad_wl, pad_wr, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

		// HWC-->CHW & /255. & BGR2RGB
		float* i_dst_host;
		size_t img_area = dst_height * dst_width;
		i_dst_host = dst_host + img_area * 3 * i;
		unsigned char* pimage = img.data;
		float* phost_r = i_dst_host + img_area * 0;
		float* phost_g = i_dst_host + img_area * 1;
		float* phost_b = i_dst_host + img_area * 2;
		for(int j = 0; j < img_area; ++j, pimage += 3){
			*phost_r++ = pimage[2] / 255.0f;
			*phost_g++ = pimage[1] / 255.0f;
			*phost_b++ = pimage[0] / 255.0f;
		}
	}
}

void compare_difference(
	int n_imgs, int dst_width, int dst_height, int dst_channels, 
	float* preprocessed_gpu, float* preprocessed_cpu
){
	int imgone_numel = dst_width * dst_height * dst_channels;
	for (int i = 0; i < n_imgs; ++i){
		for (int j = 0; j < imgone_numel; ++j){
			int idx = i * imgone_numel + j;
			float* pcpu = preprocessed_cpu + idx;
			float* pgpu = preprocessed_gpu + idx;
			float gap = *pcpu - *pgpu;
			if (gap > 0.001) 
				printf("Index: %d, cpu computed value: %f, gpu computed value: %f, gap: %f\n", idx, *pcpu, *pgpu, gap);
		}
	}
}

int main(){
	/* 使用cuda批量做yolo的前处理, 包括: resize+copyMakeBoder+BGR2RGB+/255.0
	   此代码流程: 
	   1. 开辟需要预处理的n张图片在预处理后应占用的显存地址(dst_device)
	   2. 遍历图片开始
	   		读取图片
			开辟该图片原始大小的显存地址(src_device)
			开辟该图片仿射变换后大小的显存地址(intermediate_device)
			将图片数据搬运到src_device上
			使用cuda做resize+copyMakeBoder的操作, 将src_device的数据映射到intermediate_device上
			对图片做BGR2RGB+/255.0的操作, 将intermediate_device的数据映射到dst_device上的第i张图片的位置上
			释放intermediate_device
			释放src_device
		3. 将dst_device上的数据搬运到dst_host上
		4. 比较和cpu做该预处理后数据之间的差异
	*/
	string imgsFolder = "inputs/images";
	vector<string> imgsPath;
	listdir(imgsFolder, imgsPath);
	int n_imgs = imgsPath.size();

	int dst_width = 640;
	int dst_height = 384;
	int dst_channels = 3;
	int dst_numel = n_imgs * dst_height * dst_width * dst_channels * sizeof(float);
	float* preprocessed_gpu;
	float* preprocessed_cpu;
	cudaMallocHost(&preprocessed_gpu, dst_numel);
	preprocessed_cpu = new float[dst_numel];

	printf("---->Start to preprocess with cuda......\n");
	preprocess_gpu(n_imgs, imgsPath, dst_width, dst_height, dst_channels, preprocessed_gpu);
	printf("---->Preprocess with cuda is done!\n");

	printf("---->Start to preprocess with cpu......\n");
	preprocess_cpu(n_imgs, imgsPath, dst_width, dst_height, dst_channels, preprocessed_cpu);
	printf("---->Preprocess with cpu is done!\n");

	printf("---->Compare the difference of the results between gpu and cpu...\n");
	compare_difference(n_imgs, dst_width, dst_height, dst_channels, preprocessed_gpu, preprocessed_cpu);

	delete [] preprocessed_cpu;
	cudaFreeHost(preprocessed_gpu);

	printf("All done!\n");
    return 0;
};