#include <dirent.h> 

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvOnnxParser.h>

#include "opencv2/opencv.hpp"
#include "spdlog/logger.h"                      
#include "spdlog/spdlog.h"                             

using namespace std;
using namespace cv;
using namespace nvinfer1;


template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){delete p;});
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

class TRTLogger: public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
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


class Int8EntropyCalibrator : public IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const string& calibImagesPath, int inputH, int inputW, int batchSize)
        : mInputH(inputH), mInputW(inputW), mBatchSize(batchSize), mCurrentBatch(0) {
		cv::glob(calibImagesPath + "/*.jpg", mImagePaths);
        if (mImagePaths.empty()) throw runtime_error("No calibration images found in " + calibImagesPath);
        cudaMalloc(&mDeviceInput, mBatchSize * mInputH * mInputW * 3 * sizeof(float));
        preprocessImages();
    }

    virtual ~Int8EntropyCalibrator() {
        cudaFree(mDeviceInput);
    }

    int getBatchSize() const noexcept override {
        return mBatchSize;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override {
        if (mCurrentBatch >= mBatches.size()) return false;
        cudaMemcpy(mDeviceInput, mBatches[mCurrentBatch].data(),
                   mBatches[mCurrentBatch].size() * sizeof(float), cudaMemcpyHostToDevice);
        bindings[0] = mDeviceInput;
        ++mCurrentBatch;
        return true;
    }

    const void* readCalibrationCache(size_t& length) noexcept override {
        return nullptr; // No cache for now
    }

    void writeCalibrationCache(const void* cache, size_t length) noexcept override {
        // No cache writing
    }

private:
    void preprocessImages() {
        for (const auto& path : mImagePaths) {
			cv::Mat img = cv::imread(path);
			cv::cvtColor(img, img, cv::COLOR_BGR2RGB); // 因为模型的输入数据是RGB的, 所以这里转换成RGB的
            int img_height = img.rows;
            int img_width = img.cols;
            int img_channels = img.channels();
            float scale_factor = min(static_cast<float>(mInputW) / static_cast<float>(img_width),
                            static_cast<float>(mInputH) / static_cast<float>(img_height));
            int img_new_w_unpad = img.cols * scale_factor;
            int img_new_h_unpad = img.rows * scale_factor;
            int pad_wl = round((mInputW - img_new_w_unpad - 0.01) / 2);		                   
            int pad_wr = round((mInputW - img_new_w_unpad + 0.01) / 2);
            int pad_ht = round((mInputH - img_new_h_unpad - 0.01) / 2);
            int pad_hb = round((mInputH - img_new_h_unpad + 0.01) / 2);
            cv::resize(img, img, cv::Size(img_new_w_unpad, img_new_h_unpad));
            cv::copyMakeBorder(img, img, pad_ht, pad_hb, pad_wl, pad_wr, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
			img.convertTo(img, CV_32F, 1.0 / 255.0);
			vector<float> chwImage(mInputH * mInputW * img_channels); 
			int channelSize = mInputH * mInputW;
			for (int c = 0; c < 3; ++c) { 
				for (int h = 0; h < mInputH; ++h) {
					for (int w = 0; w < mInputW; ++w) {
						chwImage[c * channelSize + h * mInputW + w] = img.at<cv::Vec3f>(h, w)[c];
					}
				}
			}
			mBatches.push_back(move(chwImage));
        }
    }

    vector<vector<float>> mBatches;
    vector<string> mImagePaths;
    int mInputH, mInputW, mBatchSize;
    int mCurrentBatch;
    void* mDeviceInput{nullptr};
};

bool quant_model(string onnxModelPath, string outEnginePath, string calibImagesPath, int maxBatchSize, string quant_type) {
	if (access(outEnginePath.c_str(), F_OK) != -1)
		spdlog::warn("Engine file {} exists! Will overwrite it with new conversion!", outEnginePath);
    
    // TensorRT components
	TRTLogger logger;
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // Parse ONNX model
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile(onnxModelPath.c_str(), 1)) {
        spdlog::error("Failed to parse {}", onnxModelPath);
        return false;
    }

    // Configure optimization profile
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();

    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
    config->addOptimizationProfile(profile);

    // Enable INT8 and set calibrator
    shared_ptr<Int8EntropyCalibrator> calibrator;
    if (quant_type == "int8"){
        calibrator = make_shared<Int8EntropyCalibrator>(calibImagesPath, input_dims.d[2], input_dims.d[3], maxBatchSize);
        config->setFlag(BuilderFlag::kINT8);
        config->setInt8Calibrator(calibrator.get());
    }else{
        config->setFlag(BuilderFlag::kFP16);
   }

    // Build engine
	spdlog::info("----> Start to serialize engine, it may take some time...");
    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr) {
        spdlog::error("Build engine failed");
        return false;
    }

    // Serialize and save model
    auto model_data = make_nvshared(engine->serialize());
    FILE* f = fopen(outEnginePath.c_str(), "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    spdlog::info("----> Build Done! Serialized engine file saved to {}", outEnginePath);
    return true;
}

int main(){
    string onnxModelPath = "your/onnx/model/path";
    string outEnginePath = "output/engine/path";
    string calibImagesPath = "your/calibrate/images/folder";
    int maxBatchSize = 8;
    string quant_type = "int8/fp16";
    quant_model(onnxModelPath, outEnginePath, calibImagesPath, maxBatchSize, quant_type);

    return 0;
}