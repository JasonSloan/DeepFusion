拉取代码：

trtpy get-series tensorrt-intergrate 

cd tensorrt-intergrate 

trtpy change-proj 1.2





main.cpp文件代码注释：

```C++
#include <cuda_runtime.h>
#include <NvInfer.h>
#include <onnx-tensorrt/NvOnnxParser.h>
#include <NvInferRuntime.h>

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>     // 智能指针
#include <functional> // 这个是哪个API用的？
#include <unistd.h>
#include <opencv2/opencv.hpp>

using namespace std;

// 固定代码
#define checkRuntime(op) __check_cuda_runtime((op), #op, __FILE__, __LINE__)
// 固定代码
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

// coco数据集的labels，关于coco：https://cocodataset.org/#home
static const char *cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"};

// 固定代码
inline const char *severity_string(nvinfer1::ILogger::Severity t)
{
    switch (t)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        return "internal_error";
    case nvinfer1::ILogger::Severity::kERROR:
        return "error";
    case nvinfer1::ILogger::Severity::kWARNING:
        return "warning";
    case nvinfer1::ILogger::Severity::kINFO:
        return "info";
    case nvinfer1::ILogger::Severity::kVERBOSE:
        return "verbose";
    default:
        return "unknow";
    }
}

// hsv转bgr，如果不保存结果图到本地，就不用
static std::tuple<uint8_t, uint8_t, uint8_t> hsv2bgr(float h, float s, float v)
{
    const int h_i = static_cast<int>(h * 6);
    const float f = h * 6 - h_i;
    const float p = v * (1 - s);
    const float q = v * (1 - f * s);
    const float t = v * (1 - (1 - f) * s);
    float r, g, b;
    switch (h_i)
    {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
        r = v;
        g = p;
        b = q;
        break;
    default:
        r = 1;
        g = 1;
        b = 1;
        break;
    }
    return make_tuple(static_cast<uint8_t>(b * 255), static_cast<uint8_t>(g * 255), static_cast<uint8_t>(r * 255));
}
// 随机产生一个颜色，画框用的。如果不保存结果图到本地，就不用
static std::tuple<uint8_t, uint8_t, uint8_t> random_color(int id)
{
    float h_plane = ((((unsigned int)id << 2) ^ 0x937151) % 100) / 100.0f;
    ;
    float s_plane = ((((unsigned int)id << 3) ^ 0x315793) % 100) / 100.0f;
    return hsv2bgr(h_plane, s_plane, 1);
}
// 固定代码
class TRTLogger : public nvinfer1::ILogger
{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const *msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
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
            if (severity == Severity::kWARNING)
            {
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if (severity <= Severity::kERROR)
            {
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else
            {
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
};

// 通过智能指针管理nv返回的指针参数
// 内存自动释放，避免泄漏
template <typename _T>
shared_ptr<_T> make_nvshared(_T *ptr)
{
    return shared_ptr<_T>(ptr, [](_T *p)
                          { p->destroy(); }); // 为什么指定一个lambda函数，是因为在在cuda_runtime的API中，释放内存使用->destroy()而不是delete []
}

bool exists(const string &path)
{ // 注意string得用std命名空间
#ifdef _WIN32
    return ::PathFIleExistsA(path.c_str()); // 在windows中是PathFIleExistsA这个函数判断文件存在
#else
    return access(path.c_str(), R_OK) == 0; // 在linux中是access这个函数判断文件存在，access包含在unistd.h头文件里
#endif // 两个函数都只接受char*类型不是string类型，所以要转成c_str()
}

bool build_model()
{
    if (exists("yolov5s.trtmodel"))
    {
        printf("yolov5s.trtmodel has exists.\n"); // 如果已经构建好了模型直接返回
        return true;
    }
    TRTLogger logger;

    // 这是基本需要的组件
    auto builder = make_nvshared(nvinfer1::createInferBuilder(logger));
    auto config = make_nvshared(builder->createBuilderConfig());
    auto network = make_nvshared(builder->createNetworkV2(1));

    // 通过onnxparser解析器解析的结果会填充到network中，类似addConv的方式添加进去
    auto parser = make_nvshared(nvonnxparser::createParser(*network, logger));
    if (!parser->parseFromFile("yolov5s.onnx", 1))
    {
        printf("Failed to parse yolov5s.onnx\n");
        return false; // 这里存在内存泄露的问题
    }

    // 设置workspace大小
    int maxBatchSize = 10;
    printf("Workspace Size = %.2f MB\n", (1 << 28) / 1024.0f / 1024.0f);
    config->setMaxWorkspaceSize(1 << 28);

    // 设置动态输入batch
    auto profile = builder->createOptimizationProfile();
    auto input_tensor = network->getInput(0);
    auto input_dims = input_tensor->getDimensions();
    input_dims.d[0] = 1;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims); // 设置最小输入尺寸
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims); // 设置最优输出尺寸
    input_dims.d[0] = maxBatchSize;
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims); // 设置最大输入尺寸
    config->addOptimizationProfile(profile);

    // 构建engine
    auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
    if (engine == nullptr)
    {
        printf("Build engine failed.\n");
        return false;
    }

    // 本地存储文件
    auto model_data = make_nvshared(engine->serialize());
    FILE *f = fopen("yolov5s.trtmodel", "wb");
    fwrite(model_data->data(), 1, model_data->size(), f);
    fclose(f);

    printf("Build Done.\n");
    return true;
}

vector<unsigned char> load_file(const string &file)
{
    ifstream in(file, ios::in | ios::binary); // 相当于python中的"rb"模式打开
    if (!in.is_open())
        return {};
    in.seekg(0, ios::end);      // 将文件指针移到文件末尾，偏移值为0
    size_t length = in.tellg(); // 计算文件大小
    vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, ios::beg);             // 将文件指针移到文件开头，偏移值为0
        data.resize(length);               // 将vector容器resize到文件大小
        in.read((char *)&data[0], length); // 取data[0]的地址强转成char*类型，将in中的内容读取到data中，读取长度为length
    }
    in.close();
    return data;
}
void inference()
{
    // 加载模型并反序列化
    TRTLogger logger;
    auto engine_data = load_file("yolov5s.trtmodel"); // 从本地加载模型
    auto runtime = make_nvshared(nvinfer1::createInferRuntime(logger));
    auto engine = make_nvshared(runtime->deserializeCudaEngine(engine_data.data(), engine_data.size())); // 将模型反序列化给engine
    if (engine == nullptr)
    {
        printf("Deserialize cuda engine failed.\n");
        runtime->destroy();
        return;
    }
    if (engine->getNbBindings() != 2)
    {
        printf("你的onnx导出有问题，必须是1个输入和1个输出，你这明显有：%d个输出.\n", engine->getNbBindings() - 1);
        return;
    }

    // 创建流和上下文管理器
    cudaStream_t stream = nullptr; // 创建流
    checkRuntime(cudaStreamCreate(&stream));
    auto execution_context = make_nvshared(engine->createExecutionContext()); // 创建上下文管理器

    // 在cpu和gpu上分配内存
    int input_batch = 1;
    int input_channel = 3;
    int input_width = 640;
    int input_height = 640;
    int input_numel = input_batch * input_channel * input_width * input_height; // 计算总共多少像素
    float *input_data_host = nullptr;                                           // 因为是要在gpu上进行计算，所以要转成float类型
    float *input_data_device = nullptr;
    checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
    checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

    // letter box
    auto image = cv::imread("car.jpg");
    // 通过双线性插值对图像做resize
    // 先计算缩放比率
    float scale_x = input_width / (float)image.cols;
    float scale_y = input_height / (float)image.rows;
    float scale = min(scale_x, scale_y);
    // 构建仿射变换矩阵
    float i2d[6], d2i[6];
    i2d[0] = scale;
    i2d[1] = 0;
    i2d[2] = (-scale * image.cols + input_width + scale - 1) * 0.5;
    i2d[3] = 0;
    i2d[4] = scale;
    i2d[5] = (-scale * image.rows + input_height + scale - 1) * 0.5;
    cv::Mat m2x3_i2d(2, 3, CV_32F, i2d);           // 将构建好的数组转成cv::Mat矩阵
    cv::Mat m2x3_d2i(2, 3, CV_32F, d2i);           // 构建一个未初始化的cv::Mat矩阵
    cv::invertAffineTransform(m2x3_i2d, m2x3_d2i); // 使用opencv自带函数计算反仿射变换矩阵

    cv::Mat input_image(input_height, input_width, CV_8UC3); // 构建一个空矩阵用来接仿射变换计算完的图
    // 仿射变换：INTER_LINEAR就是双线性插值，多出的部分用常数114填充，相当于直接letterbox了
    cv::warpAffine(image, input_image, m2x3_i2d, input_image.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar::all(114));
    cv::imwrite("input-image.jpg", input_image);

    // 将在内存中以BGRBGRBGR...的存储形式变为RRRGGGBBB...的存储形式，且进行归一化
    /*  pimage向phost的变换过程：
          image_area      image_area      image_area
        [rrrrrrrrrrrrr...][gggggggggggggg..][bbbbbbbbbbbbbb..]
            phost_b             phost_g         phost_r
                                ↑
        [[bgr][bgr][bgr][bgr][bgr][bgr][bgr][bgr][bgr]...]
                             pimage
    */
    int image_area = input_image.cols * input_image.rows;
    unsigned char *pimage = input_image.data;
    float *phost_b = input_data_host + image_area * 0;
    float *phost_g = input_data_host + image_area * 1;
    float *phost_r = input_data_host + image_area * 2;
    for (int i = 0; i < image_area; ++i, pimage += 3)
    {
        // 这里实现了bgr->rgb的变换
        *phost_r++ = pimage[0] / 255.0f; // 每次循环取出的pimage[0]是b通道的值
        *phost_g++ = pimage[1] / 255.0f; // 每次循环取出的pimage[1]是g通道的值
        *phost_b++ = pimage[2] / 255.0f; // 每次循环取出的pimage[2]是r通道的值
    }
    // 将cpu上的图片输入拷贝到gpu上
    checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream));

    auto output_dims = engine->getBindingDimensions(1);              // 获得输出维度：[N, 25200, 85]，25200=(80*80+40*40+20*20)*3
    int output_numbox = output_dims.d[1];                            // 25200边框数
    int output_numprob = output_dims.d[2];                           // 85:xywh+conf+80类别
    int num_classes = output_numprob - 5;                            // 类别数
    int output_numel = input_batch * output_numbox * output_numprob; // 总数
    float *output_data_host = nullptr;
    float *output_data_device = nullptr;
    checkRuntime(cudaMallocHost(&output_data_host, sizeof(float) * output_numel)); // 分配cpu内存
    checkRuntime(cudaMalloc(&output_data_device, sizeof(float) * output_numel));   // 分配gpu内存

    auto input_dims = engine->getBindingDimensions(0);
    input_dims.d[0] = input_batch; // 明确输入batch由-1变为1

    execution_context->setBindingDimensions(0, input_dims);                          // 固定代码
    float *bindings[] = {input_data_device, output_data_device};                     // 固定代码
    bool success = execution_context->enqueueV2((void **)bindings, stream, nullptr); // 执行推理
    // 将输出数据从gpu拷贝到cpu上
    checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream));
    checkRuntime(cudaStreamSynchronize(stream));

    // decode box：从不同尺度下的预测狂还原到原输入图上(包括:预测框，类被概率，置信度）
    vector<vector<float>> bboxes;      // 初始化变量bboxes:[[x1, y1, x2, y2, label, conf], [x1, y1, x2, y2, label, conf]...]
    float confidence_threshold = 0.25; // 置信度
    float nms_threshold = 0.5;         // iou阈值
    for (int i = 0; i < output_numbox; ++i)
    {
        float *ptr = output_data_host + i * output_numprob; // 每次偏移85
        float objness = ptr[4];                             // 获得置信度
        if (objness < confidence_threshold)
            continue;

        float *pclass = ptr + 5;                                        // 获得类别开始的地址
        int label = max_element(pclass, pclass + num_classes) - pclass; // 获得概率最大的类别
        float prob = pclass[label];                                     // 获得类别概率最大的概率值
        float confidence = prob * objness;                              // 计算后验概率
        if (confidence < confidence_threshold)
            continue;

        // 中心点、宽、高
        float cx = ptr[0];
        float cy = ptr[1];
        float width = ptr[2];
        float height = ptr[3];

        // 预测框
        float left = cx - width * 0.5;
        float top = cy - height * 0.5;
        float right = cx + width * 0.5;
        float bottom = cy + height * 0.5;

        // 对应图上的位置
        float image_base_left = d2i[0] * left + d2i[2];                                                                     // x1
        float image_base_right = d2i[0] * right + d2i[2];                                                                   // x2
        float image_base_top = d2i[0] * top + d2i[5];                                                                       // y1，这里实际应该是d2i[4] * top+d2i[5];
        float image_base_bottom = d2i[0] * bottom + d2i[5];                                                                 // y2，这里实际应该是d2i[4] * bottom+d2i[5];
        bboxes.push_back({image_base_left, image_base_top, image_base_right, image_base_bottom, (float)label, confidence}); // 放进bboxes中
    }
    printf("decoded bboxes.size = %d\n", bboxes.size());

    // nms非极大抑制
    // 通过比较索引为5(confidence)的值来将bboxes所有的框排序
    std::sort(bboxes.begin(), bboxes.end(), [](vector<float> &a, vector<float> &b)
              { return a[5] > b[5]; });
    std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags
    std::vector<vector<float>> box_result;         // box_result用来接收经过nms后保留的框
    box_result.reserve(bboxes.size());             // 给box_result保留至少bboxes.size()个存储数据的空间

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

    for (int i = 0; i < bboxes.size(); ++i)
    {
        if (remove_flags[i])
            continue; // 如果已经被标记为需要移除，则continue

        auto &ibox = bboxes[i];        // 获得第i个box
        box_result.emplace_back(ibox); // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
        for (int j = i + 1; j < bboxes.size(); ++j)
        { // 遍历剩余框，与box_result中的框做iou
            if (remove_flags[j])
                continue; // 如果已经被标记为需要移除，则continue

            auto &jbox = bboxes[j]; // 获得第j个box
            if (ibox[4] == jbox[4])
            { // 如果是同一类别才会做iou
                // class matched
                if (iou(ibox, jbox) >= nms_threshold) // iou值大于阈值，将该框标记为需要remove
                    remove_flags[j] = true;
            }
        }
    }
    printf("box_result.size = %d\n", box_result.size());

    for (int i = 0; i < box_result.size(); ++i)
    { // 画框并本地存储
        auto &ibox = box_result[i];
        float left = ibox[0];
        float top = ibox[1];
        float right = ibox[2];
        float bottom = ibox[3];
        int class_label = ibox[4];
        float confidence = ibox[5];
        cv::Scalar color;
        tie(color[0], color[1], color[2]) = random_color(class_label);
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), color, 3);

        auto name = cocolabels[class_label];
        auto caption = cv::format("%s %.2f", name, confidence);
        int text_width = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left - 3, top - 33), cv::Point(left + text_width, top), color, -1);
        cv::putText(image, caption, cv::Point(left, top - 5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    cv::imwrite("image-draw.jpg", image);

    checkRuntime(cudaStreamDestroy(stream)); // 固定代码
    checkRuntime(cudaFreeHost(input_data_host));
    checkRuntime(cudaFreeHost(output_data_host));
    checkRuntime(cudaFree(input_data_device));
    checkRuntime(cudaFree(output_data_device));
}

int main()
{
    if (!build_model())
        return -1;
    inference();
    return 0;
}
```







