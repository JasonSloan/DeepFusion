TensorRT基础

[课程地址](http://aipr.aijdjy.com/)

## 一. 环境

- 头文件（任务17，任务23）

  ```c++
  // 头文件所在目录："/root/miniconda3/envs/yolov8/lib/python3.7/site-packages/trtpy/trt8cuda112cudnn8/include/tensorRT"
  #include <NvInfer.h>
  #include <NvInferRuntime.h>
  // onnx解析器的头文件
  #include <NvOnnxParser.h>

  ```

- 检查TensorRT版本

  ```bash
  cd /root/miniconda3/envs/yolov8/lib/python3.7/site-packages/trtpy/trt8cuda112cudnn8/include/tensorRT
  cat NvInferVersion.h
  可以看到竖着的8 0 3 4，即版本号为8.0.34
  ```

  ​

## 二. 使用TensorRT硬代码构建模型Pipeline（任务17）

* 创建日志类Logger

  ```C++
  // 一般为第一步，定义日志输出级别，必须自定义且重写log函数，log函数可以什么都不写。
  // 这段代码不用太理解，知道什么意思就行，直接复制粘贴就能用
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
  class TRTLogger : public nvinfer1::ILogger{
  public:
      virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
          if(severity <= Severity::kINFO){
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
  } logger;
  ```

* 创建builder

  ```C++
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
  ```

* 创建config

  ```C++
  // config作用是将来可以指定batch-size，数据类型是int8还是float等..
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  ```

* 创建network

  ```C++
  // 这里的1指的是batch_size=1，TensorRT建议显示指定
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
  ```

* 向network中添加层

  ```C++
  const int num_input = 3;   // in_channel
  const int num_output = 2;  // out_channel
  float layer1_weight_values[] = {1.0, 2.0, 0.5, 0.1, 0.2, 0.5}; // 前3个给w1的rgb，后3个给w2的rgb 
  float layer1_bias_values[]   = {0.3, 0.8};
  //输入指定数据的名称、数据类型和完整维度，将输入层添加到网络
  nvinfer1::ITensor* input = network->addInput("image", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4(1, num_input, 1, 1)); // 数据维度NCHW
  nvinfer1::Weights layer1_weight = make_weights(layer1_weight_values, 6); 
  nvinfer1::Weights layer1_bias   = make_weights(layer1_bias_values, 2);
  //添加全连接层
  auto layer1 = network->addFullyConnected(*input, num_output, layer1_weight, layer1_bias);      // 注意对input进行了解引用
  //添加激活层 
  auto prob = network->addActivation(*layer1->getOutput(0), nvinfer1::ActivationType::kSIGMOID); // 注意更严谨的写法是*(layer1->getOutput(0)) 即对getOutput返回的指针进行解引用

  // 将我们需要的prob标记为输出
  network->markOutput(*prob->getOutput(0));
  ```

* 设置config

  ```C++
  config->setMaxWorkspaceSize(1 << 28);  // 1 << 28相当于在1后加28个0，等于2^28，等于256*1024*1024=256M。在python中也适用
  builder->setMaxBatchSize(1); // 推理时 batchSize = 1 
  ```

* 模型序列化并保存到本地

  ```C++
  nvinfer1::IHostMemory* model_data = engine->serialize();
  FILE* f = fopen("engine.trtmodel", "wb");
  fwrite(model_data->data(), 1, model_data->size(), f);
  fclose(f);
  ```

* 释放所有内存

  ```C++
  model_data->destroy();
  engine->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();
  ```

## 三. onnx模型转tensorrt模型（任务23） 

使用TensorRT的onnx解析器动态库加载onnx模型并保存成tensorrt模型

- 创建日志类Logger

  ```c++
  代码同上： 二. 使用TensorRT硬代码构建模型Pipeline（任务17）
  ```

- 创建builder

  ```c++
  nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(logger);
  ```

- 创建config

  ```c++
  // config作用是将来可以指定batch-size，数据类型是int8还是float等..
  nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
  ```

- 创建network

  ```c++
  // 这里的1指的是batch_size=1，TensorRT建议显示指定
  nvinfer1::INetworkDefinition* network = builder->createNetworkV2(1);
  ```

- 加载onnx模型

  ```c++
  // 通过onnxparser解析的结果会填充到network中，类似addConv的方式添加进去
  nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, logger);
  if(!parser->parseFromFile("demo.onnx", 1)){
    printf("Failed to parser demo.onnx\n");

    // 注意这里的几个指针还没有释放，是有内存泄漏的，后面考虑更优雅的解决
    return false;
  }
  ```

- 设置config

  ```c++
  int maxBatchSize = 10;
  config->setMaxWorkspaceSize(1 << 28);  // 1 << 28相当于在1后加28个0，等于2^28，等于256*1024*1024=256M。在python中也适用
  builder->setMaxBatchSize(maxBatchSize); // 推理时 batchSize = 1 。这里是为maxBatchSize是为了演示后面设置动态batch-size
  ```

- 设置动态batch（一般不需要）

  ```C++
  // 如果模型有多个输入，则必须多个profile
  auto profile = builder->createOptimizationProfile();
  auto input_tensor = network->getInput(0);
  int input_channel = input_tensor->getDimensions().d[1];

  // 配置输入的最小、最优、最大的范围
  profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4(1, input_channel, 3, 3));
  profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4(1, input_channel, 3, 3));
  profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4(maxBatchSize, input_channel, 5, 5));
  // 添加到配置
  config->addOptimizationProfile(profile);
  ```

- 生成engine

  ```C++
  nvinfer1::ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
  if(engine == nullptr){
    printf("Build engine failed.\n");
    return false;
  }
  ```

- 模型序列化并保存到本地

  ```c++
  nvinfer1::IHostMemory* model_data = engine->serialize();
  FILE* f = fopen("engine.trtmodel", "wb");
  fwrite(model_data->data(), 1, model_data->size(), f);
  fclose(f);
  ```

- 释放所有内存

  ```c++
  model_data->destroy();
  parser->destroy();
  engine->destroy();
  network->destroy();
  config->destroy();
  builder->destroy();
  ```

## 五. TensorRT推理模型Pipeline（任务18）

* 创建日志类Logger（同任务17）

* 加载模型文件

  ```C++
  vector<unsigned char> load_file(const string& file){
      // 这个函数的作用是读取一个文件，分配一个和文件大小相同大小的vector容器，将文件内容读取到容器中
      ifstream in(file, ios::in | ios::binary);  // 等同于ifstream in; in.open(file, ios::in | ios::binary);  in不是关键词
      if (!in.is_open())   // ifstream实例对象的方法.is_open()返回文件是否打开成功
          return {};

      in.seekg(0, ios::end);       // ifstream实例对象的方法.seekg()方法将文件读取指针移动到到文件结尾的位置
      size_t length = in.tellg();  // 返回相对于文件开头的偏移量

      std::vector<uint8_t> data;

      if (length > 0){
          in.seekg(0, ios::beg);  // 将文件指针移动到文件初始位置
          data.resize(length);

          in.read((char*)&data[0], length);  // 取data容器的第一个元素的地址，强转成char*类型的指针
      }
      in.close();
      return data;
  }
  auto engine_data = load_file("engine.trtmodel");
  ```

* 创建runtime

  ```C++
  // 执行推理前，需要创建一个推理的runtime接口实例。与builer一样，runtime需要logger：
  nvinfer1::IRuntime* runtime   = nvinfer1::createInferRuntime(logger);

  ```

* 反序列化创建engine

  ```C++
  // 将模型从读取到engine_data中，则可以对其进行反序列化以获得engine
  nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(engine_data.data(), engine_data.size());

  ```

* 创建ExecutionContext

  ```C++
  nvinfer1::IExecutionContext* execution_context = engine->createExecutionContext();

  ```

* 创建流(下面的操作全都是异步的)

  ```C++
  cudaStream_t stream = nullptr;
  // 创建CUDA流，以确定这个batch的推理是独立的
  cudaStreamCreate(&stream);

  ```

* 将数据从cpu搬运到gpu

  ```C++
  float input_data_host[] = {1, 2, 3};
  float* input_data_device = nullptr;

  float output_data_host[2];
  float* output_data_device = nullptr;
  cudaMalloc(&input_data_device, sizeof(input_data_host));
  cudaMalloc(&output_data_device, sizeof(output_data_host));
  cudaMemcpyAsync(input_data_device, input_data_host, sizeof(input_data_host), cudaMemcpyHostToDevice, stream);
  ```

* 用一个指针数组bindings指定input和output在gpu中的指针

  ```C++ 
  // 用一个指针数组指定input和output在gpu中的指针。
  float* bindings[] = {input_data_device, output_data_device};
  ```

* 执行推理

  ```C++
  bool success      = execution_context->enqueueV2((void**)bindings, stream, nullptr);  // void**是指向指针数组的指针
  ```

* 将数据搬运会cpu

  ```C++
  cudaMemcpyAsync(output_data_host, output_data_device, sizeof(output_data_host), cudaMemcpyDeviceToHost, stream);
  ```

* 等待流同步

  ```C++
  cudaStreamSynchronize(stream);
  ```

* 释放内存

  ```C++
  cudaStreamDestroy(stream);
  execution_context->destroy();
  engine->destroy();
  runtime->destroy();
  ```

## 六. python中使用ONNX修改模型（将前处理集成到模型中）（任务21）

* onnx计算图主要组成元素：

  * node：指代算子类型（conv，relu，etc。。）
  * initializer：指代算子对应的权重值
  * input：指代每个node的输入
  * output：指代每个node的输出

* **API**

  ```python
  - 首先python环境下：model = onnx.load("model.onnx")
  - 使用netron查看网络结构，网络结构中的每一层都有各种属性，根据属性可以对层进行操作
  - model.graph: 表示图结构，通常是我们netron看到的主要结构
  - model.graph.node: 表示图中的所有节点，数组，例如conv、bn等节点就是在这里的，通过input、output表示节点之间的连接关系
  - model.graph.initializer: 权重类的数据大都储存在这里
  - model.graph.initializer[index].raw_data:权重类的数据值（十六进制，需要使用np.frombuffer读取）
  - model.graph.input: 整个模型的输入储存在这里，表明哪个节点是输入节点，shape是多少
  - model.graph.output: 整个模型的输出储存在这里，表明哪个节点是输出节点，shape是多少
  - 对于anchorgrid类的常量数据，通常会储存在model.graph.node中，并指定类型为Constant，该类型节点在netron中可视化时不会显示出来
  ```

* onnx案例

  ```python
  from torch import nn
  import torch
  import onnx
  from onnx import helper
  import numpy as np
  model = onnx.load("yolov5s.onnx")
  print(model)	# 最好先把模型打印一下，看看每个节点的结构，或者使用netron看看模型每个节点的结构
  ```

  * 获取网络的某一层的权重值（普通节点，可以通过netron查看）

    ```python
    # 使用onnx获取网络的某一层的权重值（普通节点，可以通过netron查看）
    for item in model.graph.initializer:
        if item.name == "model.0.conv.weight":   # 获取第一层
            print("shape:", item.dims)
            weight = np.frombuffer(item.raw_data, dtype=np.float32).reshape(*item.dims)
    ```

  * 获取网络的某一层的权重值（特殊节点，可以通过netron查看）

    ```python
    # 使用onnx获取网络的某一层的权重值（特殊节点，可以通过netron查看）
    for item in model.graph.node:
        if item.op_type == "Constant":    # 获取后面的Add节点，类型为Constant的存储在model.graph.node中
            if "346" in item.output:
                t = item.attribute[0].t
                data = np.frombuffer(t.raw_data, dtype=np.float32).reshape(*t.dims)
                print(data.shape)
    ```

  * 修改某一层的权重值

    ```python
    # 使用onnx修改某一层的权重值
    for item in model.graph.node:
        if item.op_type == "Constant":
            if "362" in item.output:
                t = item.attribute[0].t
                # 这里为什么是int64，是因为看type(t)的时候观察到到t的datatype为TensorProto，然后去onnx-ml.proto搜索"TensorProto"查看TensorProto对应的数据类型为int64（枚举类）
                print(np.frombuffer(t.raw_data, dtype=np.int64))
                t.raw_data = np.array([100], dtype=np.int64).tobytes()
    onnx.save(model, "new.onnx")
    ```

  * 替换某一层的节点

    ```python
    # 使用onnx替换某一层的节点(本例子替换yolov5s中的reshape节点，只是将名字从"Reshape_235"替换成了"Reshape_235XXX")
    # 创建一个onnx节点
    new_node = helper.make_node("Reshape", ["394", "462"], ["401"],"Reshape_235XXX")
    for item in model.graph.node:
        if item.name == "Reshape_235":
            item.CopyFrom(new_node)   # 注意这里如果替换节点，赋值是不生效的，需要用item的实例函数CopyFrom
    onnx.save(model, "new.onnx")
    ```

  * 删除某一层的节点 （类似于删除链表中的一个节点）

    ```python
    # 使用onnx删除一个节点 （类似于删除链表中的一个节点）
    find_node_with_input = lambda name: [item for item in model.graph.node if name in item.input]  # 查找输入包含要删除节点的其他节点
    find_node_with_output = lambda name: [item for item in model.graph.node if name in item.output]  # 查找输出包含要删除节点的其他节点
    remove_nodes = []
    for item in model.graph.node:
        if item.name == "Transpose_219":
            # print(item)
            # 上一个节点的输出是当前节点的输入，当前节点的输出是下一个节点的输入
            # print(find_node_with_output(item.input[0]))
            # print(find_node_with_input(item.output[0]))
            prev_node = find_node_with_output(item.input[0])
            next_node = find_node_with_input(item.output[0])
            next_node[0].input[0] = prev_node[0].output[0]
            remove_nodes.append(item)
    for item in remove_nodes[::-1]:     # 为什么要先存起来再删除节点，就是在python中不要边遍历边删除会出现bug
        model.graph.node.remove(item)
    onnx.save(model, "new.onnx")
    ```

  * 修改整个模型输入输出的batchsize

    ```python
    # 使用onnx修改batch-size
    static_batch_size = 10
    # print(model.graph.input)
    new_input = helper.make_tensor_value_info("images", 1, [static_batch_size, 3, 640, 640])
    model.graph.input[0].CopyFrom(new_input)
    new_output = helper.make_tensor_value_info("output0", 1, [static_batch_size, 25200, 85])
    model.graph.output[0].CopyFrom(new_output)
    onnx.save(model, "new.onnx")
    ```

  * 使用onnx将预处理代码变成网络结构添加到网络中

    ```python
    # 使用onnx将预处理代码变成网络结构添加到网络中
    # step1:构建预处理网络并转成onnx
    class Preprocess(nn.Module):
        def __init__(self):
            super().__init__()
            self.mean = torch.randn(1, 1, 1, 3)
            self.std = torch.rand(1, 1, 1, 3)
        def forward(self, x):
            x = x.float()
            x = (x / 255 - self.mean) / self.std
            x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
            return x
    pre = Preprocess()
    torch.onnx.export(
        pre, (torch.zeros((1, 640, 640, 3), dtype=torch.uint8)), "pre.onnx")
    # step2:加载预处理网络，将yolov5s中的image为输入的节点，修改为pre_onnx的输出节点
    pre_onnx = onnx.load("pre.onnx")
    for item in pre_onnx.graph.node:
        # 修改当前节点的名字
        item.name = f"pre/{item.name}"
        # 修改当前节点的输入的名字
        for index in range(len(item.input)):
            item.input[index] = f"pre/{item.input[index]}"
        # 修改当前节点的输出的名字
        for index in range(len(item.output)):
            item.output[index] = f"pre/{item.output[index]}" 
    # 修改原模型的第一层的输入节点名字改为pre_onnx的输出节点的名字
    for item in model.graph.node:
        if item.name == "Conv_0":
            item.input[0] = "pre/" + pre_onnx.graph.output[0].name
    # setp3: 把pre_onnx的node全部放到yolov5s的node中
    for item in pre_onnx.graph.node:
        model.graph.node.append(item)    # 这里我看了model.graph.node这个转成列表后append不是在网络末尾追加吗，但是这个是将预处理加入到网络首部中，不应该是insert吗
    # 答：其实model.graph.node这个列表里的元素可以使完全乱序的，因为这个列表里的每个元素都标记好了他的输入是叫啥名，输出时叫啥名，所以无论在列表中顺序怎么乱，最终都能按照名字一一对应上
    # step4: 把pre_onnx的输入名称作为yolov5s的input名称
    input_name = "pre/" + pre_onnx.graph.input[0].name
    model.graph.input[0].CopyFrom(pre_onnx.graph.input[0])
    model.graph.input[0].name = input_name
    onnx.save(model, "new.onnx")
    ```

## 七. 正确导出ONNX注意事项（任务22）

![img](pics/export_onnx_properly.jpg)

**如果模型是被nn.DataParallel()或者nn.DataDistributedParalle()包裹的，需要去掉这种并行分布式的代码**

##  八. 更新onnx-tensorrt解析器（任务24）

- 如果想搞按照课程来吧，只要TensorRT版本是8.0以上9.0一下目前就不需要更新onnx-tensorrt解析器，查看TensorRT版本可以使用trtpy get-env
- 就直接使用trtpy中tensorrt课程系列中的onnx-tensorrt和onnx源码吧，不要自己搞了。。。 
