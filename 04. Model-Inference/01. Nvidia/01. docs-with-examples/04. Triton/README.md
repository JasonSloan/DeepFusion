## 一、概念

- 必要条件

  docker和nvidia-docker已安装

- 相关链接

  <https://github.com/triton-inference-server/server>

  <https://github.com/triton-inference-server/perf_analyzer>

- Docker镜像

  - Triton提供两类有关的docker镜像：
    - `nvcr.io/nvidia/tritonserver:25.02-py3`：可通过配置将模型部署上，直接启动后可对外提供HTTP(S) API、gRPC API、Native C API。
    - `nvcr.io/nvidia/tritonserver:25.02-py3-sdk`：启动后可在容器内通过工具`perf_analyzer`对前者已部署的模型进行基准测试。前者需自行编写前后处理代码，因此使用后者的`perf_analyzer`进行基准测试只能测试纯推理指标，不包含前后处理。

## 二、下载镜像并启动triton server测试示例

克隆仓库并进入示例目录

```
git clone -b r25.02 https://github.com/triton-inference-server/server.git
cd server/docs/examples
```

下载模型

```
./fetch_models.sh
```

启动triton server

```
docker run --gpus=1 --rm --net=host -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:25.02-py3 tritonserver --model-repository=/models
```

在新console执行客户端命令进行测试

```
docker run -it --rm --gpus=1 --net=host nvcr.io/nvidia/tritonserver:25.02-py3-sdk /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
```

得到结果：

```
Image '/workspace/images/mug.jpg':
    15.346230 (504) = COFFEE MUG
    13.224326 (968) = CUP
    10.422965 (505) = COFFEEPOT
```

## 三、测试自己的模型

### 1. 启动并进入triton server容器

```
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v $(pwd)/model_repository:/models nvcr.io/nvidia/tritonserver:25.02-py3
```

- **模型目录格式**
  宿主机中的`model_repository`目录与容器中的`/models`目录映射，目录`model_repository`下的模型配置应当遵循如下格式：

```
├── client.py
├── yolo11-onnx
│   ├── 1
│   │   └── model.onnx
│   └── config.pbtxt
└── yolo11-tensorrt
    ├── 1
    │   └── model.plan
    └── config.pbtxt
```

上述格式代表告诉triton要加载两个模型，一个是`model.onnx`，一个是`model.plan`。

### 2. 在triton server中编译自己的模型

```
/usr/src/tensorrt/bin/trtexec --onnx=yolo11-onnx/1/model.onnx --saveEngine=yolo11-tensorrt/1/model.plan --useCudaGraph  --fp16
```

### 3. 编写`config.pbtxt`

示例：

```
name: "yolo11-tensorrt"
platform: "tensorrt_plan"
max_batch_size : 16
input [
  {
    name: "images"
    data_type: TYPE_FP32
    format: FORMAT_NCHW
    dims: [3, 384, 640]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [5040, 5]
  }
]

dynamic_batching { }

instance_group [
    {
      count: 2
      kind: KIND_GPU
    }
]
```

- 名词解释

  - `name`：要与`config.pbtxt`的父目录的命名相同。

  - `platform`：使用的推理后端，本例中使用`tensorrt_plan`。

  - `max_batch_size`：最大批量大小。

  - ```
    input
    ```

    ：输入相关配置。

    - `name`：与onnx中模型的输入名称保持一致。
    - `data_type`：输入数据类型。
    - `format`：数据维度布局。
    - `dims`：输入维度。

  - ```
    output
    ```

    ：输出相关配置。

    - `name`：与onnx中模型的输入名称保持一致。
    - `data_type`：输入数据类型。
    - `dims`：输入维度。

  - `dynamic_batching`：允许triton动态组成批量推理。

  - `instance_group`：并发模型实例。

### 4. 在triton server的容器中启动triton模型对外服务

在triton server的容器中执行：

```
tritonserver --model-repository=/models
```

### 5. 启动基准测试容器

新开一个console并执行：

```
docker run -it --rm --gpus all --net host nvcr.io/nvidia/tritonserver:25.02-py3-sdk
```

### 6. 开始基准测试

在基准测试容器中执行：

```
perf_analyzer -m yolo11-tensorrt -b 2  --concurrency-range 2:16:2
```

- 参数解释
  - `-m`：指定要使用的模型名称。
  - `-b`：使用的batch size大小。
  - `--concurrency-range 2:16:2`：并发实例数，从2个实例开始到16个实例结束，每次增加2个实例。

### 7. 基准测试结果参数解释

- 测量设置相关

  - `Batch size`：批处理大小 → 每个推理请求处理单个图像。
  - `Service Kind`：服务类型 → 此测试在Triton推理服务器上执行。
  - `Measurement window`：测量窗口 → 性能在5秒的时间间隔内进行测量。
  - `Using synchronous calls`：使用同步调用 → 客户端一次发送一个请求并等待响应 。

- 请求相关

  - `Request concurrency`：请求并发数 → 客户端一次只发送一个请求。
  - `Request count`：请求计数 → 测试期间发出的推理请求总数。
  - `Throughput`：吞吐量 → 服务器每秒处理的推理次数约为138.3次。
  - `Avg latency`：平均延迟 → 单个推理请求的平均处理时间。

- 延迟分布相关

  - ```
    Latency breakdown
    ```

    ：延迟分布 → 不同百分比请求的处理时间分布：

    - `p50 (4693 µs, ~4.7 ms)`：第50百分位延迟（4693微秒，约4.7毫秒）→ 50%的请求处理时间 ≤4.7毫秒。
    - `p90 (6609 µs, ~6.6 ms)`：第90百分位延迟（6609微秒，约6.6毫秒）→ 90%的请求处理时间 ≤6.6毫秒。
    - `p95 (8161 µs, ~8.2 ms)`：第95百分位延迟（8161微秒，约8.2毫秒）→ 95%的请求处理时间 ≤8.2毫秒。
    - `p99 (10953 µs, ~10.95 ms)`：第99百分位延迟（10953微秒，约10.95毫秒）→ 99%的请求处理时间 ≤10.95毫秒。

  - `Avg HTTP time`：平均HTTP时间 → HTTP通信所花费的时间。

  - `Send/recv time`：发送/接收时间 → 网络传输时间。

  - `Response wait`：响应等待时间 → 等待推理结果的时间

- 推理执行相关

  - `Inference count`：推理计数 → 服务器成功处理了2981次推理。

  - `Execution count`：执行计数 → 模型执行次数与推理计数一致。

  - `Successful request count`：成功请求计数 → 没有失败的推理请求。

  - ```
    Avg request latency
    ```

    ：平均请求延迟 → 延迟的详细分解：

    - `Overhead`：开销 → Triton内部处理开销。
    - `Queue`：排队时间 → 在Triton调度队列中等待的时间。
    - `Compute input`：输入计算时间 → 准备输入张量所花费的时间。
    - `Compute infer`：推理计算时间 → 实际模型推理所花费的时间（1795微秒，1.795毫秒）。
    - `Compute output`：输出计算时间 → 准备输出张量所花费的时间。

- 其他相关

  - `Concurrency`：并发数 → 每次仅处理一个请求。
  - `Throughput`：吞吐量 → 服务器每秒处理的推理次数。
  - `Latency`：延迟 → 从客户端视角来看的总平均延迟。