TensorRT基础

[课程地址](http://aipr.aijdjy.com/)

## 九. 自定义插件（任务26）

还需要在学习一下，还没搞懂

- step1：使用pytorch自定义算子并导出为onnx格式


```python
import torch
import torch.nn as nn
import torch.onnx
import torch.autograd


# 在这里self.param实际上并没有被用到，只是为了演示
class MYSELU(nn.Module):
    def __init__(self, n):
        super().__init__()
        # nn.Parameter包裹的参数会计算梯度并更新数组
        self.param = nn.parameter.Parameter(torch.arange(n).float())

    def forward(self, x):
        # 这里必须调用apply，实际上调用的是MYSELUImpl的forward，并把x和self.param传递给x和p
        return MYSELUImpl.apply(x, self.param)


# 插件必须继承torch.autograd.Function

class MYSELUImpl(torch.autograd.Function):
    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions

    # 必须实现symbolic方法，且必须为静态方法
    @staticmethod
    def symbolic(g, x, p):  # g为固定的(代表onnx的graph)，x和p为MYSELU的x, self.param
        print("==================================call symbolic")
        # 必须返回onnx的operator即g.op。第一个参数为“MYSELU"在生成onnx节点后，在拓扑图中该节点type叫MYSELU,x和p代表是MYSELU的x, self.param
        return g.op("MYSELU", x, p,
                    # 该行只是演示，实际没有需要可以直接去掉。意思是添加该行代表添加一个Constant节点，虽然在netron中看不到
                    g.op("Constant", value_t=torch.tensor([3, 2, 1], dtype=torch.float32)),
                    attr1_s="这是字符串属性",  # 为插件的节点添加字符串属性：字符串使用attr1_s
                    attr2_i=[1, 2, 3],  # 为插件的节点添加整数属性：整数使用attr2_i
                    attr3_f=222.  # 为插件的节点添加浮点数属性：浮点数使用attr3_f
                    )

    # 必须实现forward方法，且必须为静态方法
    @staticmethod
    def forward(ctx, x, p):  # 第一个参数固定的为ctx，x和p为MYSELU的x, self.param
        return x * 1 / (1 + torch.exp(-x))  # 注意这里的所有运算操作都不会被跟踪（不会在onnx中生成新的节点）


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1))
        self.myselu = MYSELU(3)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv(x)
        x = self.myselu(x)
        return x


model = Model().eval()

input = torch.randn([1, 3, 64, 64], dtype=torch.float32)

output = model(input)

dummy = input

# 导出插件要设置enable_onnx_checker=False， 但是导出之前要将pytorch版本回退，否则enable_onnx_checker=False会报错，测过torch版本回退到1.10是没问题的
torch.onnx.export(model,
                  # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
                  (dummy,),

                  # 储存的文件路径
                  "demo.onnx",

                  # 打印详细信息
                  verbose=True,

                  # 为输入和输出节点指定名称，方便后面查看或者操作
                  input_names=["image"],
                  output_names=["output"],

                  # 对于插件，需要指定这一句固定代码
                  operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,

                  # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
                  opset_version=11,

                  # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
                  # 通常，我们只设置batch为动态，其他的避免动态
                  dynamic_axes={
                      "image": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"},
                  },

                  # 对于插件，需要禁用onnx检查
                  enable_onnx_checker=False
                  )

print("Done.!")

```

- step2：使用杜老师的easy-plugin.cu文件（注意，与此同时，也要使用杜老师的onnx-tensorrt和onnx两个文件夹下的源码，杜老师的这个源码是在官网源码的基础上增加了一些文件，增加了这些文件才能调用easy-plugin.cu）
```C++
// 这个onnxplugin.hpp是杜老师写的，不是官方带的，所以得用trtpy课程系列里的onnx-tensorrt源码
#include "onnx-tensorrt/onnxplugin.hpp"

using namespace ONNXPlugin;

static __device__ float sigmoid(float x) {
    return 1 / (1 + expf(-x));  // 定义sigmoid函数，该函数因为是在gpu上执行的，所以加device
}

static __global__ void MYSELU_kernel_fp32(const float* x,
                                          float* output,
                                          int edge) {
    int position = threadIdx.x + blockDim.x * blockIdx.x;
    if (position >= edge)
        return;

    output[position] = x[position] * sigmoid(x[position]);  // selu激活函数 = x * sigmoid(x)
}

class MYSELU : public TRTPlugin {  // 必须继承TRTPlugin
   public:
    SetupPlugin(MYSELU);  // 固定代码

    virtual void config_finish() override {  // config_finish函数这段为固定代码
        printf("\033[33minit MYSELU config: %s\033[0m\n",
               config_->info_.c_str());
        printf("weights count is %d\n", config_->weights_.size());
    }

    int enqueue(const std::vector<GTensor>& inputs,
                std::vector<GTensor>& outputs,
                const std::vector<GTensor>& weights,
                void* workspace,
                cudaStream_t stream) override {
        int n = inputs[0].count();  // 计算输入有多少个数值
        const int nthreads = 512;
        int block_size = n < nthreads ? n : nthreads;  // 三目运算，如果n<512就用n，否则用512
        int grid_size = (n + block_size - 1) / block_size;  // 算用多少个grid

        MYSELU_kernel_fp32<<<grid_size, block_size, 0, stream>>>(
            inputs[0].ptr<float>(), outputs[0].ptr<float>(),
            n);  //  调用核函数，参数固定
        return 0;
    }
};

RegisterPlugin(MYSELU);  // 固定代码
```



  ```

- 按照“五. TensorRT模型推理Pipeline（任务18）”的代码编写main.cpp推理代码，即可自动调用核函数（因为main.cpp中的Nvinfer.h会调用onnx-tensorrt下杜老师增加的代码文件，杜老师增加的代码文件会调用核函数代码，从而实现调用插件）

  ```
# 十. 插件的封装(任务27)

1. 自定义算子的python实现代码(gen-onnx.py)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.autograd
import os
import json
from torch.onnx import register_custom_op_symbolic

class MYSELUImpl(torch.autograd.Function):

    # reference: https://pytorch.org/docs/1.10/onnx.html#torch-autograd-functions
    @staticmethod
    def symbolic(g, x, p):
        print("==================================call symbolic")
        return g.op("Plugin", x, p,           # 因为杜老师已经写好了一段专门解析的代码（"onnx-tensorrt/onnxplugin.hpp"中的TRTPlugin类），所以名字统一必须叫Plugin，会根据name_s属性获取该插件真正的名字
            g.op("Constant", value_t=torch.tensor([3, 2, 1], dtype=torch.float32)),
            name_s="MYSELU",                  # 此处写插件真正的名字         
            info_s=json.dumps(
                dict(
                    attr1_s="这是字符串属性", 
                    attr2_i=[1, 2, 3], 
                    attr3_f=222
                ), ensure_ascii=False
            )
        )

    @staticmethod
    def forward(ctx, x, p):
        return x * 1 / (1 + torch.exp(-x))


class MYSELU(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.param = nn.parameter.Parameter(torch.arange(n).float())

    def forward(self, x):
        return MYSELUImpl.apply(x, self.param)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        self.myselu = MYSELU(3)
        self.conv.weight.data.fill_(1)
        self.conv.bias.data.fill_(0)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.myselu(x)
        x[..., 0] = x[..., 0].sigmoid()
        return x


# 这个包对应opset11的导出代码，如果想修改导出的细节，可以在这里修改代码
# import torch.onnx.symbolic_opset11
print("对应opset文件夹代码在这里：", os.path.dirname(torch.onnx.__file__))

model = Model().eval()
input = torch.tensor([
    # batch 0
    [
        [1,   1,   1],
        [1,   1,   1],
        [1,   1,   1],
    ],
        # batch 1
    [
        [-1,   1,   1],
        [1,   0,   1],
        [1,   1,   -1]
    ]
], dtype=torch.float32).view(2, 1, 3, 3)

output = model(input)
print(f"inference output = \n{output}")

dummy = torch.zeros(1, 1, 3, 3)
torch.onnx.export(
    model, 

    # 这里的args，是指输入给model的参数，需要传递tuple，因此用括号
    (dummy,), 

    # 储存的文件路径
    "workspace/demo.onnx", 

    # 打印详细信息
    verbose=False, 

    # 为输入和输出节点指定名称，方便后面查看或者操作
    input_names=["image"], 
    output_names=["output"], 

    # 这里的opset，指，各类算子以何种方式导出，对应于symbolic_opset11
    opset_version=11, 

    # 表示他有batch、height、width3个维度是动态的，在onnx中给其赋值为-1
    # 通常，我们只设置batch为动态，其他的避免动态
    dynamic_axes={
        "image": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },

    # 对于插件，需要禁用onnx检查
    enable_onnx_checker=False
)

import onnx
import onnx.helper as helper

model = onnx.load("workspace/demo.onnx")
for n in model.graph.node:
    if n.op_type == "ScatterND":
        n.op_type = "Plugin"
        n.attribute.append(helper.make_attribute("name", "MyScatterND"))
        n.attribute.append(helper.make_attribute("info", ""))

onnx.save(model, "workspace/demo.onnx")
print("Done.!")
```

2. 实现插件(easy-plugin.cu)

```C++
/* 如果想写一个自己的插件，只需要写一个类（本页中为class MYSELU），然后使用该类继承TRTPlugin，
*  然后主要实现enqueue方法即可，enqueue方法就是自定义算子的C++实现。虽然在onnx中可以导出这么
*  一个自定义算子，在python中也实现了这么一个自定义算子，但是tensorrt不认识，所以在插件中还得
*  再实现一遍
*/

#include "onnx-tensorrt/onnxplugin.hpp"

using namespace ONNXPlugin;

static __device__ float sigmoid(float x){
    return 1 / (1 + expf(-x));
}

// MYSELU的真正实现
static __global__ void MYSELU_kernel_fp32(const float* x, float* output, int edge) {

    int position = threadIdx.x + blockDim.x * blockIdx.x;
	if(position >= edge) return;

    output[position] = x[position] * sigmoid(x[position]);
}

class MYSELU : public TRTPlugin {
public:
	SetupPlugin(MYSELU);		// 初始化插件的一些设置

	virtual void config_finish() override{
		printf("\033[33minit MYSELU config: %s\033[0m\n", config_->info_.c_str());
		printf("weights count is %d\n", config_->weights_.size());
	}
	// 主要实现该方法，这里的GTensor是封装好的一个类，可以像使用tensor一样使用它
	int enqueue(const std::vector<GTensor>& inputs, std::vector<GTensor>& outputs, const std::vector<GTensor>& weights, void* workspace, cudaStream_t stream) override{
		// 获得输入的batch数
		int n = inputs[0].count();
		const int nthreads = 512;
		int block_size = n < nthreads ? n : nthreads;
		int grid_size = (n + block_size - 1) / block_size;

		MYSELU_kernel_fp32 <<<grid_size, block_size, 0, stream>>> (inputs[0].ptr<float>(), outputs[0].ptr<float>(), n);
		return 0;
	}
};
// 必须注册
RegisterPlugin(MYSELU);
```

