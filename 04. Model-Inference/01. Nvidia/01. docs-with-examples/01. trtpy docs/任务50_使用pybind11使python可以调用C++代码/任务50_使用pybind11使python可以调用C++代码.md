**注意**：

1. pybind11不用安装，该节代码中的pybind11.hpp文件就是全部实现的完整代码。

2. 使用pybind11时，如果C++中的代码中返回的是一张图片（数组或者std::vector形式存储），那么对应于Python中是列表，而Python中的列表中存储的是一个一个的对象，非常消耗资源。所以需要在C++中返回ndarray形式的图片数据，对应于pybind11中是就是"pybind11::array_t<uint8_t>"形式的数据。

3. C++中pybind11::array、C++中numpy的ndarray、Python中的numpy的ndarray，这三种类型的数据是相等的、互通的。

4. pybind11::array与pybind11::array_t区别可以gpt查询

5. pybind11不支持C++返回cv::Mat类型的数据，pybind11支持C++返回numpy的ndarray类型数据。

6. 如果已使用pybind11将interface.cpp编译结束，单独执行demo.py不成功，需要执行以下命令以设置环境变量

   ```bash
   source `trtpy env-source --print`
   ```

   为什么make run可以执行，但是单独执行demo.py不能执行，因为Makefile最后一句将库文件搜索路径添加到了环境变量里。

   ​

**本节是以yolov5的推理为例，使用pybind11将C++代码封装成python可以调用的库**



**interface.cpp（与demo.py的接口一一对应）代码注释**

```C++
#include <opencv2/opencv.hpp>
#include <common/ilogger.hpp>
#include "builder/trt_builder.hpp"
#include "app_yolo/yolo.hpp"
#include "pybind11.hpp"

using namespace std;
namespace py = pybind11;

class YoloInfer { 
public:
	YoloInfer(
		string engine, Yolo::Type type, int device_id, float confidence_threshold, float nms_threshold,
		Yolo::NMSMethod nms_method, int max_objects, bool use_multi_preprocess_stream
	){
		instance_ = Yolo::create_infer(
			engine, 
			type,
			device_id,
			confidence_threshold,
			nms_threshold,
			nms_method, max_objects, use_multi_preprocess_stream
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<ObjectDetector::BoxArray> commit(const py::array& image){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		return instance_->commit(cvimage);
	}

private:
	shared_ptr<Yolo::Infer> instance_;
}; 

bool compileTRT(
    int max_batch_size, string source, string output, bool fp16, int device_id, int max_workspace_size
){
    TRT::set_device(device_id);
    return TRT::compile(
        fp16 ? TRT::Mode::FP16 : TRT::Mode::FP32,
        max_batch_size, source, output, {}, nullptr, "", "", max_workspace_size
    );
}
// 对应于demo.py(import yolo)中的包名yolo，m不用变（代指当前模块）
PYBIND11_MODULE(yolo, m){
	// 没看懂
    py::class_<ObjectDetector::Box>(m, "ObjectBox")
		.def_property("left",        [](ObjectDetector::Box& self){return self.left;}, [](ObjectDetector::Box& self, float nv){self.left = nv;})
		.def_property("top",         [](ObjectDetector::Box& self){return self.top;}, [](ObjectDetector::Box& self, float nv){self.top = nv;})
		.def_property("right",       [](ObjectDetector::Box& self){return self.right;}, [](ObjectDetector::Box& self, float nv){self.right = nv;})
		.def_property("bottom",      [](ObjectDetector::Box& self){return self.bottom;}, [](ObjectDetector::Box& self, float nv){self.bottom = nv;})
		.def_property("confidence",  [](ObjectDetector::Box& self){return self.confidence;}, [](ObjectDetector::Box& self, float nv){self.confidence = nv;})
		.def_property("class_label", [](ObjectDetector::Box& self){return self.class_label;}, [](ObjectDetector::Box& self, int nv){self.class_label = nv;})
		.def_property_readonly("width", [](ObjectDetector::Box& self){return self.right - self.left;})
		.def_property_readonly("height", [](ObjectDetector::Box& self){return self.bottom - self.top;})
		.def_property_readonly("cx", [](ObjectDetector::Box& self){return (self.left + self.right) / 2;})
		.def_property_readonly("cy", [](ObjectDetector::Box& self){return (self.top + self.bottom) / 2;})
		.def("__repr__", [](ObjectDetector::Box& obj){
			return iLogger::format(
				"<Box: left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, class_label=%d, confidence=%.5f>",
				obj.left, obj.top, obj.right, obj.bottom, obj.class_label, obj.confidence
			);	
		});

	// 没看懂
    py::class_<shared_future<ObjectDetector::BoxArray>>(m, "SharedFutureObjectBoxArray")
		.def("get", &shared_future<ObjectDetector::BoxArray>::get);

    py::enum_<Yolo::Type>(m, "YoloType")
		.value("V5", Yolo::Type::V5)
		.value("V3", Yolo::Type::V3)
		.value("X", Yolo::Type::X);

	py::enum_<Yolo::NMSMethod>(m, "NMSMethod")
		.value("CPU",     Yolo::NMSMethod::CPU)
		.value("FastGPU", Yolo::NMSMethod::FastGPU);

	// 想要定义一个类，就用class_<类指针>(m, "类名")
	// 对应于demo.py中的yolo.Yolo类
    py::class_<YoloInfer>(m, "Yolo")
		// py::init相当于python中类的__init__函数，同时对应于当前文件中YoloInfer类的构造函数，参数顺序必须一致
		.def(py::init<string, Yolo::Type, int, float, float, Yolo::NMSMethod, int, bool>(), 
			py::arg("engine"), 
			py::arg("type")                 = Yolo::Type::V5, 
			py::arg("device_id")            = 0, 
			py::arg("confidence_threshold") = 0.4f,
			py::arg("nms_threshold") = 0.5f,
			py::arg("nms_method")    = Yolo::NMSMethod::FastGPU,
			py::arg("max_objects")   = 1024,
			py::arg("use_multi_preprocess_stream") = false
		)
		// def_property_readonly意思是只读属性（把YoloInfer中的成员函数当做属性使用），对应于demo.py中的yolo.valid
		.def_property_readonly("valid", &YoloInfer::valid, "Infer is valid")
		// def意思是定义一个函数，对应于demo.py中的yolo.commit函数
		.def("commit", &YoloInfer::commit, py::arg("image"));

	// 想要定义一个函数，就用def，第一个参数是函数名，第二个参数是函数指针，其余参数是函数的参数
	// 对应于demo.py中的yolo.compileTRT函数
    m.def(
		"compileTRT", compileTRT,
		py::arg("max_batch_size"),
		py::arg("source"),
		py::arg("output"),
		py::arg("fp16")                         = false,	// 默认值
		py::arg("device_id")                    = 0,		// 默认值
		py::arg("max_workspace_size")           = 1ul << 28	// 默认值
	);
}
```



**demo.py（与interface.cpp接口一一对应）代码**

```C++

import yolo
import os
import cv2

if not os.path.exists("yolov5s.trtmodel"):
    yolo.compileTRT(
        max_batch_size=1,
        source="yolov5s.onnx",
        output="yolov5s.trtmodel",
        fp16=False,
        device_id=0
    )

infer = yolo.Yolo("yolov5s.trtmodel")
if not infer.valid:
    print("invalid trtmodel")
    exit(0)

image = cv2.imread("rq.jpg")
boxes = infer.commit(image).get()

for box in boxes:
    l, t, r, b = map(int, [box.left, box.top, box.right, box.bottom])
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2, 16)

cv2.imwrite("detect.jpg", image)
```



**Makefile，有几处改变的地方，改动的地方全部以注释"# 相比于普通Makefile"标注**

```makefile
cc        := g++
name      := yolo.so		# 相比于普通Makefile改变：yolo.so中的这个yolo名字必须和interface.cpp中的PYBIND11_MODULE(yolo, m)的yolo名字相同
workdir   := workspace
srcdir    := src
objdir    := objs
stdcpp    := c++11
cuda_home := /root/software/anaconda3/envs/trtpy/lib/python3.9/site-packages/trtpy/trt8cuda115cudnn8
syslib    := /root/software/anaconda3/envs/trtpy/lib/python3.9/site-packages/trtpy/lib
cpp_pkg   := /root/software/anaconda3/envs/trtpy/lib/python3.9/site-packages/trtpy/cpp-packages
cuda_arch := 
nvcc      := $(cuda_home)/bin/nvcc -ccbin=$(cc)

# 定义cpp的路径查找和依赖项mk文件
cpp_srcs := $(shell find $(srcdir) -name "*.cpp")
cpp_objs := $(cpp_srcs:.cpp=.cpp.o)
cpp_objs := $(cpp_objs:$(srcdir)/%=$(objdir)/%)
cpp_mk   := $(cpp_objs:.cpp.o=.cpp.mk)

# 定义cu文件的路径查找和依赖项mk文件
cu_srcs := $(shell find $(srcdir) -name "*.cu")
cu_objs := $(cu_srcs:.cu=.cu.o)
cu_objs := $(cu_objs:$(srcdir)/%=$(objdir)/%)
cu_mk   := $(cu_objs:.cu.o=.cu.mk)

# 定义opencv和cuda需要用到的库文件
link_cuda      := cudart cudnn
link_trtpro    := 
link_tensorRT  := nvinfer nvinfer_plugin
link_opencv    := opencv_core opencv_imgproc opencv_imgcodecs
link_sys       := stdc++ dl protobuf python3.9	# 相比于普通Makefile增加：python3.9库(libpython3.9.so)
link_librarys  := $(link_cuda) $(link_tensorRT) $(link_sys) $(link_opencv)

# 定义头文件路径，请注意斜杠后边不能有空格
# 只需要写路径，不需要写-I
include_paths := src              \
	src/tensorRT                  \
    $(cuda_home)/include/cuda     \
	$(cuda_home)/include/tensorRT \
	$(cpp_pkg)/opencv4.2/include  \
	$(cuda_home)/include/protobuf \
	/root/software/anaconda3/envs/trtpy/include/python3.9	# 相比于普通Makefile增加：python3.9头文件

# 定义库文件路径，只需要写路径，不需要写-L，相比于普通Makefile增加：python3.9库路径
library_paths := $(cuda_home)/lib64 $(syslib) $(cpp_pkg)/opencv4.2/lib /root/software/anaconda3/envs/trtpy/lib	

# 把library path给拼接为一个字符串，例如a b c => a:b:c
# 然后使得LD_LIBRARY_PATH=a:b:c
empty := 
library_path_export := $(subst $(empty) $(empty),:,$(library_paths))

# 把库路径和头文件路径拼接起来成一个，批量自动加-I、-L、-l
run_paths     := $(foreach item,$(library_paths),-Wl,-rpath=$(item))
include_paths := $(foreach item,$(include_paths),-I$(item))
library_paths := $(foreach item,$(library_paths),-L$(item))
link_librarys := $(foreach item,$(link_librarys),-l$(item))

# 如果是其他显卡，请修改-gencode=arch=compute_75,code=sm_75为对应显卡的能力
# 显卡对应的号码参考这里：https://developer.nvidia.com/zh-cn/cuda-gpus#compute
# 如果是 jetson nano，提示找不到-m64指令，请删掉 -m64选项。不影响结果
cpp_compile_flags := -std=$(stdcpp) -w -g -O0 -m64 -fPIC -fopenmp -pthread
cu_compile_flags  := -std=$(stdcpp) -w -g -O0 -m64 $(cuda_arch) -Xcompiler "$(cpp_compile_flags)"
link_flags        := -pthread -fopenmp -Wl,-rpath='$$ORIGIN'

cpp_compile_flags += $(include_paths)
cu_compile_flags  += $(include_paths)
link_flags        += $(library_paths) $(link_librarys) $(run_paths)

# 如果头文件修改了，这里的指令可以让他自动编译依赖的cpp或者cu文件
ifneq ($(MAKECMDGOALS), clean)
-include $(cpp_mk) $(cu_mk)
endif

$(name)   : $(workdir)/$(name)

all       : $(name)
# 相比于普通Makefile增加：python demo.py，这也就是为什么make run能够运行python代码
# 相比于普通Makefile去掉：&& ./$(name) $(run_args)，防止直接运行生成的动态库
run       : $(name)
	@cd $(workdir) && python demo.py $(run_args)

# 相比于普通Makefile增加：-shared选项，生成动态库
$(workdir)/$(name) : $(cpp_objs) $(cu_objs)
	@echo Link $@
	@mkdir -p $(dir $@)
	@$(cc) -shared $^ -o $@ $(link_flags)

$(objdir)/%.cpp.o : $(srcdir)/%.cpp
	@echo Compile CXX $<
	@mkdir -p $(dir $@)
	@$(cc) -c $< -o $@ $(cpp_compile_flags)

$(objdir)/%.cu.o : $(srcdir)/%.cu
	@echo Compile CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -c $< -o $@ $(cu_compile_flags)

# 编译cpp依赖项，生成mk文件
$(objdir)/%.cpp.mk : $(srcdir)/%.cpp
	@echo Compile depends C++ $<
	@mkdir -p $(dir $@)
	@$(cc) -M $< -MF $@ -MT $(@:.cpp.mk=.cpp.o) $(cpp_compile_flags)
    
# 编译cu文件的依赖项，生成cumk文件
$(objdir)/%.cu.mk : $(srcdir)/%.cu
	@echo Compile depends CUDA $<
	@mkdir -p $(dir $@)
	@$(nvcc) -M $< -MF $@ -MT $(@:.cu.mk=.cu.o) $(cu_compile_flags)

# 定义清理指令
clean :
	@rm -rf $(objdir) $(workdir)/$(name) $(workdir)/*.trtmodel $(workdir)/*.onnx

# 防止符号被当做文件
.PHONY : clean run $(name)

# 导出依赖库路径，使得能够运行起来
export LD_LIBRARY_PATH:=$(library_path_export)
```

