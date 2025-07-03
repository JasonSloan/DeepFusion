# 一. 官方文档

```python
https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks
https://docs.openvino.ai/2022.3/notebooks/111-yolov5-quantization-migration-with-output.html
注意此代码适配安装以下版本:
openvino==2023.1.0
openvino-dev==2023.1.0
openvino-telemetry==2023.2.1
```

# 二. 使用方法

```python
将yolov5-7.0代码下载后将其改名为'yolov5'与当前quantization.py放在同级目录下
新建datasets文件夹,将coco128.zip放在datasets文件夹下并解压.
进入到yolov5文件夹,新建一个yolov5s文件夹,将yolov5s.onnx权重文件放入yolov5s文件夹下即可.

.
├── datasets
│   └── coco128
├── quantization.py
└── yolov5
    ├── yolov5s
    │   ├── yolov5s.onnx
    │   └── .....
    ├──......
    ......
```

# 三. 运行quantization.py

