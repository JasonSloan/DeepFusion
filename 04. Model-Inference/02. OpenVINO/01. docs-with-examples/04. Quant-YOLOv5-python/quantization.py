import sys
import subprocess
import os.path as osp
from pathlib import Path


import nncf
import numpy as np
import matplotlib.pyplot as plt

import openvino as ov
from openvino.tools.pot.api import DataLoader
from openvino.tools.pot.engines.ie_engine import IEEngine
from openvino.tools.pot.graph import load_model
from openvino.tools.pot.pipeline.initializer import create_pipeline
from openvino.tools.pot.graph.model_utils import compress_model_weights
from openvino.tools.pot.graph import load_model, save_model

sys.path.append("./yolov5")
from yolov5.utils.dataloaders import create_dataloader
from yolov5.utils.general import check_dataset
from yolov5.utils.general import download
from yolov5.export import attempt_load, yaml_save
from yolov5.val import run as validation_fn



IMAGE_SIZE = 640
MODEL_NAME = "yolov5s"
MODEL_PATH = f"yolov5/{MODEL_NAME}"
DATASET_CONFIG = "./yolov5/data/coco128.yaml"


def export_ov_model(onnx_path,fp32_path,fp16_path):
    """将onnx模型导出为openvino的模型"""
    # fp32 IR model
    print(f"Export ONNX to OpenVINO FP32 IR to: {fp32_path}")
    model = ov.convert_model(onnx_path)
    ov.save_model(model, fp32_path, compress_to_fp16=False)
    # fp16 IR model
    print(f"Export ONNX to OpenVINO FP16 IR to: {fp16_path}")
    model = ov.convert_model(onnx_path)
    ov.save_model(model, fp16_path, compress_to_fp16=True)

def create_data_source():
    """
    创建dataloader,就是普通的torch版本的dataloader
    """
    if not Path("datasets/coco128").exists():
        urls = ["https://ultralytics.com/assets/coco128.zip"]
        download(urls, dir="datasets")
    data = check_dataset(DATASET_CONFIG)
    val_dataloader = create_dataloader(
        data["val"], imgsz=640, batch_size=1, stride=32, pad=0.5, workers=1
    )[0]
    return val_dataloader


class YOLOv5POTDataLoader(DataLoader):
    """继承openvino.tools.pot.api中的DataLoader, 并且要实现__len__和__getitem__方法"""

    def __init__(self, data_source):
        super().__init__({})
        self._data_loader = data_source
        self._data_iter = iter(self._data_loader)

    def __len__(self):
        return len(self._data_loader.dataset)

    def __getitem__(self, item):
        try:
            batch_data = next(self._data_iter)
        except StopIteration:
            self._data_iter = iter(self._data_loader)
            batch_data = next(self._data_iter)

        im, target, path, shape = batch_data

        im = im.float()
        im /= 255
        nb, _, height, width = im.shape
        img = im.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        annotation = dict()
        annotation["image_path"] = path
        annotation["target"] = target
        annotation["batch_size"] = nb
        annotation["shape"] = shape
        annotation["width"] = width
        annotation["height"] = height
        annotation["img"] = img
        return (item, annotation), img

def transform_fn(data_item):
    """为NNCF提供数据转换函数"""
    # unpack input images tensor
    images = data_item[0]
    # convert input tensor into float format
    images = images.float()
    # scale input
    images = images / 255
    # convert torch tensor to numpy array
    images = images.cpu().detach().numpy()
    return images

def validate_model(model_path):
    """评估模型, validation_fn调用的是yolov5中的val.py中的run函数"""
    msg = colorstr(f"Checking the accuracy of model {osp.basename(model_path)}:")
    print(msg)
    fp32_metrics = validation_fn(
        data=DATASET_CONFIG,
        weights=Path(model_path).parent,
        batch_size=1,
        workers=1,
        plots=False,
        device="cpu",
        iou_thres=0.65,
    )
    ap5 = fp32_metrics[0][2]
    ap_full = fp32_metrics[0][3]
    print(f"mAP@.5 = {ap5}")
    print(f"mAP@.5:.95 = {ap_full}")
    return ap5, ap_full
    

def plot(fp32_ap5, fp32_ap_full, pot_int8_ap5, pot_int8_ap_full, nncf_int8_ap5, nncf_int8_ap_full):
    """绘制对比图, 对比得到各种精度下的AP值"""
    # plt.style.use("seaborn-deep")
    fp32_acc = np.array([fp32_ap5, fp32_ap_full])
    pot_int8_acc = np.array([pot_int8_ap5, pot_int8_ap_full])
    nncf_int8_acc = np.array([nncf_int8_ap5, nncf_int8_ap_full])
    x_data = ("AP@0.5", "AP@0.5:0.95")
    x_axis = np.arange(len(x_data))
    fig = plt.figure()
    fig.patch.set_facecolor("#FFFFFF")
    fig.patch.set_alpha(0.7)
    ax = fig.add_subplot(111)
    plt.bar(x_axis - 0.2, fp32_acc, 0.3, label="FP32")
    for i in range(0, len(x_axis)):
        plt.text(
            i - 0.3,
            round(fp32_acc[i], 3) + 0.01,
            str(round(fp32_acc[i], 3)),
            fontweight="bold",
        )
    plt.bar(x_axis + 0.15, pot_int8_acc, 0.3, label="POT INT8")
    for i in range(0, len(x_axis)):
        plt.text(
            i + 0.05,
            round(pot_int8_acc[i], 3) + 0.01,
            str(round(pot_int8_acc[i], 3)),
            fontweight="bold",
        )
    plt.bar(x_axis + 0.5, nncf_int8_acc, 0.3, label="NNCF INT8")
    for i in range(0, len(x_axis)):
        plt.text(
            i + 0.4,
            round(nncf_int8_acc[i], 3) + 0.01,
            str(round(nncf_int8_acc[i], 3)),
            fontweight="bold",
        )
    plt.xticks(x_axis, x_data)
    plt.xlabel("Average Precision")
    plt.title("Compare Yolov5 FP32 and INT8 model average precision")
    plt.legend()
    plt.savefig(f"{MODEL_PATH}/compare.png", dpi=300)

def benchmark(command):
    """benchmark,测试各种精度下的速度以及fps
    """
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    print(colorstr('Standard Out:'))
    print(result.stdout)
    print(colorstr('Standard Error:'))
    print(result.stderr)
    print(colorstr('Standard Code:'))
    print(result.returncode)

def colorstr(*input):
    """彩色打印"""
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


if __name__ == "__main__":
    #============================模型导出============================
    # 将onnx模型导出为openvino模型,为了对比,使用fp32 fp16 int8三种类型的模型进行比较
    onnx_path = f"{MODEL_PATH}/{MODEL_NAME}.onnx"
    fp32_path = f"{MODEL_PATH}/FP32_openvino_model/{MODEL_NAME}_fp32.xml"                       
    fp16_path = f"{MODEL_PATH}/FP16_openvino_model/{MODEL_NAME}_fp16.xml"
    export_ov_model(onnx_path=onnx_path, fp32_path=fp32_path, fp16_path=fp16_path)

    #============================创建openvino版本的dataloader============================
    data_source = create_data_source()      # 普通的torch版本的dataloader
    pot_data_loader = YOLOv5POTDataLoader(data_source)  # openvino版本的dataloader
    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)  # nncf版本的dataloader
    
    #============================加载模型(不使用NNCF)============================
    algorithms_config = [
        {
            "name": "DefaultQuantization",      
            "params": {
                "preset": "mixed",
                "stat_subset_size": 300,
                "target_device": "CPU"
            },
        }
    ]
    engine_config = {"device": "CPU"}
    model_config = {
        "model_name": f"{MODEL_NAME}",
        "model": fp32_path,
        "weights": fp32_path.replace(".xml", ".bin"),
    }
    #  Load model as POT model representation
    pot_model = load_model(model_config)
    #  Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=engine_config, data_loader=pot_data_loader)
    # Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms_config, engine)
    #============================执行量化(不使用NNCF)============================
    compressed_model = pipeline.run(pot_model)
    compress_model_weights(compressed_model)
    #============================保存量化后模型(不使用NNCF)============================
    optimized_save_dir = Path(f"{MODEL_PATH}/POT_INT8_openvino_model/")
    save_model(compressed_model, optimized_save_dir, model_config["model_name"] + "_int8")
    pot_int8_path = f"{optimized_save_dir}/{MODEL_NAME}_int8.xml"

    #============================加载模型(使用NNCF)============================
    subset_size = 300
    preset = nncf.QuantizationPreset.MIXED
    core = ov.Core()
    ov_model = core.read_model(fp32_path)
    #============================执行量化(不使用NNCF)============================
    quantized_model = nncf.quantize(
        ov_model, nncf_calibration_dataset, preset=preset, subset_size=subset_size
    )
    #============================保存量化后模型(使用NNCF)============================
    nncf_int8_path = f"{MODEL_PATH}/NNCF_INT8_openvino_model/{MODEL_NAME}_int8.xml"
    ov.save_model(quantized_model, nncf_int8_path, compress_to_fp16=False)

    #============================保存配置(validate_model会用到配置文件)============================
    model = attempt_load(
        f"{MODEL_PATH}/{MODEL_NAME}.pt", device="cpu", inplace=True, fuse=True
    ) 
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    yaml_save(Path(nncf_int8_path).with_suffix(".yaml"), metadata)
    yaml_save(Path(pot_int8_path).with_suffix(".yaml"), metadata)
    yaml_save(Path(fp32_path).with_suffix(".yaml"), metadata)

    #============================模型评估============================
    fp32_ap5, fp32_ap_full = validate_model(fp32_path)
    pot_int8_ap5, pot_int8_ap_full = validate_model(pot_int8_path)
    nncf_int8_ap5, nncf_int8_ap_full = validate_model(nncf_int8_path)

    #============================绘制对比图============================
    plot(fp32_ap5, fp32_ap_full, pot_int8_ap5, pot_int8_ap_full, nncf_int8_ap5, nncf_int8_ap_full)

    #============================benchmark============================
    for path in [fp32_path, fp16_path, pot_int8_path, nncf_int8_path]:
        print(colorstr(f"benchmark {path}"))
        command = [f'benchmark_app -m {path} -d CPU -api async -t 15']
        benchmark(command)




