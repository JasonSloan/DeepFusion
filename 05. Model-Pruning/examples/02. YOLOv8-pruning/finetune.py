"""
ultralytics/cfg/__init__.py中修改了增加'finetune'从overrides中pop和重赋值,防止参数检查报错
ultralytics/engine/model.py中增加了对'maskbndict'的加载
ultralytics/cfg/models/v8中新增文件yolov8-pruned.yaml
"""
from ultralytics import YOLO

weight = "weights/pruned.pt"

model = YOLO(weight)
# finetune设置为True
model.train(
    data="ultralytics/cfg/datasets/coco.yaml", 
    epochs=200, 
    finetune=True, 
    device=0
)