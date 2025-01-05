from ultralytics import YOLO

model = YOLO("weights/yolov8s.pt")
# L1正则的惩罚项系数sr
model.train(
    sr=0,
    data="ultralytics/cfg/datasets/coco.yaml", 
    epochs=200, 
    project='.', 
    name='runs/train-norm', 
    batch=48, 
    device=0
)