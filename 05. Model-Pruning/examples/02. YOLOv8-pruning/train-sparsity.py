"""
修改的代码:
ultralytics/engine/trainer.py: 禁用amp, 增加梯度惩罚项系数
ultralytics/engine/model.py: 主要是将sr参数绑定到self.trainer上
"""
from ultralytics import YOLO

model = YOLO("runs/train-norm/weights/best.pt")
# L1正则的惩罚项系数sr
model.train(
    sr=1e-2, 
    lr0=1e-3,
    data="ultralytics/cfg/datasets/coco.yaml", 
    epochs=50, 
    patience=50, 
    project='.', 
    name='runs/train-sparsity', 
    batch=48, 
    device=0
)