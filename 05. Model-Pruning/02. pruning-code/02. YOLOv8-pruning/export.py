from ultralytics import YOLO

# model = YOLO("yolov8s.yaml")
# model.export(format="onnx")

weight = ""
model = YOLO(weight)
model.export(format="onnx")         # 无需手动eval