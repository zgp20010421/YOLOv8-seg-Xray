from ultralytics import YOLO

# 加载模型
model = YOLO(model="../models/torch/yolov8s-seg-Xray-1b.pt")

if __name__ == '__main__':
    model.export(format="onnx")