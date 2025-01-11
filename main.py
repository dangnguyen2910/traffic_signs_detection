from ultralytics import YOLO

model = YOLO("yolo11n.pt")

train_result = model.train(data = "../data/data.yaml", epochs = 2, imgsz = 416, device = 0)
