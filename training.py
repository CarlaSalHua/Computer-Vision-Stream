from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
results = model.train(data="SKU-110K.yaml", epochs=50, imgsz=640)