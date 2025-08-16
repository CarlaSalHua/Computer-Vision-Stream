from ultralytics import YOLO

# Load your trained 1-class model
model = YOLO("yolo11n.pt")

# Fine-tune
model.train(
    data="data/data.yaml",
    epochs=100,
    imgsz=640
)
