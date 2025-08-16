from ultralytics import YOLO

# Load your checkpoint (after pretraining on SKU110K)
model = YOLO("runs/detect/train/weights/best.pt")

# Freeze all
for param in model.model.parameters():
    param.requires_grad = False

# Unfreeze last N layers (including detection head)
N = 50
for layer in list(model.model.modules())[-N:]:
    for param in layer.parameters(recurse=True):
        param.requires_grad = True

# Set new head for 2 classes
model.model.model[-1].nc = 2
model.model.model[-1].names = ['object', 'empty']

# Fine-tune longer
model.train(
    data="data/data.yaml",
    epochs=100,
    imgsz=640,
    lr0=1e-4,
    batch=16,
    resume=False,
)
