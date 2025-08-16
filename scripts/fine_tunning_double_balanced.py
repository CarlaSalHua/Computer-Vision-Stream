from ultralytics import YOLO

import numpy as np
import ultralytics.data.dataset as dataset
import ultralytics.data as data

class YOLOWeightedDataset(dataset.YOLODataset):
    def __init__(self, *args, mode='train', **kwargs):
        super().__init__(*args, mode=mode, **kwargs)
        self.train_mode = 'train' in self.prefix

        # Count instances per class
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.agg_func = np.mean
        self.class_weights = class_weights
        self.weights = self.calculate_weights()
        self.probs = self.calculate_probabilities()

    def count_instances(self):
        self.counts = np.zeros(len(self.data['names']), dtype=int)
        for lbl in self.labels:
            cls_ids = lbl['cls'].flatten().astype(int)
            for c in cls_ids:
                self.counts[c] += 1
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        weights = []
        for lbl in self.labels:
            cls_ids = lbl['cls'].flatten().astype(int)
            weight = self.agg_func(self.class_weights[cls_ids])
            weights.append(weight)
        return np.array(weights)

    def calculate_probabilities(self):
        return self.weights / np.sum(self.weights)

    def __getitem__(self, idx):
        chosen = np.random.choice(len(self.probs), p=self.probs)
        return super().__getitem__(chosen)

data.dataset.YOLODataset = YOLOWeightedDataset


# Load your checkpoint (after pretraining on SKU110K)
model = YOLO("runs/detect/train/weights/best.pt")

# Freeze all
for param in model.model.parameters():
    param.requires_grad = False

# Unfreeze last N layers (including detection head)
N = 20
for layer in list(model.model.modules())[-N:]:
    for param in layer.parameters(recurse=True):
        param.requires_grad = True

# Set new head for 2 classes
model.model.model[-1].nc = 2
model.model.model[-1].names = ['object', 'empty']


model.train(
    data="data/data.yaml",
    epochs=400,
    imgsz=640,
    lr0=1e-4,
    batch=16,
    resume=False,
)
