from ultralytics import YOLO
from PIL import Image
import io
import base64
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

def model_predict(image_path: str):
    results = model.predict(source=image_path, save=False)
    if not results or not results[0].boxes:
        return 0, ""

    num_objects = len(results[0].boxes)
    rendered = results[0].plot(labels=False)
    img = Image.fromarray(rendered[..., ::-1])

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return num_objects, img_base64
