from ultralytics import YOLO
from PIL import Image
import numpy as np
import io
import base64
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")
model = YOLO(MODEL_PATH)

def model_predict_image(image_path: str):
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

# Modificamos la función para que acepte un objeto de imagen (PIL.Image)
def model_predict(image: Image.Image):
    """
    Realiza la predicción en una imagen y devuelve los resultados.

    Args:
        image (Image.Image): La imagen en formato PIL.

    Returns:
        tuple: (número de objetos, imagen procesada como string base64)
    """
    # El modelo de YOLO puede aceptar directamente objetos PIL.Image
    results = model.predict(source=image, save=False)

    # Si no hay resultados, devolvemos la imagen original sin cambios
    if not results or not results[0].boxes:
        # Convertimos la imagen original a base64 para devolverla
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return 0, img_base64

    num_objects = len(results[0].boxes)

    # results[0].plot() devuelve una imagen en formato NumPy (BGR)
    rendered_image_np = results[0].plot(labels=True) # labels=True para ver "caja_llena", "caja_vacia"

    # Convertimos de NumPy BGR a PIL RGB para enviarlo de vuelta
    rendered_image_pil = Image.fromarray(rendered_image_np[..., ::-1])

    # Codificamos la imagen procesada a base64
    buffer = io.BytesIO()
    rendered_image_pil.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return num_objects, img_base64
