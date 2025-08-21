import os
import base64
import io
from PIL import Image
from pydantic import BaseModel
from fastapi import APIRouter, UploadFile, HTTPException, status, Depends, File
from fastapi.concurrency import run_in_threadpool

from app import settings as config
from app import utils
from app.auth.jwt import get_current_user
from app.model.services import model_predict, model_predict_image
from app.model.schema import PredictResponse

router = APIRouter(tags=["Model"], prefix="/model")

# Creamos un schema para la petición de streaming
class StreamRequest(BaseModel):
    image_base64: str


# Endpoint para el streaming en tiempo real
@router.post("/predict_stream", response_model=PredictResponse)
async def predict_stream(request: StreamRequest):
    try:
        # Decodificar la imagen base64 recibida
        img_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(img_bytes))

        # Ejecutar la predicción del modelo en un hilo separado
        num_objects, image_base64_processed = await run_in_threadpool(model_predict, image)

        return {
            "success": True,
            "num_objects": num_objects,
            "image_base64": image_base64_processed,
            "image_file_name": "stream_frame" # Nombre de archivo no es relevante aquí
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing stream: {str(e)}")



@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), current_user=Depends(get_current_user)):
    if not file:
        raise HTTPException(status_code=400, detail="No file provided.")

    if not utils.allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    file_content = await file.read()
    await file.seek(0)
    file_hash = await utils.get_file_hash(file)
    filename_with_extension = file_hash
    file_path = os.path.join(config.UPLOAD_FOLDER, filename_with_extension)

    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {e}")

    try:
        num_objects, image_base64, bounding_objects = await run_in_threadpool(model_predict_image, file_path)

        return {
            "success": True,
            "num_objects": num_objects,
            "bounding_objects": bounding_objects,
            "image_base64": image_base64,
            "image_file_name": filename_with_extension
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
