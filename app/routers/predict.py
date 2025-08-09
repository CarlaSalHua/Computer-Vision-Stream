import os
from fastapi import APIRouter, UploadFile, HTTPException, status, Depends, File
from fastapi.concurrency import run_in_threadpool

from app import settings as config
from app import utils
from app.auth.jwt import get_current_user
from app.model.services import model_predict
from app.model.schema import PredictResponse

router = APIRouter(tags=["Model"], prefix="/model")

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
        num_objects, image_base64 = await run_in_threadpool(model_predict, file_path)
        return {
            "success": True,
            "num_objects": num_objects,
            "image_base64": image_base64,
            "image_file_name": filename_with_extension
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
