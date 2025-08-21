from pydantic import BaseModel

class PredictResponse(BaseModel):
    success: bool
    num_objects: int
    bounding_objects: int
    image_base64: str
    image_file_name: str
