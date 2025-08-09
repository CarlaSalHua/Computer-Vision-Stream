import hashlib

def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

async def get_file_hash(file) -> str:
    contents = await file.read()
    await file.seek(0)
    ext = file.filename.rsplit('.', 1)[1].lower()
    return hashlib.sha256(contents).hexdigest() + "." + ext