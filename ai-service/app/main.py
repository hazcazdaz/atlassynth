from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from tempfile import NamedTemporaryFile
from io import BytesIO          # ← add this
import numpy as np
import rasterio
from PIL import Image
from .models import tag_tile, depth_map


app = FastAPI(title="AtlasSynth AI Service", version="0.1.0")

class TileTags(BaseModel):
    tags: list

@app.post("/classify")
async def classify_endpoint(file: UploadFile):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")
    preds = tag_tile(image)          # list of {label, score}
    return JSONResponse(preds)


@app.post("/depth")
async def depth_endpoint(file: UploadFile):
    img_bytes = await file.read()
    image = Image.open(BytesIO(img_bytes)).convert("RGB")

    # --- run model ---
    depth_pil = depth_map(image)           # <- PIL.Image.Image
    depth = np.array(depth_pil).astype("float32")  # -> NumPy array

    # --- save GeoTIFF ---
    with NamedTemporaryFile(delete=False, suffix=".tif") as tmp:
        with rasterio.open(
            tmp.name,
            "w",
            driver="GTiff",
            height=depth.shape[0],
            width=depth.shape[1],
            count=1,
            dtype="float32",
        ) as dst:
            dst.write(depth, 1)

        return FileResponse(tmp.name,
                            media_type="image/tiff",
                            filename="height.tif")

