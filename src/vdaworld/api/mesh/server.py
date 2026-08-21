import asyncio
import io
import gc
import logging
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response

from src.vdaworld.api.mesh.estimator import estimate

app = FastAPI(title="VDAWorld SAM3D Mesh Estimator Microservice")
logger = logging.getLogger(__name__)

# Serialise all GPU calls so that concurrent requests queue rather than OOM.
_gpu_lock = asyncio.Lock()

@app.post("/estimate_3d_mesh")
async def run_sam3d_endpoint(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    max_vertices: int = Form(10000)
):
    try:
        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(pil_img)

        mask_bytes = await mask.read()
        pil_mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        mask_np = np.array(pil_mask) > 128

        logger.info(f"Estimating 3D Mesh for image of size {image_np.shape}...")

        async with _gpu_lock:
            gc.collect()
            mesh = estimate(image_np, mask_np, max_vertices)
            gc.collect()

        out_buffer = io.BytesIO()
        mesh.export(file_obj=out_buffer, file_type='glb')
        out_buffer.seek(0)

        return Response(content=out_buffer.getvalue(), media_type="model/gltf-binary")

    except Exception as e:
        logger.exception("Mesh generation failed critical error.")
        raise HTTPException(status_code=500, detail=str(e))
