import io
import os
import tempfile
import logging
import gc
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response

from src.vdaworld.api.depth.estimator import estimate

app = FastAPI(title="VDAWorld VGGT Depth Estimator Microservice")
logger = logging.getLogger(__name__)


@app.post("/estimate_3d_points")
async def run_vggt_endpoint(image: UploadFile = File(...)):
    """
    Accepts an image. Returns a numpy byte buffer containing a dictionary with keys 'pts3d' and 'intrinsics'.
    """
    try:
        gc.collect()

        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width_in, height_in = pil_img.size

        logger.info(
            f"Estimating 3D Points for image of size {(width_in, height_in)}..."
        )

        with tempfile.TemporaryDirectory() as tempdir:
            pil_img.save(os.path.join(tempdir, "image_000.png"))
            result = estimate(tempdir)

        # Reshape to true resolution
        width_out = result["pts3d"].shape[-2]
        height_out = result["pts3d"].shape[-3]

        # Original script sliced index 0 for Image array inputs,
        # but estimator.py now resolves the batch dimension and scales to native resolution.
        if result["pts3d"].ndim == 4:
            result["pts3d"] = result["pts3d"][0]

        logger.info(f"Generated pts3d shape: {result['pts3d'].shape}")

        out_buffer = io.BytesIO()
        # Save compressed dictionary of numpy arrays
        np.savez(out_buffer, **result)
        out_buffer.seek(0)

        gc.collect()

        return Response(
            content=out_buffer.getvalue(), media_type="application/octet-stream"
        )

    except Exception as e:
        logger.exception("Depth estimation failed critical error.")
        raise HTTPException(status_code=500, detail=str(e))
