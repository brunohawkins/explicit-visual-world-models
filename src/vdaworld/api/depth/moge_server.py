"""
FastAPI microservice wrapping the MoGe depth and point-map estimator.

Exposes the same ``POST /estimate_3d_points`` endpoint as the VGGT server
so that ``core/api.py`` needs only to change the target port to switch
backends.

Start via ``spin_up_api.py`` (see ``api_config.yaml``) or directly::

    conda run -n moge uvicorn src.vdaworld.api.depth.moge_server:app \\
        --host 127.0.0.1 --port 5004
"""
from __future__ import annotations

import gc
import io
import logging

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.vdaworld.api.depth.moge_estimator import estimate

app = FastAPI(title="VDAWorld MoGe Depth Estimator Microservice")
logger = logging.getLogger(__name__)


@app.post("/estimate_3d_points")
async def run_moge_endpoint(image: UploadFile = File(...)):
    """Accept a PNG/JPEG image; return ``pts3d`` and ``intrinsics`` as ``.npz``."""
    try:
        gc.collect()

        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(pil_img, dtype=np.uint8)

        logger.info("MoGe: received image %s", image_np.shape)
        result = estimate(image_np)
        logger.info("MoGe: pts3d shape %s", result["pts3d"].shape)

        out_buffer = io.BytesIO()
        np.savez(out_buffer, **result)
        out_buffer.seek(0)

        gc.collect()
        return Response(content=out_buffer.getvalue(), media_type="application/octet-stream")

    except Exception as exc:
        logger.exception("MoGe estimation failed.")
        raise HTTPException(status_code=500, detail=str(exc))
