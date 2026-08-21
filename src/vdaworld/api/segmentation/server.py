import gc
import io
import logging
import os
import tempfile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(name)s: %(message)s")

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

from src.vdaworld.api.segmentation.segment import segment, segment_video

app = FastAPI(title="VDAWorld SAM3 Segmentation Microservice")
logger = logging.getLogger(__name__)


@app.post("/segment")
async def run_segmentation_endpoint(
    image: UploadFile = File(...),
    object_prompt: str = Form(...)
):
    """
    Accepts an image and a text prompt. Returns a numpy array buffer containing boolean masks.
    """
    try:
        gc.collect()

        img_bytes = await image.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        logger.info(f"Segmenting '{object_prompt}' on image size {pil_img.size}...")

        masks = segment(pil_img, object_prompt)  # (N, H, W)

        logger.info(f"Segmentation completed. Found {masks.shape[0]} masks of shape {masks.shape[1:]}.")

        out_buffer = io.BytesIO()
        np.save(out_buffer, masks)
        out_buffer.seek(0)

        gc.collect()
        return Response(content=out_buffer.getvalue(), media_type="application/octet-stream")

    except Exception as e:
        logger.exception("Segmentation failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment_video")
async def run_video_segmentation_endpoint(
    video: UploadFile = File(...),
    object_prompt: str = Form(...),
    fps: int = Form(default=30),
):
    """
    Accepts a video file and a text prompt. Returns an MP4 where each tracked object
    is painted with a solid, distinct colour against a black background.
    """
    try:
        video_bytes = await video.read()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = tmp.name

        logger.info(f"Tracking '{object_prompt}' through video ({len(video_bytes)} bytes)...")

        frames = segment_video(tmp_path, object_prompt)
        os.unlink(tmp_path)

        if not frames:
            raise HTTPException(status_code=422, detail="No objects detected for the given prompt.")

        logger.info(f"Tracking complete: {len(frames)} frames, size {frames[0].shape[:2]}.")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as out_tmp:
            out_path = out_tmp.name
        H, W = frames[0].shape[:2]
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        for frame in frames:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR for cv2
        writer.release()
        with open(out_path, "rb") as f:
            video_bytes_out = f.read()
        os.unlink(out_path)
        out_buffer = io.BytesIO(video_bytes_out)

        gc.collect()
        return Response(content=out_buffer.getvalue(), media_type="video/mp4")

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video segmentation failed.")
        raise HTTPException(status_code=500, detail=str(e))
