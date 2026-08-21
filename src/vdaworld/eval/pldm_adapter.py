"""Adapter between Felix's PLDM session-server image format and our simulator's frame format.

PLDM returns base64-encoded 64x64 RGB PNGs (white background, red dot, black walls).
Our simulator's frame contract is (H, W, 3) uint8 — currently (65, 65, 3) for two-room.
"""

from __future__ import annotations

import base64
import io
from typing import Tuple

import numpy as np
from PIL import Image


def decode_pldm_image(b64_png: str) -> np.ndarray:
    """Base64-decode a PLDM image string into an (H, W, 3) uint8 numpy array."""
    raw = base64.b64decode(b64_png)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.array(img, dtype=np.uint8)


def adapt_pldm_to_sim(
    b64_png: str, target_size: Tuple[int, int] = (65, 65)
) -> np.ndarray:
    """Decode + nearest-neighbour resize to the simulator's expected frame size.

    Nearest-neighbour is used because the source content (sharp red dot, hard wall
    edges) anti-aliases poorly under bilinear/bicubic and would smear the dot's
    centroid and the wall boundaries that the sim's `find_red_dot` / collision
    logic depend on.
    """
    arr = decode_pldm_image(b64_png)
    h, w = target_size
    if arr.shape[:2] == (h, w):
        return arr
    img = Image.fromarray(arr, mode="RGB").resize((w, h), Image.NEAREST)
    return np.array(img, dtype=np.uint8)
