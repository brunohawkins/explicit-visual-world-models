"""
MoGe monocular depth and point-map backend.

Outputs ``pts3d`` in the same OpenGL camera coordinate system used by the
VGGT backend (+X right, +Y up, -Z forward), so the rest of the pipeline is
agnostic to which backend is active.

Coordinate conversion
---------------------
MoGe outputs points in the OpenCV convention (x right, y down, z forward).
Converting to OpenGL requires negating Y and Z::

    pts3d_ogl[..., 1] *= -1   # y down  → y up
    pts3d_ogl[..., 2] *= -1   # z fwd   → -z fwd

Intrinsics
----------
MoGe returns *normalised* intrinsics where the image spans [0, 1] in both
dimensions (principal point cx=0.5, cy=0.5 for a centred camera).
Pixel-space intrinsics are recovered by::

    K_px[0, 0] *= W   # fx
    K_px[1, 1] *= H   # fy
    K_px[0, 2] *= W   # cx
    K_px[1, 2] *= H   # cy

Depth scaling
-------------
MoGe-1 (``moge-vitl``) produces relative depth. ``pts3d`` is normalised so
the mean camera distance equals 1.0, matching the VGGT convention.

MoGe-2 (``moge-2-vitl``, ``moge-2-vitl-normal``) produces **metric** depth
in metres. Normalisation is intentionally skipped so callers can exploit the
absolute scale.

Model selection
---------------
Set ``_MODEL_NAME`` below to choose the checkpoint.  The ``_IS_METRIC`` flag
is inferred automatically from the model name and controls normalisation.
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MOGE_REPO = os.environ.get("MOGE_PATH")
if _MOGE_REPO not in sys.path:
    sys.path.insert(0, _MOGE_REPO)

# ---------------------------------------------------------------------------
# Model selection — change _MODEL_NAME to switch checkpoint
# ---------------------------------------------------------------------------

# _MODEL_NAME: str = "Ruicheng/moge-vitl"            # MoGe-1 — relative depth (downloaded)
_MODEL_NAME: str = "Ruicheng/moge-2-vitl"  # MoGe-2 — metric depth
# _MODEL_NAME: str = "Ruicheng/moge-2-vitl-normal"  # MoGe-2 — metric depth + normals

_IS_METRIC: bool = "moge-2" in _MODEL_NAME

# ---------------------------------------------------------------------------
# Lazy model cache
# ---------------------------------------------------------------------------

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_CACHE = None


def _get_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        logger.info("Loading MoGe model '%s' onto %s...", _MODEL_NAME, _device)
        if "moge-2" in _MODEL_NAME:
            from moge.model.v2 import MoGeModel
        else:
            from moge.model.v1 import MoGeModel
        _MODEL_CACHE = MoGeModel.from_pretrained(_MODEL_NAME).to(_device)
        _MODEL_CACHE.eval()
        logger.info("MoGe model loaded.")
    return _MODEL_CACHE


# ---------------------------------------------------------------------------
# Public inference function
# ---------------------------------------------------------------------------


def estimate(image_rgb_uint8: np.ndarray) -> dict:
    """Run MoGe on a single RGB image and return pipeline-compatible outputs.

    Args:
        image_rgb_uint8: ``(H, W, 3)`` uint8 numpy array in RGB order.

    Returns:
        A dict with keys:

        * ``"pts3d"`` — ``(H, W, 3)`` float32 point map in the OpenGL camera
          coordinate system (+X right, +Y up, -Z forward).  For MoGe-1 the
          mean distance of all valid points equals 1.0.  For MoGe-2 the
          values are in metres.
        * ``"intrinsics"`` — ``(3, 3)`` float32 pixel-space camera intrinsics.
    """
    model = _get_model()
    h, w = image_rgb_uint8.shape[:2]

    # (H, W, 3) uint8 → (3, H, W) float32 in [0, 1]
    tensor = torch.tensor(
        image_rgb_uint8 / 255.0, dtype=torch.float32, device=_device
    ).permute(2, 0, 1)

    with torch.no_grad():
        output = model.infer(tensor, apply_mask=True)

    # -----------------------------------------------------------------------
    # Coordinate conversion: OpenCV → OpenGL
    # MoGe: (x right, y down, z forward)
    # Want: (x right, y up,   -z forward)
    # -----------------------------------------------------------------------
    pts3d = output["points"].cpu().float().numpy()  # (H, W, 3), OpenCV
    pts3d[..., 1] *= -1  # y down  → y up
    pts3d[..., 2] *= -1  # z fwd   → -z fwd (OpenGL)

    # -----------------------------------------------------------------------
    # Depth normalisation for MoGe-1 (relative depth only)
    # -----------------------------------------------------------------------
    if not _IS_METRIC:
        mask = (
            output["mask"].cpu().numpy().astype(bool)
            if "mask" in output
            else np.ones((h, w), dtype=bool)
        )
        valid_pts = pts3d[mask]
        if valid_pts.shape[0] > 0:
            mean_dist = float(np.mean(np.linalg.norm(valid_pts, axis=-1)))
            if mean_dist > 0.0:
                pts3d = pts3d / mean_dist

    # -----------------------------------------------------------------------
    # Intrinsics: normalised → pixel space
    # MoGe: K_norm where image ∈ [0,1]², so K_px = diag(W, H, 1) @ K_norm
    # -----------------------------------------------------------------------
    K_norm = output["intrinsics"].cpu().float().numpy()  # (3, 3)
    K_px = K_norm.copy()
    K_px[0, 0] *= w  # fx
    K_px[1, 1] *= h  # fy
    K_px[0, 2] *= w  # cx
    K_px[1, 2] *= h  # cy

    return {
        "pts3d": pts3d.astype(np.float32),
        "intrinsics": K_px.astype(np.float32),
    }
