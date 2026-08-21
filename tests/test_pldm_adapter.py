"""Tests for vdaworld.eval.pldm_adapter."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from vdaworld.eval.pldm_adapter import adapt_pldm_to_sim, decode_pldm_image


def _make_pldm_b64(size: int = 64) -> str:
    """Build a synthetic PLDM-shaped PNG: white bg, single red pixel at (10, 20),
    black vertical bar at x=32."""
    arr = np.full((size, size, 3), 255, dtype=np.uint8)
    arr[10, 20] = [255, 0, 0]
    arr[:, 32] = [0, 0, 0]
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_decode_returns_uint8_hwc():
    b64 = _make_pldm_b64(64)
    out = decode_pldm_image(b64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_adapt_resizes_to_target_size():
    b64 = _make_pldm_b64(64)
    out = adapt_pldm_to_sim(b64, target_size=(65, 65))
    assert out.shape == (65, 65, 3)
    assert out.dtype == np.uint8


def test_adapt_preserves_red_pixel_and_wall_under_nearest_neighbour():
    b64 = _make_pldm_b64(64)
    out = adapt_pldm_to_sim(b64, target_size=(65, 65))
    # Red pixel must survive (not bilinear-smeared into pink).
    red_mask = (out[:, :, 0] == 255) & (out[:, :, 1] == 0) & (out[:, :, 2] == 0)
    assert red_mask.sum() >= 1, "red dot lost in resize"
    # Black wall must survive as exact black somewhere.
    black_mask = (out == [0, 0, 0]).all(axis=-1)
    assert black_mask.sum() > 0, "wall lost in resize"


def test_adapt_is_noop_when_already_target_size():
    b64 = _make_pldm_b64(65)
    out = adapt_pldm_to_sim(b64, target_size=(65, 65))
    assert out.shape == (65, 65, 3)
    # Should be exactly the decoded input (no resize call).
    direct = decode_pldm_image(b64)
    np.testing.assert_array_equal(out, direct)
