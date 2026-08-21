"""Unit tests for SWM PNG adaptation and the Push-T action bridge."""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from vdaworld.eval.pusht_adapter import green_target_mask
from vdaworld.eval.swm_reference_session import (
    SWMReferenceSession,
    bridge_pusht_action,
    decode_swm_png,
    resize_swm_image,
    swm_reference_action_bounds,
)


def _image() -> np.ndarray:
    y, x = np.mgrid[:224, :224]
    return np.stack(
        (
            x.astype(np.uint8),
            y.astype(np.uint8),
            ((x + y) % 256).astype(np.uint8),
        ),
        axis=-1,
    )


def _png(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _HTTP:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def post(self, url, *, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return _Response(self.payloads.pop(0))


class _SharedServer:
    base_url = "http://reference.test:8123"


def test_png_decode_and_resize_are_uint8_rgb_bilinear():
    source = _image()
    decoded = decode_swm_png(_png(source))
    np.testing.assert_array_equal(decoded, source)

    resized = resize_swm_image(decoded, (96, 64))
    expected = np.asarray(
        Image.fromarray(source, mode="RGB").resize(
            (96, 64),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )
    assert resized.shape == (64, 96, 3)
    assert resized.dtype == np.uint8
    np.testing.assert_array_equal(resized, expected)

    native = resize_swm_image(decoded, (224, 224))
    np.testing.assert_array_equal(native, decoded)


def test_pusht_bridge_scales_relative_action_and_reports_saturation():
    native, metadata = bridge_pusht_action(
        [96.0, 0.0],
        [0.0, 512.0],
        sim_width=96,
    )

    np.testing.assert_allclose(native, [1.0, -1.0])
    np.testing.assert_allclose(metadata["native_target"], [512.0, 0.0])
    np.testing.assert_allclose(
        metadata["native_action_unclipped"],
        [5.12, -5.12],
    )
    assert metadata["saturation_mask"] == [True, True]
    assert metadata["action_saturated"] is True

    native_224, metadata_224 = bridge_pusht_action(
        [112.0, 112.0],
        [256.0, 256.0],
        sim_width=224,
    )
    np.testing.assert_allclose(metadata_224["native_target"], [256.0, 256.0])
    np.testing.assert_allclose(native_224, [0.0, 0.0])


def test_session_decodes_resizes_and_sends_pusht_native_action():
    encoded = _png(_image())
    start_payload = {
        "session_id": "session-1",
        "image": encoded,
        "goal_image": encoded,
        "info": {
            "current_state": {"pos_agent": [200.0, 300.0]},
            "primary_success": False,
        },
        "reward": 0.0,
        "done": False,
        "terminated": False,
        "truncated": False,
    }
    step_payload = {
        "session_id": "session-1",
        "image": encoded,
        "goal_image": encoded,
        "info": {
            "current_state": {"pos_agent": [210.0, 290.0]},
            "primary_success": False,
        },
        "reward": -1.5,
        "done": False,
        "terminated": False,
        "truncated": False,
    }
    http = _HTTP([start_payload, step_payload, {"closed": True}])
    session = SWMReferenceSession(
        benchmark="pusht",
        manifest_case={"case_id": "case-1", "dataset_row": 10, "goal_dataset_row": 20},
        hdf5_path="/reference/data.h5",
        sim_frame_size=(96, 64),
        server=_SharedServer(),
        http_session=http,
    )

    start = session.start(seed=7)
    assert start["image"].shape == (64, 96, 3)
    assert start["image_raw"].shape == (224, 224, 3)
    assert start["goal_image"].shape == (64, 96, 3)
    assert http.calls[0]["json"] == {
        "benchmark": "pusht",
        "manifest_case": {
            "case_id": "case-1",
            "dataset_row": 10,
            "goal_dataset_row": 20,
        },
        "hdf5_path": "/reference/data.h5",
        "task_protocol": "original",
        "seed": 7,
    }

    result = session.step([48.0, 48.0])
    sent_action = http.calls[1]["json"]["action"]
    np.testing.assert_allclose(sent_action, [0.56, -0.44], atol=1e-7)
    bridge = result["info"]["action_bridge"]
    assert bridge["raw_action"] == [48.0, 48.0]
    np.testing.assert_allclose(bridge["native_target"], [256.0, 256.0])
    assert bridge["action_saturated"] is False
    assert result["reward"] == -1.5

    session.close()
    assert http.calls[2]["json"] == {"session_id": "session-1"}


def test_action_bounds_match_vda_and_native_spaces():
    pusht_low, pusht_high = swm_reference_action_bounds("pusht", (96, 64))
    np.testing.assert_array_equal(pusht_low, [0.0, 0.0])
    np.testing.assert_array_equal(pusht_high, [96.0, 96.0])
    pusht_224_low, pusht_224_high = swm_reference_action_bounds(
        "pusht", (224, 224)
    )
    np.testing.assert_array_equal(pusht_224_low, [0.0, 0.0])
    np.testing.assert_array_equal(pusht_224_high, [224.0, 224.0])

    two_room_low, two_room_high = swm_reference_action_bounds("two_room_swm")
    np.testing.assert_array_equal(two_room_low, [-1.0, -1.0])
    np.testing.assert_array_equal(two_room_high, [1.0, 1.0])

    reacher_low, reacher_high = swm_reference_action_bounds("reacher")
    np.testing.assert_array_equal(reacher_low, [-1.0, -1.0])
    np.testing.assert_array_equal(reacher_high, [1.0, 1.0])

    cube_low, cube_high = swm_reference_action_bounds("cube")
    np.testing.assert_array_equal(cube_low, -np.ones(5))
    np.testing.assert_array_equal(cube_high, np.ones(5))


def test_pusht_session_strips_green_only_from_simulator_frames():
    source = np.full((224, 224, 3), 255, dtype=np.uint8)
    source[40:100, 80:140] = [144, 238, 144]
    source[65:80, 95:125] = [143, 163, 184]
    source[20:30, 20:30] = [78, 126, 255]
    encoded = _png(source)
    payload = {
        "session_id": "session-green",
        "image": encoded,
        "goal_image": encoded,
        "info": {"current_state": {"pos_agent": [100.0, 100.0]}},
        "done": False,
        "terminated": False,
        "truncated": False,
    }
    http = _HTTP([payload, {"closed": True}])
    session = SWMReferenceSession(
        benchmark="pusht",
        manifest_case={"case_id": "green-case"},
        hdf5_path="/reference/data.h5",
        sim_frame_size=(96, 96),
        server=_SharedServer(),
        http_session=http,
        strip_green_target_from_sim=True,
    )

    start = session.start(seed=1)

    assert green_target_mask(start["image_raw"]).any()
    assert green_target_mask(start["goal_image_raw"]).any()
    assert not green_target_mask(start["image"]).any()
    assert not green_target_mask(start["goal_image"]).any()
    session.close()
