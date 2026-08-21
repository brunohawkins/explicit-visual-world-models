"""HTTP client and minimal VDA session for the isolated SWM reference service.

The service always transports lossless 224×224 RGB PNGs.  VDA simulators may
use another frame size, so this client performs one documented conversion:
Pillow bilinear resizing from native 224 pixels to ``sim_frame_size``.  The
``sim_frame_size`` tuple follows the rest of the evaluation sessions and is
ordered as ``(width, height)``.
"""

from __future__ import annotations

import base64
import copy
import io
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from vdaworld.eval.pusht_adapter import strip_green_target
from vdaworld.eval.swm_reference_server_launcher import SWMReferenceServer


SWM_NATIVE_IMAGE_SIZE = 224
PUSHT_NATIVE_RANGE = 512.0
PUSHT_ACTION_SCALE = 100.0
SUPPORTED_BENCHMARKS = ("pusht", "two_room_swm", "reacher", "cube")
ACTION_DIMS = {"pusht": 2, "two_room_swm": 2, "reacher": 2, "cube": 5}


def decode_swm_png(encoded: str) -> np.ndarray:
    """Decode a base64 RGB PNG to an ``(H, W, 3)`` uint8 array."""

    if not isinstance(encoded, str):
        raise TypeError(f"encoded PNG must be a string, got {type(encoded).__name__}")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid base64 PNG payload") from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError("decoded payload is not a valid image") from exc
    return array


def resize_swm_image(
    image: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize an RGB image to ``(width, height)`` with bilinear interpolation."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "SWM image must be an (H, W, 3) uint8 array, got "
            f"shape={array.shape}, dtype={array.dtype}"
        )
    width, height = _frame_size(target_size)
    if array.shape[:2] == (height, width):
        return array.copy()
    resized = Image.fromarray(array, mode="RGB").resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(resized, dtype=np.uint8)


def adapt_swm_png(
    encoded: str,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Decode a server PNG and bilinearly resize it for a VDA simulator."""

    return resize_swm_image(decode_swm_png(encoded), target_size)


def _frame_size(value: tuple[int, int]) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError("sim_frame_size must be a (width, height) pair")
    width, height = value
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, np.integer))
        or not isinstance(height, (int, np.integer))
    ):
        raise ValueError("sim_frame_size entries must be integers")
    if int(width) <= 0 or int(height) <= 0:
        raise ValueError("sim_frame_size entries must be positive")
    return int(width), int(height)


def swm_reference_action_bounds(
    benchmark: str,
    sim_frame_size: tuple[int, int] = (
        SWM_NATIVE_IMAGE_SIZE,
        SWM_NATIVE_IMAGE_SIZE,
    ),
) -> tuple[np.ndarray, np.ndarray]:
    """Return VDA-planner action bounds for one SWM reference benchmark."""

    if benchmark not in SUPPORTED_BENCHMARKS:
        raise ValueError(
            f"unsupported SWM benchmark {benchmark!r}; "
            f"choose one of {SUPPORTED_BENCHMARKS}"
        )
    width, _ = _frame_size(sim_frame_size)
    if benchmark == "pusht":
        return (
            np.zeros(2, dtype=np.float64),
            np.full(2, float(width), dtype=np.float64),
        )
    dimension = ACTION_DIMS[benchmark]
    return (
        -np.ones(dimension, dtype=np.float64),
        np.ones(dimension, dtype=np.float64),
    )


def bridge_pusht_action(
    action_sim: Any,
    current_agent_native: Any,
    *,
    sim_width: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Convert a VDA absolute-pixel target to SWM's relative Push-T action.

    ``action_sim`` is an absolute pusher target in ``[0, sim_width]``.  It is
    scaled to the native 512-pixel world, converted to a displacement from the
    current native pusher position, divided by SWM's action scale (100), and
    finally clipped to the native ``[-1, 1]`` action box.
    """

    if isinstance(sim_width, bool) or not isinstance(sim_width, (int, np.integer)):
        raise ValueError("sim_width must be a positive integer")
    if int(sim_width) <= 0:
        raise ValueError("sim_width must be a positive integer")
    raw_action = np.asarray(action_sim, dtype=np.float64)
    current_agent = np.asarray(current_agent_native, dtype=np.float64)
    if raw_action.shape != (2,):
        raise ValueError(f"Push-T action must have shape (2,), got {raw_action.shape}")
    if current_agent.shape != (2,):
        raise ValueError(
            "Push-T current agent position must have shape (2,), got "
            f"{current_agent.shape}"
        )
    if not np.all(np.isfinite(raw_action)) or not np.all(np.isfinite(current_agent)):
        raise ValueError("Push-T action and current agent position must be finite")
    if np.any(raw_action < 0.0) or np.any(raw_action > float(sim_width)):
        raise ValueError(
            f"Push-T simulator action must lie in [0, {sim_width}], "
            f"got {raw_action.tolist()}"
        )

    native_target = raw_action * (PUSHT_NATIVE_RANGE / float(sim_width))
    native_unclipped = (native_target - current_agent) / PUSHT_ACTION_SCALE
    native_action = np.clip(native_unclipped, -1.0, 1.0).astype(np.float32)
    saturation_mask = np.abs(native_unclipped) > 1.0
    metadata = {
        "kind": "pusht_absolute_sim_pixels_to_swm_relative",
        "raw_action": raw_action.tolist(),
        "native_target": native_target.tolist(),
        "current_agent_native": current_agent.tolist(),
        "native_action_unclipped": native_unclipped.tolist(),
        "native_action": native_action.tolist(),
        "saturation_mask": saturation_mask.tolist(),
        "action_saturated": bool(np.any(saturation_mask)),
    }
    return native_action, metadata


class SWMReferenceClient:
    """Thin requests-based client for the reference service protocol."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        http_session: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._http = http_session or requests.Session()
        self._owns_http = http_session is None

    def _post(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        response = self._http.post(
            f"{self.base_url}{endpoint}",
            json=body,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError(
                f"SWM server {endpoint} returned non-object JSON: {payload!r}"
            )
        return payload

    def start(
        self,
        *,
        benchmark: str,
        manifest_case: dict[str, Any],
        hdf5_path: str | Path,
        seed: int | None = None,
        task_protocol: str = "original",
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "benchmark": benchmark,
            "manifest_case": manifest_case,
            "hdf5_path": str(hdf5_path),
            "task_protocol": str(task_protocol),
        }
        if seed is not None:
            body["seed"] = int(seed)
        return self._post("/start", body)

    def step(self, session_id: str, action: Any) -> dict[str, Any]:
        return self._post(
            "/step",
            {
                "session_id": session_id,
                "action": np.asarray(action).tolist(),
            },
        )

    def close_session(self, session_id: str) -> dict[str, Any]:
        return self._post("/close", {"session_id": session_id})

    def close(self, session_id: str) -> dict[str, Any]:
        """Close a remote session (alias matching the HTTP endpoint name)."""

        return self.close_session(session_id)

    def shutdown(self) -> None:
        if self._owns_http and hasattr(self._http, "close"):
            self._http.close()


class SWMReferenceSession:
    """Minimal ``start/step/close`` VDA session backed by native SWM."""

    def __init__(
        self,
        *,
        benchmark: str,
        manifest_case: dict[str, Any],
        hdf5_path: str | Path,
        sim_frame_size: tuple[int, int] = (
            SWM_NATIVE_IMAGE_SIZE,
            SWM_NATIVE_IMAGE_SIZE,
        ),
        server: SWMReferenceServer | None = None,
        port: int | None = None,
        python_path: str | Path | None = None,
        server_script: str | Path | None = None,
        startup_timeout_s: float = 120.0,
        request_timeout_s: float = 120.0,
        http_session: Any | None = None,
        strip_green_target_from_sim: bool = False,
        task_protocol: str = "original",
    ) -> None:
        if benchmark not in SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"unsupported SWM benchmark {benchmark!r}; "
                f"choose one of {SUPPORTED_BENCHMARKS}"
            )
        if not isinstance(manifest_case, dict):
            raise TypeError("manifest_case must be a dictionary")
        self.benchmark = benchmark
        self.manifest_case = copy.deepcopy(manifest_case)
        self.hdf5_path = str(hdf5_path)
        self.sim_frame_size = _frame_size(sim_frame_size)
        self._server = server
        self._own_server = server is None
        self._port = port
        self._python_path = python_path
        self._server_script = server_script
        self._startup_timeout_s = float(startup_timeout_s)
        self._request_timeout_s = float(request_timeout_s)
        self._http_session = http_session
        self.strip_green_target_from_sim = bool(strip_green_target_from_sim)
        if task_protocol not in {"original", "task_success_v1"}:
            raise ValueError(
                "task_protocol must be 'original' or 'task_success_v1'"
            )
        self.task_protocol = task_protocol
        self._client: SWMReferenceClient | None = None
        self._session_id: str | None = None
        self._goal_raw: np.ndarray | None = None
        self._goal_sim: np.ndarray | None = None
        self._current_agent_native: np.ndarray | None = None

    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return swm_reference_action_bounds(
            self.benchmark,
            self.sim_frame_size,
        )

    @staticmethod
    def _native_frame(encoded: Any, *, label: str) -> np.ndarray:
        frame = decode_swm_png(encoded)
        expected = (SWM_NATIVE_IMAGE_SIZE, SWM_NATIVE_IMAGE_SIZE, 3)
        if frame.shape != expected:
            raise ValueError(
                f"SWM {label} must decode to shape {expected}, got {frame.shape}"
            )
        return frame

    def _update_current_agent(self, info: dict[str, Any]) -> None:
        if self.benchmark != "pusht":
            return
        position = None
        current_state = info.get("current_state")
        if isinstance(current_state, dict):
            position = current_state.get("pos_agent")
        if position is None:
            position = info.get("pos_agent")
        if position is None:
            raise ValueError("Push-T server info is missing current pos_agent")
        array = np.asarray(position, dtype=np.float64)
        if array.shape != (2,) or not np.all(np.isfinite(array)):
            raise ValueError(
                "Push-T server pos_agent must be a finite shape-(2,) vector"
            )
        self._current_agent_native = array.copy()

    def _pack(
        self,
        payload: dict[str, Any],
        *,
        require_goal: bool,
    ) -> dict[str, Any]:
        if "image" not in payload:
            raise ValueError("SWM server response is missing image")
        image_raw = self._native_frame(payload["image"], label="image")
        image_for_sim = (
            strip_green_target(image_raw)
            if self.benchmark == "pusht" and self.strip_green_target_from_sim
            else image_raw
        )
        image_sim = resize_swm_image(image_for_sim, self.sim_frame_size)

        if "goal_image" in payload:
            self._goal_raw = self._native_frame(
                payload["goal_image"],
                label="goal_image",
            )
            goal_for_sim = (
                strip_green_target(self._goal_raw)
                if self.benchmark == "pusht"
                and self.strip_green_target_from_sim
                else self._goal_raw
            )
            self._goal_sim = resize_swm_image(goal_for_sim, self.sim_frame_size)
        elif require_goal or self._goal_raw is None or self._goal_sim is None:
            raise ValueError("SWM server start response is missing goal_image")

        info = payload.get("info", {})
        if not isinstance(info, dict):
            raise ValueError("SWM server response info must be an object")
        info = dict(info)
        self._update_current_agent(info)
        return {
            "image": image_sim,
            "image_raw": image_raw,
            "goal_image": self._goal_sim,
            "goal_image_raw": self._goal_raw,
            "info": info,
            "reward": float(payload.get("reward", 0.0)),
            "done": bool(payload.get("done", False)),
            "terminated": bool(payload.get("terminated", False)),
            "truncated": bool(payload.get("truncated", False)),
        }

    def _launch_owned_server(self) -> None:
        kwargs: dict[str, Any] = {
            "port": self._port,
            "python_path": self._python_path,
            "startup_timeout_s": self._startup_timeout_s,
        }
        if self._server_script is not None:
            kwargs["server_script"] = self._server_script
        self._server = SWMReferenceServer(**kwargs)
        self._server.start()

    def start(self, seed: int | None = None) -> dict[str, Any]:
        if self._session_id is not None:
            raise RuntimeError("SWM reference session is already started")
        if self._own_server:
            if self._server is not None:
                raise RuntimeError("owned SWM server is in an invalid state")
            self._launch_owned_server()
        if self._server is None:
            raise RuntimeError("SWM reference server is not configured")

        self._client = SWMReferenceClient(
            self._server.base_url,
            timeout_s=self._request_timeout_s,
            http_session=self._http_session,
        )
        try:
            payload = self._client.start(
                benchmark=self.benchmark,
                manifest_case=self.manifest_case,
                hdf5_path=self.hdf5_path,
                seed=seed,
                task_protocol=self.task_protocol,
            )
            session_id = payload.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError("SWM server start response is missing session_id")
            self._session_id = session_id
            return self._pack(payload, require_goal=True)
        except Exception:
            self._session_id = None
            self._client.shutdown()
            self._client = None
            if self._own_server and self._server is not None:
                self._server.stop()
                self._server = None
            raise

    def _bounded_native_action(self, action_sim: Any) -> np.ndarray:
        action = np.asarray(action_sim, dtype=np.float64)
        low, high = self.action_bounds()
        if action.shape != low.shape:
            raise ValueError(
                f"{self.benchmark} action must have shape {low.shape}, "
                f"got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{self.benchmark} action must be finite")
        if np.any(action < low) or np.any(action > high):
            raise ValueError(
                f"{self.benchmark} action is outside bounds: "
                f"action={action.tolist()}, low={low.tolist()}, "
                f"high={high.tolist()}"
            )
        return action

    def step(self, action_sim: Any) -> dict[str, Any]:
        if self._client is None or self._session_id is None:
            raise RuntimeError("call start() before step()")

        bridge_info: dict[str, Any] | None = None
        if self.benchmark == "pusht":
            if self._current_agent_native is None:
                raise RuntimeError("Push-T current agent state is unavailable")
            native_action, bridge_info = bridge_pusht_action(
                action_sim,
                self._current_agent_native,
                sim_width=self.sim_frame_size[0],
            )
        else:
            # Two-Room, Reacher and Cube use SWM's bounded native action space.
            native_action = self._bounded_native_action(action_sim)

        payload = self._client.step(self._session_id, native_action)
        packed = self._pack(payload, require_goal=False)
        if bridge_info is not None:
            packed["info"]["action_bridge"] = bridge_info
            packed["info"]["raw_action"] = bridge_info["raw_action"]
            packed["info"]["native_action"] = bridge_info["native_action"]
            packed["info"]["action_saturated"] = bridge_info["action_saturated"]
        return packed

    def close(self) -> None:
        error: Exception | None = None
        try:
            if self._client is not None and self._session_id is not None:
                try:
                    self._client.close_session(self._session_id)
                except Exception as exc:
                    error = exc
        finally:
            self._session_id = None
            self._goal_raw = None
            self._goal_sim = None
            self._current_agent_native = None
            if self._client is not None:
                self._client.shutdown()
                self._client = None
            if self._own_server and self._server is not None:
                self._server.stop()
                self._server = None
        if error is not None:
            raise error

    def __enter__(self) -> "SWMReferenceSession":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            self.close()
        except Exception:
            if exc is None:
                raise


# Concise alias for call sites that use a generic ``action_bounds`` helper.
action_bounds = swm_reference_action_bounds


__all__ = [
    "SWMReferenceClient",
    "SWMReferenceSession",
    "action_bounds",
    "adapt_swm_png",
    "bridge_pusht_action",
    "decode_swm_png",
    "resize_swm_image",
    "swm_reference_action_bounds",
]
