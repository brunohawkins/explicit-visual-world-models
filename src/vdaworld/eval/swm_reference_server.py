"""Standalone Stable WorldModel reference-environment HTTP service.

This file is launched directly with the Python environment from the sibling
``le-wm`` checkout.  It intentionally has no dependency on ``vdaworld`` so the
reference environments and their pinned dependencies remain process-isolated.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import threading
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import hdf5plugin  # noqa: F401  (register HDF5 compression filters before reads)
import gymnasium as gym
import h5py
import numpy as np
import stable_worldmodel  # noqa: F401  (registers the ``swm/*`` environments)
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from shapely.geometry import Polygon
from shapely.ops import unary_union


EXPECTED_SWM_VERSION = "0.1.1"
SWM_VERSION = version("stable-worldmodel")
if SWM_VERSION != EXPECTED_SWM_VERSION:
    raise RuntimeError(
        "The SWM reference server requires stable_worldmodel "
        f"{EXPECTED_SWM_VERSION}, found {SWM_VERSION}."
    )

NATIVE_IMAGE_SIZE = 224
MAX_EPISODE_STEPS = 100
BENCHMARKS = ("pusht", "two_room_swm", "reacher", "cube")
REACHER_WRIST_LIMIT = math.radians(160.0)
REACHER_IK_ENDPOINT_TOLERANCE = 1.0e-5

ENV_IDS = {
    "pusht": "swm/PushT-v1",
    "two_room_swm": "swm/TwoRoom-v1",
    "reacher": "swm/ReacherDMControl-v0",
    "cube": "swm/OGBCube-v0",
}
ENV_OPTIONS: dict[str, dict[str, Any]] = {
    "pusht": {
        "relative": True,
        "resolution": NATIVE_IMAGE_SIZE,
        "render_mode": "rgb_array",
    },
    "two_room_swm": {"render_mode": "rgb_array"},
    "reacher": {"task": "qpos_match"},
    "cube": {
        "env_type": "single",
        "ob_type": "states",
        "height": NATIVE_IMAGE_SIZE,
        "width": NATIVE_IMAGE_SIZE,
        "multiview": False,
        "visualize_info": False,
        "terminate_at_goal": True,
    },
}


class StartRequest(BaseModel):
    """Create one dataset-defined reference-environment session."""

    model_config = ConfigDict(extra="forbid")

    benchmark: Literal["pusht", "two_room_swm", "reacher", "cube"]
    manifest_case: dict[str, Any]
    hdf5_path: str = Field(min_length=1)
    seed: int | None = None
    task_protocol: Literal["original", "task_success_v1"] = "original"


class StepRequest(BaseModel):
    """Advance a session by one native SWM action."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)
    action: list[float]


class CloseRequest(BaseModel):
    """Close one reference-environment session."""

    model_config = ConfigDict(extra="forbid")

    session_id: str = Field(min_length=1)


@dataclass
class ReferenceSession:
    session_id: str
    benchmark: str
    env: gym.Env
    hdf5_path: str
    dataset_row: int
    goal_dataset_row: int
    manifest_case_id: Any
    goal_state: dict[str, Any]
    goal_image: str
    task_protocol: str = "original"
    env_step: int = 0
    finished: bool = False


_LOCK = threading.RLock()
_SESSIONS: dict[str, ReferenceSession] = {}
_HDF5_FILES: dict[str, h5py.File] = {}


def _jsonable(value: Any) -> Any:
    """Convert nested NumPy/environment values to strict JSON values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError(f"cannot encode value of type {type(value).__name__} as JSON")


def _immutable_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    array.setflags(write=False)
    return array


def _encode_png(frame: Any, *, source: str) -> str:
    array = np.asarray(frame)
    expected = (NATIVE_IMAGE_SIZE, NATIVE_IMAGE_SIZE, 3)
    if array.shape != expected:
        raise ValueError(
            f"{source} image must have shape {expected}, got {array.shape}"
        )
    if array.dtype != np.uint8:
        raise ValueError(f"{source} image must be uint8, got {array.dtype}")
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _open_hdf5(path: str) -> tuple[str, h5py.File]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"HDF5 path is not a file: {resolved}")
    key = str(resolved)
    handle = _HDF5_FILES.get(key)
    if handle is None:
        handle = h5py.File(
            resolved,
            "r",
            swmr=True,
            rdcc_nbytes=256 * 1024 * 1024,
        )
        _HDF5_FILES[key] = handle
    elif not handle.id.valid:
        raise RuntimeError(f"cached HDF5 handle is no longer valid: {resolved}")
    return key, handle


def _required_int(mapping: dict[str, Any], key: str, *, where: str) -> int:
    if key not in mapping:
        raise ValueError(f"{where} is missing required integer field {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{where}.{key} must be an integer, got {value!r}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{where}.{key} must be non-negative, got {result}")
    return result


def _validate_row(handle: h5py.File, row: int, *, label: str) -> None:
    if "pixels" not in handle:
        raise ValueError("HDF5 file is missing required 'pixels' dataset")
    row_count = int(handle["pixels"].shape[0])
    if row >= row_count:
        raise ValueError(f"{label}={row} is outside pixels row count {row_count}")


def _resolve_rows(
    handle: h5py.File,
    benchmark: str,
    manifest_case: dict[str, Any],
) -> tuple[int, int]:
    has_start = "dataset_row" in manifest_case
    has_goal = "goal_dataset_row" in manifest_case
    if has_start or has_goal:
        if not (has_start and has_goal):
            raise ValueError(
                "manifest_case must provide both dataset_row and goal_dataset_row"
            )
        start_row = _required_int(manifest_case, "dataset_row", where="manifest_case")
        goal_row = _required_int(
            manifest_case, "goal_dataset_row", where="manifest_case"
        )
    elif benchmark == "pusht":
        metadata = manifest_case.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError(
                "Push-T manifest_case needs dataset rows or a metadata object"
            )
        episode_idx = _required_int(metadata, "episode_idx", where="metadata")
        start_step = _required_int(metadata, "start_step", where="metadata")
        if "goal_step" in metadata:
            goal_step = _required_int(metadata, "goal_step", where="metadata")
        elif "goal_step_idx" in metadata:
            goal_step = _required_int(metadata, "goal_step_idx", where="metadata")
        elif "goal_offset_steps" in metadata:
            goal_step = start_step + _required_int(
                metadata, "goal_offset_steps", where="metadata"
            )
        else:
            raise ValueError(
                "Push-T metadata needs goal_step, goal_step_idx, or goal_offset_steps"
            )
        if "ep_offset" not in handle:
            raise ValueError(
                "Push-T row resolution requires the HDF5 'ep_offset' dataset"
            )
        if episode_idx >= int(handle["ep_offset"].shape[0]):
            raise ValueError(f"metadata.episode_idx={episode_idx} is outside ep_offset")
        if "ep_len" in handle:
            episode_length = int(handle["ep_len"][episode_idx])
            if start_step >= episode_length or goal_step >= episode_length:
                raise ValueError(
                    "Push-T start/goal step lies outside the selected episode: "
                    f"start={start_step}, goal={goal_step}, length={episode_length}"
                )
        offset = int(handle["ep_offset"][episode_idx])
        start_row = offset + start_step
        goal_row = offset + goal_step

        if "episode_idx" in handle:
            for row, label in ((start_row, "start"), (goal_row, "goal")):
                actual_episode = int(handle["episode_idx"][row])
                if actual_episode != episode_idx:
                    raise ValueError(
                        f"Push-T {label} row belongs to episode "
                        f"{actual_episode}, expected {episode_idx}"
                    )
        if "step_idx" in handle:
            actual_start = int(handle["step_idx"][start_row])
            actual_goal = int(handle["step_idx"][goal_row])
            if actual_start != start_step or actual_goal != goal_step:
                raise ValueError(
                    "Push-T ep_offset row resolution disagrees with step_idx: "
                    f"got ({actual_start}, {actual_goal}), expected "
                    f"({start_step}, {goal_step})"
                )
    else:
        raise ValueError(
            f"{benchmark} manifest_case requires dataset_row and goal_dataset_row"
        )

    _validate_row(handle, start_row, label="dataset_row")
    _validate_row(handle, goal_row, label="goal_dataset_row")
    return start_row, goal_row


def _row_vector(
    handle: h5py.File,
    key: str,
    row: int,
    *,
    length: int | None = None,
) -> np.ndarray:
    if key not in handle:
        raise ValueError(f"HDF5 file is missing required {key!r} dataset")
    array = np.asarray(handle[key][row], dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"HDF5 {key}[{row}] must be one-dimensional")
    if length is not None and array.shape != (length,):
        raise ValueError(
            f"HDF5 {key}[{row}] must have shape ({length},), got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"HDF5 {key}[{row}] contains non-finite values")
    return array.copy()


def _hdf_pixels(handle: h5py.File, row: int) -> np.ndarray:
    frame = np.asarray(handle["pixels"][row])
    expected = (NATIVE_IMAGE_SIZE, NATIVE_IMAGE_SIZE, 3)
    if frame.shape != expected or frame.dtype != np.uint8:
        raise ValueError(
            f"HDF5 pixels[{row}] must be uint8 with shape {expected}, "
            f"got dtype={frame.dtype}, shape={frame.shape}"
        )
    return frame.copy()


def _make_env(benchmark: str) -> gym.Env:
    return gym.make(
        ENV_IDS[benchmark],
        max_episode_steps=MAX_EPISODE_STEPS,
        **ENV_OPTIONS[benchmark],
    )


def _push_state(state: np.ndarray) -> dict[str, Any]:
    return {
        "full_state": state,
        "pos_agent": state[:2],
        "block_pose": state[2:5],
        "vel_agent": state[5:7],
    }


def _initialise_pusht(
    env: gym.Env,
    handle: h5py.File,
    start_row: int,
    goal_row: int,
) -> dict[str, Any]:
    start_state = _row_vector(handle, "state", start_row, length=7)
    goal_state = _immutable_array(_row_vector(handle, "state", goal_row, length=7))
    env.reset()
    raw = env.unwrapped
    raw._set_state(start_state)
    raw._set_goal_state(goal_state.copy())
    return _push_state(goal_state)


def _initialise_two_room(
    env: gym.Env,
    handle: h5py.File,
    start_row: int,
    goal_row: int,
) -> dict[str, Any]:
    start_state = _row_vector(handle, "proprio", start_row, length=2)
    goal_state = _immutable_array(
        _row_vector(handle, "proprio", goal_row, length=2)
    )
    env.reset()
    raw = env.unwrapped
    raw._set_goal_state(goal_state.copy())
    raw._set_state(start_state)
    return {"state": goal_state, "pos_agent": goal_state}


def _initialise_reacher(
    env: gym.Env,
    handle: h5py.File,
    start_row: int,
    goal_row: int,
) -> dict[str, Any]:
    start_qpos = _row_vector(handle, "qpos", start_row)
    start_qvel = _row_vector(handle, "qvel", start_row)
    goal_qpos = _immutable_array(_row_vector(handle, "qpos", goal_row))
    goal_qvel = _immutable_array(_row_vector(handle, "qvel", goal_row))
    env.reset()
    raw = env.unwrapped
    raw.set_state(start_qpos, start_qvel)
    raw.set_target_qpos(goal_qpos.copy())
    return {"qpos": goal_qpos, "qvel": goal_qvel}


def _set_reacher_visible_target(env: gym.Env, target_pos: np.ndarray) -> None:
    """Place and reveal the target used by task-centric Reacher evaluation."""
    physics = env.unwrapped.env.physics
    target = np.asarray(target_pos, dtype=np.float64).reshape(2)
    physics.named.model.geom_pos["target", :2] = target
    physics.named.model.mat_rgba["target", 3] = 1.0
    physics.forward()


def _select_reacher_goal_qpos(
    candidates: list[tuple[np.ndarray, float]],
    reference_qpos: np.ndarray,
    *,
    endpoint_error_tolerance: float = REACHER_IK_ENDPOINT_TOLERANCE,
    wrist_limit: float = REACHER_WRIST_LIMIT,
) -> tuple[np.ndarray, float]:
    """Choose a near-optimal IK pose on the wrist branch nearest the start."""
    reference = np.asarray(reference_qpos, dtype=np.float64).reshape(-1)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_qpos must be a finite shape-(2,) vector")
    tolerance = float(endpoint_error_tolerance)
    limit = float(wrist_limit)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("endpoint_error_tolerance must be finite and non-negative")
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("wrist_limit must be finite and positive")

    feasible: list[tuple[np.ndarray, float]] = []
    for raw_qpos, raw_error in candidates:
        qpos = np.asarray(raw_qpos, dtype=np.float64).reshape(-1)
        error = float(raw_error)
        if qpos.shape != (2,) or not np.all(np.isfinite(qpos)) or not math.isfinite(error):
            continue
        canonical = np.array(
            [
                reference[0]
                + np.arctan2(
                    np.sin(qpos[0] - reference[0]),
                    np.cos(qpos[0] - reference[0]),
                ),
                np.arctan2(np.sin(qpos[1]), np.cos(qpos[1])),
            ],
            dtype=np.float64,
        )
        if abs(float(canonical[1])) <= limit + 1.0e-9:
            feasible.append((canonical, error))
    if not feasible:
        raise ValueError("IK produced no candidate within the Reacher wrist bounds")

    best_error = min(error for _, error in feasible)
    near_optimal = [
        (qpos, error)
        for qpos, error in feasible
        if error <= best_error + tolerance
    ]
    reference_wrist = float(np.clip(reference[1], -limit, limit))

    def branch_distance(item: tuple[np.ndarray, float]) -> tuple[float, float]:
        qpos, error = item
        shoulder_delta = np.arctan2(
            np.sin(qpos[0] - reference[0]),
            np.cos(qpos[0] - reference[0]),
        )
        wrist_delta = float(qpos[1] - reference_wrist)
        return float(shoulder_delta**2 + wrist_delta**2), error

    selected_qpos, selected_error = min(near_optimal, key=branch_distance)
    return selected_qpos.copy(), float(selected_error)


def _solve_reacher_target_qpos(env: gym.Env, target_pos: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve a feasible two-joint pose whose fingertip reaches ``target_pos``."""
    from scipy.optimize import minimize

    physics = env.unwrapped.env.physics
    saved_qpos = physics.data.qpos.copy()
    saved_qvel = physics.data.qvel.copy()
    target = np.asarray(target_pos, dtype=np.float64).reshape(2)

    def objective(qpos: np.ndarray) -> float:
        physics.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        physics.data.qvel[:] = 0.0
        physics.forward()
        finger = physics.named.data.geom_xpos["finger", :2]
        return float(np.linalg.norm(finger - target))

    starts = [
        saved_qpos,
        np.zeros_like(saved_qpos),
        np.array([0.0, np.pi / 2.0]),
        np.array([np.pi / 2.0, -np.pi / 2.0]),
        np.array([-np.pi / 2.0, np.pi / 2.0]),
        np.array([np.pi, -np.pi / 2.0]),
        np.array([-np.pi, np.pi / 2.0]),
    ]
    candidates = [(saved_qpos.copy(), objective(saved_qpos))]
    try:
        for start in starts:
            result = minimize(
                lambda q: objective(np.asarray(q, dtype=np.float64)),
                np.asarray(start, dtype=np.float64),
                method="Nelder-Mead",
                options={"maxiter": 400, "xatol": 1e-7, "fatol": 1e-7},
            )
            candidate = np.asarray(result.x, dtype=np.float64)
            candidates.append((candidate.copy(), objective(candidate)))
        best_qpos, best_error = _select_reacher_goal_qpos(
            candidates,
            saved_qpos,
        )
    finally:
        physics.data.qpos[:] = saved_qpos
        physics.data.qvel[:] = saved_qvel
        physics.forward()
    return best_qpos, float(best_error)


def _initialise_reacher_task(
    env: gym.Env,
    handle: h5py.File,
    start_row: int,
    manifest_case: dict[str, Any],
) -> dict[str, Any]:
    start_qpos = _row_vector(handle, "qpos", start_row)
    start_qvel = _row_vector(handle, "qvel", start_row)
    target_pos = np.asarray(
        manifest_case.get("dataset_target_pos"), dtype=np.float64
    )
    if target_pos.shape != (2,) or not np.all(np.isfinite(target_pos)):
        raise ValueError(
            "task_success_v1 Reacher cases require finite dataset_target_pos"
        )
    env.reset()
    raw = env.unwrapped
    _set_reacher_visible_target(env, target_pos)
    raw.set_state(start_qpos, start_qvel)
    goal_qpos, goal_ik_error = _solve_reacher_target_qpos(env, target_pos)
    raw.set_state(start_qpos, start_qvel)
    _set_reacher_visible_target(env, target_pos)
    return {
        "qpos": _immutable_array(goal_qpos),
        "qvel": _immutable_array(np.zeros_like(goal_qpos)),
        "target_pos": _immutable_array(target_pos),
        "goal_ik_error": float(goal_ik_error),
    }


def _initialise_pusht_task(
    env: gym.Env,
    handle: h5py.File,
    start_row: int,
) -> dict[str, Any]:
    start_state = _row_vector(handle, "state", start_row, length=7)
    env.reset()
    raw = env.unwrapped
    target_pose = np.asarray(raw.goal_pose, dtype=np.float64).reshape(3)
    requested_agent = np.asarray(start_state[:2], dtype=np.float64)
    goal_agent, pusher_relocated = _non_occluding_pusht_agent(
        target_pose, requested_agent
    )
    goal_state = np.concatenate(
        [goal_agent, target_pose, np.zeros(2, dtype=np.float64)]
    )
    raw._set_goal_state(goal_state.copy())
    raw._set_state(start_state)
    return {
        **_push_state(_immutable_array(goal_state)),
        "target_pose": _immutable_array(target_pose),
        "goal_pusher_requested": _immutable_array(requested_agent),
        "goal_pusher_relocated_for_visibility": bool(pusher_relocated),
    }


def _non_occluding_pusht_agent(
    target_pose: np.ndarray,
    requested_agent: np.ndarray,
) -> tuple[np.ndarray, bool]:
    """Move an overlapping goal-image pusher to the farthest safe corner."""
    target_xy = np.asarray(target_pose, dtype=np.float64).reshape(3)[:2]
    requested = np.asarray(requested_agent, dtype=np.float64).reshape(2)
    if float(np.linalg.norm(requested - target_xy)) >= 145.0:
        return requested.copy(), False
    candidates = np.array(
        [[30.0, 30.0], [482.0, 30.0], [30.0, 482.0], [482.0, 482.0]],
        dtype=np.float64,
    )
    distances = np.linalg.norm(candidates - target_xy[None, :], axis=1)
    return candidates[int(np.argmax(distances))].copy(), True


def _render_task_goal(env: gym.Env, benchmark: str, goal_state: dict[str, Any]) -> np.ndarray:
    """Render a goal in which the task object/effect reaches the visible target."""
    raw = env.unwrapped
    if benchmark == "reacher":
        physics = raw.env.physics
        saved_qpos = physics.data.qpos.copy()
        saved_qvel = physics.data.qvel.copy()
        try:
            raw.set_state(
                np.asarray(goal_state["qpos"], dtype=np.float64),
                np.zeros_like(saved_qvel),
            )
            _set_reacher_visible_target(
                env, np.asarray(goal_state["target_pos"], dtype=np.float64)
            )
            return _render_live_for_env(env, benchmark)
        finally:
            raw.set_state(saved_qpos, saved_qvel)
            _set_reacher_visible_target(
                env, np.asarray(goal_state["target_pos"], dtype=np.float64)
            )
    if benchmark == "pusht":
        saved_state = np.asarray(raw._get_obs(), dtype=np.float64).copy()
        try:
            raw._set_state(np.asarray(goal_state["full_state"], dtype=np.float64))
            return _render_live_for_env(env, benchmark)
        finally:
            raw._set_state(saved_state)
    raise ValueError(f"task goal rendering is unsupported for {benchmark}")


def _initialise_cube(
    env: gym.Env,
    handle: h5py.File,
    start_row: int,
    goal_row: int,
    manifest_case: dict[str, Any],
) -> dict[str, Any]:
    cube_id = int(manifest_case.get("cube_id", 0))
    if cube_id != 0:
        raise ValueError("swm/OGBCube-v0 env_type='single' requires cube_id=0")
    start_qpos = _row_vector(handle, "qpos", start_row)
    start_qvel = _row_vector(handle, "qvel", start_row)
    goal_qpos = _immutable_array(_row_vector(handle, "qpos", goal_row))
    goal_qvel = _immutable_array(_row_vector(handle, "qvel", goal_row))
    target_pos = _immutable_array(
        _row_vector(handle, "privileged_block_0_pos", goal_row, length=3)
    )
    target_quat = _immutable_array(
        _row_vector(handle, "privileged_block_0_quat", goal_row, length=4)
    )

    # This deliberately mirrors LeWM's dataset evaluation setup: a plain reset
    # followed by explicit state and target injection.
    env.reset()
    raw = env.unwrapped
    raw.set_state(start_qpos, start_qvel)
    raw.set_target_pos(cube_id, target_pos.copy(), target_quat.copy())
    return {
        "cube_id": cube_id,
        "qpos": goal_qpos,
        "qvel": goal_qvel,
        "target_pos": target_pos,
        "target_quat": target_quat,
    }


def _render_live_for_env(env: gym.Env, benchmark: str) -> np.ndarray:
    raw = env.unwrapped
    if benchmark == "reacher":
        frame = raw.render(width=NATIVE_IMAGE_SIZE, height=NATIVE_IMAGE_SIZE)
    else:
        frame = raw.render()
    array = np.asarray(frame)
    expected = (NATIVE_IMAGE_SIZE, NATIVE_IMAGE_SIZE, 3)
    if array.shape != expected or array.dtype != np.uint8:
        raise ValueError(
            f"live {benchmark} render must be uint8 {expected}, "
            f"got dtype={array.dtype}, shape={array.shape}"
        )
    return array


def _render_live(session: ReferenceSession) -> np.ndarray:
    return _render_live_for_env(session.env, session.benchmark)


def _filtered_env_info(info: dict[str, Any] | None) -> dict[str, Any]:
    """Keep scalar/vector environment diagnostics, excluding image payloads."""

    filtered: dict[str, Any] = {}
    for key, value in (info or {}).items():
        if key in {"goal", "goal_rendered", "pixels"}:
            continue
        if isinstance(value, np.ndarray) and value.ndim >= 3:
            continue
        filtered[key] = value
    return filtered


def _pusht_target_coverage(env: gym.Env) -> float:
    """Compute exact block overlap with the rendered green target geometry."""
    raw = env.unwrapped
    target_body = raw._get_goal_pose_body(np.asarray(raw.goal_pose, dtype=np.float64))
    current_polygons = []
    target_polygons = []
    for shape in raw.block.shapes:
        if not hasattr(shape, "get_vertices"):
            continue
        vertices = shape.get_vertices()
        current_polygons.append(
            Polygon(
                [
                    tuple(float(x) for x in raw.block.local_to_world(vertex))
                    for vertex in vertices
                ]
            )
        )
        target_polygons.append(
            Polygon(
                [
                    tuple(float(x) for x in target_body.local_to_world(vertex))
                    for vertex in vertices
                ]
            )
        )
    if not current_polygons or not target_polygons:
        raise RuntimeError("Push-T block exposes no polygon geometry")
    current = unary_union(current_polygons)
    target = unary_union(target_polygons)
    return float(
        np.clip(current.intersection(target).area / max(target.area, 1e-12), 0.0, 1.0)
    )


def _pusht_info(session: ReferenceSession) -> dict[str, Any]:
    raw = session.env.unwrapped
    current = np.asarray(raw._get_obs(), dtype=np.float64)
    goal = np.asarray(session.goal_state["full_state"], dtype=np.float64)
    predicate_success, state_distance = raw.eval_state(goal, current)
    position_error = float(np.linalg.norm(goal[:4] - current[:4]))
    angle_error = float(abs(goal[4] - current[4]))
    angle_error = min(angle_error, float(2 * np.pi - angle_error))
    return {
        "current_state": _push_state(current),
        "goal_state": session.goal_state,
        "pos_agent": current[:2],
        "vel_agent": current[5:7],
        "block_pose": current[2:5],
        "goal_pose": goal[2:5],
        "target_pose": np.asarray(raw.goal_pose, dtype=np.float64).copy(),
        "coverage": _pusht_target_coverage(session.env),
        "full_state_success": bool(predicate_success),
        "full_state_distance": float(state_distance),
        "position_error": position_error,
        "angle_error": angle_error,
    }


def _two_room_info(session: ReferenceSession) -> dict[str, Any]:
    raw = session.env.unwrapped
    current = np.asarray(raw.agent_position, dtype=np.float64).reshape(2)
    goal = np.asarray(session.goal_state["state"], dtype=np.float64).reshape(2)
    distance = float(np.linalg.norm(current - goal))
    return {
        "current_state": {"state": current, "pos_agent": current},
        "goal_state": session.goal_state,
        "state": current,
        "pos_agent": current,
        "goal_position": goal,
        "distance_to_target": distance,
        "success_radius": 16.0,
    }


def _reacher_info(session: ReferenceSession) -> dict[str, Any]:
    raw = session.env.unwrapped
    physics = raw.env.physics
    qpos = np.asarray(physics.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(physics.data.qvel, dtype=np.float64).copy()
    goal_qpos = np.asarray(session.goal_state["qpos"], dtype=np.float64)
    qpos_abs_error = np.abs(qpos - goal_qpos)
    target_pos = np.asarray(
        physics.named.data.geom_xpos["target", :2], dtype=np.float64
    ).copy()
    finger_pos = np.asarray(
        physics.named.data.geom_xpos["finger", :2], dtype=np.float64
    ).copy()
    distance_to_target = float(np.linalg.norm(finger_pos - target_pos))
    target_radius = float(physics.named.model.geom_size["target", 0])
    return {
        "current_state": {"qpos": qpos, "qvel": qvel},
        "goal_state": session.goal_state,
        "qpos": qpos,
        "qvel": qvel,
        "goal_qpos": goal_qpos,
        "qpos_abs_error": qpos_abs_error,
        "qpos_max_abs_error": float(np.max(qpos_abs_error)),
        "qpos_match_success": bool(np.max(qpos_abs_error) < 0.05),
        "target_pos": target_pos,
        "finger_pos": finger_pos,
        "distance_to_target": distance_to_target,
        "target_radius": target_radius,
    }


def _cube_info(session: ReferenceSession) -> dict[str, Any]:
    raw = session.env.unwrapped
    cube_id = int(session.goal_state["cube_id"])
    qpos = np.asarray(raw._data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(raw._data.qvel, dtype=np.float64).copy()
    cube_joint = raw._data.joint(f"object_joint_{cube_id}")
    cube_pos = np.asarray(cube_joint.qpos[:3], dtype=np.float64).copy()
    cube_quat = np.asarray(cube_joint.qpos[3:], dtype=np.float64).copy()
    target_pos = np.asarray(session.goal_state["target_pos"], dtype=np.float64)
    cube_position_error = float(np.linalg.norm(cube_pos - target_pos))
    return {
        "current_state": {
            "qpos": qpos,
            "qvel": qvel,
            "cube_id": cube_id,
            "cube_pos": cube_pos,
            "cube_quat": cube_quat,
        },
        "goal_state": session.goal_state,
        "qpos": qpos,
        "qvel": qvel,
        "goal_qpos": session.goal_state["qpos"],
        "goal_qvel": session.goal_state["qvel"],
        "cube_id": cube_id,
        "cube_pos": cube_pos,
        "cube_quat": cube_quat,
        "target_pos": target_pos,
        "target_quat": session.goal_state["target_quat"],
        "cube_position_error": cube_position_error,
        "cube_position_success": bool(cube_position_error <= 0.04),
    }


def _benchmark_info(session: ReferenceSession) -> dict[str, Any]:
    if session.benchmark == "pusht":
        return _pusht_info(session)
    if session.benchmark == "two_room_swm":
        return _two_room_info(session)
    if session.benchmark == "reacher":
        return _reacher_info(session)
    if session.benchmark == "cube":
        return _cube_info(session)
    raise RuntimeError(f"unsupported session benchmark {session.benchmark!r}")


def _backend_metadata(
    session: ReferenceSession,
    *,
    frame_source: str,
) -> dict[str, Any]:
    action_repeat = (
        int(session.env.unwrapped.action_repeat)
        if session.benchmark == "reacher"
        else 1
    )
    return {
        "implementation": "stable_worldmodel",
        "version": SWM_VERSION,
        "env_id": ENV_IDS[session.benchmark],
        "env_options": ENV_OPTIONS[session.benchmark],
        "time_limit_steps": MAX_EPISODE_STEPS,
        "action_repeat": action_repeat,
        "native_image_size": [NATIVE_IMAGE_SIZE, NATIVE_IMAGE_SIZE],
        "frame_source": frame_source,
        "goal_frame_source": (
            "live_render:task_target"
            if session.task_protocol == "task_success_v1"
            else "hdf5:pixels"
        ),
        "task_protocol": session.task_protocol,
        "hdf5_path": session.hdf5_path,
        "dataset_row": session.dataset_row,
        "goal_dataset_row": session.goal_dataset_row,
    }


def _info(
    session: ReferenceSession,
    *,
    primary_success: bool,
    env_terminated: bool,
    truncated: bool,
    frame_source: str,
    env_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = {
        "benchmark": session.benchmark,
        "manifest_case_id": session.manifest_case_id,
        "env_step": session.env_step,
        "primary_success": bool(primary_success),
        "env_terminated": bool(env_terminated),
        "truncated": bool(truncated),
        "goal_state_immutable": True,
        "backend": _backend_metadata(session, frame_source=frame_source),
        **_benchmark_info(session),
    }
    filtered = _filtered_env_info(env_info)
    if filtered:
        info["environment_info"] = filtered
    return _jsonable(info)


def _start_session(request: StartRequest) -> dict[str, Any]:
    case_benchmark = request.manifest_case.get("benchmark")
    if case_benchmark is not None and case_benchmark != request.benchmark:
        raise ValueError(
            "manifest benchmark does not match request: "
            f"{case_benchmark!r} != {request.benchmark!r}"
        )

    hdf5_path, handle = _open_hdf5(request.hdf5_path)
    start_row, goal_row = _resolve_rows(
        handle, request.benchmark, request.manifest_case
    )

    env = _make_env(request.benchmark)
    try:
        if request.task_protocol == "task_success_v1" and request.benchmark == "pusht":
            goal_state = _initialise_pusht_task(env, handle, start_row)
        elif (
            request.task_protocol == "task_success_v1"
            and request.benchmark == "reacher"
        ):
            goal_state = _initialise_reacher_task(
                env, handle, start_row, request.manifest_case
            )
        elif request.benchmark == "pusht":
            goal_state = _initialise_pusht(env, handle, start_row, goal_row)
        elif request.benchmark == "two_room_swm":
            goal_state = _initialise_two_room(env, handle, start_row, goal_row)
        elif request.benchmark == "reacher":
            goal_state = _initialise_reacher(env, handle, start_row, goal_row)
        else:
            goal_state = _initialise_cube(
                env,
                handle,
                start_row,
                goal_row,
                request.manifest_case,
            )

        if request.task_protocol == "task_success_v1":
            if request.benchmark not in {"pusht", "reacher"}:
                raise ValueError(
                    "task_success_v1 currently supports only pusht and reacher"
                )
            goal_frame = _render_task_goal(env, request.benchmark, goal_state)
            start_frame = _render_live_for_env(env, request.benchmark)
            start_image = _encode_png(
                start_frame, source=f"live {request.benchmark} task start"
            )
            goal_image = _encode_png(
                goal_frame, source=f"live {request.benchmark} task goal"
            )
            frame_source = "live_render:task_start"
        else:
            start_image = _encode_png(
                _hdf_pixels(handle, start_row), source=f"HDF5 pixels[{start_row}]"
            )
            goal_image = _encode_png(
                _hdf_pixels(handle, goal_row), source=f"HDF5 pixels[{goal_row}]"
            )
            frame_source = "hdf5:pixels"

        session_id = str(uuid.uuid4())
        session = ReferenceSession(
            session_id=session_id,
            benchmark=request.benchmark,
            env=env,
            hdf5_path=hdf5_path,
            dataset_row=start_row,
            goal_dataset_row=goal_row,
            manifest_case_id=request.manifest_case.get("case_id"),
            goal_state=goal_state,
            goal_image=goal_image,
            task_protocol=request.task_protocol,
        )
        _SESSIONS[session_id] = session
    except Exception:
        env.close()
        raise

    return {
        "session_id": session_id,
        "image": start_image,
        "goal_image": goal_image,
        "info": _info(
            session,
            primary_success=False,
            env_terminated=False,
            truncated=False,
            frame_source=frame_source,
        ),
        "reward": 0.0,
        "done": False,
        "terminated": False,
        "truncated": False,
    }


def _native_action(session: ReferenceSession, action: list[float]) -> np.ndarray:
    array = np.asarray(action, dtype=np.float64)
    action_space = session.env.action_space
    if array.shape != action_space.shape:
        raise ValueError(
            f"{session.benchmark} action must have shape "
            f"{action_space.shape}, got {array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("action contains non-finite values")
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    if np.any(array < low) or np.any(array > high):
        raise ValueError(
            f"native action is outside SWM bounds: action={array.tolist()}, "
            f"low={low.tolist()}, high={high.tolist()}"
        )
    return array.astype(action_space.dtype, copy=False)


def _step_session(request: StepRequest) -> dict[str, Any]:
    session = _SESSIONS.get(request.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    if session.finished:
        raise HTTPException(status_code=409, detail="session is already finished")

    action = _native_action(session, request.action)
    _, reward, terminated, truncated, env_info = session.env.step(action)
    session.env_step += 1

    benchmark_info = _benchmark_info(session)
    if session.benchmark == "pusht":
        predicate_success = bool(benchmark_info["full_state_success"])
        original_primary_success = bool(terminated or predicate_success)
        task_success = bool(float(benchmark_info["coverage"]) >= 0.95)
    elif session.benchmark == "two_room_swm":
        distance_success = bool(
            float(benchmark_info["distance_to_target"])
            < float(benchmark_info["success_radius"])
        )
        original_primary_success = bool(terminated or distance_success)
        task_success = original_primary_success
    elif session.benchmark == "reacher":
        qpos_match = bool(benchmark_info["qpos_max_abs_error"] < 0.05)
        original_primary_success = bool(terminated or qpos_match)
        task_success = bool(
            float(benchmark_info["distance_to_target"])
            <= float(benchmark_info["target_radius"])
        )
    else:
        original_primary_success = bool(
            benchmark_info["cube_position_error"] <= 0.04
        )
        task_success = original_primary_success

    task_protocol = session.task_protocol == "task_success_v1"
    primary_success = task_success if task_protocol else original_primary_success
    session.finished = bool(truncated) if task_protocol else bool(
        primary_success or truncated
    )
    live_image = _encode_png(
        _render_live(session), source=f"live {session.benchmark} render"
    )
    info = _info(
        session,
        primary_success=primary_success,
        env_terminated=bool(terminated),
        truncated=bool(truncated),
        frame_source="live_render",
        env_info=env_info,
    )
    info["original_primary_success"] = bool(original_primary_success)
    info["task_primary_success"] = bool(task_success)
    return {
        "session_id": session.session_id,
        "image": live_image,
        "info": _jsonable(info),
        "reward": float(reward),
        "done": False if task_protocol else primary_success,
        "terminated": False if task_protocol else bool(terminated),
        "truncated": bool(truncated),
    }


def _close_session(session_id: str) -> dict[str, Any]:
    session = _SESSIONS.pop(session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="unknown session_id")
    session.env.close()
    return {"session_id": session_id, "closed": True}


def _shutdown() -> None:
    with _LOCK:
        sessions = list(_SESSIONS.values())
        _SESSIONS.clear()
        for session in sessions:
            try:
                session.env.close()
            except Exception:
                pass

        handles = list(_HDF5_FILES.values())
        _HDF5_FILES.clear()
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass


@asynccontextmanager
async def _lifespan(_: FastAPI):
    yield
    _shutdown()


app = FastAPI(title="SWM Reference Server", version="1", lifespan=_lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    with _LOCK:
        return {
            "status": "ok",
            "backend": "stable_worldmodel",
            "backend_version": SWM_VERSION,
            "sessions": len(_SESSIONS),
            "cached_hdf5_files": len(_HDF5_FILES),
        }


@app.post("/start")
def start(request: StartRequest) -> dict[str, Any]:
    with _LOCK:
        try:
            return _start_session(request)
        except HTTPException:
            raise
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/step")
def step(request: StepRequest) -> dict[str, Any]:
    with _LOCK:
        try:
            return _step_session(request)
        except HTTPException:
            raise
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/close")
def close(request: CloseRequest) -> dict[str, Any]:
    with _LOCK:
        return _close_session(request.session_id)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        workers=1,
    )


if __name__ == "__main__":
    main()
