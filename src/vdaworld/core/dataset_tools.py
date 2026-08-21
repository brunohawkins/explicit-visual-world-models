"""Dataset access tools the VLM uses during code generation.

Three helpers (``read_trajectory``, ``view_image``, ``view_action``) plus a
``_normalise_actions`` helper that accepts both layouts found across the
project's datasets:

* two-rooms ``actions.json``: ``[{"step": i, "action": [...]}, ...]``
* ogbench_cube ``actions.json``: bare ``[[...], [...], ...]``

At P0 these are importable standalone. P1 will register them as agentic-loop
tools through ``CriticSandbox``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


def _normalise_actions(raw) -> np.ndarray:
    """Return a uniform ``(T, action_dim)`` float64 ndarray from either layout.

    Empty input returns an empty 2-D array.
    """
    if not raw:
        return np.empty((0, 0), dtype=np.float64)
    first = raw[0]
    if isinstance(first, dict):
        ordered = sorted(raw, key=lambda r: r["step"])
        actions = [r["action"] for r in ordered]
    elif isinstance(first, (list, tuple)):
        actions = list(raw)
    else:
        raise ValueError(
            f"Unsupported actions.json element type: {type(first).__name__}. "
            "Expected dict with 'step'/'action' keys or list of floats."
        )
    return np.asarray(actions, dtype=np.float64)


def read_trajectory(dataset_dir: Union[str, Path], idx: int) -> dict:
    """Read trajectory ``idx`` from ``dataset_dir``.

    Returns a dict with keys:

    * ``frames``  — sorted list of PNG ``Path`` objects.
    * ``actions`` — ``(T, action_dim)`` float64 ndarray.
    * ``n_steps`` — int, number of actions.
    * ``action_contract`` — optional declared action semantics/bounds copied
      from trajectory metadata (never privileged simulator state).
    * ``state_contract`` — optional declared state topology, timing, and task
      semantics copied from metadata (never per-frame privileged state).
    """
    dataset_dir = Path(dataset_dir)
    traj_dir = dataset_dir / f"trajectory_{idx:04d}"
    if not traj_dir.is_dir():
        raise FileNotFoundError(f"No trajectory directory at {traj_dir}")

    frame_paths = sorted(traj_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {traj_dir}")

    actions_path = traj_dir / "actions.json"
    if not actions_path.exists():
        raise FileNotFoundError(f"No actions.json in {traj_dir}")
    with open(actions_path) as f:
        raw = json.load(f)
    actions = _normalise_actions(raw)
    action_contract = None
    state_contract = None
    metadata_path = traj_dir / "metadata.json"
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            declared = metadata.get("action_contract")
            if isinstance(declared, dict):
                action_contract = {
                    key: declared[key]
                    for key in ("semantics", "action_repeat", "low", "high")
                    if key in declared
                }
            declared_state = metadata.get("state_contract")
            if isinstance(declared_state, dict):
                state_contract = {
                    key: declared_state[key]
                    for key in (
                        "planner_step_seconds",
                        "components",
                        "task_target",
                    )
                    if key in declared_state
                }
            controller = metadata.get("controller")
            if (
                state_contract is None
                and isinstance(controller, dict)
                and controller.get("joint_error")
                == "wrapped_shoulder_and_raw_bounded_elbow"
            ):
                action_repeat = int((action_contract or {}).get("action_repeat", 1))
                wrist_limit = float(np.deg2rad(160.0))
                state_contract = {
                    "planner_step_seconds": 0.02 * action_repeat,
                    "components": {
                        "shoulder": {
                            "kind": "angle",
                            "periodic": True,
                            "state_keys": ["theta1", "shoulder"],
                        },
                        "wrist": {
                            "kind": "angle",
                            "periodic": False,
                            "state_keys": ["theta2", "wrist", "elbow"],
                            "low": -wrist_limit,
                            "high": wrist_limit,
                        },
                    },
                    "task_target": {
                        "kind": str(
                            metadata.get("task", "end_effector_to_visible_target")
                        ),
                        "branch_invariant": True,
                    },
                }
        except (OSError, ValueError, TypeError):
            action_contract = None
            state_contract = None

    return {
        "frames": frame_paths,
        "actions": actions,
        "n_steps": int(actions.shape[0]),
        "action_contract": action_contract,
        "state_contract": state_contract,
    }


def view_image(dataset_dir: Union[str, Path], t_idx: int, image_idx: int) -> np.ndarray:
    """Return one frame as a uint8 (H, W, 3) RGB array."""
    dataset_dir = Path(dataset_dir)
    frame_path = dataset_dir / f"trajectory_{t_idx:04d}" / f"frame_{image_idx:04d}.png"
    if not frame_path.exists():
        raise FileNotFoundError(f"No frame at {frame_path}")
    img = Image.open(frame_path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def view_action(dataset_dir: Union[str, Path], t_idx: int, step_idx: int) -> np.ndarray:
    """Return the action vector at step ``step_idx`` of trajectory ``t_idx``."""
    traj = read_trajectory(dataset_dir, t_idx)
    if step_idx < 0 or step_idx >= traj["n_steps"]:
        raise IndexError(
            f"step_idx {step_idx} out of range [0, {traj['n_steps']})"
        )
    return traj["actions"][step_idx]
