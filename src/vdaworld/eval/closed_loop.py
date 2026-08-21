"""Closed-loop evaluation: fit → plan with CEM → replay → compare.

Two evaluation surfaces:

* :func:`eval_held_out` — hold out one training trajectory. Fit on
  ``(frame_0, frame_T)``, plan with CEM, replay, compare to the held-out
  final frame in image space and to the ground-truth action sequence.
* :func:`eval_test_image` — run on the hand-made (start, goal) PNG pair.
  No ground truth; metric is the simulator's own terminal planner cost after
  rollout.

:func:`render_side_by_side` writes an MP4 of (predicted | reference) for
human inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from vdaworld.core.cem import CEM
from vdaworld.core.dataset_tools import read_trajectory, view_image
from vdaworld.core.simulator import ActionConditionedSimulatorBase


def _load_png(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _rollout(
    sim: ActionConditionedSimulatorBase, actions: np.ndarray
) -> list[np.ndarray]:
    """Render the current state, then apply each action and render after.

    ``render_frame()`` may return a live view into a reused backing buffer
    (e.g. a pygame surface via ``surfarray.pixels3d``); without copying, every
    frame in this list would alias the same memory and collapse to the final
    rendered state. Copy each frame to snapshot it.
    """
    frames = [np.asarray(sim.render_frame()).copy()]
    for a in actions:
        sim.update(a)
        frames.append(np.asarray(sim.render_frame()).copy())
    return frames


def _image_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def eval_held_out(
    sim_class,
    dataset_dir: Union[str, Path],
    traj_idx: int,
    cem_kwargs: dict,
    frame_size: tuple[int, int] = (65, 65),
    api=None,
) -> dict:
    """Held-out trajectory eval.

    Returns dict with: predicted_frames, true_frames, predicted_actions,
    true_actions, image_distance, action_distance, cem_history.
    """
    dataset_dir = Path(dataset_dir)
    traj = read_trajectory(dataset_dir, traj_idx)
    n_frames = len(traj["frames"])

    image_A = view_image(dataset_dir, traj_idx, 0)
    image_B = view_image(dataset_dir, traj_idx, n_frames - 1)

    sim = sim_class(frame_size=frame_size, api=api)
    sim.fit(image_A, image_B)

    cem = CEM(sim, **cem_kwargs)
    predicted_actions = cem.plan()
    predicted_frames = _rollout(sim, predicted_actions)

    true_actions = traj["actions"]
    true_frames = [view_image(dataset_dir, traj_idx, i) for i in range(n_frames)]

    image_distance = _image_distance(predicted_frames[-1], true_frames[-1])
    n_compare = min(predicted_actions.shape[0], true_actions.shape[0])
    if n_compare > 0:
        action_distance = float(
            np.linalg.norm(predicted_actions[:n_compare] - true_actions[:n_compare])
        )
    else:
        action_distance = float("nan")

    return {
        "predicted_frames": predicted_frames,
        "true_frames": true_frames,
        "predicted_actions": predicted_actions,
        "true_actions": true_actions,
        "image_distance": image_distance,
        "action_distance": action_distance,
        "cem_history": cem.history,
    }


def eval_test_image(
    sim_class,
    start_png_path: Union[str, Path],
    goal_png_path: Union[str, Path],
    cem_kwargs: dict,
    frame_size: tuple[int, int] = (65, 65),
    api=None,
) -> dict:
    """Hand-made (start, goal) PNG eval. No ground-truth final frame."""
    image_A = _load_png(Path(start_png_path))
    image_B = _load_png(Path(goal_png_path))

    sim = sim_class(frame_size=frame_size, api=api)
    sim.fit(image_A, image_B)

    cem = CEM(sim, **cem_kwargs)
    predicted_actions = cem.plan()
    predicted_frames = _rollout(sim, predicted_actions)
    final_loss = float(sim.planning_objective())

    return {
        "predicted_frames": predicted_frames,
        "predicted_actions": predicted_actions,
        "image_A": image_A,
        "image_B": image_B,
        "final_loss": final_loss,
        "cem_history": cem.history,
    }


def render_side_by_side(
    predicted: list[np.ndarray],
    reference: Optional[list[np.ndarray]],
    out_path: Union[str, Path],
    fps: int = 5,
) -> None:
    """Write an MP4 of (predicted | reference). If reference is None, predicted only."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if reference is None:
        composite = predicted
    else:
        n = max(len(predicted), len(reference))
        pred = predicted + [predicted[-1]] * (n - len(predicted))
        refr = reference + [reference[-1]] * (n - len(reference))
        composite = [np.concatenate([p, r], axis=1) for p, r in zip(pred, refr)]

    imageio.mimwrite(out_path, composite, fps=fps, codec="libx264")
