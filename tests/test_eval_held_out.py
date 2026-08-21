"""Smoke-level test for eval_held_out using a kinematic fixture sim."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vdaworld.core.simulator import ActionConditionedSimulatorBase
from vdaworld.eval.closed_loop import eval_held_out

_DATASET_DIR = Path(__file__).resolve().parents[2] / "Datasets" / "two-rooms"


class _KinematicSim(ActionConditionedSimulatorBase):
    """Same fixture as the smoke script — red-dot centroid extraction + clip-to-frame."""

    @staticmethod
    def _red_centroid(img):
        rgb = img.astype(np.int16)
        reddish = (rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])) > 30
        ys, xs = np.where(reddish)
        if len(xs) == 0:
            raise ValueError("No red pixels in image.")
        return np.array([xs.mean(), ys.mean()], dtype=np.float64)

    def fit(self, image_A, image_B):
        self.state = {"agent_xy": self._red_centroid(image_A)}
        self.target_state = {"agent_xy": self._red_centroid(image_B)}

    def update(self, a):
        new_xy = self.state["agent_xy"] + np.asarray(a, dtype=np.float64)
        h, w = self.frame_size[1], self.frame_size[0]
        new_xy[0] = np.clip(new_xy[0], 0, w - 1)
        new_xy[1] = np.clip(new_xy[1], 0, h - 1)
        self.state["agent_xy"] = new_xy

    def loss_to_target(self):
        return float(
            np.linalg.norm(self.state["agent_xy"] - self.target_state["agent_xy"])
        )

    def render_frame(self):
        w, h = self.frame_size
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        x, y = int(round(self.state["agent_xy"][0])), int(round(self.state["agent_xy"][1]))
        x = max(1, min(w - 2, x))
        y = max(1, min(h - 2, y))
        canvas[y - 1:y + 2, x - 1:x + 2] = (255, 0, 0)
        return canvas


@pytest.mark.skipif(not _DATASET_DIR.exists(), reason="two-rooms dataset not available")
def test_eval_held_out_smoke():
    cem_kwargs = dict(
        action_dim=2,
        action_low=np.array([-1.5, -1.5]),
        action_high=np.array([1.5, 1.5]),
        horizon=10,
        population=50,
        elite_frac=0.1,
        iters=3,
        seed=0,
    )
    result = eval_held_out(_KinematicSim, _DATASET_DIR, traj_idx=9, cem_kwargs=cem_kwargs)

    # Required keys present.
    for k in (
        "predicted_frames",
        "true_frames",
        "predicted_actions",
        "true_actions",
        "image_distance",
        "action_distance",
        "cem_history",
    ):
        assert k in result, f"missing key {k!r}"

    # Shape sanity.
    assert result["predicted_actions"].shape == (10, 2)
    assert result["true_actions"].shape == (10, 2)
    assert len(result["predicted_frames"]) == 11    # horizon + 1
    assert len(result["true_frames"]) == 11
    assert isinstance(result["image_distance"], float)
    assert isinstance(result["action_distance"], float)
    assert len(result["cem_history"]) == 3
