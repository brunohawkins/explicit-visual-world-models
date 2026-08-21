"""Smoke-level test for eval_test_image using a kinematic fixture sim."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vdaworld.core.simulator import ActionConditionedSimulatorBase
from vdaworld.eval.closed_loop import eval_test_image

_TEST_IMG_DIR = (
    Path(__file__).resolve().parents[1] / "scenes" / "two_room" / "test_time"
)


class _KinematicSim(ActionConditionedSimulatorBase):
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


@pytest.mark.skipif(
    not (_TEST_IMG_DIR / "start.png").exists()
    or not (_TEST_IMG_DIR / "goal.png").exists(),
    reason="hand-made test PNGs not generated yet (run scripts/generate_test_time_image.py)",
)
def test_eval_test_image_smoke():
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
    result = eval_test_image(
        _KinematicSim,
        start_png_path=_TEST_IMG_DIR / "start.png",
        goal_png_path=_TEST_IMG_DIR / "goal.png",
        cem_kwargs=cem_kwargs,
    )

    for k in (
        "predicted_frames",
        "predicted_actions",
        "image_A",
        "image_B",
        "final_loss",
        "cem_history",
    ):
        assert k in result, f"missing key {k!r}"

    assert result["predicted_actions"].shape == (10, 2)
    assert len(result["predicted_frames"]) == 11
    assert result["image_A"].shape == (65, 65, 3)
    assert result["image_B"].shape == (65, 65, 3)
    assert isinstance(result["final_loss"], float)
    assert len(result["cem_history"]) == 3
