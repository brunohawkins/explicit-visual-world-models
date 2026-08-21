"""Regression tests: eval helpers must thread WorldAPI through sim construction."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.test_calibrate_parameters import FRAME, _GOOD_SIM, _build_dataset, _render
from vdaworld.core.simulator import ActionConditionedSimulatorBase
from vdaworld.eval.closed_loop import eval_held_out, eval_test_image
from vdaworld.eval.mpc import eval_mpc

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
import score_simulator_fidelity as ssf  # noqa: E402


class _RecordingSim(ActionConditionedSimulatorBase):
    """Records the api kwarg passed to __init__ on the class for inspection."""

    last_api = object()

    def __init__(self, frame_size=(65, 65), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        type(self).last_api = api

    @staticmethod
    def _centroid(img):
        rgb = img.astype(np.int16)
        reddish = (rgb[:, :, 0] - np.maximum(rgb[:, :, 1], rgb[:, :, 2])) > 30
        ys, xs = np.where(reddish)
        if len(xs) == 0:
            return np.array([32.0, 32.0], dtype=np.float64)
        return np.array([xs.mean(), ys.mean()], dtype=np.float64)

    def fit(self, image_A, image_B):
        self.state = {"p": self._centroid(image_A)}
        self.target_state = {"p": self._centroid(image_B)}

    def update(self, a):
        self.state["p"] = self.state["p"] + np.asarray(a, dtype=np.float64)

    def loss_to_target(self):
        return float(np.linalg.norm(self.state["p"] - self.target_state["p"]))

    def render_frame(self):
        w, h = self.frame_size
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        x, y = int(round(self.state["p"][0])), int(round(self.state["p"][1]))
        x = max(1, min(w - 2, x))
        y = max(1, min(h - 2, y))
        canvas[y - 1 : y + 2, x - 1 : x + 2] = (255, 0, 0)
        return canvas


class _MockSession:
    def __init__(self, frame=(65, 65)):
        self._frame = frame

    def start(self, seed):
        img = np.full((*self._frame, 3), 255, dtype=np.uint8)
        img[30, 30] = (255, 0, 0)
        img[50, 50] = (255, 0, 0)
        return {"image": img, "goal_image": img, "done": False, "truncated": False}

    def step(self, action):
        img = np.full((*self._frame, 3), 255, dtype=np.uint8)
        img[30, 30] = (255, 0, 0)
        return {"image": img, "done": True, "truncated": False}

    def close(self):
        pass


def _cem_kwargs():
    return dict(
        action_dim=2,
        action_low=np.array([-1.5, -1.5]),
        action_high=np.array([1.5, 1.5]),
        horizon=2,
        population=10,
        elite_frac=0.2,
        iters=1,
        seed=0,
    )


def test_eval_held_out_threads_api(tmp_path):
    _build_dataset(tmp_path, n_traj=1, n_steps=3)
    sentinel = object()
    eval_held_out(
        _RecordingSim,
        tmp_path,
        traj_idx=0,
        cem_kwargs=_cem_kwargs(),
        frame_size=(FRAME, FRAME),
        api=sentinel,
    )
    assert _RecordingSim.last_api is sentinel

    eval_held_out(
        _RecordingSim,
        tmp_path,
        traj_idx=0,
        cem_kwargs=_cem_kwargs(),
        frame_size=(FRAME, FRAME),
    )
    assert _RecordingSim.last_api is None


def test_eval_test_image_threads_api(tmp_path):
    start = tmp_path / "start.png"
    goal = tmp_path / "goal.png"
    _render(np.array([16.0, 16.0])).save(start)
    _render(np.array([40.0, 40.0])).save(goal)

    sentinel = object()
    eval_test_image(
        _RecordingSim,
        start_png_path=start,
        goal_png_path=goal,
        cem_kwargs=_cem_kwargs(),
        frame_size=(FRAME, FRAME),
        api=sentinel,
    )
    assert _RecordingSim.last_api is sentinel

    eval_test_image(
        _RecordingSim,
        start_png_path=start,
        goal_png_path=goal,
        cem_kwargs=_cem_kwargs(),
        frame_size=(FRAME, FRAME),
    )
    assert _RecordingSim.last_api is None


def test_eval_mpc_threads_api():
    sentinel = object()
    eval_mpc(
        _RecordingSim,
        session=_MockSession(frame=(65, 65)),
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        horizon=1,
        apply_steps=1,
        max_episode_steps=1,
        cem_kwargs={"population": 5, "iters": 1, "seed": 0},
        sim_frame_size=(65, 65),
        api=sentinel,
        seed=0,
    )
    assert _RecordingSim.last_api is sentinel

    eval_mpc(
        _RecordingSim,
        session=_MockSession(frame=(65, 65)),
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        horizon=1,
        apply_steps=1,
        max_episode_steps=1,
        cem_kwargs={"population": 5, "iters": 1, "seed": 0},
        sim_frame_size=(65, 65),
        seed=0,
    )
    assert _RecordingSim.last_api is None


def test_score_one_forwards_no_api(tmp_path, monkeypatch):
    _build_dataset(tmp_path, n_traj=1, n_steps=3)
    sim_path = tmp_path / "simulator_gen.py"
    sim_path.write_text(_GOOD_SIM, encoding="utf-8")

    captured: dict = {}

    class _MockSandbox:
        def __init__(self, **kwargs):
            captured["no_api"] = kwargs.get("no_api")

        def validate_against_training(self, trajectory_index=0):
            return "reduction_ratio=0.5"

    monkeypatch.setattr(ssf, "PlanningCriticSandbox", _MockSandbox)

    ssf.score_one(
        sim_path,
        tmp_path,
        "Simulator",
        (FRAME, FRAME),
        30,
        held_out=0,
        no_api=False,
    )
    assert captured["no_api"] is False

    ssf.score_one(
        sim_path,
        tmp_path,
        "Simulator",
        (FRAME, FRAME),
        30,
        held_out=0,
    )
    assert captured["no_api"] is True
