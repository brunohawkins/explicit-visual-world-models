"""Regression tests: subprocess validators must thread WorldAPI like run_simulation."""

from __future__ import annotations

import re

import numpy as np
import pytest

from tests.test_calibrate_parameters import (
    FRAME,
    _GOOD_SIM,
    _build_dataset,
)
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox

# Same dynamics as _GOOD_SIM but fit() requires a non-None self.api.
_API_REQUIRED_SIM = f"""
import numpy as np


class Simulator(ActionConditionedSimulatorBase):
    def __init__(self, frame_size=(64, 64), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.params = {{"gain": 0.2}}

    def _centroid(self, image):
        mask = np.asarray(image)[:, :, 0] > 128
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return np.array([32.0, 32.0])
        return np.array([xs.mean(), ys.mean()])

    def fit(self, image_A, image_B):
        if self.api is None:
            raise AttributeError("'NoneType' object has no attribute 'segment'")
        self.state = {{"p": self._centroid(image_A)}}
        self.target_state = {{"p": self._centroid(image_B)}}

    def update(self, a):
        self.state["p"] = self.state["p"] + self.params["gain"] * np.asarray(a, dtype=float)

    def loss_to_target(self):
        return float(np.linalg.norm(self.state["p"] - self.target_state["p"]))

    def render_frame(self):
        img = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        x, y = int(round(self.state["p"][0])), int(round(self.state["p"][1]))
        x = min(max(x, 2), self.frame_size[0] - 3)
        y = min(max(y, 2), self.frame_size[1] - 3)
        img[y - 2 : y + 3, x - 2 : x + 3, 0] = 255
        return img
"""

# validate_with_cem_states bypasses fit(); require api during CEM (loss_to_target).
_API_REQUIRED_LOSS_SIM = f"""
import numpy as np


class Simulator(ActionConditionedSimulatorBase):
    def __init__(self, frame_size=(64, 64), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.params = {{"gain": 0.7}}

    def fit(self, image_A, image_B):
        self.state = {{"p": np.array([16.0, 16.0])}}
        self.target_state = {{"p": np.array([40.0, 40.0])}}

    def update(self, a):
        self.state["p"] = self.state["p"] + self.params["gain"] * np.asarray(a, dtype=float)

    def loss_to_target(self):
        if self.api is None:
            raise AttributeError("'NoneType' object has no attribute 'segment'")
        return float(np.linalg.norm(self.state["p"] - self.target_state["p"]))

    def render_frame(self):
        img = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        x, y = int(round(self.state["p"][0])), int(round(self.state["p"][1]))
        x = min(max(x, 2), self.frame_size[0] - 3)
        y = min(max(y, 2), self.frame_size[1] - 3)
        img[y - 2 : y + 3, x - 2 : x + 3, 0] = 255
        return img
"""


def _make_sandbox(
    tmp_path,
    code,
    dataset_root,
    *,
    no_api: bool,
    deployed_action_bounds=None,
):
    start = dataset_root / "trajectory_0000" / "frame_0000.png"
    goal = dataset_root / "trajectory_0000" / "frame_0008.png"
    return PlanningCriticSandbox(
        code=code,
        fps=10,
        n_frames=8,
        frame_size=(FRAME, FRAME),
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset_root),
        simulator_class_name="Simulator",
        sandbox_dir=str(tmp_path / "sandbox"),
        tool_calls_log_dir=None,
        no_api=no_api,
        deployed_action_bounds=deployed_action_bounds,
    )


def _extract_reduction_ratio(summary: str) -> float | None:
    m = re.search(r"reduction_ratio=([\d.]+|n/a)", summary)
    if not m or m.group(1) == "n/a":
        return None
    return float(m.group(1))


def _assert_api_ok(summary: str) -> None:
    assert "runner exited" not in summary
    assert "AttributeError" not in summary
    assert "FIT_FAILED" not in summary


def _assert_api_blocked(summary: str) -> None:
    assert (
        "runner exited" in summary
        or "AttributeError" in summary
        or "FIT_FAILED" in summary
    )


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root)
    return root


# --- validate_against_training ------------------------------------------------

def test_numpy_sim_unchanged_with_no_api(tmp_path, dataset):
    """Pure-numpy sim (two-rooms style): no_api=True path unchanged."""
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset, no_api=True)
    summary = sandbox.validate_against_training(trajectory_index=0)
    assert "runner exited" not in summary
    ratio = _extract_reduction_ratio(summary)
    assert ratio is not None
    assert ratio < 1.0


def test_validate_api_required_gets_world_api_when_enabled(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_SIM, dataset, no_api=False)
    summary = sandbox.validate_against_training(trajectory_index=0)
    _assert_api_ok(summary)
    ratio = _extract_reduction_ratio(summary)
    assert ratio is not None
    assert ratio < 1.0


def test_validate_api_required_still_gets_none_when_no_api(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_SIM, dataset, no_api=True)
    summary = sandbox.validate_against_training(trajectory_index=0)
    _assert_api_blocked(summary)


# --- calibrate_parameters -----------------------------------------------------

def test_calibrate_numpy_sim_unchanged_with_no_api(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset, no_api=True)
    summary = sandbox.calibrate_parameters(bounds={"gain": [0.1, 2.0]}, budget=30)
    assert "calibrate_parameters: fitted" in summary


def test_calibrate_api_required_gets_world_api_when_enabled(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_SIM, dataset, no_api=False)
    summary = sandbox.calibrate_parameters(
        bounds={"gain": [0.1, 2.0]}, budget=30, trajectory_indices=[0]
    )
    _assert_api_ok(summary)
    assert "calibrate_parameters: fitted" in summary


def test_calibrate_api_required_still_gets_none_when_no_api(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_SIM, dataset, no_api=True)
    summary = sandbox.calibrate_parameters(bounds={"gain": [0.1, 2.0]}, budget=30)
    _assert_api_blocked(summary)


# --- validate_with_cem ----------------------------------------------------------

def test_cem_numpy_sim_unchanged_with_no_api(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset, no_api=True)
    summary = sandbox.validate_with_cem(
        0, 0, 1, 0, horizon=5, cem_iters=2, cem_population=20
    )
    _assert_api_ok(summary)
    assert "initial_loss=" in summary


def test_cem_api_required_gets_world_api_when_enabled(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_SIM, dataset, no_api=False)
    summary = sandbox.validate_with_cem(
        0, 0, 1, 0, horizon=5, cem_iters=2, cem_population=20
    )
    _assert_api_ok(summary)
    assert "initial_loss=" in summary


def test_cem_api_required_still_gets_none_when_no_api(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_SIM, dataset, no_api=True)
    summary = sandbox.validate_with_cem(
        0, 0, 1, 0, horizon=5, cem_iters=2, cem_population=20
    )
    _assert_api_blocked(summary)


def test_cem_defaults_to_deployed_action_bounds_when_configured(tmp_path, dataset):
    sandbox = _make_sandbox(
        tmp_path,
        _GOOD_SIM,
        dataset,
        no_api=True,
        deployed_action_bounds=(
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
        ),
    )
    summary = sandbox.validate_with_cem(
        0, 0, 1, 0, horizon=5, cem_iters=2, cem_population=20
    )
    assert "CEM action_bounds (deployed)" in summary
    assert "-1.000" in summary and "1.000" in summary


# --- validate_with_cem_states -------------------------------------------------

def test_cem_states_numpy_sim_unchanged_with_no_api(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset, no_api=True)
    summary = sandbox.validate_with_cem_states(
        start_state={"p": [16.0, 16.0]},
        goal_state={"p": [40.0, 40.0]},
        horizon=5,
        cem_iters=2,
        cem_population=20,
    )
    _assert_api_ok(summary)
    assert "initial_loss=" in summary


def test_cem_states_api_required_gets_world_api_when_enabled(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_LOSS_SIM, dataset, no_api=False)
    summary = sandbox.validate_with_cem_states(
        start_state={"p": [16.0, 16.0]},
        goal_state={"p": [40.0, 40.0]},
        horizon=5,
        cem_iters=2,
        cem_population=20,
    )
    _assert_api_ok(summary)
    assert "initial_loss=" in summary


def test_cem_states_api_required_still_gets_none_when_no_api(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_REQUIRED_LOSS_SIM, dataset, no_api=True)
    summary = sandbox.validate_with_cem_states(
        start_state={"p": [16.0, 16.0]},
        goal_state={"p": [40.0, 40.0]},
        horizon=5,
        cem_iters=2,
        cem_population=20,
    )
    _assert_api_blocked(summary)


def test_cem_states_defaults_to_deployed_action_bounds_when_configured(tmp_path, dataset):
    sandbox = _make_sandbox(
        tmp_path,
        _GOOD_SIM,
        dataset,
        no_api=True,
        deployed_action_bounds=(
            np.array([-1.0, -1.0]),
            np.array([1.0, 1.0]),
        ),
    )
    summary = sandbox.validate_with_cem_states(
        start_state={"p": [16.0, 16.0]},
        goal_state={"p": [40.0, 40.0]},
        horizon=5,
        cem_iters=2,
        cem_population=20,
    )
    assert "CEM action_bounds (deployed)" in summary
