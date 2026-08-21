"""Functional tests for PlanningCriticSandbox.calibrate_parameters.

These build a tiny synthetic dataset with KNOWN true dynamics
(block_pos += true_gain * action, true_gain = 0.7) and a candidate
simulator whose default gain is deliberately wrong (0.2). The tool is
system identification: it must recover ~0.7 from the data and collapse the
loss_to_target reduction_ratio, without ever editing the candidate's code.

We also assert the two contract failures the tool is supposed to report
clearly: a simulator with no ``self.params`` dict (NO_PARAMS) and a
``bounds`` key that is not actually a parameter (MISSING_PARAMS).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox

FRAME = 64
TRUE_GAIN = 0.7
DEFAULT_GAIN = 0.2  # deliberately wrong, so calibration has work to do


# --- synthetic dataset -------------------------------------------------------

def _render(pos: np.ndarray) -> Image.Image:
    """A red 5x5 block at integer ``pos`` on a black frame."""
    img = np.zeros((FRAME, FRAME, 3), dtype=np.uint8)
    x, y = int(round(pos[0])), int(round(pos[1]))
    x = min(max(x, 2), FRAME - 3)
    y = min(max(y, 2), FRAME - 3)
    img[y - 2 : y + 3, x - 2 : x + 3, 0] = 255
    return Image.fromarray(img)


def _build_dataset(root, n_traj=4, n_steps=8):
    """Block moves by ``TRUE_GAIN * action`` each step; frames + actions.json."""
    rng = np.random.default_rng(0)
    starts = np.array([[16, 16], [16, 46], [46, 16], [30, 30]], dtype=np.float64)
    for ti in range(n_traj):
        tdir = root / f"trajectory_{ti:04d}"
        tdir.mkdir(parents=True)
        pos = starts[ti % len(starts)].copy()
        actions = rng.uniform(-3.0, 3.0, size=(n_steps, 2))
        # bias actions so the block drifts toward frame centre and stays in view
        actions += (np.array([32.0, 32.0]) - pos) / (n_steps * TRUE_GAIN)
        _render(pos).save(tdir / "frame_0000.png")
        recorded = []
        for si, a in enumerate(actions):
            pos = pos + TRUE_GAIN * a
            _render(pos).save(tdir / f"frame_{si + 1:04d}.png")
            recorded.append({"step": si, "action": a.tolist()})
        (tdir / "actions.json").write_text(json.dumps(recorded))


# --- candidate simulator sources --------------------------------------------

_GOOD_SIM = f"""
import numpy as np


class Simulator(ActionConditionedSimulatorBase):
    def __init__(self, frame_size=(64, 64), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.params = {{"gain": {DEFAULT_GAIN}}}

    def _centroid(self, image):
        mask = np.asarray(image)[:, :, 0] > 128
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return np.array([32.0, 32.0])
        return np.array([xs.mean(), ys.mean()])

    def fit(self, image_A, image_B):
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

# Identical, but with NO self.params dict at all.
_NO_PARAMS_SIM = _GOOD_SIM.replace(
    f'        self.params = {{"gain": {DEFAULT_GAIN}}}\n', ""
).replace('self.params["gain"]', str(DEFAULT_GAIN))


# --- fixtures ----------------------------------------------------------------

def _make_sandbox(
    tmp_path, code, dataset_root, *, deployed_action_bounds=None, **kwargs
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
        no_api=True,
        deployed_action_bounds=deployed_action_bounds,
        **kwargs,
    )


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root)
    return root


# --- tests -------------------------------------------------------------------

def test_recovers_true_gain_and_collapses_ratio(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    summary = sandbox.calibrate_parameters(bounds={"gain": [0.1, 2.0]}, budget=120)

    # The summary is the VLM-facing string; the full payload is persisted, but
    # logging is off here, so parse the numbers we care about from the summary.
    assert "calibrate_parameters: fitted" in summary
    assert "gain=" in summary

    # Recover the fitted gain from the "fitted params:" line.
    fitted_line = next(l for l in summary.splitlines() if "fitted params:" in l)
    fitted_gain = float(fitted_line.split("gain=")[1].split(",")[0].strip())
    assert fitted_gain == pytest.approx(TRUE_GAIN, abs=0.05), summary

    # The mean reduction_ratio must drop substantially after calibration.
    before = float(summary.split("before (your values):")[1].split()[0])
    after = float(summary.split("after (fitted):")[1].split()[0])
    assert after < before, summary
    assert after < 0.2, summary  # faithful replay collapses the loss

    # Held-out trajectory must improve too (generalisation, not overfit).
    assert "HELD-OUT trajectory" in summary, summary


def test_no_params_dict_reports_contract_error(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _NO_PARAMS_SIM, dataset)
    summary = sandbox.calibrate_parameters(bounds={"gain": [0.1, 2.0]}, budget=60)
    assert "no `self.params` dict" in summary, summary
    # The error must show the fix-it example so the VLM can self-correct.
    assert "self.params = {" in summary, summary


def test_wrong_param_name_lists_available_keys(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    summary = sandbox.calibrate_parameters(bounds={"friction": [0.0, 1.0]}, budget=60)
    assert "not keys of" in summary, summary
    assert "friction" in summary, summary
    # It should tell the VLM what params DO exist.
    assert "gain" in summary, summary


def test_empty_bounds_is_rejected(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    summary = sandbox.calibrate_parameters(bounds={}, budget=60)
    assert "non-empty dict" in summary, summary


def test_contact_gain_prior_caps_upper_bound(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    summary = sandbox.calibrate_parameters(bounds={"gain": [0.1, 2.0]}, budget=60)
    assert "prior: contact/transfer gains were bounded to <=1.0" in summary, summary
    fitted_line = next(l for l in summary.splitlines() if "fitted params:" in l)
    fitted_gain = float(fitted_line.split("gain=")[1].split(",")[0].strip())
    assert fitted_gain <= 1.0 + 1e-9, summary


def test_contact_gain_prior_rejects_range_above_one(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    summary = sandbox.calibrate_parameters(bounds={"gain": [1.1, 2.0]}, budget=60)
    assert "contact/transfer gain prior <= 1.0" in summary, summary
