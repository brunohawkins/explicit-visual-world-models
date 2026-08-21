"""Tests for the two planner-faithfulness feedback signals added 2026-07-07.

1. ``validate_state_consistency`` — perception-anchored per-step fidelity:
   replay ground-truth actions, re-parse the TRUE frame at checkpoints with
   the sim's own ``fit()``, and report the per-component gap. A sim whose
   ``update`` uses the true gain must track its own parses (OK); a sim with a
   wrong gain must be flagged (DRIFT) with an onset step.

2. Goal-reach in ``validate_with_cem`` — cost reduction alone must not pass:
   the components named in ``target_state`` must approach their targets in
   state space, measured in units of per-step motion from training replay.
   The classic Push-T failure (cost falls via an easy shaping term while the
   task component barely moves) must be flagged as FAIL.

The synthetic dataset (true dynamics: p += 0.7 * action) and sandbox factory
are reused from the calibrate tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_calibrate_parameters import (
    FRAME,
    TRUE_GAIN,
    _GOOD_SIM,
    _build_dataset,
    _make_sandbox,
)
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox

_TRUE_GAIN_SIM = _GOOD_SIM.replace(
    'self.params = {"gain": 0.2}',
    f'self.params = {{"gain": {TRUE_GAIN}}}',
)

# Push-T-pattern sim: cost is dominated by an easy, fully drivable component
# (q, the "pusher"), while the component named in target_state (p, the
# "block") barely responds to actions. CEM collapses the cost without moving
# p toward the target — exactly the failure that used to pass CEM checks.
_COST_STATE_MISMATCH_SIM = f"""
import numpy as np


class Simulator(ActionConditionedSimulatorBase):
    def __init__(self, frame_size=(64, 64), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.params = {{"gain": {TRUE_GAIN}}}

    def _centroid(self, image):
        mask = np.asarray(image)[:, :, 0] > 128
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return np.array([32.0, 32.0])
        return np.array([xs.mean(), ys.mean()])

    def fit(self, image_A, image_B):
        c = self._centroid(image_A)
        self.state = {{"p": c.copy(), "q": c.copy()}}
        self.target_state = {{"p": self._centroid(image_B)}}

    def update(self, a):
        a = np.asarray(a, dtype=float)
        self.state["q"] = self.state["q"] + self.params["gain"] * a
        self.state["p"] = self.state["p"] + 0.005 * a

    def loss_to_target(self):
        tp = self.target_state["p"]
        return float(
            np.linalg.norm(self.state["q"] - tp)
            + 0.01 * np.linalg.norm(self.state["p"] - tp)
        )

    def render_frame(self):
        img = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        x, y = int(round(self.state["p"][0])), int(round(self.state["p"][1]))
        x = min(max(x, 2), self.frame_size[0] - 3)
        y = min(max(y, 2), self.frame_size[1] - 3)
        img[y - 2 : y + 3, x - 2 : x + 3, 0] = 255
        return img
"""

# Target stored under a key that never appears in self.state: state-space
# goal-reach cannot be audited, and the tool must say so without crashing.
_NO_COMMON_COMPONENT_SIM = _TRUE_GAIN_SIM.replace(
    '        self.target_state = {"p": self._centroid(image_B)}',
    '        self.target_state = {"tp": self._centroid(image_B)}',
).replace(
    'return float(np.linalg.norm(self.state["p"] - self.target_state["p"]))',
    'return float(np.linalg.norm(self.state["p"] - self.target_state["tp"]))',
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root)
    return root


# --- validate_state_consistency ----------------------------------------------

def test_component_delta_wraps_multiple_revolutions_and_joint_arrays():
    scalar = PlanningCriticSandbox._component_delta_norm(
        "theta1",
        np.array([0.1]),
        np.array([0.3 + 8.0 * np.pi]),
    )
    vector = PlanningCriticSandbox._component_delta_norm(
        "qpos",
        np.array([0.1, -0.2]),
        np.array([0.3 + 6.0 * np.pi, -0.5 - 4.0 * np.pi]),
    )
    bounded_wrist = PlanningCriticSandbox._component_delta_norm(
        "theta2",
        np.array([np.pi - 0.1]),
        np.array([-np.pi + 0.1]),
        periodic=False,
    )

    assert scalar == pytest.approx(0.2)
    assert vector == pytest.approx(np.linalg.norm([0.2, -0.3]))
    assert bounded_wrist == pytest.approx(2.0 * np.pi - 0.2)


def test_state_consistency_true_gain_tracks_parses(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    summary = sandbox.validate_state_consistency(0)
    assert "[validate_state_consistency] trajectory 0" in summary
    assert "DRIFT" not in summary, summary
    assert "All components track your own perception" in summary, summary


def test_state_consistency_flags_wrong_gain_with_onset(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)  # gain 0.2 vs true 0.7
    summary = sandbox.validate_state_consistency(0)
    assert "DRIFT" in summary, summary
    assert "fix update(a)" in summary, summary
    # Onset step must be reported so the VLM can inspect the right frames.
    assert "exceeds 25% from ~t=" in summary, summary


def test_state_consistency_check_payload_shape(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    res = sandbox._run_state_consistency_check(0, max_checkpoints=6)
    assert res["ok"], res
    assert res["components"], res
    assert "p" in res["components"]
    assert res["components"]["p"]["verdict"] in {"OK", "DRIFT"}
    assert res["oneline"].startswith("t0:")
    assert res["compact_lines"]


# --- goal-reach in validate_with_cem ------------------------------------------

def test_target_configuration_requires_a_changed_task_target(tmp_path, dataset):
    faithful = _make_sandbox(tmp_path / "faithful", _TRUE_GAIN_SIM, dataset)
    missing = _make_sandbox(tmp_path / "missing", _NO_COMMON_COMPONENT_SIM, dataset)

    faithful_result = faithful._run_target_configuration_check([0, 1, 2])
    missing_result = missing._run_target_configuration_check([0, 1, 2])

    assert faithful_result["passed"], faithful_result
    assert missing_result["passed"] is False
    assert "no-changed-cost-sensitive-task-target" in missing_result["summary"]


def test_goal_reach_reached_for_faithful_sim(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    summary = sandbox.validate_with_cem(
        start_trajectory_index=0,
        start_frame_index=0,
        goal_trajectory_index=1,
        goal_frame_index=8,
        horizon=20,
        cem_iters=6,
        cem_population=150,
    )
    assert "goal-reach" in summary, summary
    assert "REACHED" in summary, summary
    assert "FAIL:" not in summary, summary


def test_goal_reach_flags_cost_state_mismatch(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _COST_STATE_MISMATCH_SIM, dataset)
    summary = sandbox.validate_with_cem(
        start_trajectory_index=0,
        start_frame_index=0,
        goal_trajectory_index=1,
        goal_frame_index=8,
        horizon=20,
        cem_iters=6,
        cem_population=150,
    )
    # The cost is drivable (q reaches the target), so the old cost-based
    # checks would pass; the state-space audit must fail on p.
    assert "FAIL: goal-reach" in summary, summary
    assert "'p'" in summary, summary


def test_goal_reach_missing_common_components_is_hard_fail(tmp_path, dataset):
    # 2026-07-07 change: "unavailable" used to be a NOTE that the CEM suite
    # treated as a pass (Reacher repeat_1 shipped through that hole); a
    # state/target key mismatch is now a structural FAIL.
    sandbox = _make_sandbox(tmp_path, _NO_COMMON_COMPONENT_SIM, dataset)
    summary = sandbox.validate_with_cem(
        start_trajectory_index=0,
        start_frame_index=0,
        goal_trajectory_index=1,
        goal_frame_index=8,
        horizon=20,
        cem_iters=4,
        cem_population=100,
    )
    assert "no numeric components shared" in summary, summary
    assert "FAIL: goal-reach audit impossible" in summary, summary
    assert "SAME keys" in summary, summary


# --- gate advisory + complaint hint -------------------------------------------

def test_gate_carries_state_consistency_advisory_and_hint(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)  # poor ratio
    verdict = sandbox.final_fidelity_gate([0, 1, 2], max_worst_ratio=0.5)
    assert verdict["passed"] is False
    sc = verdict["state_consistency"]
    assert sc is not None and sc.get("ok"), sc
    assert sc["oneline"]
    assert "HINT (state-consistency" in verdict["complaint"], verdict["complaint"]


def test_auto_validate_feedback_appends_state_consistency(tmp_path, dataset):
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0008.png"
    sandbox = PlanningCriticSandbox(
        code=_GOOD_SIM,
        fps=10,
        n_frames=8,
        frame_size=(FRAME, FRAME),
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset),
        simulator_class_name="Simulator",
        sandbox_dir=str(tmp_path / "sandbox"),
        tool_calls_log_dir=None,
        no_api=True,
        auto_validate_trajectories=[0, 1, 2],
        keep_best=False,
    )
    msg = sandbox.write_code_from_scratch(_GOOD_SIM)
    assert "[state-consistency on t" in msg, msg


# --- deployment goal contract (2026-07-07, task-agnostic Push-T r1 fix) --------

# A sim that parses the goal from a PERSISTENT cue (here: a fixed landmark)
# keeps a nonzero target residual even when image_B is the start observation.
_MARKER_TARGET_SIM = _TRUE_GAIN_SIM.replace(
    '        self.target_state = {"p": self._centroid(image_B)}',
    '        self.target_state = {"p": np.array([10.0, 10.0])}',
)


def _make_deploy_sandbox(tmp_path, code, dataset, mode):
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0008.png"
    return PlanningCriticSandbox(
        code=code,
        fps=10,
        n_frames=8,
        frame_size=(FRAME, FRAME),
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset),
        simulator_class_name="Simulator",
        sandbox_dir=str(tmp_path / "sandbox"),
        tool_calls_log_dir=None,
        no_api=True,
        gate_trajectories=[0, 1, 2],
        keep_best=False,
        deployment_goal_mode=mode,
    )


def test_deployment_goal_check_skipped_in_final_mode(tmp_path, dataset):
    sandbox = _make_deploy_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset, "final")
    res = sandbox._run_deployment_goal_check([0, 1, 2])
    assert res["passed"] is True
    assert res["skipped"] is True


def test_deployment_goal_check_flags_movable_target_parse(tmp_path, dataset):
    # Target parsed from the movable dot in image_B: faithful on training
    # pairs, degenerate when image_B is the start observation (the Push-T
    # repeat_1 failure).
    sandbox = _make_deploy_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset, "start")
    res = sandbox._run_deployment_goal_check([0, 1, 2])
    assert res["passed"] is False, res
    assert res["n_bad"] >= 2, res
    assert any(p["verdict"] == "degenerate" for p in res["probes"]), res


def test_deployment_goal_check_passes_persistent_cue_target(tmp_path, dataset):
    sandbox = _make_deploy_sandbox(tmp_path, _MARKER_TARGET_SIM, dataset, "start")
    res = sandbox._run_deployment_goal_check([0, 1, 2])
    assert res["passed"] is True, res
    assert all(p["verdict"] == "ok" for p in res["probes"]), res


def test_gate_fails_and_explains_deployment_goal_degeneracy(tmp_path, dataset):
    sandbox = _make_deploy_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset, "start")
    verdict = sandbox.final_fidelity_gate([0, 1, 2], max_worst_ratio=None)
    assert verdict["deployment_goal_ok"] is False
    assert verdict["passed"] is False
    assert "FAIL (deployment-goal)" in verdict["complaint"], verdict["complaint"]
    assert "persistent goal cues" in verdict["complaint"], verdict["complaint"]
