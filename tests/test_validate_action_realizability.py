"""Tests for PlanningCriticSandbox.validate_action_realizability."""

from __future__ import annotations

import pytest

from tests.test_calibrate_parameters import (
    FRAME,
    TRUE_GAIN,
    _GOOD_SIM,
    _build_dataset,
    _make_sandbox,
)
from vdaworld.core.planning_agentic_generator import PlanningAgenticGenerator
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox

_TRUE_GAIN_SIM = _GOOD_SIM.replace(
    'self.params = {"gain": 0.2}',
    f'self.params = {{"gain": {TRUE_GAIN}}}',
)

_EXPLOIT_SIM = _TRUE_GAIN_SIM.replace(
    '        self.state["p"] = self.state["p"] + self.params["gain"] * np.asarray(a, dtype=float)',
    '        a = np.asarray(a, dtype=float)\n'
    '        step = self.params["gain"] * a\n'
    "        # Out-of-distribution exploit: huge one-step jump on extreme actions.\n"
    "        if float(np.linalg.norm(a)) > 8.0:\n"
    "            step = 20.0 * a\n"
    '        self.state["p"] = self.state["p"] + step',
)

_SECONDARY_EXPLOIT_SIM = _TRUE_GAIN_SIM.replace(
    'self.state = {"p": self._centroid(image_A)}',
    'self.state = {"p": self._centroid(image_A), "secondary": 0.0}',
).replace(
    '        self.state["p"] = self.state["p"] + self.params["gain"] * np.asarray(a, dtype=float)',
    '        a = np.asarray(a, dtype=float)\n'
    '        self.state["p"] = self.state["p"] + self.params["gain"] * a\n'
    "        if float(np.linalg.norm(a)) > 8.0:\n"
    '            self.state["secondary"] += 20.0 * float(a[0])',
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root, n_traj=4, n_steps=8)
    return root


def _make_generator(dataset, gate_trajectories):
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0008.png"
    return PlanningAgenticGenerator(
        vlm=None,
        prompts_path="",
        max_turns=5,
        n_frames=8,
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset),
        fps=10,
        frame_size=(FRAME, FRAME),
        simulator_class_name="Simulator",
        output_dir="",
        cache_dir=None,
        available_tools="",
        caption="",
        gate_trajectories=gate_trajectories,
    )


def test_realizable_sim_passes(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    summary = sandbox.validate_action_realizability()
    assert "PASS" in summary
    assert "training-derived one-step bounds" in summary


def test_exploit_sim_fails(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _EXPLOIT_SIM, dataset)
    summary = sandbox.validate_action_realizability(
        action_low=[-20.0, -20.0],
        action_high=[20.0, 20.0],
    )
    assert "FAIL" in summary
    assert "unrealizable one-step jump" in summary


def test_secondary_component_exploit_fails(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _SECONDARY_EXPLOIT_SIM, dataset)
    summary = sandbox.validate_action_realizability(
        action_low=[-20.0, -20.0],
        action_high=[20.0, 20.0],
    )

    assert "FAIL" in summary
    assert "secondary" in summary


def test_plan_audit_checks_non_primary_wrist_component() -> None:
    violations, missing = PlanningCriticSandbox._realizability_violations(
        {"theta1": 0.1, "theta2": 6.2},
        {"theta1": 1.0, "theta2": 0.25},
    )

    assert missing == []
    assert [item["component"] for item in violations] == ["theta2"]


def test_tool_registered_on_planning_generator(dataset):
    gen = _make_generator(dataset, gate_trajectories=[0, 1, 2])
    sandbox = gen._build_sandbox(
        sandbox_dir=str(dataset / "sb"),
        tool_calls_log_dir=str(dataset / "logs"),
        world_api_log_dir=str(dataset / "api_logs"),
    )
    names = [t.__name__ for t in gen._build_tool_callables(sandbox)]
    assert "validate_action_realizability" in names
    sandbox.cleanup()
