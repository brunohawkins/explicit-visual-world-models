"""Tests for PlanningCriticSandbox.validate_goal_invariance."""

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

_TRUE_GAIN_SIM = _GOOD_SIM.replace(
    'self.params = {"gain": 0.2}',
    f'self.params = {{"gain": {TRUE_GAIN}}}',
)

_GOAL_DRIFT_SIM = _TRUE_GAIN_SIM.replace(
    '        self.target_state = {"p": self._centroid(image_B)}',
    '        self.target_state = {"p": self._centroid(image_A)}',
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


def test_image_b_goal_sim_passes_goal_invariance_tool(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    summary = sandbox.validate_goal_invariance(trajectory_indices=[0, 1, 2])

    assert "validate_goal_invariance" in summary
    assert "PASS" in summary
    assert "target_state stayed invariant" in summary


def test_image_a_goal_sim_fails_goal_invariance_tool(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOAL_DRIFT_SIM, dataset)
    summary = sandbox.validate_goal_invariance(trajectory_indices=[0, 1, 2])

    assert "FAIL" in summary
    assert "target_state drift" in summary
    assert "image_B" in summary


def test_goal_invariance_tool_agrees_with_gate_criterion(tmp_path, dataset):
    passing_sandbox = _make_sandbox(tmp_path / "pass", _TRUE_GAIN_SIM, dataset)
    passing_summary = passing_sandbox.validate_goal_invariance(
        trajectory_indices=[0, 1, 2]
    )
    passing_gate = passing_sandbox.final_fidelity_gate(
        [0, 1, 2],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=True,
    )

    failing_sandbox = _make_sandbox(tmp_path / "fail", _GOAL_DRIFT_SIM, dataset)
    failing_summary = failing_sandbox.validate_goal_invariance(
        trajectory_indices=[0, 1, 2]
    )
    failing_gate = failing_sandbox.final_fidelity_gate(
        [0, 1, 2],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=True,
    )

    assert ("PASS" in passing_summary) is passing_gate["goal_invariance_ok"]
    assert passing_gate["passed"] is True
    assert ("PASS" in failing_summary) is failing_gate["goal_invariance_ok"]
    assert failing_gate["passed"] is False


def test_goal_invariance_tool_registered_on_planning_generator(dataset):
    gen = _make_generator(dataset, gate_trajectories=[0, 1, 2])
    sandbox = gen._build_sandbox(
        sandbox_dir=str(dataset / "sb"),
        tool_calls_log_dir=str(dataset / "logs"),
        world_api_log_dir=str(dataset / "api_logs"),
    )
    names = [t.__name__ for t in gen._build_tool_callables(sandbox)]
    assert "validate_goal_invariance" in names
    sandbox.cleanup()
