"""Tests for PlanningCriticSandbox.validate_all_trajectories."""

from __future__ import annotations

import pytest

from tests.test_calibrate_parameters import (
    _GOOD_SIM,
    _build_dataset,
    _make_sandbox,
)
from tests.test_final_fidelity_gate import _CRASH_SIM, _make_generator


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root, n_traj=4, n_steps=8)
    return root


def test_validate_all_trajectories_summary(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    sandbox._gate_trajectories = [0, 1, 2, 3]
    summary = sandbox.validate_all_trajectories()
    assert "validate_all_trajectories" in summary
    for t in (0, 1, 2, 3):
        assert f"t{t}=" in summary
    assert "mean reduction_ratio" in summary
    assert "WORST = trajectory" in summary
    assert "reduction_ratio=" in summary


def test_crashing_sim_reported_as_failure(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _CRASH_SIM, dataset)
    sandbox._gate_trajectories = [0, 1, 2]
    summary = sandbox.validate_all_trajectories()
    assert "FAILED/degenerate" in summary
    assert "t0" in summary


def test_tool_registered_on_planning_generator(dataset):
    gen = _make_generator(dataset, gate_trajectories=[0, 1, 2])
    sandbox = gen._build_sandbox(
        sandbox_dir=str(dataset / "sb"),
        tool_calls_log_dir=str(dataset / "logs"),
        world_api_log_dir=str(dataset / "api_logs"),
    )
    names = [t.__name__ for t in gen._build_tool_callables(sandbox)]
    assert "validate_all_trajectories" in names
    sandbox.cleanup()
