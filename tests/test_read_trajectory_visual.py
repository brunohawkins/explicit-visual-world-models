from __future__ import annotations

from PIL import Image

from tests.test_calibrate_parameters import FRAME, _build_dataset
from vdaworld.core.agentic_generator import _STUB_CODE
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox


def _make_sandbox(tmp_path, dataset):
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0008.png"
    return PlanningCriticSandbox(
        code=_STUB_CODE,
        fps=10,
        n_frames=8,
        frame_size=(FRAME, FRAME),
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset),
        simulator_class_name="Simulator",
        sandbox_dir=str(tmp_path / "sandbox"),
        tool_calls_log_dir=str(tmp_path / "tool_calls"),
        no_api=True,
    )


def test_read_trajectory_returns_text_by_default(tmp_path):
    dataset = tmp_path / "dataset"
    _build_dataset(dataset, n_traj=2, n_steps=8)
    (dataset / "trajectory_0000" / "metadata.json").write_text(
        '{"action_contract":{"semantics":"normalized_torque",'
        '"action_repeat":2,"low":[-1,-1],"high":[1,1]},'
        '"state_contract":{"planner_step_seconds":0.04,'
        '"components":{"shoulder":{"kind":"angle","periodic":true},'
        '"wrist":{"kind":"angle","periodic":false,"low":-2.79,"high":2.79}},'
        '"task_target":{"kind":"end_effector_to_visible_target",'
        '"branch_invariant":true}}}',
        encoding="utf-8",
    )
    sandbox = _make_sandbox(tmp_path, dataset)

    try:
        result = sandbox.read_trajectory(0, max_frames=4)
    finally:
        sandbox.cleanup()

    assert isinstance(result, str)
    assert "Trajectory 0:" in result
    assert "frame(s)" in result
    assert "Per-dimension statistics:" in result
    assert "a[0]" in result
    assert "mean=" in result
    assert "sum=" in result
    assert "DECLARED ACTION CONTRACT (authoritative)" in result
    assert "force-like control, not a pose increment" in result
    assert "DECLARED STATE/TASK CONTRACT (authoritative)" in result
    assert "branch_invariant" in result
    assert "bounded non-periodic coordinates must never cross" in result


def test_read_trajectory_can_return_visual_contact_sheet(tmp_path):
    dataset = tmp_path / "dataset"
    _build_dataset(dataset, n_traj=2, n_steps=8)
    sandbox = _make_sandbox(tmp_path, dataset)

    try:
        result = sandbox.read_trajectory(0, include_images=True, max_frames=4)
    finally:
        sandbox.cleanup()

    assert isinstance(result, Image.Image)
    assert result.width > FRAME
    assert result.height > FRAME
