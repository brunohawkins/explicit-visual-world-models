from pathlib import Path

import pytest

from tests.test_calibrate_parameters import FRAME, _build_dataset
from vdaworld.core.agentic_generator import _STUB_CODE
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox


GOOD_CODE = """
class Simulator:
    MARKER = "GOOD"
"""

WORSE_CODE = """
class Simulator:
    MARKER = "WORSE"
"""

DEGENERATE_CODE = """
class Simulator:
    MARKER = "DEGENERATE"
"""


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root)
    return root


def _make_sandbox(tmp_path, dataset, *, keep_best=True):
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0008.png"
    sandbox = PlanningCriticSandbox(
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
        gate_trajectories=[0, 1],
        keep_best=keep_best,
    )
    sandbox._auto_validate_feedback = lambda: ""
    return sandbox


def _fake_score_for_current_code(sandbox):
    code = Path(sandbox._sandbox_path).read_text(encoding="utf-8")
    if "GOOD" in code:
        return {
            "per_traj": {0: 0.2, 1: 0.4},
            "worst": 0.4,
            "mean": 0.3,
            "n_failed": 0,
        }
    if "WORSE" in code:
        return {
            "per_traj": {0: 0.8, 1: 0.9},
            "worst": 0.9,
            "mean": 0.85,
            "n_failed": 0,
        }
    return {
        "per_traj": {0: None, 1: None},
        "worst": None,
        "mean": None,
        "n_failed": 2,
    }


def test_keep_best_restores_better_earlier_write(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, dataset)
    sandbox._score_full_train_fidelity = lambda: _fake_score_for_current_code(sandbox)

    sandbox.write_code_from_scratch(GOOD_CODE)
    sandbox.write_code_from_scratch(WORSE_CODE)

    assert "WORSE" in Path(sandbox._sandbox_path).read_text(encoding="utf-8")
    assert sandbox.restore_best_checkpoint() is True
    shipped = Path(sandbox._sandbox_path).read_text(encoding="utf-8")
    assert "GOOD" in shipped
    assert "WORSE" not in shipped

    record = sandbox.get_keep_best_record()
    assert record["keep_best_enabled"] is True
    assert record["shipped_checkpoint_tool_call"] == 0
    assert record["shipped_worst_ratio"] == pytest.approx(0.4)
    assert record["final_sandbox_worst_ratio"] == pytest.approx(0.9)
    assert record["keep_best_restored"] is True


def test_degenerate_later_write_does_not_replace_good_checkpoint(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, dataset)
    sandbox._score_full_train_fidelity = lambda: _fake_score_for_current_code(sandbox)

    sandbox.write_code_from_scratch(GOOD_CODE)
    sandbox.write_code_from_scratch(DEGENERATE_CODE)

    assert sandbox.restore_best_checkpoint() is True
    shipped = Path(sandbox._sandbox_path).read_text(encoding="utf-8")
    assert "GOOD" in shipped
    assert "DEGENERATE" not in shipped

    record = sandbox.get_keep_best_record()
    assert record["shipped_checkpoint_tool_call"] == 0
    assert record["shipped_worst_ratio"] == pytest.approx(0.4)
    assert record["final_sandbox_worst_ratio"] is None


def test_keep_best_disabled_ships_last_write(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, dataset, keep_best=False)
    sandbox._score_full_train_fidelity = pytest.fail

    sandbox.write_code_from_scratch(GOOD_CODE)
    sandbox.write_code_from_scratch(WORSE_CODE)

    assert sandbox.restore_best_checkpoint() is False
    shipped = Path(sandbox._sandbox_path).read_text(encoding="utf-8")
    assert "WORSE" in shipped
    assert "GOOD" not in shipped

    record = sandbox.get_keep_best_record()
    assert record["keep_best_enabled"] is False
    assert record["shipped_checkpoint_tool_call"] is None


def test_scorer_aborts_early_against_incumbent_and_is_not_checkpointed(
    tmp_path, dataset
):
    """The real scorer sweeps worst-first and stops once the incumbent best
    cannot be beaten; such partial scores must never become checkpoints."""
    sandbox = _make_sandbox(tmp_path, dataset)
    sandbox.write_code_from_scratch(GOOD_CODE.replace("GOOD", "REAL"))

    # Incumbent best: worst 0.4. Last score ranked t1 as previously worst.
    sandbox._keep_best_best = {
        "tool_call": 0,
        "code": "incumbent",
        "score": {"per_traj": {0: 0.2, 1: 0.4}, "worst": 0.4, "mean": 0.3, "n_failed": 0},
    }
    sandbox._keep_best_last = {
        "tool_call": 0,
        "code": "incumbent",
        "score": {"per_traj": {0: 0.2, 1: 0.9}, "worst": 0.9, "mean": 0.55, "n_failed": 0},
    }

    calls: list[int] = []

    def fake_validate(trajectory_index, **kwargs):
        calls.append(trajectory_index)
        return f"reduction_ratio={0.95 if trajectory_index == 1 else 0.1}"

    sandbox.validate_against_training = fake_validate
    score = sandbox._score_full_train_fidelity()

    # Worst-first ordering per the last score: t1 first; 0.95 > 0.4 aborts.
    assert calls == [1], calls
    assert score.get("aborted") is True
    assert sandbox._keep_best_score_is_better(
        score, sandbox._keep_best_best["score"]
    ) is False


def test_keep_best_prioritises_required_quality_over_replay_ratio():
    perception_good = {
        "worst": 0.8,
        "mean": 0.7,
        "n_failed": 0,
        "quality_checks": 2,
        "quality_failures": 0,
    }
    perception_bad = {
        "worst": 0.2,
        "mean": 0.1,
        "n_failed": 0,
        "quality_checks": 2,
        "quality_failures": 1,
    }

    assert PlanningCriticSandbox._keep_best_score_is_better(
        perception_good, perception_bad
    )
    assert not PlanningCriticSandbox._keep_best_score_is_better(
        perception_bad, perception_good
    )
