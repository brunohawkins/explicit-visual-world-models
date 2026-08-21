"""Focused tests for the Reacher torque dataset and action-repeat contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vdaworld.eval.reacher_env import (
    REACHER_WRIST_LIMIT,
    ReacherDMControlSession,
    _select_reacher_goal_qpos,
)


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_reacher_torque_expert_dataset as torque_builder  # noqa: E402


class _Timestep:
    def __init__(self, *, reward: float, last: bool = False) -> None:
        self.reward = reward
        self._last = last

    def last(self) -> bool:
        return self._last


class _FakeEnv:
    def __init__(self, timesteps: list[_Timestep]) -> None:
        self.timesteps = list(timesteps)
        self.actions: list[np.ndarray] = []

    def step(self, action: np.ndarray) -> _Timestep:
        self.actions.append(np.asarray(action, dtype=np.float64).copy())
        return self.timesteps[len(self.actions) - 1]

    def close(self) -> None:
        return None


def _install_fake_env(
    session: ReacherDMControlSession,
    timesteps: list[_Timestep],
    *,
    successful_calls: set[int] | None = None,
) -> _FakeEnv:
    env = _FakeEnv(timesteps)
    successes = successful_calls or set()
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    session._env = env
    session._goal_image = frame.copy()
    session._render = lambda: frame.copy()
    session._info = lambda: {
        "distance_to_target": 0.0 if len(env.actions) in successes else 1.0,
        "target_radius": 0.1,
        "success": len(env.actions) in successes,
    }
    return env


def test_default_action_repeat_executes_one_planner_action_twice() -> None:
    session = ReacherDMControlSession()
    env = _install_fake_env(
        session,
        [_Timestep(reward=1.0), _Timestep(reward=2.0)],
    )

    result = session.step([3.0, -3.0])

    assert len(env.actions) == 2
    for action in env.actions:
        np.testing.assert_array_equal(action, [1.0, -1.0])
    assert result["reward"] == 2.0
    assert result["done"] is False
    assert result["truncated"] is False
    assert result["info"]["action_repeat"] == 2
    assert result["info"]["executed_action_repeat"] == 2
    assert [item["substep_index"] for item in result["info"]["substeps"]] == [0, 1]


def test_manifest_repeat_stops_after_success_at_first_substep() -> None:
    session = ReacherDMControlSession(
        manifest_case={
            "case_id": "reacher-7",
            "action_contract": {"action_repeat": 4},
        },
        action_repeat=2,
    )
    env = _install_fake_env(
        session,
        [_Timestep(reward=0.5)],
        successful_calls={1},
    )

    result = session.step([0.25, -0.5])

    assert len(env.actions) == 1
    assert result["done"] is True
    assert result["truncated"] is False
    assert result["info"]["manifest_case_id"] == "reacher-7"
    assert result["info"]["action_repeat"] == 4
    assert result["info"]["executed_action_repeat"] == 1
    assert result["info"]["substeps"][0]["success"] is True


def test_action_repeat_stops_when_environment_is_last() -> None:
    session = ReacherDMControlSession(action_repeat=4)
    env = _install_fake_env(
        session,
        [
            _Timestep(reward=1.0),
            _Timestep(reward=2.0, last=True),
        ],
    )

    result = session.step([0.0, 0.0])

    assert len(env.actions) == 2
    assert result["done"] is False
    assert result["truncated"] is True
    assert result["info"]["executed_action_repeat"] == 2
    assert result["info"]["substeps"][-1]["env_last"] is True


def test_goal_qpos_selection_prefers_nearby_wrist_branch() -> None:
    reference = np.array([2.3, -2.15])
    selected, error = _select_reacher_goal_qpos(
        [
            (np.array([-2.7, 2.17]), 1.0e-9),
            (np.array([-0.53, -2.17]), 2.0e-9),
        ],
        reference,
    )

    assert selected[1] == pytest.approx(-2.17)
    assert abs(selected[0] - reference[0]) < np.pi
    assert error == pytest.approx(2.0e-9)


def test_goal_qpos_selection_rejects_out_of_range_wrist() -> None:
    reference = np.array([0.1, REACHER_WRIST_LIMIT - 0.05])
    selected, _ = _select_reacher_goal_qpos(
        [
            (np.array([0.2, REACHER_WRIST_LIMIT + 0.02]), 1.0e-10),
            (np.array([0.25, REACHER_WRIST_LIMIT - 0.10]), 1.0e-8),
        ],
        reference,
    )

    assert selected[1] == pytest.approx(REACHER_WRIST_LIMIT - 0.10)


def test_goal_qpos_selection_requires_a_feasible_candidate() -> None:
    with pytest.raises(ValueError, match="wrist bounds"):
        _select_reacher_goal_qpos(
            [(np.array([0.0, np.pi]), 0.0)],
            np.zeros(2),
        )


def test_builder_declares_bounded_wrist_state_contract() -> None:
    contract = torque_builder._state_contract(action_repeat=2)

    assert contract["planner_step_seconds"] == pytest.approx(0.04)
    assert contract["components"]["shoulder"]["periodic"] is True
    wrist = contract["components"]["wrist"]
    assert wrist["periodic"] is False
    assert wrist["state_keys"] == ["theta2", "wrist", "elbow"]
    assert wrist["low"] == pytest.approx(-REACHER_WRIST_LIMIT)
    assert wrist["high"] == pytest.approx(REACHER_WRIST_LIMIT)
    assert contract["task_target"]["branch_invariant"] is True


def test_source_manifest_helpers_preserve_order_seeds_and_reset_identity(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "manifest.json"
    source_path.write_text(
        json.dumps(
            {
                "benchmark": "reacher",
                "trajectories": [
                    {
                        "trajectory_index": 9,
                        "seed": 101,
                        "start_qpos": [0.1, -0.2],
                        "target_pos": [0.3, 0.4],
                    },
                    {
                        "trajectory_index": 2,
                        "seed": 55,
                        "target_pos": [-0.3, 0.2],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    _, source_cases = torque_builder._load_source_manifest(source_path)

    assert [case["trajectory_index"] for case in source_cases] == [9, 2]
    assert [case["seed"] for case in source_cases] == [101, 55]
    torque_builder._validate_source_reset(
        source_cases[0],
        seed=101,
        start_qpos=np.array([0.1, -0.2]),
        target_pos=np.array([0.3, 0.4]),
    )
    with pytest.raises(ValueError, match="target position"):
        torque_builder._validate_source_reset(
            source_cases[0],
            seed=101,
            start_qpos=np.array([0.1, -0.2]),
            target_pos=np.array([0.3, 0.5]),
        )
