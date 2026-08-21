"""Unit tests for dataset_tools."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vdaworld.core.dataset_tools import (
    _normalise_actions,
    read_trajectory,
    view_action,
    view_image,
)

_DATASET_DIR = Path(__file__).resolve().parents[2] / "Datasets" / "two-rooms"


def test_normalise_list_of_dicts():
    raw = [
        {"step": 1, "action": [0.5, -0.3]},
        {"step": 0, "action": [0.1, 0.2]},
        {"step": 2, "action": [-0.4, 0.7]},
    ]
    arr = _normalise_actions(raw)
    assert arr.shape == (3, 2)
    np.testing.assert_allclose(arr[0], [0.1, 0.2])
    np.testing.assert_allclose(arr[2], [-0.4, 0.7])


def test_normalise_bare_list():
    raw = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    arr = _normalise_actions(raw)
    assert arr.shape == (3, 2)
    np.testing.assert_allclose(arr[1], [0.3, 0.4])


def test_normalise_empty():
    arr = _normalise_actions([])
    assert arr.shape == (0, 0)


def test_normalise_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported actions.json element type"):
        _normalise_actions(["string_not_a_dict_or_list"])


@pytest.mark.skipif(not _DATASET_DIR.exists(), reason="two-rooms dataset not available")
def test_read_trajectory_two_rooms():
    traj = read_trajectory(_DATASET_DIR, 0)
    assert traj["n_steps"] == 10
    assert traj["actions"].shape == (10, 2)
    assert len(traj["frames"]) == 11
    # Frame paths should be sorted in order.
    names = [p.name for p in traj["frames"]]
    assert names == sorted(names)


@pytest.mark.skipif(not _DATASET_DIR.exists(), reason="two-rooms dataset not available")
def test_view_image_returns_uint8_rgb():
    img = view_image(_DATASET_DIR, 0, 0)
    assert img.dtype == np.uint8
    assert img.ndim == 3
    assert img.shape[2] == 3
    assert img.shape[:2] == (65, 65)


@pytest.mark.skipif(not _DATASET_DIR.exists(), reason="two-rooms dataset not available")
def test_view_action_returns_correct_step():
    a0 = view_action(_DATASET_DIR, 0, 0)
    traj = read_trajectory(_DATASET_DIR, 0)
    np.testing.assert_allclose(a0, traj["actions"][0])

    with pytest.raises(IndexError):
        view_action(_DATASET_DIR, 0, 100)


def test_read_trajectory_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_trajectory(tmp_path, 99)


def test_read_trajectory_exposes_only_declared_action_contract(tmp_path):
    traj_dir = tmp_path / "trajectory_0000"
    traj_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        traj_dir / "frame_0000.png"
    )
    (traj_dir / "actions.json").write_text(
        json.dumps([{"step": 0, "action": [0.1, -0.2]}]),
        encoding="utf-8",
    )
    (traj_dir / "metadata.json").write_text(
        json.dumps(
            {
                "action_contract": {
                    "semantics": "normalized_torque",
                    "action_repeat": 2,
                    "low": [-1.0, -1.0],
                    "high": [1.0, 1.0],
                    "private_field": "not exposed",
                },
                "state_contract": {
                    "planner_step_seconds": 0.04,
                    "components": {
                        "shoulder": {"kind": "angle", "periodic": True},
                        "wrist": {
                            "kind": "angle",
                            "periodic": False,
                            "low": -2.79,
                            "high": 2.79,
                        },
                    },
                    "task_target": {
                        "kind": "end_effector_to_visible_target",
                        "branch_invariant": True,
                    },
                    "private_field": "not exposed",
                },
                "qpos": [[1.0, 2.0]],
            }
        ),
        encoding="utf-8",
    )

    trajectory = read_trajectory(tmp_path, 0)

    assert trajectory["action_contract"] == {
        "semantics": "normalized_torque",
        "action_repeat": 2,
        "low": [-1.0, -1.0],
        "high": [1.0, 1.0],
    }
    assert trajectory["state_contract"] == {
        "planner_step_seconds": 0.04,
        "components": {
            "shoulder": {"kind": "angle", "periodic": True},
            "wrist": {
                "kind": "angle",
                "periodic": False,
                "low": -2.79,
                "high": 2.79,
            },
        },
        "task_target": {
            "kind": "end_effector_to_visible_target",
            "branch_invariant": True,
        },
    }
    assert "qpos" not in trajectory


def test_read_trajectory_derives_legacy_bounded_elbow_contract(tmp_path):
    traj_dir = tmp_path / "trajectory_0000"
    traj_dir.mkdir()
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
        traj_dir / "frame_0000.png"
    )
    (traj_dir / "actions.json").write_text(
        json.dumps([{"step": 0, "action": [0.0, 0.0]}]),
        encoding="utf-8",
    )
    (traj_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task": "end_effector_to_visible_target",
                "action_contract": {
                    "semantics": "dm_control_reacher_normalized_torque",
                    "action_repeat": 2,
                },
                "controller": {
                    "joint_error": "wrapped_shoulder_and_raw_bounded_elbow"
                },
            }
        ),
        encoding="utf-8",
    )

    contract = read_trajectory(tmp_path, 0)["state_contract"]

    assert contract["planner_step_seconds"] == pytest.approx(0.04)
    assert contract["components"]["shoulder"]["periodic"] is True
    assert contract["components"]["wrist"]["periodic"] is False
    assert contract["components"]["wrist"]["state_keys"] == [
        "theta2",
        "wrist",
        "elbow",
    ]


def test_view_image_missing_file(tmp_path):
    (tmp_path / "trajectory_0000").mkdir()
    with pytest.raises(FileNotFoundError):
        view_image(tmp_path, 0, 0)
