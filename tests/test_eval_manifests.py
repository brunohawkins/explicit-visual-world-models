"""Tests for opt-in strict evaluation-manifest validation."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from vdaworld.eval.manifest import (
    load_manifest,
    manifest_sha256,
    validate_manifest,
)


def _image_ref(row: int) -> dict:
    return {
        "type": "hdf5_dataset_row",
        "path": "/data/source.h5",
        "key": "pixels",
        "row": row,
    }


def _case_state(benchmark: str) -> dict:
    if benchmark == "pusht":
        state = [1.0, 2.0, 3.0, 4.0, 0.5, 6.0, 7.0]
        goal_state = [2.0, 3.0, 4.0, 5.0, 0.75, 7.0, 8.0]
        return {
            "start": {
                "state": state,
                "pos_agent": state[:2],
                "block_pose": state[2:5],
                "vel_agent": state[5:],
            },
            "goal": {
                "state": goal_state,
                "pos_agent": goal_state[:2],
                "block_pose": goal_state[2:5],
                "vel_agent": goal_state[5:],
            },
        }
    if benchmark == "reacher":
        return {
            "qpos": [0.1, 0.2],
            "qvel": [0.3, 0.4],
            "goal_qpos": [0.5, 0.6],
            "goal_qvel": [0.7, 0.8],
        }
    if benchmark == "two_room_swm":
        return {
            "start": {
                "state": [40.0, 80.0],
                "observation": [40.0, 80.0, 160.0, 140.0, 112.0, 49.0, 0, 0, 0, 0],
            },
            "goal": {
                "state": [160.0, 140.0],
                "observation": [160.0, 140.0, 160.0, 140.0, 112.0, 49.0, 0, 0, 0, 0],
            },
        }
    return {
        "qpos": [0.1] * 21,
        "qvel": [0.2] * 20,
        "goal_qpos": [0.3] * 21,
        "goal_qvel": [0.4] * 20,
        "target_pos": [0.4, 0.0, 0.02],
        "target_quat": [1.0, 0.0, 0.0, 0.0],
        "goal_privileged_block_0_pos": [0.4, 0.0, 0.02],
        "goal_privileged_block_0_quat": [1.0, 0.0, 0.0, 0.0],
    }


def _environment(benchmark: str) -> dict:
    if benchmark == "pusht":
        return {
            "id": "swm/PushT-v1",
            "native_resolution": 224,
            "options": {
                "relative": True,
                "resolution": 224,
                "render_mode": "rgb_array",
                "max_episode_steps": 100,
            },
        }
    if benchmark == "reacher":
        return {
            "id": "swm/ReacherDMControl-v0",
            "native_resolution": 224,
            "options": {
                "task": "qpos_match",
                "max_episode_steps": 100,
            },
        }
    if benchmark == "two_room_swm":
        return {
            "id": "swm/TwoRoom-v1",
            "native_resolution": 224,
            "options": {
                "render_mode": "rgb_array",
                "max_episode_steps": 100,
            },
        }
    return {
        "id": "swm/OGBCube-v0",
        "native_resolution": 224,
        "options": {
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": True,
            "max_episode_steps": 100,
        },
    }


def _semantics(benchmark: str) -> tuple[dict, dict]:
    if benchmark == "pusht":
        return (
            {
                "name": "future_full_state_pose_match",
                "position_threshold": 20.0,
                "position_comparator": "<",
                "angle_threshold_expression": "pi/9",
                "angle_comparator": "<",
                "velocity_used_for_success": False,
            },
            {
                "action_scale": 100.0,
                "physics_steps_per_action": 10,
            },
        )
    if benchmark == "reacher":
        return (
            {
                "name": "qpos_match",
                "angle_wrapping": False,
                "threshold": 0.05,
                "comparator": "<",
                "all_joints_required": True,
            },
            {"action_repeat": 2},
        )
    if benchmark == "two_room_swm":
        return (
            {
                "name": "distance_to_future_position",
                "threshold": 16.0,
                "comparator": "<",
            },
            {
                "low": [-1.0, -1.0],
                "high": [1.0, 1.0],
                "speed_pixels_per_step": 5.0,
            },
        )
    return (
        {
            "name": "cube_position",
            "threshold": 0.04,
            "comparator": "<=",
            "orientation_used_for_success": False,
        },
        {},
    )


def _valid_manifest(benchmark: str) -> dict:
    success, action_contract = _semantics(benchmark)
    cases = []
    for index, seed in enumerate(range(42, 92)):
        row = 1000 + index * 100
        cases.append(
            {
                "case_id": f"{benchmark}-{index}",
                "seed": seed,
                "benchmark": benchmark,
                "dataset_row": row,
                "goal_dataset_row": row + 25,
                "episode_idx": index,
                "step_idx": 10,
                "goal_step_idx": 35,
                "goal_offset_steps": 25,
                "eval_budget": 50,
                "start_image": _image_ref(row),
                "goal_image": _image_ref(row + 25),
                **_case_state(benchmark),
            }
        )
    return {
        "schema_version": "v2",
        "protocol": "lewm_aligned_n50_v1",
        "benchmark": benchmark,
        "selection_seed": 42,
        "case_seed_start": 42,
        "num_eval": 50,
        "goal_offset_steps": 25,
        "eval_budget": 50,
        "source_hdf5": "/data/source.h5",
        "environment": _environment(benchmark),
        "success": success,
        "action_contract": action_contract,
        "cases": cases,
    }


@pytest.mark.parametrize(
    "benchmark",
    ["pusht", "two_room_swm", "reacher", "cube"],
)
def test_strict_lewm_manifest_accepts_each_benchmark(
    benchmark: str,
) -> None:
    assert validate_manifest(_valid_manifest(benchmark)) == []


def test_strict_validator_returns_all_useful_errors() -> None:
    manifest = copy.deepcopy(_valid_manifest("reacher"))
    manifest["goal_offset_steps"] = 24
    manifest["cases"][1]["seed"] = 42
    manifest["cases"][2]["goal_dataset_row"] += 1
    del manifest["cases"][3]["goal_image"]

    errors = validate_manifest(manifest, raise_on_error=False)

    assert any("goal_offset_steps" in error for error in errors)
    assert any("seeds must be unique" in error for error in errors)
    assert any("goal_dataset_row" in error for error in errors)
    assert any("cases[3].goal_image" in error for error in errors)
    with pytest.raises(ValueError, match="invalid manifest"):
        validate_manifest(manifest)


def test_strict_validator_rejects_pusht_goal_pose_alias() -> None:
    manifest = _valid_manifest("pusht")
    manifest["cases"][0]["goal"]["goal_pose"] = [256.0, 256.0, 0.0]

    errors = validate_manifest(manifest, raise_on_error=False)

    assert any("goal.goal_pose" in error for error in errors)


def test_load_manifest_keeps_legacy_format_behavior(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"version": 1, "cases": []}), encoding="utf-8")

    assert load_manifest(path) == {"version": 1, "cases": []}


def test_manifest_sha256_hashes_exact_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = b'{\n  "cases": []\n}\n'
    path.write_bytes(payload)

    assert manifest_sha256(path) == hashlib.sha256(payload).hexdigest()
