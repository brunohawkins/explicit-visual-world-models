"""Shared matched-evaluation manifest helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


LEWM_ALIGNED_PROFILE = "lewm_aligned_n50_v1"
_LEWM_BENCHMARKS = {"pusht", "two_room_swm", "reacher", "cube"}
_LEWM_ENV_IDS = {
    "pusht": "swm/PushT-v1",
    "two_room_swm": "swm/TwoRoom-v1",
    "reacher": "swm/ReacherDMControl-v0",
    "cube": "swm/OGBCube-v0",
}


def load_manifest(path: Path | str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("cases"), list):
        raise ValueError(f"invalid manifest format in {manifest_path}")
    return payload


def case_for_seed(manifest: dict[str, Any] | None, seed: int) -> dict[str, Any] | None:
    if manifest is None:
        return None
    for case in manifest.get("cases", []):
        if int(case.get("seed")) == int(seed):
            return dict(case)
    raise KeyError(f"manifest has no case for seed={seed}")


def manifest_sha256(path: Path | str) -> str:
    """Return the SHA-256 of the manifest file's exact bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _expect_equal(
    errors: list[str],
    path: str,
    actual: Any,
    expected: Any,
) -> None:
    if actual != expected:
        errors.append(f"{path}: expected {expected!r}, got {actual!r}")


def _numeric_sequence(
    errors: list[str],
    path: str,
    value: Any,
    length: int,
) -> list[int | float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
        or any(not _is_number(item) for item in value)
    ):
        errors.append(f"{path}: expected a numeric sequence of length {length}")
        return None
    return list(value)


def _validate_image_ref(
    errors: list[str],
    *,
    path: str,
    value: Any,
    expected_hdf5: str | None,
    expected_row: int | None,
) -> None:
    if not isinstance(value, Mapping):
        errors.append(f"{path}: expected an HDF5 image reference object")
        return
    _expect_equal(
        errors,
        f"{path}.type",
        value.get("type"),
        "hdf5_dataset_row",
    )
    ref_path = value.get("path")
    if not isinstance(ref_path, str) or not ref_path:
        errors.append(f"{path}.path: expected a non-empty string")
    elif expected_hdf5 is not None and ref_path != expected_hdf5:
        errors.append(
            f"{path}.path: expected top-level source_hdf5 "
            f"{expected_hdf5!r}, got {ref_path!r}"
        )
    _expect_equal(errors, f"{path}.key", value.get("key"), "pixels")
    row = value.get("row")
    if not _is_int(row):
        errors.append(f"{path}.row: expected an integer")
    elif expected_row is not None and row != expected_row:
        errors.append(f"{path}.row: expected {expected_row}, got {row}")


def _validate_environment(
    errors: list[str],
    manifest: Mapping[str, Any],
    benchmark: str,
) -> None:
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        errors.append("environment: expected an object")
        return
    _expect_equal(
        errors,
        "environment.id",
        environment.get("id"),
        _LEWM_ENV_IDS[benchmark],
    )
    _expect_equal(
        errors,
        "environment.native_resolution",
        environment.get("native_resolution"),
        224,
    )
    options = environment.get("options")
    if not isinstance(options, Mapping):
        errors.append("environment.options: expected an object")
        return

    if benchmark == "pusht":
        for key, expected in {
            "relative": True,
            "resolution": 224,
            "render_mode": "rgb_array",
            "max_episode_steps": 100,
        }.items():
            _expect_equal(
                errors,
                f"environment.options.{key}",
                options.get(key),
                expected,
            )
    elif benchmark == "two_room_swm":
        for key, expected in {
            "render_mode": "rgb_array",
            "max_episode_steps": 100,
        }.items():
            _expect_equal(
                errors,
                f"environment.options.{key}",
                options.get(key),
                expected,
            )
    elif benchmark == "reacher":
        _expect_equal(
            errors,
            "environment.options.task",
            options.get("task"),
            "qpos_match",
        )
        _expect_equal(
            errors,
            "environment.options.max_episode_steps",
            options.get("max_episode_steps"),
            100,
        )
    else:
        for key, expected in {
            "env_type": "single",
            "ob_type": "states",
            "multiview": False,
            "width": 224,
            "height": 224,
            "visualize_info": False,
            "terminate_at_goal": True,
            "max_episode_steps": 100,
        }.items():
            _expect_equal(
                errors,
                f"environment.options.{key}",
                options.get(key),
                expected,
            )


def _validate_semantics(
    errors: list[str],
    manifest: Mapping[str, Any],
    benchmark: str,
) -> None:
    success = manifest.get("success")
    if not isinstance(success, Mapping):
        errors.append("success: expected an object")
    elif benchmark == "pusht":
        for key, expected in {
            "name": "future_full_state_pose_match",
            "position_threshold": 20.0,
            "position_comparator": "<",
            "angle_threshold_expression": "pi/9",
            "angle_comparator": "<",
            "velocity_used_for_success": False,
        }.items():
            _expect_equal(
                errors,
                f"success.{key}",
                success.get(key),
                expected,
            )
    elif benchmark == "two_room_swm":
        for key, expected in {
            "name": "distance_to_future_position",
            "threshold": 16.0,
            "comparator": "<",
        }.items():
            _expect_equal(
                errors,
                f"success.{key}",
                success.get(key),
                expected,
            )
    elif benchmark == "reacher":
        for key, expected in {
            "name": "qpos_match",
            "angle_wrapping": False,
            "threshold": 0.05,
            "comparator": "<",
            "all_joints_required": True,
        }.items():
            _expect_equal(
                errors,
                f"success.{key}",
                success.get(key),
                expected,
            )
    else:
        for key, expected in {
            "name": "cube_position",
            "threshold": 0.04,
            "comparator": "<=",
            "orientation_used_for_success": False,
        }.items():
            _expect_equal(
                errors,
                f"success.{key}",
                success.get(key),
                expected,
            )

    action = manifest.get("action_contract")
    if not isinstance(action, Mapping):
        errors.append("action_contract: expected an object")
    elif benchmark == "pusht":
        _expect_equal(
            errors,
            "action_contract.action_scale",
            action.get("action_scale"),
            100.0,
        )
        _expect_equal(
            errors,
            "action_contract.physics_steps_per_action",
            action.get("physics_steps_per_action"),
            10,
        )
    elif benchmark == "two_room_swm":
        _numeric_sequence(
            errors,
            "action_contract.low",
            action.get("low"),
            2,
        )
        _numeric_sequence(
            errors,
            "action_contract.high",
            action.get("high"),
            2,
        )
        _expect_equal(
            errors,
            "action_contract.speed_pixels_per_step",
            action.get("speed_pixels_per_step"),
            5.0,
        )
    elif benchmark == "reacher":
        _expect_equal(
            errors,
            "action_contract.action_repeat",
            action.get("action_repeat"),
            2,
        )


def _validate_case_state(
    errors: list[str],
    *,
    case: Mapping[str, Any],
    case_path: str,
    benchmark: str,
) -> None:
    if benchmark == "pusht":
        for state_name in ("start", "goal"):
            state_path = f"{case_path}.{state_name}"
            state_value = case.get(state_name)
            if not isinstance(state_value, Mapping):
                errors.append(f"{state_path}: expected an object")
                continue
            state = _numeric_sequence(
                errors,
                f"{state_path}.state",
                state_value.get("state"),
                7,
            )
            pos_agent = _numeric_sequence(
                errors,
                f"{state_path}.pos_agent",
                state_value.get("pos_agent"),
                2,
            )
            block_pose = _numeric_sequence(
                errors,
                f"{state_path}.block_pose",
                state_value.get("block_pose"),
                3,
            )
            vel_agent = _numeric_sequence(
                errors,
                f"{state_path}.vel_agent",
                state_value.get("vel_agent"),
                2,
            )
            if (
                state is not None
                and pos_agent is not None
                and block_pose is not None
                and vel_agent is not None
                and state != pos_agent + block_pose + vel_agent
            ):
                errors.append(
                    f"{state_path}: split components do not reconstruct state"
                )
        goal = case.get("goal")
        if isinstance(goal, Mapping) and "goal_pose" in goal:
            errors.append(
                f"{case_path}.goal.goal_pose: fixed rendered goal_pose must "
                "not alias the future block pose"
            )
    elif benchmark == "two_room_swm":
        for state_name in ("start", "goal"):
            state_path = f"{case_path}.{state_name}"
            state_value = case.get(state_name)
            if not isinstance(state_value, Mapping):
                errors.append(f"{state_path}: expected an object")
                continue
            _numeric_sequence(
                errors,
                f"{state_path}.state",
                state_value.get("state"),
                2,
            )
            _numeric_sequence(
                errors,
                f"{state_path}.observation",
                state_value.get("observation"),
                10,
            )
    elif benchmark == "reacher":
        for key in ("qpos", "qvel", "goal_qpos", "goal_qvel"):
            _numeric_sequence(
                errors,
                f"{case_path}.{key}",
                case.get(key),
                2,
            )
    else:
        for key, length in {
            "qpos": 21,
            "qvel": 20,
            "goal_qpos": 21,
            "goal_qvel": 20,
            "target_pos": 3,
            "target_quat": 4,
            "goal_privileged_block_0_pos": 3,
            "goal_privileged_block_0_quat": 4,
        }.items():
            _numeric_sequence(
                errors,
                f"{case_path}.{key}",
                case.get(key),
                length,
            )


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    profile: str = LEWM_ALIGNED_PROFILE,
    raise_on_error: bool = True,
) -> list[str]:
    """Strictly validate a frozen evaluation manifest.

    This is intentionally opt-in: :func:`load_manifest` retains its legacy,
    format-only behavior. Set ``raise_on_error=False`` to collect every
    structural error instead of raising one aggregated ``ValueError``.
    """
    errors: list[str] = []
    if profile != LEWM_ALIGNED_PROFILE:
        errors.append(
            f"profile: unsupported strict profile {profile!r}; expected "
            f"{LEWM_ALIGNED_PROFILE!r}"
        )
    if not isinstance(manifest, Mapping):
        errors.append("manifest: expected an object")
        if raise_on_error:
            raise ValueError("invalid manifest:\n- " + "\n- ".join(errors))
        return errors

    _expect_equal(
        errors,
        "schema_version",
        manifest.get("schema_version"),
        "v2",
    )
    _expect_equal(
        errors,
        "protocol",
        manifest.get("protocol"),
        LEWM_ALIGNED_PROFILE,
    )
    benchmark = manifest.get("benchmark")
    if benchmark not in _LEWM_BENCHMARKS:
        errors.append(
            "benchmark: expected one of "
            f"{sorted(_LEWM_BENCHMARKS)!r}, got {benchmark!r}"
        )
        benchmark = None

    for key, expected in {
        "selection_seed": 42,
        "case_seed_start": 42,
        "num_eval": 50,
        "goal_offset_steps": 25,
        "eval_budget": 50,
    }.items():
        _expect_equal(errors, key, manifest.get(key), expected)

    source_hdf5 = manifest.get("source_hdf5")
    if not isinstance(source_hdf5, str) or not source_hdf5:
        errors.append("source_hdf5: expected a non-empty string")
        source_hdf5 = None

    if benchmark is not None:
        _validate_environment(errors, manifest, benchmark)
        _validate_semantics(errors, manifest, benchmark)

    cases = manifest.get("cases")
    if not isinstance(cases, list):
        errors.append("cases: expected a list")
        if raise_on_error and errors:
            raise ValueError("invalid manifest:\n- " + "\n- ".join(errors))
        return errors
    if len(cases) != 50:
        errors.append(f"cases: expected exactly 50 cases, got {len(cases)}")

    seeds: list[int] = []
    case_ids: list[str] = []
    dataset_rows: list[int] = []
    for index, case in enumerate(cases):
        case_path = f"cases[{index}]"
        if not isinstance(case, Mapping):
            errors.append(f"{case_path}: expected an object")
            continue

        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            errors.append(f"{case_path}.case_id: expected a non-empty string")
        else:
            case_ids.append(case_id)

        seed = case.get("seed")
        if not _is_int(seed):
            errors.append(f"{case_path}.seed: expected an integer")
        else:
            seeds.append(seed)

        if benchmark is not None:
            _expect_equal(
                errors,
                f"{case_path}.benchmark",
                case.get("benchmark"),
                benchmark,
            )
        _expect_equal(
            errors,
            f"{case_path}.goal_offset_steps",
            case.get("goal_offset_steps"),
            25,
        )
        _expect_equal(
            errors,
            f"{case_path}.eval_budget",
            case.get("eval_budget"),
            50,
        )

        row = case.get("dataset_row")
        goal_row = case.get("goal_dataset_row")
        if not _is_int(row):
            errors.append(f"{case_path}.dataset_row: expected an integer")
            row = None
        else:
            dataset_rows.append(row)
        if not _is_int(goal_row):
            errors.append(f"{case_path}.goal_dataset_row: expected an integer")
            goal_row = None
        if row is not None and goal_row is not None and goal_row != row + 25:
            errors.append(
                f"{case_path}.goal_dataset_row: expected dataset_row + 25 "
                f"({row + 25}), got {goal_row}"
            )

        episode = case.get("episode_idx")
        if not _is_int(episode):
            errors.append(f"{case_path}.episode_idx: expected an integer")
        step = case.get("step_idx")
        goal_step = case.get("goal_step_idx")
        if not _is_int(step):
            errors.append(f"{case_path}.step_idx: expected an integer")
        if not _is_int(goal_step):
            errors.append(f"{case_path}.goal_step_idx: expected an integer")
        if _is_int(step) and _is_int(goal_step) and goal_step != step + 25:
            errors.append(
                f"{case_path}.goal_step_idx: expected step_idx + 25 "
                f"({step + 25}), got {goal_step}"
            )

        _validate_image_ref(
            errors,
            path=f"{case_path}.start_image",
            value=case.get("start_image"),
            expected_hdf5=source_hdf5,
            expected_row=row,
        )
        _validate_image_ref(
            errors,
            path=f"{case_path}.goal_image",
            value=case.get("goal_image"),
            expected_hdf5=source_hdf5,
            expected_row=goal_row,
        )
        if benchmark is not None:
            _validate_case_state(
                errors,
                case=case,
                case_path=case_path,
                benchmark=benchmark,
            )

    if len(seeds) == len(cases):
        expected_seeds = list(range(42, 92))
        if seeds != expected_seeds:
            errors.append(
                f"cases[*].seed: expected ordered seeds 42 through 91, got {seeds!r}"
            )
        if len(set(seeds)) != len(seeds):
            errors.append("cases[*].seed: seeds must be unique")
    if len(case_ids) == len(cases) and len(set(case_ids)) != len(case_ids):
        errors.append("cases[*].case_id: case IDs must be unique")
    if len(dataset_rows) == len(cases) and dataset_rows != sorted(dataset_rows):
        errors.append("cases[*].dataset_row: rows must be sorted increasingly")

    if errors and raise_on_error:
        raise ValueError("invalid manifest:\n- " + "\n- ".join(errors))
    return errors
