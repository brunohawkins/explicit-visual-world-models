"""Synthetic-HDF regression tests for canonical LeWM manifest builders."""

from __future__ import annotations

import sys
from pathlib import Path

import hdf5plugin  # noqa: F401  # Register filters before importing h5py.
import h5py
import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_pusht_lewm_future_manifest as pusht_builder  # noqa: E402
import build_reacher_lewm_eval_manifest as reacher_builder  # noqa: E402
import build_cube_lewm_eval_manifest as cube_builder  # noqa: E402
import build_lewm_tworoom_swm_manifest as tworoom_builder  # noqa: E402


def _episode_columns(
    lengths: list[int],
    episode_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    episodes = np.concatenate(
        [
            np.full(length, episode, dtype=np.int64)
            for length, episode in zip(lengths, episode_ids)
        ]
    )
    steps = np.concatenate([np.arange(length, dtype=np.int64) for length in lengths])
    return episodes, steps


def _canonical_eval_rows(
    episodes: np.ndarray,
    steps: np.ndarray,
    *,
    goal_offset: int,
    num_eval: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    episode_ids = np.unique(episodes)
    episode_lengths = np.array(
        [np.max(steps[episodes == episode]) + 1 for episode in episode_ids]
    )
    max_start = episode_lengths - goal_offset - 1
    max_start_by_episode = dict(zip(episode_ids, max_start))
    max_start_per_row = np.array(
        [max_start_by_episode[episode] for episode in episodes]
    )
    valid = np.nonzero(steps <= max_start_per_row)[0]
    offsets = np.random.default_rng(seed).choice(
        len(valid) - 1,
        size=num_eval,
        replace=False,
    )
    return np.sort(valid[offsets]), valid


def _create_pixels(h5: h5py.File, rows: int) -> None:
    h5.create_dataset(
        "pixels",
        shape=(rows, 224, 224, 3),
        dtype=np.uint8,
    )


def test_tworoom_builder_samples_only_strict_opposite_room_interiors(
    tmp_path: Path,
) -> None:
    h5_path = tmp_path / "tworoom.h5"
    rows = 14
    x = np.asarray(
        [50.0, 50.0, 130.0, 130.0, 50.0, 50.0, 130.0, 130.0, 50.0, 50.0, 130.0, 130.0, 50.0, 50.0],
        dtype=np.float32,
    )
    proprio = np.column_stack([x, np.full(rows, 80.0, dtype=np.float32)])
    with h5py.File(h5_path, "w") as h5:
        h5["ep_idx"] = np.zeros(rows, dtype=np.int64)
        h5["step_idx"] = np.arange(rows, dtype=np.int64)
        h5["proprio"] = proprio

    selected, _, _ = tworoom_builder.sample_lewm_eval_rows(
        h5_path,
        seed=42,
        num_eval=6,
        goal_offset_steps=2,
    )

    starts = proprio[selected]
    goals = proprio[selected + 2]
    assert np.all(
        tworoom_builder._opposite_room_interior_mask(starts, goals)
    )
    assert np.all(
        ((starts[:, 0] <= 100.0) & (goals[:, 0] >= 124.0))
        | ((starts[:, 0] >= 124.0) & (goals[:, 0] <= 100.0))
    )


def test_cube_builder_samples_only_displacements_of_at_least_eight_cm() -> None:
    rows = 14
    episode_idx = np.zeros(rows, dtype=np.int64)
    step_idx = np.arange(rows, dtype=np.int64)
    block_pos = np.zeros((rows, 3), dtype=np.float64)
    block_pos[:, 0] = np.asarray(
        [0.0, 0.0, 0.10, 0.10, 0.20, 0.20, 0.30, 0.30, 0.40, 0.40, 0.50, 0.50, 0.60, 0.60]
    )

    selected = cube_builder.sample_eval_rows(
        episode_idx=episode_idx,
        step_idx=step_idx,
        num_eval=6,
        goal_offset=2,
        seed=42,
        block_pos=block_pos,
        min_cube_displacement=0.08,
    )

    displacement = np.linalg.norm(
        block_pos[selected + 2] - block_pos[selected],
        axis=1,
    )
    assert np.all(displacement >= 0.08)


def test_pusht_builder_matches_eval_py_sampling(tmp_path: Path) -> None:
    episodes, steps = _episode_columns(
        lengths=[8, 9, 10],
        episode_ids=[4, 9, 15],
    )
    h5_path = tmp_path / "pusht.h5"
    states = np.arange(len(steps) * 7, dtype=np.float32).reshape(-1, 7)
    with h5py.File(h5_path, "w") as h5:
        h5["episode_idx"] = episodes
        h5["step_idx"] = steps
        h5["state"] = states
        _create_pixels(h5, len(steps))

    expected_rows, valid_rows = _canonical_eval_rows(
        episodes,
        steps,
        goal_offset=2,
        num_eval=5,
        seed=42,
    )
    manifest = pusht_builder.build_manifest(
        h5_path=h5_path,
        num_eval=5,
        goal_offset=2,
        eval_budget=7,
        sample_seed=42,
        case_seed_start=100,
    )

    cases = manifest["cases"]
    assert [case["dataset_row"] for case in cases] == expected_rows.tolist()
    assert valid_rows[-1] not in expected_rows
    assert [case["seed"] for case in cases] == list(range(100, 105))
    for case in cases:
        row = case["dataset_row"]
        goal_row = case["goal_dataset_row"]
        assert episodes[goal_row] == episodes[row]
        assert steps[goal_row] == steps[row] + 2
        assert case["start"]["state"] == states[row].astype(float).tolist()
        assert case["goal"]["state"] == states[goal_row].astype(float).tolist()
        assert (
            case["start"]["pos_agent"]
            + case["start"]["block_pose"]
            + case["start"]["vel_agent"]
            == case["start"]["state"]
        )
        assert "goal_pose" not in case["goal"]
        assert case["start_image"]["row"] == row
        assert case["goal_image"]["row"] == goal_row


def test_reacher_builder_matches_eval_py_sampling(tmp_path: Path) -> None:
    episodes, steps = _episode_columns(
        lengths=[9, 10, 11],
        episode_ids=[3, 8, 20],
    )
    h5_path = tmp_path / "reacher.h5"
    qpos = np.arange(len(steps) * 2, dtype=np.float64).reshape(-1, 2)
    qvel = -qpos
    target_pos = np.stack(
        [np.linspace(0.0, 1.0, len(steps))] * 2,
        axis=1,
    )
    with h5py.File(h5_path, "w") as h5:
        h5["ep_idx"] = episodes
        h5["step_idx"] = steps
        h5["qpos"] = qpos
        h5["qvel"] = qvel
        h5["target_pos"] = target_pos
        _create_pixels(h5, len(steps))

    expected_rows, valid_rows = _canonical_eval_rows(
        episodes,
        steps,
        goal_offset=3,
        num_eval=6,
        seed=7,
    )
    manifest = reacher_builder.build_manifest(
        h5_path=h5_path,
        num_eval=6,
        goal_offset=3,
        eval_budget=9,
        selection_seed=7,
        case_seed_start=30,
    )

    cases = manifest["cases"]
    assert [case["dataset_row"] for case in cases] == expected_rows.tolist()
    assert valid_rows[-1] not in expected_rows
    for case in cases:
        row = case["dataset_row"]
        goal_row = case["goal_dataset_row"]
        assert episodes[goal_row] == episodes[row]
        assert steps[goal_row] == steps[row] + 3
        assert case["qpos"] == qpos[row].tolist()
        assert case["qvel"] == qvel[row].tolist()
        assert case["goal_qpos"] == qpos[goal_row].tolist()
        assert case["goal_qvel"] == qvel[goal_row].tolist()
        assert case["start_image"] == {
            "type": "hdf5_dataset_row",
            "path": str(h5_path),
            "key": "pixels",
            "row": row,
        }
        assert case["goal_image"]["row"] == goal_row
