"""Tests for the benchmark registry descriptors."""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np

from vdaworld.core.dataset_tools import read_trajectory
from vdaworld.eval.benchmarks import (
    BENCHMARKS,
    SHARED_GATE_MAX_WORST_RATIO,
    available_benchmarks,
    get_benchmark,
)


def test_registry_contains_current_scope_only():
    assert available_benchmarks() == [
        "cube",
        "pusht",
        "reacher",
        "two_room",
        "two_room_lewm",
    ]
    assert set(BENCHMARKS) == {
        "cube",
        "pusht",
        "reacher",
        "two_room",
        "two_room_lewm",
    }


def test_specs_have_required_descriptor_fields():
    for name in available_benchmarks():
        spec = get_benchmark(name)
        assert spec.dataset_dir.is_dir()
        assert spec.frame_size[0] > 0 and spec.frame_size[1] > 0
        assert spec.n_frames > 0
        fit_a, fit_b = spec.sandbox_fit_paths()
        assert fit_a.is_file()
        assert fit_b.is_file()
        assert spec.held_out_traj >= 0
        dep_low, dep_high = spec.deployed_bounds_copy()
        p2_low, p2_high = spec.p2_bounds_copy()
        assert dep_low.shape == dep_high.shape
        assert p2_low.shape == p2_high.shape
        assert np.all(dep_low < dep_high)
        assert np.all(p2_low < p2_high)
        session = spec.p2_session_factory()
        try:
            assert hasattr(session, "start")
            assert hasattr(session, "step")
            assert hasattr(session, "close")
        finally:
            session.close()


def test_reacher_cube_and_lewm_two_room_specs_match_datasets():
    two_room_lewm = get_benchmark("two_room_lewm")
    assert two_room_lewm.frame_size == (224, 224)
    assert two_room_lewm.n_frames == 51
    assert two_room_lewm.sandbox_fit_frame_a == 0
    assert two_room_lewm.sandbox_fit_frame_b == 25
    assert two_room_lewm.held_out_traj == 9
    np.testing.assert_allclose(
        two_room_lewm.deployed_bounds_copy()[0], [-1.0, -1.0]
    )
    np.testing.assert_allclose(
        two_room_lewm.deployed_bounds_copy()[1], [1.0, 1.0]
    )

    pusht = get_benchmark("pusht")
    assert pusht.frame_size == (224, 224)
    assert pusht.dataset_dir.name == "expert_only"
    assert pusht.dataset_dir.parent.name == "pusht_no_green_target_224"
    np.testing.assert_allclose(pusht.deployed_bounds_copy()[0], [0.0, 0.0])
    np.testing.assert_allclose(pusht.deployed_bounds_copy()[1], [224.0, 224.0])
    np.testing.assert_allclose(pusht.p2_bounds_copy()[1], [224.0, 224.0])

    reacher = get_benchmark("reacher")
    assert reacher.frame_size == (224, 224)
    assert reacher.n_frames == 41
    assert reacher.sandbox_fit_frame_a == 0
    assert reacher.sandbox_fit_frame_b == 40
    assert reacher.held_out_traj == 9
    np.testing.assert_allclose(reacher.deployed_bounds_copy()[0], [-1.0, -1.0])
    np.testing.assert_allclose(reacher.deployed_bounds_copy()[1], [1.0, 1.0])

    cube = get_benchmark("cube")
    assert cube.frame_size == (200, 200)
    assert cube.n_frames == 11
    assert cube.sandbox_fit_frame_a == 0
    assert cube.sandbox_fit_frame_b == 10
    assert cube.held_out_traj == 9
    actions = []
    for i in range(10):
        tr = read_trajectory(cube.dataset_dir, i)
        assert len(tr["frames"]) >= cube.n_frames
        assert tr["actions"].shape == (len(tr["frames"]) - 1, 5)
        actions.append(tr["actions"][: cube.n_frames - 1])
    all_actions = np.concatenate(actions, axis=0)
    dep_low, dep_high = cube.deployed_bounds_copy()
    assert np.all(all_actions.min(axis=0) >= dep_low - 1e-6)
    assert np.all(all_actions.max(axis=0) <= dep_high + 1e-6)
    p2_low, p2_high = cube.p2_bounds_copy()
    np.testing.assert_allclose(p2_low, [-1.0, -1.0, -1.0, -1.0, -1.0])
    np.testing.assert_allclose(p2_high, [1.0, 1.0, 1.0, 1.0, 1.0])


def test_p1_dry_run_does_not_construct_p2_sessions(monkeypatch):
    import vdaworld.eval.benchmarks as benchmarks

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "smoke_p1.py"
    spec = importlib.util.spec_from_file_location("smoke_p1_for_test", script_path)
    assert spec is not None and spec.loader is not None
    smoke_p1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke_p1)

    def fail_factory():
        raise AssertionError("P1 dry-run constructed a P2 session")

    originals = dict(benchmarks.BENCHMARKS)
    for benchmark in available_benchmarks():
        patched = dict(benchmarks.BENCHMARKS)
        patched[benchmark] = dataclasses.replace(
            patched[benchmark],
            p2_session_factory=fail_factory,
        )
        monkeypatch.setattr(benchmarks, "BENCHMARKS", patched)
        monkeypatch.setattr(
            sys,
            "argv",
            ["smoke_p1.py", "--benchmark", benchmark, "--dry-run"],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            assert smoke_p1.main() == 0
        monkeypatch.setattr(benchmarks, "BENCHMARKS", originals)


def test_gate_worst_ratio_threshold_is_shared():
    assert SHARED_GATE_MAX_WORST_RATIO == 0.5
