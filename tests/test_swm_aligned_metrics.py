"""Regression tests for post-step SWM reference metrics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from vdaworld.core.local_2d import (
    LOCAL_2D_TOOLBOX_COMPATIBLE_VERSIONS,
    LOCAL_2D_TOOLBOX_VERSION,
)


def _load_smoke_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "smoke_p2_mpc.py"
    spec = importlib.util.spec_from_file_location("smoke_p2_aligned_metrics", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backend_info(step: int) -> dict:
    return {
        "env_step": step,
        "backend": {"implementation": "stable_worldmodel"},
    }


def test_p2_provenance_hashes_are_content_addressed(tmp_path):
    smoke = _load_smoke_module()
    artifact = tmp_path / "simulator.py"
    artifact.write_bytes(b"generated simulator\n")

    assert (
        smoke._file_sha256(artifact)
        == "91cad59b83cef27ef8baa0604407fe41f6f6a7c24977c468bc3b1ed49d145ea6"
    )
    array = np.arange(6, dtype=np.uint8).reshape(2, 3)
    assert smoke._array_sha256(array) == smoke._array_sha256(array.copy())
    assert smoke._array_sha256(array) != smoke._array_sha256(array.astype(np.int64))
    assert smoke._array_sha256(None) is None


def test_toolbox_v4_declares_v3_runtime_compatibility():
    assert LOCAL_2D_TOOLBOX_VERSION == "4"
    assert LOCAL_2D_TOOLBOX_COMPATIBLE_VERSIONS == {"3", "4"}


def test_cem_kwargs_plumb_generic_settings_and_resolve_planner_seed():
    smoke = _load_smoke_module()
    args = SimpleNamespace(
        cem_population=12,
        cem_iters=3,
        elite_frac=0.25,
        init_std_scale=0.8,
        cem_use_stage_cost=True,
        cem_action_smoothness_weight=1.5,
        cem_action_knots=4,
        planner_seed=None,
    )

    resolved = smoke._cem_kwargs_for_seed(args, seed=42)
    assert resolved == {
        "population": 12,
        "iters": 3,
        "elite_frac": 0.25,
        "init_std_scale": 0.8,
        "use_stage_cost": True,
        "action_smoothness_weight": 1.5,
        "num_action_knots": 4,
        "seed": 42,
    }

    args.planner_seed = 99
    assert smoke._cem_kwargs_for_seed(args, seed=42)["seed"] == 99


def test_p2_diagnostic_flags_and_primary_success_metric_plumbing():
    smoke = _load_smoke_module()
    args = SimpleNamespace(
        propagate_prev_state=False,
        stop_on_primary_success=True,
    )

    assert smoke._p2_diagnostic_kwargs(args) == {
        "propagate_prev_state": False,
        "stop_on_primary_success": True,
    }
    assert smoke._native_success(
        {"done": False, "primary_success_observed": True}
    )
    assert not smoke._native_success(
        {"done": False, "primary_success_observed": False}
    )


def test_summary_records_resolved_planner_seed_metadata():
    smoke = _load_smoke_module()
    args = SimpleNamespace(benchmark="generic", seed_values=[42, 43], planner_seed=None)
    metrics = [
        {
            "done": False,
            "is_success": False,
            "terminal_distance": 1.0,
            "planner_seed": 42,
            "cem_use_stage_cost": True,
            "cem_action_smoothness_weight": 0.5,
            "cem_num_action_knots": 3,
            "propagate_prev_state": False,
            "stop_on_primary_success": True,
        },
        {
            "done": False,
            "is_success": False,
            "terminal_distance": 2.0,
            "planner_seed": 43,
        },
    ]

    summary = smoke._summary(
        metrics,
        args=args,
        sim_path=Path("simulator.py"),
        out_dir=Path("out"),
    )

    assert summary["planner_seed"] is None
    assert summary["planner_seed_source"] == "environment_seed"
    assert summary["planner_seeds"] == [42, 43]
    assert summary["cem_use_stage_cost"] is True
    assert summary["cem_action_smoothness_weight"] == 0.5
    assert summary["cem_num_action_knots"] == 3
    assert summary["propagate_prev_state"] is False
    assert summary["stop_on_primary_success"] is True


def test_push_full_state_uses_manifest_future_agent_and_skips_reset():
    smoke = _load_smoke_module()
    reset = {
        **_backend_info(0),
        "pos_agent": [9.0, 9.0],
        "block_pose": [10.0, 10.0, 0.0],
        "goal_pose": [10.0, 10.0, 0.0],
        "goal_state": {"pos_agent": [20.0, 20.0]},
        "goal_render_info": {"pos_agent": [9.0, 9.0]},
    }
    after_step = {
        **_backend_info(1),
        "pos_agent": [20.0, 20.0],
        "block_pose": [10.0, 10.0, 0.0],
        "goal_pose": [10.0, 10.0, 0.0],
        "goal_state": {"pos_agent": [20.0, 20.0]},
        "goal_render_info": {"pos_agent": [9.0, 9.0]},
    }

    metrics = smoke._pusht_pose_metrics({"gt_infos": [reset, after_step]})
    assert metrics["pusht_paper_full_state_ever_success"] is True
    assert metrics["pusht_paper_full_state_final_success"] is True
    assert metrics["pusht_pose_max_coverage"] == 1.0
    assert metrics["pusht_pose_ever_ge_0_95"] is True


def test_pusht_summary_uses_coverage_as_headline_success():
    smoke = _load_smoke_module()
    args = SimpleNamespace(benchmark="pusht", seed_values=[42, 43])
    metrics = [
        {
            "done": False,
            "is_success": True,
            "native_is_success": False,
            "terminal_distance": 1.0,
            "pusht_final_coverage": 0.96,
            "pusht_max_coverage": 0.96,
            "pusht_success_at_0_95": True,
            "pusht_pose_max_coverage": 0.96,
            "pusht_pose_final_coverage": 0.96,
            "pusht_pose_ever_ge_0_95": True,
            "pusht_paper_block_final_success": True,
            "pusht_paper_block_ever_success": True,
            "pusht_paper_full_state_final_success": False,
            "pusht_paper_full_state_ever_success": False,
        },
        {
            "done": False,
            "is_success": False,
            "native_is_success": False,
            "terminal_distance": 2.0,
            "pusht_final_coverage": 0.5,
            "pusht_max_coverage": 0.7,
            "pusht_success_at_0_95": False,
            "pusht_pose_max_coverage": 0.7,
            "pusht_pose_final_coverage": 0.5,
            "pusht_pose_ever_ge_0_95": False,
            "pusht_paper_block_final_success": False,
            "pusht_paper_block_ever_success": False,
            "pusht_paper_full_state_final_success": False,
            "pusht_paper_full_state_ever_success": False,
        },
    ]

    summary = smoke._summary(
        metrics,
        args=args,
        sim_path=Path("simulator.py"),
        out_dir=Path("out"),
    )

    assert summary["n_success"] == 1
    assert summary["success_metric"] == "pusht_success_at_0_95"
    assert summary["task_primary_n_success"] == 1
    assert summary["paper_primary_n_success"] == 0


def test_reacher_qpos_primary_uses_every_post_step():
    smoke = _load_smoke_module()
    reset = {
        **_backend_info(0),
        "qpos": [0.0, 0.0],
        "goal_qpos": [0.0, 0.0],
    }
    miss = {
        **_backend_info(1),
        "qpos": [0.2, 0.0],
        "goal_qpos": [0.0, 0.0],
    }
    hit = {
        **_backend_info(2),
        "qpos": [0.049, -0.049],
        "goal_qpos": [0.0, 0.0],
    }

    metrics = smoke._reacher_qpos_metrics({"gt_infos": [reset, miss, hit]})
    assert metrics["reacher_paper_qpos_ever_success"] is True
    assert metrics["reacher_paper_qpos_final_success"] is True
    assert metrics["reacher_paper_qpos_min_max_abs_error"] == 0.049


def test_reacher_target_success_uses_fingertip_distance_not_qpos():
    smoke = _load_smoke_module()
    miss = {
        **_backend_info(1),
        "distance_to_target": 0.05,
        "target_radius": 0.015,
    }
    hit = {
        **_backend_info(2),
        "distance_to_target": 0.014,
        "target_radius": 0.015,
    }

    metrics = smoke._reacher_distance_metrics({"gt_infos": [miss, hit]})
    assert metrics["reacher_target_ever_success"] is True
    assert metrics["reacher_target_final_success"] is True


def test_reacher_summary_uses_endpoint_as_headline_success():
    smoke = _load_smoke_module()
    args = SimpleNamespace(benchmark="reacher", seed_values=[42, 43])
    metrics = [
        {
            "done": False,
            "is_success": True,
            "native_is_success": False,
            "terminal_distance": 1.0,
            "reacher_min_distance_to_target": 0.01,
            "reacher_final_distance_to_target": 0.02,
            "reacher_target_radius": 0.015,
            "reacher_target_final_success": False,
            "reacher_target_ever_success": True,
            "reacher_paper_qpos_final_success": False,
            "reacher_paper_qpos_ever_success": False,
            "reacher_paper_qpos_min_max_abs_error": 0.2,
        },
        {
            "done": False,
            "is_success": False,
            "native_is_success": False,
            "terminal_distance": 2.0,
            "reacher_min_distance_to_target": 0.03,
            "reacher_final_distance_to_target": 0.04,
            "reacher_target_radius": 0.015,
            "reacher_target_final_success": False,
            "reacher_target_ever_success": False,
            "reacher_paper_qpos_final_success": True,
            "reacher_paper_qpos_ever_success": True,
            "reacher_paper_qpos_min_max_abs_error": 0.01,
        },
    ]

    summary = smoke._summary(
        metrics,
        args=args,
        sim_path=Path("simulator.py"),
        out_dir=Path("out"),
    )

    assert summary["n_success"] == 1
    assert summary["success_metric"] == "reacher_target_ever_success"
    assert summary["task_primary_n_success"] == 1
    assert summary["paper_primary_n_success"] == 1


def test_cube_reset_proximity_is_not_counted_as_success():
    smoke = _load_smoke_module()
    reset = {
        **_backend_info(0),
        "cube_pos": [0.0, 0.0, 0.0],
        "target_pos": [0.0, 0.0, 0.0],
        "primary_success": False,
    }
    after_step = {
        **_backend_info(1),
        "cube_pos": [0.05, 0.0, 0.0],
        "target_pos": [0.0, 0.0, 0.0],
        "primary_success": False,
    }

    metrics = smoke._cube_position_metrics({"gt_infos": [reset, after_step]})
    assert metrics["cube_paper_ever_success"] is False
    assert metrics["cube_paper_final_success"] is False
    assert metrics["cube_paper_min_distance"] == 0.05
