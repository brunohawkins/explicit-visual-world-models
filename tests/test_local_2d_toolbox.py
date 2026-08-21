"""Tests for deterministic helpers exposed to generated 2-D simulators."""

from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import pytest
from PIL import Image

from vdaworld.core import dataset_tools
from vdaworld.core.local_2d import (
    calibrate_articulated_chain_geometry,
    component_geometry,
    damped_dynamics_step,
    estimate_observed_velocity,
    fit_articulated_chain,
    foreground_from_border,
    linear_dynamics_step,
    mask_from_appearance,
    planar_push_step,
    select_connected_component,
    template_from_mask,
)
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox
from vdaworld.core.simulator import ActionConditionedSimulatorBase


def _lab_model(rgb: tuple[int, int, int]) -> dict:
    centre = cv2.cvtColor(
        np.asarray(rgb, dtype=np.uint8).reshape(1, 1, 3),
        cv2.COLOR_RGB2LAB,
    ).reshape(3)
    return {
        "center_lab": centre.astype(float).tolist(),
        "scale_lab": [3.0, 3.0, 3.0],
        "max_distance": 2.0,
    }


def test_appearance_mask_and_geometry_ignore_background_noise():
    image = np.full((64, 64, 3), 245, dtype=np.uint8)
    cv2.rectangle(image, (18, 20), (42, 38), (80, 150, 220), thickness=-1)
    image[2, 2] = np.array([80, 150, 220], dtype=np.uint8)

    mask = mask_from_appearance(
        image,
        _lab_model((80, 150, 220)),
        min_area=20,
        close_radius=1,
        largest=True,
    )
    geometry = component_geometry(mask)

    assert not bool(mask[2, 2])
    assert geometry["area"] == 25 * 19
    assert geometry["centroid"] == [30.0, 29.0]
    assert len(geometry["contour"]) == 4


def test_component_selection_uses_reference_geometry_not_largest():
    mask = np.zeros((40, 50), dtype=bool)
    mask[3:13, 3:15] = True
    mask[25:29, 35:40] = True

    selected = select_connected_component(
        mask,
        min_area=10,
        reference_centroid=[37.0, 26.5],
        expected_area=20,
    )

    assert selected["valid"] is True
    assert selected["area"] == 20
    assert selected["centroid"] == [37.0, 26.5]
    assert selected["component_count"] == 2
    assert selected["candidate_count"] == 2
    assert 0.5 < selected["confidence"] <= 1.0
    assert np.array_equal(selected["mask"], mask & (np.indices(mask.shape)[0] >= 25))

    largest = ActionConditionedSimulatorBase.select_connected_component(mask)
    assert largest["area"] == 120
    assert largest["centroid"] == [8.5, 7.5]


def test_component_selection_reports_invalid_and_validates_inputs():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    result = select_connected_component(mask, min_area=5)

    assert not result["mask"].any()
    assert {key: value for key, value in result.items() if key != "mask"} == {
        "valid": False,
        "confidence": 0.0,
        "area": 0,
        "centroid": None,
        "component_count": 1,
        "candidate_count": 0,
    }
    with pytest.raises(ValueError, match="expected_area"):
        select_connected_component(mask, expected_area=0)
    with pytest.raises(ValueError, match="reference_centroid"):
        select_connected_component(mask, reference_centroid=[np.nan, 1.0])
    with pytest.raises(ValueError, match="connectivity"):
        select_connected_component(mask, connectivity=6)


def test_foreground_from_border_and_template_roundtrip():
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    mask_true = np.zeros((48, 48), dtype=np.uint8)
    polygon = np.array(
        [[12, 10], [35, 10], [35, 17], [26, 17], [26, 36], [20, 36], [20, 17], [12, 17]],
        dtype=np.int32,
    )
    cv2.fillPoly(mask_true, [polygon], 1)
    image[mask_true.astype(bool)] = np.array([130, 155, 180], dtype=np.uint8)

    mask = foreground_from_border(
        image,
        lab_distance=10.0,
        min_area=20,
        largest=True,
    )
    template = template_from_mask(mask, simplify_px=0.5)
    sim = ActionConditionedSimulatorBase(frame_size=(48, 48))
    pose = sim.fit_pose_to_mask(mask, template, n_angles=90)

    assert pose["dice"] > 0.95
    assert pose["iou"] > 0.90


def test_identified_dynamics_helpers():
    next_state = linear_dynamics_step(
        np.array([1.0, 2.0]),
        np.array([0.5, -1.0]),
        np.array([[2.0, 0.0], [0.0, 0.5]]),
        max_delta=np.array([2.0, 2.0]),
    )
    assert np.allclose(next_state, [2.0, 1.5])

    result = damped_dynamics_step(
        np.array([0.0]),
        np.array([1.0]),
        np.array([0.0]),
        np.array([[1.0]]),
        damping=0.5,
        dt=0.2,
    )
    assert np.allclose(result["velocity"], [0.9])
    assert np.allclose(result["position"], [0.18])

    bounded = damped_dynamics_step(
        np.array([3.1, 2.78]),
        np.array([1.0, 1.0]),
        np.zeros(2),
        np.zeros((2, 2)),
        damping=0.0,
        dt=0.1,
        position_low=np.array([-np.inf, -2.7925]),
        position_high=np.array([np.inf, 2.7925]),
        periodic=[True, False],
    )
    assert bounded["position"][0] == pytest.approx(
        np.arctan2(np.sin(3.2), np.cos(3.2))
    )
    assert bounded["position"][1] == pytest.approx(2.7925)
    assert bounded["velocity"][0] == pytest.approx(1.0)
    assert bounded["velocity"][1] == pytest.approx(0.0)


def test_observed_velocity_wraps_only_declared_periodic_dimensions():
    previous = np.array([np.pi - 0.1, 2.7])
    current = np.array([-np.pi + 0.1, -2.7])

    velocity = estimate_observed_velocity(
        current,
        previous,
        elapsed_seconds=0.5,
        periodic=[True, False],
    )
    wrapped_by_simulator = ActionConditionedSimulatorBase.estimate_observed_velocity(
        current,
        previous,
        elapsed_seconds=0.5,
        periodic=[True, False],
    )

    assert velocity == pytest.approx([0.4, -10.8])
    assert np.array_equal(velocity, wrapped_by_simulator)
    with pytest.raises(ValueError, match="same non-empty shape"):
        estimate_observed_velocity(current, previous[:1], 0.5, [True, False])
    with pytest.raises(ValueError, match="elapsed_seconds"):
        estimate_observed_velocity(current, previous, 0.0, [True, False])
    with pytest.raises(ValueError, match="periodic"):
        estimate_observed_velocity(current, previous, 0.5, [True, 2])


def test_observed_velocity_accepts_scalar_periodic_and_nonperiodic_values():
    periodic = estimate_observed_velocity(
        np.float64(-np.pi + 0.1),
        np.float64(np.pi - 0.1),
        elapsed_seconds=0.5,
        periodic=True,
    )
    nonperiodic = estimate_observed_velocity(
        np.float64(-2.7),
        np.float64(2.7),
        elapsed_seconds=0.5,
        periodic=False,
    )
    inherited = ActionConditionedSimulatorBase.estimate_observed_velocity(
        np.float64(-np.pi + 0.1),
        np.float64(np.pi - 0.1),
        elapsed_seconds=0.5,
        periodic=True,
    )

    assert np.ndim(periodic) == 0
    assert periodic == pytest.approx(0.4)
    assert nonperiodic == pytest.approx(-10.8)
    assert inherited == pytest.approx(periodic)


def test_articulated_chain_fitter_recovers_end_effector():
    mask = np.zeros((96, 96), dtype=np.uint8)
    base = np.array([48.0, 48.0])
    lengths = np.array([20.0, 18.0])
    true_angles = np.array([0.55, -1.05])
    elbow = base + lengths[0] * np.array(
        [np.cos(true_angles[0]), np.sin(true_angles[0])]
    )
    tip = elbow + lengths[1] * np.array(
        [
            np.cos(true_angles.sum()),
            np.sin(true_angles.sum()),
        ]
    )
    cv2.line(
        mask,
        tuple(np.rint(base).astype(int)),
        tuple(np.rint(elbow).astype(int)),
        1,
        thickness=5,
    )
    cv2.line(
        mask,
        tuple(np.rint(elbow).astype(int)),
        tuple(np.rint(tip).astype(int)),
        1,
        thickness=5,
    )

    result = fit_articulated_chain(
        mask,
        base,
        lengths,
        n_starts=96,
    )

    assert np.linalg.norm(result["end_effector"] - tip) < 2.0
    assert result["chamfer_rmse"] < 3.0


def test_articulated_chain_search_covers_folded_branches():
    base = np.array([48.0, 48.0])
    lengths = np.array([22.0, 19.0])
    for true_angles in (
        np.array([0.3, 2.2]),
        np.array([-1.1, -2.3]),
        np.array([2.4, 1.7]),
    ):
        headings = np.cumsum(true_angles)
        elbow = base + lengths[0] * np.array(
            [np.cos(headings[0]), np.sin(headings[0])]
        )
        tip = elbow + lengths[1] * np.array(
            [np.cos(headings[1]), np.sin(headings[1])]
        )
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.line(
            mask,
            tuple(np.rint(base).astype(int)),
            tuple(np.rint(elbow).astype(int)),
            1,
            thickness=5,
        )
        cv2.line(
            mask,
            tuple(np.rint(elbow).astype(int)),
            tuple(np.rint(tip).astype(int)),
            1,
            thickness=5,
        )

        result = fit_articulated_chain(mask, base, lengths, n_starts=128)

        assert np.linalg.norm(result["end_effector"] - tip) < 2.0
        assert result["candidates"]
        assert "ambiguous" in result
        assert "ambiguity_margin" in result


def test_multi_frame_chain_geometry_calibration_recovers_fixed_dimensions():
    true_base = np.array([43.0, 51.0])
    true_lengths = np.array([22.0, 17.0])
    masks = []
    for angles in (
        np.array([-1.2, 0.7]),
        np.array([-0.2, 1.8]),
        np.array([0.8, -1.5]),
        np.array([1.9, 1.1]),
    ):
        headings = np.cumsum(angles)
        elbow = true_base + true_lengths[0] * np.array(
            [np.cos(headings[0]), np.sin(headings[0])]
        )
        tip = elbow + true_lengths[1] * np.array(
            [np.cos(headings[1]), np.sin(headings[1])]
        )
        mask = np.zeros((96, 96), dtype=np.uint8)
        cv2.line(
            mask,
            tuple(np.rint(true_base).astype(int)),
            tuple(np.rint(elbow).astype(int)),
            1,
            thickness=5,
        )
        cv2.line(
            mask,
            tuple(np.rint(elbow).astype(int)),
            tuple(np.rint(tip).astype(int)),
            1,
            thickness=5,
        )
        masks.append(mask)

    result = calibrate_articulated_chain_geometry(
        masks,
        base_xy=np.array([47.0, 47.0]),
        link_lengths=np.array([20.0, 20.0]),
        n_starts=64,
    )

    assert np.linalg.norm(result["base_xy"] - true_base) < 3.0
    assert np.max(np.abs(result["link_lengths"] - true_lengths)) < 3.0
    assert result["worst_chamfer_rmse"] < 3.0


def test_planar_push_step_is_local_and_deterministic():
    square = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])
    kwargs = {
        "body_polygons": [square],
        "body_position": np.array([20.0, 20.0]),
        "body_angle": 0.0,
        "pusher_position": np.array([14.0, 20.0]),
        "pusher_target": np.array([22.0, 20.0]),
        "pusher_radius": 2.0,
        "kp": 100.0,
        "kv": 20.0,
        "dt": 0.01,
        "substeps": 20,
    }
    first = planar_push_step(**kwargs)
    second = ActionConditionedSimulatorBase.planar_push_step(**kwargs)

    for key in (
        "body_position",
        "body_velocity",
        "pusher_position",
        "pusher_velocity",
    ):
        assert np.allclose(first[key], second[key])
    assert first["body_position"][0] > 20.0
    assert first["pusher_position"][0] > 14.0


def test_derive_visual_model_emits_runtime_compatible_literals(monkeypatch):
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    reference = np.zeros((48, 48), dtype=np.uint8)
    polygon = np.array(
        [[10, 10], [36, 10], [36, 17], [26, 17], [26, 38], [20, 38], [20, 17], [10, 17]],
        dtype=np.int32,
    )
    cv2.fillPoly(reference, [polygon], 255)
    image[reference > 0] = np.array([120, 150, 180], dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(reference).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    monkeypatch.setattr(dataset_tools, "view_image", lambda *_: image)
    sandbox = object.__new__(PlanningCriticSandbox)
    sandbox._dataset_dir = None
    sandbox._call_gemini_segmentation = lambda *_: [
        {
            "label": "rigid object",
            "mask": f"data:image/png;base64,{encoded}",
        }
    ]
    logged = {}
    sandbox._log_critic_tool_call = lambda **kwargs: logged.update(kwargs)

    result = sandbox.derive_visual_model(0, 0, "the rigid object")

    assert "[derive_visual_model]" in result
    assert '"local_reproduction_iou": 1.0' in result
    assert '"appearance_model"' in result
    assert '"template_polygons"' in result
    assert logged["tool_name"] == "derive_visual_model"
    assert "appearance_model.json" in logged["extra_files"]


def test_validate_visual_reconstruction_uses_diverse_advisory_probes():
    sandbox = object.__new__(PlanningCriticSandbox)
    sandbox._gate_trajectories = [0, 1, 2]
    sandbox._gate_fit_quality_max_rmse = 0.3
    logged = {}
    sandbox._log_critic_tool_call = lambda **kwargs: logged.update(kwargs)
    calls = {}

    def fake_check(trajectories, *, max_rmse, max_probes):
        calls.update(
            trajectories=trajectories,
            max_rmse=max_rmse,
            max_probes=max_probes,
        )
        return {
            "passed": False,
            "n_probes": max_probes,
            "worst_normalized_rmse": 0.4,
        }

    sandbox._run_fit_quality_check = fake_check

    result = sandbox.validate_visual_reconstruction([0, 2], max_probes=6)

    assert "[validate_visual_reconstruction]" in result
    assert calls == {
        "trajectories": [0, 2],
        "max_rmse": 0.3,
        "max_probes": 6,
    }
    assert logged["tool_name"] == "validate_visual_reconstruction"
    assert "visual_reconstruction.json" in logged["extra_files"]


def test_keep_best_cem_reranking_is_bounded_and_restores_file(tmp_path):
    simulator_path = tmp_path / "simulator.py"
    simulator_path.write_text("FINAL", encoding="utf-8")
    sandbox = object.__new__(PlanningCriticSandbox)
    sandbox._keep_best_include_cem = True
    sandbox._gate_require_cem_suite = True
    sandbox._gate_trajectories = [0]
    sandbox._keep_best_candidates = [
        {
            "tool_call": 1,
            "code": "RATIO_BEST_PLANNER_BAD",
            "score": {
                "worst": 0.1,
                "mean": 0.1,
                "n_failed": 0,
                "quality_failures": 0,
                "cem_failures": 0,
            },
        },
        {
            "tool_call": 2,
            "code": "RATIO_SECOND_PLANNER_GOOD",
            "score": {
                "worst": 0.2,
                "mean": 0.2,
                "n_failed": 0,
                "quality_failures": 0,
                "cem_failures": 0,
            },
        },
    ]
    sandbox._sandbox_path = str(simulator_path)
    sandbox._tool_calls_log_dir = str(tmp_path)
    sandbox._suppress_tool_call_log = False
    sandbox._keep_best_best = sandbox._keep_best_candidates[0]
    sandbox._keep_best_cem_record = None
    calls = []

    def fake_cem(*_args, **_kwargs):
        code = simulator_path.read_text(encoding="utf-8")
        calls.append(code)
        passed = "PLANNER_GOOD" in code
        return {
            "passed": passed,
            "pairs": [1, 2, 3],
            "results": [{"passed": passed} for _ in range(3)],
        }

    sandbox._run_cem_suite_check = fake_cem
    sandbox._rerank_keep_best_with_cem(max_candidates=3)

    assert calls == ["RATIO_BEST_PLANNER_BAD", "RATIO_SECOND_PLANNER_GOOD"]
    assert sandbox._keep_best_best["tool_call"] == 2
    assert simulator_path.read_text(encoding="utf-8") == "FINAL"
    assert sandbox._keep_best_cem_record["cem_candidates_evaluated"] == 2
