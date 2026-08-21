"""Focused tests for generic rigid geometry and planar push helpers."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from vdaworld.core.local_2d import (
    calibrate_component_geometry,
    convex_decompose_polygon,
    planar_push_step,
)
from vdaworld.core import dataset_tools
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox
from vdaworld.core.simulator import ActionConditionedSimulatorBase


def _t_outline() -> np.ndarray:
    return np.array(
        [
            [-6.0, -6.0],
            [6.0, -6.0],
            [6.0, -2.0],
            [2.0, -2.0],
            [2.0, 6.0],
            [-2.0, 6.0],
            [-2.0, -2.0],
            [-6.0, -2.0],
        ]
    )


def _polygon_area(polygon: np.ndarray) -> float:
    return 0.5 * abs(
        float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
    )


def test_multi_frame_component_calibration_measures_pusher_radius():
    masks = []
    for center, radius in (
        ((20, 22), 5),
        ((32, 30), 6),
        ((43, 39), 7),
    ):
        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, thickness=-1)
        masks.append(mask)

    result = calibrate_component_geometry(masks, simplify_px=0.25)
    via_base = ActionConditionedSimulatorBase.calibrate_component_geometry(
        masks,
        simplify_px=0.25,
    )
    areas = [int(mask.sum()) for mask in masks]
    expected_median_area = float(np.median(areas))

    assert result == via_base
    assert result["n_frames"] == 3
    assert result["n_valid_frames"] == 3
    assert result["median_area"] == expected_median_area
    assert result["median_area_equivalent_radius"] == pytest.approx(
        np.sqrt(expected_median_area / np.pi)
    )
    assert result["median_area_equivalent_radius"] == pytest.approx(6.0, abs=0.5)
    assert result["median_centroid"] == [32.0, 30.0]
    assert all(
        record["area_equivalent_radius"] > 0.0
        and record["bbox_dimensions"][0] == record["bbox_dimensions"][1]
        for record in result["frames"]
    )
    assert len(result["canonical_template"]) == 1
    assert len(result["canonical_template"][0]) >= 3
    json.dumps(result)


def test_codegen_rigid_geometry_tool_samples_local_colour_masks(monkeypatch):
    images = {}
    for sample, center in (
        ((0, 0), (18, 20)),
        ((0, 1), (30, 31)),
        ((1, 0), (44, 40)),
    ):
        image = np.full((64, 64, 3), 255, dtype=np.uint8)
        cv2.circle(image, center, 3, (60, 100, 230), thickness=-1)
        images[sample] = image
    monkeypatch.setattr(
        dataset_tools,
        "view_image",
        lambda _dataset, trajectory, frame: images[(trajectory, frame)],
    )
    sandbox = object.__new__(PlanningCriticSandbox)
    sandbox._dataset_dir = None
    logged = {}
    sandbox._log_critic_tool_call = lambda **kwargs: logged.update(kwargs)

    summary = sandbox.calibrate_rigid_geometry(
        [50, 90, 220],
        [70, 110, 240],
        samples=[[0, 0], [0, 1], [1, 0]],
        min_area=5,
    )

    assert "[calibrate_rigid_geometry]" in summary
    payload = json.loads(summary.split("\n", 1)[1])
    assert payload["passed"] is True
    assert payload["n_valid_frames"] == 3
    assert payload["median_area_equivalent_radius"] == pytest.approx(3.0, abs=0.5)
    assert logged["tool_name"] == "calibrate_rigid_geometry"
    assert "rigid_geometry_calibration.json" in logged["extra_files"]


def test_concave_t_decomposition_is_deterministic_and_preserves_area():
    outline = _t_outline()

    first = convex_decompose_polygon(outline)
    second = ActionConditionedSimulatorBase.convex_decompose_polygon(outline)

    assert len(first) > 1
    assert len(first) == len(second)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    assert sum(_polygon_area(piece) for piece in first) == pytest.approx(
        _polygon_area(outline)
    )
    assert all(len(convex_decompose_polygon(piece)) == 1 for piece in first)
    original_vertices = {tuple(vertex) for vertex in outline}
    assert all(
        tuple(vertex) in original_vertices
        for piece in first
        for vertex in piece
    )


def test_concave_t_contact_moves_body_deterministically():
    kwargs = {
        "body_polygons": [_t_outline()],
        "body_position": np.array([20.0, 20.0]),
        "body_angle": 0.0,
        "pusher_position": np.array([10.0, 16.0]),
        "pusher_target": np.array([24.0, 16.0]),
        "pusher_radius": 2.0,
        "kp": 100.0,
        "kv": 20.0,
        "dt": 0.01,
        "substeps": 30,
    }

    first = planar_push_step(**kwargs)
    second = planar_push_step(**kwargs)

    assert set(first) == {
        "body_position",
        "body_angle",
        "body_velocity",
        "body_angular_velocity",
        "pusher_position",
        "pusher_velocity",
    }
    for key in first:
        assert np.allclose(first[key], second[key])
    assert first["body_position"][0] > kwargs["body_position"][0]


def test_planar_push_preserves_velocities_and_caller_coordinate_direction():
    left = np.array([[-3.0, -2.0], [0.0, -2.0], [0.0, 2.0], [-3.0, 2.0]])
    right = np.array([[0.0, -2.0], [3.0, -2.0], [3.0, 2.0], [0.0, 2.0]])
    body_velocity = np.array([1.5, -0.5])
    pusher_velocity = np.array([0.3, 0.4])
    result = ActionConditionedSimulatorBase.planar_push_step(
        body_polygons=[left, right],
        body_position=np.array([20.0, 20.0]),
        body_angle=0.2,
        body_velocity=body_velocity,
        body_angular_velocity=0.25,
        pusher_position=np.array([2.0, 2.0]),
        pusher_target=np.array([2.0, 2.0]),
        pusher_velocity=pusher_velocity,
        pusher_radius=1.0,
        damping=1.0,
        kp=0.0,
        kv=0.0,
        dt=0.01,
        substeps=5,
    )

    assert np.allclose(result["body_velocity"], body_velocity)
    assert result["body_angular_velocity"] == pytest.approx(0.25)
    assert np.allclose(result["pusher_velocity"], pusher_velocity)
    assert result["body_position"][1] < 20.0
    assert result["pusher_position"][1] > 2.0


@pytest.mark.parametrize("damping", [-0.01, 1.01, np.nan])
def test_planar_push_rejects_invalid_damping(damping):
    square = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])

    with pytest.raises(ValueError, match=r"damping.*\[0, 1\]"):
        planar_push_step(
            body_polygons=[square],
            body_position=np.array([10.0, 10.0]),
            body_angle=0.0,
            pusher_position=np.array([2.0, 2.0]),
            pusher_target=np.array([2.0, 2.0]),
            pusher_radius=1.0,
            damping=damping,
        )
