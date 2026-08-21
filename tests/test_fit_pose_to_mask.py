import numpy as np
import pytest

from vdaworld.core.api import WorldAPI, _rasterize_template_polygons
from vdaworld.core.simulator import ActionConditionedSimulatorBase


def _centre_template(
    template: list[np.ndarray],
    hw: tuple[int, int] = (96, 128),
) -> list[np.ndarray]:
    """Shift polygons so rasterized pixel centroid matches the placement point."""
    h, w = hw
    cx, cy = w / 2.0, h / 2.0
    mask = _rasterize_template_polygons(template, cx, cy, 0.0, (w, h))
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return template
    local_centroid = np.array([xs.mean() - cx, ys.mean() - cy], dtype=float)
    return [np.asarray(poly, dtype=float) - local_centroid for poly in template]


def _asymmetric_l_template() -> list[np.ndarray]:
    horizontal = np.array([[-20.0, -5.0], [20.0, -5.0], [20.0, 5.0], [-20.0, 5.0]])
    vertical = np.array([[-20.0, -5.0], [-10.0, -5.0], [-10.0, 25.0], [-20.0, 25.0]])
    return _centre_template([horizontal, vertical])


def _near_symmetric_template() -> list[np.ndarray]:
    left = np.array([[-18.0, -4.0], [-8.0, -4.0], [-8.0, 4.0], [-18.0, 4.0]])
    right = np.array([[9.0, -3.5], [19.0, -3.5], [19.0, 3.5], [9.0, 3.5]])
    return _centre_template([left, right], hw=(100, 100))


def _rasterize(
    template: list[np.ndarray],
    cx: float,
    cy: float,
    theta: float,
    hw: tuple[int, int],
) -> np.ndarray:
    h, w = hw
    return _rasterize_template_polygons(template, cx, cy, theta, (w, h))


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def _angle_error(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


@pytest.fixture
def api() -> WorldAPI:
    return WorldAPI(output_dir="dummy_out")


@pytest.fixture
def simulator() -> ActionConditionedSimulatorBase:
    return ActionConditionedSimulatorBase(frame_size=(128, 96), api=None)


@pytest.mark.parametrize(
    "theta_true",
    [np.pi / 4, np.pi / 2, np.pi, 3 * np.pi / 2],
    ids=["45deg", "90deg", "180deg", "270deg"],
)
def test_recovers_known_pose(api: WorldAPI, theta_true: float) -> None:
    template = _asymmetric_l_template()
    n_angles = 180
    cx, cy = 64.0, 48.0
    mask = _rasterize(template, cx, cy, theta_true, (96, 128))
    expected_cx, expected_cy = _mask_centroid(mask)

    result = api.fit_pose_to_mask(mask, template, n_angles=n_angles)

    assert abs(result["cx"] - cx) <= 1.0
    assert abs(result["cy"] - cy) <= 1.0
    assert abs(result["cx"] - expected_cx) <= 1.5
    assert abs(result["cy"] - expected_cy) <= 1.5
    assert _angle_error(result["theta"], theta_true) <= 2 * np.pi / n_angles + 1e-9
    assert result["iou"] > 0.85
    assert result["valid"] is True
    assert result["confidence"] == pytest.approx(result["iou"])


def test_empty_mask_returns_frame_centre(api: WorldAPI) -> None:
    template = _asymmetric_l_template()
    mask = np.zeros((96, 128), dtype=bool)

    result = api.fit_pose_to_mask(mask, template)

    assert result == {
        "cx": 64.0,
        "cy": 48.0,
        "theta": 0.0,
        "iou": 0.0,
        "valid": False,
        "confidence": 0.0,
    }


def test_iou_picks_unique_orientation_over_symmetric_alternative(api: WorldAPI) -> None:
    template = _near_symmetric_template()
    n_angles = 360
    theta_true = np.pi / 2
    cx, cy = 70.0, 50.0
    mask = _rasterize(template, cx, cy, theta_true, (100, 100))

    result = api.fit_pose_to_mask(mask, template, n_angles=n_angles)

    assert _angle_error(result["theta"], theta_true) <= 2 * (2 * np.pi / n_angles) + 1e-9
    assert _angle_error(result["theta"], theta_true + np.pi / 2) > 2 * np.pi / n_angles
    assert result["iou"] > 0.85


@pytest.mark.parametrize("theta_true", [np.pi / 4, np.pi, 3 * np.pi / 2])
def test_simulator_helper_recovers_pose_without_world_api(
    simulator: ActionConditionedSimulatorBase,
    theta_true: float,
) -> None:
    template = _asymmetric_l_template()
    mask = _rasterize(template, 64.0, 48.0, theta_true, (96, 128))

    result = simulator.fit_pose_to_mask(mask, template, n_angles=180)

    assert simulator.api is None
    assert _angle_error(result["theta"], theta_true) <= 2 * np.pi / 180 + 1e-9
    assert result["iou"] > 0.85
    assert result["valid"] is True
    assert result["confidence"] == pytest.approx(result["iou"])


def test_simulator_helper_validates_inputs(
    simulator: ActionConditionedSimulatorBase,
) -> None:
    template = _asymmetric_l_template()

    with pytest.raises(ValueError, match="mask must be a 2D"):
        simulator.fit_pose_to_mask(np.zeros((4, 4, 1)), template)
    with pytest.raises(ValueError, match="n_angles must be a positive integer"):
        simulator.fit_pose_to_mask(np.zeros((4, 4)), template, n_angles=0)
    with pytest.raises(ValueError, match="at least one polygon"):
        simulator.fit_pose_to_mask(np.zeros((4, 4)), [])


def test_recovers_encoded_origin_for_uncentred_template(
    simulator: ActionConditionedSimulatorBase,
) -> None:
    template = [
        np.array([[0.0, 0.0], [28.0, 0.0], [28.0, 8.0], [0.0, 8.0]]),
        np.array([[10.0, 8.0], [18.0, 8.0], [18.0, 30.0], [10.0, 30.0]]),
    ]
    cx, cy, theta = 42.0, 31.0, np.pi / 3.0
    mask = _rasterize(template, cx, cy, theta, (96, 128))

    result = simulator.fit_pose_to_mask(mask, template, n_angles=180)

    assert abs(result["cx"] - cx) <= 1.5
    assert abs(result["cy"] - cy) <= 1.5
    assert _angle_error(result["theta"], theta) <= np.deg2rad(3)
    assert result["iou"] > 0.8
    assert result["dice"] > 0.88


def test_largest_connected_component_removes_noise(
    simulator: ActionConditionedSimulatorBase,
) -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:12, 4:13] = True
    mask[1, 1] = True
    mask[17:19, 17:19] = True

    cleaned = simulator.largest_connected_component(mask, min_area=10)

    assert int(cleaned.sum()) == 63
    assert cleaned[5:12, 4:13].all()
    assert not cleaned[1, 1]
