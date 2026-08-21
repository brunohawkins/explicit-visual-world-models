import numpy as np
import pytest

from vdaworld.core.api import WorldAPI


@pytest.fixture
def api() -> WorldAPI:
    return WorldAPI()


def test_clamp_step_caps_magnitude(api: WorldAPI) -> None:
    result = api.clamp_step(
        prev_pos=np.array([0.0, 0.0]),
        target_pos=np.array([3.0, 4.0]),
        max_step=2.0,
    )

    np.testing.assert_allclose(result, np.array([1.2, 1.6]))
    assert np.linalg.norm(result) == pytest.approx(2.0)


def test_clamp_step_identity_within_bound(api: WorldAPI) -> None:
    target = np.array([0.3, -0.4])

    result = api.clamp_step(np.array([0.0, 0.0]), target, max_step=1.0)

    np.testing.assert_allclose(result, target)


def test_resolve_contact_pushes_penetrating_point_outward(api: WorldAPI) -> None:
    result = api.resolve_contact(
        body_points_world=np.array([[0.5, 0.0], [2.0, 0.0]]),
        body_center=np.array([1.0, 0.0]),
        pusher_pos=np.array([0.0, 0.0]),
        pusher_radius=1.0,
        k_trans=1.0,
        k_rot=1.0,
    )

    np.testing.assert_allclose(result["d_pos"], np.array([0.5, 0.0]))
    assert result["d_theta"] == pytest.approx(0.0)


def test_resolve_contact_zero_without_penetration(api: WorldAPI) -> None:
    result = api.resolve_contact(
        body_points_world=np.array([[2.0, 0.0], [0.0, 2.0]]),
        body_center=np.array([1.0, 1.0]),
        pusher_pos=np.array([0.0, 0.0]),
        pusher_radius=1.0,
        k_trans=3.0,
        k_rot=4.0,
    )

    np.testing.assert_allclose(result["d_pos"], np.zeros(2))
    assert result["d_theta"] == pytest.approx(0.0)


def test_forward_kinematics_two_link_case(api: WorldAPI) -> None:
    result = api.forward_kinematics(
        base_pos=np.array([0.0, 0.0]),
        angles=np.array([0.0, np.pi / 2.0]),
        link_lengths=np.array([1.0, 2.0]),
    )

    expected_joints = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]])
    np.testing.assert_allclose(result["joints"], expected_joints, atol=1e-12)
    np.testing.assert_allclose(result["end_effector"], expected_joints[-1], atol=1e-12)


def test_forward_kinematics_three_link_case(api: WorldAPI) -> None:
    result = api.forward_kinematics(
        base_pos=np.array([1.0, 1.0]),
        angles=np.array([np.pi / 2.0, -np.pi / 2.0, np.pi]),
        link_lengths=np.array([2.0, 3.0, 1.0]),
    )

    expected_joints = np.array([[1.0, 1.0], [1.0, 3.0], [4.0, 3.0], [3.0, 3.0]])
    np.testing.assert_allclose(result["joints"], expected_joints, atol=1e-12)
    np.testing.assert_allclose(result["end_effector"], expected_joints[-1], atol=1e-12)


def test_collide_occupancy_stops_at_wall_and_slides(api: WorldAPI) -> None:
    occupancy = np.zeros((7, 7), dtype=bool)
    occupancy[:, 3] = True

    stopped = api.collide_occupancy(
        pos=np.array([1.0, 3.0]),
        step=np.array([4.0, 0.0]),
        occupancy_mask=occupancy,
        radius=0.25,
    )
    assert 1.9 <= stopped[0] <= 2.1
    assert stopped[1] == pytest.approx(3.0)

    slid = api.collide_occupancy(
        pos=np.array([1.0, 1.0]),
        step=np.array([4.0, 2.0]),
        occupancy_mask=occupancy,
        radius=0.25,
    )
    assert 1.9 <= slid[0] <= 2.1
    assert slid[1] == pytest.approx(3.0)


def test_dynamics_toolbox_reports_bad_shapes(api: WorldAPI) -> None:
    with pytest.raises(ValueError, match="prev_pos"):
        api.clamp_step(np.zeros(3), np.zeros(2), max_step=1.0)

    with pytest.raises(ValueError, match="body_points_world"):
        api.resolve_contact(np.zeros(3), np.zeros(2), np.zeros(2), 1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="same length"):
        api.forward_kinematics(np.zeros(2), np.zeros(2), np.ones(3))

    with pytest.raises(ValueError, match="occupancy_mask"):
        api.collide_occupancy(np.zeros(2), np.zeros(2), np.zeros(3), radius=0.0)
