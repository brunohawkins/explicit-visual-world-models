from __future__ import annotations

import numpy as np

from vdaworld.eval.pusht_adapter import (
    green_target_mask,
    native_xy_to_sim,
    non_occluding_pusht_agent,
    sim_action_bounds,
    sim_action_to_native,
    strip_green_target,
)
from vdaworld.eval.pusht_env import GymPushTSession


def test_strip_green_target_preserves_block_pusher_and_input() -> None:
    frame = np.full((4, 5, 3), 255, dtype=np.uint8)
    frame[0, 0] = [144, 238, 144]  # exact LightGreen target
    frame[0, 1] = [236, 252, 236]  # antialiased green edge
    frame[1, 0] = [143, 163, 184]  # grey-blue T
    frame[1, 1] = [78, 126, 255]  # blue pusher
    original = frame.copy()

    stripped = strip_green_target(frame)

    np.testing.assert_array_equal(frame, original)
    np.testing.assert_array_equal(stripped[0, 0], [255, 255, 255])
    np.testing.assert_array_equal(stripped[0, 1], [255, 255, 255])
    np.testing.assert_array_equal(stripped[1, 0], original[1, 0])
    np.testing.assert_array_equal(stripped[1, 1], original[1, 1])
    assert not green_target_mask(stripped).any()


def test_green_target_mask_rejects_non_rgb_frames() -> None:
    with np.testing.assert_raises_regex(ValueError, "shape"):
        green_target_mask(np.zeros((4, 5), dtype=np.uint8))


def test_coordinate_bridge_supports_legacy_and_native_224_frames() -> None:
    np.testing.assert_allclose(sim_action_to_native([48.0, 48.0]), [256.0, 256.0])
    np.testing.assert_allclose(
        sim_action_to_native([112.0, 112.0], (224, 224)),
        [256.0, 256.0],
    )
    np.testing.assert_allclose(
        native_xy_to_sim([256.0, 128.0], frame_size=(224, 224)),
        [112.0, 56.0],
    )
    np.testing.assert_allclose(
        native_xy_to_sim(
            [256.0, 128.0],
            frame_size=(224, 224),
            flip_y=True,
        ),
        [112.0, 168.0],
    )
    low, high = sim_action_bounds((224, 224))
    np.testing.assert_array_equal(low, [0.0, 0.0])
    np.testing.assert_array_equal(high, [224.0, 224.0])


def test_swm_goal_pusher_visibility_rule_preserves_safe_and_moves_overlap() -> None:
    target = np.array([256.0, 256.0, 0.0])
    safe, safe_moved = non_occluding_pusht_agent(target, [30.0, 30.0])
    np.testing.assert_array_equal(safe, [30.0, 30.0])
    assert safe_moved is False

    moved, overlap_moved = non_occluding_pusht_agent(target, [256.0, 256.0])
    assert overlap_moved is True
    assert np.linalg.norm(moved - target[:2]) >= 145.0


def test_native_session_strips_only_simulator_inputs() -> None:
    session = GymPushTSession(
        sim_frame_size=(96, 96),
        strip_green_target_from_sim=True,
    )
    try:
        start = session.start(seed=42)
        assert green_target_mask(start["image_raw"]).any()
        assert not green_target_mask(start["image"]).any()
        assert np.any(start["image"] != start["image_raw"])

        step = session.step([48.0, 48.0])
        assert green_target_mask(step["image_raw"]).any()
        assert not green_target_mask(step["image"]).any()
        assert np.any(step["image"] != step["image_raw"])
    finally:
        session.close()


def test_native_224_session_renders_directly_and_relocates_occluding_goal_pusher() -> None:
    session = GymPushTSession(
        sim_frame_size=(224, 224),
        manifest_case={
            "case_id": "overlap",
            "start": {
                "pos_agent": [256.0, 256.0],
                "block_pose": [180.0, 180.0, 0.0],
            },
            "goal": {
                "pos_agent": [256.0, 256.0],
                "goal_pose": [256.0, 256.0, np.pi / 4.0],
            },
        },
    )
    try:
        start = session.start(seed=7)
        assert start["image"].shape == (224, 224, 3)
        assert start["goal_image"].shape == (224, 224, 3)
        goal_info = start["info"]["goal_render_info"]
        assert goal_info["goal_pusher_relocated_for_visibility"] is True
        requested = np.asarray(goal_info["goal_pusher_requested_native"])
        rendered = np.asarray(goal_info["goal_pusher_rendered_native"])
        np.testing.assert_array_equal(requested, [256.0, 256.0])
        assert np.linalg.norm(rendered - np.array([256.0, 256.0])) >= 145.0

        step = session.step([112.0, 112.0])
        np.testing.assert_allclose(
            step["info"]["pos_agent"],
            [256.0, 256.0],
            atol=1.0,
        )
    finally:
        session.close()
