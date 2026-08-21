from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from vdaworld.core.cem import CEM
from vdaworld.eval.mpc import load_simulator_class


ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = ROOT / "scripts" / "ogbench_cube_reference_simulator.py"


@pytest.fixture(scope="module")
def simulator_class():
    return load_simulator_class(SIM_PATH, "GeneratedSimulator")


def _parsed(
    cube_pos: tuple[float, float, float],
    *,
    cube_visible: bool = True,
    contact_score: float | None = None,
) -> dict:
    cube = np.asarray(cube_pos, dtype=np.float64)
    if contact_score is None:
        contact_score = 1.0 if cube[2] > 0.055 else 0.0
    return {
        "cube_xy": cube[:2].copy(),
        "cube_pos": cube,
        "gripper_xy": cube[:2].copy(),
        "gripper_z": max(0.03, float(cube[2] - 0.003)),
        "cube_yaw": 0.0,
        "cube_visible": cube_visible,
        "gripper_visible": True,
        "contact_score": contact_score,
    }


def _fit_with_parsed(sim, current: dict, goal: dict) -> None:
    parsed = iter((current, goal))
    sim._parse_image = lambda _image: next(parsed)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    sim.fit(frame, frame)


def _assert_state_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    for key in left:
        if isinstance(left[key], np.ndarray):
            np.testing.assert_allclose(left[key], right[key])
        else:
            assert left[key] == pytest.approx(right[key])


def test_pose_constants_and_mask_estimator_load(simulator_class):
    assert simulator_class.CUBE_POSE_EXPANDED_MEAN.shape == (54,)
    assert simulator_class.CUBE_POSE_EXPANDED_SCALE.shape == (54,)
    assert simulator_class.CUBE_POSE_COEFFICIENT.shape == (55, 3)

    mask = np.zeros((200, 200), dtype=bool)
    mask[95:110, 80:96] = True
    position = simulator_class._estimate_cube_pos(mask)
    assert position is not None
    assert position.shape == (3,)
    assert np.isfinite(position).all()
    assert (position >= simulator_class.WORKSPACE_LOW).all()
    assert (position <= simulator_class.WORKSPACE_HIGH).all()


def test_fit_detects_initial_attachment_and_lifted_goal(simulator_class):
    sim = simulator_class(frame_size=(200, 200))
    current_pos = np.array([0.40, 0.08, 0.18])
    goal_pos = np.array([0.50, -0.10, 0.22])
    _fit_with_parsed(sim, _parsed(tuple(current_pos)), _parsed(tuple(goal_pos)))

    assert sim.state["holding"] is True
    assert sim.state["grip_closed"] >= 0.7
    np.testing.assert_allclose(sim.state["cube_pos"], current_pos)
    np.testing.assert_allclose(
        sim.state["ee_pos"][:2],
        current_pos[:2] + sim.GRASP_OFFSET_XY,
    )
    assert sim.target_state["goal_mode"] == "lifted"
    np.testing.assert_allclose(sim.target_state["cube_pos"], goal_pos)


def test_update_is_target_independent(simulator_class):
    current = _parsed((0.40, 0.02, 0.14))
    sim_left = simulator_class(frame_size=(200, 200))
    sim_right = simulator_class(frame_size=(200, 200))
    _fit_with_parsed(sim_left, current, _parsed((0.48, -0.08, 0.18)))
    _fit_with_parsed(sim_right, current, _parsed((0.32, 0.20, 0.02)))
    sim_right.state = copy.deepcopy(sim_left.state)

    action = np.array([0.4, -0.2, 0.5, 0.1, 0.2])
    sim_left.update(action)
    sim_right.update(action)
    _assert_state_equal(sim_left.state, sim_right.state)


def test_visible_held_cube_regrounds_propagated_attachment(simulator_class):
    sim = simulator_class(frame_size=(200, 200))
    _fit_with_parsed(
        sim,
        _parsed((0.40, 0.04, 0.14)),
        _parsed((0.48, -0.08, 0.20)),
    )
    sim.update(np.array([0.8, 0.6, 0.5, 0.0, 0.5]))
    propagated = copy.deepcopy(sim.state)
    observed_cube = np.array([0.41, 0.05, 0.16])

    parsed = iter(
        (
            _parsed(tuple(observed_cube)),
            _parsed((0.48, -0.08, 0.20)),
        )
    )
    sim._parse_image = lambda _image: next(parsed)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    sim.fit(frame, frame, prev_state=propagated)

    assert sim.state["holding"] is True
    np.testing.assert_allclose(sim.state["cube_pos"], observed_cube)
    np.testing.assert_allclose(
        sim.state["ee_pos"],
        np.array([observed_cube[0], observed_cube[1], observed_cube[2] - 0.003]),
    )


def test_stale_attachment_unlatches_without_visual_contact(simulator_class):
    sim = simulator_class(frame_size=(200, 200))
    _fit_with_parsed(
        sim,
        _parsed((0.40, 0.04, 0.14)),
        _parsed((0.48, -0.08, 0.20)),
    )
    propagated = copy.deepcopy(sim.state)

    parsed = iter(
        (
            _parsed((0.40, 0.04, 0.02), contact_score=0.1),
            _parsed((0.48, -0.08, 0.20)),
        )
    )
    sim._parse_image = lambda _image: next(parsed)
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    sim.fit(frame, frame, prev_state=propagated)

    assert sim.state["holding"] is False
    assert sim.state["cube_pos"][2] == pytest.approx(sim.TABLE_Z)


def test_lifted_goal_cem_reduces_3d_cube_error(simulator_class):
    sim = simulator_class(frame_size=(200, 200))
    start = np.array([0.40, 0.00, 0.12])
    target = np.array([0.46, 0.04, 0.16])
    _fit_with_parsed(sim, _parsed(tuple(start)), _parsed(tuple(target)))
    initial_error = float(np.linalg.norm(sim.state["cube_pos"] - target))

    planner = CEM(
        sim,
        action_dim=5,
        action_low=-np.ones(5),
        action_high=np.ones(5),
        horizon=5,
        population=120,
        elite_frac=0.1,
        iters=6,
        init_std_scale=1.0,
        seed=7,
    )
    plan = planner.plan()
    for action in plan:
        sim.update(action)

    final_error = float(np.linalg.norm(sim.state["cube_pos"] - target))
    assert sim.state["holding"] is True
    assert final_error < initial_error * 0.55


def test_render_contract_for_lifted_state(simulator_class):
    sim = simulator_class(frame_size=(160, 120))
    _fit_with_parsed(
        sim,
        _parsed((0.40, 0.02, 0.14)),
        _parsed((0.46, -0.06, 0.20)),
    )
    frame = sim.render_frame()
    assert frame.shape == (120, 160, 3)
    assert frame.dtype == np.uint8
