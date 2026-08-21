"""Integration tests for vdaworld.eval.mpc.eval_mpc.

Uses a deterministic minimal-fixture simulator (no walls, exact red-dot
tracker, identity update) so we test the loop plumbing — not the quality of
the planner against a real test env. That harder test is the smoke script.
"""

from __future__ import annotations

import numpy as np
import pytest

from vdaworld.core.simulator import ActionConditionedSimulatorBase
from vdaworld.eval.mpc import (
    EVAL_PROTOCOL_LEWM_ALIGNED_N50,
    EVAL_PROTOCOL_LEWM_MATCHED,
    EVAL_PROTOCOL_VDA_DEFAULT,
    _apply_eval_protocol,
    _resolve_p2_diagnostics,
    eval_mpc,
)


class _FixtureSim(ActionConditionedSimulatorBase):
    """No collisions, no rendering geometry — just enough to drive eval_mpc end-to-end.
    Centroid-tracks the red dot in the start image; update integrates actions
    linearly; loss is Euclidean to the target dot position."""

    def __init__(self, frame_size=(65, 65), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)

    def _find_red(self, img: np.ndarray) -> np.ndarray:
        red = (img[..., 0].astype(int) - np.maximum(img[..., 1], img[..., 2])) > 50
        if not red.any():
            return np.array([img.shape[1] / 2, img.shape[0] / 2], dtype=float)
        ys, xs = np.where(red)
        return np.array([xs.mean(), ys.mean()], dtype=float)

    def fit(self, image_A: np.ndarray, image_B: np.ndarray) -> None:
        self.state = {"pos": self._find_red(image_A)}
        self.target_state = {"pos": self._find_red(image_B)}

    def update(self, a: np.ndarray) -> None:
        self.state["pos"] = self.state["pos"] + np.asarray(a, dtype=float)

    def loss_to_target(self) -> float:
        return float(np.linalg.norm(self.state["pos"] - self.target_state["pos"]))

    def render_frame(self) -> np.ndarray:
        img = np.full((self.frame_size[0], self.frame_size[1], 3), 255, dtype=np.uint8)
        x, y = self.state["pos"]
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < img.shape[0] and 0 <= ix < img.shape[1]:
            img[iy, ix] = [255, 0, 0]
        return img


class _CountingOpenLoopSim(ActionConditionedSimulatorBase):
    def __init__(self, frame_size=(9, 9), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)

    @staticmethod
    def _point_from_img(img: np.ndarray) -> np.ndarray:
        pos = np.argwhere(img[..., 0] > 0)
        if len(pos) == 0:
            return np.array([4.0, 4.0], dtype=np.float64)
        y, x = pos[0]
        return np.array([float(x), float(y)], dtype=np.float64)

    def fit(self, image_A: np.ndarray, image_B: np.ndarray) -> None:
        self.state = {"pos": self._point_from_img(image_A)}
        self.target_state = {"pos": self._point_from_img(image_B)}

    def update(self, a: np.ndarray) -> None:
        self.state["pos"] = self.state["pos"] + np.asarray(a, dtype=np.float64)

    def loss_to_target(self) -> float:
        return float(np.linalg.norm(self.state["pos"] - self.target_state["pos"]))

    def render_frame(self) -> np.ndarray:
        img = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        x, y = self.state["pos"]
        ix = int(np.clip(round(x), 0, self.frame_size[0] - 1))
        iy = int(np.clip(round(y), 0, self.frame_size[1] - 1))
        img[iy, ix, 0] = 255
        return img


class _RuntimeContractSim(_CountingOpenLoopSim):
    constructed_fps = None
    fit_api_values = []
    rollout_api_values = []

    def __init__(self, frame_size=(9, 9), api=None, fps=30):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        type(self).constructed_fps = fps

    def fit(self, image_A: np.ndarray, image_B: np.ndarray) -> None:
        type(self).fit_api_values.append(self.api)
        super().fit(image_A, image_B)

    def update(self, a: np.ndarray) -> None:
        type(self).rollout_api_values.append(self.api)
        super().update(a)


class _PrevStateSim(_CountingOpenLoopSim):
    fit_received_prev_state = []

    def fit(
        self,
        image_A: np.ndarray,
        image_B: np.ndarray,
        prev_state=None,
    ) -> None:
        type(self).fit_received_prev_state.append(prev_state is not None)
        super().fit(image_A, image_B)
        self.state["latent_counter"] = (
            int(prev_state["latent_counter"]) if prev_state is not None else 0
        )

    def update(self, a: np.ndarray) -> None:
        super().update(a)
        self.state["latent_counter"] += 1


class _FixtureSession:
    def __init__(self, frame_size=(65, 65)):
        self.frame_size = frame_size
        self.pos = np.array([10.0, 10.0], dtype=np.float64)
        self.goal = np.array([20.0, 20.0], dtype=np.float64)
        self.closed = False

    def _render(self, pos):
        w, h = self.frame_size
        img = np.full((h, w, 3), 255, dtype=np.uint8)
        x, y = np.asarray(pos, dtype=np.float64)
        img[int(round(y)), int(round(x))] = [255, 0, 0]
        return img

    def start(self, seed):
        return {
            "image": self._render(self.pos),
            "goal_image": self._render(self.goal),
            "info": {"pos": self.pos.tolist()},
            "done": False,
            "truncated": False,
            "reward": 0.0,
        }

    def step(self, action):
        self.pos = self.pos + np.asarray(action, dtype=np.float64)
        done = bool(np.linalg.norm(self.pos - self.goal) < 1.0)
        return {
            "image": self._render(self.pos),
            "goal_image": self._render(self.goal),
            "info": {"pos": self.pos.tolist()},
            "done": done,
            "truncated": False,
            "reward": -float(np.linalg.norm(self.pos - self.goal)),
        }

    def close(self):
        self.closed = True


class _NoDoneSession:
    def __init__(self, frame_size=(9, 9)):
        self.frame_size = frame_size
        self.pos = np.array([1.0, 1.0], dtype=np.float64)
        self.goal = np.array([7.0, 7.0], dtype=np.float64)
        self.step_calls = 0

    def _render(self, pos):
        w, h = self.frame_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        x = int(np.clip(round(pos[0]), 0, w - 1))
        y = int(np.clip(round(pos[1]), 0, h - 1))
        img[y, x, 0] = 255
        return img

    def start(self, seed):
        return {
            "image": self._render(self.pos),
            "goal_image": self._render(self.goal),
            "info": {"step": self.step_calls},
            "done": False,
            "truncated": False,
            "reward": 0.0,
        }

    def step(self, action):
        self.step_calls += 1
        self.pos = self.pos + np.asarray(action, dtype=np.float64)
        return {
            "image": self._render(self.pos),
            "goal_image": self._render(self.goal),
            "info": {"step": self.step_calls},
            "done": False,
            "truncated": False,
            "reward": 0.0,
        }

    def close(self):
        pass


class _PrimarySuccessSession(_NoDoneSession):
    def __init__(self, frame_size=(9, 9), success_step=2):
        super().__init__(frame_size=frame_size)
        self.success_step = success_step

    def step(self, action):
        result = super().step(action)
        result["info"]["primary_success"] = self.step_calls >= self.success_step
        return result


@pytest.mark.integration
def test_eval_mpc_defaults_to_receding_horizon_mode():
    result = eval_mpc(
        _FixtureSim,
        session=_FixtureSession(frame_size=(65, 65)),
        action_low=np.array([-2.0, -2.0], dtype=np.float64),
        action_high=np.array([2.0, 2.0], dtype=np.float64),
        max_episode_steps=3,
        cem_kwargs={"population": 20, "iters": 2, "seed": 0},
        seed=0,
    )
    assert result["plan_mode"] == "mpc"
    assert result["eval_protocol"] == EVAL_PROTOCOL_VDA_DEFAULT
    assert len(result["cem_histories"]) >= 1
    assert result["propagate_prev_state"] is True
    assert result["stop_on_primary_success"] is False
    assert result["timing"]["total_seconds"] >= 0.0
    assert result["timing"]["cem_total_seconds"] >= 0.0
    assert len(result["timing"]["environment_step_seconds"]) == result["n_steps"]
    assert result["estimated_simulator_transitions"] > 0


def test_lewm_matched_protocol_applies_low_level_defaults():
    horizon, apply_steps, max_steps, cem_kwargs = _apply_eval_protocol(
        eval_protocol=EVAL_PROTOCOL_LEWM_MATCHED,
        horizon=5,
        apply_steps=2,
        max_episode_steps=50,
        cem_kwargs={},
    )

    assert horizon == 25
    assert apply_steps == 25
    assert max_steps == 50
    assert cem_kwargs["population"] == 300
    assert cem_kwargs["iters"] == 30
    assert cem_kwargs["elite_frac"] == 0.1


def test_frozen_profile_forces_default_p2_diagnostics():
    assert _resolve_p2_diagnostics(
        eval_protocol=EVAL_PROTOCOL_LEWM_ALIGNED_N50,
        propagate_prev_state=False,
        stop_on_primary_success=True,
    ) == (True, False)
    assert _resolve_p2_diagnostics(
        eval_protocol=EVAL_PROTOCOL_VDA_DEFAULT,
        propagate_prev_state=False,
        stop_on_primary_success=True,
    ) == (False, True)


@pytest.mark.integration
def test_prev_state_propagation_defaults_on_and_can_be_disabled():
    def run(propagate_prev_state):
        _PrevStateSim.fit_received_prev_state = []
        kwargs = {}
        if propagate_prev_state is not None:
            kwargs["propagate_prev_state"] = propagate_prev_state
        result = eval_mpc(
            _PrevStateSim,
            session=_NoDoneSession(frame_size=(9, 9)),
            action_low=np.array([-1.0, -1.0]),
            action_high=np.array([1.0, 1.0]),
            horizon=1,
            apply_steps=1,
            max_episode_steps=2,
            cem_kwargs={"population": 4, "iters": 1, "seed": 0},
            sim_frame_size=(9, 9),
            seed=0,
            **kwargs,
        )
        return result, list(_PrevStateSim.fit_received_prev_state)

    default_result, default_received = run(None)
    disabled_result, disabled_received = run(False)

    assert default_received == [False, True]
    assert default_result["propagate_prev_state"] is True
    assert disabled_received == [False, False]
    assert disabled_result["propagate_prev_state"] is False


@pytest.mark.integration
def test_primary_success_does_not_stop_by_default():
    result = eval_mpc(
        _CountingOpenLoopSim,
        session=_PrimarySuccessSession(frame_size=(9, 9), success_step=2),
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        horizon=2,
        apply_steps=2,
        max_episode_steps=4,
        cem_kwargs={"population": 4, "iters": 1, "seed": 0},
        sim_frame_size=(9, 9),
        seed=0,
    )

    assert result["n_steps"] == 4
    assert result["primary_success_observed"] is True
    assert result["stop_reason"] is None
    assert result["stop_on_primary_success"] is False


@pytest.mark.integration
@pytest.mark.parametrize("plan_mode", ["mpc", "open_loop"])
def test_primary_success_diagnostic_stops_and_records_reason(plan_mode):
    result = eval_mpc(
        _CountingOpenLoopSim,
        session=_PrimarySuccessSession(frame_size=(9, 9), success_step=2),
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        horizon=4,
        apply_steps=4,
        max_episode_steps=6,
        cem_kwargs={"population": 4, "iters": 1, "seed": 0},
        sim_frame_size=(9, 9),
        seed=0,
        plan_mode=plan_mode,
        plan_horizon=6,
        stop_on_primary_success=True,
    )

    assert result["n_steps"] == 2
    assert result["done"] is False
    assert result["primary_success_observed"] is True
    assert result["gt_infos"][-1]["primary_success"] is True
    assert result["stop_reason"] == "primary_success"
    assert result["stop_on_primary_success"] is True


@pytest.mark.integration
def test_eval_mpc_enforces_fps_and_fit_scoped_api():
    marker_api = object()
    _RuntimeContractSim.fit_api_values = []
    _RuntimeContractSim.rollout_api_values = []
    result = eval_mpc(
        _RuntimeContractSim,
        session=_NoDoneSession(frame_size=(9, 9)),
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        horizon=1,
        apply_steps=1,
        max_episode_steps=1,
        cem_kwargs={"population": 4, "iters": 1, "seed": 0},
        sim_frame_size=(9, 9),
        sim_fps=7,
        api=marker_api,
        seed=0,
    )

    assert _RuntimeContractSim.constructed_fps == 7
    assert _RuntimeContractSim.fit_api_values == [marker_api]
    assert _RuntimeContractSim.rollout_api_values
    assert all(value is None for value in _RuntimeContractSim.rollout_api_values)
    assert result["sim_fps"] == 7
    assert result["world_api_fit_enabled"] is True
    assert result["world_api_rollout_enabled"] is False


@pytest.mark.integration
def test_eval_mpc_runs_end_to_end():
    """Smallest viable MPC run: horizon=1, apply_steps=1, max_episode_steps=3.
    Asserts the loop ran, produced expected dict keys, and the action count is
    consistent with the step budget."""
    result = eval_mpc(
        _FixtureSim,
        session=_FixtureSession(frame_size=(65, 65)),
        action_low=np.array([-2.0, -2.0], dtype=np.float64),
        action_high=np.array([2.0, 2.0], dtype=np.float64),
        horizon=1,
        apply_steps=1,
        max_episode_steps=3,
        cem_kwargs={"population": 20, "iters": 2, "seed": 0},
        seed=0,
        plan_mode="mpc",
    )
    expected_keys = {
        "sim_predicted_frames",
        "observations_sim",
        "observations_raw",
        "goal_image_sim",
        "goal_image_raw",
        "actions_applied",
        "cem_histories",
        "terminal_distance",
        "n_steps",
        "done",
        "truncated",
    }
    assert expected_keys.issubset(result.keys())
    assert result["n_steps"] >= 1
    assert result["n_steps"] <= 3
    assert result["actions_applied"].shape == (result["n_steps"], 2)
    # observations include a trailing post-final-step observation.
    assert len(result["observations_raw"]) == result["n_steps"] + 1
    # The detailed visualization trace contains every real action, its matching
    # simulator prediction, fit/replan markers, and an explicit final marker.
    trace_lengths = {
        len(result["execution_observations_raw"]),
        len(result["execution_sim_predicted_frames"]),
        len(result["execution_env_steps"]),
        len(result["execution_mpc_cycles"]),
        len(result["execution_events"]),
        len(result["execution_action_indices"]),
        len(result["execution_sim_losses"]),
    }
    assert len(trace_lengths) == 1
    assert result["execution_events"][0] == "fit_cem_replan"
    assert result["execution_events"][-1] == "final_observation"
    assert result["execution_events"].count("env_update") == result["n_steps"]
    action_trace_steps = [
        step
        for step, event in zip(
            result["execution_env_steps"], result["execution_events"]
        )
        if event == "env_update"
    ]
    assert action_trace_steps == list(range(1, result["n_steps"] + 1))
    assert len(result["cem_histories"]) >= 1
    # Goal images are valid uint8 frames of the right shape.
    assert result["goal_image_raw"].shape == (65, 65, 3)
    assert result["goal_image_sim"].shape == (65, 65, 3)
    assert result["terminal_distance"] >= 0
    assert result["cem_use_stage_cost"] is False
    assert result["cem_action_smoothness_weight"] == 0.0
    assert result["cem_num_action_knots"] is None
    assert result["cem_seed"] == 0


@pytest.mark.integration
def test_eval_open_loop_plans_once_and_applies_full_horizon():
    plan_horizon = 6
    result = eval_mpc(
        _CountingOpenLoopSim,
        session=_NoDoneSession(frame_size=(9, 9)),
        action_low=np.array([-1.0, -1.0], dtype=np.float64),
        action_high=np.array([1.0, 1.0], dtype=np.float64),
        horizon=3,
        apply_steps=1,
        max_episode_steps=100,
        cem_kwargs={"population": 12, "iters": 2, "seed": 0},
        sim_frame_size=(9, 9),
        seed=0,
        plan_mode="open_loop",
        plan_horizon=plan_horizon,
    )
    assert result["plan_mode"] == "open_loop"
    assert result["plan_horizon"] == plan_horizon
    # Open-loop does one planner fit and one planner invocation.
    assert len(result["sim_fit_states"]) == 1
    assert len(result["cem_histories"]) == 1
    assert len(result["actions_applied"]) == plan_horizon
    assert len(result["sim_predicted_rollout"]) == plan_horizon + 1
    assert len(result["observations_raw"]) == plan_horizon + 1


@pytest.mark.integration
def test_eval_mpc_mode_respects_receding_horizon_budget():
    result = eval_mpc(
        _FixtureSim,
        session=_FixtureSession(frame_size=(65, 65)),
        action_low=np.array([-2.0, -2.0], dtype=np.float64),
        action_high=np.array([2.0, 2.0], dtype=np.float64),
        horizon=2,
        apply_steps=1,
        max_episode_steps=3,
        cem_kwargs={"population": 20, "iters": 2, "seed": 0},
        seed=0,
        plan_mode="mpc",
        plan_horizon=11,
    )
    assert result["plan_mode"] == "mpc"
    assert result["plan_horizon"] == 11
    assert result["n_steps"] <= 3
    assert len(result["cem_histories"]) == result["n_steps"]
    assert len(result["sim_predicted_rollout"]) == 0


@pytest.mark.integration
def test_eval_mpc_lewm_matched_protocol_records_metadata_with_overrides():
    result = eval_mpc(
        _FixtureSim,
        session=_NoDoneSession(frame_size=(65, 65)),
        action_low=np.array([-0.1, -0.1], dtype=np.float64),
        action_high=np.array([0.1, 0.1], dtype=np.float64),
        cem_kwargs={"population": 4, "iters": 1, "elite_frac": 0.5, "seed": 0},
        seed=0,
        plan_mode="mpc",
        eval_protocol=EVAL_PROTOCOL_LEWM_MATCHED,
        start_goal_source="unit_test",
        num_eval=5,
    )

    assert result["eval_protocol"] == EVAL_PROTOCOL_LEWM_MATCHED
    assert result["start_goal_source"] == "unit_test"
    assert result["num_eval"] == 5
    assert result["horizon"] == 25
    assert result["apply_steps"] == 25
    assert result["max_episode_steps"] == 50
    assert result["effective_plan_steps"] == 25
    assert result["effective_apply_steps"] == 25
    assert result["cem_population"] == 4
    assert result["cem_iters"] == 1
    assert result["cem_elite_frac"] == 0.5
    assert result["cem_topk"] == 2
    assert result["n_steps"] == 50
    assert len(result["cem_histories"]) == 2
    assert len(result["gt_infos"]) == 51
    assert [info["step"] for info in result["gt_infos"]] == list(range(51))


def test_lewm_aligned_n50_profile_freezes_controller_and_cem_values():
    horizon, apply_steps, max_steps, cem_kwargs = _apply_eval_protocol(
        eval_protocol=EVAL_PROTOCOL_LEWM_ALIGNED_N50,
        horizon=3,
        apply_steps=2,
        max_episode_steps=7,
        cem_kwargs={
            "population": 4,
            "iters": 1,
            "elite_frac": 0.5,
            "init_std_scale": 0.25,
            "seed": 9,
            "use_stage_cost": True,
            "action_smoothness_weight": 2.5,
            "num_action_knots": 3,
        },
    )

    assert horizon == 25
    assert apply_steps == 25
    assert max_steps == 50
    assert cem_kwargs["population"] == 300
    assert cem_kwargs["iters"] == 30
    assert cem_kwargs["elite_frac"] == 0.1
    assert cem_kwargs["init_std_scale"] == 1.0
    assert cem_kwargs["seed"] == 9
    assert cem_kwargs["use_stage_cost"] is False
    assert cem_kwargs["action_smoothness_weight"] == 0.0
    assert cem_kwargs["num_action_knots"] is None


@pytest.mark.integration
def test_eval_mpc_records_resolved_generic_cem_settings():
    result = eval_mpc(
        _FixtureSim,
        session=_FixtureSession(frame_size=(65, 65)),
        action_low=np.array([-2.0, -2.0], dtype=np.float64),
        action_high=np.array([2.0, 2.0], dtype=np.float64),
        horizon=3,
        apply_steps=1,
        max_episode_steps=1,
        cem_kwargs={
            "population": 8,
            "iters": 1,
            "seed": 17,
            "use_stage_cost": True,
            "action_smoothness_weight": 0.75,
            "num_action_knots": 2,
        },
        seed=4,
    )

    assert result["cem_use_stage_cost"] is True
    assert result["cem_action_smoothness_weight"] == 0.75
    assert result["cem_num_action_knots"] == 2
    assert result["cem_seed"] == 17
