"""Unit tests for CEM planner."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vdaworld.core.cem import CEM, CEMHistoryEntry
from vdaworld.core.simulator import ActionConditionedSimulatorBase


class _Toy1D(ActionConditionedSimulatorBase):
    """1-D agent on the real line. update(a) adds a scalar to position.

    Target is x=10. CEM should drive cost monotonically down.
    """

    def __init__(self, frame_size=(65, 65), api=None, fps=10):
        super().__init__(frame_size=frame_size, api=api, fps=fps)

    def fit(self, image_A, image_B):
        self.state = {"x": 0.0}
        self.target_state = {"x": 10.0}

    def update(self, a):
        self.state["x"] += float(np.asarray(a).item())

    def loss_to_target(self):
        return abs(self.state["x"] - self.target_state["x"])

    def render_frame(self):
        return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)


class _TerminalCostToy1D(_Toy1D):
    def terminal_cost(self):
        return abs(self.state["x"] - self.target_state["x"])


class _StageCostToy1D(_Toy1D):
    def __init__(self, frame_size=(65, 65), api=None, fps=10):
        super().__init__(frame_size=frame_size, api=api, fps=fps)
        self.stage_calls = []

    def stage_cost(self, action, step_index):
        action_value = float(np.asarray(action).item())
        self.stage_calls.append((step_index, action_value, self.state["x"]))
        return self.state["x"] ** 2


class _NonFiniteStageCostToy1D(_Toy1D):
    def stage_cost(self, action, step_index):
        return float("inf")


class _ConstantObjectiveToy1D(_Toy1D):
    def loss_to_target(self):
        return 0.0


def _make_cem(sim, seed=0, iters=8):
    return CEM(
        sim,
        action_dim=1,
        action_low=np.array([-1.5]),
        action_high=np.array([1.5]),
        horizon=10,
        population=100,
        elite_frac=0.1,
        iters=iters,
        seed=seed,
    )


def test_plan_returns_correct_shape():
    sim = _Toy1D()
    sim.fit(None, None)
    cem = _make_cem(sim, iters=3)
    actions = cem.plan()
    assert actions.shape == (10, 1)


def test_plan_accepts_terminal_cost_method():
    sim = _TerminalCostToy1D()
    sim.fit(None, None)
    cem = _make_cem(sim, iters=3)
    actions = cem.plan()
    assert actions.shape == (10, 1)


def test_history_records_each_iteration():
    sim = _Toy1D()
    sim.fit(None, None)
    cem = _make_cem(sim, iters=5)
    cem.plan()
    assert len(cem.history) == 5
    assert all(isinstance(h, CEMHistoryEntry) for h in cem.history)


def test_elite_mean_cost_decreases_overall():
    sim = _Toy1D()
    sim.fit(None, None)
    cem = _make_cem(sim, iters=10)
    cem.plan()
    first = cem.history[0].elite_mean_cost
    last = cem.history[-1].elite_mean_cost
    # Strong condition: last iteration's elite mean must be strictly better
    # than first iteration's. CEM should easily solve toy 1-D in 10 iters.
    assert last < first


def test_state_restored_after_plan():
    sim = _Toy1D()
    sim.fit(None, None)
    snapshot_before = copy.deepcopy(sim.state)
    cem = _make_cem(sim, iters=3)
    cem.plan()
    assert sim.state == snapshot_before


def test_plan_raises_if_not_fit():
    sim = _Toy1D()
    cem = _make_cem(sim, iters=2)
    with pytest.raises(RuntimeError, match="before sim.fit"):
        cem.plan()


def test_deterministic_under_seed():
    sim_a = _Toy1D()
    sim_a.fit(None, None)
    cem_a = _make_cem(sim_a, seed=42, iters=4)
    actions_a = cem_a.plan()

    sim_b = _Toy1D()
    sim_b.fit(None, None)
    cem_b = _make_cem(sim_b, seed=42, iters=4)
    actions_b = cem_b.plan()

    np.testing.assert_allclose(actions_a, actions_b)


def test_default_options_preserve_legacy_seeded_output_exactly():
    expected = np.array(
        [
            [0.8245023081896499],
            [1.5],
            [0.7434477116519813],
            [1.3075153838906255],
            [0.6217004822687132],
            [1.2019183163197995],
            [1.5],
            [0.7278570269123095],
            [1.3390848160141455],
            [0.2536395959827694],
        ]
    )
    for optional_kwargs in (
        {},
        {
            "use_stage_cost": False,
            "action_smoothness_weight": 0.0,
            "num_action_knots": None,
        },
        {"num_action_knots": 10},
    ):
        sim = _Toy1D()
        sim.fit(None, None)
        actions = CEM(
            sim,
            action_dim=1,
            action_low=np.array([-1.5]),
            action_high=np.array([1.5]),
            horizon=10,
            population=100,
            elite_frac=0.1,
            iters=4,
            seed=42,
            **optional_kwargs,
        ).plan()
        np.testing.assert_array_equal(actions, expected)


def test_stage_cost_runs_after_each_update_with_step_index():
    sim = _StageCostToy1D()
    sim.fit(None, None)
    cem = CEM(
        sim,
        action_dim=1,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        horizon=3,
        population=5,
        iters=1,
        seed=7,
        use_stage_cost=True,
    )
    cem.plan()

    assert len(sim.stage_calls) == 15
    assert [call[0] for call in sim.stage_calls] == [0, 1, 2] * 5
    running_x = 0.0
    for step_index, action, updated_x in sim.stage_calls:
        if step_index == 0:
            running_x = 0.0
        running_x += action
        assert updated_x == pytest.approx(running_x)


def test_nonfinite_stage_cost_restores_state_and_raises():
    sim = _NonFiniteStageCostToy1D()
    sim.fit(None, None)
    state_before = copy.deepcopy(sim.state)
    cem = CEM(
        sim,
        action_dim=1,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        horizon=2,
        population=2,
        iters=1,
        seed=0,
        use_stage_cost=True,
    )

    with pytest.raises(ValueError, match="finite scalar"):
        cem.plan()
    assert sim.state == state_before


def test_smoothness_penalty_selects_smoother_actions():
    def plan(weight):
        sim = _ConstantObjectiveToy1D()
        sim.fit(None, None)
        return CEM(
            sim,
            action_dim=1,
            action_low=np.array([-1.0]),
            action_high=np.array([1.0]),
            horizon=8,
            population=100,
            iters=4,
            seed=4,
            action_smoothness_weight=weight,
        ).plan()

    default_actions = plan(0.0)
    smooth_actions = plan(100.0)
    default_penalty = np.mean((np.diff(default_actions[:, 0]) / 2.0) ** 2)
    smooth_penalty = np.mean((np.diff(smooth_actions[:, 0]) / 2.0) ** 2)
    assert smooth_penalty < default_penalty


@pytest.mark.parametrize("num_knots", [1, 2, 4])
def test_action_knots_return_interpolated_full_horizon(num_knots):
    sim = _ConstantObjectiveToy1D()
    sim.fit(None, None)
    actions = CEM(
        sim,
        action_dim=1,
        action_low=np.array([-1.0]),
        action_high=np.array([1.0]),
        horizon=7,
        population=10,
        iters=1,
        seed=3,
        num_action_knots=num_knots,
    ).plan()

    assert actions.shape == (7, 1)
    assert np.all(actions >= -1.0)
    assert np.all(actions <= 1.0)
    if num_knots == 1:
        np.testing.assert_array_equal(actions, np.repeat(actions[:1], 7, axis=0))
    if num_knots == 2:
        np.testing.assert_allclose(np.diff(actions[:, 0], n=2), 0.0, atol=1e-15)


def test_actions_clipped_to_bounds():
    sim = _Toy1D()
    sim.fit(None, None)
    cem = _make_cem(sim, iters=4)
    actions = cem.plan()
    assert (actions >= -1.5 - 1e-9).all()
    assert (actions <= 1.5 + 1e-9).all()


def test_action_dim_shape_mismatch_raises():
    sim = _Toy1D()
    with pytest.raises(ValueError, match="action_low/high must have shape"):
        CEM(
            sim,
            action_dim=2,
            action_low=np.array([-1.0]),  # wrong shape: should be (2,)
            action_high=np.array([1.0, 1.0]),
            horizon=5,
            population=10,
            iters=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"horizon": 0}, "horizon"),
        ({"population": 0}, "population"),
        ({"elite_frac": 0.0}, "elite_frac"),
        ({"iters": 0}, "iters"),
        ({"init_std_scale": -1.0}, "init_std_scale"),
        ({"action_smoothness_weight": -1.0}, "action_smoothness_weight"),
        ({"num_action_knots": 6}, "num_action_knots"),
    ],
)
def test_invalid_planner_options_raise(kwargs, message):
    options = {
        "action_dim": 1,
        "action_low": np.array([-1.0]),
        "action_high": np.array([1.0]),
        "horizon": 5,
        "population": 10,
        "elite_frac": 0.1,
        "iters": 1,
    }
    options.update(kwargs)
    with pytest.raises(ValueError, match=message):
        CEM(_Toy1D(), **options)
