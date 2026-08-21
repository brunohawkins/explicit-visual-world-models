"""Unit tests for ActionConditionedSimulatorBase contract."""

from __future__ import annotations

import numpy as np
import pytest

from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase


class _MinimalSubclass(ActionConditionedSimulatorBase):
    def fit(self, image_A, image_B):
        self.state = {"x": 0.0}
        self.target_state = {"x": 1.0}

    def update(self, a):
        self.state["x"] += float(np.asarray(a).item())

    def loss_to_target(self):
        return abs(self.state["x"] - self.target_state["x"])

    def render_frame(self):
        return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)


class _TerminalCostSubclass(ActionConditionedSimulatorBase):
    def fit(self, image_A, image_B):
        self.state = {"x": 0.0}
        self.target_state = {"x": 1.0}

    def update(self, a):
        self.state["x"] += float(np.asarray(a).item())

    def terminal_cost(self):
        return abs(self.state["x"] - self.target_state["x"])

    def render_frame(self):
        return np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)


def test_inherits_from_simulator_base():
    assert issubclass(ActionConditionedSimulatorBase, SimulatorBase)


def test_base_class_raises_not_implemented():
    sim = ActionConditionedSimulatorBase(frame_size=(65, 65))
    with pytest.raises(NotImplementedError):
        sim.update(np.zeros(2))
    with pytest.raises(NotImplementedError):
        sim.fit(np.zeros((65, 65, 3), dtype=np.uint8), np.zeros((65, 65, 3), dtype=np.uint8))
    with pytest.raises(NotImplementedError):
        sim.loss_to_target()
    with pytest.raises(NotImplementedError):
        sim.terminal_cost()
    with pytest.raises(NotImplementedError):
        sim.planning_objective()


def test_state_defaults_to_none():
    sim = ActionConditionedSimulatorBase(frame_size=(65, 65))
    assert sim.state is None
    assert sim.target_state is None


def test_default_stage_cost_is_zero_and_side_effect_free():
    sim = _MinimalSubclass(frame_size=(65, 65))
    sim.fit(None, None)
    state_before = {"x": sim.state["x"]}

    assert sim.stage_cost(np.array([0.25]), 3) == 0.0
    assert sim.state == state_before


def test_update_simulation_and_next_are_disabled():
    sim = _MinimalSubclass(frame_size=(65, 65))
    sim.fit(None, None)
    with pytest.raises(NotImplementedError):
        sim.update_simulation(0.1)
    with pytest.raises(NotImplementedError):
        next(sim)


def test_minimal_subclass_roundtrip():
    sim = _MinimalSubclass(frame_size=(65, 65))
    sim.fit(None, None)
    assert sim.state == {"x": 0.0}
    assert sim.target_state == {"x": 1.0}
    assert sim.loss_to_target() == pytest.approx(1.0)
    sim.update(np.array([0.4]))
    assert sim.loss_to_target() == pytest.approx(0.6)
    frame = sim.render_frame()
    assert frame.shape == (65, 65, 3)
    assert frame.dtype == np.uint8


def test_terminal_cost_is_primary_planning_objective():
    sim = _TerminalCostSubclass(frame_size=(65, 65))
    sim.fit(None, None)
    assert sim.terminal_cost() == pytest.approx(1.0)
    assert sim.loss_to_target() == pytest.approx(1.0)
    assert sim.planning_objective() == pytest.approx(1.0)
    sim.update(np.array([0.4]))
    assert sim.planning_objective() == pytest.approx(0.6)
