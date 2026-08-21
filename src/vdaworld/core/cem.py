"""Cross-Entropy Method planner for action-conditioned simulators.

Plain Gaussian CEM: at each iteration, sample ``population`` action sequences
from a per-step Gaussian, roll each through the simulator, keep the
``elite_frac`` lowest-cost sequences, refit the Gaussian. Returns the
single best action sequence ever seen across all iterations.

The simulator must subclass :class:`~vdaworld.core.simulator.ActionConditionedSimulatorBase`
and follow the snapshot/restore contract: per-rollout mutable state lives in
``self.state``; the planner snapshots via ``copy.deepcopy`` and restores by
assignment.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from vdaworld.core.simulator import ActionConditionedSimulatorBase


@dataclass
class CEMHistoryEntry:
    iteration: int
    mean_cost: float
    std_cost: float
    best_cost: float
    elite_mean_cost: float


class CEM:
    """Cross-Entropy Method planner.

    Usage::

        sim.fit(image_A, image_B)
        cem = CEM(sim, action_dim=2,
                  action_low=np.array([-1.5, -1.5]),
                  action_high=np.array([1.5, 1.5]))
        actions = cem.plan()       # (horizon, action_dim)
        history = cem.history       # list[CEMHistoryEntry] per iter

    Args:
        sim:            Fitted ActionConditionedSimulatorBase instance.
        action_dim:     Dimensionality of each action vector.
        action_low:     (action_dim,) lower clip bounds.
        action_high:    (action_dim,) upper clip bounds.
        horizon:        Number of action steps per rollout.
        population:     Samples per iteration.
        elite_frac:     Top fraction kept for refitting.
        iters:          Number of refitting iterations.
        init_std_scale: Initial Gaussian std as a fraction of action half-range.
        seed:           Optional RNG seed.
        use_stage_cost: If true, accumulate ``sim.stage_cost(action, t)`` after
                        every simulator update.
        action_smoothness_weight:
                        Weight on the mean squared consecutive-action change,
                        normalized by each action dimension's range.
        num_action_knots:
                        Optional number of temporally interpolated action knots.
                        ``None`` or ``horizon`` samples every action as before.
    """

    def __init__(
        self,
        sim: ActionConditionedSimulatorBase,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        horizon: int = 10,
        population: int = 200,
        elite_frac: float = 0.1,
        iters: int = 10,
        init_std_scale: float = 0.5,
        seed: int | None = None,
        use_stage_cost: bool = False,
        action_smoothness_weight: float = 0.0,
        num_action_knots: int | None = None,
    ) -> None:
        if isinstance(action_dim, bool) or not isinstance(action_dim, (int, np.integer)):
            raise ValueError(f"action_dim must be a positive integer; got {action_dim!r}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer; got {action_dim!r}")
        self.sim = sim
        self.action_dim = int(action_dim)
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)
        if self.action_low.shape != (action_dim,) or self.action_high.shape != (action_dim,):
            raise ValueError(
                f"action_low/high must have shape ({action_dim},); got "
                f"{self.action_low.shape}/{self.action_high.shape}"
            )
        if not np.all(np.isfinite(self.action_low)) or not np.all(
            np.isfinite(self.action_high)
        ):
            raise ValueError("action_low/high must contain only finite values")
        if np.any(self.action_high <= self.action_low):
            raise ValueError("action_high must be greater than action_low in every dimension")
        self.horizon = self._positive_integer("horizon", horizon)
        self.population = self._positive_integer("population", population)
        self.elite_frac = self._finite_float("elite_frac", elite_frac)
        if not 0.0 < self.elite_frac <= 1.0:
            raise ValueError(f"elite_frac must be finite and in (0, 1]; got {elite_frac!r}")
        self.elite_n = max(1, int(round(self.population * self.elite_frac)))
        self.iters = self._positive_integer("iters", iters)
        self.init_std_scale = self._finite_float("init_std_scale", init_std_scale)
        if self.init_std_scale < 0.0:
            raise ValueError(
                f"init_std_scale must be finite and non-negative; got {init_std_scale!r}"
            )
        if not isinstance(use_stage_cost, (bool, np.bool_)):
            raise ValueError(f"use_stage_cost must be a bool; got {use_stage_cost!r}")
        self.use_stage_cost = bool(use_stage_cost)
        self.action_smoothness_weight = self._finite_float(
            "action_smoothness_weight", action_smoothness_weight
        )
        if self.action_smoothness_weight < 0.0:
            raise ValueError(
                "action_smoothness_weight must be finite and non-negative; "
                f"got {action_smoothness_weight!r}"
            )
        if num_action_knots is None:
            self.num_action_knots = None
        else:
            self.num_action_knots = self._positive_integer(
                "num_action_knots", num_action_knots
            )
            if self.num_action_knots > self.horizon:
                raise ValueError(
                    "num_action_knots must be <= horizon; "
                    f"got {self.num_action_knots} > {self.horizon}"
                )
        self.rng = np.random.default_rng(seed)
        self.history: list[CEMHistoryEntry] = []

    @staticmethod
    def _positive_integer(name: str, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
        return value

    @staticmethod
    def _finite_float(name: str, value) -> float:
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
            raise ValueError(f"{name} must be a finite scalar; got {value!r}")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a finite scalar; got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be a finite scalar; got {value!r}")
        return value

    def _interpolate_knots(self, knots: np.ndarray) -> np.ndarray:
        """Linearly interpolate ``(population, knots, action_dim)`` samples."""
        if knots.shape[1] == 1:
            return np.repeat(knots, self.horizon, axis=1)
        knot_times = np.linspace(0.0, float(self.horizon - 1), knots.shape[1])
        action_times = np.arange(self.horizon, dtype=np.float64)
        right = np.searchsorted(knot_times, action_times, side="left")
        right = np.clip(right, 1, knots.shape[1] - 1)
        left = right - 1
        weight = (action_times - knot_times[left]) / (
            knot_times[right] - knot_times[left]
        )
        return (
            knots[:, left, :] * (1.0 - weight[None, :, None])
            + knots[:, right, :] * weight[None, :, None]
        )

    @staticmethod
    def _finite_scalar_stage_cost(value, step_index: int) -> float:
        array = np.asarray(value)
        if array.ndim != 0:
            raise ValueError(
                "sim.stage_cost(action, step_index) must return a finite scalar; "
                f"got shape {array.shape} at step {step_index}"
            )
        try:
            cost = float(array)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sim.stage_cost(action, step_index) must return a finite scalar; "
                f"got {value!r} at step {step_index}"
            ) from exc
        if not np.isfinite(cost):
            raise ValueError(
                "sim.stage_cost(action, step_index) must return a finite scalar; "
                f"got {cost!r} at step {step_index}"
            )
        return cost

    def _smoothness_cost(self, actions: np.ndarray) -> float:
        if self.action_smoothness_weight == 0.0 or self.horizon < 2:
            return 0.0
        action_range = self.action_high - self.action_low
        normalized_deltas = np.diff(actions, axis=0) / action_range
        return self.action_smoothness_weight * float(np.mean(normalized_deltas**2))

    def plan(self) -> np.ndarray:
        if self.sim.state is None:
            raise RuntimeError(
                "CEM.plan() called before sim.fit(A, B); sim.state is None."
            )

        mid = (self.action_low + self.action_high) / 2.0
        half_range = (self.action_high - self.action_low) / 2.0
        sample_steps = self.num_action_knots or self.horizon
        mean = np.tile(mid, (sample_steps, 1))
        std = np.tile(half_range * self.init_std_scale, (sample_steps, 1))

        saved_state = copy.deepcopy(self.sim.state)

        best_actions = np.tile(mid, (self.horizon, 1))
        best_cost = float("inf")

        try:
            for it in range(self.iters):
                parameter_samples = self.rng.normal(
                    loc=mean,
                    scale=std,
                    size=(self.population, sample_steps, self.action_dim),
                )
                parameter_samples = np.clip(
                    parameter_samples, self.action_low, self.action_high
                )
                samples = (
                    parameter_samples
                    if sample_steps == self.horizon
                    else self._interpolate_knots(parameter_samples)
                )

                costs = np.empty(self.population, dtype=np.float64)
                for i in range(self.population):
                    self.sim.state = copy.deepcopy(saved_state)
                    stage_cost = 0.0
                    for t in range(self.horizon):
                        self.sim.update(samples[i, t])
                        if self.use_stage_cost:
                            stage_cost += self._finite_scalar_stage_cost(
                                self.sim.stage_cost(samples[i, t], t), t
                            )
                            if not np.isfinite(stage_cost):
                                raise ValueError(
                                    "accumulated simulator stage cost must remain "
                                    f"finite; overflowed at step {t}"
                                )
                    costs[i] = (
                        float(self.sim.planning_objective())
                        + stage_cost
                        + self._smoothness_cost(samples[i])
                    )

                elite_idx = np.argsort(costs)[: self.elite_n]
                elites = parameter_samples[elite_idx]
                mean = elites.mean(axis=0)
                std = elites.std(axis=0) + 1e-6

                iter_best = float(costs[elite_idx[0]])
                if iter_best < best_cost:
                    best_cost = iter_best
                    best_actions = samples[elite_idx[0]].copy()

                self.history.append(
                    CEMHistoryEntry(
                        iteration=it,
                        mean_cost=float(costs.mean()),
                        std_cost=float(costs.std()),
                        best_cost=iter_best,
                        elite_mean_cost=float(costs[elite_idx].mean()),
                    )
                )
        finally:
            # Restore sim state even if a simulator objective raises.
            self.sim.state = copy.deepcopy(saved_state)
        return best_actions
