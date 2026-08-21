"""DM Control Reacher session for the shared P2 MPC harness."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

import numpy as np


REACHER_ACTION_LOW = np.array([-1.0, -1.0], dtype=np.float64)
REACHER_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float64)
REACHER_WRIST_LIMIT = float(np.deg2rad(160.0))
REACHER_IK_ENDPOINT_TOLERANCE = 1.0e-5


def _select_reacher_goal_qpos(
    candidates: list[tuple[np.ndarray, float]],
    reference_qpos: np.ndarray,
    *,
    endpoint_error_tolerance: float = REACHER_IK_ENDPOINT_TOLERANCE,
    wrist_limit: float = REACHER_WRIST_LIMIT,
) -> tuple[np.ndarray, float]:
    """Choose a near-optimal IK pose on the wrist branch nearest the start."""
    reference = np.asarray(reference_qpos, dtype=np.float64).reshape(-1)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_qpos must be a finite shape-(2,) vector")
    tolerance = float(endpoint_error_tolerance)
    limit = float(wrist_limit)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("endpoint_error_tolerance must be finite and non-negative")
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("wrist_limit must be finite and positive")

    feasible: list[tuple[np.ndarray, float]] = []
    for raw_qpos, raw_error in candidates:
        qpos = np.asarray(raw_qpos, dtype=np.float64).reshape(-1)
        error = float(raw_error)
        if qpos.shape != (2,) or not np.all(np.isfinite(qpos)) or not np.isfinite(error):
            continue
        canonical = np.array(
            [
                reference[0]
                + np.arctan2(
                    np.sin(qpos[0] - reference[0]),
                    np.cos(qpos[0] - reference[0]),
                ),
                np.arctan2(np.sin(qpos[1]), np.cos(qpos[1])),
            ],
            dtype=np.float64,
        )
        if abs(float(canonical[1])) <= limit + 1.0e-9:
            feasible.append((canonical, error))
    if not feasible:
        raise ValueError("IK produced no candidate within the Reacher wrist bounds")

    best_error = min(error for _, error in feasible)
    near_optimal = [
        (qpos, error)
        for qpos, error in feasible
        if error <= best_error + tolerance
    ]
    reference_wrist = float(np.clip(reference[1], -limit, limit))

    def branch_distance(item: tuple[np.ndarray, float]) -> tuple[float, float]:
        qpos, error = item
        shoulder_delta = np.arctan2(
            np.sin(qpos[0] - reference[0]),
            np.cos(qpos[0] - reference[0]),
        )
        wrist_delta = float(qpos[1] - reference_wrist)
        return float(shoulder_delta**2 + wrist_delta**2), error

    selected_qpos, selected_error = min(near_optimal, key=branch_distance)
    return selected_qpos.copy(), float(selected_error)


def _validated_action_repeat(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"Reacher action_repeat must be a positive integer, got {value!r}")
    action_repeat = int(value)
    if action_repeat <= 0:
        raise ValueError(f"Reacher action_repeat must be positive, got {action_repeat}")
    return action_repeat


class ReacherDMControlSession:
    """In-process ``dm_control`` Reacher wrapper exposing start/step/close."""

    def __init__(
        self,
        *,
        task: str = "hard",
        sim_frame_size: tuple[int, int] = (224, 224),
        goal_image_mode: str = "goal_state",
        manifest_case: dict[str, Any] | None = None,
        action_repeat: int = 2,
    ) -> None:
        if goal_image_mode not in {"goal_state", "start"}:
            raise ValueError(f"unsupported Reacher goal_image_mode={goal_image_mode!r}")
        selected_action_repeat: Any = action_repeat
        if manifest_case is not None and manifest_case.get("action_contract") is not None:
            action_contract = manifest_case["action_contract"]
            if not isinstance(action_contract, Mapping):
                raise ValueError("manifest_case.action_contract must be an object")
            if action_contract.get("action_repeat") is not None:
                selected_action_repeat = action_contract["action_repeat"]
        self.task = task
        self.sim_frame_size = sim_frame_size
        self.goal_image_mode = goal_image_mode
        self.manifest_case = manifest_case
        self.action_repeat = _validated_action_repeat(selected_action_repeat)
        self._env = None
        self._goal_image = None
        self._target_radius = float("nan")
        self._goal_ik_error = float("nan")
        self._goal_qpos = None

    def _render(self) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("call start() before rendering")
        width, height = self.sim_frame_size
        frame = self._env.physics.render(height=int(height), width=int(width), camera_id=0)
        return np.asarray(frame, dtype=np.uint8)

    def _info(self) -> dict[str, Any]:
        if self._env is None:
            return {}
        physics = self._env.physics
        target = physics.named.data.geom_xpos["target", :2].copy()
        finger = physics.named.data.geom_xpos["finger", :2].copy()
        qpos = physics.data.qpos.copy()
        qvel = physics.data.qvel.copy()
        distance = float(np.linalg.norm(finger - target))
        goal_qpos = None if self._goal_qpos is None else self._goal_qpos.copy()
        qpos_abs_error = None
        qpos_max_abs_error = None
        qpos_match_success = None
        if goal_qpos is not None:
            qpos_abs_error = np.abs(qpos - goal_qpos)
            qpos_max_abs_error = float(np.max(qpos_abs_error))
            qpos_match_success = bool(np.all(qpos_abs_error < 0.05))
        return {
            "target_pos": target,
            "finger_pos": finger,
            "qpos": qpos,
            "qvel": qvel,
            "goal_qpos": goal_qpos,
            "qpos_abs_error": qpos_abs_error,
            "qpos_max_abs_error": qpos_max_abs_error,
            "qpos_match_success": qpos_match_success,
            "distance_to_target": distance,
            "target_radius": self._target_radius,
            "goal_ik_error": self._goal_ik_error,
            "success": bool(distance <= self._target_radius),
        }

    def _finger_to_target_error(self, qpos: np.ndarray, target: np.ndarray) -> float:
        physics = self._env.physics
        physics.data.qpos[:] = qpos
        physics.data.qvel[:] = 0.0
        physics.forward()
        finger = physics.named.data.geom_xpos["finger", :2]
        return float(np.linalg.norm(finger - target))

    def _solve_goal_qpos(self) -> np.ndarray:
        """Find a two-joint Reacher pose whose fingertip lies on the target."""
        physics = self._env.physics
        target = physics.named.data.geom_xpos["target", :2].copy()
        current_qpos = physics.data.qpos.copy()

        starts = [
            current_qpos,
            np.zeros_like(current_qpos),
            np.array([0.0, np.pi / 2.0], dtype=np.float64),
            np.array([np.pi / 2.0, -np.pi / 2.0], dtype=np.float64),
            np.array([-np.pi / 2.0, np.pi / 2.0], dtype=np.float64),
            np.array([np.pi, -np.pi / 2.0], dtype=np.float64),
            np.array([-np.pi, np.pi / 2.0], dtype=np.float64),
        ]

        candidates = [
            (
                current_qpos.copy(),
                self._finger_to_target_error(current_qpos, target),
            )
        ]

        try:
            from scipy.optimize import minimize

            for start in starts:
                result = minimize(
                    lambda q: self._finger_to_target_error(np.asarray(q), target),
                    np.asarray(start, dtype=np.float64),
                    method="Nelder-Mead",
                    options={"maxiter": 400, "xatol": 1e-6, "fatol": 1e-6},
                )
                candidate = np.asarray(result.x, dtype=np.float64)
                candidates.append(
                    (
                        candidate.copy(),
                        self._finger_to_target_error(candidate, target),
                    )
                )
        except Exception:
            # Fallback for environments without scipy: coarse grid then local
            # coordinate search. This is only used to render image_B, not control.
            best_qpos, best_error = min(candidates, key=lambda item: item[1])
            grid = np.linspace(-np.pi, np.pi, 33)
            for q0 in grid:
                for q1 in grid:
                    q = np.array([q0, q1], dtype=np.float64)
                    err = self._finger_to_target_error(q, target)
                    candidates.append((q.copy(), err))
                    if err < best_error:
                        best_error = err
                        best_qpos = q.copy()
            step = 0.25
            while step > 1e-3:
                improved = False
                for dim in range(best_qpos.size):
                    for sign in (-1.0, 1.0):
                        q = best_qpos.copy()
                        q[dim] += sign * step
                        err = self._finger_to_target_error(q, target)
                        if err < best_error:
                            best_error = err
                            best_qpos = q.copy()
                            candidates.append((q.copy(), err))
                            improved = True
                if not improved:
                    step *= 0.5

        goal_qpos, goal_error = _select_reacher_goal_qpos(
            candidates,
            current_qpos,
        )
        self._goal_ik_error = goal_error
        return goal_qpos

    def _solve_goal_qpos_preserving_state(self) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("call start() before solving a goal qpos")
        physics = self._env.physics
        saved_qpos = physics.data.qpos.copy()
        saved_qvel = physics.data.qvel.copy()
        try:
            return self._solve_goal_qpos()
        finally:
            physics.data.qpos[:] = saved_qpos
            physics.data.qvel[:] = saved_qvel
            physics.forward()

    def _render_goal_state(self) -> np.ndarray:
        """Render image_B: Reacher with fingertip placed on the red target."""
        if self._env is None:
            raise RuntimeError("call start() before rendering a goal state")
        physics = self._env.physics
        saved_qpos = physics.data.qpos.copy()
        saved_qvel = physics.data.qvel.copy()
        try:
            goal_qpos = self._solve_goal_qpos()
            self._goal_qpos = goal_qpos.copy()
            physics.data.qpos[:] = goal_qpos
            physics.data.qvel[:] = 0.0
            physics.forward()
            return self._render()
        finally:
            physics.data.qpos[:] = saved_qpos
            physics.data.qvel[:] = saved_qvel
            physics.forward()

    def _render_goal_qpos(self, goal_qpos: np.ndarray) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("call start() before rendering a manifest goal state")
        physics = self._env.physics
        saved_qpos = physics.data.qpos.copy()
        saved_qvel = physics.data.qvel.copy()
        try:
            self._goal_qpos = np.asarray(goal_qpos, dtype=np.float64).copy()
            physics.data.qpos[:] = self._goal_qpos
            physics.data.qvel[:] = 0.0
            physics.forward()
            return self._render()
        finally:
            physics.data.qpos[:] = saved_qpos
            physics.data.qvel[:] = saved_qvel
            physics.forward()

    def _apply_manifest_start(self) -> None:
        if self._env is None or self.manifest_case is None:
            return
        physics = self._env.physics
        start = dict(self.manifest_case.get("start") or {})
        if start.get("qpos") is not None:
            physics.data.qpos[:] = np.asarray(start["qpos"], dtype=np.float64)
        if start.get("qvel") is not None:
            physics.data.qvel[:] = np.asarray(start["qvel"], dtype=np.float64)
        else:
            physics.data.qvel[:] = 0.0
        physics.forward()

    def start(self, seed: Optional[int] = None) -> dict[str, Any]:
        from dm_control import suite

        self.close()
        if self.manifest_case is not None and self.manifest_case.get("env_seed") is not None:
            seed = int(self.manifest_case["env_seed"])
        self._env = suite.load(
            domain_name="reacher",
            task_name=self.task,
            task_kwargs={"random": seed},
        )
        timestep = self._env.reset()
        self._target_radius = float(self._env.physics.named.model.geom_size["target", 0])
        self._apply_manifest_start()
        frame = self._render()
        goal = dict(self.manifest_case.get("goal") or {}) if self.manifest_case is not None else {}
        if goal.get("qpos") is not None:
            self._goal_image = self._render_goal_qpos(np.asarray(goal["qpos"], dtype=np.float64))
        elif self.goal_image_mode == "start":
            # Diagnostic mode: keep the red target marker visible in image_B
            # without showing a final arm pose.
            self._goal_qpos = self._solve_goal_qpos_preserving_state()
            self._goal_image = frame.copy()
        else:
            # image_B should be a final-state target frame: render a temporary
            # IK pose with the fingertip on the red target, then restore env.
            self._goal_image = self._render_goal_state()
        return {
            "image": frame,
            "image_raw": frame,
            "goal_image": self._goal_image,
            "goal_image_raw": self._goal_image,
            "info": {
                **self._info(),
                "manifest_case_id": None if self.manifest_case is None else self.manifest_case.get("case_id"),
                "action_repeat": self.action_repeat,
            },
            "reward": float(timestep.reward or 0.0),
            "done": bool(timestep.last()),
            "truncated": False,
        }

    def step(self, action_sim) -> dict[str, Any]:
        if self._env is None:
            raise RuntimeError("call start() before step()")
        action = np.asarray(action_sim, dtype=np.float64).reshape(2)
        action = np.clip(action, REACHER_ACTION_LOW, REACHER_ACTION_HIGH)
        substeps: list[dict[str, Any]] = []
        timestep = None
        info: dict[str, Any] = {}
        for substep_index in range(self.action_repeat):
            timestep = self._env.step(action)
            info = self._info()
            success = bool(info.get("success", False))
            substeps.append(
                {
                    **info,
                    "substep_index": substep_index,
                    "reward": float(timestep.reward or 0.0),
                    "env_last": bool(timestep.last()),
                }
            )
            if success or timestep.last():
                break
        if timestep is None:  # pragma: no cover - guarded by constructor validation.
            raise AssertionError("Reacher action_repeat executed no environment steps")
        frame = self._render()
        if self.manifest_case is not None:
            info["manifest_case_id"] = self.manifest_case.get("case_id")
        info["action_repeat"] = self.action_repeat
        info["executed_action_repeat"] = len(substeps)
        info["substeps"] = substeps
        success = bool(info.get("success", False))
        return {
            "image": frame,
            "image_raw": frame,
            "goal_image": self._goal_image,
            "goal_image_raw": self._goal_image,
            "info": info,
            "reward": float(timestep.reward or 0.0),
            "done": bool(success),
            "truncated": bool(timestep.last() and not success),
        }

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

