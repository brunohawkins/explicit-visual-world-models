"""PlanningCriticSandbox: CriticSandbox extension for action-conditioned codegen.

Differences from :class:`vdaworld.core.critic_toolbox.CriticSandbox`:

* Constructor takes ``start_image_path`` and ``goal_image_path`` (two images)
  instead of a single ``input_image_path``, plus a ``dataset_dir`` pointing
  at the training trajectories.
* ``run_simulation`` subprocesses
  :mod:`vdaworld.core.action_conditioned_sandbox_runner` with both image
  paths, so the generated simulator is exercised through its
  ``fit(image_A, image_B)`` API + ``update(a)``-driven rollout.
* Adds dataset and validation tools the VLM can call during generation:

  * :meth:`read_trajectory` — frame-list + actions summary for one training trajectory.
  * :meth:`view_image` — load one PNG from a training trajectory.
  * :meth:`view_action` — return one action vector from a training trajectory.
  * :meth:`validate_action_realizability` — offline one-step realizability probe.
  * :meth:`validate_goal_invariance` — offline fixed-goal target drift probe.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import py_compile
import ast
import base64
import binascii
import io
import inspect
import re
from itertools import product
from pathlib import Path
from typing import Any, Union

import numpy as np
from PIL import Image, ImageDraw

from vdaworld.core import dataset_tools as _dt
from vdaworld.core.api import _draw_segment_overlay
from vdaworld.core.critic_toolbox import CriticSandbox
from vdaworld.utils.error_handling import AbortLoopError

logger = logging.getLogger(__name__)

_RUNNER_MODULE = "vdaworld.core.action_conditioned_sandbox_runner"


def _generic_fit_quality_color_masks(
    image: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return coarse advisory colour proxies used by the fit-quality gate."""
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"image must have shape (H, W, >=3), got {array.shape}")
    rgb = array[:, :, :3].astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    corner = np.concatenate(
        [
            rgb[:5, :5].reshape(-1, 3),
            rgb[:5, -5:].reshape(-1, 3),
            rgb[-5:, :5].reshape(-1, 3),
            rgb[-5:, -5:].reshape(-1, 3),
        ],
        axis=0,
    )
    background = np.median(corner, axis=0)
    foreground = np.linalg.norm(rgb - background.reshape(1, 1, 3), axis=2) > 35.0
    maximum = np.max(rgb, axis=2)
    minimum = np.min(rgb, axis=2)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
    masks = {
        "red": (r > g + 35) & (r > b + 35) & (r > 80),
        "orange": (r > 130) & (g > 45) & (g < r - 25) & (b < g + 20),
        "green": (g > r + 35) & (g > b + 35) & (g > 80),
        "blue": (
            (b > 80)
            & (b - np.maximum(r, g) >= 30)
            & (saturation >= 0.30)
        ),
        "yellow": (r > 120) & (g > 120) & (b < 120) & (np.abs(r - g) < 80),
        "purple": (r > g + 20) & (b > g + 20) & (r > 50) & (b > 50),
    }
    return masks, foreground


def summarize_reduction_ratios(
    per_traj: dict[int, float | None],
) -> dict[str, Any]:
    """Benchmark-agnostic distribution stats for reduction_ratio values.

    ``None`` means crash/degenerate and counts against trajectory fractions. The
    numeric summaries are over trajectories that produced a ratio.
    """
    total = len(per_traj)
    vals = [
        float(r)
        for _, r in sorted(per_traj.items())
        if r is not None and np.isfinite(float(r))
    ]
    n_valid = len(vals)
    n_failed = total - n_valid
    if vals:
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        p75 = float(np.percentile(arr, 75))
        worst = float(np.max(arr))
        best = float(np.min(arr))
    else:
        mean = median = p75 = worst = best = None

    denom = float(total) if total else float("nan")
    frac_below_1 = (
        float(sum(1 for v in vals if v < 1.0) / denom) if total else None
    )
    frac_below_05 = (
        float(sum(1 for v in vals if v < 0.5) / denom) if total else None
    )
    return {
        "mean": mean,
        "median": median,
        "p75": p75,
        "worst": worst,
        "best": best,
        "n_trajectories": total,
        "n_valid": n_valid,
        "n_failed": n_failed,
        "frac_below_1.0": frac_below_1,
        "frac_below_0.5": frac_below_05,
    }


def candidate_gate_criteria(stats: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Uniform no-op-anchored candidate criteria for reporting only."""
    crash_ok = int(stats.get("n_failed", 0)) == 0
    worst = stats.get("worst")
    mean = stats.get("mean")
    median = stats.get("median")
    p75 = stats.get("p75")
    frac_below_1 = stats.get("frac_below_1.0")

    def _ok(value) -> bool:
        return value is not None and np.isfinite(float(value))

    return {
        "C1": {
            "passed": bool(crash_ok and _ok(worst) and float(worst) <= 0.5),
            "description": "worst<=0.5 (current/enforced)",
        },
        "C2": {
            "passed": bool(
                crash_ok
                and _ok(worst)
                and _ok(mean)
                and float(worst) < 1.0
                and float(mean) <= 0.5
            ),
            "description": "worst<1.0 AND mean<=0.5",
        },
        "C3": {
            "passed": bool(
                crash_ok
                and _ok(median)
                and _ok(frac_below_1)
                and float(median) <= 0.5
                and float(frac_below_1) >= 0.8
            ),
            "description": "median<=0.5 AND frac_below_1.0>=0.8",
        },
        "C4": {
            "passed": bool(crash_ok and _ok(p75) and float(p75) <= 0.5),
            "description": "p75<=0.5",
        },
    }


class PlanningCriticSandbox(CriticSandbox):
    """Sandbox for action-conditioned simulator generation.

    Args:
        start_image_path: Path to the start-state PNG (used as ``image_A`` in fit).
        goal_image_path:  Path to the goal-state PNG (used as ``image_B`` in fit).
        dataset_dir:      Root of the training-trajectory dataset.
        All other args are forwarded to :class:`CriticSandbox`.
    """

    def __init__(
        self,
        code: str,
        fps: int,
        n_frames: int,
        frame_size: tuple[int, int],
        start_image_path: str,
        goal_image_path: str,
        dataset_dir: Union[str, Path],
        simulator_class_name: str,
        sandbox_dir: str,
        tool_calls_log_dir: str | None = None,
        cache_dir: str | None = None,
        world_api_log_dir: str | None = None,
        no_api: bool = False,
        auto_validate_trajectories: list[int] | None = None,
        gate_trajectories: list[int] | None = None,
        deployed_action_bounds: tuple[np.ndarray, np.ndarray] | None = None,
        gate_max_worst_ratio: float | None = 1.25,
        gate_realizability_margin: float = 1.25,
        gate_require_action_realizability: bool = True,
        gate_require_goal_invariance: bool = True,
        gate_require_cem_suite: bool = False,
        gate_require_p2_safety: bool = False,
        gate_goal_invariance_abs_tol: float = 3.0,
        gate_goal_invariance_rel_tol: float = 0.02,
        gate_require_target_independence: bool = True,
        gate_target_independence_abs_tol: float = 1.0,
        gate_target_independence_rel_tol: float = 0.02,
        gate_null_action_steps: int = 20,
        gate_null_action_cost_drop_frac: float = 0.15,
        gate_require_state_consistency: bool = False,
        gate_require_fit_quality: bool = False,
        gate_fit_quality_max_rmse: float = 0.30,
        gate_fit_quality_max_scene_frac: float = 0.35,
        gate_fit_quality_mode: str = "rmse_plus_foreground",
        gate_state_consistency_skip_latent: bool = True,
        keep_best: bool = True,
        keep_best_include_cem: bool = False,
        deployment_goal_mode: str | None = None,
        gate_ratio_advisory: bool = False,
    ) -> None:
        # Parent uses input_image_path; we pass start_image_path so existing
        # bookkeeping (e.g. read_code) still works without surprise.
        super().__init__(
            code=code,
            fps=fps,
            n_frames=n_frames,
            frame_size=frame_size,
            input_image_path=start_image_path,
            simulator_class_name=simulator_class_name,
            sandbox_dir=sandbox_dir,
            tool_calls_log_dir=tool_calls_log_dir,
            cache_dir=cache_dir,
            world_api_log_dir=world_api_log_dir,
            no_api=no_api,
        )
        self._start_image_path = start_image_path
        self._goal_image_path = goal_image_path
        self._dataset_dir = Path(dataset_dir)
        # Trajectories the post-code-change auto-validate feedback runs on. A
        # SINGLE trajectory (the default) lets the VLM overfit it; passing a
        # diverse spread of TRAINING trajectories (excluding the held-out eval
        # trajectory) forces the feedback to reflect generalisation. This is
        # feedback, not selection — the VLM still decides what to ship.
        self._auto_validate_trajs = (
            list(auto_validate_trajectories)
            if auto_validate_trajectories
            else [self._AUTO_VALIDATE_TRAJ]
        )
        self._auto_visual_code_changes = 0
        self._auto_visual_last_change = -100
        self._auto_visual_runs = 0
        # Full training set for validate_all_trajectories / final_fidelity_gate
        # (all trajectories except held-out eval — threaded from smoke/batch scripts).
        self._gate_trajectories = (
            list(gate_trajectories) if gate_trajectories else None
        )
        self._deployed_action_bounds: tuple[np.ndarray, np.ndarray] | None = None
        if deployed_action_bounds is not None:
            low, high = deployed_action_bounds
            low_arr = np.asarray(low, dtype=np.float64).reshape(-1)
            high_arr = np.asarray(high, dtype=np.float64).reshape(-1)
            if low_arr.shape == high_arr.shape and low_arr.size > 0:
                self._deployed_action_bounds = (low_arr.copy(), high_arr.copy())
        self._gate_max_worst_ratio = (
            float(gate_max_worst_ratio)
            if gate_max_worst_ratio is not None
            else None
        )
        self._gate_realizability_margin = float(gate_realizability_margin)
        self._gate_require_action_realizability = bool(
            gate_require_action_realizability
        )
        self._gate_require_goal_invariance = bool(gate_require_goal_invariance)
        self._gate_require_cem_suite = bool(gate_require_cem_suite)
        self._gate_require_p2_safety = bool(gate_require_p2_safety)
        self._gate_goal_invariance_abs_tol = float(gate_goal_invariance_abs_tol)
        self._gate_goal_invariance_rel_tol = float(gate_goal_invariance_rel_tol)
        self._gate_require_target_independence = bool(gate_require_target_independence)
        self._gate_target_independence_abs_tol = float(gate_target_independence_abs_tol)
        self._gate_target_independence_rel_tol = float(gate_target_independence_rel_tol)
        self._gate_null_action_steps = int(gate_null_action_steps)
        self._gate_null_action_cost_drop_frac = float(gate_null_action_cost_drop_frac)
        self._gate_require_state_consistency = bool(gate_require_state_consistency)
        self._gate_require_fit_quality = bool(gate_require_fit_quality)
        self._gate_fit_quality_max_rmse = float(gate_fit_quality_max_rmse)
        self._gate_fit_quality_max_scene_frac = float(gate_fit_quality_max_scene_frac)
        mode = str(gate_fit_quality_mode or "rmse_plus_foreground").strip().lower()
        if mode not in {"all_labels", "primary_labels", "rmse_plus_foreground"}:
            raise ValueError(
                "gate_fit_quality_mode must be one of "
                "all_labels, primary_labels, rmse_plus_foreground; "
                f"got {gate_fit_quality_mode!r}"
            )
        self._gate_fit_quality_mode = mode
        self._gate_state_consistency_skip_latent = bool(
            gate_state_consistency_skip_latent
        )
        # Keep-best: quietly score every code version the VLM writes and keep
        # the one with the lowest full-training-set worst reduction_ratio so
        # a last-turn regression never ships (mechanism here; the restore
        # POLICY — only when the final gate fails — lives in the generator).
        self._keep_best_enabled = bool(keep_best)
        # When CEM is required, optionally rerank only the top few checkpoints
        # with a cheap CEM stress test immediately before a failed-final-gate
        # restore. This avoids running CEM after every VLM edit while still
        # preventing ratio-good but planner-dead restores.
        self._keep_best_include_cem = bool(keep_best_include_cem)
        self._keep_best_best: dict[str, Any] | None = None
        self._keep_best_last: dict[str, Any] | None = None
        self._keep_best_candidates: list[dict[str, Any]] = []
        self._keep_best_cem_record: dict[str, Any] | None = None
        self._keep_best_record: dict[str, Any] | None = None
        # Deployment goal contract: how image_B is constructed at P2 time.
        # "final" (or None) = a completed-task frame, matching training pairs;
        # "start" = the CURRENT observation (goal cues must be persistent
        # markers in the scene). Mirrors the P2 protocol config for this run.
        self._deployment_goal_mode = (
            str(deployment_goal_mode).strip().lower()
            if deployment_goal_mode
            else "final"
        )
        # Ratio-advisory gate: report the worst-ratio fidelity criterion in
        # every verdict/complaint but exclude it from pass/fail. Use when the
        # fixed bar is unreachable for a benchmark's dynamics (e.g. Push-T
        # contact: no draw has beaten 0.887 vs a 0.5 bar), so that the
        # planner-relevant criteria (CEM suite, deployment-goal, invariances)
        # drive selection instead of a constant failure.
        self._gate_ratio_advisory = bool(gate_ratio_advisory)

    def has_valid_simulator_class(self) -> bool:
        """Return True once the sandbox source defines the expected class."""
        try:
            source = Path(self._sandbox_path).read_text(encoding="utf-8")
        except OSError:
            return False
        if f"class {self._simulator_class_name}" not in source:
            return False
        try:
            py_compile.compile(self._sandbox_path, doraise=True)
        except py_compile.PyCompileError:
            return False
        return True

    def _fidelity_train_trajectories(self) -> list[int]:
        """Training trajectory indices for full-set fidelity sweeps."""
        if self._gate_trajectories is not None:
            return list(self._gate_trajectories)
        idxs: list[int] = []
        for p in sorted(self._dataset_dir.iterdir()):
            if not p.is_dir() or not p.name.startswith("trajectory_"):
                continue
            try:
                idxs.append(int(p.name.split("_", 1)[1]))
            except ValueError:
                continue
        return sorted(idxs)

    def _collect_training_actions(self) -> np.ndarray | None:
        """Stack all available training actions, or None if unavailable."""
        all_actions: list[np.ndarray] = []
        for idx in self._fidelity_train_trajectories():
            try:
                tr = _dt.read_trajectory(self._dataset_dir, int(idx))
            except (FileNotFoundError, ValueError):
                continue
            if tr["n_steps"] > 0:
                all_actions.append(np.asarray(tr["actions"], dtype=np.float64))
        if not all_actions:
            return None
        return np.concatenate(all_actions, axis=0)

    @staticmethod
    def _subprocess_env() -> dict[str, str]:
        """Environment for sandbox subprocesses with bounded numeric threads."""
        env = os.environ.copy()
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "OPENCV_FOR_THREADS_NUM",
        ):
            env[key] = "1"
        return env

    @staticmethod
    def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size < 2 or x.size != y.size:
            return None
        x0 = x - float(np.mean(x))
        y0 = y - float(np.mean(y))
        denom = float(np.linalg.norm(x0) * np.linalg.norm(y0))
        if denom <= 1e-12:
            return None
        return float(np.dot(x0, y0) / denom)

    @staticmethod
    def _linear_gain(x: np.ndarray, y: np.ndarray) -> float | None:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.size < 2 or x.size != y.size:
            return None
        x0 = x - float(np.mean(x))
        denom = float(np.dot(x0, x0))
        if denom <= 1e-12:
            return None
        return float(np.dot(x0, y - float(np.mean(y))) / denom)

    def _extract_simple_color_tracks(
        self,
        image: np.ndarray,
        *,
        min_area: int = 5,
    ) -> dict[str, dict[str, Any]]:
        """Task-agnostic visible-object centroids from simple colour masks.

        This is deliberately simple and deterministic. It is not a benchmark
        parser; it gives the VLM coarse measurement evidence (what moved with
        which action dimension) before it writes its own task-specific `fit()`.
        """
        from scipy import ndimage as _ndi

        img = np.asarray(image, dtype=np.uint8)
        if img.ndim != 3 or img.shape[2] < 3:
            return {}
        arr = img[:, :, :3].astype(np.float64)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        corner = np.concatenate(
            [
                arr[:5, :5].reshape(-1, 3),
                arr[:5, -5:].reshape(-1, 3),
                arr[-5:, :5].reshape(-1, 3),
                arr[-5:, -5:].reshape(-1, 3),
            ],
            axis=0,
        )
        bg = np.median(corner, axis=0)
        fg = np.linalg.norm(arr - bg.reshape(1, 1, 3), axis=2) > 35.0
        masks = {
            "red": (r > g + 70) & (r > b + 70) & (r > 120) & (g < 130),
            "orange": (
                (r > 160)
                & (g >= 80)
                & (g < 190)
                & (b < 130)
                & (r > g + 30)
                & (g > b + 20)
            ),
            "green": (g > r + 35) & (g > b + 35) & (g > 80),
            "blue": (b > r + 60) & (b > g + 40) & (b > 100),
            "yellow": (r > 120) & (g > 120) & (b < 120) & (np.abs(r - g) < 80),
            "purple": (r > g + 20) & (b > g + 20) & (r > 50) & (b > 50),
            "grey": (np.abs(r - g) < 30) & (np.abs(g - b) < 30) & fg,
            "foreground": fg,
        }
        tracks: dict[str, dict[str, Any]] = {}
        for label, mask in masks.items():
            source_area = int(np.count_nonzero(mask))
            if source_area < int(min_area):
                continue
            # A global colour mask can merge a moving object with a similarly
            # coloured cue or background. Track one coherent component rather
            # than the centroid of every matching pixel in the frame.
            labelled, n_components = _ndi.label(np.asarray(mask, dtype=bool))
            if int(n_components) <= 0:
                continue
            component_areas = np.bincount(labelled.reshape(-1))[1:]
            if component_areas.size == 0:
                continue
            component_id = int(np.argmax(component_areas)) + 1
            component = labelled == component_id
            ys, xs = np.where(component)
            area = int(xs.size)
            area_fraction = float(area / max(1, mask.size))
            # Near-full-frame masks are scene appearance, not trackable
            # objects. In Reacher the old "blue" track covered ~98% of pixels
            # and produced authoritative-looking but meaningless gains.
            if area < int(min_area) or area_fraction > 0.35:
                continue
            tracks[label] = {
                "centroid": [float(np.mean(xs)), float(np.mean(ys))],
                "area": area,
                "area_fraction": area_fraction,
                "source_area": source_area,
                "n_components": int(n_components),
            }
        return tracks

    def _measurement_trajectories(
        self,
        trajectory_indices: list[int] | None,
        max_trajectories: int,
    ) -> list[int]:
        if trajectory_indices is not None:
            return [int(t) for t in trajectory_indices][: max(1, int(max_trajectories))]
        trajs = self._fidelity_train_trajectories()
        if not trajs:
            return []
        n = max(1, min(int(max_trajectories), len(trajs)))
        idxs = np.linspace(0, len(trajs) - 1, n).round().astype(int)
        return [int(trajs[i]) for i in dict.fromkeys(idxs.tolist())]

    def _training_action_bounds(
        self,
        *,
        pad_fraction: float = 0.05,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Per-dim min/max action bounds from training, with optional padding."""
        stacked = self._collect_training_actions()
        if stacked is None or stacked.ndim != 2 or stacked.shape[0] == 0:
            return None
        per_dim_min = stacked.min(axis=0)
        per_dim_max = stacked.max(axis=0)
        per_dim_range = per_dim_max - per_dim_min
        if pad_fraction > 0:
            pad = np.where(per_dim_range > 0, pad_fraction * per_dim_range, pad_fraction)
            per_dim_min = per_dim_min - pad
            per_dim_max = per_dim_max + pad
        return per_dim_min.astype(np.float64), per_dim_max.astype(np.float64)

    def _resolve_action_bounds(
        self,
        action_dim: int,
        *,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ) -> tuple[list[float] | None, list[float] | None, str]:
        """Resolve action bounds (custom > deployed default > training fallback)."""
        if (action_low is None) ^ (action_high is None):
            return (None, None, "invalid: provide both action_low and action_high")

        if action_low is not None and action_high is not None:
            low = np.asarray(action_low, dtype=np.float64).reshape(-1)
            high = np.asarray(action_high, dtype=np.float64).reshape(-1)
            if low.shape != high.shape or low.size != action_dim:
                return (
                    None,
                    None,
                    "invalid: custom bounds dimensionality mismatch",
                )
            return (low.tolist(), high.tolist(), "custom")

        if self._deployed_action_bounds is not None:
            dep_low, dep_high = self._deployed_action_bounds
            if dep_low.size == action_dim and dep_high.size == action_dim:
                return (dep_low.tolist(), dep_high.tolist(), "deployed")

        training = self._training_action_bounds(pad_fraction=0.0)
        if training is None:
            return (None, None, "invalid: no usable training actions")
        low, high = training
        if low.size != action_dim or high.size != action_dim:
            return (None, None, "invalid: derived training bounds dimensionality mismatch")
        return (low.tolist(), high.tolist(), "training")

    @staticmethod
    def _numeric_components(state: Any) -> dict[str, np.ndarray]:
        """Extract numeric state components as flat float arrays."""
        if isinstance(state, dict):
            items = state.items()
        else:
            items = [("__state__", state)]
        out: dict[str, np.ndarray] = {}
        for key, value in items:
            try:
                arr = np.asarray(value)
            except Exception:
                continue
            if arr.dtype.kind not in "iufb":
                continue
            flat = np.asarray(arr, dtype=np.float64).reshape(-1)
            if flat.size == 0:
                continue
            out[str(key)] = flat
        return out

    @classmethod
    def _step_delta_components(
        cls,
        before: Any,
        after: Any,
        *,
        component_periodicity: dict[str, bool] | None = None,
    ) -> dict[str, float]:
        """Per-component ||delta|| between two state snapshots."""
        a = cls._numeric_components(before)
        b = cls._numeric_components(after)
        periodicity = component_periodicity or {}
        out: dict[str, float] = {}
        for key in sorted(set(a.keys()) & set(b.keys())):
            if a[key].shape != b[key].shape:
                continue
            out[key] = cls._component_delta_norm(
                key,
                a[key],
                b[key],
                periodic=periodicity.get(key),
            )
        return out

    @classmethod
    def _max_step_deltas_from_states(
        cls,
        states: list[Any],
        *,
        component_periodicity: dict[str, bool] | None = None,
    ) -> dict[str, float]:
        """Per-component max ||delta|| across a state trajectory."""
        maxima: dict[str, float] = {}
        if len(states) < 2:
            return maxima
        for i in range(1, len(states)):
            step = cls._step_delta_components(
                states[i - 1],
                states[i],
                component_periodicity=component_periodicity,
            )
            for key, delta in step.items():
                if delta > maxima.get(key, 0.0):
                    maxima[key] = delta
        return maxima

    @staticmethod
    def _is_latent_state_key(key: str) -> bool:
        """True for velocity / rate / latent keys not re-parsable from a still frame."""
        k = str(key).lower()
        if re.search(
            r"(^|_)(delta_|vel|velocity|omega|latent|hidden|momentum|dd)(_|$|[a-z])",
            k,
        ):
            return True
        # Common first-derivative spellings: dtheta*, dq*, d_angle*, ...
        # Require an explicit derivative prefix — bare leading "d" is too broad
        # (would match door, disk, etc.).
        if re.match(r"d(theta|angle|q|pos|x|y|z)", k):
            return True
        if re.search(r"(^|_)d(theta|angle|q|pos|x|y|z)", k):
            return True
        return False

    @staticmethod
    def _is_angle_state_key(key: str) -> bool:
        """True for angular pose keys (not angular rates like dtheta)."""
        k = str(key).lower()
        if PlanningCriticSandbox._is_latent_state_key(k):
            return False
        if k in {
            "q",
            "qpos",
            "joint",
            "joints",
            "joint_pos",
            "joint_positions",
        }:
            return True
        return bool(re.search(r"(^|_)(theta|angle)(\d*|_|$)", k))

    @staticmethod
    def _component_delta_norm(
        key: str,
        reference: np.ndarray,
        current: np.ndarray,
        *,
        periodic: bool | None = None,
    ) -> float:
        """Norm-like delta with periodic handling for angle components."""
        a = np.asarray(reference, dtype=np.float64).reshape(-1)
        b = np.asarray(current, dtype=np.float64).reshape(-1)
        if a.shape != b.shape or a.size == 0:
            return float(np.inf)
        use_periodic_delta = (
            PlanningCriticSandbox._is_angle_state_key(key)
            if periodic is None
            else bool(periodic)
        )
        if use_periodic_delta:
            delta = b - a
            wrapped = np.arctan2(np.sin(delta), np.cos(delta))
            return float(np.linalg.norm(wrapped))
        return float(np.linalg.norm(b - a))

    def _declared_state_periodicity(self) -> dict[str, bool]:
        """Map simulator state-key aliases to declared periodicity."""
        for trajectory_index in self._fidelity_train_trajectories():
            try:
                trajectory = _dt.read_trajectory(
                    self._dataset_dir,
                    int(trajectory_index),
                )
            except (FileNotFoundError, ValueError):
                continue
            contract = trajectory.get("state_contract")
            if not isinstance(contract, dict):
                continue
            components = contract.get("components")
            if not isinstance(components, dict):
                continue
            periodicity: dict[str, bool] = {}
            for component_name, component in components.items():
                if not isinstance(component, dict) or not isinstance(
                    component.get("periodic"),
                    bool,
                ):
                    continue
                state_keys = component.get("state_keys", [])
                aliases = [str(component_name)]
                if isinstance(state_keys, list):
                    aliases.extend(str(key) for key in state_keys)
                for alias in aliases:
                    periodicity[alias] = bool(component["periodic"])
            if periodicity:
                return periodicity
        return {}

    @staticmethod
    def _realizability_violations(
        plan_max_deltas: dict[str, float],
        train_bounds: dict[str, float],
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Compare every training-bounded state component against a plan."""
        violations: list[dict[str, Any]] = []
        missing: list[str] = []
        for component, raw_bound in sorted(train_bounds.items()):
            if component not in plan_max_deltas:
                missing.append(component)
                continue
            delta = float(plan_max_deltas[component])
            bound = float(raw_bound)
            if delta <= bound + 1.0e-9:
                continue
            ratio = (delta / bound) if bound > 0.0 else np.inf
            violations.append(
                {
                    "component": component,
                    "delta": delta,
                    "bound": bound,
                    "ratio": float(ratio),
                }
            )
        return violations, missing

    def _realizability_limits_cached(
        self,
        *,
        margin: float,
    ) -> tuple[dict | None, str | None]:
        """Cached wrapper around `_derive_realizability_limits`.

        Deriving limits replays every training trajectory in a subprocess, and
        `validate_with_cem`/goal-reach may need them several times per code
        version (e.g. the 5-pair gate CEM suite). Cache on (margin, source hash).
        """
        import hashlib as _hashlib

        try:
            src = Path(self._sandbox_path).read_bytes()
            key = (float(margin), _hashlib.sha1(src).hexdigest())
        except OSError:
            key = None
        if key is not None and getattr(self, "_limits_cache_key", None) == key:
            return self._limits_cache_val
        val = self._derive_realizability_limits(margin=margin)
        if key is not None:
            self._limits_cache_key = key
            self._limits_cache_val = val
        return val

    def _assess_goal_reach(
        self,
        *,
        target_state: Any,
        states: list[Any],
        horizon: int,
        limits: dict | None,
    ) -> dict[str, Any]:
        """State-space goal-reach assessment for a planned rollout.

        Compares each rollout state against ``target_state`` per shared numeric
        component, normalised by the per-step motion scale observed during
        training replay (``limits['observed_max']``). Unit-free: residuals are
        reported in "one-step units" = how many maximal single steps of that
        component the remaining distance represents.

        Returns a dict with per-component numbers plus:
          reached      — all scaled components ended within ~2 one-step units
          state_stuck  — some component closed <25% of the achievable distance
                         while remaining far from target (the planner reduced
                         its internal cost without real state progress)
        """
        tgt = self._numeric_components(target_state)
        if not tgt or not states:
            return {
                "available": False,
                "note": "target_state has no numeric components",
                "components": {},
                "reached": None,
                "state_stuck": False,
                "worst": None,
            }
        last = self._numeric_components(states[-1])
        common = [
            k for k in sorted(set(tgt.keys()) & set(last.keys()))
            if tgt[k].shape == last[k].shape
        ]
        if not common:
            return {
                "available": False,
                "note": (
                    "no numeric components shared between self.state and "
                    "self.target_state — state-space goal-reach cannot be "
                    "audited (only cost-based checks apply). Consider "
                    "representing the target in the same state components "
                    "your update() simulates."
                ),
                "components": {},
                "reached": None,
                "state_stuck": False,
                "worst": None,
            }

        scales: dict[str, float] = {}
        if limits is not None:
            for k, v in dict(limits.get("observed_max", {})).items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv > 1e-12:
                    scales[str(k)] = fv

        max_scale = max(scales.values()) if scales else None
        components: dict[str, dict[str, Any]] = {}
        reached_flags: list[bool] = []
        flagged: list[dict[str, Any]] = []
        for key in common:
            residuals: list[float] = []
            for s in states:
                comps = self._numeric_components(s)
                if key not in comps or comps[key].shape != tgt[key].shape:
                    residuals.append(float("nan"))
                    continue
                residuals.append(
                    self._component_delta_norm(key, tgt[key], comps[key])
                )
            valid = [
                (i, r) for i, r in enumerate(residuals) if np.isfinite(r)
            ]
            if not valid:
                continue
            initial_res = valid[0][1]
            final_res = valid[-1][1]
            min_i, min_res = min(valid, key=lambda ir: ir[1])
            scale = scales.get(key)
            entry: dict[str, Any] = {
                "initial_residual": float(initial_res),
                "final_residual": float(final_res),
                "min_residual": float(min_res),
                "min_at_step": int(min_i),
                "scale": (float(scale) if scale is not None else None),
                "kind": None,
            }
            if scale is not None:
                steps_init = initial_res / scale
                steps_min = min_res / scale
                entry["steps_initial"] = float(steps_init)
                entry["steps_min"] = float(steps_min)
                start_near = steps_init <= 2.0
                reached_k = steps_min <= 2.0
                achievable = min(initial_res, float(horizon) * scale)
                progress = initial_res - min_res
                rel_progress = (
                    progress / achievable if achievable > 1e-12 else None
                )
                entry["closed_frac_of_achievable"] = (
                    float(rel_progress) if rel_progress is not None else None
                )
                entry["reached"] = bool(reached_k)
                # Failure kinds, unit-free:
                #  stuck    — cost fell but this component closed <25% of what
                #             the horizon could plausibly close.
                #  immobile — target is so many one-step units away that no
                #             plannable horizon could ever cover it; per-step
                #             motion for this component is effectively dead
                #             relative to task distances.
                if not reached_k and steps_init > 25.0 * max(1.0, float(horizon)):
                    entry["kind"] = "immobile"
                elif (
                    not reached_k
                    and not start_near
                    and rel_progress is not None
                    and rel_progress < 0.25
                ):
                    entry["kind"] = "stuck"
                reached_flags.append(bool(reached_k))
            else:
                # Component named in target_state but with no motion scale:
                # it never moved during training replay. If the residual is
                # larger than anything the sim moves in one step anywhere,
                # the task needs it to move and the dynamics never do.
                entry["reached"] = None
                if max_scale is not None and initial_res > max_scale:
                    entry["kind"] = "dead"
            if entry["kind"] is not None:
                flagged.append({"component": key, **entry})
            components[key] = entry

        def _severity(e: dict[str, Any]) -> tuple:
            order = {"dead": 2, "immobile": 1, "stuck": 0}
            return (
                order.get(e.get("kind"), -1),
                e.get("steps_min", float("inf")),
            )

        worst = max(flagged, key=_severity) if flagged else None
        return {
            "available": True,
            "note": None,
            "components": components,
            "reached": (all(reached_flags) if reached_flags else None),
            "state_stuck": worst is not None,
            "worst": worst,
        }

    @staticmethod
    def _format_goal_reach_lines(goal_reach: dict[str, Any]) -> list[str]:
        """Human/VLM-readable lines for a goal-reach assessment."""
        if not goal_reach.get("available"):
            note = goal_reach.get("note") or "unavailable"
            return [f"  goal-reach audit: NOTE — {note}"]
        lines = [
            "  goal-reach (state-space, per target component; 'steps' = "
            "residual distance in units of that component's max one-step "
            "motion from training replay):"
        ]
        for key, e in goal_reach.get("components", {}).items():
            kind = e.get("kind")
            if e.get("scale") is not None:
                frac = e.get("closed_frac_of_achievable")
                frac_str = (
                    f", closed {100.0 * frac:.0f}% of achievable"
                    if frac is not None
                    else ""
                )
                if e.get("reached"):
                    verdict = "REACHED"
                elif kind:
                    verdict = kind.upper()
                else:
                    verdict = "PARTIAL"
                lines.append(
                    f"    {key}: initial={e['steps_initial']:.1f} steps, "
                    f"best={e['steps_min']:.1f} at t={e['min_at_step']}, "
                    f"final={e['final_residual'] / e['scale']:.1f}"
                    f"{frac_str} -> {verdict}"
                )
            else:
                verdict = (
                    "DEAD (never moves under update, but the task needs it "
                    "to move)"
                    if kind == "dead"
                    else "no motion scale — report only"
                )
                lines.append(
                    f"    {key}: initial_residual={e['initial_residual']:.3f}, "
                    f"best={e['min_residual']:.3f} at t={e['min_at_step']}, "
                    f"final={e['final_residual']:.3f} -> {verdict}"
                )
        return lines

    @staticmethod
    def _format_goal_reach_unavailable_fail(goal_reach: dict[str, Any]) -> str:
        """FAIL text when state-space goal-reach cannot be audited at all.

        A target that shares no numeric components with ``self.state`` makes
        planner progress unauditable — historically this let cost-only checks
        pass while deployment failed, so it is a hard failure, not a note.
        """
        note = goal_reach.get("note") or "goal-reach audit unavailable"
        return (
            f"\n  FAIL: goal-reach audit impossible — {note} "
            "This is a structural bug, not a tuning issue: planner progress "
            "toward the goal cannot be verified in state space. Store the "
            "goal in target_state under the SAME keys (same shapes) as the "
            "components update(a) evolves in self.state."
        )

    @staticmethod
    def _format_goal_reach_fail(
        goal_reach: dict[str, Any],
        initial_loss: float,
        min_loss: float,
    ) -> str:
        """Kind-specific FAIL text for a goal-reach failure."""
        w = goal_reach.get("worst") or {}
        key = w.get("component")
        kind = w.get("kind")
        cost_str = f"({initial_loss:.2f} -> {min_loss:.2f})"
        if kind == "dead":
            return (
                f"\n  FAIL: goal-reach — component '{key}' appears in "
                "target_state but NEVER moved during training-action replay, "
                f"while its distance to target ({w.get('initial_residual', 0.0):.3g}) "
                "exceeds anything your sim moves in one step. The task needs "
                "this component to move and your update(a) never moves it — "
                "check how actions couple to it (contact, grasping, "
                "kinematics) before touching terminal_cost."
            )
        if kind == "immobile":
            return (
                f"\n  FAIL: goal-reach — component '{key}' is "
                f"{w.get('steps_initial', float('nan')):.0f} one-step units "
                "from target: at its observed per-step motion no plannable "
                "horizon can cover that distance. Its response to actions is "
                "effectively dead relative to task distances — update(a) "
                "under-drives it (wrong gain/contact model)."
            )
        frac = w.get("closed_frac_of_achievable")
        frac_str = f"{100.0 * frac:.0f}%" if frac is not None else "n/a"
        return (
            f"\n  FAIL: goal-reach — CEM reduced its internal cost "
            f"{cost_str} but component '{key}' closed only {frac_str} of the "
            f"achievable distance to target (initial "
            f"{w.get('steps_initial', float('nan')):.1f} -> best "
            f"{w.get('steps_min', float('nan')):.1f} one-step units). "
            "The cost falls without real state progress, which predicts "
            "planning failure in the real environment. Either update(a) "
            "under-responds for this component (e.g. contact/coupling too "
            "weak), or terminal_cost rewards something other than moving it "
            "(e.g. dominated by an easy shaping term), or the cost's minimum "
            "is not reachable under your dynamics (encode feasible-path "
            "distance, not direct distance)."
        )

    @staticmethod
    def _is_contact_transfer_gain_param(name: str) -> bool:
        """Heuristic: scalar gains likely governing contact/transfer response."""
        n = str(name).lower()
        if "contact" in n or "transfer" in n:
            return True
        if "k_trans" in n or "ktrans" in n:
            return True
        return "gain" in n

    @staticmethod
    def _is_geometry_visibility_param(name: str) -> bool:
        """Heuristic: scalar parameters likely controlling rendered/physical size."""
        n = str(name).lower()
        tokens = (
            "radius",
            "diameter",
            "width",
            "height",
            "size",
            "thickness",
            "extent",
        )
        return any(tok in n for tok in tokens)

    @staticmethod
    def _build_probe_actions(action_low: np.ndarray, action_high: np.ndarray) -> list[list[float]]:
        """Stress actions within provided bounds (corners + axis extremes)."""
        dim = int(action_low.size)
        low = np.asarray(action_low, dtype=np.float64).reshape(-1)
        high = np.asarray(action_high, dtype=np.float64).reshape(-1)
        mid = 0.5 * (low + high)
        probes: list[np.ndarray] = [low, high, mid]
        if dim <= 4:
            for bits in product([0, 1], repeat=dim):
                probes.append(np.where(np.asarray(bits, dtype=bool), high, low))
        else:
            probes.append(np.where((np.arange(dim) % 2) == 0, low, high))
            probes.append(np.where((np.arange(dim) % 2) == 0, high, low))
        for i in range(dim):
            lo = mid.copy()
            hi = mid.copy()
            lo[i] = low[i]
            hi[i] = high[i]
            probes.extend([lo, hi])

        uniq: list[list[float]] = []
        seen: set[tuple[float, ...]] = set()
        for p in probes:
            key = tuple(np.round(np.asarray(p, dtype=np.float64), 12).tolist())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(np.asarray(p, dtype=np.float64).tolist())
        return uniq

    @staticmethod
    def _build_training_probe_actions(stacked_actions: np.ndarray) -> list[list[float]]:
        """Probe actions sampled from REAL observed actions (no synthetic corners)."""
        stacked = np.asarray(stacked_actions, dtype=np.float64)
        if stacked.ndim != 2 or stacked.shape[0] == 0:
            return []
        norms = np.linalg.norm(stacked, axis=1)
        idxs: set[int] = {int(np.argmax(norms)), int(np.argmin(norms))}
        for d in range(stacked.shape[1]):
            idxs.add(int(np.argmin(stacked[:, d])))
            idxs.add(int(np.argmax(stacked[:, d])))
        probes = [stacked[i] for i in sorted(idxs)]
        probes.append(np.mean(stacked, axis=0))
        uniq: list[list[float]] = []
        seen: set[tuple[float, ...]] = set()
        for p in probes:
            key = tuple(np.round(np.asarray(p, dtype=np.float64), 12).tolist())
            if key in seen:
                continue
            seen.add(key)
            uniq.append(np.asarray(p, dtype=np.float64).tolist())
        return uniq

    # ------------------------------------------------------------------
    # Overridden run_simulation — uses the action-conditioned runner
    # ------------------------------------------------------------------

    def _default_test_action(self) -> list[float] | None:
        """A real dataset action for smoke rollouts (dimension-correct).

        Uses the largest-norm action from the first readable training
        trajectory so `run_simulation` exercises `update(a)` with an action
        whose dimensionality and scale match the training data.
        """
        for t in self._fidelity_train_trajectories() or [0]:
            try:
                actions = _dt.read_trajectory(self._dataset_dir, int(t))["actions"]
            except (FileNotFoundError, ValueError):
                continue
            if actions.shape[0] == 0:
                continue
            idx = int(np.argmax(np.linalg.norm(actions, axis=1)))
            return [float(x) for x in actions[idx]]
        return None

    def run_simulation(self) -> str:
        """Execute the action-conditioned sim in a subprocess.

        Calls ``sim.fit(image_A, image_B)`` then rolls forward ``n_frames``
        steps with a fixed dataset-derived test action, verifying the
        contract and saving frames for inspection.
        """
        if os.path.exists(self._frames_npy):
            os.remove(self._frames_npy)
        if os.path.exists(self._debug_view_npy):
            os.remove(self._debug_view_npy)

        if self._world_api_log_dir:
            if os.path.isdir(self._world_api_log_dir):
                shutil.rmtree(self._world_api_log_dir)
            os.makedirs(self._world_api_log_dir)

        cmd = [
            sys.executable,
            "-m",
            _RUNNER_MODULE,
            "--simulator-path",
            self._sandbox_path,
            "--simulator-class",
            self._simulator_class_name,
            "--fps",
            str(self._fps),
            "--n-frames",
            str(self._n_frames),
            "--frame-size",
            str(self._frame_size[0]),
            str(self._frame_size[1]),
            "--start-image-path",
            self._start_image_path,
            "--goal-image-path",
            self._goal_image_path,
            "--frames-npy",
            self._frames_npy,
            "--debug-view-npy",
            self._debug_view_npy,
        ]
        if self._no_api:
            cmd.append("--no-api")
        if self._cache_dir:
            cmd += ["--cache-dir", self._cache_dir]
        if self._world_api_log_dir:
            cmd += ["--api-calls-dir", self._world_api_log_dir]
        test_action = self._default_test_action()
        if test_action is not None:
            cmd += ["--test-action", json.dumps(test_action)]

        import time as _time

        logger.debug("PlanningCriticSandbox: running subprocess: %s", " ".join(cmd))
        _t = _time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env=self._subprocess_env(),
        )
        print(
            f"[planning_sandbox] run_simulation subprocess: {_time.perf_counter() - _t:.1f}s"
        )
        self._last_stdout = result.stdout
        self._last_stderr = result.stderr

        combined = ""
        if result.stdout:
            combined += result.stdout
        if result.stderr:
            combined += "\n--- stderr ---\n" + result.stderr
        if result.returncode != 0:
            combined += (
                f"\n[run_simulation] Process exited with code {result.returncode}"
            )

        extra_files: dict = {
            "terminal_output.txt": combined.strip() or "No output produced."
        }
        if os.path.exists(self._frames_npy):
            _frames_log = np.load(self._frames_npy)
            if len(_frames_log) > 0:
                extra_files["first_frame.png"] = Image.fromarray(_frames_log[0])
            if len(_frames_log) > 1:
                extra_files["last_frame.png"] = Image.fromarray(_frames_log[-1])

        self._log_critic_tool_call(
            tool_name="run_simulation",
            args={"n_frames": self._n_frames, "fps": self._fps},
            result_summary=combined[:300] if combined else "No output.",
            extra_files=extra_files,
        )

        if "APIInternalError" in self._last_stderr:
            combined += self._register_api_internal_error()

        return combined.strip() or "[run_simulation] No output produced."

    def _register_api_internal_error(self, *, max_tolerated: int = 2) -> str:
        """Handle an APIInternalError seen in sandbox stderr.

        A backend/API failure is infrastructure, not VLM code — but a single
        flaky call must not kill a 40-turn codegen run (2026-07-07: both cube
        runs died mid-loop this way). Surface the first occurrences as tool
        feedback so the model can route around the broken backend; abort only
        when it keeps recurring (service truly down AND the model keeps
        depending on it).
        """
        self._api_internal_error_count = (
            getattr(self, "_api_internal_error_count", 0) + 1
        )
        if self._api_internal_error_count > max_tolerated:
            logger.error(
                "APIInternalError in sandbox stderr %d times — aborting "
                "agentic loop.",
                self._api_internal_error_count,
            )
            raise AbortLoopError(
                "APIInternalError in sandbox stderr — persistent "
                "infrastructure failure."
            )
        logger.warning(
            "APIInternalError in sandbox stderr (%d/%d tolerated) — "
            "returned to VLM as feedback.",
            self._api_internal_error_count,
            max_tolerated,
        )
        return (
            "\n[run_simulation] WARNING: an API backend call failed "
            "(APIInternalError above). This backend appears unavailable "
            "— do NOT keep relying on it. Rewrite the failing code path "
            "to use local computation (e.g. color/geometry parsing) "
            "instead of that api.* call."
        )

    # ------------------------------------------------------------------
    # Auto-validate feedback: every code change is immediately scored against
    # ground-truth-action replay, so the VLM never flies blind or unknowingly
    # submits a degenerate / unvalidated simulator. This is FEEDBACK only — it
    # appends the fidelity result to the tool output; it does NOT alter or select
    # the VLM's code. Task-agnostic: it uses the simulator's own terminal_cost.
    # ------------------------------------------------------------------

    _AUTO_VALIDATE_TRAJ = 0

    def _auto_validate_feedback(self) -> str:
        """Run validate_against_training across the auto-validate trajectory set
        and format it as a feedback suffix appended to a code-change result.

        With a single trajectory this is the old behaviour (full summary). With a
        diverse spread it reports the per-trajectory reduction_ratio and the WORST
        (least-faithful) one, with full detail for the worst — so the VLM optimises
        for fidelity across all of them instead of overfitting one. Feedback only;
        the harness never selects which code ships."""
        import re as _re

        trajs = self._auto_validate_trajs
        rows: list[tuple[int, float]] = []
        failures: list[tuple[int, str]] = []
        worst_t: int | None = None
        worst_ratio = -1.0
        worst_summary = ""
        for t in trajs:
            try:
                summary = self.validate_against_training(trajectory_index=t)
            except Exception as exc:  # never let feedback crash the code-change tool
                failures.append((t, f"{type(exc).__name__}: {exc}"))
                continue
            m = _re.search(r"reduction_ratio=([0-9.]+|n/a)", summary)
            if m and m.group(1) != "n/a":
                ratio = float(m.group(1))
                rows.append((t, ratio))
                if ratio > worst_ratio:  # worst = highest ratio = least faithful
                    worst_ratio, worst_t, worst_summary = ratio, t, summary
            else:
                failures.append((t, "no reduction_ratio (degenerate/crash)"))

        # Single-trajectory mode preserves the original verbose feedback.
        if len(trajs) == 1 and rows and not failures:
            return (
                f"\n\n[auto-validate on trajectory {trajs[0]} — the fidelity of the "
                f"code you just wrote; judge by reduction_ratio, fix if it failed]\n"
                f"{worst_summary}"
            )

        ratio_line = ", ".join(f"t{t}={r:.3f}" for t, r in rows) or "(none parsed)"
        msg = (
            f"\n\n[auto-validate across trajectories {trajs} — a faithful simulator "
            f"must reproduce ALL of them, not one. Judge by the WORST (highest) ratio; "
            f"a low ratio on one trajectory while others stay high means you have "
            f"OVERFIT that trajectory, not captured the dynamics.]\n"
            f"  reduction_ratio per trajectory: {ratio_line}\n"
        )
        if failures:
            msg += (
                "  FAILED/degenerate: "
                + ", ".join(f"t{t} ({why})" for t, why in failures)
                + "\n"
            )
        if worst_t is not None:
            msg += (
                f"  WORST = trajectory {worst_t} (ratio {worst_ratio:.3f}); detail "
                f"below — improve the dynamics that fail here without breaking the "
                f"others:\n{worst_summary}"
            )
            try:
                sc = self._run_state_consistency_check(worst_t, max_checkpoints=8)
                if sc.get("ok") and sc.get("compact_lines"):
                    msg += (
                        f"\n  [state-consistency on t{worst_t} — gap between "
                        "open-loop rollout and your own fit() parse of the true "
                        "frames; localises WHICH component and WHEN dynamics "
                        "diverge (run validate_state_consistency for the full "
                        "table)]\n"
                        + "\n".join("    " + line for line in sc["compact_lines"])
                    )
            except Exception:
                pass  # feedback must never break the code-change tool
        return msg

    def _auto_visual_reconstruction_feedback(self) -> str:
        """Inject bounded true-pixel feedback after representative code edits."""
        self._auto_visual_code_changes += 1
        changes_since_last = (
            self._auto_visual_code_changes - self._auto_visual_last_change
        )
        if self._auto_visual_runs >= 3 or (
            self._auto_visual_runs > 0 and changes_since_last < 8
        ):
            return ""
        if not self.has_valid_simulator_class():
            return ""

        self._auto_visual_runs += 1
        self._auto_visual_last_change = self._auto_visual_code_changes
        try:
            payload = self._run_fit_quality_check(
                self._auto_validate_trajs,
                max_rmse=self._gate_fit_quality_max_rmse,
                max_probes=3,
            )
        except Exception as exc:
            return (
                "\n\n[auto state-to-pixel reconstruction — advisory]\n"
                f"  check failed to execute: {type(exc).__name__}: {exc}\n"
                "  Repair fit()/render_frame() execution before tuning dynamics."
            )

        if payload.get("error"):
            return (
                "\n\n[auto state-to-pixel reconstruction — advisory]\n"
                f"  {payload['error']}\n"
                "  Repair fit()/render_frame() execution before tuning dynamics."
            )

        rows = []
        structural_failures = 0
        for probe in payload.get("probes", []):
            if probe.get("error"):
                structural_failures += 1
                rows.append(
                    f"  t{probe.get('trajectory')}/f{probe.get('frame_index')}: "
                    f"ERROR {probe['error']}"
                )
                continue
            foreground = probe.get("foreground_check") or {}
            failed_labels = [
                str(item.get("label"))
                for item in probe.get("component_checks", [])
                if not item.get("passed", True)
                and not item.get("skipped_scene_mask", False)
            ]
            foreground_ok = bool(foreground.get("passed", True))
            if failed_labels or not foreground_ok:
                structural_failures += 1
            rows.append(
                f"  t{probe.get('trajectory')}/f{probe.get('frame_index')}: "
                f"rmse={float(probe.get('normalized_rmse', float('nan'))):.3f} "
                f"foreground_iou={float(foreground.get('iou', float('nan'))):.3f} "
                f"foreground_ok={foreground_ok} "
                f"failed_components={failed_labels or 'none'}"
            )

        verdict = (
            "STRUCTURAL MISMATCH: repair perception/geometry now"
            if structural_failures
            else "no coarse structural mismatch detected"
        )
        return (
            "\n\n[auto state-to-pixel reconstruction — advisory, bounded; "
            "true frame versus render from your parsed state]\n"
            + "\n".join(rows)
            + f"\n  verdict: {verdict}. This check is independent of replay "
            "ratio; do not tune dynamics while projected geometry is wrong."
        )

    def _cem_suite_pairs(self, trajectories: list[int], *, n_pairs: int = 5) -> list[tuple[int, int, int, int]]:
        """Choose deterministic, visually separated start/goal pairs.

        The selection is benchmark-agnostic: score trajectory start/end pairs by
        image-space separation and choose diverse far pairs. For two-room this
        tends to pick cross-room pairs; for manipulation/reacher tasks it tends
        to pick large pose/configuration changes.
        """
        available = []
        for t in trajectories:
            try:
                tr = _dt.read_trajectory(self._dataset_dir, int(t))
            except (FileNotFoundError, ValueError):
                continue
            if tr["n_steps"] > 0 and len(tr["frames"]) > 1:
                available.append((int(t), len(tr["frames"]) - 1))
        if not available:
            return []

        try:
            starts: dict[int, np.ndarray] = {}
            goals: dict[int, np.ndarray] = {}
            for t, last in available:
                starts[t] = _dt.view_image(self._dataset_dir, t, 0).astype(np.float32)
                goals[t] = _dt.view_image(self._dataset_dir, t, last).astype(np.float32)

            candidates: list[tuple[float, int, int, int]] = []
            for start_t, _ in available:
                for goal_t, goal_last in available:
                    # Prefer cross-trajectory pairs when the dataset has enough
                    # trajectories; same-trajectory endpoints are a fallback.
                    if len(available) > 1 and start_t == goal_t:
                        continue
                    distance = float(np.linalg.norm(starts[start_t] - goals[goal_t]))
                    candidates.append((distance, start_t, goal_t, goal_last))
            candidates.sort(reverse=True)

            pairs: list[tuple[int, int, int, int]] = []
            used_start: set[int] = set()
            used_goal: set[int] = set()
            for _, start_t, goal_t, goal_last in candidates:
                if start_t in used_start or goal_t in used_goal:
                    continue
                pairs.append((start_t, 0, goal_t, goal_last))
                used_start.add(start_t)
                used_goal.add(goal_t)
                if len(pairs) >= n_pairs:
                    return pairs
            for _, start_t, goal_t, goal_last in candidates:
                pair = (start_t, 0, goal_t, goal_last)
                if pair not in pairs:
                    pairs.append(pair)
                if len(pairs) >= n_pairs:
                    return pairs
            return pairs
        except Exception:
            if len(available) <= n_pairs:
                selected = available
            else:
                picks = np.linspace(0, len(available) - 1, n_pairs, dtype=int).tolist()
                selected = [available[i] for i in dict.fromkeys(picks)]

            pairs: list[tuple[int, int, int, int]] = []
            for i, (start_t, _) in enumerate(selected):
                goal_t, goal_last = selected[-(i + 1)]
                pairs.append((start_t, 0, goal_t, goal_last))
            return pairs

    def _run_target_configuration_check(
        self,
        trajectories: list[int],
        *,
        n_probes: int = 3,
    ) -> dict[str, Any]:
        """Check that at least one changed task-space configuration is scored.

        For each probe, ``fit(start, goal)`` supplies the candidate target while
        ``fit(goal, goal)`` supplies the simulator's own parse of the desired
        observable configuration. At least one non-latent component that changes
        between those parses must have a same-key target value, agree with the
        goal parse, and measurably affect the terminal objective. Other changed
        components remain diagnostics: joints, pushers, and other mechanism state
        may be necessary for dynamics without defining task success.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        available: list[tuple[int, str, str]] = []
        for trajectory_index in trajectories:
            try:
                trajectory = _dt.read_trajectory(
                    self._dataset_dir, int(trajectory_index)
                )
            except (FileNotFoundError, ValueError):
                continue
            if trajectory["n_steps"] <= 0 or len(trajectory["frames"]) < 2:
                continue
            available.append(
                (
                    int(trajectory_index),
                    str(trajectory["frames"][0]),
                    str(trajectory["frames"][-1]),
                )
            )
        if not available:
            return {
                "passed": False,
                "summary": "no usable start/goal probes",
                "probes": [],
            }
        if len(available) > int(n_probes):
            indices = np.linspace(
                0, len(available) - 1, int(n_probes)
            ).round().astype(int)
            available = [available[i] for i in dict.fromkeys(indices.tolist())]

        runner_src = (
            "import sys, json, copy, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.planning_critic_toolbox import _generic_fit_quality_color_masks\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "def _mk():\n"
            "    api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "    return cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "def _numeric(state):\n"
            "    items = state.items() if isinstance(state, dict) else [('__state__', state)]\n"
            "    out = {}\n"
            "    for key, value in items:\n"
            "        try:\n"
            "            arr = np.asarray(value)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if arr.dtype.kind not in 'iufb':\n"
            "            continue\n"
            "        flat = np.asarray(arr, dtype=np.float64).reshape(-1)\n"
            "        if flat.size:\n"
            "            out[str(key)] = flat.tolist()\n"
            "    return out\n"
            "records = []\n"
            "for probe in args['probes']:\n"
            "    record = {'trajectory': probe['trajectory']}\n"
            "    try:\n"
            "        start = np.array(Image.open(probe['start']).convert('RGB'))\n"
            "        goal = np.array(Image.open(probe['goal']).convert('RGB'))\n"
            "        sim = _mk(); sim.fit(start, goal)\n"
            "        desired = _mk(); desired.fit(goal, goal.copy())\n"
            "        record['state'] = _numeric(sim.state)\n"
            "        record['target'] = _numeric(sim.target_state)\n"
            "        record['desired'] = _numeric(desired.state)\n"
            "        sensitivity = {}\n"
            "        if isinstance(sim.state, dict) and isinstance(sim.target_state, dict):\n"
            "            initial_state = copy.deepcopy(sim.state)\n"
            "            for key in set(sim.state) & set(sim.target_state):\n"
            "                try:\n"
            "                    if np.asarray(sim.state[key]).shape == np.asarray(sim.target_state[key]).shape:\n"
            "                        sim.state[key] = copy.deepcopy(sim.target_state[key])\n"
            "                except Exception:\n"
            "                    pass\n"
            "            aligned_cost = float(sim.planning_objective())\n"
            "            record['aligned_cost'] = aligned_cost\n"
            "            for key in set(sim.state) & set(sim.target_state):\n"
            "                original = copy.deepcopy(sim.target_state[key])\n"
            "                try:\n"
            "                    sim.target_state[key] = copy.deepcopy(initial_state[key])\n"
            "                    sensitivity[str(key)] = float(sim.planning_objective())\n"
            "                except Exception:\n"
            "                    sensitivity[str(key)] = None\n"
            "                finally:\n"
            "                    sim.target_state[key] = original\n"
            "        record['cost_with_target_replaced_by_initial'] = sensitivity\n"
            "    except Exception as exc:\n"
            "        record['error'] = f'{type(exc).__name__}: {exc}'\n"
            "    records.append(record)\n"
            "print(json.dumps(records))\n"
        )
        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name
        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
                "probes": [
                    {
                        "trajectory": trajectory,
                        "start": start,
                        "goal": goal,
                    }
                    for trajectory, start, goal in available
                ],
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=180,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass
        if result.returncode != 0:
            return {
                "passed": False,
                "summary": (
                    f"probe runner exited {result.returncode}: "
                    f"{result.stderr[-500:]}"
                ),
                "probes": [],
            }
        try:
            records = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {
                "passed": False,
                "summary": f"could not parse probe output: {exc}",
                "probes": [],
            }

        latent_tokens = (
            "velocity",
            "_vel",
            "omega",
            "rate",
            "counter",
            "step",
            "time",
            "contact",
            "holding",
            "mode",
        )
        probes: list[dict[str, Any]] = []
        for record in records:
            probe: dict[str, Any] = {"trajectory": record.get("trajectory")}
            if record.get("error"):
                probe.update({"passed": False, "error": record["error"]})
                probes.append(probe)
                continue
            state = self._numeric_components(record.get("state"))
            target = self._numeric_components(record.get("target"))
            desired = self._numeric_components(record.get("desired"))
            changed: list[str] = []
            valid_targets: list[str] = []
            failures: list[str] = []
            details: dict[str, Any] = {}
            for key in sorted(set(state) & set(desired)):
                if state[key].shape != desired[key].shape:
                    continue
                lower = key.lower()
                if any(token in lower for token in latent_tokens):
                    continue
                motion = self._component_delta_norm(key, state[key], desired[key])
                if not np.isfinite(motion) or motion <= 1e-3:
                    continue
                changed.append(key)
                detail: dict[str, Any] = {"observable_change": float(motion)}
                if key not in target or target[key].shape != desired[key].shape:
                    detail["diagnostic"] = "not a terminal target"
                    details[key] = detail
                    continue
                target_error = self._component_delta_norm(
                    key, desired[key], target[key]
                )
                detail["target_vs_goal_error"] = float(target_error)
                target_matches = target_error <= max(1e-3, 0.25 * motion)
                if target_error > max(1e-3, 0.25 * motion):
                    detail["failure"] = "target does not match image_B configuration"
                alternative_cost = (
                    record.get("cost_with_target_replaced_by_initial") or {}
                ).get(key)
                aligned_cost = record.get("aligned_cost")
                detail["aligned_cost"] = aligned_cost
                detail["cost_with_initial_as_target"] = alternative_cost
                cost_sensitive = False
                if alternative_cost is None or aligned_cost is None:
                    detail["failure"] = "terminal-cost sensitivity unavailable"
                else:
                    cost_delta = abs(float(alternative_cost) - float(aligned_cost))
                    detail["target_cost_sensitivity"] = float(cost_delta)
                    cost_sensitive = cost_delta > 1e-6 * max(
                        1.0, abs(float(aligned_cost))
                    )
                    if not cost_sensitive:
                        detail["failure"] = "target component ignored by terminal_cost"
                if target_matches and cost_sensitive:
                    detail["valid_task_target"] = True
                    valid_targets.append(key)
                details[key] = detail
            if not changed:
                failures.append("no-changed-observable-components")
            elif not valid_targets:
                failures.append("no-changed-cost-sensitive-task-target")
            probe.update(
                {
                    "passed": not failures,
                    "changed_components": changed,
                    "valid_task_targets": valid_targets,
                    "failures": failures,
                    "details": details,
                }
            )
            probes.append(probe)

        failed = [probe for probe in probes if not probe.get("passed")]
        if failed:
            first = failed[0]
            summary = (
                f"{len(failed)}/{len(probes)} probes failed; "
                f"trajectory {first.get('trajectory')}: "
                + ", ".join(first.get("failures") or ["unknown"])
            )
        else:
            summary = (
                f"{len(probes)}/{len(probes)} probes expose a changed, "
                "cost-sensitive task-space target"
            )
        return {
            "passed": bool(probes) and not failed,
            "summary": summary,
            "probes": probes,
        }

    def _run_cem_suite_check(
        self,
        trajectories: list[int],
        *,
        n_pairs: int = 5,
        horizon: int = 20,
        cem_iters: int = 5,
        cem_population: int = 100,
    ) -> dict[str, Any]:
        """Run a fixed CEM stress suite chosen by the harness, not the VLM."""
        target_configuration = self._run_target_configuration_check(trajectories)
        if not target_configuration.get("passed"):
            return {
                "passed": False,
                "error": (
                    "target-configuration coverage failed: "
                    + str(target_configuration.get("summary") or "unknown failure")
                ),
                "target_configuration": target_configuration,
                "pairs": [],
                "results": [],
            }
        pairs = self._cem_suite_pairs(trajectories, n_pairs=n_pairs)
        if not pairs:
            return {
                "passed": False,
                "error": "no valid trajectory pairs available for CEM suite",
                "pairs": [],
                "results": [],
            }

        results = []
        for s_t, s_f, g_t, g_f in pairs:
            summary = self.validate_with_cem(
                start_trajectory_index=s_t,
                start_frame_index=s_f,
                goal_trajectory_index=g_t,
                goal_frame_index=g_f,
                horizon=horizon,
                cem_iters=cem_iters,
                cem_population=cem_population,
            )
            failed = "FAIL:" in summary or summary.startswith("[validate_with_cem]")
            results.append(
                {
                    "pair": {
                        "start_trajectory_index": s_t,
                        "start_frame_index": s_f,
                        "goal_trajectory_index": g_t,
                        "goal_frame_index": g_f,
                    },
                    "passed": not failed,
                    "summary": summary,
                }
            )
        return {
            "passed": all(r["passed"] for r in results),
            "target_configuration": target_configuration,
            "pairs": pairs,
            "results": results,
        }

    def _run_deployment_goal_check(
        self,
        trajectories: list[int],
        *,
        n_probes: int = 3,
    ) -> dict[str, Any]:
        """Validate fit() under the DEPLOYMENT goal-image distribution.

        Task-agnostic train/deploy alignment check. Training pairs use a
        completed-task frame as image_B, but some P2 protocols construct
        image_B from the CURRENT observation (``deployment_goal_mode ==
        "start"``). A fit() that parses the target from the movable
        components' own pose looks faithful in training yet degenerates at
        deployment: target_state == current state, terminal_cost ≈ 0, and the
        planner has no incentive to act.

        Probes ``fit(frame_0, frame_0)`` on up to ``n_probes`` trajectories.
        A probe fails when every shared numeric component's target sits within
        noise of the current state (degenerate) or when no components are
        shared (unauditable). The check fails when more than half the probes
        fail — single-probe tolerance avoids false positives on episodes that
        genuinely start at the goal.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        mode = self._deployment_goal_mode
        if mode == "final":
            return {"passed": True, "skipped": True, "mode": mode, "probes": []}
        if mode != "start":
            return {
                "passed": False,
                "skipped": False,
                "mode": mode,
                "error": f"unknown deployment_goal_mode '{mode}'",
                "probes": [],
            }

        probe_pairs: list[tuple[int, str]] = []
        for t in trajectories:
            try:
                tr = _dt.read_trajectory(self._dataset_dir, int(t))
            except (FileNotFoundError, ValueError):
                continue
            if tr["n_steps"] <= 0 or len(tr["frames"]) < 2:
                continue
            # Motion guard: only probe episodes where the task visibly changes
            # the scene, so "starts at the goal" episodes cannot false-flag.
            try:
                first = np.asarray(
                    Image.open(tr["frames"][0]).convert("RGB"), dtype=np.float32
                )
                final = np.asarray(
                    Image.open(tr["frames"][-1]).convert("RGB"), dtype=np.float32
                )
                if float(np.abs(first - final).mean()) < 1.0:
                    continue
            except OSError:
                continue
            probe_pairs.append((int(t), str(tr["frames"][0])))
            if len(probe_pairs) >= n_probes:
                break
        if not probe_pairs:
            return {
                "passed": False,
                "skipped": False,
                "mode": mode,
                "error": "no usable trajectories for deployment-goal probes",
                "probes": [],
            }

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "def _repr(s):\n"
            "    if isinstance(s, dict):\n"
            "        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}\n"
            "    if isinstance(s, np.ndarray):\n"
            "        return s.tolist()\n"
            "    return str(s)\n"
            "out = []\n"
            "for probe in args['probes']:\n"
            "    entry = {'trajectory': probe['trajectory']}\n"
            "    try:\n"
            "        sim = cls(frame_size=tuple(args['frame_size']), api=None, fps=args['fps'])\n"
            "        img = np.array(Image.open(probe['frame']).convert('RGB'))\n"
            "        sim.fit(img, img.copy())\n"
            "        entry['state'] = _repr(sim.state)\n"
            "        entry['target_state'] = _repr(sim.target_state)\n"
            "        entry['terminal_cost'] = float(sim.planning_objective())\n"
            "    except Exception as exc:\n"
            "        entry['error'] = f'{type(exc).__name__}: {exc}'\n"
            "    out.append(entry)\n"
            "print(json.dumps(out))\n"
        )
        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name
        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "probes": [
                    {"trajectory": t, "frame": frame}
                    for t, frame in probe_pairs
                ],
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=120,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass
        if result.returncode != 0:
            return {
                "passed": False,
                "skipped": False,
                "mode": mode,
                "error": (
                    "deployment-goal probe runner failed: "
                    f"{result.stderr[:500]}"
                ),
                "probes": [],
            }
        try:
            payloads = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {
                "passed": False,
                "skipped": False,
                "mode": mode,
                "error": f"could not parse probe runner output: {exc}",
                "probes": [],
            }

        limits, _limits_err = self._realizability_limits_cached(
            margin=self._gate_realizability_margin
        )
        scales: dict[str, float] = {}
        if limits is not None:
            for k, v in dict(limits.get("observed_max", {})).items():
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if fv > 1e-12:
                    scales[str(k)] = fv

        probes: list[dict[str, Any]] = []
        n_bad = 0
        for entry in payloads:
            probe: dict[str, Any] = {"trajectory": entry.get("trajectory")}
            if entry.get("error"):
                probe["verdict"] = "error"
                probe["detail"] = entry["error"]
                n_bad += 1
                probes.append(probe)
                continue
            state = self._numeric_components(entry.get("state"))
            target = self._numeric_components(entry.get("target_state"))
            common = [
                k for k in sorted(set(state.keys()) & set(target.keys()))
                if state[k].shape == target[k].shape
            ]
            probe["terminal_cost"] = entry.get("terminal_cost")
            if not common:
                probe["verdict"] = "unauditable"
                probe["detail"] = (
                    "no numeric components shared between state and "
                    "target_state"
                )
                n_bad += 1
                probes.append(probe)
                continue
            residuals: dict[str, float] = {}
            degenerate = True
            for key in common:
                res = self._component_delta_norm(key, target[key], state[key])
                residuals[key] = float(res)
                scale = scales.get(key)
                # Within a quarter of one observed step (or ~exact when the
                # motion scale is unknown) counts as "same as current state".
                tol = 0.25 * scale if scale is not None else 1e-6
                if res > max(tol, 1e-9):
                    degenerate = False
            probe["residuals"] = residuals
            probe["verdict"] = "degenerate" if degenerate else "ok"
            if degenerate:
                n_bad += 1
            probes.append(probe)

        passed = n_bad <= len(probes) // 2
        return {
            "passed": passed,
            "skipped": False,
            "mode": mode,
            "n_probes": len(probes),
            "n_bad": n_bad,
            "probes": probes,
        }

    def _scan_runtime_source_safety(self) -> dict[str, Any]:
        """Reject shipped simulators that depend on training files at P2 time."""
        try:
            source = Path(self._sandbox_path).read_text(encoding="utf-8")
        except OSError as exc:
            return {"passed": False, "error": f"could not read simulator source: {exc}"}

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return {"passed": False, "error": f"could not parse simulator source: {exc}"}

        forbidden_modules = {"os", "pathlib", "glob", "subprocess", "shutil"}
        forbidden_calls = {
            "open",
            "Path",
            "os.path.exists",
            "os.path.isfile",
            "os.path.isdir",
            "os.path.join",
            "os.listdir",
            "os.walk",
            "glob.glob",
            "subprocess.run",
            "subprocess.Popen",
            "shutil.copy",
            "np.load",
            "numpy.load",
            "json.load",
            "Image.open",
        }
        dataset_markers = {
            str(self._dataset_dir.resolve()),
            "MLMI_research_project/Datasets",
            "/Datasets/",
            "trajectory_",
            "actions.json",
            ".npy",
            ".npz",
            ".h5",
            ".hdf5",
        }

        def _name(node: ast.AST) -> str:
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                base = _name(node.value)
                return f"{base}.{node.attr}" if base else node.attr
            return ""

        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name.split(".")[0] for alias in getattr(node, "names", [])]
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module.split(".")[0])
                for name in names:
                    if name in forbidden_modules:
                        violations.append(
                            f"line {getattr(node, 'lineno', '?')}: imports `{name}`"
                        )
            elif isinstance(node, ast.Call):
                call_name = _name(node.func)
                if call_name in forbidden_calls:
                    violations.append(
                        f"line {getattr(node, 'lineno', '?')}: calls `{call_name}`"
                    )
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
                for marker in dataset_markers:
                    if marker and marker in value:
                        violations.append(
                            f"line {getattr(node, 'lineno', '?')}: embeds dataset/file marker `{marker}`"
                        )
                        break

        if violations:
            preview = "; ".join(violations[:6])
            extra = "" if len(violations) <= 6 else f"; ... +{len(violations) - 6} more"
            return {
                "passed": False,
                "error": (
                    "generated simulator appears to read or locate training files at P2 runtime. "
                    "Use dataset tools during codegen, then bake learned constants into ordinary "
                    f"Python data structures. Violations: {preview}{extra}"
                ),
                "violations": violations,
            }

        return {"passed": True}

    def _run_p2_safety_check(self) -> dict[str, Any]:
        """Smoke-test that the generated sim can run under the P2 API contract.

        This does not forbid local deterministic physics libraries. It catches
        dependence on WorldAPI/SAM/VLM-style calls during CEM rollouts and
        broken simulator contracts before P2. WorldAPI is allowed for
        ``fit(image_A, image_B)`` when enabled by the harness.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        source_safety = self._scan_runtime_source_safety()
        if not source_safety.get("passed"):
            return source_safety

        stacked = self._collect_training_actions()
        if stacked is None or stacked.ndim != 2 or stacked.shape[0] == 0:
            return {"passed": False, "error": "could not derive action bounds"}
        action_dim = int(stacked.shape[1])
        action_low, action_high, bounds_source = self._resolve_action_bounds(
            action_dim=action_dim,
        )
        if action_low is None or action_high is None:
            return {"passed": False, "error": f"could not resolve action bounds ({bounds_source})"}

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "from vdaworld.core.cem import CEM\n"
            "class _NoWorldAPIInRollout:\n"
            "    def __getattr__(self, name):\n"
            "        raise RuntimeError(f'WorldAPI is only allowed during fit(), not rollout method {name!r}')\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_p2_safety', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "sys.modules['simulator_p2_safety'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "image_A = np.array(Image.open(args['start_path']).convert('RGB'))\n"
            "image_B = np.array(Image.open(args['goal_path']).convert('RGB'))\n"
            "sim.fit(image_A, image_B)\n"
            "sim.api = _NoWorldAPIInRollout()\n"
            "frame = np.asarray(sim.render_frame())\n"
            "loss = float(sim.planning_objective())\n"
            "cem = CEM(sim, action_dim=args['action_dim'], action_low=np.asarray(args['action_low']), action_high=np.asarray(args['action_high']), horizon=3, population=24, iters=2, seed=7)\n"
            "plan = cem.plan()\n"
            "for a in plan[:2]:\n"
            "    sim.update(np.asarray(a, dtype=np.float64))\n"
            "loss_after = float(sim.planning_objective())\n"
            "print(json.dumps({'loss': loss, 'loss_after': loss_after, 'frame_shape': list(frame.shape), 'plan_shape': list(plan.shape)}))\n"
        )
        with _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name
        try:
            args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "start_path": self._start_image_path,
                "goal_path": self._goal_image_path,
                "action_dim": action_dim,
                "action_low": action_low,
                "action_high": action_high,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(args)],
                capture_output=True,
                text=True,
                timeout=90,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "OPENCV_FOR_THREADS_NUM": "1",
                },
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return {
                "passed": False,
                "error": result.stderr[-1500:] or result.stdout[-1500:] or "unknown P2 safety failure",
            }
        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {
                "passed": False,
                "error": f"could not parse P2 safety output: {exc}",
                "stdout": result.stdout[-1500:],
            }
        frame_shape = payload.get("frame_shape") or []
        passed = len(frame_shape) == 3 and frame_shape[2] == 3
        return {"passed": bool(passed), "payload": payload, "bounds_source": bounds_source}

    def final_fidelity_gate(
        self,
        trajectories: list[int] | None = None,
        *,
        max_worst_ratio: float | None = None,
        require_action_realizability: bool | None = None,
        realizability_margin: float | None = None,
        require_goal_invariance: bool | None = None,
        require_cem_suite: bool | None = None,
        require_p2_safety: bool | None = None,
        goal_invariance_abs_tol: float | None = None,
        goal_invariance_rel_tol: float | None = None,
        require_target_independence: bool | None = None,
        require_state_consistency: bool | None = None,
        require_fit_quality: bool | None = None,
        fit_quality_max_rmse: float | None = None,
    ) -> dict:
        """Final gate over shipped code: no crash, bounded worst-ratio, realizable, goal-stable.

        The gate sweeps training trajectories via :meth:`validate_against_training`
        and passes only if:
        1) every trajectory runs and yields a reduction_ratio,
        2) worst reduction_ratio <= ``max_worst_ratio`` (when configured), and
        3) :meth:`validate_action_realizability` passes (when enabled), and
        4) within-episode target state stays invariant to current-observation frame.

        ``trajectories=None`` uses the configured training set; if no trajectories
        are configured, the caller should treat the gate as opted out.
        """
        import re as _re

        trajs = (
            [int(t) for t in trajectories]
            if trajectories is not None
            else self._fidelity_train_trajectories()
        )
        if max_worst_ratio is None:
            max_worst_ratio = self._gate_max_worst_ratio
        if require_action_realizability is None:
            require_action_realizability = self._gate_require_action_realizability
        if realizability_margin is None:
            realizability_margin = self._gate_realizability_margin
        if require_goal_invariance is None:
            require_goal_invariance = self._gate_require_goal_invariance
        if require_cem_suite is None:
            require_cem_suite = self._gate_require_cem_suite
        if require_p2_safety is None:
            require_p2_safety = self._gate_require_p2_safety
        if goal_invariance_abs_tol is None:
            goal_invariance_abs_tol = self._gate_goal_invariance_abs_tol
        if goal_invariance_rel_tol is None:
            goal_invariance_rel_tol = self._gate_goal_invariance_rel_tol
        if require_target_independence is None:
            require_target_independence = self._gate_require_target_independence
        if require_state_consistency is None:
            require_state_consistency = self._gate_require_state_consistency
        if require_fit_quality is None:
            require_fit_quality = self._gate_require_fit_quality
        if fit_quality_max_rmse is None:
            fit_quality_max_rmse = self._gate_fit_quality_max_rmse

        per_traj: dict[int, float | None] = {}
        failures: list[tuple[int, str]] = []
        rows: list[tuple[int, float]] = []
        worst_summary = ""
        for t in trajs:
            t = int(t)
            try:
                summary = self.validate_against_training(trajectory_index=t)
            except Exception as exc:  # never let the gate crash the loop
                per_traj[t] = None
                failures.append((t, f"{type(exc).__name__}: {exc}"))
                continue
            m = _re.search(r"reduction_ratio=([0-9.]+|n/a)", summary)
            if m and m.group(1) != "n/a":
                ratio = float(m.group(1))
                per_traj[t] = ratio
                rows.append((t, ratio))
            else:
                per_traj[t] = None
                failures.append(
                    (t, "ran but produced no reduction_ratio (crash/degenerate)")
                )
                if not worst_summary:
                    worst_summary = summary  # surface a crash detail in the complaint

        stats = summarize_reduction_ratios(per_traj)
        candidates = candidate_gate_criteria(stats)
        n_failed = int(stats["n_failed"])
        worst = stats["worst"]
        mean = stats["mean"]
        crash_ok = n_failed == 0
        ratio_ok = (
            crash_ok
            and (
                max_worst_ratio is None
                or (worst is not None and float(worst) <= float(max_worst_ratio))
            )
        )
        realizability = None
        realizability_ok = True
        if require_action_realizability:
            realizability = self._run_action_realizability_check(
                margin=float(realizability_margin),
            )
            realizability_ok = bool(realizability.get("passed"))

        goal_invariance = None
        goal_invariance_ok = True
        if require_goal_invariance:
            try:
                goal_invariance = self._run_goal_invariance_check(
                    trajectories=trajs,
                    abs_tolerance=float(goal_invariance_abs_tol),
                    rel_tolerance=float(goal_invariance_rel_tol),
                )
            except Exception as exc:  # gate checks must never crash the loop
                goal_invariance = {
                    "passed": False,
                    "error": f"goal-invariance check errored: {type(exc).__name__}: {exc}",
                }
            goal_invariance_ok = bool(goal_invariance.get("passed"))

        target_independence = None
        target_independence_ok = True
        if require_target_independence:
            try:
                target_independence = self._run_target_independence_check(
                    trajectories=trajs,
                )
            except Exception as exc:
                target_independence = {
                    "passed": False,
                    "error": f"target-independence check errored: {type(exc).__name__}: {exc}",
                }
            target_independence_ok = bool(target_independence.get("passed"))

        cem_suite = None
        cem_suite_ok = True
        if require_cem_suite:
            cem_suite = self._run_cem_suite_check(trajs)
            cem_suite_ok = bool(cem_suite.get("passed"))

        p2_safety = None
        p2_safety_ok = True
        if require_p2_safety:
            p2_safety = self._run_p2_safety_check()
            p2_safety_ok = bool(p2_safety.get("passed"))

        deployment_goal = None
        deployment_goal_ok = True
        if self._deployment_goal_mode != "final":
            try:
                deployment_goal = self._run_deployment_goal_check(trajs)
            except Exception as exc:  # gate checks must never crash the loop
                deployment_goal = {
                    "passed": False,
                    "error": (
                        "deployment-goal check errored: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            deployment_goal_ok = bool(deployment_goal.get("passed"))

        # Perception-anchored state consistency on the worst-ratio trajectory,
        # to make the ratio complaint actionable ("which component, from which
        # step"). Optional gate requirement in measurement-selector runs.
        state_consistency = None
        if rows:
            worst_row_t = max(rows, key=lambda tr: tr[1])[0]
            try:
                state_consistency = self._run_state_consistency_check(
                    worst_row_t, max_checkpoints=8
                )
            except Exception as exc:
                state_consistency = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "trajectory": worst_row_t,
                }

        state_consistency_ok = True
        if require_state_consistency:
            state_consistency_ok = bool(
                state_consistency
                and state_consistency.get("ok")
                and state_consistency.get("components")
                and not state_consistency.get("drifting_components")
                and not state_consistency.get("parse_errors")
            )

        fit_quality = None
        fit_quality_ok = True
        if require_fit_quality:
            fit_quality = self._run_fit_quality_check(
                trajs,
                max_rmse=float(fit_quality_max_rmse),
            )
            fit_quality_ok = bool(fit_quality.get("passed"))

        ratio_gates_pass = bool(ratio_ok or self._gate_ratio_advisory)
        passed = bool(
            crash_ok
            and ratio_gates_pass
            and realizability_ok
            and goal_invariance_ok
            and target_independence_ok
            and cem_suite_ok
            and p2_safety_ok
            and deployment_goal_ok
            and state_consistency_ok
            and fit_quality_ok
        )
        complaint = ""
        if not passed:
            ratio_line = (
                ", ".join(f"t{t}={r:.3f}" for t, r in rows) or "(none ran)"
            )
            lines = [
                "[MANDATORY FINAL VALIDATION GATE — before you finish, your "
                "simulator must be functional, faithful, and realizable.]",
                f"  trajectories that ran: {ratio_line}",
            ]
            if not crash_ok:
                fail_line = ", ".join(f"t{t} ({why})" for t, why in failures)
                lines.append(
                    "  FAIL (crash): could not produce reduction_ratio on "
                    f"{fail_line}"
                )
                if worst_summary:
                    lines.append(f"  failure detail:\n{worst_summary}")
            if max_worst_ratio is not None and not ratio_ok:
                ratio_tag = (
                    "ADVISORY (fidelity)"
                    if self._gate_ratio_advisory
                    else "FAIL (fidelity)"
                )
                lines.append(
                    f"  {ratio_tag}: worst reduction_ratio "
                    f"{('n/a' if worst is None else f'{worst:.3f}')} exceeds "
                    f"the gate threshold {float(max_worst_ratio):.3f}."
                    + (
                        " (Advisory only — fix the FAIL items first.)"
                        if self._gate_ratio_advisory
                        else ""
                    )
                )
                if state_consistency and state_consistency.get("ok"):
                    lines.append(
                        "  HINT (state-consistency on worst trajectory "
                        f"t{state_consistency['trajectory']} — which component "
                        "diverges from your own fit() parse of the true "
                        "frames, and from when):"
                    )
                    lines.extend(
                        "    " + line
                        for line in state_consistency.get("compact_lines", [])
                    )
            if require_state_consistency and not state_consistency_ok:
                if not state_consistency or not state_consistency.get("ok"):
                    lines.append(
                        "  FAIL (state-consistency): "
                        f"{(state_consistency or {}).get('error', 'check unavailable')}"
                    )
                else:
                    lines.append(
                        "  FAIL (state-consistency): open-loop rollout diverges "
                        "from the simulator's own fit() parse of true frames. "
                        "Fix update(a) before shipping."
                    )
                    lines.extend(
                        "    " + line
                        for line in state_consistency.get("compact_lines", [])
                    )
            if require_fit_quality and not fit_quality_ok:
                if fit_quality and fit_quality.get("error"):
                    lines.append(
                        "  FAIL (fit-quality): "
                        f"{fit_quality['error']}"
                    )
                else:
                    n_vis = int((fit_quality or {}).get("n_visibility_failures") or 0)
                    worst_rmse = (fit_quality or {}).get("worst_normalized_rmse")
                    worst_rmse_text = (
                        "n/a"
                        if worst_rmse is None
                        else f"{float(worst_rmse):.3f}"
                    )
                    lines.append(
                        "  FAIL (fit-quality): render_frame() immediately after "
                        "fit() does not structurally match the true frame "
                        f"({n_vis} visibility failures). "
                        f"worst_normalized_rmse={worst_rmse_text} "
                        f"(threshold {float(fit_quality_max_rmse):.3f}). Fix fit()/render_frame() before dynamics."
                )
            if require_action_realizability and not realizability_ok:
                if realizability and realizability.get("error"):
                    lines.append(
                        "  FAIL (realizability): "
                        f"{realizability['error']}"
                    )
                else:
                    worst_v = (realizability or {}).get("worst_violation") or {}
                    lines.append(
                        "  FAIL (realizability): one-step state jump exceeds "
                        "training-derived bound. "
                        f"Component={worst_v.get('component')}, "
                        f"delta={float(worst_v.get('delta', 0.0)):.3f}, "
                        f"bound={float(worst_v.get('bound', 0.0)):.3f}."
                    )
            if require_goal_invariance and not goal_invariance_ok:
                if goal_invariance and goal_invariance.get("error"):
                    lines.append(
                        "  FAIL (goal-invariance): "
                        f"{goal_invariance['error']}"
                    )
                else:
                    worst_goal = (goal_invariance or {}).get("worst_violation") or {}
                    lines.append(
                        "  FAIL (goal-invariance): target_state drifts within one episode "
                        "when only the current observation frame changes. "
                        f"Trajectory={worst_goal.get('trajectory_index')}, "
                        f"component={worst_goal.get('component')}, "
                        f"drift={float(worst_goal.get('drift', 0.0)):.3f}, "
                        f"tol={float(worst_goal.get('tolerance', 0.0)):.3f}."
                    )
            if require_target_independence and not target_independence_ok:
                if target_independence and target_independence.get("error"):
                    lines.append(
                        "  FAIL (target-independence): "
                        f"{target_independence['error']}"
                    )
                else:
                    if (target_independence or {}).get("leak_violations"):
                        w = (target_independence or {}).get("worst_leak") or {}
                        lines.append(
                            "  FAIL (target-independence): identical actions from the same "
                            "start produced different state trajectories under two different "
                            f"goals (component={w.get('component')}, step={w.get('step')}, "
                            f"drift={float(w.get('drift', 0.0)):.4f} > tol={float(w.get('tolerance', 0.0)):.4f}). "
                            "update(a) must never read self.target_state or goal-derived values."
                        )
                    if (target_independence or {}).get("null_violations"):
                        w = (target_independence or {}).get("worst_null") or {}
                        lines.append(
                            "  FAIL (null-action drift): terminal_cost fell from "
                            f"{float(w.get('cost_initial', 0.0)):.4f} to {float(w.get('cost_final', 0.0)):.4f} "
                            "under all-zero actions. Cost must reflect physical progress toward "
                            "the target, not elapsed rollout steps."
                        )
            if require_cem_suite and not cem_suite_ok:
                if cem_suite and cem_suite.get("error"):
                    lines.append(f"  FAIL (CEM suite): {cem_suite['error']}")
                else:
                    failed = [
                        r for r in (cem_suite or {}).get("results", [])
                        if not r.get("passed")
                    ]
                    if failed:
                        pair = failed[0].get("pair", {})
                        lines.append(
                            "  FAIL (CEM suite): planner failed on "
                            f"start=traj{pair.get('start_trajectory_index')}/frame{pair.get('start_frame_index')} "
                            f"goal=traj{pair.get('goal_trajectory_index')}/frame{pair.get('goal_frame_index')} "
                            f"({len(failed)}/{len((cem_suite or {}).get('results', []))} pairs failed). "
                            "Revise update(a) or terminal_cost so CEM can drive the explicit state to the target."
                        )
                        first_fail = failed[0].get("summary", "")
                        for fl in first_fail.splitlines():
                            if "FAIL:" in fl:
                                lines.append("    reason: " + fl.strip()[:400])
                                break
            if require_p2_safety and not p2_safety_ok:
                lines.append(
                    "  FAIL (runtime-safety): simulator crashed or violated the P2 contract when "
                    "used by CEM. WorldAPI/SAM/VLM calls are allowed in fit(image_A, image_B) "
                    "when enabled, but rollout methods must be local and deterministic. "
                    f"{(p2_safety or {}).get('error', '')[:500]}"
                )
            if deployment_goal is not None and not deployment_goal_ok:
                if deployment_goal.get("error"):
                    lines.append(
                        "  FAIL (deployment-goal): "
                        f"{deployment_goal['error']}"
                    )
                else:
                    n_bad = deployment_goal.get("n_bad", 0)
                    n_probes = deployment_goal.get("n_probes", 0)
                    kinds = {
                        p.get("verdict")
                        for p in deployment_goal.get("probes", [])
                        if p.get("verdict") not in (None, "ok")
                    }
                    lines.append(
                        "  FAIL (deployment-goal): at test time image_B is "
                        "built from the CURRENT observation, not a "
                        "completed-task frame. Probing fit(frame_0, frame_0) "
                        f"failed on {n_bad}/{n_probes} trajectories "
                        f"({', '.join(sorted(kinds))}): target_state came "
                        "back ≈ the movable components' current state, so "
                        "terminal_cost starts near 0 and the planner has no "
                        "incentive to act. Derive target_state from "
                        "persistent goal cues visible in image_B (static "
                        "markers indicating where components must GO), never "
                        "from the movable components' own pose."
                    )
            complaint = "\n".join(lines)

        return {
            "passed": passed,
            "per_traj": per_traj,
            "worst": worst,
            "mean": mean,
            "median": stats["median"],
            "p75": stats["p75"],
            "best": stats["best"],
            "ratio_stats": stats,
            "candidate_criteria": candidates,
            "n_failed": n_failed,
            "failures": failures,
            "crash_ok": crash_ok,
            "ratio_ok": ratio_ok,
            "ratio_advisory": self._gate_ratio_advisory,
            "realizability_ok": realizability_ok,
            "goal_invariance_ok": goal_invariance_ok,
            "max_worst_ratio": max_worst_ratio,
            "require_action_realizability": require_action_realizability,
            "realizability_margin": realizability_margin,
            "realizability": realizability,
            "require_goal_invariance": require_goal_invariance,
            "goal_invariance_abs_tol": goal_invariance_abs_tol,
            "goal_invariance_rel_tol": goal_invariance_rel_tol,
            "goal_invariance": goal_invariance,
            "require_target_independence": require_target_independence,
            "target_independence_ok": target_independence_ok,
            "target_independence": target_independence,
            "require_cem_suite": require_cem_suite,
            "cem_suite_ok": cem_suite_ok,
            "cem_suite": cem_suite,
            "require_p2_safety": require_p2_safety,
            "p2_safety_ok": p2_safety_ok,
            "p2_safety": p2_safety,
            "deployment_goal_mode": self._deployment_goal_mode,
            "deployment_goal_ok": deployment_goal_ok,
            "deployment_goal": deployment_goal,
            "require_state_consistency": require_state_consistency,
            "state_consistency_ok": state_consistency_ok,
            "state_consistency": state_consistency,
            "require_fit_quality": require_fit_quality,
            "fit_quality_ok": fit_quality_ok,
            "fit_quality": fit_quality,
            "complaint": complaint,
        }

    def write_code_from_scratch(self, code: str) -> str:
        msg = super().write_code_from_scratch(code)
        self._keep_best_observe_code_change()
        return (
            msg
            + self._auto_validate_feedback()
            + self._auto_visual_reconstruction_feedback()
        )

    def edit_code(self, search: str, replace: str) -> str:
        msg = super().edit_code(search, replace)
        self._keep_best_observe_code_change()
        return (
            msg
            + self._auto_validate_feedback()
            + self._auto_visual_reconstruction_feedback()
        )

    # ------------------------------------------------------------------
    # Keep-best checkpointing (harness-side; never VLM-visible)
    # ------------------------------------------------------------------

    def _score_full_train_fidelity(self) -> dict[str, Any]:
        """Quietly score the current sandbox code over the full training set.

        Runs :meth:`validate_against_training` on every fidelity trajectory
        with tool-call logging suppressed (indices stay VLM-only). Returns
        ``{"per_traj", "worst", "mean", "n_failed"}`` where worst/mean cover
        the trajectories that produced a ratio.

        Cost controls: non-compiling code is scored degenerate without any
        replays, trajectories are swept worst-first (per the last score), and
        the sweep aborts early once a ratio already exceeds the incumbent
        best checkpoint's worst — a partial sweep can only prove "not
        better", which is all the keep-best comparison needs (``aborted``
        marks such scores so they are never checkpointed).
        """
        import re as _re

        trajs = [int(t) for t in self._fidelity_train_trajectories()]
        if not self.has_valid_simulator_class():
            return {
                "per_traj": {t: None for t in trajs},
                "worst": None,
                "mean": None,
                "n_failed": len(trajs),
            }
        prev = (self._keep_best_last or {}).get("score") or {}
        prev_ratios = {
            int(k): v
            for k, v in (prev.get("per_traj") or {}).items()
            if v is not None
        }
        trajs.sort(key=lambda t: -prev_ratios.get(t, float("inf")))
        best_score = (self._keep_best_best or {}).get("score") or {}
        abort_above = (
            None
            if self._gate_require_fit_quality
            or self._gate_require_state_consistency
            else best_score.get("worst")
        )

        per_traj: dict[int, float | None] = {}
        ratios: list[float] = []
        n_failed = 0
        aborted = False
        self._suppress_tool_call_log = True
        try:
            for t in trajs:
                try:
                    summary = self.validate_against_training(trajectory_index=t)
                except Exception:
                    per_traj[t] = None
                    n_failed += 1
                    continue
                m = _re.search(r"reduction_ratio=([0-9.]+|n/a)", summary)
                if m and m.group(1) != "n/a":
                    ratio = float(m.group(1))
                    per_traj[t] = ratio
                    ratios.append(ratio)
                    if abort_above is not None and ratio > float(abort_above):
                        aborted = True
                        break
                else:
                    per_traj[t] = None
                    n_failed += 1
        finally:
            self._suppress_tool_call_log = False
        score: dict[str, Any] = {
            "per_traj": per_traj,
            "worst": max(ratios) if ratios else None,
            "mean": (sum(ratios) / len(ratios)) if ratios else None,
            "n_failed": n_failed,
        }
        if aborted:
            score["aborted"] = True
        quality_checks = 0
        quality_failures = 0
        if not aborted and int(n_failed) == 0 and ratios:
            self._suppress_tool_call_log = True
            original_log_dir = self._tool_calls_log_dir
            self._tool_calls_log_dir = None
            try:
                if self._gate_require_fit_quality:
                    quality_checks += 1
                    fit_quality = self._run_fit_quality_check(
                        trajs,
                        max_rmse=self._gate_fit_quality_max_rmse,
                        max_probes=3,
                    )
                    score["fit_quality_passed"] = bool(fit_quality.get("passed"))
                    score["fit_quality_worst_rmse"] = fit_quality.get(
                        "worst_normalized_rmse"
                    )
                    if not score["fit_quality_passed"]:
                        quality_failures += 1
                if self._gate_require_state_consistency:
                    quality_checks += 1
                    worst_t = max(
                        per_traj,
                        key=lambda t: (
                            float("-inf")
                            if per_traj[t] is None
                            else float(per_traj[t])
                        ),
                    )
                    consistency = self._run_state_consistency_check(
                        int(worst_t),
                        max_checkpoints=8,
                    )
                    state_consistency_passed = bool(
                        consistency.get("ok")
                        and consistency.get("components")
                        and not consistency.get("drifting_components")
                        and not consistency.get("parse_errors")
                    )
                    score["state_consistency_passed"] = state_consistency_passed
                    if not state_consistency_passed:
                        quality_failures += 1
            except Exception as exc:
                quality_failures += max(1, quality_checks - quality_failures)
                score["quality_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                self._tool_calls_log_dir = original_log_dir
                self._suppress_tool_call_log = False
        score["quality_checks"] = int(quality_checks)
        score["quality_failures"] = int(quality_failures)
        score.setdefault("cem_failures", 0)
        return score

    @staticmethod
    def _keep_best_score_is_better(
        candidate: dict[str, Any] | None,
        incumbent: dict[str, Any] | None,
    ) -> bool:
        """Strictly-better ordering: valid beats invalid, then quality, CEM, ratio."""
        def _valid(s: dict[str, Any] | None) -> bool:
            return bool(
                s is not None
                and s.get("worst") is not None
                and int(s.get("n_failed") or 0) == 0
                and not s.get("aborted")
            )

        if not _valid(candidate):
            return False
        if not _valid(incumbent):
            return True
        candidate_quality = int(candidate.get("quality_failures") or 0)
        incumbent_quality = int(incumbent.get("quality_failures") or 0)
        if candidate_quality != incumbent_quality:
            return candidate_quality < incumbent_quality
        candidate_cem = int(candidate.get("cem_failures") or 0)
        incumbent_cem = int(incumbent.get("cem_failures") or 0)
        if candidate_cem != incumbent_cem:
            return candidate_cem < incumbent_cem
        cw, iw = float(candidate["worst"]), float(incumbent["worst"])
        if abs(cw - iw) > 1e-12:
            return cw < iw
        cm = candidate.get("mean")
        im = incumbent.get("mean")
        if cm is not None and im is not None:
            return float(cm) < float(im) - 1e-12
        return False

    def _keep_best_observe_code_change(self) -> None:
        """Score the just-written code and checkpoint it if it is the best so far."""
        if not self._keep_best_enabled:
            return
        # Index of the write/edit tool call itself (super() logged it already).
        call_idx = max(0, self._tool_call_idx - 1)
        try:
            code = Path(self._sandbox_path).read_text(encoding="utf-8")
        except OSError:
            return
        try:
            score = self._score_full_train_fidelity()
        except Exception:
            score = None
        entry = {"tool_call": call_idx, "code": code, "score": score}
        self._keep_best_last = entry
        self._keep_best_candidates.append(entry)
        # Keep memory bounded during long repair loops. This list is only used
        # for final bounded CEM reranking; preserving the best ratio/quality
        # candidates is sufficient.
        if len(self._keep_best_candidates) > 24:
            ordered: list[dict[str, Any]] = []
            seen_code: set[str] = set()
            for candidate in reversed(self._keep_best_candidates):
                if candidate["code"] in seen_code:
                    continue
                seen_code.add(candidate["code"])
                insert_at = len(ordered)
                for index, incumbent in enumerate(ordered):
                    if self._keep_best_score_is_better(
                        candidate.get("score"),
                        incumbent.get("score"),
                    ):
                        insert_at = index
                        break
                ordered.insert(insert_at, candidate)
            self._keep_best_candidates = ordered[:12]
        if self._keep_best_score_is_better(
            score, (self._keep_best_best or {}).get("score")
        ):
            self._keep_best_best = entry

    def _rerank_keep_best_with_cem(self, max_candidates: int = 3) -> None:
        """CEM-rerank only the top few checkpoints once, immediately before restore."""
        if (
            not self._keep_best_include_cem
            or not self._gate_require_cem_suite
            or not self._gate_trajectories
            or not self._keep_best_candidates
        ):
            return

        ordered: list[dict[str, Any]] = []
        seen_code: set[str] = set()
        for candidate in reversed(self._keep_best_candidates):
            if candidate["code"] in seen_code:
                continue
            seen_code.add(candidate["code"])
            insert_at = len(ordered)
            for index, incumbent in enumerate(ordered):
                if self._keep_best_score_is_better(
                    candidate.get("score"),
                    incumbent.get("score"),
                ):
                    insert_at = index
                    break
            ordered.insert(insert_at, candidate)
        candidates = ordered[: max(1, int(max_candidates))]
        try:
            original_code = Path(self._sandbox_path).read_text(encoding="utf-8")
        except OSError:
            return

        evaluated: list[dict[str, Any]] = []
        original_log_dir = self._tool_calls_log_dir
        self._tool_calls_log_dir = None
        self._suppress_tool_call_log = True
        try:
            for candidate in candidates:
                score = dict(candidate.get("score") or {})
                if int(score.get("quality_failures") or 0) > 0:
                    score["cem_failures"] = 999
                    score["cem_suite_passed"] = False
                else:
                    Path(self._sandbox_path).write_text(
                        candidate["code"],
                        encoding="utf-8",
                    )
                    try:
                        cem = self._run_cem_suite_check(
                            list(self._gate_trajectories),
                            n_pairs=3,
                            horizon=15,
                            cem_iters=3,
                            cem_population=50,
                        )
                        results = list(cem.get("results") or [])
                        if cem.get("passed"):
                            failures = 0
                        elif results:
                            failures = sum(
                                0 if row.get("passed") else 1 for row in results
                            )
                        else:
                            failures = max(
                                1,
                                int(len(cem.get("pairs") or []) or 1),
                            )
                        score["cem_failures"] = int(failures)
                        score["cem_suite_passed"] = bool(cem.get("passed"))
                    except Exception as exc:
                        score["cem_failures"] = 999
                        score["cem_suite_passed"] = False
                        score["cem_error"] = f"{type(exc).__name__}: {exc}"
                evaluated.append({**candidate, "score": score})
        finally:
            Path(self._sandbox_path).write_text(original_code, encoding="utf-8")
            self._tool_calls_log_dir = original_log_dir
            self._suppress_tool_call_log = False

        best: dict[str, Any] | None = None
        for candidate in evaluated:
            if self._keep_best_score_is_better(
                candidate.get("score"),
                (best or {}).get("score"),
            ):
                best = candidate
        if best is not None:
            self._keep_best_best = best
        self._keep_best_cem_record = {
            "cem_reranked": True,
            "cem_candidates_evaluated": len(evaluated),
            "candidates": [
                {
                    "tool_call": candidate["tool_call"],
                    "worst_ratio": (candidate.get("score") or {}).get("worst"),
                    "quality_failures": (candidate.get("score") or {}).get(
                        "quality_failures"
                    ),
                    "cem_failures": (candidate.get("score") or {}).get(
                        "cem_failures"
                    ),
                }
                for candidate in evaluated
            ],
        }

    def restore_best_checkpoint(self) -> bool:
        """Restore the best-scoring code version into the sandbox file.

        Returns True only when the sandbox content actually changed. Always
        refreshes the record returned by :meth:`get_keep_best_record` and, when
        a tool-call log dir exists, persists it as ``keep_best.json``.
        """
        self._rerank_keep_best_with_cem()
        best = self._keep_best_best
        last = self._keep_best_last
        record: dict[str, Any] = {
            "keep_best_enabled": self._keep_best_enabled,
            "shipped_checkpoint_tool_call": None,
            "shipped_worst_ratio": None,
            "final_sandbox_worst_ratio": (
                (last.get("score") or {}).get("worst")
                if last is not None
                else None
            ),
            "keep_best_restored": False,
            **(self._keep_best_cem_record or {}),
        }
        restored = False
        if self._keep_best_enabled and best is not None:
            record["shipped_checkpoint_tool_call"] = best["tool_call"]
            record["shipped_worst_ratio"] = (best.get("score") or {}).get("worst")
            try:
                current = Path(self._sandbox_path).read_text(encoding="utf-8")
            except OSError:
                current = None
            if current != best["code"]:
                Path(self._sandbox_path).write_text(
                    best["code"], encoding="utf-8"
                )
                restored = True
        record["keep_best_restored"] = restored
        self._keep_best_record = record
        self._persist_keep_best_record()
        return restored

    def get_keep_best_record(self) -> dict[str, Any]:
        """Keep-best bookkeeping for metrics/logs (computed lazily if needed)."""
        if self._keep_best_record is not None:
            return dict(self._keep_best_record)
        best = self._keep_best_best
        last = self._keep_best_last
        return {
            "keep_best_enabled": self._keep_best_enabled,
            "shipped_checkpoint_tool_call": (
                best["tool_call"] if self._keep_best_enabled and best else None
            ),
            "shipped_worst_ratio": (
                (best.get("score") or {}).get("worst")
                if self._keep_best_enabled and best
                else None
            ),
            "final_sandbox_worst_ratio": (
                (last.get("score") or {}).get("worst")
                if last is not None
                else None
            ),
            "keep_best_restored": False,
        }

    def _persist_keep_best_record(self) -> None:
        if not self._tool_calls_log_dir or self._keep_best_record is None:
            return
        try:
            path = Path(self._tool_calls_log_dir) / "keep_best.json"
            path.write_text(
                json.dumps(self._keep_best_record, indent=2, default=str),
                encoding="utf-8",
            )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Dataset tools exposed to the VLM
    # ------------------------------------------------------------------

    def _trajectory_contact_sheet(
        self,
        *,
        idx: int,
        traj: dict,
        summary: str,
        max_frames: int,
    ) -> Image.Image:
        n_frames = len(traj["frames"])
        n_steps = int(traj["n_steps"])
        max_frames = max(2, min(int(max_frames), n_frames))
        frame_indices = np.linspace(0, n_frames - 1, max_frames, dtype=int).tolist()
        # De-duplicate indices for very short trajectories while preserving order.
        frame_indices = list(dict.fromkeys(frame_indices))

        thumbs: list[Image.Image] = []
        labels: list[list[str]] = []
        for frame_idx in frame_indices:
            img = Image.open(traj["frames"][frame_idx]).convert("RGB")
            img.thumbnail((120, 120), Image.Resampling.NEAREST)
            thumbs.append(img.copy())

            label = [f"frame {frame_idx}/{n_frames - 1}"]
            if n_steps > 0:
                if frame_idx < n_steps:
                    action_idx = frame_idx
                else:
                    action_idx = n_steps - 1
                action = np.asarray(traj["actions"][action_idx], dtype=np.float64)
                action_str = np.array2string(
                    action,
                    precision=2,
                    suppress_small=True,
                    max_line_width=80,
                )
                label.append(f"a[{action_idx}]={action_str}")
            labels.append(label)

        cell_w = max(140, max(t.width for t in thumbs) + 16)
        cell_h = max(170, max(t.height for t in thumbs) + 48)
        cols = min(4, len(thumbs))
        rows = int(np.ceil(len(thumbs) / cols))
        header_h = 76
        sheet = Image.new("RGB", (cols * cell_w, header_h + rows * cell_h), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((8, 8), f"Trajectory {idx} contact sheet", fill=(0, 0, 0))
        draw.text((8, 28), summary, fill=(0, 0, 0))
        draw.text(
            (8, 48),
            "Evenly spaced frames; action label is the action from this frame when available.",
            fill=(80, 80, 80),
        )

        for j, (thumb, label_lines) in enumerate(zip(thumbs, labels)):
            row, col = divmod(j, cols)
            x0 = col * cell_w
            y0 = header_h + row * cell_h
            x = x0 + (cell_w - thumb.width) // 2
            y = y0 + 8
            sheet.paste(thumb, (x, y))
            draw.rectangle((x0 + 4, y0 + 4, x0 + cell_w - 5, y0 + cell_h - 5), outline=(210, 210, 210))
            text_y = y0 + thumb.height + 14
            for line in label_lines:
                draw.text((x0 + 8, text_y), line, fill=(0, 0, 0))
                text_y += 14
        return sheet

    def read_trajectory(
        self,
        trajectory_index: int,
        include_images: bool | None = None,
        max_frames: int = 8,
    ) -> Image.Image | str:
        """Inspect one training trajectory with a visual contact sheet by default.

        Use this to understand trajectory-level dynamics before reading specific
        frames or actions. Trajectory indices are zero-based and dataset-specific.
        Set `include_images=False` to return metadata text only. The default can
        also be disabled with `VDAWORLD_READ_TRAJECTORY_VISUAL=0`.

        Args:
            trajectory_index: Which trajectory to summarise.
            include_images: Whether to attach a contact sheet of sampled frames.
            max_frames: Maximum number of evenly spaced frames in the contact sheet.

        Returns:
            A contact sheet image with embedded summary/action labels, or a
            short metadata string when visual mode is disabled. Use view_image
            only for close-up inspection of individual frames.
        """
        try:
            idx = int(trajectory_index)
            traj = _dt.read_trajectory(self._dataset_dir, idx)
        except (FileNotFoundError, ValueError) as exc:
            err = f"[read_trajectory] {exc}"
            self._log_critic_tool_call(
                tool_name="read_trajectory",
                args={"trajectory_index": str(trajectory_index)},
                result_summary=err,
            )
            return err

        n_frames = len(traj["frames"])
        n_steps = traj["n_steps"]
        action_dim = traj["actions"].shape[1] if traj["actions"].ndim == 2 else 0
        action_min = float(traj["actions"].min()) if n_steps > 0 else 0.0
        action_max = float(traj["actions"].max()) if n_steps > 0 else 0.0

        result = (
            f"Trajectory {idx}: {n_frames} frame(s) (indices 0..{n_frames - 1}), "
            f"{n_steps} action(s) of dim {action_dim} "
            f"in range [{action_min:.3f}, {action_max:.3f}]."
        )
        if n_steps > 0 and action_dim > 0:
            action_rows = []
            for dim in range(action_dim):
                values = traj["actions"][:, dim]
                action_rows.append(
                    f"a[{dim}] min={float(values.min()):.4f} "
                    f"max={float(values.max()):.4f} "
                    f"mean={float(values.mean()):.4f} "
                    f"sum={float(values.sum()):.4f}"
                )
            result += " Per-dimension statistics: " + "; ".join(action_rows) + "."
        action_contract = traj.get("action_contract")
        if isinstance(action_contract, dict) and action_contract:
            semantics = str(action_contract.get("semantics", "unspecified"))
            action_repeat = action_contract.get("action_repeat")
            declared_low = action_contract.get("low")
            declared_high = action_contract.get("high")
            result += (
                " DECLARED ACTION CONTRACT (authoritative): "
                f"semantics={semantics!r}"
                + (
                    f", action_repeat={int(action_repeat)}"
                    if action_repeat is not None
                    else ""
                )
                + (
                    f", bounds low={declared_low} high={declared_high}"
                    if declared_low is not None and declared_high is not None
                    else ""
                )
                + "."
            )
            semantics_lower = semantics.lower()
            if any(
                token in semantics_lower
                for token in ("torque", "force", "acceleration")
            ):
                result += (
                    " This is force-like control, not a pose increment: model "
                    "velocity/momentum and damping, and verify sign/lag from "
                    "multi-step state consistency."
                )
            elif any(
                token in semantics_lower
                for token in ("setpoint", "absolute", "target_position")
            ):
                result += (
                    " This is an absolute/setpoint action, not a displacement; "
                    "derive motion from action minus current controlled state."
                )
            if self._deployed_action_bounds is not None:
                deploy_low, deploy_high = self._deployed_action_bounds
                result += (
                    " P2 deployed bounds: "
                    f"low={deploy_low.tolist()} high={deploy_high.tolist()}."
                )
                if declared_low is not None and declared_high is not None:
                    low_arr = np.asarray(declared_low, dtype=np.float64).reshape(-1)
                    high_arr = np.asarray(declared_high, dtype=np.float64).reshape(-1)
                    bounds_match = bool(
                        low_arr.shape == deploy_low.shape
                        and high_arr.shape == deploy_high.shape
                        and np.allclose(low_arr, deploy_low)
                        and np.allclose(high_arr, deploy_high)
                    )
                    result += (
                        " Dataset/P2 bounds MATCH."
                        if bounds_match
                        else " WARNING: dataset and P2 action bounds DO NOT MATCH."
                    )
        state_contract = traj.get("state_contract")
        if isinstance(state_contract, dict) and state_contract:
            result += (
                " DECLARED STATE/TASK CONTRACT (authoritative): "
                + json.dumps(
                    state_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + ". Periodic coordinates may wrap; bounded non-periodic "
                "coordinates must never cross their limits. A branch-invariant "
                "task target must not force one visually equivalent pose branch."
            )
        visual_default = os.environ.get("VDAWORLD_READ_TRAJECTORY_VISUAL", "0") != "0"
        use_images = visual_default if include_images is None else bool(include_images)
        extra_files: dict[str, Any] = {
            "actions.json": json.dumps(traj["actions"].tolist())
        }
        if isinstance(state_contract, dict) and state_contract:
            extra_files["state_contract.json"] = json.dumps(
                state_contract,
                indent=2,
                sort_keys=True,
            )
        if use_images:
            sheet = self._trajectory_contact_sheet(
                idx=idx,
                traj=traj,
                summary=result,
                max_frames=max_frames,
            )
            extra_files["trajectory_contact_sheet.png"] = sheet
            return_value: Image.Image | str = sheet
            result_summary = (
                result
                + f" Returned visual contact sheet with {min(max(2, int(max_frames)), n_frames)} sampled frame(s)."
            )
        else:
            return_value = result
            result_summary = result

        self._log_critic_tool_call(
            tool_name="read_trajectory",
            args={
                "trajectory_index": idx,
                "include_images": str(include_images),
                "max_frames": int(max_frames),
            },
            result_summary=result_summary,
            extra_files=extra_files,
        )
        return return_value

    def analyze_action_effects(
        self,
        trajectory_indices: list[int] | None = None,
        max_trajectories: int = 3,
        min_track_area: int = 5,
    ) -> str:
        """Measure how observed actions correlate with visible object motion.

        This is a codegen-time system-identification aid. It does not know the
        benchmark semantics; it extracts coarse connected colour components from
        real frames, then reports which action dimensions correlate with each
        component's signed image displacement. These are advisory image-space
        regressions, not latent-state or physics gains.
        """
        traj_ids = self._measurement_trajectories(
            trajectory_indices, max_trajectories=max_trajectories
        )
        if not traj_ids:
            err = "[analyze_action_effects] no usable trajectories found."
            self._log_critic_tool_call(
                tool_name="analyze_action_effects",
                args={"trajectory_indices": trajectory_indices},
                result_summary=err,
            )
            return err

        action_rows: list[np.ndarray] = []
        component_series: dict[str, dict[str, list[float]]] = {}
        component_areas: dict[str, list[int]] = {}
        co_motion_rows: dict[tuple[str, str], dict[str, list[float]]] = {}
        errors: dict[int, str] = {}

        for tid in traj_ids:
            try:
                traj = _dt.read_trajectory(self._dataset_dir, int(tid))
            except (FileNotFoundError, ValueError) as exc:
                errors[int(tid)] = str(exc)
                continue
            actions = np.asarray(traj["actions"], dtype=np.float64)
            frames = list(traj["frames"])
            n = min(int(actions.shape[0]), max(0, len(frames) - 1))
            if n <= 0:
                continue
            try:
                tracks = [
                    self._extract_simple_color_tracks(
                        np.asarray(Image.open(p).convert("RGB")),
                        min_area=int(min_track_area),
                    )
                    for p in frames[: n + 1]
                ]
            except Exception as exc:
                errors[int(tid)] = f"{type(exc).__name__}: {exc}"
                continue

            for step in range(n):
                a = np.asarray(actions[step], dtype=np.float64).reshape(-1)
                action_rows.append(a)
                before = tracks[step]
                after = tracks[step + 1]
                labels = sorted(set(before) & set(after))
                deltas: dict[str, np.ndarray] = {}
                for label in labels:
                    p0 = np.asarray(before[label]["centroid"], dtype=np.float64)
                    p1 = np.asarray(after[label]["centroid"], dtype=np.float64)
                    delta = p1 - p0
                    deltas[label] = delta
                    rec = component_series.setdefault(
                        label,
                        {
                            "dx": [],
                            "dy": [],
                            "speed": [],
                            "x_before": [],
                            "y_before": [],
                            "x_after": [],
                            "y_after": [],
                            "action_index": [],
                            "trajectory": [],
                            "step": [],
                        },
                    )
                    rec["dx"].append(float(delta[0]))
                    rec["dy"].append(float(delta[1]))
                    rec["speed"].append(float(np.linalg.norm(delta)))
                    rec["x_before"].append(float(p0[0]))
                    rec["y_before"].append(float(p0[1]))
                    rec["x_after"].append(float(p1[0]))
                    rec["y_after"].append(float(p1[1]))
                    rec["action_index"].append(float(len(action_rows) - 1))
                    rec["trajectory"].append(int(tid))
                    rec["step"].append(int(step))
                    component_areas.setdefault(label, []).append(int(before[label]["area"]))

                for i, a_label in enumerate(labels):
                    for b_label in labels[i + 1 :]:
                        key = (a_label, b_label)
                        d0 = deltas.get(a_label)
                        d1 = deltas.get(b_label)
                        if d0 is None or d1 is None:
                            continue
                        dist = float(
                            np.linalg.norm(
                                np.asarray(before[a_label]["centroid"], dtype=np.float64)
                                - np.asarray(before[b_label]["centroid"], dtype=np.float64)
                            )
                        )
                        rec = co_motion_rows.setdefault(
                            key, {"dot": [], "dist": [], "speed_a": [], "speed_b": []}
                        )
                        rec["dot"].append(float(np.dot(d0, d1)))
                        rec["dist"].append(dist)
                        rec["speed_a"].append(float(np.linalg.norm(d0)))
                        rec["speed_b"].append(float(np.linalg.norm(d1)))

        if not action_rows:
            err = "[analyze_action_effects] no action/frame pairs could be measured."
            self._log_critic_tool_call(
                tool_name="analyze_action_effects",
                args={"trajectory_indices": traj_ids},
                result_summary=err,
                extra_files={"errors.json": json.dumps(errors, indent=2)},
            )
            return err

        actions = np.vstack(action_rows)
        action_stats = {
            "dim": int(actions.shape[1]),
            "n_steps": int(actions.shape[0]),
            "mean": actions.mean(axis=0).tolist(),
            "std": actions.std(axis=0).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        }
        all_training_actions = self._collect_training_actions()
        full_action_stats = None
        if (
            all_training_actions is not None
            and all_training_actions.ndim == 2
            and all_training_actions.shape[1] == actions.shape[1]
        ):
            full_action_stats = {
                "n_steps": int(all_training_actions.shape[0]),
                "mean": all_training_actions.mean(axis=0).tolist(),
                "std": all_training_actions.std(axis=0).tolist(),
                "min": all_training_actions.min(axis=0).tolist(),
                "max": all_training_actions.max(axis=0).tolist(),
            }

        components: dict[str, Any] = {}
        top_effects: list[dict[str, Any]] = []
        for label, series in sorted(component_series.items()):
            idxs = np.asarray(series["action_index"], dtype=int)
            if idxs.size < 3:
                continue
            comp_payload = {
                "n_steps": int(idxs.size),
                "mean_area": float(np.mean(component_areas.get(label, [0]))),
                "mean_speed": float(np.mean(series["speed"])),
                "effects": [],
            }
            for dim in range(actions.shape[1]):
                # Correlating a signed action with non-negative speed creates
                # high but uninterpretable effects when demonstrations are
                # action-biased. Keep only signed displacement axes.
                for axis in ("dx", "dy"):
                    y = np.asarray(series[axis], dtype=np.float64)
                    step_rows = np.asarray(series["step"], dtype=int)
                    best_lag = 0
                    corr = None
                    gain = None
                    for lag in range(4):
                        valid = step_rows >= lag
                        if int(np.sum(valid)) < 3:
                            continue
                        lagged_indices = idxs[valid] - lag
                        x = actions[lagged_indices, dim]
                        y_valid = y[valid]
                        candidate_corr = self._pearson_corr(x, y_valid)
                        candidate_gain = self._linear_gain(x, y_valid)
                        if candidate_corr is None:
                            continue
                        if corr is None or abs(candidate_corr) > abs(corr):
                            best_lag = int(lag)
                            corr = candidate_corr
                            gain = candidate_gain
                    effect = {
                        "action_dim": int(dim),
                        "component": label,
                        "axis": axis,
                        "corr": corr,
                        "gain": gain,
                        "action_lag": best_lag,
                    }
                    comp_payload["effects"].append(effect)
                    if corr is not None:
                        top_effects.append(effect)
            components[label] = comp_payload

        top_effects = sorted(
            top_effects,
            key=lambda e: abs(float(e["corr"])) if e.get("corr") is not None else -1.0,
            reverse=True,
        )[:12]

        setpoint_hints: list[dict[str, Any]] = []
        if actions.shape[1] >= 2:
            width, height = self._frame_size
            for label, series in sorted(component_series.items()):
                idxs = np.asarray(series["action_index"], dtype=int)
                if idxs.size < 3:
                    continue
                targets = actions[idxs, :2]
                in_image_coordinates = bool(
                    np.mean(
                        (targets[:, 0] >= 0.0)
                        & (targets[:, 0] <= float(width))
                        & (targets[:, 1] >= 0.0)
                        & (targets[:, 1] <= float(height))
                    )
                    >= 0.95
                )
                if not in_image_coordinates:
                    continue
                before = np.column_stack(
                    [series["x_before"], series["y_before"]]
                ).astype(np.float64)
                after = np.column_stack(
                    [series["x_after"], series["y_after"]]
                ).astype(np.float64)
                delta = after - before
                to_target = targets - before
                before_error = np.linalg.norm(to_target, axis=1)
                after_error = np.linalg.norm(targets - after, axis=1)
                toward = np.sum(delta * to_target, axis=1) > 0.0
                progress = before_error - after_error
                toward_fraction = float(np.mean(toward))
                mean_progress = float(np.mean(progress))
                if toward_fraction < 0.70 or mean_progress <= 0.0:
                    continue
                setpoint_hints.append(
                    {
                        "component": label,
                        "toward_fraction": toward_fraction,
                        "mean_distance_before": float(np.mean(before_error)),
                        "mean_distance_after": float(np.mean(after_error)),
                        "mean_progress": mean_progress,
                        "corr_next_x_action0": self._pearson_corr(
                            targets[:, 0], after[:, 0]
                        ),
                        "corr_next_y_action1": self._pearson_corr(
                            targets[:, 1], after[:, 1]
                        ),
                    }
                )
            setpoint_hints.sort(
                key=lambda row: (
                    float(row["toward_fraction"]),
                    float(row["mean_progress"]),
                ),
                reverse=True,
            )

        co_motion: list[dict[str, Any]] = []
        for (a_label, b_label), rec in sorted(co_motion_rows.items()):
            if len(rec["dot"]) < 3:
                continue
            dist = np.asarray(rec["dist"], dtype=np.float64)
            sp_a = np.asarray(rec["speed_a"], dtype=np.float64)
            sp_b = np.asarray(rec["speed_b"], dtype=np.float64)
            co_motion.append(
                {
                    "components": [a_label, b_label],
                    "n_steps": int(len(dist)),
                    "mean_distance": float(np.mean(dist)),
                    "close_fraction_lt_20px": float(np.mean(dist < 20.0)),
                    "speed_corr": self._pearson_corr(sp_a, sp_b),
                    "mean_motion_dot": float(np.mean(rec["dot"])),
                }
            )
        co_motion = sorted(
            co_motion,
            key=lambda r: (
                float(r["close_fraction_lt_20px"]),
                abs(float(r["speed_corr"])) if r["speed_corr"] is not None else 0.0,
            ),
            reverse=True,
        )[:8]

        payload = {
            "trajectories": traj_ids,
            "action_stats": action_stats,
            "full_training_action_stats": full_action_stats,
            "components": components,
            "top_effects": top_effects,
            "setpoint_hints": setpoint_hints,
            "co_motion_hints": co_motion,
            "errors": errors,
        }

        lines = [
            "[analyze_action_effects] measured visible tracks vs actions",
            (
                f"  trajectories={traj_ids} steps={action_stats['n_steps']} "
                f"action_dim={action_stats['dim']}"
            ),
            "  selected-trajectory action ranges: "
            + ", ".join(
                f"a[{i}]=[{lo:.3g},{hi:.3g}]"
                for i, (lo, hi) in enumerate(zip(action_stats["min"], action_stats["max"]))
            ),
        ]
        if full_action_stats is not None:
            lines.append(
                "  full-training action ranges: "
                + ", ".join(
                    f"a[{i}]=[{lo:.3g},{hi:.3g}]"
                    for i, (lo, hi) in enumerate(
                        zip(full_action_stats["min"], full_action_stats["max"])
                    )
                )
            )
        reliable_effects = [
            effect
            for effect in top_effects
            if effect.get("component") != "foreground"
            and effect.get("corr") is not None
            and abs(float(effect["corr"])) >= 0.30
        ]
        if reliable_effects:
            lines.append("  strongest measured action effects:")
            for e in reliable_effects[:8]:
                corr = e["corr"]
                gain = e["gain"]
                lines.append(
                    f"    {e['component']}.{e['axis']} vs a[{e['action_dim']}]: "
                    f"corr={corr:.3f} pixel_gain={gain:.3g} "
                    f"lag={int(e.get('action_lag', 0))}"
                    if corr is not None and gain is not None
                    else f"    {e['component']}.{e['axis']} vs a[{e['action_dim']}]: n/a"
                )
        else:
            lines.append(
                "  no reliable immediate signed component effect "
                "(|corr| >= 0.30); inspect lag/setpoint hints and explicit state."
            )
        if setpoint_hints:
            lines.append(
                "  possible absolute image-coordinate setpoint semantics "
                "(action resembles desired next XY, not displacement):"
            )
            for row in setpoint_hints[:4]:
                lines.append(
                    f"    {row['component']}: moves toward action XY on "
                    f"{100.0 * row['toward_fraction']:.0f}% of steps; "
                    f"mean distance {row['mean_distance_before']:.1f} -> "
                    f"{row['mean_distance_after']:.1f}px"
                )
        if co_motion:
            lines.append("  co-motion / attachment hints:")
            for r in co_motion[:5]:
                corr = r["speed_corr"]
                lines.append(
                    f"    {r['components'][0]} <-> {r['components'][1]}: "
                    f"mean_dist={r['mean_distance']:.1f}px "
                    f"close<20px={100.0 * r['close_fraction_lt_20px']:.0f}% "
                    f"speed_corr={'n/a' if corr is None else f'{corr:.2f}'}"
                )
        lines.append(
            "  Advisory only: gains are pixels of immediate signed displacement per "
            "action unit, not joint/pose dynamics constants. Inspect masks and confirm "
            "sign, lag, and magnitude with your explicit parser plus "
            "validate_state_consistency before writing update(a)."
        )
        summary = "\n".join(lines)
        self._log_critic_tool_call(
            tool_name="analyze_action_effects",
            args={
                "trajectory_indices": traj_ids,
                "max_trajectories": int(max_trajectories),
                "min_track_area": int(min_track_area),
            },
            result_summary=summary[:500],
            extra_files={"action_effects.json": json.dumps(payload, indent=2, default=str)},
        )
        return summary

    def view_image(self, trajectory_index: int, image_index: int) -> Image.Image | str:
        """Load one PNG frame from a training trajectory.

        Args:
            trajectory_index: Which trajectory (e.g. 0).
            image_index:      Which frame inside that trajectory (e.g. 0 for start, 10 for end).

        Returns:
            The PIL Image, or an error string if the indices are out of range.
        """
        try:
            t_idx = int(trajectory_index)
            i_idx = int(image_index)
            arr = _dt.view_image(self._dataset_dir, t_idx, i_idx)
        except (FileNotFoundError, ValueError) as exc:
            err = f"[view_image] {exc}"
            self._log_critic_tool_call(
                tool_name="view_image",
                args={
                    "trajectory_index": str(trajectory_index),
                    "image_index": str(image_index),
                },
                result_summary=err,
            )
            return err

        img = Image.fromarray(arr)
        self._log_critic_tool_call(
            tool_name="view_image",
            args={"trajectory_index": t_idx, "image_index": i_idx},
            result_summary=f"Returned trajectory {t_idx} frame {i_idx} ({img.width}x{img.height})",
            extra_files={"frame.png": img},
        )
        return img

    def view_fit_comparison(
        self,
        trajectory_index: int,
        frame_index: int = 0,
        goal_trajectory_index: int | None = None,
        goal_frame_index: int | None = None,
    ) -> Image.Image | str:
        """SEE your own perception: true frame vs render_frame() after fit().

        Runs ``fit(true_frame, goal_frame)`` on real training images, calls
        ``render_frame()``, and returns a labelled side-by-side image:
        [TRUE frame | your render | 50/50 blend]. The middle panel must
        structurally match the left one — same number of objects, same poses,
        same proportions (arm link angles, block shape/orientation, marker
        positions). If it does not, your ``fit()`` parse or your
        ``render_frame()`` geometry is wrong, and NO amount of dynamics
        tuning will fix the simulator — repair perception first.

        Use on 2-3 diverse frames right after implementing ``fit()``, and
        again whenever ``validate_state_consistency`` reports DRIFT (to tell
        a bad parse from bad dynamics).

        Args:
            trajectory_index: Trajectory supplying the TRUE frame (image_A).
            frame_index: Frame index for image_A (default 0).
            goal_trajectory_index: Trajectory supplying image_B (defaults to
                the same trajectory).
            goal_frame_index: Frame index for image_B (defaults to the last
                frame of the goal trajectory).

        Returns:
            A PIL image with the three panels, or an error string.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        try:
            t_idx = int(trajectory_index)
            f_idx = int(frame_index)
            g_t = int(goal_trajectory_index) if goal_trajectory_index is not None else t_idx
            traj = _dt.read_trajectory(self._dataset_dir, t_idx)
            goal_traj = (
                traj if g_t == t_idx else _dt.read_trajectory(self._dataset_dir, g_t)
            )
            g_f = (
                int(goal_frame_index)
                if goal_frame_index is not None
                else len(goal_traj["frames"]) - 1
            )
            if not (0 <= f_idx < len(traj["frames"])):
                raise ValueError(
                    f"frame_index {f_idx} out of range [0, {len(traj['frames'])})"
                )
            if not (0 <= g_f < len(goal_traj["frames"])):
                raise ValueError(
                    f"goal_frame_index {g_f} out of range "
                    f"[0, {len(goal_traj['frames'])})"
                )
        except (FileNotFoundError, ValueError) as exc:
            err = f"[view_fit_comparison] {exc}"
            self._log_critic_tool_call(
                tool_name="view_fit_comparison",
                args={
                    "trajectory_index": str(trajectory_index),
                    "frame_index": str(frame_index),
                    "goal_trajectory_index": str(goal_trajectory_index),
                    "goal_frame_index": str(goal_frame_index),
                },
                result_summary=err,
            )
            return err

        render_npy = os.path.join(self._sandbox_dir, "_fit_comparison_render.npy")
        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "image_A = np.array(Image.open(args['start_path']).convert('RGB'))\n"
            "image_B = np.array(Image.open(args['goal_path']).convert('RGB'))\n"
            "sim.fit(image_A, image_B)\n"
            "frame = np.asarray(sim.render_frame())\n"
            "np.save(args['render_npy'], frame)\n"
            "def _repr(s):\n"
            "    if isinstance(s, dict):\n"
            "        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}\n"
            "    if isinstance(s, np.ndarray):\n"
            "        return s.tolist()\n"
            "    return str(s)\n"
            "print(json.dumps({'state': _repr(sim.state), 'target_state': _repr(sim.target_state)}, default=str))\n"
        )
        with _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name
        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "start_path": str(traj["frames"][f_idx]),
                "goal_path": str(goal_traj["frames"][g_f]),
                "render_npy": render_npy,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=120,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0 or not os.path.exists(render_npy):
            err = (
                "[view_fit_comparison] fit()/render_frame() failed:\n"
                f"{(result.stderr or result.stdout)[:1200]}"
            )
            self._log_critic_tool_call(
                tool_name="view_fit_comparison",
                args={"start": f"({t_idx},{f_idx})", "goal": f"({g_t},{g_f})"},
                result_summary=err[:300],
                extra_files={"stderr.txt": result.stderr or ""},
            )
            return err

        state_line = ""
        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
            state_line = (
                f"state={_json.dumps(payload.get('state'))[:180]} "
                f"target={_json.dumps(payload.get('target_state'))[:120]}"
            )
        except (ValueError, IndexError):
            pass

        true_img = Image.open(traj["frames"][f_idx]).convert("RGB")
        render_arr = np.load(render_npy)
        try:
            os.remove(render_npy)
        except OSError:
            pass
        render_img = Image.fromarray(render_arr.astype(np.uint8)).convert("RGB")
        if render_img.size != true_img.size:
            render_img = render_img.resize(true_img.size, Image.Resampling.NEAREST)
        blend = Image.blend(true_img, render_img, 0.5)

        scale = max(1, int(np.ceil(256 / max(true_img.size))))
        panels = [
            (true_img, f"TRUE frame t{t_idx}/f{f_idx}"),
            (render_img, "your render after fit()"),
            ("blend", "50/50 blend (misalignment shows double)"),
        ]
        panels[2] = (blend, panels[2][1])
        pw = true_img.width * scale
        ph = true_img.height * scale
        pad = 6
        header = 22
        footer = 30
        sheet = Image.new(
            "RGB", (3 * pw + 4 * pad, header + ph + footer), (255, 255, 255)
        )
        draw = ImageDraw.Draw(sheet)
        for i, (im, label) in enumerate(panels):
            x = pad + i * (pw + pad)
            sheet.paste(im.resize((pw, ph), Image.Resampling.NEAREST), (x, header))
            draw.text((x, 4), label, fill=(0, 0, 0))
        if state_line:
            draw.text((pad, header + ph + 4), state_line[:180], fill=(60, 60, 60))

        self._log_critic_tool_call(
            tool_name="view_fit_comparison",
            args={
                "trajectory_index": t_idx,
                "frame_index": f_idx,
                "goal_trajectory_index": g_t,
                "goal_frame_index": g_f,
            },
            result_summary=(
                f"fit comparison on traj{t_idx}/frame{f_idx} vs goal "
                f"traj{g_t}/frame{g_f}. {state_line}"
            ),
            extra_files={"comparison.png": sheet},
        )
        return sheet

    def _run_fit_quality_check(
        self,
        trajectories: list[int],
        *,
        max_rmse: float,
        max_probes: int = 3,
    ) -> dict[str, Any]:
        """Numeric true-frame vs fit()+render_frame() check for final gates."""
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        probe_trajs = self._measurement_trajectories(
            [int(t) for t in trajectories],
            max_trajectories=max_probes,
        )
        probe_candidates: list[dict[str, Any]] = []
        for tid in probe_trajs:
            try:
                traj = _dt.read_trajectory(self._dataset_dir, int(tid))
            except (FileNotFoundError, ValueError) as exc:
                probe_candidates.append({"trajectory": int(tid), "error": str(exc)})
                continue
            frames = list(traj["frames"])
            if len(frames) < 2:
                probe_candidates.append(
                    {"trajectory": int(tid), "error": "trajectory has <2 frames"}
                )
                continue
            sample_idxs = sorted(set([0, len(frames) // 2, len(frames) - 1]))
            for frame_idx in sample_idxs:
                phase = (
                    "start"
                    if frame_idx == 0
                    else "late"
                    if frame_idx == len(frames) - 1
                    else "middle"
                )
                probe_candidates.append(
                    {
                        "trajectory": int(tid),
                        "frame_index": int(frame_idx),
                        "phase": phase,
                        "start_path": str(frames[frame_idx]),
                        "goal_path": str(frames[-1]),
                    }
                )

        valid_candidates = [p for p in probe_candidates if "error" not in p]
        errors = [p for p in probe_candidates if "error" in p]
        probes: list[dict[str, Any]] = []
        if valid_candidates:
            ordered = sorted(
                valid_candidates,
                key=lambda p: (
                    {"start": 0, "middle": 1, "late": 2}.get(str(p["phase"]), 3),
                    int(p["trajectory"]),
                ),
            )
            selected = np.linspace(
                0,
                len(ordered) - 1,
                min(int(max_probes), len(ordered)),
                dtype=int,
            ).tolist()
            probes.extend(ordered[i] for i in dict.fromkeys(selected))
        probes.extend(errors[: max(0, int(max_probes) - len(probes))])

        if not probes:
            return {"passed": False, "error": "no fit-quality probes could be built"}

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            f"{inspect.getsource(_generic_fit_quality_color_masks)}\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "from scipy.ndimage import label\n"
            "def _mask_stats(mask):\n"
            "    mask = np.asarray(mask, dtype=bool)\n"
            "    structure = np.ones((3,3), dtype=np.uint8)\n"
            "    labels, count = label(mask, structure=structure)\n"
            "    min_area = max(3, int(round(mask.size * 0.0005)))\n"
            "    components = []\n"
            "    for idx in range(1, int(count) + 1):\n"
            "        ys, xs = np.where(labels == idx)\n"
            "        if xs.size < min_area:\n"
            "            continue\n"
            "        components.append({'area': int(xs.size), 'centroid': [float(xs.mean()), float(ys.mean())]})\n"
            "    components.sort(key=lambda item: item['area'], reverse=True)\n"
            "    ys, xs = np.where(mask)\n"
            "    return {'area': int(xs.size), 'centroid': ([float(xs.mean()), float(ys.mean())] if xs.size else None), 'n_components': len(components), 'components': components[:8]}\n"
            "def _color_masks(img):\n"
            "    return _generic_fit_quality_color_masks(img)\n"
            "def _mask_overlap(true_mask, render_mask):\n"
            "    inter = int(np.logical_and(true_mask, render_mask).sum())\n"
            "    union = int(np.logical_or(true_mask, render_mask).sum())\n"
            "    return float(inter / union) if union else 1.0\n"
            "def _perception_checks(true, render, max_scene_frac, perception_mode):\n"
            "    true_masks, true_fg = _color_masks(true)\n"
            "    render_masks, render_fg = _color_masks(render)\n"
            "    checks = []\n"
            "    frame_pixels = float(true_fg.size)\n"
            "    min_label_area = max(3, int(round(frame_pixels * 0.0005)))\n"
            "    for name, true_mask in true_masks.items():\n"
            "        true_stats = _mask_stats(true_mask)\n"
            "        if int(true_stats['area']) < min_label_area:\n"
            "            continue\n"
            "        area_frac = float(true_stats['area']) / frame_pixels\n"
            "        render_mask = render_masks[name]\n"
            "        render_stats = _mask_stats(render_mask)\n"
            "        ratio = float(render_stats['area'] / max(1, true_stats['area']))\n"
            "        iou = _mask_overlap(true_mask, render_mask)\n"
            "        if true_stats['centroid'] is not None and render_stats['centroid'] is not None:\n"
            "            centroid_error = float(np.linalg.norm(np.asarray(true_stats['centroid']) - np.asarray(render_stats['centroid'])))\n"
            "        else:\n"
            "            centroid_error = float('inf')\n"
            "        radius = max(5.0, 1.5 * (float(true_stats['area']) / np.pi) ** 0.5)\n"
            "        count_delta = abs(int(true_stats['n_components']) - int(render_stats['n_components']))\n"
            "        geometrically_ok = bool(0.5 <= ratio <= 2.0 and iou >= 0.15 and centroid_error <= radius and count_delta <= 1)\n"
            "        skipped_scene = bool(area_frac > float(max_scene_frac))\n"
            "        checks.append({'label': name, 'true': true_stats, 'render': render_stats, 'area_ratio': ratio, 'area_fraction': area_frac, 'iou': iou, 'centroid_error': centroid_error, 'centroid_tolerance': radius, 'component_count_delta': count_delta, 'skipped_scene_mask': skipped_scene, 'passed': True if skipped_scene else geometrically_ok})\n"
            "    ranked = sorted([c for c in checks if not c.get('skipped_scene_mask')], key=lambda c: int(c['true']['area']), reverse=True)\n"
            "    if perception_mode == 'rmse_plus_foreground':\n"
            "        primary_names = set()\n"
            "    elif perception_mode == 'primary_labels':\n"
            "        primary_names = {c['label'] for c in ranked[:2]}\n"
            "    else:\n"
            "        primary_names = {c['label'] for c in ranked}\n"
            "    for c in checks:\n"
            "        c['primary'] = bool(c['label'] in primary_names)\n"
            "        if c.get('skipped_scene_mask'):\n"
            "            c['hard_fail'] = False\n"
            "        elif perception_mode == 'rmse_plus_foreground':\n"
            "            c['hard_fail'] = False\n"
            "            c['advisory_fail'] = not bool(c['passed'])\n"
            "        else:\n"
            "            c['hard_fail'] = bool(c['primary'] and not c['passed'])\n"
            "            c['advisory_fail'] = bool((not c['primary']) and (not c['passed']))\n"
            "    true_fg_stats = _mask_stats(true_fg)\n"
            "    render_fg_stats = _mask_stats(render_fg)\n"
            "    fg_ratio = float(render_fg_stats['area'] / max(1, true_fg_stats['area']))\n"
            "    fg_iou = _mask_overlap(true_fg, render_fg)\n"
            "    fg_count_delta = abs(int(true_fg_stats['n_components']) - int(render_fg_stats['n_components']))\n"
            "    fg_area_frac = float(true_fg_stats['area']) / frame_pixels\n"
            "    fg_skipped = bool(fg_area_frac > float(max_scene_frac))\n"
            "    fg_geom_ok = bool(true_fg_stats['area'] > 0 and 0.5 <= fg_ratio <= 2.0 and fg_iou >= 0.20 and fg_count_delta <= max(2, int(true_fg_stats['n_components']) // 2))\n"
            "    fg_passed = True if fg_skipped else fg_geom_ok\n"
            "    foreground = {'label': 'foreground', 'true': true_fg_stats, 'render': render_fg_stats, 'area_ratio': fg_ratio, 'area_fraction': fg_area_frac, 'iou': fg_iou, 'component_count_delta': fg_count_delta, 'skipped_scene_mask': fg_skipped, 'passed': fg_passed, 'hard_fail': bool((not fg_skipped) and (not fg_geom_ok))}\n"
            "    union = np.logical_or(true_fg, render_fg)\n"
            "    foreground_rmse = float(np.sqrt(np.mean((render[union] - true[union]) ** 2)) / 255.0) if np.any(union) else 0.0\n"
            "    return checks, foreground, foreground_rmse\n"
            "out = []\n"
            "for p in args['probes']:\n"
            "    rec = {k: v for k, v in p.items() if k not in ('start_path', 'goal_path')}\n"
            "    if 'error' in p:\n"
            "        rec['error'] = p['error']; out.append(rec); continue\n"
            "    try:\n"
            "        api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "        sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "        image_A = np.array(Image.open(p['start_path']).convert('RGB'))\n"
            "        image_B = np.array(Image.open(p['goal_path']).convert('RGB'))\n"
            "        sim.fit(image_A, image_B)\n"
            "        render = np.asarray(sim.render_frame()).astype(np.float32)\n"
            "        true = image_A.astype(np.float32)\n"
            "        if render.ndim == 2:\n"
            "            render = np.repeat(render[:, :, None], 3, axis=2)\n"
            "        if render.shape[:2] != true.shape[:2]:\n"
            "            render = np.asarray(Image.fromarray(np.clip(render, 0, 255).astype(np.uint8)).resize((true.shape[1], true.shape[0]), Image.Resampling.NEAREST), dtype=np.float32)\n"
            "        render = render[:, :, :3]\n"
            "        denom = max(1.0, (true.size ** 0.5) * 255.0)\n"
            "        rmse = float(np.linalg.norm(render - true) / denom)\n"
            "        component_checks, foreground_check, foreground_rmse = _perception_checks(true, render, args.get('max_scene_frac', 0.35), args.get('perception_mode', 'rmse_plus_foreground'))\n"
            "        failures = [v for v in component_checks if v.get('hard_fail')]\n"
            "        if foreground_check.get('hard_fail'):\n"
            "            failures.append(foreground_check)\n"
            "        advisory = [v for v in component_checks if v.get('advisory_fail')]\n"
            "        rec.update({'normalized_rmse': rmse, 'foreground_rmse': foreground_rmse, 'render_shape': list(render.shape), 'component_checks': component_checks, 'foreground_check': foreground_check, 'n_perception_failures': len(failures), 'n_advisory_perception_failures': len(advisory), 'perception_mode': args.get('perception_mode', 'rmse_plus_foreground')})\n"
            "    except Exception as exc:\n"
            "        rec['error'] = f'{type(exc).__name__}: {exc}'\n"
            "    out.append(rec)\n"
            "print(json.dumps(out))\n"
        )
        with _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name
        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "probes": probes,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
                "max_scene_frac": float(self._gate_fit_quality_max_scene_frac),
                "perception_mode": str(self._gate_fit_quality_mode),
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=180,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return {
                "passed": False,
                "error": f"runner exited with code {result.returncode}: {result.stderr[-800:]}",
            }
        try:
            results = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {
                "passed": False,
                "error": f"could not parse fit-quality output: {exc}",
                "stdout": result.stdout[-800:],
            }
        rmses = [
            float(r["normalized_rmse"])
            for r in results
            if r.get("normalized_rmse") is not None and np.isfinite(float(r["normalized_rmse"]))
        ]
        failed = [r for r in results if r.get("error")]
        perception_failures = [
            r
            for r in results
            if int(r.get("n_perception_failures") or 0) > 0
        ]
        worst = max(rmses) if rmses else None
        mean = float(np.mean(rmses)) if rmses else None
        passed = bool(
            not failed
            and not perception_failures
            and worst is not None
            and float(worst) <= float(max_rmse)
        )
        payload = {
            "passed": passed,
            "max_rmse": float(max_rmse),
            "max_scene_frac": float(self._gate_fit_quality_max_scene_frac),
            "perception_mode": str(self._gate_fit_quality_mode),
            "worst_normalized_rmse": worst,
            "mean_normalized_rmse": mean,
            "n_probes": len(results),
            "n_failed": len(failed),
            "n_visibility_failures": len(perception_failures),
            "n_perception_failures": len(perception_failures),
            "probes": results,
        }
        if self._tool_calls_log_dir:
            try:
                Path(self._tool_calls_log_dir, "fit_quality_gate.json").write_text(
                    json.dumps(payload, indent=2, default=str),
                    encoding="utf-8",
                )
            except OSError:
                pass
        return payload

    def validate_visual_reconstruction(
        self,
        trajectory_indices: list[int] | None = None,
        max_probes: int = 6,
    ) -> str:
        """Check fit()+render geometry over diverse real training frames.

        This is advisory codegen feedback, not a hard gate. It reuses the
        fit-quality probe on several start/middle/late frames and reports
        foreground/component overlap, centroids, component counts, and pixel
        RMSE. Use it before dynamics identification so good regression cannot
        hide a distorted visual coordinate system.

        Args:
            trajectory_indices: Optional training trajectories to inspect.
            max_probes: Number of diverse frame probes (3..9).
        """
        available = set(int(i) for i in self._fidelity_train_trajectories())
        requested = (
            sorted(set(int(i) for i in trajectory_indices))
            if trajectory_indices is not None
            else sorted(available)
        )
        selected = [idx for idx in requested if idx in available]
        args = {
            "trajectory_indices": requested,
            "max_probes": int(max_probes),
        }
        if not selected:
            result = (
                "[validate_visual_reconstruction] no requested training "
                f"trajectories are available; valid={sorted(available)}"
            )
            self._log_critic_tool_call(
                tool_name="validate_visual_reconstruction",
                args=args,
                result_summary=result,
            )
            return result

        probe_count = max(3, min(9, int(max_probes)))
        payload = self._run_fit_quality_check(
            selected,
            max_rmse=self._gate_fit_quality_max_rmse,
            max_probes=probe_count,
        )
        summary = (
            "[validate_visual_reconstruction] advisory multi-frame "
            "fit()+render check\n"
            + json.dumps(payload, indent=2, default=str)
        )
        self._log_critic_tool_call(
            tool_name="validate_visual_reconstruction",
            args=args,
            result_summary=summary[:600],
            extra_files={
                "visual_reconstruction.json": json.dumps(
                    payload,
                    indent=2,
                    default=str,
                )
            },
        )
        return summary

    def calibrate_articulated_geometry(
        self,
        rgb_min: list[float],
        rgb_max: list[float],
        base_xy: list[float],
        link_lengths: list[float],
        samples: list[list[int]] | None = None,
        max_samples: int = 9,
    ) -> str:
        """Calibrate fixed serial-chain geometry across diverse training frames.

        Supply inclusive RGB bounds for the visible links after inspecting
        training images. The tool builds deterministic local masks, samples
        start/middle/end frames across trajectories by default, and jointly
        estimates one fixed base and set of link lengths. No Gemini/API call is
        used. Bake the returned constants into ``fit()``; do not recalibrate
        geometry independently on each P2 observation.
        """
        from vdaworld.core.local_2d import (
            calibrate_articulated_chain_geometry,
            clean_mask,
        )

        lower = np.asarray(rgb_min, dtype=np.float64).reshape(-1)
        upper = np.asarray(rgb_max, dtype=np.float64).reshape(-1)
        base = np.asarray(base_xy, dtype=np.float64).reshape(-1)
        lengths = np.asarray(link_lengths, dtype=np.float64).reshape(-1)
        args = {
            "rgb_min": lower.tolist(),
            "rgb_max": upper.tolist(),
            "base_xy": base.tolist(),
            "link_lengths": lengths.tolist(),
            "samples": samples,
            "max_samples": int(max_samples),
        }
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or np.any(lower > upper)
            or base.shape != (2,)
            or lengths.size == 0
            or np.any(lengths <= 0)
        ):
            message = (
                "[calibrate_articulated_geometry] invalid RGB bounds, base, "
                "or link-length hints"
            )
            self._log_critic_tool_call(
                tool_name="calibrate_articulated_geometry",
                args=args,
                result_summary=message,
            )
            return message

        selected: list[tuple[int, int]] = []
        if samples is not None:
            for item in samples:
                if len(item) != 2:
                    continue
                selected.append((int(item[0]), int(item[1])))
        else:
            trajectories = self._measurement_trajectories(
                self._fidelity_train_trajectories(),
                max_trajectories=3,
            )
            for trajectory_index in trajectories:
                try:
                    trajectory = _dt.read_trajectory(
                        self._dataset_dir,
                        int(trajectory_index),
                    )
                except (FileNotFoundError, ValueError):
                    continue
                n_frames = len(trajectory["frames"])
                for frame_index in sorted(
                    set([0, n_frames // 2, n_frames - 1])
                ):
                    selected.append((int(trajectory_index), int(frame_index)))
        selected = list(dict.fromkeys(selected))[: max(2, min(12, int(max_samples)))]

        masks: list[np.ndarray] = []
        used_samples: list[list[int]] = []
        rejected: list[dict[str, Any]] = []
        for trajectory_index, frame_index in selected:
            try:
                image = np.asarray(
                    _dt.view_image(
                        self._dataset_dir,
                        trajectory_index,
                        frame_index,
                    )
                )[..., :3]
                mask = np.all(
                    (image.astype(np.float64) >= lower.reshape(1, 1, 3))
                    & (image.astype(np.float64) <= upper.reshape(1, 1, 3)),
                    axis=2,
                )
                mask = clean_mask(
                    mask,
                    # Articulated links can be anti-aliased into several thin
                    # colour fragments; geometry calibration and skeleton
                    # fitting need all of them rather than only a large blob.
                    min_area=1,
                    close_radius=1,
                    largest=False,
                )
                if int(mask.sum()) < 3:
                    raise ValueError("RGB bounds produced fewer than 3 pixels")
                masks.append(mask)
                used_samples.append([trajectory_index, frame_index])
            except (FileNotFoundError, ValueError) as exc:
                rejected.append(
                    {
                        "sample": [trajectory_index, frame_index],
                        "error": str(exc),
                    }
                )

        if len(masks) < 2:
            payload = {
                "passed": False,
                "error": "fewer than two usable masks",
                "samples": used_samples,
                "rejected": rejected,
            }
        else:
            try:
                calibration = calibrate_articulated_chain_geometry(
                    masks,
                    base,
                    lengths,
                )
                frame_records = []
                for sample, record in zip(used_samples, calibration["frames"]):
                    frame_records.append(
                        {
                            "sample": sample,
                            "angles": record["angles"],
                            "joints": record["joints"],
                            "end_effector": record["end_effector"],
                            "chamfer_rmse": record["chamfer_rmse"],
                            "ambiguous": record["ambiguous"],
                            "ambiguity_margin": record["ambiguity_margin"],
                        }
                    )
                payload = {
                    "passed": True,
                    "base_xy": calibration["base_xy"],
                    "link_lengths": calibration["link_lengths"],
                    "geometry_delta": calibration["geometry_delta"],
                    "mean_chamfer_rmse": calibration["mean_chamfer_rmse"],
                    "worst_chamfer_rmse": calibration["worst_chamfer_rmse"],
                    "samples": frame_records,
                    "rejected": rejected,
                    "ready_to_embed": {
                        "base_xy": calibration["base_xy"],
                        "link_lengths": calibration["link_lengths"],
                    },
                    "warning": (
                        "Inspect ambiguous frame records and validate projected "
                        "state pixels before trusting these constants."
                    ),
                }
            except (RuntimeError, ValueError) as exc:
                payload = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "samples": used_samples,
                    "rejected": rejected,
                }

        serialized = json.dumps(payload, indent=2, default=lambda value: np.asarray(value).tolist())
        message = "[calibrate_articulated_geometry]\n" + serialized
        self._log_critic_tool_call(
            tool_name="calibrate_articulated_geometry",
            args=args,
            result_summary=message[:800],
            extra_files={"chain_geometry_calibration.json": serialized},
        )
        return message

    def calibrate_rigid_geometry(
        self,
        rgb_min: list[float],
        rgb_max: list[float],
        samples: list[list[int]] | None = None,
        max_samples: int = 9,
        min_area: int = 3,
        simplify_px: float = 1.0,
    ) -> str:
        """Measure one rigid/local component over diverse training frames.

        Supply inclusive RGB bounds after inspecting representative pixels. The
        tool uses only deterministic local masks and reports robust area,
        centroid/bounds, area-equivalent radius, and one centroid-centred
        canonical template. Circular components intentionally report unstable
        orientation; use their median radius rather than a pose angle.
        """
        from vdaworld.core.local_2d import (
            calibrate_component_geometry,
            clean_mask,
        )

        lower = np.asarray(rgb_min, dtype=np.float64).reshape(-1)
        upper = np.asarray(rgb_max, dtype=np.float64).reshape(-1)
        args = {
            "rgb_min": lower.tolist(),
            "rgb_max": upper.tolist(),
            "samples": samples,
            "max_samples": int(max_samples),
            "min_area": int(min_area),
            "simplify_px": float(simplify_px),
        }
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or np.any(lower > upper)
            or int(min_area) < 1
            or not np.isfinite(float(simplify_px))
            or float(simplify_px) < 0
        ):
            message = (
                "[calibrate_rigid_geometry] invalid RGB bounds, min_area, "
                "or simplify_px"
            )
            self._log_critic_tool_call(
                tool_name="calibrate_rigid_geometry",
                args=args,
                result_summary=message,
            )
            return message

        selected: list[tuple[int, int]] = []
        if samples is not None:
            for item in samples:
                if len(item) == 2:
                    selected.append((int(item[0]), int(item[1])))
        else:
            trajectories = self._measurement_trajectories(
                self._fidelity_train_trajectories(),
                max_trajectories=3,
            )
            for trajectory_index in trajectories:
                try:
                    trajectory = _dt.read_trajectory(
                        self._dataset_dir,
                        int(trajectory_index),
                    )
                except (FileNotFoundError, ValueError):
                    continue
                n_frames = len(trajectory["frames"])
                for frame_index in sorted(set([0, n_frames // 2, n_frames - 1])):
                    selected.append((int(trajectory_index), int(frame_index)))
        selected = list(dict.fromkeys(selected))[: max(2, min(12, int(max_samples)))]

        masks: list[np.ndarray] = []
        used_samples: list[list[int]] = []
        rejected: list[dict[str, Any]] = []
        for trajectory_index, frame_index in selected:
            try:
                image = np.asarray(
                    _dt.view_image(
                        self._dataset_dir,
                        trajectory_index,
                        frame_index,
                    )
                )[..., :3]
                mask = np.all(
                    (image.astype(np.float64) >= lower.reshape(1, 1, 3))
                    & (image.astype(np.float64) <= upper.reshape(1, 1, 3)),
                    axis=2,
                )
                mask = clean_mask(
                    mask,
                    min_area=int(min_area),
                    close_radius=1,
                    largest=True,
                )
                if int(mask.sum()) < int(min_area):
                    raise ValueError("RGB bounds produced too few component pixels")
                masks.append(mask)
                used_samples.append([trajectory_index, frame_index])
            except (FileNotFoundError, ValueError) as exc:
                rejected.append(
                    {
                        "sample": [trajectory_index, frame_index],
                        "error": str(exc),
                    }
                )

        if len(masks) < 2:
            payload: dict[str, Any] = {
                "passed": False,
                "error": "fewer than two usable masks",
                "samples": used_samples,
                "rejected": rejected,
            }
        else:
            try:
                calibration = calibrate_component_geometry(
                    masks,
                    min_area=int(min_area),
                    close_radius=0,
                    largest=True,
                    simplify_px=float(simplify_px),
                )
                for sample, record in zip(used_samples, calibration["frames"]):
                    record["sample"] = sample
                payload = {
                    "passed": True,
                    **calibration,
                    "rejected": rejected,
                    "ready_to_embed": {
                        "area_equivalent_radius": calibration[
                            "median_area_equivalent_radius"
                        ],
                        "canonical_template": calibration["canonical_template"],
                    },
                    "warning": (
                        "Validate the emitted mask/template on held-out poses. "
                        "Use radius for near-circular components and convex-"
                        "decompose concave collision templates."
                    ),
                }
            except ValueError as exc:
                payload = {
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "samples": used_samples,
                    "rejected": rejected,
                }

        serialized = json.dumps(
            payload,
            indent=2,
            default=lambda value: np.asarray(value).tolist(),
        )
        message = "[calibrate_rigid_geometry]\n" + serialized
        self._log_critic_tool_call(
            tool_name="calibrate_rigid_geometry",
            args=args,
            result_summary=message[:800],
            extra_files={"rigid_geometry_calibration.json": serialized},
        )
        return message

    @staticmethod
    def _extract_json_list(text: str) -> list[dict[str, Any]]:
        cleaned = str(text or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            lo = cleaned.find("[")
            hi = cleaned.rfind("]")
            if lo < 0 or hi <= lo:
                raise
            payload = json.loads(cleaned[lo : hi + 1])
        if isinstance(payload, dict):
            payload = payload.get("masks") or payload.get("segments") or payload.get("objects") or [payload]
        if not isinstance(payload, list):
            raise ValueError(f"Gemini segmentation JSON must be a list, got {type(payload).__name__}")
        return [item for item in payload if isinstance(item, dict)]

    @staticmethod
    def _bbox_to_mask(box: Any, shape: tuple[int, int]) -> np.ndarray:
        h, w = shape
        arr = np.asarray(box, dtype=np.float64).reshape(-1)
        mask = np.zeros((h, w), dtype=bool)
        if arr.size != 4 or not np.all(np.isfinite(arr)):
            return mask
        y0, x0, y1, x1 = arr.tolist()
        # Gemini spatial outputs are commonly 0..1000; accept pixel boxes too.
        if max(abs(v) for v in (x0, y0, x1, y1)) > max(h, w) + 5:
            x0, x1 = x0 * w / 1000.0, x1 * w / 1000.0
            y0, y1 = y0 * h / 1000.0, y1 * h / 1000.0
        xa, xb = sorted((int(round(x0)), int(round(x1))))
        ya, yb = sorted((int(round(y0)), int(round(y1))))
        xa, xb = max(0, xa), min(w, xb)
        ya, yb = max(0, ya), min(h, yb)
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = True
        return mask

    @staticmethod
    def _decode_gemini_mask(mask_payload: Any, shape: tuple[int, int]) -> np.ndarray | None:
        if not isinstance(mask_payload, str) or not mask_payload.strip():
            return None
        raw = mask_payload.split(",", 1)[-1].strip()
        try:
            data = base64.b64decode(raw, validate=False)
            img = Image.open(io.BytesIO(data)).convert("L")
        except (binascii.Error, OSError, ValueError):
            return None
        h, w = shape
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.NEAREST)
        return np.asarray(img, dtype=np.uint8) > 0

    def _call_gemini_segmentation(
        self,
        image: np.ndarray,
        query: str,
        *,
        timeout_s: float = 90.0,
    ) -> list[dict[str, Any]]:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; cannot call Gemini segmentation.")
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
        from google import genai
        from google.genai import types

        prompt = (
            "Give the segmentation masks for the objects matching this query: "
            f"{query!r}.\n"
            "Output a JSON list of segmentation masks where each entry contains "
            'the 2D bounding box in the key "box_2d", the segmentation mask in '
            'key "mask", and the text label in the key "label". Use descriptive labels.'
        )

        def _generate() -> str:
            client = genai.Client(api_key=api_key, vertexai=False)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, Image.fromarray(image)],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            return getattr(response, "text", "") or ""

        # Do not use the executor as a context manager: on timeout it would still
        # block in shutdown(wait=True) while the hung HTTP call finishes.
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            future = pool.submit(_generate)
            try:
                text = future.result(timeout=float(timeout_s))
            except FuturesTimeoutError as exc:
                future.cancel()
                raise TimeoutError(
                    f"Gemini segmentation timed out after {timeout_s:.0f}s"
                ) from exc
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        return self._extract_json_list(text)

    def segment_image_with_gemini(
        self,
        trajectory_index: int,
        image_index: int,
        query: str,
    ) -> str:
        """Use Gemini 2.5 conversational segmentation on one training frame.

        Args:
            trajectory_index: Zero-based trajectory index under the training dataset.
            image_index: Zero-based frame index within that trajectory.
            query: Natural-language referring expression, e.g. "the red cube" or "the robot gripper".
        """
        try:
            t_idx = int(trajectory_index)
            i_idx = int(image_index)
            arr = _dt.view_image(self._dataset_dir, t_idx, i_idx)
            raw_items = self._call_gemini_segmentation(arr, str(query))
        except Exception as exc:
            err = f"[segment_image_with_gemini] {type(exc).__name__}: {exc}"
            self._log_critic_tool_call(
                tool_name="segment_image_with_gemini",
                args={"trajectory_index": trajectory_index, "image_index": image_index, "query": query},
                result_summary=err,
            )
            return err

        masks: list[np.ndarray] = []
        rows: list[dict[str, Any]] = []
        for idx, item in enumerate(raw_items):
            mask = self._decode_gemini_mask(item.get("mask"), arr.shape[:2])
            if mask is None:
                mask = self._bbox_to_mask(item.get("box_2d"), arr.shape[:2])
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue
            masks.append(mask)
            rows.append(
                {
                    "index": idx,
                    "label": str(item.get("label", f"segment_{idx}")),
                    "area": int(mask.sum()),
                    "centroid_xy": [float(xs.mean()), float(ys.mean())],
                    "box_2d": item.get("box_2d"),
                }
            )

        if masks:
            mask_stack = np.asarray(masks, dtype=bool)
            overlay = Image.fromarray(_draw_segment_overlay(arr, mask_stack))
        else:
            overlay = Image.fromarray(arr)

        summary = (
            f"Gemini segmentation for trajectory {t_idx} frame {i_idx}, query={query!r}: "
            f"{len(rows)} segment(s).\n"
            + json.dumps(rows, indent=2)
        )
        self._log_critic_tool_call(
            tool_name="segment_image_with_gemini",
            args={"trajectory_index": t_idx, "image_index": i_idx, "query": query},
            result_summary=summary[:500],
            extra_files={
                "overlay.png": overlay,
                "segments.json": json.dumps(rows, indent=2),
            },
        )
        return summary

    def derive_visual_model(
        self,
        trajectory_index: int,
        image_index: int,
        query: str,
        component_index: int = 0,
        simplify_px: float = 1.0,
    ) -> str:
        """Turn a training-frame Gemini mask into deploy-time local CV constants.

        This bridges code-generation perception and P2: Gemini identifies an
        object on a TRAINING frame once, then this tool derives a deterministic
        Lab appearance model and centroid-centred polygon template. Generated
        code embeds those JSON/Python literals and calls the always-available
        ``self.mask_from_appearance`` / ``self.fit_pose_to_mask`` helpers. No
        Gemini, network, dataset access, or WorldAPI is needed at P2.

        Args:
            trajectory_index: Training trajectory containing the reference frame.
            image_index: Frame within the trajectory.
            query: Natural-language referring expression for the object.
            component_index: Area-ranked matching segment to use (0 = largest).
            simplify_px: OpenCV contour simplification tolerance in pixels.
        """
        import cv2

        from vdaworld.core.local_2d import (
            clean_mask,
            mask_from_appearance,
            template_from_mask,
        )

        args = {
            "trajectory_index": trajectory_index,
            "image_index": image_index,
            "query": query,
            "component_index": component_index,
            "simplify_px": simplify_px,
        }
        try:
            t_idx = int(trajectory_index)
            i_idx = int(image_index)
            selected_index = int(component_index)
            if selected_index < 0:
                raise ValueError("component_index must be non-negative")
            arr = _dt.view_image(self._dataset_dir, t_idx, i_idx)
            raw_items = self._call_gemini_segmentation(arr, str(query))

            decoded: list[tuple[int, str, np.ndarray]] = []
            for raw_index, item in enumerate(raw_items):
                mask = self._decode_gemini_mask(item.get("mask"), arr.shape[:2])
                if mask is None:
                    # A box is useful for inspection but cannot support a
                    # faithful appearance distribution or rigid template.
                    # Fail explicitly instead of emitting a confident-looking
                    # rectangular model from missing segmentation data.
                    continue
                mask = clean_mask(
                    mask,
                    min_area=max(3, int(round(mask.size * 0.0003))),
                    close_radius=1,
                    largest=True,
                )
                if np.any(mask):
                    decoded.append(
                        (
                            raw_index,
                            str(item.get("label", f"segment_{raw_index}")),
                            mask,
                        )
                    )
            decoded.sort(key=lambda row: int(row[2].sum()), reverse=True)
            if selected_index >= len(decoded):
                raise ValueError(
                    f"component_index={selected_index} but Gemini returned "
                    f"{len(decoded)} usable pixel mask(s); bounding boxes are "
                    "not accepted for visual-model derivation"
                )
            raw_index, label, reference_mask = decoded[selected_index]

            rgb = np.asarray(arr[..., :3], dtype=np.uint8)
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
            pixels = lab[reference_mask]
            center = np.median(pixels, axis=0)
            mad = np.median(np.abs(pixels - center), axis=0)
            # Lab quantization and flat-colour regions otherwise produce zero
            # scales. A small floor preserves anti-aliased boundary colours.
            scale = np.maximum(1.4826 * mad, np.array([3.0, 3.0, 3.0]))
            distances = np.linalg.norm((pixels - center) / scale, axis=1)

            min_area = max(3, int(round(float(reference_mask.sum()) * 0.20)))
            candidate_quantiles = (0.90, 0.95, 0.975, 0.99, 0.995, 1.0)
            best: tuple[float, float, np.ndarray] | None = None
            for quantile in candidate_quantiles:
                threshold = max(1.0, float(np.quantile(distances, quantile)) + 0.15)
                candidate_model = {
                    "center_lab": center.tolist(),
                    "scale_lab": scale.tolist(),
                    "max_distance": threshold,
                }
                predicted = mask_from_appearance(
                    rgb,
                    candidate_model,
                    min_area=min_area,
                    close_radius=1,
                    largest=True,
                )
                intersection = int(np.logical_and(reference_mask, predicted).sum())
                union = int(np.logical_or(reference_mask, predicted).sum())
                iou = float(intersection / union) if union else 0.0
                # Prefer a smaller threshold when overlap ties; this limits
                # same-colour background leakage on unseen frames.
                score = iou - 1e-6 * threshold
                if best is None or score > best[0]:
                    best = (score, threshold, predicted)
            assert best is not None
            _, threshold, predicted_mask = best

            appearance_model = {
                "center_lab": [round(float(value), 4) for value in center],
                "scale_lab": [round(float(value), 4) for value in scale],
                "max_distance": round(float(threshold), 4),
            }
            template_arrays = template_from_mask(
                reference_mask,
                simplify_px=max(0.0, float(simplify_px)),
                max_vertices=64,
            )
            template = [
                [[round(float(x), 3), round(float(y), 3)] for x, y in polygon]
                for polygon in template_arrays
            ]
            ys, xs = np.where(reference_mask)
            intersection = int(np.logical_and(reference_mask, predicted_mask).sum())
            union = int(np.logical_or(reference_mask, predicted_mask).sum())
            iou = float(intersection / union) if union else 0.0
            dice = float(
                2.0 * intersection
                / max(1, int(reference_mask.sum()) + int(predicted_mask.sum()))
            )
            result_payload = {
                "trajectory": t_idx,
                "frame": i_idx,
                "query": str(query),
                "gemini_label": label,
                "gemini_raw_index": raw_index,
                "area": int(reference_mask.sum()),
                "centroid_xy": [
                    round(float(xs.mean()), 4),
                    round(float(ys.mean()), 4),
                ],
                "appearance_model": appearance_model,
                "recommended_min_area": int(min_area),
                "local_reproduction_iou": round(iou, 4),
                "local_reproduction_dice": round(dice, 4),
                "template_polygons": template,
                "template_note": (
                    "Use for a rigid asymmetric component. For articulated or "
                    "deformable components use the appearance mask plus "
                    "component/keypoint geometry instead of one rigid pose."
                ),
            }

            overlay = rgb.copy()
            overlay[reference_mask] = (
                0.45 * overlay[reference_mask]
                + 0.55 * np.array([255, 70, 50], dtype=np.float64)
            ).astype(np.uint8)
            predicted_edge = cv2.morphologyEx(
                predicted_mask.astype(np.uint8),
                cv2.MORPH_GRADIENT,
                np.ones((3, 3), dtype=np.uint8),
            ).astype(bool)
            overlay[predicted_edge] = np.array([0, 255, 255], dtype=np.uint8)

            snippet = (
                "Embed these literals in your simulator, then in fit():\n"
                f"appearance_model = {appearance_model!r}\n"
                f"template = [np.array(poly, dtype=float) for poly in {template!r}]\n"
                "mask = self.mask_from_appearance(image, appearance_model, "
                f"min_area={min_area}, close_radius=1, largest=True)\n"
                "pose = self.fit_pose_to_mask(mask, template)  # rigid objects only"
            )
            summary = (
                "[derive_visual_model] TRAINING-mask -> deterministic P2 model\n"
                + json.dumps(result_payload, indent=2)
                + "\n"
                + snippet
                + "\nValidate this model on other frames with "
                "view_fit_comparison; one reference mask is not proof of "
                "cross-frame generalisation."
            )
            self._log_critic_tool_call(
                tool_name="derive_visual_model",
                args=args,
                result_summary=summary[:500],
                extra_files={
                    "appearance_model.json": json.dumps(
                        result_payload,
                        indent=2,
                    ),
                    "model_overlay.png": Image.fromarray(overlay),
                },
            )
            return summary
        except Exception as exc:
            error = f"[derive_visual_model] {type(exc).__name__}: {exc}"
            self._log_critic_tool_call(
                tool_name="derive_visual_model",
                args=args,
                result_summary=error,
            )
            return error

    def get_runtime_toolbox_documentation(
        self,
        category: str = "all",
    ) -> str:
        """Return concise P1/P2-safe helper and library documentation."""
        requested = str(category or "all").strip().lower()
        if requested not in {"all", "perception", "dynamics", "libraries"}:
            message = (
                "[get_runtime_toolbox_documentation] category must be one of "
                "all, perception, dynamics, libraries"
            )
        else:
            sections: list[str] = []
            if requested in {"all", "perception"}:
                sections.append(
                    "PERCEPTION (available as inherited methods with api=None):\n"
                    "- self.mask_from_appearance(image, model, min_area=1, "
                    "open_radius=0, close_radius=1, largest=True) -> bool mask. "
                    "Use the model emitted by derive_visual_model.\n"
                    "- self.foreground_from_border(image, lab_distance=12, "
                    "border_width=3, min_area=1, ..., largest=False) -> bool mask.\n"
                    "- self.clean_mask(mask, min_area=1, open_radius=0, "
                    "close_radius=1, largest=True) -> bool mask.\n"
                    "- self.select_connected_component(mask, min_area=1, "
                    "reference_centroid=None, expected_area=None, connectivity=8) "
                    "-> mask plus valid/confidence/area/centroid/component-count "
                    "metadata. Never average disconnected components; use prior "
                    "observations and calibrated area when available.\n"
                    "- self.component_geometry(mask, simplify_px=1, "
                    "max_vertices=64) -> area/centroid/bbox/PCA angle/contour.\n"
                    "- self.calibrate_component_geometry(masks, ...) -> robust "
                    "multi-frame dimensions, area-equivalent radius, and one "
                    "canonical template. During codegen, prefer dataset tool "
                    "calibrate_rigid_geometry(...) for sampled training frames.\n"
                    "- self.template_from_mask(mask, ...) -> centroid-centred "
                    "polygons. Keep one cross-frame-validated canonical template; "
                    "never derive a new template inside each fit().\n"
                    "- self.fit_pose_to_mask(mask, template, n_angles=180) -> "
                    "cx/cy/theta/IoU/Dice plus valid/confidence. Check validity "
                    "before using the legacy centre fallback. It fits pose; it "
                    "cannot repair a bad mask or bad template.\n"
                    "- self.fit_articulated_chain(mask, base_xy, link_lengths, "
                    "initial_angles=None, relative_angles=True, n_starts=128, "
                    "prior_weight=.02) -> angles/joints/end_effector/Chamfer "
                    "error plus alternate candidates and ambiguity diagnostics. "
                    "Use for fixed-length linked objects; pass previous angles "
                    "to preserve temporal continuity.\n"
                    "- self.calibrate_articulated_chain_geometry(masks, base_xy, "
                    "link_lengths, ...) estimates one fixed base/link geometry "
                    "from several masks. During codegen, prefer the dataset tool "
                    "calibrate_articulated_geometry(...) so constants are baked "
                    "into deployed code."
                )
            if requested in {"all", "dynamics"}:
                sections.append(
                    "DYNAMICS (local/deterministic; safe in update/CEM):\n"
                    "- self.clamp_step(previous, target, max_step)\n"
                    "- self.estimate_observed_velocity(current_positions, "
                    "previous_observed_positions, elapsed_seconds, periodic) "
                    "uses wrapped deltas only on declared periodic dimensions. "
                    "Compare observed poses to earlier observed poses over the "
                    "real elapsed time, not to propagated latent state.\n"
                    "- self.linear_dynamics_step(state, action, gain, bias=None, "
                    "max_delta=None): identified first-order model.\n"
                    "- self.damped_dynamics_step(position, velocity, action, "
                    "gain, damping=..., dt=..., max_velocity=...): second-order.\n"
                    "  Optional position_low/position_high and periodic control "
                    "bounded versus wrapped dimensions. Bounded dimensions clamp "
                    "and cancel outward velocity; do not mark a bounded joint periodic.\n"
                    "- self.planar_push_step(body_polygons=..., body_position=..., "
                    "body_angle=..., pusher_position=..., pusher_target=..., "
                    "pusher_radius=..., body_velocity=..., "
                    "body_angular_velocity=..., pusher_velocity=..., kp=100, "
                    "kv=20, dt=.01, substeps=10, friction=0, damping=1, "
                    "bounds=None) -> dict with body_position/body_angle/"
                    "body_velocity/body_angular_velocity/pusher_position/"
                    "pusher_velocity. Concave outlines are decomposed into "
                    "convex pieces. Inputs and outputs remain in the caller's "
                    "coordinate system; never flip positions without also "
                    "transforming local polygons. Infer parameters from demos."
                )
            if requested in {"all", "libraries"}:
                sections.append(
                    "INSTALLED IN BOTH P1/P2: numpy, scipy, opencv-python (cv2), "
                    "scikit-image, Pillow, shapely, pymunk, pygame, mujoco. "
                    "Prefer established local libraries over handwritten "
                    "segmentation, collision, or rigid-body solvers. Generated "
                    "code must remain deterministic/network-free and must not "
                    "load datasets or real environments."
                )
            message = "[get_runtime_toolbox_documentation]\n" + "\n\n".join(sections)
        self._log_critic_tool_call(
            tool_name="get_runtime_toolbox_documentation",
            args={"category": category},
            result_summary=message[:500],
        )
        return message

    def segment_trajectory_with_gemini(
        self,
        trajectory_index: int,
        query: str,
        max_frames: int = 6,
    ) -> str:
        """Segment sampled frames from one trajectory and summarize object centroids.

        Args:
            trajectory_index: Zero-based trajectory index under the training dataset.
            query: Natural-language referring expression to segment in each sampled frame.
            max_frames: Number of evenly spaced frames to sample from the trajectory.
        """
        try:
            t_idx = int(trajectory_index)
            traj = _dt.read_trajectory(self._dataset_dir, t_idx)
            n_frames = len(traj["frames"])
            if n_frames <= 0:
                raise ValueError(f"trajectory {t_idx} has no frames")
            indices = np.linspace(0, n_frames - 1, max(1, min(int(max_frames), n_frames)), dtype=int).tolist()
            indices = list(dict.fromkeys(indices))
        except Exception as exc:
            err = f"[segment_trajectory_with_gemini] {type(exc).__name__}: {exc}"
            self._log_critic_tool_call(
                tool_name="segment_trajectory_with_gemini",
                args={"trajectory_index": trajectory_index, "query": query, "max_frames": max_frames},
                result_summary=err,
            )
            return err

        frame_summaries = []
        for i_idx in indices:
            result = self.segment_image_with_gemini(t_idx, int(i_idx), query)
            try:
                json_part = result[result.index("[") :]
                segments = json.loads(json_part)
            except Exception:
                segments = []
            frame_summaries.append({"frame": int(i_idx), "segments": segments})

        summary = (
            f"Trajectory {t_idx} Gemini segmentation track for query={query!r} "
            f"over frames {indices}:\n"
            + json.dumps(frame_summaries, indent=2)
        )
        self._log_critic_tool_call(
            tool_name="segment_trajectory_with_gemini",
            args={"trajectory_index": t_idx, "query": query, "max_frames": max_frames},
            result_summary=summary[:500],
            extra_files={"trajectory_segments.json": json.dumps(frame_summaries, indent=2)},
        )
        return summary

    def validate_against_training(
        self,
        trajectory_index: int,
        max_steps: int | None = None,
    ) -> str:
        """Replay a training trajectory through your simulator and report fidelity.

        Loads the trajectory's first and last frames, calls `sim.fit(frame_first,
        frame_last)`, then applies the trajectory's ground-truth actions through
        `sim.update`. After each step, records the simulator's `terminal_cost()`
        value and renders a frame. Finally computes the L2 image-distance between
        the simulator's predicted final rendered frame and the ground-truth final
        frame.

        Use this to verify your simulator is non-degenerate. If `update(a)` is a
        no-op, the predicted final position stays at the start and the image
        distance will be very high. If `terminal_cost` is a constant, you will
        see it not decreasing across the trajectory.

        Args:
            trajectory_index: Which training trajectory to validate against.
            max_steps: Optional cap on number of actions to replay (defaults to all).

        Returns:
            A summary string. PRIMARY signal: the terminal_cost reduction_ratio
            (final/initial) — task-agnostic and render-independent; a faithful
            simulator drives it toward 0 under ground-truth-action replay (the
            target was fit from the true final frame). SECONDARY: the pixel
            image_distance, which is render-style dependent (a high constant floor
            usually means the renderer doesn't pixel-match the dataset, not a
            dynamics error). Includes flags for a constant loss, a constant
            predicted frame, and a reduction_ratio that fails to fall.
        """
        try:
            t_idx = int(trajectory_index)
            traj = _dt.read_trajectory(self._dataset_dir, t_idx)
        except (FileNotFoundError, ValueError) as exc:
            err = f"[validate_against_training] {exc}"
            self._log_critic_tool_call(
                tool_name="validate_against_training",
                args={"trajectory_index": str(trajectory_index)},
                result_summary=err,
            )
            return err

        # Spawn a subprocess that imports the current sandbox code, fits on
        # (first, last) frames of the trajectory, replays ground-truth actions,
        # and reports the requested metrics. We use a fresh subprocess so any
        # crash in the generated code does not poison this process.
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        ground_truth_actions = traj["actions"]
        if max_steps is not None:
            ground_truth_actions = ground_truth_actions[: int(max_steps)]

        gt_first_path = str(traj["frames"][0])
        gt_last_path = str(traj["frames"][-1])

        # Inline runner script (kept here so we don't add another module).
        # Pass all ground-truth frame paths so the runner can compute per-step distance.
        gt_frame_paths = [str(p) for p in traj["frames"]]

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "def _numeric_components(state):\n"
            "    out = {}\n"
            "    if isinstance(state, dict):\n"
            "        items = state.items()\n"
            "    else:\n"
            "        items = [('state', state)]\n"
            "    for k, v in items:\n"
            "        try:\n"
            "            arr = np.array(v, dtype=np.float64).reshape(-1).copy()\n"
            "        except Exception:\n"
            "            continue\n"
            "        if arr.size and np.all(np.isfinite(arr)):\n"
            "            out[str(k)] = arr\n"
            "    return out\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "image_A = np.array(Image.open(args['gt_first']).convert('RGB'))\n"
            "image_B = np.array(Image.open(args['gt_last']).convert('RGB'))\n"
            "sim.fit(image_A, image_B)\n"
            "losses = [float(sim.planning_objective())]\n"
            "states = [_numeric_components(sim.state)]\n"
            "gt_frames = [np.array(Image.open(p).convert('RGB')).astype(np.float32) for p in args['gt_frames']]\n"
            "predicted = [sim.render_frame().astype(np.float32)]\n"
            "for a in args['actions']:\n"
            "    sim.update(np.asarray(a, dtype=np.float64))\n"
            "    losses.append(float(sim.planning_objective()))\n"
            "    states.append(_numeric_components(sim.state))\n"
            "    predicted.append(sim.render_frame().astype(np.float32))\n"
            "n_compare = min(len(predicted), len(gt_frames))\n"
            "per_step_distances = [float(np.linalg.norm(predicted[i] - gt_frames[i])) for i in range(n_compare)]\n"
            "state_tracking = {}\n"
            "if n_compare:\n"
            "    n_checks = min(6, n_compare)\n"
            "    check_indices = sorted(set(np.linspace(0, n_compare - 1, n_checks).round().astype(int).tolist()))\n"
            "    for idx in check_indices:\n"
            "        try:\n"
            "            parser = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "            parser.fit(gt_frames[idx].astype(np.uint8), image_B)\n"
            "            parsed = _numeric_components(parser.state)\n"
            "            rollout = states[idx] if idx < len(states) else {}\n"
            "            for name in sorted(set(parsed) & set(rollout)):\n"
            "                a = np.asarray(parsed[name], dtype=np.float64).reshape(-1)\n"
            "                b = np.asarray(rollout[name], dtype=np.float64).reshape(-1)\n"
            "                if a.shape != b.shape:\n"
            "                    continue\n"
            "                state_tracking.setdefault(name, []).append({'step': int(idx), 'error': float(np.linalg.norm(a - b))})\n"
            "        except Exception:\n"
            "            pass\n"
            "state_tracking_summary = {}\n"
            "for name, rows in state_tracking.items():\n"
            "    errs = [float(r['error']) for r in rows]\n"
            "    state_tracking_summary[name] = {'mean_error': float(np.mean(errs)), 'max_error': float(max(errs)), 'samples': rows}\n"
            "out = {'image_distance_final': per_step_distances[-1] if per_step_distances else 0.0,"
            " 'image_distance_per_step': per_step_distances,"
            " 'image_distance_mean': float(np.mean(per_step_distances)) if per_step_distances else 0.0,"
            " 'image_distance_max': float(max(per_step_distances)) if per_step_distances else 0.0,"
            " 'state_tracking': state_tracking_summary,"
            " 'losses': losses,"
            " 'predicted_final_shape': list(predicted[-1].shape),"
            " 'predicted_final_min': int(predicted[-1].min()),"
            " 'predicted_final_max': int(predicted[-1].max())}\n"
            "print(json.dumps(out))\n"
        )

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "gt_first": gt_first_path,
                "gt_last": gt_last_path,
                "gt_frames": gt_frame_paths,
                "actions": ground_truth_actions.tolist(),
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=60,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            err = (
                f"[validate_against_training] runner exited with code "
                f"{result.returncode}\nstderr:\n{result.stderr[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="validate_against_training",
                args={
                    "trajectory_index": t_idx,
                    "max_steps": str(max_steps),
                },
                result_summary=err[:300],
                extra_files={"stderr.txt": result.stderr},
            )
            return err

        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            err = (
                f"[validate_against_training] could not parse runner output: "
                f"{exc}\nstdout:\n{result.stdout[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="validate_against_training",
                args={"trajectory_index": t_idx},
                result_summary=err[:300],
            )
            return err

        losses = payload["losses"]
        per_step_distances = payload["image_distance_per_step"]
        image_distance_final = payload["image_distance_final"]
        image_distance_mean = payload["image_distance_mean"]
        image_distance_max = payload["image_distance_max"]
        loss_initial = losses[0] if losses else 0.0
        loss_final = losses[-1] if losses else 0.0
        loss_constant = (
            len(losses) > 1
            and min(losses) == max(losses)
        )
        # PRIMARY fidelity signal: the simulator's OWN terminal_cost, in its own
        # state space, after replaying ground-truth actions. target_state was fit
        # from the TRUE final frame, so a faithful simulator drives the loss toward
        # zero -> reduction_ratio (final/initial) toward 0. This is task-agnostic
        # (the VLM defines state/fit/loss; the harness assumes nothing about the
        # task) and render-independent, unlike image_distance below.
        loss_reduction_ratio = (
            loss_final / loss_initial if loss_initial > 0 else None
        )
        ratio_str = (
            f"{loss_reduction_ratio:.3f}" if loss_reduction_ratio is not None else "n/a"
        )
        pf_min = payload["predicted_final_min"]
        pf_max = payload["predicted_final_max"]
        state_tracking = payload.get("state_tracking") or {}

        per_step_preview = ", ".join(f"{d:.1f}" for d in per_step_distances[:6])
        if len(per_step_distances) > 6:
            per_step_preview += f", ... (total {len(per_step_distances)} steps)"
        if state_tracking:
            tracking_line = "; ".join(
                f"{name}: mean={float(vals.get('mean_error', 0.0)):.3f}, "
                f"max={float(vals.get('max_error', 0.0)):.3f}"
                for name, vals in sorted(state_tracking.items())
            )
        else:
            tracking_line = "n/a (no shared numeric state components could be parsed)"

        summary = (
            f"Validated against trajectory {t_idx} ({len(ground_truth_actions)} actions).\n"
            f"  PRIMARY fidelity — terminal_cost (your own state-space metric):\n"
            f"    initial={loss_initial:.4f}  final={loss_final:.4f}  "
            f"reduction_ratio={ratio_str}  (min={min(losses):.4f}, max={max(losses):.4f})\n"
            f"    Replaying the ground-truth actions should drive terminal_cost toward 0 "
            f"(target_state is fit from the true final frame). reduction_ratio near 0 = "
            f"faithful dynamics; near 1 = the actions barely move the state toward target.\n"
            f"  SECONDARY — image_distance (render-style dependent; a high floor here is "
            f"often just a renderer that doesn't pixel-match the dataset, NOT a dynamics "
            f"error — judge fidelity by terminal_cost above):\n"
            f"    per step [t=0..]: {per_step_preview}\n"
            f"    final={image_distance_final:.2f}, mean={image_distance_mean:.2f}, max={image_distance_max:.2f}\n"
            f"  STATE TRACKING — rollout state vs fit(true intermediate frame) at checkpoints:\n"
            f"    {tracking_line}\n"
            f"  predicted_final_frame pixel range: [{pf_min}, {pf_max}]"
        )
        # PRIMARY correctness flag: ground-truth-action replay should collapse the
        # loss. If it doesn't, update(a) or fit(goal) is wrong — independent of any
        # render mismatch. (loss_constant is the degenerate case, flagged below.)
        if (
            loss_reduction_ratio is not None
            and not loss_constant
            and loss_reduction_ratio > 0.5
        ):
            summary += (
                f"\n  WARNING: terminal_cost only fell to {loss_reduction_ratio:.0%} of "
                "its initial value when replaying the ground-truth actions. A faithful "
                "simulator should drive it close to 0 (the true actions, by definition, "
                "reproduce the true final frame from which target_state was fit). This "
                "points to a dynamics error in update(a) or a goal mis-parse in fit() — "
                "not a rendering issue."
            )
            if loss_reduction_ratio > 1.0:
                summary += (
                    " NOTE: reduction_ratio > 1 means the loss GREW — your dynamics move "
                    "the state AWAY from the target under the true (correct) actions. "
                    "Possible causes: a coordinate sign/axis flip (e.g. image y-down vs "
                    "physics y-up), a wrong update rule, or — for contact/pushing tasks — "
                    "the action not actually transferring motion to the object the way the "
                    "data shows (the pushed object barely moves, or moves the wrong way). "
                    "Look at HOW each part of the state moves under the true actions to tell "
                    "which it is, rather than assuming a single cause."
                )
        # SECONDARY hint (only meaningful when your renderer pixel-matches the
        # dataset — otherwise the render-mismatch floor dominates): intermediate
        # frames much worse than the final one suggests the path diverges even
        # though the endpoint lands close — e.g. update(a) ignores obstacles.
        if image_distance_mean > 1.5 * image_distance_final and image_distance_final > 0:
            summary += (
                "\n  (secondary) mean per-step image_distance >> final image_distance: "
                "if your renderer matches the dataset, this suggests the simulator's "
                "intermediate path diverges from the data even though the endpoint is "
                "close (e.g. update(a) does not enforce geometric constraints / obstacles)."
            )
        if loss_constant:
            summary += (
                "\n  WARNING: terminal_cost is CONSTANT across the trajectory. "
                "This means either update(a) does not mutate self.state, or "
                "terminal_cost does not depend on self.state."
            )
        if pf_min == pf_max:
            summary += (
                f"\n  WARNING: predicted_final_frame is a CONSTANT colour "
                f"({pf_min}). render_frame() likely ignores self.state."
            )

        self._log_critic_tool_call(
            tool_name="validate_against_training",
            args={
                "trajectory_index": t_idx,
                "max_steps": str(max_steps),
            },
            result_summary=summary[:500],
            extra_files={
                "losses.json": _json.dumps(losses),
                "predicted_final_shape.json": _json.dumps(payload["predicted_final_shape"]),
            },
        )
        return summary

    def validate_all_trajectories(
        self,
        max_steps: int | None = None,
    ) -> str:
        """Replay every training trajectory and summarise fidelity.

        Sweeps the full training set (``gate_trajectories`` when configured,
        otherwise every ``trajectory_*`` under the dataset root). Returns a
        per-trajectory ``reduction_ratio`` table, the mean, and the WORST
        (least-faithful) trajectory with its full ``validate_against_training``
        detail. Read-only — does not edit or select code.
        """
        import re as _re

        trajs = self._fidelity_train_trajectories()
        if not trajs:
            err = (
                "[validate_all_trajectories] no training trajectories found "
                f"under {self._dataset_dir}"
            )
            self._log_critic_tool_call(
                tool_name="validate_all_trajectories",
                args={"max_steps": str(max_steps)},
                result_summary=err,
            )
            return err

        rows: list[tuple[int, float]] = []
        failures: list[tuple[int, str]] = []
        worst_t: int | None = None
        worst_ratio = -1.0
        worst_summary = ""
        for t in trajs:
            try:
                summary = self.validate_against_training(
                    trajectory_index=t, max_steps=max_steps
                )
            except Exception as exc:
                failures.append((t, f"{type(exc).__name__}: {exc}"))
                continue
            m = _re.search(r"reduction_ratio=([0-9.]+|n/a)", summary)
            if m and m.group(1) != "n/a":
                ratio = float(m.group(1))
                rows.append((t, ratio))
                if ratio > worst_ratio:
                    worst_ratio, worst_t, worst_summary = ratio, t, summary
            else:
                failures.append((t, "no reduction_ratio (degenerate/crash)"))

        ratio_line = ", ".join(f"t{t}={r:.3f}" for t, r in rows) or "(none parsed)"
        mean_ratio = sum(r for _, r in rows) / len(rows) if rows else None
        mean_str = f"{mean_ratio:.3f}" if mean_ratio is not None else "n/a"

        msg = (
            f"[validate_all_trajectories] swept {len(trajs)} training trajectory/ies "
            f"({trajs}).\n"
            f"  reduction_ratio per trajectory: {ratio_line}\n"
            f"  mean reduction_ratio: {mean_str}\n"
        )
        if failures:
            msg += (
                "  FAILED/degenerate: "
                + ", ".join(f"t{t} ({why})" for t, why in failures)
                + "\n"
            )
        if worst_t is not None:
            msg += (
                f"  WORST = trajectory {worst_t} (ratio {worst_ratio:.3f}); "
                f"full detail below — improve dynamics here without breaking "
                f"the others:\n{worst_summary}"
            )
        elif failures and not rows:
            msg += (
                "  WORST = (none ran successfully); see failure detail above.\n"
            )

        self._log_critic_tool_call(
            tool_name="validate_all_trajectories",
            args={"max_steps": str(max_steps)},
            result_summary=msg[:500],
        )
        return msg

    def _run_state_consistency_check(
        self,
        trajectory_index: int,
        *,
        max_checkpoints: int = 12,
        return_states: bool = False,
        checkpoint_steps: list[int] | None = None,
        max_numeric_state_dim: int | None = None,
        max_numeric_state_keys: int | None = None,
    ) -> dict[str, Any]:
        """Perception-anchored per-step fidelity for one trajectory.

        Rolls the simulator forward from ``fit(frame_0, frame_last)`` under
        ground-truth actions, and at each checkpoint re-parses the TRUE frame
        with a fresh ``fit(frame_t, frame_last)``. The per-component gap
        between the rollout state and the sim's own parse of reality localises
        WHERE (which step) and WHICH component the dynamics diverge —
        independent of terminal_cost shape and renderer style.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        try:
            t_idx = int(trajectory_index)
            traj = _dt.read_trajectory(self._dataset_dir, t_idx)
        except (FileNotFoundError, ValueError) as exc:
            return {"ok": False, "error": str(exc), "trajectory": trajectory_index}

        actions = np.asarray(traj["actions"], dtype=np.float64)
        n_actions = int(actions.shape[0])
        if n_actions == 0 or len(traj["frames"]) < 2:
            return {
                "ok": False,
                "error": "trajectory has no actions/frames",
                "trajectory": t_idx,
            }

        if checkpoint_steps is None:
            n_ck = max(3, min(int(max_checkpoints), n_actions + 1))
            checkpoints = sorted(
                {int(round(x)) for x in np.linspace(0, n_actions, n_ck)}
            )
        else:
            checkpoints = sorted(
                {
                    max(0, min(n_actions, int(step)))
                    for step in checkpoint_steps
                }
            )
            if not checkpoints:
                return {
                    "ok": False,
                    "error": "checkpoint_steps selected no valid checkpoints",
                    "trajectory": t_idx,
                }
        frame_paths = [str(p) for p in traj["frames"]]

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "def _mk():\n"
            "    api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "    return cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "skipped = {}\n"
            "skip_overflow = {'count': 0}\n"
            "def _skip(k, reason, arr=None):\n"
            "    key = str(k)[:80]\n"
            "    if key in skipped:\n"
            "        return\n"
            "    if len(skipped) >= 64:\n"
            "        skip_overflow['count'] += 1\n"
            "        return\n"
            "    item = {'reason': reason}\n"
            "    if arr is not None:\n"
            "        item['shape'] = list(arr.shape)[:4]\n"
            "        item['size'] = int(arr.size)\n"
            "        item['dtype'] = str(arr.dtype)[:24]\n"
            "    skipped[key] = item\n"
            "def _numeric(s):\n"
            "    items = s.items() if isinstance(s, dict) else [('__state__', s)]\n"
            "    out = {}\n"
            "    for k, v in items:\n"
            "        key = str(k)[:80]\n"
            "        key_limit = args.get('max_numeric_state_keys')\n"
            "        if key_limit is not None and len(out) >= int(key_limit):\n"
            "            _skip(key, 'component_count_limit')\n"
            "            continue\n"
            "        try:\n"
            "            arr = np.asarray(v)\n"
            "        except Exception:\n"
            "            _skip(key, 'not_array_like')\n"
            "            continue\n"
            "        dim_limit = args.get('max_numeric_state_dim')\n"
            "        if dim_limit is not None and arr.dtype.kind == 'b':\n"
            "            _skip(key, 'boolean_or_mask', arr)\n"
            "            continue\n"
            "        if arr.dtype.kind not in 'iufb':\n"
            "            _skip(key, 'non_numeric', arr)\n"
            "            continue\n"
            "        if dim_limit is not None and arr.ndim >= 2 and arr.size > int(dim_limit):\n"
            "            _skip(key, 'image_shaped', arr)\n"
            "            continue\n"
            "        if dim_limit is not None and arr.size > int(dim_limit):\n"
            "            _skip(key, 'oversized', arr)\n"
            "            continue\n"
            "        flat = np.asarray(arr, dtype=np.float64).reshape(-1)\n"
            "        if not flat.size:\n"
            "            _skip(key, 'empty', arr)\n"
            "            continue\n"
            "        if not np.all(np.isfinite(flat)):\n"
            "            _skip(key, 'non_finite', arr)\n"
            "            continue\n"
            "        out[key] = flat.tolist()\n"
            "    return out\n"
            "frames = args['frame_paths']\n"
            "goal = np.array(Image.open(frames[-1]).convert('RGB'))\n"
            "cks = list(args['checkpoints'])\n"
            "ck_set = set(cks)\n"
            "sim = _mk()\n"
            "sim.fit(np.array(Image.open(frames[0]).convert('RGB')), goal)\n"
            "rollout = {}\n"
            "if 0 in ck_set:\n"
            "    rollout['0'] = _numeric(sim.state)\n"
            "for t, a in enumerate(args['actions'], start=1):\n"
            "    sim.update(np.asarray(a, dtype=np.float64))\n"
            "    if t in ck_set:\n"
            "        rollout[str(t)] = _numeric(sim.state)\n"
            "parsed = {}\n"
            "errors = {}\n"
            "for t in cks:\n"
            "    try:\n"
            "        s2 = _mk()\n"
            "        s2.fit(np.array(Image.open(frames[t]).convert('RGB')), goal)\n"
            "        parsed[str(t)] = _numeric(s2.state)\n"
            "    except Exception as exc:\n"
            "        errors[str(t)] = f'{type(exc).__name__}: {exc}'\n"
            "if skip_overflow['count']:\n"
            "    skipped['__additional_skipped__'] = {'reason': 'report_limit', 'count': skip_overflow['count']}\n"
            "print(json.dumps({'rollout': rollout, 'parsed': parsed, 'errors': errors, 'skipped': skipped}))\n"
        )

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "frame_paths": frame_paths,
                "actions": actions.tolist(),
                "checkpoints": checkpoints,
                "max_numeric_state_dim": max_numeric_state_dim,
                "max_numeric_state_keys": max_numeric_state_keys,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=180,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return {
                "ok": False,
                "error": (
                    f"runner exited with code {result.returncode}: "
                    f"{result.stderr[-800:]}"
                ),
                "trajectory": t_idx,
            }
        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {
                "ok": False,
                "error": f"could not parse runner output: {exc}",
                "trajectory": t_idx,
            }

        rollout = {int(k): v for k, v in payload.get("rollout", {}).items()}
        parsed = {int(k): v for k, v in payload.get("parsed", {}).items()}
        parse_errors = {int(k): v for k, v in payload.get("errors", {}).items()}
        skipped_components = payload.get("skipped", {})

        usable_cks = [t for t in checkpoints if t in rollout and t in parsed]
        if not usable_cks:
            return {
                "ok": False,
                "error": (
                    "no usable checkpoints (fit() failed on intermediate "
                    f"frames: {dict(list(parse_errors.items())[:3])})"
                ),
                "trajectory": t_idx,
            }

        base_t = usable_cks[0]
        keys: set[str] = set()
        for t in usable_cks:
            keys |= set(rollout[t].keys()) & set(parsed[t].keys())

        components: dict[str, dict[str, Any]] = {}
        for key in sorted(keys):
            drifts: list[tuple[int, float]] = []
            motions: list[float] = []
            for t in usable_cks:
                r = rollout[t].get(key)
                p = parsed[t].get(key)
                if r is None or p is None or len(r) != len(p):
                    continue
                drifts.append(
                    (
                        t,
                        self._component_delta_norm(
                            key, np.asarray(p), np.asarray(r)
                        ),
                    )
                )
                p0 = parsed[base_t].get(key)
                if p0 is not None and len(p0) == len(p):
                    motions.append(
                        self._component_delta_norm(
                            key, np.asarray(p0), np.asarray(p)
                        )
                    )
            if not drifts:
                continue
            motion = max(motions) if motions else 0.0
            max_t, max_drift = max(drifts, key=lambda td: td[1])
            final_drift = drifts[-1][1]
            static = motion <= 1e-9
            latent_key = self._is_latent_state_key(key)
            if static:
                if max_drift <= 1e-6:
                    verdict = "static"
                    onset = None
                elif (
                    self._gate_state_consistency_skip_latent
                    and latent_key
                ):
                    # Image re-parse cannot observe velocities/latents: fit()
                    # typically re-initialises them to 0 while rollout integrates.
                    verdict = "latent_unobserved"
                    onset = None
                else:
                    verdict = "DRIFT"
                    onset = next(
                        (t for t, d in drifts if d > 1e-6),
                        max_t,
                    )
            elif max_drift <= 0.25 * motion:
                verdict = "OK"
                onset = None
            else:
                verdict = "DRIFT"
                onset = next(
                    (t for t, d in drifts if d > 0.25 * motion), max_t
                )
            components[key] = {
                "motion": float(motion),
                "max_drift": float(max_drift),
                "max_drift_at": int(max_t),
                "final_drift": float(final_drift),
                "drift_frac_of_motion": (
                    float(max_drift / motion) if motion > 1e-9 else None
                ),
                "onset_step": onset,
                "verdict": verdict,
                "latent": bool(latent_key),
                "drifts": [(int(t), float(d)) for t, d in drifts],
            }

        def _fmt_component_line(key: str, c: dict[str, Any]) -> str:
            if c["verdict"] == "static":
                return f"{key}: static in data (skipped)"
            if c["verdict"] == "latent_unobserved":
                return (
                    f"{key}: latent/unobserved in still-frame parse (skipped "
                    f"for hard gate; rollout gap to {c['max_drift']:.3g})"
                )
            if c["verdict"] == "DRIFT" and c["drift_frac_of_motion"] is None:
                return (
                    f"{key}: DRIFT — static in parsed frames but rollout gap "
                    f"grows to {c['max_drift']:.3g} by t={c['max_drift_at']}, "
                    f"from ~t={c['onset_step']}. If this quantity never moves "
                    "under update(a), store it in target_state / attrs, not "
                    "self.state."
                )
            pct = (
                f"{100.0 * c['drift_frac_of_motion']:.0f}% of its motion"
                if c["drift_frac_of_motion"] is not None
                else "n/a"
            )
            if c["verdict"] == "OK":
                return (
                    f"{key}: OK — max gap {c['max_drift']:.3g} "
                    f"({pct}) at t={c['max_drift_at']}"
                )
            return (
                f"{key}: DRIFT — gap grows to {c['max_drift']:.3g} "
                f"({pct}) by t={c['max_drift_at']}, "
                f"exceeds 25% from ~t={c['onset_step']}"
            )

        compact_lines = [_fmt_component_line(k, c) for k, c in components.items()]
        if parse_errors:
            compact_lines.append(
                f"fit() failed on {len(parse_errors)} checkpoint frame(s): "
                + ", ".join(f"t{t}" for t in sorted(parse_errors)[:5])
            )

        drifting = [k for k, c in components.items() if c["verdict"] == "DRIFT"]
        oneline = (
            f"t{t_idx}: " + "; ".join(
                f"{k}={components[k]['verdict']}"
                + (
                    (
                        f"({100.0 * components[k]['drift_frac_of_motion']:.0f}%"
                        f"@t{components[k]['onset_step']})"
                        if components[k]["drift_frac_of_motion"] is not None
                        else f"(static@t{components[k]['onset_step']})"
                    )
                    if components[k]["verdict"] == "DRIFT"
                    else ""
                )
                for k in sorted(components)
            )
            if components
            else f"t{t_idx}: no comparable components"
        )

        table_lines: list[str] = []
        for key, c in components.items():
            picks = c["drifts"]
            if len(picks) > 7:
                idxs = np.linspace(0, len(picks) - 1, 7).round().astype(int)
                picks = [picks[i] for i in dict.fromkeys(idxs.tolist())]
            table_lines.append(
                f"  {key} (motion={c['motion']:.3g}): "
                + "  ".join(f"t{t}:{d:.3g}" for t, d in picks)
                + f"  -> {_fmt_component_line(key, c)}"
            )

        result_payload = {
            "ok": True,
            "error": None,
            "trajectory": t_idx,
            "n_actions": n_actions,
            "checkpoints": usable_cks,
            "components": components,
            "parse_errors": parse_errors,
            "skipped_components": skipped_components,
            "drifting_components": drifting,
            "oneline": oneline,
            "compact_lines": compact_lines,
            "table_lines": table_lines,
        }
        if return_states:
            # Internal system-identification callers need the simulator's own
            # true-frame parses. Keep this opt-in so ordinary gate payloads and
            # metrics do not balloon with every checkpoint state.
            result_payload["parsed_states"] = parsed
            result_payload["rollout_states"] = rollout
        return result_payload

    def estimate_dynamics_models(
        self,
        trajectory_indices: list[int] | None = None,
        max_trajectories: int = 3,
        max_state_dim: int = 16,
        max_action_dim: int = 16,
        max_components: int = 12,
    ) -> str:
        """Fit simple candidate dynamics to states parsed from consecutive frames.

        The generated simulator's current ``fit()`` defines the state
        coordinates. This tool reparses every true frame and compares:

        * first-order: ``delta_state ~ action``;
        * inertial: ``delta_state ~ action + previous_delta``;
        * setpoint (matching dimensions only): ``delta_state ~ action - state``.

        Only compact observable numeric components are admitted. Boolean masks,
        image-shaped arrays, non-finite values, and components above
        ``max_state_dim`` are skipped before conversion/serialization. It does
        not modify code. Low error suggests a reusable inherited dynamics
        helper; uniformly poor fits or sparse motion suggest contact/event/
        kinematic structure rather than another global gain.
        """
        if not self.has_valid_simulator_class():
            message = (
                "[estimate_dynamics_models] Write a runnable simulator with a "
                "credible fit() first; dynamics identification uses its parsed "
                "state coordinates."
            )
            self._log_critic_tool_call(
                tool_name="estimate_dynamics_models",
                args={},
                result_summary=message,
            )
            return message

        max_state_dim = max(1, min(int(max_state_dim), 32))
        max_action_dim = max(1, min(int(max_action_dim), 32))
        max_components = max(1, min(int(max_components), 24))
        if trajectory_indices is None:
            candidates = list(self._gate_trajectories or self._auto_validate_trajs)
        else:
            candidates = [int(value) for value in trajectory_indices]
        trajectories = self._measurement_trajectories(
            candidates,
            max_trajectories=max(1, int(max_trajectories)),
        )
        rows: list[dict[str, Any]] = []
        errors: list[str] = []
        skipped: dict[str, dict[str, Any]] = {}
        actions_by_traj: dict[int, np.ndarray] = {}
        parsed_by_traj: dict[int, dict[int, dict[str, list[float]]]] = {}
        for trajectory in trajectories:
            try:
                data = _dt.read_trajectory(self._dataset_dir, int(trajectory))
                actions = np.asarray(data["actions"], dtype=np.float64)
                if actions.ndim != 2 or actions.shape[1] > max_action_dim:
                    skipped["__actions__"] = {
                        "reason": "oversized_action_dimension",
                        "shape": list(actions.shape),
                        "limit": max_action_dim,
                    }
                    errors.append(
                        f"t{trajectory}: action shape {tuple(actions.shape)} exceeds "
                        f"compact limit {max_action_dim}"
                    )
                    continue
                n_actions = int(actions.shape[0])
                transition_starts = sorted(
                    {
                        int(round(value))
                        for value in np.linspace(
                            0,
                            max(0, n_actions - 1),
                            min(24, max(1, n_actions)),
                        )
                    }
                )
                checkpoint_steps = sorted(
                    {
                        step
                        for start in transition_starts
                        for step in (max(0, start - 1), start, start + 1)
                    }
                )
                check = self._run_state_consistency_check(
                    int(trajectory),
                    max_checkpoints=len(checkpoint_steps),
                    return_states=True,
                    checkpoint_steps=checkpoint_steps,
                    max_numeric_state_dim=max_state_dim,
                    max_numeric_state_keys=max_components,
                )
                if not check.get("ok"):
                    errors.append(
                        (
                            f"t{trajectory}: "
                            f"{check.get('error', 'parse failed')}"
                        )[:500]
                    )
                    continue
                actions_by_traj[int(trajectory)] = actions
                parsed_by_traj[int(trajectory)] = {
                    int(step): state
                    for step, state in (check.get("parsed_states") or {}).items()
                }
                for key, reason in (check.get("skipped_components") or {}).items():
                    skipped.setdefault(str(key), reason)
            except Exception as exc:
                errors.append(
                    f"t{trajectory}: {type(exc).__name__}: {str(exc)[:400]}"
                )

        keys: set[str] = set()
        for parsed in parsed_by_traj.values():
            for state in parsed.values():
                keys.update(state)

        def _fit_candidate(
            features: np.ndarray,
            targets: np.ndarray,
        ) -> dict[str, Any] | None:
            if features.ndim != 2 or targets.ndim != 2 or len(features) < 6:
                return None
            split = max(4, int(round(0.8 * len(features))))
            split = min(split, len(features) - 1)
            train_x, test_x = features[:split], features[split:]
            train_y, test_y = targets[:split], targets[split:]
            design = np.column_stack([train_x, np.ones(len(train_x))])
            coefficients, *_ = np.linalg.lstsq(design, train_y, rcond=None)
            predicted = np.column_stack([test_x, np.ones(len(test_x))]) @ coefficients
            error = predicted - test_y
            rmse = float(np.sqrt(np.mean(error * error)))
            scale = float(np.sqrt(np.mean(test_y * test_y)))
            normalized = float(rmse / max(scale, 1e-9))
            residual = float(np.sum(error * error))
            centered = test_y - np.mean(test_y, axis=0, keepdims=True)
            total = float(np.sum(centered * centered))
            r2 = float(1.0 - residual / max(total, 1e-12))
            weight_array = np.round(coefficients[:-1].T, 6)
            coefficient_count = int(weight_array.size + coefficients[-1].size)
            model = {
                "n_train": int(len(train_x)),
                "n_test": int(len(test_x)),
                "normalized_rmse": round(normalized, 4),
                "r2": round(r2, 4),
                "coefficient_count": coefficient_count,
                "weight_shape": list(weight_array.shape),
                "bias": np.round(coefficients[-1], 6).tolist(),
            }
            # Keep complete coefficients for ordinary compact scalar/vector
            # states. Larger allowed systems get a small, explicitly labelled
            # preview rather than hundreds of API-bound numbers.
            if coefficient_count <= 48:
                model["weights"] = weight_array.tolist()
            else:
                model["weights_preview"] = weight_array.reshape(-1)[:32].tolist()
                model["coefficients_truncated"] = True
            return model

        for key in sorted(keys):
            action_features: list[np.ndarray] = []
            inertial_features: list[np.ndarray] = []
            setpoint_features: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            setpoint_targets: list[np.ndarray] = []
            delta_norms: list[float] = []
            state_size: int | None = None
            action_size: int | None = None

            for trajectory in trajectories:
                actions = actions_by_traj.get(int(trajectory))
                parsed = parsed_by_traj.get(int(trajectory))
                if actions is None or not parsed:
                    continue
                previous_delta: np.ndarray | None = None
                for step in range(int(actions.shape[0])):
                    if step not in parsed or step + 1 not in parsed:
                        previous_delta = None
                        continue
                    current_raw = parsed[step].get(key)
                    next_raw = parsed[step + 1].get(key)
                    if current_raw is None or next_raw is None:
                        previous_delta = None
                        continue
                    current = np.asarray(current_raw, dtype=np.float64).reshape(-1)
                    following = np.asarray(next_raw, dtype=np.float64).reshape(-1)
                    action = np.asarray(actions[step], dtype=np.float64).reshape(-1)
                    if current.shape != following.shape or current.size == 0:
                        previous_delta = None
                        continue
                    delta = following - current
                    if self._is_angle_state_key(key):
                        delta = np.arctan2(np.sin(delta), np.cos(delta))
                    state_size = int(current.size)
                    action_size = int(action.size)
                    action_features.append(action)
                    targets.append(delta)
                    delta_norms.append(float(np.linalg.norm(delta)))
                    if previous_delta is not None:
                        inertial_features.append(
                            np.concatenate([action, previous_delta])
                        )
                    else:
                        # Keep feature/target alignment by omitting the first
                        # transition of each trajectory from the inertial fit.
                        pass
                    if current.size == action.size:
                        error_to_setpoint = action - current
                        if self._is_angle_state_key(key):
                            error_to_setpoint = np.arctan2(
                                np.sin(error_to_setpoint),
                                np.cos(error_to_setpoint),
                            )
                        setpoint_features.append(error_to_setpoint)
                        setpoint_targets.append(delta)
                    previous_delta = delta

            if not targets or state_size is None or action_size is None:
                continue
            target_array = np.asarray(targets, dtype=np.float64)
            first = _fit_candidate(
                np.asarray(action_features, dtype=np.float64),
                target_array,
            )
            # Inertial targets omit the first transition of each trajectory.
            inertial_targets: list[np.ndarray] = []
            for trajectory in trajectories:
                actions = actions_by_traj.get(int(trajectory))
                parsed = parsed_by_traj.get(int(trajectory))
                if actions is None or not parsed:
                    continue
                for step in range(1, int(actions.shape[0])):
                    if step not in parsed or step + 1 not in parsed:
                        continue
                    current_raw = parsed[step].get(key)
                    next_raw = parsed[step + 1].get(key)
                    previous_raw = parsed[step - 1].get(key)
                    if (
                        current_raw is None
                        or next_raw is None
                        or previous_raw is None
                    ):
                        continue
                    current = np.asarray(current_raw, dtype=float).reshape(-1)
                    following = np.asarray(next_raw, dtype=float).reshape(-1)
                    prior = np.asarray(previous_raw, dtype=float).reshape(-1)
                    if current.shape != following.shape or current.shape != prior.shape:
                        continue
                    delta = following - current
                    if self._is_angle_state_key(key):
                        delta = np.arctan2(np.sin(delta), np.cos(delta))
                    inertial_targets.append(delta)
            inertial = _fit_candidate(
                np.asarray(inertial_features, dtype=np.float64),
                np.asarray(inertial_targets, dtype=np.float64),
            )
            setpoint = _fit_candidate(
                np.asarray(setpoint_features, dtype=np.float64),
                np.asarray(setpoint_targets, dtype=np.float64),
            )
            moving_fraction = float(
                np.mean(np.asarray(delta_norms, dtype=float) > 1e-6)
            )
            models = {
                name: model
                for name, model in (
                    ("first_order_action", first),
                    ("inertial_action_plus_previous_delta", inertial),
                    ("setpoint_error", setpoint),
                )
                if model is not None
            }
            ranked = sorted(
                models,
                key=lambda name: float(models[name]["normalized_rmse"]),
            )
            best = ranked[0] if ranked else None
            warning = None
            if moving_fraction < 0.20:
                warning = (
                    "motion is sparse; a global regression can look good by "
                    "predicting no motion. Inspect contact/event conditions."
                )
            elif best and float(models[best]["normalized_rmse"]) > 0.75:
                warning = (
                    "all simple models fit poorly; use structured kinematics, "
                    "contact, mode switches, or improve perception."
                )
            rows.append(
                {
                    "state_key": key,
                    "state_dim": state_size,
                    "action_dim": action_size,
                    "n_transitions": int(len(targets)),
                    "moving_fraction": round(moving_fraction, 4),
                    "best_model": best,
                    "models": models,
                    "warning": warning,
                }
            )

        payload = {
            "trajectories": trajectories,
            "limits": {
                "max_state_dim": max_state_dim,
                "max_action_dim": max_action_dim,
                "max_components": max_components,
                "max_coefficients_per_model": 48,
                "coefficient_preview_values": 32,
            },
            "components": rows,
            "skipped_components": skipped,
            "errors": errors,
            "interpretation": (
                "Use coefficients only after perception is stable. "
                "normalized_rmse <~0.35 is strong evidence; compare models, "
                "motion sparsity, action metadata, and held-out replay. "
                "Poor object fits with a well-fit controllable/pusher state "
                "usually indicate contact/event dynamics."
            ),
        }
        artifact_payload = payload
        artifact_text = json.dumps(artifact_payload, indent=2)
        # The defaults above already keep this small. This second hard cap
        # protects against unusually long key/error strings while preserving a
        # valid JSON artifact.
        if len(artifact_text) > 60000:
            for component in artifact_payload["components"]:
                for model in component.get("models", {}).values():
                    model.pop("weights", None)
                    model.pop("weights_preview", None)
                    model.pop("bias", None)
                    model["coefficients_omitted_from_artifact"] = True
            artifact_payload["artifact_truncated"] = True
            artifact_text = json.dumps(artifact_payload, indent=2)

        response_payload = json.loads(json.dumps(artifact_payload))
        message = "[estimate_dynamics_models]\n" + json.dumps(
            response_payload, indent=2
        )
        if len(message) > 14000:
            for component in response_payload["components"]:
                for model in component.get("models", {}).values():
                    model.pop("weights", None)
                    model.pop("weights_preview", None)
                    model.pop("bias", None)
                    model["coefficients_in_artifact"] = True
            response_payload["response_truncated"] = True
            message = "[estimate_dynamics_models]\n" + json.dumps(
                response_payload, indent=2
            )
        if len(message) > 14000:
            kept: list[dict[str, Any]] = []
            for component in response_payload["components"]:
                candidate = dict(response_payload)
                candidate["components"] = kept + [component]
                if len(json.dumps(candidate, indent=2)) > 13000:
                    break
                kept.append(component)
            response_payload["components"] = kept
            response_payload["components_omitted_from_response"] = (
                len(rows) - len(kept)
            )
            message = "[estimate_dynamics_models]\n" + json.dumps(
                response_payload, indent=2
            )
        if len(message) > 14000:
            message = (
                message[:13800]
                + "\n  [response hard-capped; bounded JSON artifact has the report]"
            )
        self._log_critic_tool_call(
            tool_name="estimate_dynamics_models",
            args={
                "trajectory_indices": trajectory_indices,
                "max_trajectories": max_trajectories,
                "max_state_dim": max_state_dim,
                "max_action_dim": max_action_dim,
                "max_components": max_components,
            },
            result_summary=message[:500],
            extra_files={"dynamics_models.json": artifact_text},
        )
        return message

    def validate_state_consistency(
        self,
        trajectory_index: int,
        max_checkpoints: int = 12,
    ) -> str:
        """Localise dynamics error: open-loop rollout vs your own parse of the true frames.

        Replays trajectory ``trajectory_index``'s ground-truth actions from
        ``fit(frame_0, frame_last)``, and at ~``max_checkpoints`` evenly spaced
        steps re-parses the TRUE frame at that step with a fresh
        ``fit(frame_t, frame_last)``. It then reports, per state component, the
        gap between the rollout state and your own perception of reality.

        This decomposes what ``reduction_ratio`` conflates:
        - gap ~0 everywhere            -> dynamics track the data; a poor ratio
          then points at terminal_cost shape or goal parsing, not update(a).
        - gap grows steadily           -> systematic dynamics bias for that
          component (wrong scale/sign/gain).
        - gap jumps at a specific step -> a mis-modelled event (contact,
          collision, grasp) around that step; inspect those frames.
        - gap only at checkpoint 0     -> fit() itself is unstable
          (perception noise floor).

        The drift is measured in YOUR state units against YOUR OWN fit() —
        it needs no rendering match and no ground-truth state access.

        Args:
            trajectory_index: training trajectory to analyse.
            max_checkpoints: number of frames to re-parse (default 12).

        Returns:
            A per-component drift table with onset steps and verdicts
            (OK / DRIFT / static), plus parse-failure notes.
        """
        res = self._run_state_consistency_check(
            trajectory_index, max_checkpoints=max_checkpoints
        )
        if not res.get("ok"):
            msg = (
                f"[validate_state_consistency] {res.get('error')}"
            )
            self._log_critic_tool_call(
                tool_name="validate_state_consistency",
                args={"trajectory_index": str(trajectory_index)},
                result_summary=msg[:300],
            )
            return msg

        msg = (
            f"[validate_state_consistency] trajectory {res['trajectory']} "
            f"({res['n_actions']} actions, checkpoints {res['checkpoints']})\n"
            "Per-component gap between open-loop rollout state and your own "
            "fit() parse of the TRUE frame at each checkpoint (units = your "
            "state units; 'motion' = total observed movement of that "
            "component in the parses):\n"
            + "\n".join(res["table_lines"])
        )
        if res["drifting_components"]:
            msg += (
                "\n  DRIFTING: "
                + ", ".join(res["drifting_components"])
                + " — fix update(a) for these components first: a gap that "
                "grows steadily means wrong scale/sign/gain; a sudden jump "
                "means a mis-modelled contact/event near the onset step."
            )
        else:
            msg += (
                "\n  All components track your own perception of the data. "
                "If reduction_ratio is still poor, the problem is likely "
                "terminal_cost shape or goal parsing in fit(image_B), not "
                "update(a)."
            )
        if res["parse_errors"]:
            msg += (
                f"\n  NOTE: fit() failed on {len(res['parse_errors'])} "
                "checkpoint frame(s): "
                + ", ".join(f"t{t}" for t in sorted(res["parse_errors"])[:5])
                + " — your parser must handle every intermediate frame, not "
                "only trajectory endpoints."
            )

        self._log_critic_tool_call(
            tool_name="validate_state_consistency",
            args={
                "trajectory_index": int(trajectory_index),
                "max_checkpoints": int(max_checkpoints),
            },
            result_summary=msg[:500],
            extra_files={
                "state_consistency.json": json.dumps(
                    {
                        k: v
                        for k, v in res.items()
                        if k not in ("table_lines", "compact_lines")
                    },
                    indent=2,
                    default=str,
                )
            },
        )
        return msg

    def _derive_realizability_limits(
        self,
        *,
        margin: float,
        trajectories: list[int] | None = None,
    ) -> tuple[dict | None, str | None]:
        """Offline Step-1 bound from replayed training trajectories."""
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        try:
            margin = float(margin)
        except (TypeError, ValueError):
            return (None, "margin must be a positive float")
        if margin <= 0:
            return (None, "margin must be > 0")

        trajs = (
            [int(t) for t in trajectories]
            if trajectories is not None
            else self._fidelity_train_trajectories()
        )
        if not trajs:
            return (
                None,
                f"no training trajectories found under {self._dataset_dir}",
            )

        replay_specs = []
        for t in trajs:
            try:
                tr = _dt.read_trajectory(self._dataset_dir, int(t))
            except (FileNotFoundError, ValueError):
                continue
            if tr["n_steps"] <= 0:
                continue
            replay_specs.append(
                {
                    "trajectory_index": int(t),
                    "first_path": str(tr["frames"][0]),
                    "last_path": str(tr["frames"][-1]),
                    "actions": np.asarray(tr["actions"], dtype=np.float64).tolist(),
                }
            )
        if not replay_specs:
            return (None, "no usable training trajectories with actions")

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "def _numeric_components(state):\n"
            "    items = state.items() if isinstance(state, dict) else [('__state__', state)]\n"
            "    out = {}\n"
            "    for k, v in items:\n"
            "        try:\n"
            "            arr = np.asarray(v)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if arr.dtype.kind not in 'iufb':\n"
            "            continue\n"
            "        flat = np.asarray(arr, dtype=np.float64).reshape(-1)\n"
            "        if flat.size == 0:\n"
            "            continue\n"
            "        out[str(k)] = flat\n"
            "    return out\n"
            "state_periodicity = {str(k): bool(v) for k, v in args.get('state_periodicity', {}).items()}\n"
            "def _delta(key, before, after):\n"
            "    delta = after - before\n"
            "    periodic = state_periodicity.get(key)\n"
            "    if periodic is None:\n"
            "        lower = key.lower()\n"
            "        periodic = ('theta' in lower or 'angle' in lower or lower in {'q', 'qpos', 'joint', 'joints'})\n"
            "    if periodic:\n"
            "        delta = np.arctan2(np.sin(delta), np.cos(delta))\n"
            "    return float(np.linalg.norm(delta))\n"
            "maxima = {}\n"
            "n_steps = 0\n"
            "action_dim = None\n"
            "failures = []\n"
            "for tr in args['replay_specs']:\n"
            "    try:\n"
            "        image_A = np.array(Image.open(tr['first_path']).convert('RGB'))\n"
            "        image_B = np.array(Image.open(tr['last_path']).convert('RGB'))\n"
            "        sim.fit(image_A, image_B)\n"
            "        for a in tr['actions']:\n"
            "            a_arr = np.asarray(a, dtype=np.float64)\n"
            "            if action_dim is None:\n"
            "                action_dim = int(a_arr.size)\n"
            "            before = _numeric_components(sim.state)\n"
            "            sim.update(a_arr)\n"
            "            after = _numeric_components(sim.state)\n"
            "            for key in (set(before.keys()) & set(after.keys())):\n"
            "                if before[key].shape != after[key].shape:\n"
            "                    continue\n"
            "                d = _delta(key, before[key], after[key])\n"
            "                maxima.setdefault(key, 0.0)\n"
            "                if d > float(maxima.get(key, 0.0)):\n"
            "                    maxima[key] = d\n"
            "            n_steps += 1\n"
            "    except Exception as exc:\n"
            "        failures.append({'trajectory_index': tr.get('trajectory_index'), 'error': f'{type(exc).__name__}: {exc}'})\n"
            "out = {'ok': bool(maxima) and (action_dim is not None), 'max_step_deltas': maxima, 'n_steps': n_steps, 'action_dim': int(action_dim or 0), 'failures': failures}\n"
            "print(json.dumps(out))\n"
        )

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "replay_specs": replay_specs,
                "state_periodicity": self._declared_state_periodicity(),
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=240,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return (
                None,
                f"realizability bound runner exited with code {result.returncode}: {result.stderr[:600]}",
            )
        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return (None, f"could not parse realizability-bound runner output: {exc}")

        if not payload.get("ok"):
            fail_preview = payload.get("failures", [])
            return (
                None,
                f"could not derive per-step state bounds from replay (failures={fail_preview})",
            )

        observed = {
            str(k): float(v)
            for k, v in dict(payload.get("max_step_deltas", {})).items()
            if float(v) >= 0.0
        }
        if not observed:
            return (None, "no numeric per-step state components found during replay")
        primary_component = max(observed.items(), key=lambda kv: kv[1])[0]
        bounded = {k: float(v) * margin for k, v in observed.items()}
        return (
            {
                "trajectories": trajs,
                "margin": margin,
                "n_replay_steps": int(payload.get("n_steps", 0)),
                "action_dim": int(payload.get("action_dim", 0)),
                "observed_max": observed,
                "bound_max": bounded,
                "primary_component": primary_component,
                "failures": payload.get("failures", []),
            },
            None,
        )

    def _run_goal_invariance_check(
        self,
        *,
        trajectories: list[int] | None = None,
        abs_tolerance: float,
        rel_tolerance: float,
    ) -> dict:
        """Check that fit(obs_i, goal).target_state is invariant within episode."""
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        try:
            abs_tolerance = float(abs_tolerance)
            rel_tolerance = float(rel_tolerance)
        except (TypeError, ValueError):
            return {"passed": False, "error": "goal-invariance tolerances must be floats"}
        if abs_tolerance < 0 or rel_tolerance < 0:
            return {"passed": False, "error": "goal-invariance tolerances must be >= 0"}

        trajs = (
            [int(t) for t in trajectories]
            if trajectories is not None
            else self._fidelity_train_trajectories()
        )
        if not trajs:
            return {"passed": False, "error": "no training trajectories available"}

        checks: list[dict[str, Any]] = []
        for t in trajs:
            try:
                tr = _dt.read_trajectory(self._dataset_dir, int(t))
            except (FileNotFoundError, ValueError):
                continue
            frames = [str(p) for p in tr["frames"]]
            if len(frames) < 2:
                continue
            sample_idxs = sorted(set([0, len(frames) // 2, len(frames) - 1]))
            checks.append(
                {
                    "trajectory_index": int(t),
                    "goal_path": frames[-1],
                    "obs_paths": [frames[i] for i in sample_idxs],
                    "obs_indices": [int(i) for i in sample_idxs],
                    "goal_frame_index": len(frames) - 1,
                }
            )

        if not checks:
            return {
                "passed": False,
                "error": "no trajectories had enough frames for goal-invariance check",
            }

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "def _numeric_components(state):\n"
            "    items = state.items() if isinstance(state, dict) else [('__state__', state)]\n"
            "    out = {}\n"
            "    for k, v in items:\n"
            "        try:\n"
            "            arr = np.asarray(v)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if arr.dtype.kind not in 'iufb':\n"
            "            continue\n"
            "        flat = np.asarray(arr, dtype=np.float64).reshape(-1)\n"
            "        if flat.size == 0 or flat.size > int(args['max_target_elements']):\n"
            "            continue\n"
            "        out[str(k)] = flat.tolist()\n"
            "    return out\n"
            "results = []\n"
            "for check in args['checks']:\n"
            "    out = {'trajectory_index': int(check['trajectory_index']), 'samples': []}\n"
            "    try:\n"
            "        goal = np.array(Image.open(check['goal_path']).convert('RGB'))\n"
            "        for obs_idx, obs_path in zip(check['obs_indices'], check['obs_paths']):\n"
            "            obs = np.array(Image.open(obs_path).convert('RGB'))\n"
            "            sim.fit(obs, goal)\n"
            "            out['samples'].append({'obs_index': int(obs_idx), 'target_components': _numeric_components(sim.target_state)})\n"
            "    except Exception as exc:\n"
            "        out['error'] = f'{type(exc).__name__}: {exc}'\n"
            "    results.append(out)\n"
            "print(json.dumps({'results': results}))\n"
        )

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "checks": checks,
                "max_target_elements": 64,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            try:
                result = _subprocess.run(
                    [_sys.executable, runner_path, _json.dumps(runner_args)],
                    capture_output=True,
                    text=True,
                    timeout=240,
                    env=self._subprocess_env(),
                )
            except _subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "error": "goal-invariance check timed out after 240s",
                }
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return {
                "passed": False,
                "error": (
                    "goal-invariance runner failed with code "
                    f"{result.returncode}: {result.stderr[:600]}"
                ),
            }

        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {"passed": False, "error": f"could not parse goal-invariance output: {exc}"}

        failures: list[dict[str, Any]] = []
        violations: list[dict[str, Any]] = []
        per_traj_max_drift: dict[int, float] = {}

        for row in payload.get("results", []):
            t = int(row.get("trajectory_index", -1))
            if row.get("error"):
                failures.append({"trajectory_index": t, "error": row["error"]})
                continue
            samples = list(row.get("samples", []))
            if len(samples) < 2:
                failures.append(
                    {
                        "trajectory_index": t,
                        "error": "fewer than 2 samples available for comparison",
                    }
                )
                continue
            ref_components = {
                str(k): np.asarray(v, dtype=np.float64).reshape(-1)
                for k, v in dict(samples[0].get("target_components", {})).items()
            }
            if not ref_components:
                failures.append(
                    {
                        "trajectory_index": t,
                        "error": "no numeric target_state components <=64 elements",
                    }
                )
                continue

            traj_max = 0.0
            comparable_found = False
            for sample in samples[1:]:
                cur_components = {
                    str(k): np.asarray(v, dtype=np.float64).reshape(-1)
                    for k, v in dict(sample.get("target_components", {})).items()
                }
                missing_keys = sorted(set(ref_components) - set(cur_components))
                extra_keys = sorted(set(cur_components) - set(ref_components))
                for key in missing_keys + extra_keys:
                    violations.append(
                        {
                            "trajectory_index": t,
                            "baseline_obs_index": int(samples[0]["obs_index"]),
                            "obs_index": int(sample["obs_index"]),
                            "component": key,
                            "drift": float(abs_tolerance + rel_tolerance + 1.0),
                            "tolerance": float(abs_tolerance + rel_tolerance),
                            "reference_norm": 0.0,
                            "reason": (
                                "target component disappeared"
                                if key in missing_keys
                                else "observation-dependent target component appeared"
                            ),
                        }
                    )
                common = sorted(set(ref_components.keys()) & set(cur_components.keys()))
                if not common:
                    continue
                for key in common:
                    ref = ref_components[key]
                    cur = cur_components[key]
                    if ref.shape != cur.shape:
                        continue
                    comparable_found = True
                    drift = self._component_delta_norm(key, ref, cur)
                    ref_norm = float(np.linalg.norm(ref))
                    tol = abs_tolerance + rel_tolerance * max(ref_norm, 1.0)
                    if drift > traj_max:
                        traj_max = drift
                    if drift > tol + 1e-9:
                        violations.append(
                            {
                                "trajectory_index": t,
                                "baseline_obs_index": int(samples[0]["obs_index"]),
                                "obs_index": int(sample["obs_index"]),
                                "component": key,
                                "drift": float(drift),
                                "tolerance": float(tol),
                                "reference_norm": ref_norm,
                            }
                        )
            if not comparable_found:
                failures.append(
                    {
                        "trajectory_index": t,
                        "error": "no shared comparable target_state components across sampled observations",
                    }
                )
                continue
            per_traj_max_drift[t] = float(traj_max)

        worst_violation = None
        if violations:
            worst_violation = max(violations, key=lambda x: float(x["drift"] / max(x["tolerance"], 1e-9)))

        passed = (not failures) and (not violations)
        return {
            "passed": bool(passed),
            "abs_tolerance": float(abs_tolerance),
            "rel_tolerance": float(rel_tolerance),
            "checked_trajectories": [int(c["trajectory_index"]) for c in checks],
            "per_traj_max_drift": per_traj_max_drift,
            "failures": failures,
            "violations": violations,
            "worst_violation": worst_violation,
        }

    def _run_target_independence_check(
        self,
        *,
        trajectories: list[int] | None = None,
        abs_tolerance: float | None = None,
        rel_tolerance: float | None = None,
        null_steps: int | None = None,
        null_cost_drop_frac: float | None = None,
        max_replay_steps: int = 20,
    ) -> dict:
        """Anti-cheat probe: dynamics must not depend on the goal, and cost
        must not fall under a declared neutral action.

        Part A (target independence): fit the same start frame against two
        different goal images, replay the identical expert-action prefix, and
        require (near-)identical `self.state` trajectories. Catches `update`
        implementations that steer state toward `target_state`.

        Part B (neutral-action drift): when zero lies inside a signed deployed
        action space, apply zero actions and require that `terminal_cost` does
        not drop by more than ``null_cost_drop_frac`` of its initial value.
        Non-negative image-coordinate/setpoint spaces have no universal fixed
        neutral action, so this subtest is skipped rather than teaching the
        simulator that a real corner target is a no-op.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        abs_tolerance = (
            self._gate_target_independence_abs_tol
            if abs_tolerance is None
            else float(abs_tolerance)
        )
        rel_tolerance = (
            self._gate_target_independence_rel_tol
            if rel_tolerance is None
            else float(rel_tolerance)
        )
        null_steps = (
            self._gate_null_action_steps if null_steps is None else int(null_steps)
        )
        null_cost_drop_frac = (
            self._gate_null_action_cost_drop_frac
            if null_cost_drop_frac is None
            else float(null_cost_drop_frac)
        )
        neutral_action = None
        neutral_action_source = "not_defined_for_action_contract"
        action_bounds = self._deployed_action_bounds
        if action_bounds is None:
            action_bounds = self._training_action_bounds(pad_fraction=0.0)
            neutral_action_source = "not_defined_for_training_bounds"
        if action_bounds is not None:
            low, high = action_bounds
            if (
                low.shape == high.shape
                and low.size > 0
                and np.all(low < 0.0)
                and np.all(high > 0.0)
            ):
                neutral_action = np.zeros(low.size, dtype=np.float64).tolist()
                neutral_action_source = (
                    "zero_in_signed_deployed_bounds"
                    if self._deployed_action_bounds is not None
                    else "zero_in_signed_training_bounds"
                )

        trajs = (
            [int(t) for t in trajectories]
            if trajectories is not None
            else self._fidelity_train_trajectories()
        )
        if not trajs:
            return {"passed": False, "error": "no training trajectories available"}

        loaded: dict[int, dict] = {}
        for t in trajs:
            try:
                tr = _dt.read_trajectory(self._dataset_dir, int(t))
            except (FileNotFoundError, ValueError):
                continue
            if len(tr["frames"]) >= 2 and tr["actions"].shape[0] >= 1:
                loaded[int(t)] = tr
        usable = sorted(loaded)
        if not usable:
            return {
                "passed": False,
                "error": "no trajectories with frames and actions for target-independence check",
            }

        checks: list[dict[str, Any]] = []
        for i, t in enumerate(usable[:3]):
            tr = loaded[t]
            frames = [str(p) for p in tr["frames"]]
            n_actions = min(int(max_replay_steps), int(tr["actions"].shape[0]))
            # Alternative goal: the final frame of a different trajectory when
            # available; otherwise the trajectory's own start frame.
            alt = usable[(i + 1) % len(usable)]
            if alt != t:
                goal_b = str(loaded[alt]["frames"][-1])
            else:
                goal_b = frames[0]
            checks.append(
                {
                    "trajectory_index": int(t),
                    "start_path": frames[0],
                    "goal_a_path": frames[-1],
                    "goal_b_path": goal_b,
                    "actions": tr["actions"][:n_actions].tolist(),
                }
            )

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "def _numeric_components(state):\n"
            "    items = state.items() if isinstance(state, dict) else [('__state__', state)]\n"
            "    out = {}\n"
            "    for k, v in items:\n"
            "        try:\n"
            "            arr = np.asarray(v)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if arr.dtype.kind not in 'iufb':\n"
            "            continue\n"
            "        flat = np.asarray(arr, dtype=np.float64).reshape(-1)\n"
            "        if flat.size == 0 or flat.size > int(args['max_target_elements']):\n"
            "            continue\n"
            "        out[str(k)] = flat.tolist()\n"
            "    return out\n"
            "def _rollout(start, goal, actions):\n"
            "    sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "    sim.fit(start, goal)\n"
            "    states = [_numeric_components(sim.state)]\n"
            "    for a in actions:\n"
            "        sim.update(np.asarray(a, dtype=np.float64))\n"
            "        states.append(_numeric_components(sim.state))\n"
            "    return states\n"
            "results = []\n"
            "for check in args['checks']:\n"
            "    out = {'trajectory_index': int(check['trajectory_index'])}\n"
            "    try:\n"
            "        start = np.array(Image.open(check['start_path']).convert('RGB'))\n"
            "        goal_a = np.array(Image.open(check['goal_a_path']).convert('RGB'))\n"
            "        goal_b = np.array(Image.open(check['goal_b_path']).convert('RGB'))\n"
            "        actions = check['actions']\n"
            "        out['states_goal_a'] = _rollout(start, goal_a, actions)\n"
            "        out['states_goal_b'] = _rollout(start, goal_b, actions)\n"
            "        sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "        sim.fit(start, goal_a)\n"
            "        neutral = args.get('neutral_action')\n"
            "        if neutral is not None:\n"
            "            neutral = np.asarray(neutral, dtype=np.float64)\n"
            "            costs = [float(sim.planning_objective())]\n"
            "            for _ in range(int(args['null_steps'])):\n"
            "                sim.update(neutral)\n"
            "                costs.append(float(sim.planning_objective()))\n"
            "            out['null_cost_initial'] = costs[0]\n"
            "            out['null_cost_final'] = costs[-1]\n"
            "            out['null_cost_min'] = min(costs)\n"
            "        else:\n"
            "            out['null_action_skipped'] = True\n"
            "    except Exception as exc:\n"
            "        out['error'] = f'{type(exc).__name__}: {exc}'\n"
            "    results.append(out)\n"
            "print(json.dumps({'results': results}))\n"
        )

        with _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "checks": checks,
                "null_steps": int(null_steps),
                "neutral_action": neutral_action,
                "max_target_elements": 64,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            try:
                result = _subprocess.run(
                    [_sys.executable, runner_path, _json.dumps(runner_args)],
                    capture_output=True,
                    text=True,
                    timeout=240,
                    env=self._subprocess_env(),
                )
            except _subprocess.TimeoutExpired:
                return {
                    "passed": False,
                    "error": "target-independence check timed out after 240s",
                }
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return {
                "passed": False,
                "error": (
                    "target-independence runner failed with code "
                    f"{result.returncode}: {result.stderr[:600]}"
                ),
            }

        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {
                "passed": False,
                "error": f"could not parse target-independence output: {exc}",
            }

        failures: list[dict[str, Any]] = []
        leak_violations: list[dict[str, Any]] = []
        null_violations: list[dict[str, Any]] = []

        for row in payload.get("results", []):
            t = int(row.get("trajectory_index", -1))
            if row.get("error"):
                failures.append({"trajectory_index": t, "error": row["error"]})
                continue

            states_a = list(row.get("states_goal_a", []))
            states_b = list(row.get("states_goal_b", []))
            for step in range(min(len(states_a), len(states_b))):
                comp_a = {
                    str(k): np.asarray(v, dtype=np.float64).reshape(-1)
                    for k, v in dict(states_a[step]).items()
                }
                comp_b = {
                    str(k): np.asarray(v, dtype=np.float64).reshape(-1)
                    for k, v in dict(states_b[step]).items()
                }
                for key in sorted(set(comp_a) & set(comp_b)):
                    ref, cur = comp_a[key], comp_b[key]
                    if ref.shape != cur.shape:
                        continue
                    drift = self._component_delta_norm(key, ref, cur)
                    tol = abs_tolerance + rel_tolerance * max(
                        float(np.linalg.norm(ref)), 1.0
                    )
                    if drift > tol + 1e-9:
                        leak_violations.append(
                            {
                                "trajectory_index": t,
                                "step": int(step),
                                "component": key,
                                "drift": float(drift),
                                "tolerance": float(tol),
                            }
                        )

            c0 = row.get("null_cost_initial")
            c_final = row.get("null_cost_final")
            if c0 is not None and c_final is not None and np.isfinite([c0, c_final]).all():
                drop = float(c0) - float(c_final)
                allowed = max(null_cost_drop_frac * abs(float(c0)), 1e-6)
                if drop > allowed:
                    null_violations.append(
                        {
                            "trajectory_index": t,
                            "cost_initial": float(c0),
                            "cost_final": float(c_final),
                            "drop": drop,
                            "allowed_drop": allowed,
                        }
                    )

        worst_leak = None
        if leak_violations:
            worst_leak = max(
                leak_violations,
                key=lambda x: float(x["drift"] / max(x["tolerance"], 1e-9)),
            )
        worst_null = None
        if null_violations:
            worst_null = max(null_violations, key=lambda x: float(x["drop"]))

        passed = not failures and not leak_violations and not null_violations
        return {
            "passed": bool(passed),
            "abs_tolerance": float(abs_tolerance),
            "rel_tolerance": float(rel_tolerance),
            "null_steps": int(null_steps),
            "null_cost_drop_frac": float(null_cost_drop_frac),
            "neutral_action": neutral_action,
            "neutral_action_source": neutral_action_source,
            "null_action_skipped": neutral_action is None,
            "checked_trajectories": [int(c["trajectory_index"]) for c in checks],
            "failures": failures,
            "leak_violations": leak_violations,
            "null_violations": null_violations,
            "worst_leak": worst_leak,
            "worst_null": worst_null,
        }

    def validate_target_independence(
        self,
        trajectory_indices: list[int] | None = None,
    ) -> str:
        """Verify dynamics ignore the goal and cost cannot fall for free.

        Two probes on training data: (1) fit the same start frame against two
        different goal images and replay identical expert actions — the state
        trajectories must match, otherwise `update` is steering toward
        `target_state`; (2) when the deployed action contract has a fixed zero
        neutral action, roll it out and require that `terminal_cost` does not
        drop materially. Both applicable checks are enforced by the final gate.

        Args:
            trajectory_indices: Optional training trajectory indices to check.
                Defaults to the configured gate training set.
        """
        import json as _json

        verdict = self._run_target_independence_check(
            trajectories=trajectory_indices,
        )
        args = {"trajectory_indices": str(trajectory_indices)}

        if verdict.get("error"):
            summary = f"validate_target_independence: FAIL ({verdict['error']})"
        elif verdict["passed"]:
            neutral_note = (
                f"and terminal_cost did not drop under {verdict['null_steps']} "
                "neutral-action steps."
                if not verdict.get("null_action_skipped")
                else "the neutral-action subtest was not applicable to this "
                "setpoint-style action contract."
            )
            summary = (
                "validate_target_independence: PASS — identical action replays "
                "under two different goals produced matching state trajectories "
                f"(tol abs={verdict['abs_tolerance']}, rel={verdict['rel_tolerance']}), "
                f"{neutral_note}"
            )
        else:
            lines = ["validate_target_independence: FAIL"]
            if verdict.get("leak_violations"):
                w = verdict.get("worst_leak") or {}
                lines.append(
                    f"  TARGET LEAK: state component {w.get('component')!r} diverged by "
                    f"{float(w.get('drift', 0.0)):.4f} (tol {float(w.get('tolerance', 0.0)):.4f}) "
                    f"at step {w.get('step')} of trajectory {w.get('trajectory_index')} when only "
                    "the goal image changed. update(a) must not read self.target_state "
                    "or any goal-derived quantity to move state."
                )
            if verdict.get("null_violations"):
                w = verdict.get("worst_null") or {}
                lines.append(
                    f"  NULL-ACTION DRIFT: terminal_cost fell from {float(w.get('cost_initial', 0.0)):.4f} "
                    f"to {float(w.get('cost_final', 0.0)):.4f} under all-zero actions "
                    f"(allowed drop {float(w.get('allowed_drop', 0.0)):.4f}). Cost must reflect "
                    "physical progress, not elapsed steps."
                )
            if verdict.get("failures"):
                first = verdict["failures"][0]
                lines.append(
                    f"  ERROR on trajectory {first.get('trajectory_index')}: {first.get('error')}"
                )
            summary = "\n".join(lines)

        self._log_critic_tool_call(
            tool_name="validate_target_independence",
            args=args,
            result_summary=summary[:1000],
            extra_files={"verdict.json": _json.dumps(verdict, indent=2, default=str)},
        )
        return summary

    def _run_action_realizability_check(
        self,
        *,
        margin: float | None = None,
        fit_trajectory_index: int | None = None,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ) -> dict:
        """Run Step-2 over-reachability probe and return a structured verdict."""
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        if margin is None:
            margin = self._gate_realizability_margin
        bounds_data, bounds_err = self._realizability_limits_cached(
            margin=float(margin),
        )
        if bounds_err or bounds_data is None:
            return {"passed": False, "error": bounds_err or "unknown error"}

        action_dim = int(bounds_data["action_dim"])
        low, high, source = self._resolve_action_bounds(
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )
        if low is None or high is None:
            return {
                "passed": False,
                "error": f"could not resolve probe action bounds ({source})",
            }

        if fit_trajectory_index is None:
            fit_trajectory_index = int(bounds_data["trajectories"][0])
        try:
            fit_traj = _dt.read_trajectory(
                self._dataset_dir, int(fit_trajectory_index)
            )
        except (FileNotFoundError, ValueError) as exc:
            return {"passed": False, "error": f"fit trajectory error: {exc}"}

        low_arr = np.asarray(low, dtype=np.float64)
        high_arr = np.asarray(high, dtype=np.float64)
        if source == "training":
            stacked = self._collect_training_actions()
            probes = []
            if stacked is not None and stacked.ndim == 2 and stacked.shape[1] == action_dim:
                probes = self._build_training_probe_actions(stacked)
            if not probes:
                probes = self._build_probe_actions(low_arr, high_arr)
        else:
            probes = self._build_probe_actions(low_arr, high_arr)

        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "image_A = np.array(Image.open(args['fit_first']).convert('RGB'))\n"
            "image_B = np.array(Image.open(args['fit_last']).convert('RGB'))\n"
            "def _numeric_components(state):\n"
            "    items = state.items() if isinstance(state, dict) else [('__state__', state)]\n"
            "    out = {}\n"
            "    for k, v in items:\n"
            "        try:\n"
            "            arr = np.asarray(v)\n"
            "        except Exception:\n"
            "            continue\n"
            "        if arr.dtype.kind not in 'iufb':\n"
            "            continue\n"
            "        flat = np.asarray(arr, dtype=np.float64).reshape(-1)\n"
            "        if flat.size == 0:\n"
            "            continue\n"
            "        out[str(k)] = flat\n"
            "    return out\n"
            "state_periodicity = {str(k): bool(v) for k, v in args.get('state_periodicity', {}).items()}\n"
            "def _delta(key, before, after):\n"
            "    delta = after - before\n"
            "    periodic = state_periodicity.get(key)\n"
            "    if periodic is None:\n"
            "        lower = key.lower()\n"
            "        periodic = ('theta' in lower or 'angle' in lower or lower in {'q', 'qpos', 'joint', 'joints'})\n"
            "    if periodic:\n"
            "        delta = np.arctan2(np.sin(delta), np.cos(delta))\n"
            "    return float(np.linalg.norm(delta))\n"
            "results = []\n"
            "for a in args['probe_actions']:\n"
            "    try:\n"
            "        sim.fit(image_A, image_B)\n"
            "        before = _numeric_components(sim.state)\n"
            "        sim.update(np.asarray(a, dtype=np.float64))\n"
            "        after = _numeric_components(sim.state)\n"
            "        deltas = {}\n"
            "        for key in (set(before.keys()) & set(after.keys())):\n"
            "            if before[key].shape != after[key].shape:\n"
            "                continue\n"
            "            deltas[key] = _delta(key, before[key], after[key])\n"
            "        results.append({'action': list(map(float, a)), 'deltas': deltas})\n"
            "    except Exception as exc:\n"
            "        results.append({'action': list(map(float, a)), 'error': f'{type(exc).__name__}: {exc}'})\n"
            "print(json.dumps({'results': results}))\n"
        )

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "fit_first": str(fit_traj["frames"][0]),
                "fit_last": str(fit_traj["frames"][-1]),
                "probe_actions": probes,
                "state_periodicity": self._declared_state_periodicity(),
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=180,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "OPENCV_FOR_THREADS_NUM": "1",
                },
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return {
                "passed": False,
                "error": (
                    f"realizability probe runner exited with code {result.returncode}: "
                    f"{result.stderr[:600]}"
                ),
            }

        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return {"passed": False, "error": f"could not parse probe output: {exc}"}

        train_bound = dict(bounds_data["bound_max"])
        primary_component = str(bounds_data["primary_component"])
        checked_components = sorted(train_bound)
        violations: list[dict[str, Any]] = []
        probe_errors: list[str] = []
        missing_components: dict[str, list[int]] = {}
        for i, row in enumerate(payload.get("results", [])):
            if row.get("error"):
                probe_errors.append(f"probe#{i}: {row['error']}")
                continue
            deltas = dict(row.get("deltas", {}))
            for component in checked_components:
                if component not in deltas:
                    missing_components.setdefault(component, []).append(i)
                    continue
                delta = float(deltas[component])
                bound = float(train_bound[component])
                if delta > bound + 1e-9:
                    ratio = (delta / bound) if bound > 0 else np.inf
                    violations.append(
                        {
                            "probe_index": i,
                            "action": row.get("action"),
                            "component": component,
                            "delta": delta,
                            "bound": bound,
                            "ratio": float(ratio),
                        }
                    )
        for component, probe_indices in sorted(missing_components.items()):
            probe_errors.append(
                f"component '{component}' missing from probes {probe_indices}"
            )

        worst_violation = None
        if violations:
            worst_violation = max(violations, key=lambda x: float(x["ratio"]))

        passed = (not violations) and (not probe_errors)
        return {
            "passed": bool(passed),
            "fit_trajectory_index": int(fit_trajectory_index),
            "action_bounds_source": source,
            "action_low": low,
            "action_high": high,
            "probe_actions": probes,
            "train_observed_max": bounds_data["observed_max"],
            "train_bound_max": train_bound,
            "primary_component": primary_component,
            "checked_components": checked_components,
            "bound_margin": float(bounds_data["margin"]),
            "training_failures": bounds_data["failures"],
            "probe_results": payload.get("results", []),
            "probe_errors": probe_errors,
            "violations": violations,
            "worst_violation": worst_violation,
            "missing_component_probes": missing_components,
            "missing_primary_component_probes": missing_components.get(
                primary_component,
                [],
            ),
        }

    def validate_goal_invariance(
        self,
        trajectory_indices: list[int] | None = None,
        abs_tolerance: float | None = None,
        rel_tolerance: float | None = None,
    ) -> str:
        """Check that target_state stays fixed when only the observation changes.

        Re-fits the simulator as ``fit(obs_i, fixed_goal)`` across early,
        middle, and late frames from one or more training trajectories. The
        current movable state may change with ``obs_i``, but the goal/target
        decoded into ``target_state`` must stay invariant because the second
        image is the fixed goal frame.

        Args:
            trajectory_indices: Optional training trajectory indices to check.
                Defaults to the configured gate training set.
            abs_tolerance: Absolute drift tolerance for numeric target_state
                components. Defaults to the gate tolerance.
            rel_tolerance: Relative drift tolerance, scaled by target norm.
                Defaults to the gate tolerance.
        """
        import json as _json

        if abs_tolerance is None:
            abs_tolerance = self._gate_goal_invariance_abs_tol
        if rel_tolerance is None:
            rel_tolerance = self._gate_goal_invariance_rel_tol

        verdict = self._run_goal_invariance_check(
            trajectories=trajectory_indices,
            abs_tolerance=float(abs_tolerance),
            rel_tolerance=float(rel_tolerance),
        )
        args = {
            "trajectory_indices": str(trajectory_indices),
            "abs_tolerance": str(abs_tolerance),
            "rel_tolerance": str(rel_tolerance),
        }

        if verdict.get("error"):
            summary = f"validate_goal_invariance: FAIL ({verdict['error']})"
            self._log_critic_tool_call(
                tool_name="validate_goal_invariance",
                args=args,
                result_summary=summary,
                extra_files={"goal_invariance.json": _json.dumps(verdict, indent=2)},
            )
            return summary

        drifts = dict(verdict.get("per_traj_max_drift", {}))
        drift_line = (
            ", ".join(f"t{int(t)}={float(d):.3f}" for t, d in sorted(drifts.items()))
            or "(none)"
        )
        summary = (
            "validate_goal_invariance:\n"
            f"  checked trajectories: {verdict.get('checked_trajectories', [])}\n"
            f"  tolerances: abs={float(verdict['abs_tolerance']):.3f}, "
            f"rel={float(verdict['rel_tolerance']):.3f}\n"
            f"  max target_state drift per trajectory: {drift_line}\n"
        )

        failures = list(verdict.get("failures", []))
        if failures:
            fail_line = "; ".join(
                f"t{f.get('trajectory_index')} ({f.get('error')})" for f in failures
            )
            summary += f"  check failures: {fail_line}\n"

        if verdict["passed"]:
            summary += (
                "  PASS: target_state stayed invariant when refitting with "
                "different current observation frames and the same fixed goal frame."
            )
        elif verdict.get("worst_violation"):
            worst = verdict.get("worst_violation") or {}
            summary += (
                "  FAIL: target_state drifts when only image_A/current observation "
                "changes. Read the goal/target from image_B (the fixed goal frame), "
                "not solely from image_A. "
                f"Trajectory={worst.get('trajectory_index')}, "
                f"baseline_obs={worst.get('baseline_obs_index')}, "
                f"obs={worst.get('obs_index')}, "
                f"component={worst.get('component')}, "
                f"drift={float(worst.get('drift', 0.0)):.3f}, "
                f"tol={float(worst.get('tolerance', 0.0)):.3f}."
            )
        else:
            summary += (
                "  FAIL: goal-invariance check could not compare a stable numeric "
                "target_state across observations. Ensure fit(image_A, image_B) sets "
                "a numeric target_state from the fixed goal frame image_B."
            )

        self._log_critic_tool_call(
            tool_name="validate_goal_invariance",
            args=args,
            result_summary=summary[:500],
            extra_files={"goal_invariance.json": _json.dumps(verdict, indent=2)},
        )
        return summary

    def validate_action_realizability(
        self,
        fit_trajectory_index: int | None = None,
        margin: float | None = None,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ) -> str:
        """Check whether one-step state changes stay within training-derived limits.

        Derives per-component max ||delta state|| from replaying the TRAINING
        trajectories (offline, no env), inflates each max by ``margin``, then
        probes the simulator with extreme one-step actions under the planner's
        action bounds. Fails if any state component moves farther in one update
        than training ever demonstrated.
        """
        import json as _json

        verdict = self._run_action_realizability_check(
            margin=margin,
            fit_trajectory_index=fit_trajectory_index,
            action_low=action_low,
            action_high=action_high,
        )
        if verdict.get("error"):
            summary = f"validate_action_realizability: FAIL ({verdict['error']})"
            self._log_critic_tool_call(
                tool_name="validate_action_realizability",
                args={
                    "fit_trajectory_index": str(fit_trajectory_index),
                    "margin": str(margin),
                },
                result_summary=summary,
                extra_files={"realizability.json": _json.dumps(verdict, indent=2)},
            )
            return summary

        train_bounds = verdict["train_bound_max"]
        bounds_line = ", ".join(
            f"{k}<= {float(v):.3f}" for k, v in sorted(train_bounds.items())
        ) or "(none)"
        summary = (
            "validate_action_realizability:\n"
            f"  training-derived one-step bounds (margin x{verdict['bound_margin']:.3f}): {bounds_line}\n"
            f"  checked state components: {verdict['checked_components']}\n"
            f"  probe action bounds source: {verdict['action_bounds_source']} "
            f"(low={verdict['action_low']}, high={verdict['action_high']})\n"
        )
        if verdict["probe_errors"]:
            summary += "  probe errors: " + "; ".join(verdict["probe_errors"]) + "\n"

        if verdict["passed"]:
            summary += (
                "  PASS: all probed one-step state-component displacements stayed "
                "within training-derived realizability bounds."
            )
        else:
            worst = verdict.get("worst_violation") or {}
            summary += (
                "  FAIL: unrealizable one-step jump detected. "
                f"Component '{worst.get('component')}' moved "
                f"{float(worst.get('delta', 0.0)):.3f} in one update under probe "
                f"action {worst.get('action')}, while the training-derived bound is "
                f"{float(worst.get('bound', 0.0)):.3f}. "
                "The planner can exploit this mismatch."
            )

        self._log_critic_tool_call(
            tool_name="validate_action_realizability",
            args={
                "fit_trajectory_index": str(fit_trajectory_index),
                "margin": str(margin),
                "action_low": str(action_low),
                "action_high": str(action_high),
            },
            result_summary=summary[:500],
            extra_files={"realizability.json": _json.dumps(verdict, indent=2)},
        )
        return summary

    def calibrate_transition_parameters(
        self,
        bounds: dict,
        trajectory_indices: list[int] | None = None,
        state_keys: list[str] | None = None,
        max_transitions: int = 36,
        budget: int = 72,
        max_state_dim: int = 16,
    ) -> str:
        """Calibrate ``self.params`` from teacher-forced one-step transitions.

        For every sampled transition this tool fits the current TRUE frame
        against that trajectory's fixed final-frame goal, applies exactly one
        recorded action, and compares the predicted observable state with a
        fresh fit of the next TRUE frame. Errors are normalized per component;
        angle-like keys use wrapped differences. Boolean masks, image-shaped or
        oversized arrays, latent/rate keys, static components, and components
        not consistently observable in both frames are excluded and reported.

        Training and held-out transitions are disjoint. Only the parameter keys
        listed in ``bounds`` are searched, with a bounded deterministic
        differential-evolution budget. This tool never edits simulator source;
        copy accepted values into ``self.params`` yourself and revalidate.

        Args:
            bounds: Selected ``self.params`` names mapped to narrow
                ``[low, high]`` ranges.
            trajectory_indices: Optional training trajectories to sample.
            state_keys: Optional observable state keys to score. By default all
                safe, dynamic, consistently observable keys are used.
            max_transitions: Total train plus held-out transitions (6..48).
            budget: Approximate optimizer evaluations (12..180).
            max_state_dim: Maximum scalar count in one state component (<=32).
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        tool_name = "calibrate_transition_parameters"

        def _fail(message: str, args: dict | None = None) -> str:
            text = f"[{tool_name}] {message}"
            self._log_critic_tool_call(
                tool_name=tool_name,
                args=args or {},
                result_summary=text[:500],
            )
            return text

        if not self.has_valid_simulator_class():
            return _fail("write a runnable simulator with a credible fit() first.")
        if not isinstance(bounds, dict) or not bounds:
            return _fail(
                "`bounds` must be a non-empty dict of self.params names to "
                "narrow [low, high] ranges."
            )
        if len(bounds) > 6:
            return _fail("at most 6 parameters may be calibrated in one bounded call.")

        param_names: list[str] = []
        bounds_list: list[list[float]] = []
        for raw_name, raw_range in bounds.items():
            name = str(raw_name)
            if not name or len(name) > 80:
                return _fail(f"invalid parameter name {name!r}.")
            try:
                lo = float(raw_range[0])
                hi = float(raw_range[1])
            except (TypeError, ValueError, IndexError):
                return _fail(
                    f"bound for {name!r} must be [low, high]; got {raw_range!r}."
                )
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return _fail(
                    f"bound for {name!r} must be finite with high > low; "
                    f"got [{lo}, {hi}]."
                )
            if max(abs(lo), abs(hi), hi - lo) > 1e6:
                return _fail(
                    f"bound for {name!r} is not a narrow finite calibration range."
                )
            param_names.append(name)
            bounds_list.append([lo, hi])

        try:
            max_transitions = max(6, min(int(max_transitions), 48))
            budget = max(12, min(int(budget), 180))
            max_state_dim = max(1, min(int(max_state_dim), 32))
        except (TypeError, ValueError):
            return _fail("max_transitions, budget, and max_state_dim must be integers.")

        requested_keys = None
        if state_keys is not None:
            requested_keys = [str(key)[:80] for key in state_keys]
            requested_keys = list(dict.fromkeys(requested_keys))[:16]
            if not requested_keys:
                return _fail("state_keys selected no usable component names.")

        available = sorted(int(i) for i in self._fidelity_train_trajectories())
        if trajectory_indices is not None:
            selected = list(
                dict.fromkeys(int(i) for i in trajectory_indices if int(i) in available)
            )
            if not selected:
                return _fail(
                    f"none of trajectory_indices {trajectory_indices} exist; "
                    f"available: {available}."
                )
        else:
            selected = self._measurement_trajectories(
                available, max_trajectories=min(3, len(available))
            )
        if not selected:
            return _fail("no training trajectories were found.")

        per_trajectory: list[list[dict[str, Any]]] = []
        action_dim: int | None = None
        for trajectory_index in selected:
            try:
                trajectory = _dt.read_trajectory(
                    self._dataset_dir, int(trajectory_index)
                )
            except (FileNotFoundError, ValueError) as exc:
                return _fail(str(exc))
            actions = np.asarray(trajectory["actions"], dtype=np.float64)
            if actions.ndim != 2 or actions.shape[0] == 0:
                continue
            if actions.shape[1] > 32:
                return _fail(
                    f"trajectory {trajectory_index} action dimension "
                    f"{actions.shape[1]} exceeds the bounded limit 32."
                )
            if action_dim is None:
                action_dim = int(actions.shape[1])
            elif action_dim != int(actions.shape[1]):
                return _fail("selected trajectories have inconsistent action dimensions.")
            n_steps = min(int(actions.shape[0]), len(trajectory["frames"]) - 1)
            if n_steps <= 0:
                continue
            count = min(
                n_steps,
                max(2, int(np.ceil(max_transitions / max(1, len(selected))))),
            )
            starts = sorted(
                {
                    int(round(value))
                    for value in np.linspace(0, n_steps - 1, count)
                }
            )
            per_trajectory.append(
                [
                    {
                        "trajectory": int(trajectory_index),
                        "step": int(step),
                        "current": str(trajectory["frames"][step]),
                        "next": str(trajectory["frames"][step + 1]),
                        "goal": str(trajectory["frames"][-1]),
                        "action": actions[step].tolist(),
                    }
                    for step in starts
                ]
            )

        # Round-robin keeps every selected trajectory represented when the hard
        # transition cap truncates the combined sample.
        transition_specs: list[dict[str, Any]] = []
        depth = 0
        while len(transition_specs) < max_transitions:
            added = False
            for group in per_trajectory:
                if depth < len(group) and len(transition_specs) < max_transitions:
                    transition_specs.append(group[depth])
                    added = True
            if not added:
                break
            depth += 1
        if len(transition_specs) < 2:
            return _fail("at least two consecutive-frame transitions are required.")

        holdout_positions = {
            index
            for index in range(len(transition_specs))
            if index % 4 == 3
        }
        if not holdout_positions:
            holdout_positions = {len(transition_specs) - 1}
        for index, spec in enumerate(transition_specs):
            spec["split"] = "held_out" if index in holdout_positions else "train"

        runner_src = r"""
import importlib.util
import json
import re
import sys
from functools import lru_cache

import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution
from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI
from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase

args = json.loads(sys.argv[1])
spec = importlib.util.spec_from_file_location("simulator_sandbox", args["sim_path"])
module = importlib.util.module_from_spec(spec)
module.SimulatorBase = SimulatorBase
module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase
module.WorldAPI = GeometryOnlyWorldAPI
sys.modules["simulator_sandbox"] = module
spec.loader.exec_module(module)
cls = getattr(module, args["class_name"])
names = list(args["param_names"])
requested_keys = args.get("state_keys")
max_dim = int(args["max_state_dim"])
max_keys = 16
skipped = {}
skip_overflow = 0
overwritten = set()
runtime_failures = []

api = None
if not args["no_api"]:
    api = GeometryOnlyWorldAPI(
        cache_dir=args.get("cache_dir"),
        api_calls_dir=args.get("api_calls_dir"),
    )

def build():
    return cls(
        frame_size=tuple(args["frame_size"]),
        api=api,
        fps=args["fps"],
    )

@lru_cache(maxsize=4)
def load_image(path):
    with Image.open(path) as image:
        return np.array(image.convert("RGB"))

def note_skip(key, reason, arr=None):
    global skip_overflow
    key = str(key)[:80]
    if key in skipped:
        return
    if len(skipped) >= 64:
        skip_overflow += 1
        return
    item = {"reason": reason}
    if arr is not None:
        item.update({
            "shape": list(arr.shape)[:4],
            "size": int(arr.size),
            "dtype": str(arr.dtype)[:24],
        })
    skipped[key] = item

def numeric(state):
    items = state.items() if isinstance(state, dict) else [("__state__", state)]
    out = {}
    for raw_key, value in items:
        key = str(raw_key)[:80]
        try:
            arr = np.asarray(value)
        except Exception:
            note_skip(key, "not_array_like")
            continue
        if arr.dtype.kind == "b":
            note_skip(key, "boolean_or_mask", arr)
            continue
        if arr.dtype.kind not in "iuf":
            note_skip(key, "non_numeric", arr)
            continue
        if arr.ndim >= 2 and arr.size > max_dim:
            note_skip(key, "image_shaped", arr)
            continue
        if arr.size > max_dim:
            note_skip(key, "oversized", arr)
            continue
        if arr.size == 0:
            note_skip(key, "empty", arr)
            continue
        flat = np.asarray(arr, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(flat)):
            note_skip(key, "non_finite", arr)
            continue
        if len(out) >= max_keys:
            note_skip(key, "component_count_limit", arr)
            continue
        out[key] = flat
    return out

def latent(key):
    key = str(key).lower()
    if re.search(
        r"(^|_)(delta_|vel|velocity|omega|latent|hidden|momentum|dd)(_|$|[a-z])",
        key,
    ):
        return True
    return bool(
        re.match(r"d(theta|angle|q|pos|x|y|z)", key)
        or re.search(r"(^|_)d(theta|angle|q|pos|x|y|z)", key)
    )

def angle(key):
    key = str(key).lower()
    if latent(key):
        return False
    if key in {"q", "qpos", "joint", "joints", "joint_pos", "joint_positions"}:
        return True
    return bool(re.search(r"(^|_)(theta|angle)(\d*|_|$)", key))

def delta(key, prediction, reference):
    difference = np.asarray(prediction) - np.asarray(reference)
    if angle(key):
        difference = np.arctan2(np.sin(difference), np.cos(difference))
    return difference

first = args["transitions"][0]
try:
    initial_sim = build()
    initial_sim.fit(load_image(first["current"]), load_image(first["goal"]))
except Exception as exc:
    print(json.dumps({"error": "FIT_FAILED", "detail": repr(exc)[:500]}))
    sys.exit(0)
if not hasattr(initial_sim, "params") or not isinstance(initial_sim.params, dict):
    print(json.dumps({"error": "NO_PARAMS"}))
    sys.exit(0)
missing = [name for name in names if name not in initial_sim.params]
if missing:
    print(json.dumps({
        "error": "MISSING_PARAMS",
        "missing": missing,
        "have": [str(key)[:80] for key in list(initial_sim.params)[:32]],
    }))
    sys.exit(0)
try:
    defaults = {name: float(initial_sim.params[name]) for name in names}
except Exception as exc:
    print(json.dumps({"error": "NONSCALAR_PARAMS", "detail": repr(exc)[:500]}))
    sys.exit(0)
if not all(np.isfinite(value) for value in defaults.values()):
    print(json.dumps({"error": "NONFINITE_PARAMS"}))
    sys.exit(0)

records = []
for transition in args["transitions"]:
    try:
        current_sim = build()
        current_sim.fit(
            load_image(transition["current"]),
            load_image(transition["goal"]),
        )
        current = numeric(current_sim.state)
        # A distinct instance and fit make the next-state reference genuinely
        # teacher-forced rather than a continuation of the current simulation.
        next_sim = build()
        next_sim.fit(load_image(transition["next"]), load_image(transition["goal"]))
        target = numeric(next_sim.state)
        records.append({
            **transition,
            "current_true": current,
            "target_true": target,
        })
    except Exception as exc:
        runtime_failures.append({
            "trajectory": transition.get("trajectory"),
            "step": transition.get("step"),
            "phase": "true_frame_fit",
            "error": f"{type(exc).__name__}: {exc}"[:300],
        })

if len(records) < 2:
    print(json.dumps({
        "error": "TOO_FEW_TRANSITIONS",
        "failures": runtime_failures[:12],
    }))
    sys.exit(0)

all_keys = sorted({
    key
    for record in records
    for key in set(record["current_true"]) | set(record["target_true"])
})
eligible = []
scales = {}
for key in all_keys:
    if requested_keys is not None and key not in requested_keys:
        note_skip(key, "not_selected")
        continue
    if latent(key):
        note_skip(key, "latent_or_unobservable_rate")
        continue
    pairs = []
    missing_count = 0
    shape_mismatch = False
    for record in records:
        current = record["current_true"].get(key)
        target = record["target_true"].get(key)
        if current is None or target is None:
            missing_count += 1
            continue
        if current.shape != target.shape:
            shape_mismatch = True
            continue
        pairs.append((current, target))
    if shape_mismatch or missing_count or len(pairs) != len(records):
        note_skip(key, "not_consistently_observable")
        continue
    movements = np.concatenate([
        delta(key, target, current).reshape(-1)
        for current, target in pairs
    ])
    scale = float(np.sqrt(np.mean(np.square(movements))))
    if not np.isfinite(scale) or scale <= 1e-8:
        note_skip(key, "static_in_sampled_transitions")
        continue
    eligible.append(key)
    scales[key] = max(scale, 1e-6)
    if len(eligible) >= max_keys:
        break

if requested_keys is not None:
    observed_names = set(all_keys) | set(skipped)
    for key in requested_keys:
        if key not in observed_names:
            note_skip(key, "not_observed")
if not eligible:
    print(json.dumps({
        "error": "NO_OBSERVABLE_COMPONENTS",
        "skipped_components": skipped,
        "failures": runtime_failures[:12],
    }))
    sys.exit(0)

train_records = [record for record in records if record["split"] == "train"]
held_records = [record for record in records if record["split"] == "held_out"]
if not train_records or not held_records:
    print(json.dumps({"error": "SPLIT_FAILED"}))
    sys.exit(0)

def predict(params, record):
    sim = build()
    if not hasattr(sim, "params") or not isinstance(sim.params, dict):
        sim.params = {}
    sim.params.update(params)
    sim.fit(load_image(record["current"]), load_image(record["goal"]))
    if not hasattr(sim, "params") or not isinstance(sim.params, dict):
        raise RuntimeError("NO_PARAMS")
    missing_now = [name for name in names if name not in sim.params]
    if missing_now:
        raise RuntimeError("MISSING_PARAMS:" + ",".join(missing_now))
    for name, value in params.items():
        try:
            if abs(float(sim.params[name]) - float(value)) > 1e-9:
                overwritten.add(name)
        except Exception:
            overwritten.add(name)
    # Reassert after fit so update() sees the candidate even if fit reset it.
    sim.params.update(params)
    # Exactly one recorded action is applied for this teacher-forced sample.
    sim.update(np.asarray(record["action"], dtype=np.float64))
    return numeric(sim.state)

def metrics(params, subset):
    squared = {key: [] for key in eligible}
    failures = 0
    for record in subset:
        try:
            prediction = predict(params, record)
        except Exception:
            prediction = {}
            failures += 1
        for key in eligible:
            target = record["target_true"][key]
            value = prediction.get(key)
            if value is None or value.shape != target.shape:
                squared[key].append(25.0)
                continue
            normalized = delta(key, value, target) / scales[key]
            if not np.all(np.isfinite(normalized)):
                squared[key].append(25.0)
            else:
                squared[key].append(float(np.mean(np.square(normalized))))
    per_key = {
        key: float(np.sqrt(np.mean(values)))
        for key, values in squared.items()
        if values
    }
    total = float(np.sqrt(np.mean([
        value * value for value in per_key.values()
    ]))) if per_key else 5.0
    return {"total": total, "per_key": per_key, "prediction_failures": failures}

before_train = metrics(defaults, train_records)
before_held = metrics(defaults, held_records)
evaluations = {"count": 0}

def objective(vector):
    evaluations["count"] += 1
    params = {name: float(vector[index]) for index, name in enumerate(names)}
    return metrics(params, train_records)["total"]

population = 6
maxiter = max(
    1,
    int(args["budget"]) // max(1, population * len(names)) - 1,
)
try:
    result = differential_evolution(
        objective,
        args["bounds"],
        popsize=population,
        maxiter=maxiter,
        tol=0.01,
        seed=0,
        polish=False,
        workers=1,
        updating="immediate",
    )
    fitted = {
        name: float(result.x[index])
        for index, name in enumerate(names)
    }
except RuntimeError as exc:
    print(json.dumps({"error": "PARAM_CONTRACT", "detail": str(exc)[:500]}))
    sys.exit(0)

after_train = metrics(fitted, train_records)
# Never recommend an optimizer draw that is worse than the in-bounds default.
if (
    all(args["bounds"][i][0] <= defaults[name] <= args["bounds"][i][1]
        for i, name in enumerate(names))
    and before_train["total"] < after_train["total"]
):
    fitted = defaults.copy()
    after_train = before_train
after_held = metrics(fitted, held_records)

if skip_overflow:
    skipped["__additional_skipped__"] = {
        "reason": "report_limit",
        "count": int(skip_overflow),
    }
out = {
    "param_names": names,
    "default_params": defaults,
    "fitted_params": fitted,
    "bounds": args["bounds"],
    "nfev": int(evaluations["count"]),
    "train_transition_count": len(train_records),
    "held_out_transition_count": len(held_records),
    "train_transition_ids": [
        [int(record["trajectory"]), int(record["step"])]
        for record in train_records
    ],
    "held_out_transition_ids": [
        [int(record["trajectory"]), int(record["step"])]
        for record in held_records
    ],
    "component_scales": scales,
    "angle_keys": [key for key in eligible if angle(key)],
    "before": {"train": before_train, "held_out": before_held},
    "after": {"train": after_train, "held_out": after_held},
    "skipped_components": skipped,
    "overwritten_params": sorted(overwritten),
    "runtime_failures": runtime_failures[:12],
}
print(json.dumps(out))
"""

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as temporary:
            temporary.write(runner_src)
            runner_path = temporary.name
        runner_args = {
            "sim_path": self._sandbox_path,
            "class_name": self._simulator_class_name,
            "frame_size": list(self._frame_size),
            "fps": self._fps,
            "param_names": param_names,
            "bounds": bounds_list,
            "transitions": transition_specs,
            "state_keys": requested_keys,
            "budget": budget,
            "max_state_dim": max_state_dim,
            "no_api": self._no_api,
            "cache_dir": self._cache_dir,
            "api_calls_dir": self._world_api_log_dir,
        }
        try:
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=360,
                env=self._subprocess_env(),
            )
        except _subprocess.TimeoutExpired:
            return _fail(
                "bounded transition calibration timed out after 360 seconds.",
                {"bounds": str(bounds)[:200]},
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            return _fail(
                f"runner exited with code {result.returncode}: "
                f"{result.stderr[-1200:]}",
                {"bounds": str(bounds)[:200]},
            )
        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            return _fail(
                f"could not parse bounded runner output: {exc}; "
                f"stdout tail={result.stdout[-800:]!r}",
                {"bounds": str(bounds)[:200]},
            )

        error = payload.get("error")
        if error == "NO_PARAMS":
            return _fail(
                "your simulator has no `self.params` dict; expose selected "
                "dynamics constants there before calibration."
            )
        if error == "MISSING_PARAMS":
            return _fail(
                f"requested parameters {payload.get('missing')} are not in "
                f"self.params; available keys: {payload.get('have')}."
            )
        if error:
            detail = payload.get("detail") or payload.get("failures") or ""
            skipped_detail = payload.get("skipped_components")
            suffix = f"; skipped={skipped_detail}" if skipped_detail else ""
            return _fail(f"{error}: {detail}{suffix}")

        def _fmt_params(values: dict) -> str:
            return ", ".join(
                f"{name}={float(value):.6g}" for name, value in values.items()
            )

        def _fmt_error(value: Any) -> str:
            try:
                return f"{float(value):.4f}"
            except (TypeError, ValueError):
                return "n/a"

        before = payload["before"]
        after = payload["after"]
        summary_lines = [
            (
                "calibrate_transition_parameters: fitted "
                f"{payload.get('param_names')} on "
                f"{payload.get('train_transition_count')} teacher-forced train "
                f"transitions; evaluated {payload.get('held_out_transition_count')} "
                f"held-out transitions ({payload.get('nfev')} optimizer evaluations)."
            ),
            (
                "  protocol: fit(TRUE current, fixed trajectory goal) -> exactly "
                "one recorded action -> compare with fresh fit(TRUE next, same goal)."
            ),
            f"  default params: {_fmt_params(payload.get('default_params', {}))}",
            f"  fitted params:  {_fmt_params(payload.get('fitted_params', {}))}",
            "  NORMALIZED ONE-STEP RMSE (lower is better):",
            (
                "    TRAIN total: "
                f"{_fmt_error(before['train']['total'])} -> "
                f"{_fmt_error(after['train']['total'])}"
            ),
        ]
        train_keys = sorted(
            set(before["train"].get("per_key", {}))
            | set(after["train"].get("per_key", {}))
        )
        for key in train_keys:
            angle_note = " [wrapped angle]" if key in payload.get("angle_keys", []) else ""
            summary_lines.append(
                f"      {key}: "
                f"{_fmt_error(before['train']['per_key'].get(key))} -> "
                f"{_fmt_error(after['train']['per_key'].get(key))}; "
                f"scale={_fmt_error(payload.get('component_scales', {}).get(key))}"
                f"{angle_note}"
            )
        summary_lines.append(
            "    HELD-OUT total: "
            f"{_fmt_error(before['held_out']['total'])} -> "
            f"{_fmt_error(after['held_out']['total'])}"
        )
        held_keys = sorted(
            set(before["held_out"].get("per_key", {}))
            | set(after["held_out"].get("per_key", {}))
        )
        for key in held_keys:
            summary_lines.append(
                f"      {key}: "
                f"{_fmt_error(before['held_out']['per_key'].get(key))} -> "
                f"{_fmt_error(after['held_out']['per_key'].get(key))}"
            )
        skipped = payload.get("skipped_components") or {}
        if skipped:
            skip_text = ", ".join(
                f"{str(key)[:80]}={str(info.get('reason', info))[:80]}"
                for key, info in list(skipped.items())[:24]
            )
            summary_lines.append(f"  skipped components: {skip_text}")
        if payload.get("overwritten_params"):
            summary_lines.append(
                "  WARNING: fit() overwrote candidate values for "
                f"{payload['overwritten_params']}; values were reasserted before "
                "the one update, but fit-dependent parameter effects are not "
                "identified by this tool."
            )
        summary_lines.append(
            "  NEXT: source code was not changed. Adopt values manually only if "
            "held-out transition errors also improve, then re-run state consistency."
        )
        summary = "\n".join(summary_lines)
        if len(summary) > 12000:
            summary = summary[:11800] + "\n  [response bounded; full report is in artifact]"

        artifact_text = _json.dumps(payload, indent=2)
        if len(artifact_text) > 60000:
            payload["train_transition_ids"] = payload.get(
                "train_transition_ids", []
            )[:16]
            payload["held_out_transition_ids"] = payload.get(
                "held_out_transition_ids", []
            )[:16]
            payload["artifact_truncated"] = True
            artifact_text = _json.dumps(payload, indent=2)
        self._log_critic_tool_call(
            tool_name=tool_name,
            args={
                "bounds": str(bounds)[:200],
                "trajectory_indices": selected,
                "state_keys": requested_keys,
                "max_transitions": max_transitions,
                "budget": budget,
                "max_state_dim": max_state_dim,
            },
            result_summary=summary[:500],
            extra_files={"transition_calibration.json": artifact_text},
        )
        return summary

    def calibrate_parameters(
        self,
        bounds: dict,
        trajectory_indices: list[int] | None = None,
        budget: int = 60,
    ) -> str:
        """Fit your simulator's free physics constants to the training data.

        Instead of hand-guessing dynamics constants (push gain, friction,
        contact stiffness, substep count, mass ratios, …), expose them and let
        this tool MEASURE them from the demonstrations. This is system
        identification: it searches the constants that make your `update`
        reproduce the trajectories most faithfully.

        Contract — how to make a parameter fittable:

        - Store every tunable scalar in a dict `self.params` set in `__init__`
          (e.g. `self.params = {"k_trans": 0.8, "k_rot": 0.005}`), and READ it
          live inside `update`/`terminal_cost` (`self.params["k_trans"]`).
        - Do NOT reassign `self.params` wholesale inside `fit` — `fit` may read
          it, but if `fit` overwrites the dict with hard-coded defaults this
          tool cannot inject candidates (it will warn you if it detects this).
        - Keep `self.params` OFF `self.state` — it is constant within a rollout,
          so it must not be snapshotted/restored by the planner.

        What this tool does: for each candidate setting of `self.params` it
        re-fits your simulator on a training trajectory's (first, last) frames,
        replays that trajectory's ground-truth actions through `update`, and
        scores the result by the terminal_cost reduction_ratio (final/initial)
        — the same render-independent fidelity metric as
        `validate_against_training`. It minimises the mean reduction_ratio over
        the fitting trajectories with a black-box optimiser
        (`scipy.optimize.differential_evolution`, gradient-free), then reports a
        held-out trajectory's ratio so you can see whether the fit generalises.

        IMPORTANT: this tool does NOT modify your code. It returns the fitted
        values; YOU must write them into `self.params` (via `edit_code`) and
        re-validate. The decision to adopt them stays with you.

        Args:
            bounds: dict mapping each parameter name (a key of `self.params`) to
                a `[low, high]` search range, e.g.
                `{"k_trans": [0.1, 2.0], "k_rot": [0.0, 0.05]}`. Only the names
                you list are fitted; others keep their current values.
            trajectory_indices: which trajectories to FIT against. Defaults to
                the first few available. A different trajectory is held out for
                the generalisation report when possible.
            budget: rough number of objective evaluations (default 60). Higher =
                more thorough but slower.

        Returns:
            A summary string with the fitted parameter values, the mean
            reduction_ratio before vs after, the per-trajectory ratios, a
            held-out ratio, and warnings (param pinned at a bound, no
            improvement / irreducible floor, or `fit` overwriting `self.params`).
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        # --- validate bounds -------------------------------------------------
        if not isinstance(bounds, dict) or not bounds:
            err = (
                "[calibrate_parameters] `bounds` must be a non-empty dict mapping "
                "parameter names to [low, high], e.g. {'k_trans': [0.1, 2.0]}."
            )
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=err,
            )
            return err
        param_names = list(bounds.keys())
        bounds_list = []
        contact_prior_capped: list[str] = []
        geometry_prior_capped: list[str] = []
        for name in param_names:
            rng = bounds[name]
            try:
                lo, hi = float(rng[0]), float(rng[1])
            except (TypeError, ValueError, IndexError):
                err = (
                    f"[calibrate_parameters] bound for '{name}' must be "
                    f"[low, high]; got {rng!r}."
                )
                self._log_critic_tool_call(
                    tool_name="calibrate_parameters", args={}, result_summary=err,
                )
                return err
            if self._is_contact_transfer_gain_param(name):
                cap_hi = min(hi, 1.0)
                if lo > 1.0:
                    err = (
                        f"[calibrate_parameters] bound for '{name}' is incompatible "
                        "with the contact/transfer gain prior <= 1.0; "
                        f"got [{lo}, {hi}]. Lower the range."
                    )
                    self._log_critic_tool_call(
                        tool_name="calibrate_parameters", args={}, result_summary=err,
                    )
                    return err
                if cap_hi < hi:
                    hi = cap_hi
                    contact_prior_capped.append(str(name))
            # Only clamp clearly pixel-scale search ranges (hi >= 2 px). Ranges
            # that stay below ~2 are likely normalized/physical units and must
            # not be forced to a pixel floor.
            if self._is_geometry_visibility_param(name) and hi >= 2.0 and lo < 0.75:
                lo = 0.75
                geometry_prior_capped.append(str(name))
            if not (hi > lo):
                err = (
                    f"[calibrate_parameters] bound for '{name}' must have high > low; "
                    f"got [{lo}, {hi}]."
                )
                self._log_critic_tool_call(
                    tool_name="calibrate_parameters", args={}, result_summary=err,
                )
                return err
            bounds_list.append([lo, hi])

        # --- choose fit + held-out trajectories ------------------------------
        # Respect the configured fidelity-training split. The benchmark-held-out
        # trajectory must never leak into parameter search or its validation
        # report.
        available = sorted(int(i) for i in self._fidelity_train_trajectories())
        if not available:
            err = "[calibrate_parameters] no training trajectories found."
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=err,
            )
            return err

        if trajectory_indices:
            fit_indices = [int(i) for i in trajectory_indices if int(i) in available]
            if not fit_indices:
                err = (
                    f"[calibrate_parameters] none of trajectory_indices "
                    f"{trajectory_indices} exist; available: {available}."
                )
                self._log_critic_tool_call(
                    tool_name="calibrate_parameters", args={}, result_summary=err,
                )
                return err
        else:
            fit_indices = available[: min(3, len(available))]
        holdout_index = next((i for i in available if i not in fit_indices), None)

        # --- load (first, last, actions) for each chosen trajectory ----------
        def _load(idx):
            tr = _dt.read_trajectory(self._dataset_dir, idx)
            return {
                "index": idx,
                "first": str(tr["frames"][0]),
                "last": str(tr["frames"][-1]),
                "actions": tr["actions"].tolist(),
            }

        try:
            fit_trajectories = [_load(i) for i in fit_indices]
            holdout_trajectory = _load(holdout_index) if holdout_index is not None else None
        except (FileNotFoundError, ValueError) as exc:
            err = f"[calibrate_parameters] {exc}"
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=err,
            )
            return err

        try:
            budget = max(10, int(budget))
        except (TypeError, ValueError):
            budget = 60

        # --- subprocess: runs the whole optimisation in isolation ------------
        runner_src = (
            "import sys, json, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from scipy.optimize import differential_evolution\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "names = args['param_names']\n"
            "overwritten = set()\n"
            "_api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "def _build():\n"
            "    return cls(frame_size=tuple(args['frame_size']), api=_api, fps=args['fps'])\n"
            "def _ratio(params, traj, inject):\n"
            "    sim = _build()\n"
            "    if inject is not None:\n"
            "        if not hasattr(sim, 'params') or not isinstance(getattr(sim, 'params', None), dict):\n"
            "            sim.params = {}\n"
            "        sim.params.update(inject)\n"
            "    image_A = np.array(Image.open(traj['first']).convert('RGB'))\n"
            "    image_B = np.array(Image.open(traj['last']).convert('RGB'))\n"
            "    sim.fit(image_A, image_B)\n"
            "    if not hasattr(sim, 'params') or not isinstance(getattr(sim, 'params', None), dict):\n"
            "        raise RuntimeError('NO_PARAMS')\n"
            "    missing = [n for n in names if n not in sim.params]\n"
            "    if missing:\n"
            "        raise RuntimeError('MISSING_PARAMS:' + ','.join(missing))\n"
            "    if inject is not None:\n"
            "        for k, v in inject.items():\n"
            "            if abs(float(sim.params.get(k, 1e30)) - float(v)) > 1e-9:\n"
            "                overwritten.add(k)\n"
            "        sim.params.update(inject)\n"   # re-assert so update() sees our values
            "    initial = float(sim.planning_objective())\n"
            "    for a in traj['actions']:\n"
            "        sim.update(np.asarray(a, dtype=np.float64))\n"
            "    final = float(sim.planning_objective())\n"
            "    if not np.isfinite(initial) or not np.isfinite(final) or initial <= 0:\n"
            "        return None\n"
            "    return final / initial\n"
            # --- baseline (default params, no injection) ---
            "default_params = {}\n"
            "try:\n"
            "    sim0 = _build()\n"
            "    iA = np.array(Image.open(args['fit_trajectories'][0]['first']).convert('RGB'))\n"
            "    iB = np.array(Image.open(args['fit_trajectories'][0]['last']).convert('RGB'))\n"
            "    sim0.fit(iA, iB)\n"
            "    if not hasattr(sim0, 'params') or not isinstance(getattr(sim0, 'params', None), dict):\n"
            "        print(json.dumps({'error': 'NO_PARAMS'})); sys.exit(0)\n"
            "    missing0 = [n for n in names if n not in sim0.params]\n"
            "    if missing0:\n"
            "        print(json.dumps({'error': 'MISSING_PARAMS', 'missing': missing0, 'have': list(sim0.params.keys())})); sys.exit(0)\n"
            "    default_params = {n: float(sim0.params[n]) for n in names}\n"
            "except Exception as e:\n"
            "    print(json.dumps({'error': 'FIT_FAILED', 'detail': repr(e)})); sys.exit(0)\n"
            "before_per = [(_ratio(default_params, t, None)) for t in args['fit_trajectories']]\n"
            "before_clean = [r for r in before_per if r is not None]\n"
            "before_mean = float(np.mean(before_clean)) if before_clean else None\n"
            # --- objective ---
            "evals = {'n': 0}\n"
            "def objective(x):\n"
            "    evals['n'] += 1\n"
            "    p = {names[i]: float(x[i]) for i in range(len(names))}\n"
            "    rs = []\n"
            "    for t in args['fit_trajectories']:\n"
            "        try:\n"
            "            r = _ratio(p, t, p)\n"
            "        except RuntimeError:\n"
            "            raise\n"
            "        except Exception:\n"
            "            r = None\n"
            "        rs.append(r if r is not None else 5.0)\n"
            "    return float(np.mean(rs))\n"
            "n_params = len(names)\n"
            "popsize = 10\n"
            "maxiter = max(2, int(args['budget'] // (popsize * n_params)))\n"
            "fatal = None\n"
            "try:\n"
            "    res = differential_evolution(objective, args['bounds'], popsize=popsize,\n"
            "        maxiter=maxiter, tol=0.01, seed=0, polish=False, init='latinhypercube')\n"
            "    best_x = res.x; nfev = int(res.nfev)\n"
            "except RuntimeError as e:\n"
            "    fatal = str(e)\n"
            "if fatal is not None:\n"
            "    out = {'error': 'PARAM_CONTRACT', 'detail': fatal}\n"
            "    print(json.dumps(out)); sys.exit(0)\n"
            "best_params = {names[i]: float(best_x[i]) for i in range(len(names))}\n"
            "after_per = [(_ratio(best_params, t, best_params)) for t in args['fit_trajectories']]\n"
            "after_clean = [r for r in after_per if r is not None]\n"
            "after_mean = float(np.mean(after_clean)) if after_clean else None\n"
            "holdout = args.get('holdout_trajectory')\n"
            "holdout_before = holdout_after = None\n"
            "if holdout is not None:\n"
            "    holdout_before = _ratio(default_params, holdout, None)\n"
            "    holdout_after = _ratio(best_params, holdout, best_params)\n"
            "out = {\n"
            "  'param_names': names,\n"
            "  'default_params': default_params,\n"
            "  'best_params': best_params,\n"
            "  'before_mean': before_mean,\n"
            "  'after_mean': after_mean,\n"
            "  'before_per_traj': [None if r is None else round(r, 4) for r in before_per],\n"
            "  'after_per_traj': [None if r is None else round(r, 4) for r in after_per],\n"
            "  'fit_indices': [t['index'] for t in args['fit_trajectories']],\n"
            "  'holdout_index': (holdout['index'] if holdout is not None else None),\n"
            "  'holdout_before': holdout_before,\n"
            "  'holdout_after': holdout_after,\n"
            "  'nfev': nfev,\n"
            "  'overwritten': sorted(overwritten),\n"
            "  'bounds': args['bounds'],\n"
            "}\n"
            "print(json.dumps(out))\n"
        )

        with _tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "param_names": param_names,
                "bounds": bounds_list,
                "fit_trajectories": fit_trajectories,
                "holdout_trajectory": holdout_trajectory,
                "budget": budget,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=600,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            err = (
                f"[calibrate_parameters] runner exited with code "
                f"{result.returncode}\nstderr:\n{result.stderr[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="calibrate_parameters",
                args={"bounds": str(bounds)[:200]},
                result_summary=err[:300],
                extra_files={"stderr.txt": result.stderr},
            )
            return err

        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            err = (
                f"[calibrate_parameters] could not parse runner output: {exc}\n"
                f"stdout:\n{result.stdout[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=err[:300],
            )
            return err

        # --- contract / fit-failure errors -----------------------------------
        if payload.get("error") == "NO_PARAMS":
            msg = (
                "[calibrate_parameters] your simulator has no `self.params` dict. "
                "To calibrate, store your tunable constants in a dict named "
                "`self.params` (set in `__init__`) and read them inside `update`/"
                "`terminal_cost` — e.g. `self.params = {'k_trans': 0.8}` then "
                "`self.state['block_pos'] += self.params['k_trans'] * push`."
            )
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=msg,
            )
            return msg
        if payload.get("error") == "MISSING_PARAMS":
            msg = (
                f"[calibrate_parameters] these requested names are not keys of "
                f"self.params: {payload.get('missing')}. self.params currently has: "
                f"{payload.get('have')}. Fix the names in `bounds` or add the params."
            )
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=msg[:300],
            )
            return msg
        if payload.get("error") == "PARAM_CONTRACT":
            detail = payload.get("detail", "")
            hint = ""
            if detail.startswith("MISSING_PARAMS"):
                hint = (
                    " (a parameter disappeared from self.params on a re-fit — make "
                    "sure every name in `bounds` is always present in self.params)."
                )
            msg = (
                f"[calibrate_parameters] aborted on the parameter contract: {detail}.{hint}"
            )
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=msg[:300],
            )
            return msg
        if payload.get("error") == "FIT_FAILED":
            msg = (
                f"[calibrate_parameters] your simulator's fit()/terminal_cost() raised "
                f"before calibration could start: {payload.get('detail')}. Make "
                "run_simulation / validate_against_training pass first."
            )
            self._log_critic_tool_call(
                tool_name="calibrate_parameters", args={}, result_summary=msg[:300],
            )
            return msg

        # --- format the success summary --------------------------------------
        before_mean = payload.get("before_mean")
        after_mean = payload.get("after_mean")
        best_params = payload.get("best_params", {})
        default_params = payload.get("default_params", {})

        def _fmt(d):
            return ", ".join(f"{k}={v:.5g}" for k, v in d.items())

        def _r(x):
            return "n/a" if x is None else f"{x:.4f}"

        summary = (
            f"calibrate_parameters: fitted {payload.get('param_names')} on "
            f"trajectories {payload.get('fit_indices')} "
            f"({payload.get('nfev')} evaluations).\n"
            f"  PRIMARY — mean reduction_ratio (lower = more faithful):\n"
            f"    before (your values): {_r(before_mean)}   "
            f"after (fitted):  {_r(after_mean)}\n"
            f"    per-trajectory before: {payload.get('before_per_traj')}\n"
            f"    per-trajectory after:  {payload.get('after_per_traj')}\n"
            f"  default params: {_fmt(default_params)}\n"
            f"  fitted params:  {_fmt(best_params)}\n"
        )
        if contact_prior_capped:
            summary += (
                "  prior: contact/transfer gains were bounded to <=1.0 for "
                f"{sorted(set(contact_prior_capped))}.\n"
            )
        if geometry_prior_capped:
            summary += (
                "  prior: geometry/visibility parameters were bounded to >=0.75 px "
                "to avoid sub-pixel object collapse: "
                f"{sorted(set(geometry_prior_capped))}.\n"
            )
        if payload.get("holdout_index") is not None:
            summary += (
                f"  HELD-OUT trajectory {payload.get('holdout_index')} (not used in the fit): "
                f"ratio before={_r(payload.get('holdout_before'))}, "
                f"after={_r(payload.get('holdout_after'))} "
                "(this is the generalisation check — adopt the fit only if this improved too).\n"
            )
        summary += (
            "  NEXT: this tool did NOT change your code. Write the fitted values into "
            "self.params with edit_code, then re-run validate_against_training to confirm."
        )

        # --- warnings ---------------------------------------------------------
        bounds_arr = payload.get("bounds", [])
        at_bound = []
        for i, name in enumerate(payload.get("param_names", [])):
            v = best_params.get(name)
            if v is None or i >= len(bounds_arr):
                continue
            lo, hi = bounds_arr[i]
            span = hi - lo
            if span > 0 and (abs(v - lo) < 0.02 * span or abs(v - hi) < 0.02 * span):
                at_bound.append(name)
        if at_bound:
            summary += (
                f"\n  WARNING: {at_bound} settled at the edge of their search range — "
                "the true optimum may lie outside `bounds`. Widen the range and re-run."
            )
        if (
            before_mean is not None
            and after_mean is not None
            and after_mean >= 0.98 * before_mean
        ):
            summary += (
                f"\n  NOTE: calibration barely improved the ratio "
                f"({_r(before_mean)} -> {_r(after_mean)}). Either your constants were "
                "already near-optimal, or these parameters don't affect the residual, "
                "or your update()'s STRUCTURE is incomplete (an irreducible floor no "
                "constant can remove — e.g. a missing friction or rotation term). "
                "Tuning won't fix a structural gap."
            )
        if payload.get("overwritten"):
            summary += (
                f"\n  WARNING: fit() reassigned these params after we set them: "
                f"{payload.get('overwritten')}. If any are used INSIDE fit() (not just "
                "update()), calibration of them won't take effect. Set their defaults in "
                "__init__ and have fit() read self.params rather than overwrite it."
            )

        self._log_critic_tool_call(
            tool_name="calibrate_parameters",
            args={
                "bounds": str(bounds)[:200],
                "fit_indices": payload.get("fit_indices"),
                "budget": budget,
            },
            result_summary=summary[:500],
            extra_files={
                "calibration.json": _json.dumps(payload, indent=2),
            },
        )
        return summary

    def validate_with_cem(
        self,
        start_trajectory_index: int,
        start_frame_index: int,
        goal_trajectory_index: int,
        goal_frame_index: int,
        horizon: int = 20,
        cem_iters: int = 10,
        cem_population: int = 200,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ) -> str:
        """Self-validate the simulator by running CEM against arbitrary start/goal frames.

        Pick a start frame and a goal frame from the training data — typically
        from DIFFERENT trajectories or far-apart steps so the planner must
        discover a non-trivial action sequence (a single straight line is
        unlikely to reach the goal). Calls ``sim.fit(start, goal)``, runs CEM
        to optimise a ``horizon``-step action sequence against your
        ``terminal_cost``, applies the plan through ``update``, and reports
        whether CEM made meaningful progress.

        Use this to detect cases where ``terminal_cost`` traps the planner.
        Example failure mode: a pure straight-line distance pulls CEM samples
        toward the goal; if the feasible path requires a detour around an
        obstacle, the cost gradient points into the obstacle and CEM converges
        to whatever state is closest-to-goal it can physically reach — never
        exploring the detour path. ``final_loss`` then stays close to
        ``initial_loss`` and the tool flags it as stuck.

        PASS requires more than a falling cost: the tool also audits
        GOAL-REACH in state space — the planned rollout must move the
        components named in ``target_state`` toward their target values,
        measured in units of each component's max one-step motion from
        training replay. A plan whose internal cost collapses while the
        task-relevant state stays far from target is flagged as a FAIL
        (that pattern predicts real-environment planning failure).

        Pick start/goal pairs that exercise long-range planning:
        ``frame_0`` of one trajectory + ``frame_last`` of a different
        trajectory is a good default. Run on at least 2-3 diverse pairs.

        By default, CEM uses the deployed action bounds when available (the
        same bounds the test-time planner uses). If deployed bounds are not
        configured for this run, it falls back to the observed training min/max.
        You can override bounds explicitly via ``action_low`` / ``action_high``.

        Args:
            start_trajectory_index, start_frame_index: training frame used as image_A.
            goal_trajectory_index, goal_frame_index:   training frame used as image_B.
            horizon: number of action steps CEM optimises over (default 20).
            cem_iters: CEM refit iterations (default 10).
            cem_population: samples per CEM iteration (default 200).
            action_low, action_high: optional explicit CEM bounds.

        Returns:
            A summary with initial_loss, final_loss, per-step loss preview,
            per-step state preview, and a "STUCK" warning if final_loss is
            close to initial_loss.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        try:
            s_t = int(start_trajectory_index)
            s_f = int(start_frame_index)
            g_t = int(goal_trajectory_index)
            g_f = int(goal_frame_index)
            horizon = int(horizon)
            cem_iters = int(cem_iters)
            cem_population = int(cem_population)
            start_traj = _dt.read_trajectory(self._dataset_dir, s_t)
            goal_traj = _dt.read_trajectory(self._dataset_dir, g_t)
        except (FileNotFoundError, ValueError) as exc:
            err = f"[validate_with_cem] {exc}"
            self._log_critic_tool_call(
                tool_name="validate_with_cem",
                args={
                    "start": f"({start_trajectory_index},{start_frame_index})",
                    "goal":  f"({goal_trajectory_index},{goal_frame_index})",
                },
                result_summary=err,
            )
            return err

        if not (0 <= s_f < len(start_traj["frames"])):
            err = (
                f"[validate_with_cem] start_frame_index {s_f} out of range "
                f"[0, {len(start_traj['frames'])})"
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem",
                args={"start_frame_index": s_f},
                result_summary=err,
            )
            return err
        if not (0 <= g_f < len(goal_traj["frames"])):
            err = (
                f"[validate_with_cem] goal_frame_index {g_f} out of range "
                f"[0, {len(goal_traj['frames'])})"
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem",
                args={"goal_frame_index": g_f},
                result_summary=err,
            )
            return err

        stacked = self._collect_training_actions()
        if stacked is None or stacked.ndim != 2 or stacked.shape[0] == 0:
            err = "[validate_with_cem] could not derive action bounds — no usable training trajectories found."
            self._log_critic_tool_call(
                tool_name="validate_with_cem", args={}, result_summary=err,
            )
            return err
        action_dim = stacked.shape[1]
        resolved_low, resolved_high, bounds_source = self._resolve_action_bounds(
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )
        if resolved_low is None or resolved_high is None:
            err = f"[validate_with_cem] could not resolve action bounds ({bounds_source})."
            self._log_critic_tool_call(
                tool_name="validate_with_cem", args={}, result_summary=err,
            )
            return err

        start_path = str(start_traj["frames"][s_f])
        goal_path = str(goal_traj["frames"][g_f])

        runner_src = (
            "import sys, json, copy, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "from vdaworld.core.cem import CEM\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "image_A = np.array(Image.open(args['start_path']).convert('RGB'))\n"
            "image_B = np.array(Image.open(args['goal_path']).convert('RGB'))\n"
            "sim.fit(image_A, image_B)\n"
            "def _state_repr(s):\n"
            "    if isinstance(s, dict):\n"
            "        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}\n"
            "    if isinstance(s, np.ndarray):\n"
            "        return s.tolist()\n"
            "    return str(s)\n"
            "initial_state = _state_repr(sim.state)\n"
            "target_state  = _state_repr(sim.target_state)\n"
            "initial_loss = float(sim.planning_objective())\n"
            "cem = CEM(sim, action_dim=args['action_dim'],\n"
            "          action_low=np.asarray(args['action_low']),\n"
            "          action_high=np.asarray(args['action_high']),\n"
            "          horizon=args['horizon'],\n"
            "          population=args['cem_population'],\n"
            "          iters=args['cem_iters'],\n"
            "          seed=args['seed'])\n"
            "plan = cem.plan()\n"
            "cem_history = [{'iter': e.iteration, 'best': e.best_cost, 'mean': e.mean_cost} for e in cem.history]\n"
            "# CEM restores sim.state — apply plan freshly to record the trajectory.\n"
            "losses = [initial_loss]\n"
            "states = [initial_state]\n"
            "for a in plan:\n"
            "    sim.update(np.asarray(a, dtype=np.float64))\n"
            "    losses.append(float(sim.planning_objective()))\n"
            "    states.append(_state_repr(sim.state))\n"
            "out = {\n"
            "  'initial_state': initial_state,\n"
            "  'target_state': target_state,\n"
            "  'initial_loss': initial_loss,\n"
            "  'final_loss': losses[-1],\n"
            "  'min_loss': float(min(losses)),\n"
            "  'losses': losses,\n"
            "  'states': states,\n"
            "  'cem_history': cem_history,\n"
            "  'actions': [list(map(float, a)) for a in plan.tolist()],\n"
            "}\n"
            "print(json.dumps(out))\n"
        )

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        seed_for_cem = 42  # deterministic across calls so VLM sees reproducible results
        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "start_path": start_path,
                "goal_path": goal_path,
                "action_dim": int(action_dim),
                "action_low": resolved_low,
                "action_high": resolved_high,
                "horizon": horizon,
                "cem_iters": cem_iters,
                "cem_population": cem_population,
                "seed": seed_for_cem,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=180,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            err = (
                f"[validate_with_cem] runner exited with code "
                f"{result.returncode}\nstderr:\n{result.stderr[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem",
                args={"start": f"({s_t},{s_f})", "goal": f"({g_t},{g_f})"},
                result_summary=err[:300],
                extra_files={"stderr.txt": result.stderr},
            )
            return err

        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            err = (
                f"[validate_with_cem] could not parse runner output: {exc}\n"
                f"stdout:\n{result.stdout[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem", args={}, result_summary=err[:300],
            )
            return err

        losses = payload["losses"]
        states = payload["states"]
        initial_loss = payload["initial_loss"]
        final_loss = payload["final_loss"]
        min_loss = payload["min_loss"]

        n = len(losses)
        sample_idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        loss_preview = ", ".join(
            f"t={i}:{losses[i]:.3f}" for i in sample_idxs
        )
        state_preview = "\n".join(
            f"    t={i}: {states[i]}" for i in sample_idxs
        )

        # Stuck criterion (a): min_loss within 90% of initial_loss — CEM could
        # not find a meaningfully better trajectory than the starting position.
        stuck_completely = (initial_loss > 0) and (min_loss >= 0.9 * initial_loss)
        # CEM internal: did CEM's own best_cost decrease meaningfully?
        cem_history = payload.get("cem_history", [])
        cem_first_best = cem_history[0]["best"] if cem_history else None
        cem_last_best = cem_history[-1]["best"] if cem_history else None

        if cem_history:
            cem_line = (
                f"  CEM best_cost: first_iter={cem_first_best:.4f}, "
                f"last_iter={cem_last_best:.4f}"
            )
        else:
            cem_line = "  (no CEM history available)"

        plan_max_deltas = self._max_step_deltas_from_states(
            states,
            component_periodicity=self._declared_state_periodicity(),
        )
        audit_payload: dict[str, Any]
        limits, limits_err = self._realizability_limits_cached(
            margin=self._gate_realizability_margin
        )
        # Stuck criterion (b), unit-free: the plan must move the components
        # named in target_state toward their targets in STATE space, not just
        # reduce the sim's own cost (whose units the VLM itself chose).
        goal_reach = self._assess_goal_reach(
            target_state=payload["target_state"],
            states=states,
            horizon=horizon,
            limits=limits,
        )
        state_stuck = bool(goal_reach.get("state_stuck"))
        if limits_err or limits is None:
            audit_payload = {
                "available": False,
                "error": limits_err or "unknown",
                "plan_max_deltas": plan_max_deltas,
            }
            audit_line = (
                f"  NOTE: realizability audit unavailable ({audit_payload['error']})."
            )
        else:
            train_bounds = dict(limits["bound_max"])
            violations, missing_components = self._realizability_violations(
                plan_max_deltas,
                train_bounds,
            )
            if missing_components:
                audit_payload = {
                    "available": False,
                    "error": (
                        "state components missing in planned trajectory: "
                        f"{missing_components}"
                    ),
                    "plan_max_deltas": plan_max_deltas,
                    "train_bounds": train_bounds,
                }
                audit_line = (
                    "  NOTE: realizability audit unavailable "
                    f"(state components missing in planned trajectory: {missing_components})."
                )
            else:
                if violations:
                    worst_violation = max(violations, key=lambda x: x["ratio"])
                    audit_line = (
                        "  FAIL: plan realizability audit — this CEM plan relies on "
                        "one-step state jumps larger than the training-derived bound. "
                        f"Worst component={worst_violation['component']} "
                        f"delta={worst_violation['delta']:.3f} > "
                        f"bound={worst_violation['bound']:.3f}."
                    )
                else:
                    audit_line = (
                        "  realizability audit: PASS (all one-step displacements of "
                        "all state components stayed within training-derived bounds)."
                    )
                audit_payload = {
                    "available": True,
                    "plan_max_deltas": plan_max_deltas,
                    "train_bounds": train_bounds,
                    "checked_components": sorted(train_bounds),
                    "violations": violations,
                }

        summary = (
            f"validate_with_cem: start=traj{s_t}/frame{s_f}, "
            f"goal=traj{g_t}/frame{g_f}, horizon={horizon}\n"
            f"  initial_state: {payload['initial_state']}\n"
            f"  target_state:  {payload['target_state']}\n"
            f"  CEM action_bounds ({bounds_source}): "
            f"low={[f'{x:.3f}' for x in resolved_low]}, "
            f"high={[f'{x:.3f}' for x in resolved_high]}\n"
            f"{cem_line}\n"
            f"  loss trajectory: {loss_preview}\n"
            f"  state trajectory:\n{state_preview}\n"
            f"  summary: initial_loss={initial_loss:.4f}, "
            f"final_loss={final_loss:.4f}, min_loss={min_loss:.4f}\n"
            + "\n".join(self._format_goal_reach_lines(goal_reach))
        )
        if not goal_reach.get("available"):
            summary += self._format_goal_reach_unavailable_fail(goal_reach)
        elif stuck_completely:
            summary += (
                "\n  FAIL: CEM made essentially no progress — min_loss is ≥90% of "
                "initial_loss. Either your update(a) does not respond to action, "
                "or terminal_cost's gradient points away from any state CEM can "
                "reach. Re-examine both."
            )
        elif state_stuck:
            summary += self._format_goal_reach_fail(
                goal_reach, initial_loss, min_loss
            )
        elif goal_reach.get("reached") is False:
            summary += (
                "\n  ADVISORY: structured goal-reach remained incomplete even "
                "though the rollout was not classified as stuck. Do not treat "
                "this pair as evidence of full target reach."
            )
        elif final_loss > min_loss * 1.2:
            summary += (
                "\n  NOTE: final_loss > min_loss across the trajectory — CEM "
                "found a low-cost point mid-rollout but the horizon overshot it."
            )
        summary += "\n" + audit_line

        self._log_critic_tool_call(
            tool_name="validate_with_cem",
            args={
                "start": f"({s_t},{s_f})",
                "goal": f"({g_t},{g_f})",
                "horizon": horizon,
                "cem_iters": cem_iters,
                "cem_population": cem_population,
                "action_bounds_source": bounds_source,
            },
            result_summary=summary[:500],
            extra_files={
                "losses.json": _json.dumps(losses),
                "states.json": _json.dumps(states),
                "actions.json": _json.dumps(payload["actions"]),
                "cem_history.json": _json.dumps(cem_history),
                "realizability_audit.json": _json.dumps(audit_payload, indent=2),
                "goal_reach.json": _json.dumps(goal_reach, indent=2, default=str),
            },
        )
        return summary

    def validate_with_cem_states(
        self,
        start_state: dict,
        goal_state: dict,
        horizon: int = 20,
        cem_iters: int = 10,
        cem_population: int = 200,
        action_low: list[float] | None = None,
        action_high: list[float] | None = None,
    ) -> str:
        """Self-validate CEM against INVENTED (not data-anchored) start/goal states.

        Use this when ``validate_with_cem`` cannot construct a stressful enough
        test case from the training data alone. The training trajectories may
        cover only a subset of the state space the planner will encounter at
        test time — e.g. a navigation dataset may show the agent only near
        certain regions, while at test time the agent could appear anywhere
        the scene physically supports. This tool lets you construct OUT-OF-
        DISTRIBUTION scenarios on purpose, by passing state dictionaries
        directly.

        Bypasses ``fit()`` — sets ``sim.state = start_state`` and
        ``sim.target_state = goal_state`` directly. Pass dicts in the same
        schema your ``fit()`` produces (you wrote it, so you know the keys
        and types). Lists are coerced to numpy arrays automatically.

        Construct hard cases that probe your ``terminal_cost``:

        - Endpoints at opposite extremes of the state space, separated by
          regions the training data implies are impassable.
        - Pairs where the straight-line direction from start to goal points
          INTO an impassable region. This is where a naive Euclidean loss
          will trap CEM: the gradient pulls toward the goal, but the
          feasible trajectory must detour around the obstacle. If CEM
          plateaus at a state pressed against the obstacle, ``terminal_cost``
          needs to encode feasible-path distance, not direct distance.

        Args:
            start_state: dict matching your state schema (e.g. {"pos": [x, y]}).
            goal_state:  dict matching your state schema for the target.
            horizon, cem_iters, cem_population: CEM hyperparameters.
            action_low, action_high: optional explicit CEM bounds.

        Returns:
            Summary string with initial_loss, final_loss, per-step loss
            preview, per-step state preview, and a STUCK warning if CEM
            could not improve materially beyond initial_loss.
        """
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile

        try:
            horizon = int(horizon)
            cem_iters = int(cem_iters)
            cem_population = int(cem_population)
        except (TypeError, ValueError) as exc:
            err = f"[validate_with_cem_states] bad hyperparameter: {exc}"
            self._log_critic_tool_call(
                tool_name="validate_with_cem_states", args={}, result_summary=err,
            )
            return err

        if not isinstance(start_state, dict) or not isinstance(goal_state, dict):
            err = (
                "[validate_with_cem_states] start_state and goal_state must be dicts "
                "matching your sim's state schema (e.g. {'pos': [x, y]}). "
                f"Got start_state={type(start_state).__name__}, "
                f"goal_state={type(goal_state).__name__}."
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem_states", args={}, result_summary=err,
            )
            return err

        stacked = self._collect_training_actions()
        if stacked is None or stacked.ndim != 2 or stacked.shape[0] == 0:
            err = "[validate_with_cem_states] could not derive action bounds — no usable training trajectories found."
            self._log_critic_tool_call(
                tool_name="validate_with_cem_states", args={}, result_summary=err,
            )
            return err
        action_dim = stacked.shape[1]
        resolved_low, resolved_high, bounds_source = self._resolve_action_bounds(
            action_dim=action_dim,
            action_low=action_low,
            action_high=action_high,
        )
        if resolved_low is None or resolved_high is None:
            err = (
                "[validate_with_cem_states] could not resolve action bounds "
                f"({bounds_source})."
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem_states", args={}, result_summary=err,
            )
            return err

        runner_src = (
            "import sys, json, copy, numpy as np, importlib.util\n"
            "from PIL import Image\n"
            "from vdaworld.core.api import GeometryOnlyWorldAPI, WorldAPI\n"
            "from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase\n"
            "from vdaworld.core.cem import CEM\n"
            "args = json.loads(sys.argv[1])\n"
            "spec = importlib.util.spec_from_file_location('simulator_sandbox', args['sim_path'])\n"
            "module = importlib.util.module_from_spec(spec)\n"
            "module.SimulatorBase = SimulatorBase\n"
            "module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase\n"
            "module.WorldAPI = GeometryOnlyWorldAPI\n"
            "sys.modules['simulator_sandbox'] = module\n"
            "spec.loader.exec_module(module)\n"
            "cls = getattr(module, args['class_name'])\n"
            "api = None if args['no_api'] else GeometryOnlyWorldAPI(cache_dir=args.get('cache_dir'), api_calls_dir=args.get('api_calls_dir'))\n"
            "sim = cls(frame_size=tuple(args['frame_size']), api=api, fps=args['fps'])\n"
            "def _coerce(s):\n"
            "    out = {}\n"
            "    for k, v in s.items():\n"
            "        if isinstance(v, list):\n"
            "            out[k] = np.asarray(v, dtype=np.float64)\n"
            "        else:\n"
            "            out[k] = v\n"
            "    return out\n"
            "sim.state = _coerce(args['start_state'])\n"
            "sim.target_state = _coerce(args['goal_state'])\n"
            "def _state_repr(s):\n"
            "    if isinstance(s, dict):\n"
            "        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}\n"
            "    if isinstance(s, np.ndarray):\n"
            "        return s.tolist()\n"
            "    return str(s)\n"
            "initial_state = _state_repr(sim.state)\n"
            "target_state  = _state_repr(sim.target_state)\n"
            "initial_loss = float(sim.planning_objective())\n"
            "cem = CEM(sim, action_dim=args['action_dim'],\n"
            "          action_low=np.asarray(args['action_low']),\n"
            "          action_high=np.asarray(args['action_high']),\n"
            "          horizon=args['horizon'],\n"
            "          population=args['cem_population'],\n"
            "          iters=args['cem_iters'],\n"
            "          seed=args['seed'])\n"
            "plan = cem.plan()\n"
            "cem_history = [{'iter': e.iteration, 'best': e.best_cost, 'mean': e.mean_cost} for e in cem.history]\n"
            "losses = [initial_loss]\n"
            "states = [initial_state]\n"
            "for a in plan:\n"
            "    sim.update(np.asarray(a, dtype=np.float64))\n"
            "    losses.append(float(sim.planning_objective()))\n"
            "    states.append(_state_repr(sim.state))\n"
            "out = {\n"
            "  'initial_state': initial_state,\n"
            "  'target_state': target_state,\n"
            "  'initial_loss': initial_loss,\n"
            "  'final_loss': losses[-1],\n"
            "  'min_loss': float(min(losses)),\n"
            "  'losses': losses,\n"
            "  'states': states,\n"
            "  'cem_history': cem_history,\n"
            "  'actions': [list(map(float, a)) for a in plan.tolist()],\n"
            "}\n"
            "print(json.dumps(out))\n"
        )

        with _tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False
        ) as tmp:
            tmp.write(runner_src)
            runner_path = tmp.name

        try:
            runner_args = {
                "sim_path": self._sandbox_path,
                "class_name": self._simulator_class_name,
                "frame_size": list(self._frame_size),
                "fps": self._fps,
                "start_state": start_state,
                "goal_state": goal_state,
                "action_dim": int(action_dim),
                "action_low": resolved_low,
                "action_high": resolved_high,
                "horizon": horizon,
                "cem_iters": cem_iters,
                "cem_population": cem_population,
                "seed": 42,
                "no_api": self._no_api,
                "cache_dir": self._cache_dir,
                "api_calls_dir": self._world_api_log_dir,
            }
            result = _subprocess.run(
                [_sys.executable, runner_path, _json.dumps(runner_args)],
                capture_output=True,
                text=True,
                timeout=180,
                env=self._subprocess_env(),
            )
        finally:
            try:
                os.remove(runner_path)
            except OSError:
                pass

        if result.returncode != 0:
            err = (
                f"[validate_with_cem_states] runner exited with code "
                f"{result.returncode}\nstderr:\n{result.stderr[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem_states",
                args={"start_state": str(start_state)[:200], "goal_state": str(goal_state)[:200]},
                result_summary=err[:300],
                extra_files={"stderr.txt": result.stderr},
            )
            return err

        try:
            payload = _json.loads(result.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            err = (
                f"[validate_with_cem_states] could not parse runner output: {exc}\n"
                f"stdout:\n{result.stdout[:1500]}"
            )
            self._log_critic_tool_call(
                tool_name="validate_with_cem_states", args={}, result_summary=err[:300],
            )
            return err

        losses = payload["losses"]
        states = payload["states"]
        initial_loss = payload["initial_loss"]
        final_loss = payload["final_loss"]
        min_loss = payload["min_loss"]

        n = len(losses)
        sample_idxs = sorted(set([0, n // 4, n // 2, 3 * n // 4, n - 1]))
        loss_preview = ", ".join(f"t={i}:{losses[i]:.3f}" for i in sample_idxs)
        state_preview = "\n".join(f"    t={i}: {states[i]}" for i in sample_idxs)

        # Stuck criterion (a): loss floor near the initial loss.
        stuck_completely = (initial_loss > 0) and (min_loss >= 0.9 * initial_loss)

        cem_history = payload.get("cem_history", [])
        if cem_history:
            cem_line = (
                f"  CEM best_cost: first_iter={cem_history[0]['best']:.4f}, "
                f"last_iter={cem_history[-1]['best']:.4f}"
            )
        else:
            cem_line = "  (no CEM history available)"

        plan_max_deltas = self._max_step_deltas_from_states(
            states,
            component_periodicity=self._declared_state_periodicity(),
        )
        audit_payload: dict[str, Any]
        limits, limits_err = self._realizability_limits_cached(
            margin=self._gate_realizability_margin
        )
        # Stuck criterion (b), unit-free state-space goal-reach (see
        # validate_with_cem): cost falling is not enough — the target
        # components must actually approach their target values.
        goal_reach = self._assess_goal_reach(
            target_state=payload["target_state"],
            states=states,
            horizon=horizon,
            limits=limits,
        )
        state_stuck = bool(goal_reach.get("state_stuck"))
        if limits_err or limits is None:
            audit_payload = {
                "available": False,
                "error": limits_err or "unknown",
                "plan_max_deltas": plan_max_deltas,
            }
            audit_line = (
                f"  NOTE: realizability audit unavailable ({audit_payload['error']})."
            )
        else:
            train_bounds = dict(limits["bound_max"])
            violations, missing_components = self._realizability_violations(
                plan_max_deltas,
                train_bounds,
            )
            if missing_components:
                audit_payload = {
                    "available": False,
                    "error": (
                        "state components missing in planned trajectory: "
                        f"{missing_components}"
                    ),
                    "plan_max_deltas": plan_max_deltas,
                    "train_bounds": train_bounds,
                }
                audit_line = (
                    "  NOTE: realizability audit unavailable "
                    f"(state components missing in planned trajectory: {missing_components})."
                )
            else:
                if violations:
                    worst_violation = max(violations, key=lambda x: x["ratio"])
                    audit_line = (
                        "  FAIL: plan realizability audit — this CEM plan relies on "
                        "one-step state jumps larger than the training-derived bound. "
                        f"Worst component={worst_violation['component']} "
                        f"delta={worst_violation['delta']:.3f} > "
                        f"bound={worst_violation['bound']:.3f}."
                    )
                else:
                    audit_line = (
                        "  realizability audit: PASS (all one-step displacements of "
                        "all state components stayed within training-derived bounds)."
                    )
                audit_payload = {
                    "available": True,
                    "plan_max_deltas": plan_max_deltas,
                    "train_bounds": train_bounds,
                    "checked_components": sorted(train_bounds),
                    "violations": violations,
                }

        summary = (
            f"validate_with_cem_states (synthetic OOD test): horizon={horizon}\n"
            f"  initial_state: {payload['initial_state']}\n"
            f"  target_state:  {payload['target_state']}\n"
            f"  CEM action_bounds ({bounds_source}): "
            f"low={[f'{x:.3f}' for x in resolved_low]}, "
            f"high={[f'{x:.3f}' for x in resolved_high]}\n"
            f"{cem_line}\n"
            f"  loss trajectory: {loss_preview}\n"
            f"  state trajectory:\n{state_preview}\n"
            f"  summary: initial_loss={initial_loss:.4f}, "
            f"final_loss={final_loss:.4f}, min_loss={min_loss:.4f}\n"
            + "\n".join(self._format_goal_reach_lines(goal_reach))
        )
        if not goal_reach.get("available"):
            summary += self._format_goal_reach_unavailable_fail(goal_reach)
        elif stuck_completely:
            summary += (
                "\n  FAIL: CEM made essentially no progress — min_loss is ≥90% of "
                "initial_loss. Either your update(a) does not respond to action, "
                "or terminal_cost's gradient points away from any state CEM can "
                "reach. Re-examine both."
            )
        elif state_stuck:
            summary += self._format_goal_reach_fail(
                goal_reach, initial_loss, min_loss
            )
        elif goal_reach.get("reached") is False:
            summary += (
                "\n  ADVISORY: structured goal-reach remained incomplete even "
                "though the rollout was not classified as stuck. Do not treat "
                "this pair as evidence of full target reach."
            )
        elif final_loss > min_loss * 1.2:
            summary += (
                "\n  NOTE: final_loss > min_loss across the trajectory — CEM "
                "found a low-cost state mid-rollout but the horizon overshot it."
            )
        summary += "\n" + audit_line

        self._log_critic_tool_call(
            tool_name="validate_with_cem_states",
            args={
                "start_state": str(start_state)[:200],
                "goal_state": str(goal_state)[:200],
                "horizon": horizon,
                "cem_iters": cem_iters,
                "cem_population": cem_population,
                "action_bounds_source": bounds_source,
            },
            result_summary=summary[:500],
            extra_files={
                "losses.json": _json.dumps(losses),
                "states.json": _json.dumps(states),
                "actions.json": _json.dumps(payload["actions"]),
                "cem_history.json": _json.dumps(cem_history),
                "realizability_audit.json": _json.dumps(audit_payload, indent=2),
                "goal_reach.json": _json.dumps(goal_reach, indent=2, default=str),
            },
        )
        return summary

    def view_action(self, trajectory_index: int, step_index: int) -> str:
        """Return the action vector at one step of a training trajectory.

        Args:
            trajectory_index: Which trajectory.
            step_index:       Which step within the trajectory (0..n_steps-1).

        Returns:
            A JSON string with the action vector, or an error message.
        """
        try:
            t_idx = int(trajectory_index)
            s_idx = int(step_index)
            action = _dt.view_action(self._dataset_dir, t_idx, s_idx)
        except (FileNotFoundError, IndexError, ValueError) as exc:
            err = f"[view_action] {exc}"
            self._log_critic_tool_call(
                tool_name="view_action",
                args={
                    "trajectory_index": str(trajectory_index),
                    "step_index": str(step_index),
                },
                result_summary=err,
            )
            return err

        result = json.dumps(action.tolist())
        self._log_critic_tool_call(
            tool_name="view_action",
            args={"trajectory_index": t_idx, "step_index": s_idx},
            result_summary=f"Returned action of dim {action.shape[0]}: {result}",
        )
        return result
