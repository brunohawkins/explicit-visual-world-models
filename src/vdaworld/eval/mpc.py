"""Test-time control evaluation loops.

Two planning modes are supported:
    - ``mpc`` (default): receding-horizon control — at each cycle, re-fit on
      the current real observation, run CEM over a short ``horizon``, apply
      only ``apply_steps`` actions in the real session, then repeat.
    - ``open_loop``: fit once on ``(start_obs, goal_obs)``, plan once with CEM
      over ``plan_horizon`` actions, then execute that full plan without
      re-fit / re-plan (ablation / blind rollout).

Three evaluation protocol profiles are supported:
    - ``vda_mpc_default``: existing VDA-World default
      (``horizon=5``, ``apply_steps=2``, ``max_episode_steps=50``).
    - ``lewm_matched_low_level``: low-level control budget matched to LeWM's
      ``horizon=5`` x ``action_block=5`` setup
      (``horizon=25``, ``apply_steps=25``, CEM population/iters/top-k 300/30/30).
    - ``lewm_aligned_n50_v1``: frozen N=50 release profile using the same
      low-level budget plus immutable CEM 300/30/top-30/scale-1 settings.

The loop is test-env agnostic: it talks only to a session object exposing
`start(seed) / step(action_sim) / close()`. Each session owns its own
coordinate conversion and returns observations already at the simulator's frame
size, so this file never needs to know which benchmark it is driving.
"""

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from vdaworld.core.cem import CEM
from vdaworld.core.simulator import (
    ActionConditionedSimulatorBase,
    SimulatorBase,
)
from vdaworld.eval.profiles import LEWM_ALIGNED_N50_V1

EVAL_PROTOCOL_VDA_DEFAULT = "vda_mpc_default"
EVAL_PROTOCOL_LEWM_MATCHED = "lewm_matched_low_level"
EVAL_PROTOCOL_LEWM_ALIGNED_N50 = LEWM_ALIGNED_N50_V1.profile_id
SUPPORTED_EVAL_PROTOCOLS = {
    EVAL_PROTOCOL_VDA_DEFAULT,
    EVAL_PROTOCOL_LEWM_MATCHED,
    EVAL_PROTOCOL_LEWM_ALIGNED_N50,
}


def load_simulator_class(simulator_path: Path, class_name: str):
    """Load a `GeneratedSimulator`-style class from a standalone .py file.

    Mirrors the pattern in `core/action_conditioned_sandbox_runner.py` so the
    base-class names match what the VLM-generated code expects.
    """
    spec = importlib.util.spec_from_file_location(
        "simulator_loaded_for_mpc", str(simulator_path)
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create module spec from {simulator_path}")
    module = importlib.util.module_from_spec(spec)
    module.SimulatorBase = SimulatorBase
    module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase
    sys.modules["simulator_loaded_for_mpc"] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _image_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))


def _state_snapshot(state) -> Optional[dict]:
    """Deep-copyable snapshot of a sim state dict (arrays → lists)."""
    try:
        return {
            k: (v.tolist() if hasattr(v, "tolist") else v)
            for k, v in (state or {}).items()
        }
    except Exception:
        return None


def _fit_supports_prev_state(sim) -> bool:
    """True if the generated `fit` accepts a `prev_state` keyword.

    Receding-horizon refits erase latched state (e.g. attachment flags) that a
    single observation cannot re-establish. Generated simulators may opt in to
    state persistence by declaring `fit(self, image_A, image_B, prev_state=None)`.
    """
    try:
        params = inspect.signature(sim.fit).parameters
    except (TypeError, ValueError):
        return False
    if "prev_state" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _apply_eval_protocol(
    *,
    eval_protocol: str,
    horizon: int,
    apply_steps: int,
    max_episode_steps: int,
    cem_kwargs: dict[str, Any],
) -> tuple[int, int, int, dict[str, Any]]:
    if eval_protocol not in SUPPORTED_EVAL_PROTOCOLS:
        raise ValueError(
            f"unsupported eval_protocol={eval_protocol!r}; expected one of "
            f"{sorted(SUPPORTED_EVAL_PROTOCOLS)}"
        )

    cem_kwargs = dict(cem_kwargs)
    if eval_protocol == EVAL_PROTOCOL_LEWM_MATCHED:
        # LeWM plans 5 high-level blocks, each block containing 5 env actions.
        # Our explicit simulator consumes low-level actions, so the closest
        # controller-protocol match is a 25-step terminal CEM plan and applying
        # that whole low-level plan before replanning.
        horizon = 25
        apply_steps = 25
        max_episode_steps = 50
        cem_kwargs.setdefault("population", 300)
        cem_kwargs.setdefault("iters", 30)
        cem_kwargs.setdefault("elite_frac", 0.1)
    elif eval_protocol == EVAL_PROTOCOL_LEWM_ALIGNED_N50:
        # Frozen release profile. Unlike the legacy budget-match profile, these
        # values deliberately replace caller values so an N=50 comparison
        # cannot silently drift via command-line overrides.
        horizon = LEWM_ALIGNED_N50_V1.horizon
        apply_steps = LEWM_ALIGNED_N50_V1.apply_steps
        max_episode_steps = LEWM_ALIGNED_N50_V1.max_episode_steps
        cem_kwargs.update(
            {
                "population": LEWM_ALIGNED_N50_V1.cem_population,
                "iters": LEWM_ALIGNED_N50_V1.cem_iters,
                "elite_frac": LEWM_ALIGNED_N50_V1.cem_elite_frac,
                "init_std_scale": LEWM_ALIGNED_N50_V1.cem_init_std_scale,
                "use_stage_cost": False,
                "action_smoothness_weight": 0.0,
                "num_action_knots": None,
            }
        )

    return horizon, apply_steps, max_episode_steps, cem_kwargs


def _resolve_p2_diagnostics(
    *,
    eval_protocol: str,
    propagate_prev_state: bool,
    stop_on_primary_success: bool,
) -> tuple[bool, bool]:
    """Resolve diagnostic toggles, preserving the frozen release protocol."""
    if eval_protocol == EVAL_PROTOCOL_LEWM_ALIGNED_N50:
        return True, False
    return bool(propagate_prev_state), bool(stop_on_primary_success)


def eval_mpc(
    sim_class,
    *,
    session: Any,
    action_low: np.ndarray,
    action_high: np.ndarray,
    horizon: int = 5,
    apply_steps: int = 2,
    max_episode_steps: int = 50,
    cem_kwargs: Optional[dict] = None,
    sim_frame_size: tuple[int, int] = (65, 65),
    sim_fps: int = 10,
    api=None,
    state_override_fn: Optional[Any] = None,
    seed: Optional[int] = None,
    plan_mode: str = "mpc",
    plan_horizon: int = 50,
    eval_protocol: str = EVAL_PROTOCOL_VDA_DEFAULT,
    start_goal_source: str = "session",
    num_eval: Optional[int] = None,
    propagate_prev_state: bool = True,
    stop_on_primary_success: bool = False,
) -> dict[str, Any]:
    """Run test-time control evaluation against a test-env ``session``.

    `session` must expose `start(seed)`, `step(action_sim)`, `close()`, each of
    `start`/`step` returning a dict with keys: `image` (sim-frame obs), optional
    `image_raw` (env-native obs for video), `goal_image`/`goal_image_raw`,
    `info` (env ground-truth dict, used only for diagnostics), `reward`, `done`,
    `truncated`.

    `action_low`/`action_high` are the CEM bounds in the **simulator's** action
    space.

    `propagate_prev_state=False` disables latent-state carry-over between MPC
    refits. `stop_on_primary_success=True` is a diagnostic early-stop triggered
    by a post-step ``info["primary_success"]`` value.

    Returns generic keys only; benchmark-specific summaries belong in callers.
    """
    if action_low is None or action_high is None:
        raise ValueError("action_low/action_high are required")
    action_low = np.asarray(action_low, dtype=np.float64)
    action_high = np.asarray(action_high, dtype=np.float64)

    if plan_mode not in {"open_loop", "mpc"}:
        raise ValueError(
            f"unsupported plan_mode={plan_mode!r}; expected 'open_loop' or 'mpc'"
        )
    if plan_horizon <= 0:
        raise ValueError(f"plan_horizon must be > 0; got {plan_horizon}")

    horizon, apply_steps, max_episode_steps, cem_kwargs = _apply_eval_protocol(
        eval_protocol=eval_protocol,
        horizon=int(horizon),
        apply_steps=int(apply_steps),
        max_episode_steps=int(max_episode_steps),
        cem_kwargs=dict(cem_kwargs or {}),
    )
    propagate_prev_state, stop_on_primary_success = _resolve_p2_diagnostics(
        eval_protocol=eval_protocol,
        propagate_prev_state=propagate_prev_state,
        stop_on_primary_success=stop_on_primary_success,
    )
    cem_kwargs.setdefault("action_dim", int(action_low.shape[0]))
    cem_kwargs.setdefault("action_low", action_low)
    cem_kwargs.setdefault("action_high", action_high)
    if plan_mode == "mpc":
        cem_kwargs.setdefault("horizon", horizon)
    cem_population = int(cem_kwargs.get("population", 200))
    cem_iters = int(cem_kwargs.get("iters", 10))
    cem_elite_frac = float(cem_kwargs.get("elite_frac", 0.1))
    cem_init_std_scale = float(cem_kwargs.get("init_std_scale", 0.5))
    cem_use_stage_cost = bool(cem_kwargs.get("use_stage_cost", False))
    cem_action_smoothness_weight = float(
        cem_kwargs.get("action_smoothness_weight", 0.0)
    )
    cem_num_action_knots = cem_kwargs.get("num_action_knots")
    cem_seed = cem_kwargs.get("seed")
    cem_topk = max(1, int(round(cem_population * cem_elite_frac)))

    obs_raw_list: list[np.ndarray] = []
    obs_sim_list: list[np.ndarray] = []
    sim_predicted: list[np.ndarray] = []
    sim_predicted_rollout: list[np.ndarray] = []
    cem_histories: list[list] = []
    actions_applied: list[np.ndarray] = []
    rewards: list[float] = []
    gt_infos: list[dict] = []
    sim_fit_states: list[Any] = []
    sim_target_states: list[Any] = []
    sim_loss_each_step: list[float] = []
    observation_env_steps: list[int] = []
    observation_mpc_cycles: list[int | None] = []
    observation_events: list[str] = []
    action_mpc_cycles: list[int] = []
    action_apply_indices: list[int] = []
    execution_obs_raw: list[np.ndarray] = []
    execution_sim_predicted: list[np.ndarray | None] = []
    execution_env_steps: list[int] = []
    execution_mpc_cycles: list[int | None] = []
    execution_events: list[str] = []
    execution_action_indices: list[int | None] = []
    execution_sim_losses: list[float] = []
    done = False
    truncated = False
    steps_applied = 0
    primary_success_observed = False
    stop_requested = False
    stop_reason: str | None = None
    evaluation_t0 = time.perf_counter()
    session_start_seconds = 0.0
    fit_seconds: list[float] = []
    cem_seconds: list[float] = []
    environment_step_seconds: list[float] = []

    try:
        session_t0 = time.perf_counter()
        current = session.start(seed)
        session_start_seconds = time.perf_counter() - session_t0
        goal_sim = current["goal_image"]
        goal_raw = current.get("goal_image_raw", goal_sim)
        gt_infos.append(dict(current.get("info", {})))

        # Match P1 construction exactly. Previously P2 omitted fps and silently
        # depended on each generated class's default (often 10, sometimes the
        # base-class 30), making identified dynamics run at a different rate.
        sim = sim_class(frame_size=sim_frame_size, api=api, fps=int(sim_fps))

        if plan_mode == "mpc":
            fit_takes_prev_state = _fit_supports_prev_state(sim)
            prev_state: Any = None
            while (
                steps_applied < max_episode_steps
                and not done
                and not truncated
                and not stop_requested
            ):
                cycle_idx = len(cem_histories)
                obs_sim = current["image"]
                obs_raw = current.get("image_raw", obs_sim)
                obs_sim_list.append(obs_sim)
                obs_raw_list.append(obs_raw)
                observation_env_steps.append(int(steps_applied))
                observation_mpc_cycles.append(int(cycle_idx))
                observation_events.append("fit_cem_replan")

                # WorldAPI is fit-scoped. Reattach it only while parsing the
                # new observation, then remove it before objective/CEM/update
                # so real P2 enforces the same rollout contract as the P1
                # runtime-safety check.
                sim.api = api
                fit_t0 = time.perf_counter()
                if fit_takes_prev_state:
                    sim.fit(
                        obs_sim,
                        goal_sim,
                        prev_state=prev_state if propagate_prev_state else None,
                    )
                else:
                    sim.fit(obs_sim, goal_sim)
                fit_seconds.append(time.perf_counter() - fit_t0)
                if state_override_fn is not None:
                    state_override_fn(sim, current.get("info", {}))
                sim.api = None
                fitted_render = sim.render_frame()
                sim_predicted.append(fitted_render)
                sim_fit_states.append(_state_snapshot(sim.state))
                sim_target_states.append(_state_snapshot(sim.target_state))
                try:
                    fitted_loss = float(sim.planning_objective())
                except Exception:
                    fitted_loss = float("nan")
                sim_loss_each_step.append(fitted_loss)

                # Detailed video trace. Keep this separate from fit-boundary
                # observations so evaluation metrics retain their old meaning.
                execution_obs_raw.append(obs_raw)
                execution_sim_predicted.append(fitted_render.copy())
                execution_env_steps.append(int(steps_applied))
                execution_mpc_cycles.append(int(cycle_idx))
                execution_events.append("fit_cem_replan")
                execution_action_indices.append(None)
                execution_sim_losses.append(fitted_loss)

                # Snapshot the freshly fitted belief before CEM rollouts mutate it.
                try:
                    fitted_state = copy.deepcopy(sim.state)
                except Exception:
                    fitted_state = None

                cem = CEM(sim, **cem_kwargs)
                cem_t0 = time.perf_counter()
                plan = cem.plan()  # (horizon, action_dim)
                cem_seconds.append(time.perf_counter() - cem_t0)
                cem_histories.append(cem.history)

                n_to_apply = min(
                    apply_steps, len(plan), max_episode_steps - steps_applied
                )

                # Roll the fitted simulator through the exact action prefix in
                # parallel with the real session. This is for visual diagnosis;
                # CEM scoring and real-environment execution are unchanged.
                prediction_available = fitted_state is not None
                if prediction_available:
                    try:
                        sim.state = copy.deepcopy(fitted_state)
                    except Exception:
                        prediction_available = False

                for k in range(n_to_apply):
                    a = plan[k]
                    predicted_frame = None
                    predicted_loss = float("nan")
                    if prediction_available:
                        try:
                            sim.update(a)
                            predicted_frame = sim.render_frame().copy()
                            predicted_loss = float(sim.planning_objective())
                        except Exception:
                            prediction_available = False
                            predicted_frame = None
                            predicted_loss = float("nan")

                    env_t0 = time.perf_counter()
                    current = session.step(a)
                    environment_step_seconds.append(time.perf_counter() - env_t0)
                    actions_applied.append(np.asarray(a, dtype=np.float64).copy())
                    action_mpc_cycles.append(int(cycle_idx))
                    action_apply_indices.append(int(k))
                    rewards.append(float(current.get("reward", float("nan"))))
                    steps_applied += 1
                    step_info = dict(current.get("info", {}))
                    gt_infos.append(step_info)
                    step_primary_success = bool(
                        step_info.get("primary_success", False)
                    )
                    primary_success_observed = (
                        primary_success_observed or step_primary_success
                    )
                    done = bool(current.get("done", False))
                    truncated = bool(current.get("truncated", False))

                    execution_obs_raw.append(
                        current.get("image_raw", current["image"])
                    )
                    execution_sim_predicted.append(predicted_frame)
                    execution_env_steps.append(int(steps_applied))
                    execution_mpc_cycles.append(int(cycle_idx))
                    execution_events.append("env_update")
                    execution_action_indices.append(int(k))
                    execution_sim_losses.append(predicted_loss)

                    if stop_on_primary_success and step_primary_success:
                        stop_requested = True
                        stop_reason = "primary_success"
                        break
                    if done or truncated:
                        stop_reason = "done" if done else "truncated"
                        break

                # Propagate the simulator's belief through the actions that were
                # actually applied, so the next refit can preserve latched state
                # (e.g. attachment flags) that one image cannot re-establish.
                prev_state = None
                if (
                    propagate_prev_state
                    and fit_takes_prev_state
                    and prediction_available
                ):
                    try:
                        prev_state = copy.deepcopy(sim.state)
                    except Exception:
                        prev_state = None

            # Final observation after the last applied action.
            final_sim = current["image"]
            final_raw = current.get("image_raw", final_sim)
            obs_sim_list.append(final_sim)
            obs_raw_list.append(final_raw)
            observation_env_steps.append(int(steps_applied))
            observation_mpc_cycles.append(None)
            observation_events.append("final_observation")

            # Duplicate the last action-level observation with an explicit
            # final status frame so success/failure colouring remains visible.
            execution_obs_raw.append(final_raw)
            execution_sim_predicted.append(
                execution_sim_predicted[-1].copy()
                if execution_sim_predicted
                and execution_sim_predicted[-1] is not None
                else None
            )
            execution_env_steps.append(int(steps_applied))
            execution_mpc_cycles.append(None)
            execution_events.append("final_observation")
            execution_action_indices.append(None)
            execution_sim_losses.append(
                execution_sim_losses[-1]
                if execution_sim_losses
                else float("nan")
            )
        else:
            obs0 = current["image"]
            obs0_raw = current.get("image_raw", obs0)
            obs_sim_list.append(obs0)
            obs_raw_list.append(obs0_raw)
            observation_env_steps.append(0)
            observation_mpc_cycles.append(0)
            observation_events.append("fit_open_loop_plan")

            sim.api = api
            fit_t0 = time.perf_counter()
            sim.fit(obs0, goal_sim)
            fit_seconds.append(time.perf_counter() - fit_t0)
            if state_override_fn is not None:
                state_override_fn(sim, current.get("info", {}))
            sim.api = None
            sim_fit_states.append(_state_snapshot(sim.state))
            sim_target_states.append(_state_snapshot(sim.target_state))
            try:
                sim_loss_each_step.append(float(sim.planning_objective()))
            except Exception:
                sim_loss_each_step.append(float("nan"))

            open_loop_cem_kwargs = dict(cem_kwargs)
            open_loop_cem_kwargs["horizon"] = int(plan_horizon)
            cem = CEM(sim, **open_loop_cem_kwargs)
            cem_t0 = time.perf_counter()
            plan = cem.plan()  # (plan_horizon, action_dim)
            cem_seconds.append(time.perf_counter() - cem_t0)
            cem_histories.append(cem.history)

            # Re-fit to the same start state before rendering a forward rollout.
            sim.api = api
            sim.fit(obs0, goal_sim)
            if state_override_fn is not None:
                state_override_fn(sim, current.get("info", {}))
            sim.api = None

            sim_predicted_rollout = [sim.render_frame().copy()]
            for a in plan:
                sim.update(a)
                sim_predicted_rollout.append(sim.render_frame().copy())

            # Reuse the rollout frames as the sim-side visualization in open-loop mode.
            sim_predicted = list(sim_predicted_rollout)

            for a in plan:
                env_t0 = time.perf_counter()
                current = session.step(a)
                environment_step_seconds.append(time.perf_counter() - env_t0)
                actions_applied.append(np.asarray(a, dtype=np.float64).copy())
                action_mpc_cycles.append(0)
                action_apply_indices.append(int(steps_applied))
                rewards.append(float(current.get("reward", float("nan"))))
                steps_applied += 1

                obs_sim = current["image"]
                obs_raw = current.get("image_raw", obs_sim)
                obs_sim_list.append(obs_sim)
                obs_raw_list.append(obs_raw)
                step_info = dict(current.get("info", {}))
                gt_infos.append(step_info)
                step_primary_success = bool(step_info.get("primary_success", False))
                primary_success_observed = (
                    primary_success_observed or step_primary_success
                )
                observation_env_steps.append(int(steps_applied))
                observation_mpc_cycles.append(0)
                observation_events.append("env_update")

                done = bool(current.get("done", False))
                truncated = bool(current.get("truncated", False))
                if stop_on_primary_success and step_primary_success:
                    stop_requested = True
                    stop_reason = "primary_success"
                    break
                if done or truncated:
                    stop_reason = "done" if done else "truncated"
                    break

            final_sim = obs_sim_list[-1]
            final_raw = obs_raw_list[-1]

        terminal_distance = _image_distance(final_raw, goal_raw)
    finally:
        session.close()

    actions_arr = (
        np.asarray(actions_applied, dtype=np.float64)
        if actions_applied
        else np.zeros((0, int(action_low.shape[0])), dtype=np.float64)
    )

    total_seconds = time.perf_counter() - evaluation_t0
    effective_horizon = int(horizon if plan_mode == "mpc" else plan_horizon)
    estimated_simulator_transitions = (
        len(cem_histories) * cem_population * cem_iters * effective_horizon
    )
    result = {
        # Generic, env-agnostic keys.
        "sim_predicted_frames": sim_predicted,
        "sim_predicted_rollout": sim_predicted_rollout,
        "observations_sim": obs_sim_list,
        "observations_raw": obs_raw_list,
        "goal_image_sim": goal_sim,
        "goal_image_raw": goal_raw,
        "actions_applied": actions_arr,
        "rewards": rewards,
        "cem_histories": cem_histories,
        "terminal_distance": terminal_distance,
        "n_steps": steps_applied,
        "done": done,
        "truncated": truncated,
        "primary_success_observed": primary_success_observed,
        "stop_reason": stop_reason,
        "gt_infos": gt_infos,
        "sim_fit_states": sim_fit_states,
        "sim_target_states": sim_target_states,
        "sim_loss_each_step": sim_loss_each_step,
        "sim_fps": int(sim_fps),
        "world_api_fit_enabled": api is not None,
        "world_api_rollout_enabled": False,
        "observation_env_steps": observation_env_steps,
        "observation_mpc_cycles": observation_mpc_cycles,
        "observation_events": observation_events,
        "action_mpc_cycles": action_mpc_cycles,
        "action_apply_indices": action_apply_indices,
        "execution_observations_raw": execution_obs_raw,
        "execution_sim_predicted_frames": execution_sim_predicted,
        "execution_env_steps": execution_env_steps,
        "execution_mpc_cycles": execution_mpc_cycles,
        "execution_events": execution_events,
        "execution_action_indices": execution_action_indices,
        "execution_sim_losses": execution_sim_losses,
        "plan_mode": plan_mode,
        "plan_horizon": int(plan_horizon),
        "eval_protocol": eval_protocol,
        "start_goal_source": start_goal_source,
        "num_eval": num_eval,
        "horizon": int(horizon),
        "apply_steps": int(apply_steps),
        "max_episode_steps": int(max_episode_steps),
        "effective_plan_steps": int(horizon if plan_mode == "mpc" else plan_horizon),
        "effective_apply_steps": int(
            apply_steps if plan_mode == "mpc" else min(plan_horizon, steps_applied)
        ),
        "cem_population": cem_population,
        "cem_iters": cem_iters,
        "cem_elite_frac": cem_elite_frac,
        "cem_topk": cem_topk,
        "cem_init_std_scale": cem_init_std_scale,
        "cem_use_stage_cost": cem_use_stage_cost,
        "cem_action_smoothness_weight": cem_action_smoothness_weight,
        "cem_num_action_knots": cem_num_action_knots,
        "cem_seed": cem_seed,
        "propagate_prev_state": propagate_prev_state,
        "stop_on_primary_success": stop_on_primary_success,
        "timing": {
            "total_seconds": total_seconds,
            "session_start_seconds": session_start_seconds,
            "fit_seconds": fit_seconds,
            "fit_total_seconds": float(sum(fit_seconds)),
            "cem_seconds": cem_seconds,
            "cem_total_seconds": float(sum(cem_seconds)),
            "environment_step_seconds": environment_step_seconds,
            "environment_total_seconds": float(sum(environment_step_seconds)),
        },
        "estimated_simulator_transitions": int(estimated_simulator_transitions),
    }
    return result
