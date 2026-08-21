#!/usr/bin/env python
"""Shared P2 smoke: evaluate generated simulators with CEM/MPC in real envs.

Default protocol is the original VDA receding-horizon MPC. Use
``--eval-protocol lewm_matched_low_level`` for the LeWorldModel-aligned
controller budget, or ``lewm_aligned_n50_v1`` for the frozen canonical
50-case manifests and isolated SWM reference environments.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

from vdaworld.core.api import GeometryOnlyWorldAPI
from vdaworld.core.local_2d import (
    LOCAL_2D_TOOLBOX_COMPATIBLE_VERSIONS,
    LOCAL_2D_TOOLBOX_VERSION,
)
from vdaworld.eval.mpc import (
    EVAL_PROTOCOL_LEWM_ALIGNED_N50,
    EVAL_PROTOCOL_VDA_DEFAULT,
    SUPPORTED_EVAL_PROTOCOLS,
    eval_mpc,
    load_simulator_class,
)
from vdaworld.eval.manifest import (
    case_for_seed,
    load_manifest,
    manifest_sha256,
    validate_manifest,
)
from vdaworld.eval.profiles import (
    LEWM_ALIGNED_MANIFEST_SHA256,
    LEWM_ALIGNED_N50_V1,
)
from vdaworld.eval.pusht_adapter import sim_action_bounds as pusht_action_bounds
from vdaworld.eval.pusht_env import GymPushTSession
from vdaworld.eval.reacher_env import (
    REACHER_ACTION_HIGH,
    REACHER_ACTION_LOW,
    ReacherDMControlSession,
)
from vdaworld.eval.swm_reference_server_launcher import SWMReferenceServer
from vdaworld.eval.swm_reference_session import SWMReferenceSession
from vdaworld.eval.swm_two_room_env import (
    SWM_TWO_ROOM_ACTION_HIGH,
    SWM_TWO_ROOM_ACTION_LOW,
    SWM_TWO_ROOM_FRAME_SIZE,
    SWMTwoRoomSession,
)

P2_CASE_SCHEMA_VERSION = "p2_case_metrics_v2"
P2_SUMMARY_SCHEMA_VERSION = "p2_summary_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

REPO_ROOT = Path(__file__).resolve().parents[1]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str | None:
    if value is None:
        return None
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _git_command(*args: str) -> bytes | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def _pipeline_git_provenance() -> dict[str, Any]:
    commit_bytes = _git_command("rev-parse", "HEAD")
    status_bytes = _git_command("status", "--porcelain=v1", "--untracked-files=all")
    diff_bytes = _git_command("diff", "--binary", "HEAD")
    commit = None if commit_bytes is None else commit_bytes.decode().strip() or None
    dirty = None if status_bytes is None else bool(status_bytes.strip())
    diff_sha256 = (
        None
        if diff_bytes is None or not diff_bytes
        else hashlib.sha256(diff_bytes).hexdigest()
    )
    return {
        "commit": commit,
        "dirty": dirty,
        "diff_sha256": diff_sha256,
    }


def _resize(frame: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    w, h = int(shape[0]), int(shape[1])
    arr = np.asarray(frame, dtype=np.uint8)
    if arr.shape[:2] == (h, w):
        return arr
    return np.asarray(
        Image.fromarray(arr).resize((w, h), Image.NEAREST), dtype=np.uint8
    )


def _draw_border(
    arr: np.ndarray, color: tuple[int, int, int], width: int = 4
) -> np.ndarray:
    out = np.asarray(arr, dtype=np.uint8).copy()
    if out.size == 0:
        return out
    out[:width, :, :] = color
    out[-width:, :, :] = color
    out[:, :width, :] = color
    out[:, -width:, :] = color
    return out


def _text_panel(lines: list[str], width: int, height: int = 58) -> np.ndarray:
    img = Image.new("RGB", (width, height), (20, 20, 24))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 6
    for line in lines:
        draw.text((8, y), line, fill=(235, 235, 235), font=font)
        y += 16
    return np.asarray(img, dtype=np.uint8)


def _label_panel(frame: np.ndarray, title: str) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.uint8)
    label_h = 20
    label = Image.new("RGB", (arr.shape[1], label_h), (35, 35, 38))
    draw = ImageDraw.Draw(label)
    draw.text((6, 4), title, fill=(240, 240, 240), font=ImageFont.load_default())
    return np.concatenate([np.asarray(label, dtype=np.uint8), arr], axis=0)


def _blank_frame(shape: tuple[int, int, int], text: str = "no fit frame") -> np.ndarray:
    h, w = shape[:2]
    img = Image.new("RGB", (w, h), (35, 35, 35))
    draw = ImageDraw.Draw(img)
    draw.text((8, 8), text, fill=(210, 210, 210), font=ImageFont.load_default())
    return np.asarray(img, dtype=np.uint8)


def _write_comparison_video(
    result: dict[str, Any], out_path: Path, fps: int = 4
) -> None:
    detailed_trace = bool(result.get("execution_observations_raw"))
    obs = (
        result.get("execution_observations_raw")
        if detailed_trace
        else result.get("observations_raw")
    ) or []
    goal = result.get("goal_image_raw")
    sim_fit = (
        result.get("execution_sim_predicted_frames")
        if detailed_trace
        else result.get("sim_predicted_frames")
    ) or []
    env_steps = (
        result.get("execution_env_steps")
        if detailed_trace
        else result.get("observation_env_steps")
    ) or []
    cycles = (
        result.get("execution_mpc_cycles")
        if detailed_trace
        else result.get("observation_mpc_cycles")
    ) or []
    events = (
        result.get("execution_events")
        if detailed_trace
        else result.get("observation_events")
    ) or []
    action_indices = (
        result.get("execution_action_indices") if detailed_trace else []
    ) or []
    apply_steps = int(result.get("apply_steps") or 0)
    action_cycles = result.get("action_mpc_cycles") or []
    sim_losses = (
        result.get("execution_sim_losses")
        if detailed_trace
        else result.get("sim_loss_each_step")
    ) or []
    if not obs:
        return
    reacher_status = None
    metric_infos = _metric_infos(result)
    if any(
        isinstance(info, dict) and info.get("benchmark") == "reacher"
        for info in metric_infos
    ):
        qpos_metrics = _reacher_qpos_metrics(result)
        target_metrics = _reacher_distance_metrics(result)
        reacher_status = {
            "paper_ever": bool(
                qpos_metrics.get("reacher_paper_qpos_ever_success")
            ),
            "paper_final": bool(
                qpos_metrics.get("reacher_paper_qpos_final_success")
            ),
            "target_ever": bool(
                target_metrics.get("reacher_target_ever_success")
            ),
            "target_final": bool(
                target_metrics.get("reacher_target_final_success")
            ),
        }
    frames = []
    for i, frame in enumerate(obs):
        frame = np.asarray(frame, dtype=np.uint8)
        goal_resized = (
            _resize(goal, (frame.shape[1], frame.shape[0]))
            if goal is not None
            else _blank_frame(frame.shape, "no goal")
        )
        predicted = sim_fit[i] if i < len(sim_fit) else None
        sim_frame = (
            _resize(predicted, (frame.shape[1], frame.shape[0]))
            if predicted is not None
            else _blank_frame(frame.shape, "prediction unavailable")
        )

        event = events[i] if i < len(events) else "observation"
        cycle = cycles[i] if i < len(cycles) else None
        env_step = env_steps[i] if i < len(env_steps) else None
        n_actions_this_cycle = (
            sum(1 for c in action_cycles if c == cycle) if cycle is not None else 0
        )
        loss = sim_losses[i] if i < len(sim_losses) else None

        if event == "fit_cem_replan":
            border = (50, 140, 255)  # blue: fit/replan
            phase = "FIT -> CEM PLAN"
        elif event == "env_update":
            border = (255, 120, 40)  # orange/red: env update
            action_idx = action_indices[i] if i < len(action_indices) else None
            phase = (
                f"APPLY ACTION {int(action_idx) + 1}/{n_actions_this_cycle}"
                if action_idx is not None
                else "ENV UPDATE"
            )
        elif event == "final_observation":
            if reacher_status is not None:
                if reacher_status["paper_ever"]:
                    border = (80, 220, 110)
                    outcome = "PAPER QPOS SUCCESS"
                elif reacher_status["target_ever"]:
                    border = (235, 180, 45)
                    outcome = "ENDPOINT ONLY — WRONG/UNMATCHED JOINT STATE"
                else:
                    border = (230, 80, 80)
                    outcome = "TARGET NOT REACHED"
                phase = f"FINAL OBSERVATION | {outcome}"
            else:
                border = (80, 220, 110) if result.get("done") else (230, 80, 80)
                phase = "FINAL OBSERVATION"
        else:
            border = (180, 180, 180)
            phase = event.upper()

        if event == "fit_cem_replan":
            sim_label = "sim.render_frame() immediately after fit"
        elif event == "env_update":
            sim_label = "sim prediction after the same action"
        elif event == "final_observation":
            sim_label = "last sim prediction (no final refit)"
        else:
            sim_label = "sim.render_frame()"

        panels = [
            _label_panel(_draw_border(frame, border), "real/env observation"),
            _label_panel(
                _draw_border(sim_frame, border),
                sim_label,
            ),
            _label_panel(goal_resized, "image_B target passed to fit"),
        ]
        body = np.concatenate(panels, axis=1)
        header_lines = [
            f"{phase} | mpc_cycle={cycle if cycle is not None else 'n/a'} | env_step={env_step if env_step is not None else 'n/a'} / {result.get('n_steps')}",
            f"planned_horizon={result.get('horizon')} apply_steps={apply_steps} actions_applied_this_cycle={n_actions_this_cycle}",
            f"sim_loss_after_fit={loss if loss is not None else 'n/a'}",
        ]
        if event == "final_observation" and reacher_status is not None:
            header_lines.append(
                "Reacher metrics: "
                f"qpos-ever={reacher_status['paper_ever']} "
                f"qpos-final={reacher_status['paper_final']} "
                f"endpoint-ever={reacher_status['target_ever']} "
                f"endpoint-final={reacher_status['target_final']}"
            )
        header = _text_panel(header_lines, width=body.shape[1])
        frames.append(np.concatenate([header, body], axis=0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_path, frames, fps=fps, codec="libx264")


def _plot_cem_costs(result: dict[str, Any], out_path: Path) -> None:
    histories = result.get("cem_histories") or []
    if not histories:
        return
    best = []
    elite = []
    labels = []
    for cycle_idx, history in enumerate(histories):
        if not history:
            continue
        last = history[-1]
        best.append(float(getattr(last, "best_cost", np.nan)))
        elite.append(float(getattr(last, "elite_mean_cost", np.nan)))
        labels.append(cycle_idx)
    if not labels:
        return
    plt.figure(figsize=(6, 4))
    plt.plot(labels, best, marker="o", label="cycle final best")
    plt.plot(labels, elite, marker="s", label="cycle final elite mean")
    plt.xlabel("replan cycle")
    plt.ylabel("terminal planner cost")
    plt.title("CEM cost by replan cycle")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()


def _save_frame(path: Path, frame: np.ndarray | None) -> None:
    if frame is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(frame, dtype=np.uint8)).save(path)


def _default_sim_path(benchmark: str) -> Path | None:
    mapping = {
        "pusht": REPO_ROOT / "viz_output" / "smoke_p1_pusht",
        "two_room": REPO_ROOT / "viz_output" / "smoke_p1_two_room",
        "reacher": REPO_ROOT / "viz_output" / "smoke_p1_reacher",
        "cube": REPO_ROOT / "viz_output" / "smoke_p1_cube",
    }
    root = mapping.get(benchmark)
    if root is None or not root.exists():
        return None
    sims = sorted(
        root.glob("*/run/simulator_gen.py"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return sims[0] if sims else None


def _make_session(benchmark: str, frame_size: tuple[int, int], max_steps: int, args):
    manifest_case = getattr(args, "_current_manifest_case", None)
    if args.eval_backend == "swm_reference":
        if benchmark not in {"pusht", "two_room_swm", "reacher", "cube"}:
            raise ValueError(
                "swm_reference backend supports pusht, two_room_swm, "
                "reacher, and cube"
            )
        if manifest_case is None:
            raise ValueError("swm_reference backend requires a manifest case")
        session = SWMReferenceSession(
            benchmark=benchmark,
            manifest_case=manifest_case,
            hdf5_path=args.swm_hdf5_path,
            sim_frame_size=frame_size,
            server=args._swm_server,
            strip_green_target_from_sim=args.pusht_strip_green_target,
            task_protocol=args.task_success_protocol,
        )
        low, high = session.action_bounds()
        return session, low, high
    if benchmark == "pusht":
        return GymPushTSession(
            sim_frame_size=frame_size,
            max_episode_steps=max_steps,
            goal_image_mode=args.pusht_goal_image_mode,
            manifest_case=manifest_case,
            strip_green_target_from_sim=args.pusht_strip_green_target,
        )
    if benchmark == "two_room":
        try:
            from vdaworld.eval.pldm_session import (
                PLDM_ACTION_HIGH,
                PLDM_ACTION_LOW,
                PLDMSession,
            )
        except Exception as exc:  # pragma: no cover - depends on optional PLDM files.
            raise RuntimeError(
                "two_room P2 requires PLDM session support, but it could not be imported. "
                "Check vdaworld.eval.pldm_server / pldm_envs installation."
            ) from exc
        return (
            PLDMSession(sim_frame_size=frame_size, manifest_case=manifest_case),
            PLDM_ACTION_LOW,
            PLDM_ACTION_HIGH,
        )
    if benchmark == "two_room_swm":
        return (
            SWMTwoRoomSession(
                sim_frame_size=frame_size,
                max_episode_steps=max_steps,
                manifest_case=manifest_case,
            ),
            SWM_TWO_ROOM_ACTION_LOW,
            SWM_TWO_ROOM_ACTION_HIGH,
        )
    if benchmark == "reacher":
        return (
            ReacherDMControlSession(
                sim_frame_size=frame_size,
                goal_image_mode=args.reacher_goal_image_mode,
                manifest_case=manifest_case,
            ),
            REACHER_ACTION_LOW,
            REACHER_ACTION_HIGH,
        )
    if benchmark == "cube":
        from vdaworld.eval.ogbench_cube_env import (
            CUBE_ACTION_HIGH,
            CUBE_ACTION_LOW,
            OGBenchCubeSession,
        )

        return (
            OGBenchCubeSession(
                sim_frame_size=frame_size,
                manifest_case=manifest_case,
            ),
            CUBE_ACTION_LOW,
            CUBE_ACTION_HIGH,
        )
    raise ValueError(f"unsupported benchmark={benchmark!r}")


def _session_and_bounds(
    benchmark: str, frame_size: tuple[int, int], max_steps: int, args
):
    made = _make_session(benchmark, frame_size, max_steps, args)
    if isinstance(made, tuple):
        return made
    if benchmark == "pusht":
        low, high = pusht_action_bounds(frame_size)
        return made, low, high
    raise AssertionError(f"bounds missing for benchmark={benchmark}")


def _frame_size_for(benchmark: str) -> tuple[int, int]:
    return {
        "pusht": (224, 224),
        "two_room": (65, 65),
        "two_room_swm": SWM_TWO_ROOM_FRAME_SIZE,
        "reacher": (224, 224),
        "cube": (200, 200),
    }[benchmark]


def _metric_infos(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return per-action diagnostics, excluding reset for SWM reference runs."""

    infos = [info for info in result.get("gt_infos", []) if isinstance(info, dict)]
    is_reference = any(
        isinstance(info.get("backend"), dict)
        and info["backend"].get("implementation") == "stable_worldmodel"
        for info in infos
    )
    if is_reference:
        return [
            info
            for info in infos
            if isinstance(info.get("env_step"), (int, float))
            and int(info["env_step"]) > 0
        ]
    return infos


def _pusht_coverage_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract Push-T coverage progress from ground-truth env diagnostics."""
    coverages = [
        float(info["coverage"])
        for info in _metric_infos(result)
        if isinstance(info, dict) and isinstance(info.get("coverage"), (int, float))
    ]
    if not coverages:
        return {
            "pusht_final_coverage": None,
            "pusht_max_coverage": None,
            "pusht_success_at_0_50": False,
            "pusht_success_at_0_75": False,
            "pusht_success_at_0_90": False,
            "pusht_success_at_0_95": False,
        }
    max_coverage = max(coverages)
    return {
        "pusht_final_coverage": coverages[-1],
        "pusht_max_coverage": max_coverage,
        "pusht_success_at_0_50": bool(max_coverage >= 0.50),
        "pusht_success_at_0_75": bool(max_coverage >= 0.75),
        "pusht_success_at_0_90": bool(max_coverage >= 0.90),
        "pusht_success_at_0_95": bool(max_coverage >= 0.95),
    }


def _angle_diff_radians(a: float, b: float) -> float:
    diff = abs((float(a) % (2.0 * np.pi)) - (float(b) % (2.0 * np.pi)))
    return float(min(diff, 2.0 * np.pi - diff))


def _pusht_pose_coverage(block_pose: np.ndarray, goal_pose: np.ndarray) -> float:
    """T-shape overlap for SWM's hard-coded scale-30 block geometry."""
    scale = 30.0
    crossbar = Polygon(
        [(-2 * scale, -scale), (2 * scale, -scale), (2 * scale, 0), (-2 * scale, 0)]
    )
    stem = Polygon(
        [
            (-scale / 2, -scale),
            (-scale / 2, -4 * scale),
            (scale / 2, -4 * scale),
            (scale / 2, -scale),
        ]
    )
    local = unary_union([crossbar, stem])

    def transform(pose: np.ndarray):
        geometry = affinity.rotate(
            local, float(pose[2]), origin=(0.0, 0.0), use_radians=True
        )
        return affinity.translate(
            geometry, xoff=float(pose[0]), yoff=float(pose[1])
        )

    current = transform(np.asarray(block_pose, dtype=np.float64).reshape(3))
    target = transform(np.asarray(goal_pose, dtype=np.float64).reshape(3))
    return float(
        np.clip(current.intersection(target).area / max(target.area, 1e-9), 0.0, 1.0)
    )


def _pusht_pose_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract LeWM/DINO-style Push-T pose-threshold diagnostics."""
    infos = [
        info
        for info in _metric_infos(result)
        if isinstance(info, dict)
        and isinstance(info.get("block_pose"), (list, tuple, np.ndarray))
        and isinstance(info.get("goal_pose"), (list, tuple, np.ndarray))
    ]
    if not infos:
        return {
            "pusht_paper_block_final_success": None,
            "pusht_paper_block_ever_success": None,
            "pusht_paper_block_final_pos_diff": None,
            "pusht_paper_block_final_angle_diff": None,
            "pusht_paper_full_state_final_success": None,
            "pusht_paper_full_state_ever_success": None,
            "pusht_paper_full_state_final_pos_diff": None,
            "pusht_pose_final_coverage": None,
            "pusht_pose_max_coverage": None,
            "pusht_pose_ever_ge_0_95": None,
        }

    def block_stats(info: dict[str, Any]) -> tuple[float, float, bool]:
        block = np.asarray(info["block_pose"], dtype=np.float64).reshape(3)
        goal = np.asarray(info["goal_pose"], dtype=np.float64).reshape(3)
        pos_diff = float(np.linalg.norm(block[:2] - goal[:2]))
        angle_diff = _angle_diff_radians(block[2], goal[2])
        return pos_diff, angle_diff, bool(pos_diff < 20.0 and angle_diff < np.pi / 9.0)

    goal_agent = None
    for info in infos:
        goal_state = info.get("goal_state")
        if isinstance(goal_state, dict) and goal_state.get("pos_agent") is not None:
            goal_agent = np.asarray(goal_state["pos_agent"], dtype=np.float64).reshape(
                2
            )
            break
        manifest_goal = info.get("manifest_goal_state")
        if (
            isinstance(manifest_goal, dict)
            and manifest_goal.get("pos_agent") is not None
        ):
            goal_agent = np.asarray(
                manifest_goal["pos_agent"], dtype=np.float64
            ).reshape(2)
            break
        goal_render_info = info.get("goal_render_info")
        if (
            isinstance(goal_render_info, dict)
            and goal_render_info.get("pos_agent") is not None
        ):
            goal_agent = np.asarray(
                goal_render_info["pos_agent"], dtype=np.float64
            ).reshape(2)
            break

    def full_state_stats(info: dict[str, Any]) -> tuple[float | None, bool | None]:
        if goal_agent is None or info.get("pos_agent") is None:
            return None, None
        agent = np.asarray(info["pos_agent"], dtype=np.float64).reshape(2)
        block_pos_diff, angle_diff, _ = block_stats(info)
        pos_diff = float(np.sqrt(np.sum((agent - goal_agent) ** 2) + block_pos_diff**2))
        return pos_diff, bool(pos_diff < 20.0 and angle_diff < np.pi / 9.0)

    final_block_pos, final_angle, final_block_success = block_stats(infos[-1])
    final_full_pos, final_full_success = full_state_stats(infos[-1])
    full_successes = [full_state_stats(info)[1] for info in infos]
    pose_coverages = [
        _pusht_pose_coverage(info["block_pose"], info["goal_pose"]) for info in infos
    ]
    return {
        "pusht_paper_block_final_success": final_block_success,
        "pusht_paper_block_ever_success": any(block_stats(info)[2] for info in infos),
        "pusht_paper_block_final_pos_diff": final_block_pos,
        "pusht_paper_block_final_angle_diff": final_angle,
        "pusht_paper_full_state_final_success": final_full_success,
        "pusht_paper_full_state_ever_success": (
            any(value for value in full_successes if value is not None)
            if any(value is not None for value in full_successes)
            else None
        ),
        "pusht_paper_full_state_final_pos_diff": final_full_pos,
        "pusht_pose_final_coverage": pose_coverages[-1],
        "pusht_pose_max_coverage": max(pose_coverages),
        "pusht_pose_ever_ge_0_95": any(value >= 0.95 for value in pose_coverages),
    }


def _two_room_paper_metrics(result: dict[str, Any]) -> dict[str, Any]:
    infos = [
        info
        for info in result.get("gt_infos", [])
        if isinstance(info, dict)
        and isinstance(info.get("dot_position"), (list, tuple, np.ndarray))
    ]
    if not infos:
        return {
            "two_room_final_distance_to_target": None,
            "two_room_min_distance_to_target": None,
            "two_room_lewm_scaled16_success": None,
            "two_room_dinowm_wall45_success": None,
            "two_room_pldm_calibrated_final_distance_to_true_target": None,
            "two_room_pldm_calibrated_min_distance_to_true_target": None,
            "two_room_pldm_calibrated_success": None,
            "two_room_pldm_calibrated_target_source": None,
        }

    def distances_to(target_key: str) -> list[float]:
        vals = []
        for info in infos:
            if not isinstance(info.get(target_key), (list, tuple, np.ndarray)):
                continue
            vals.append(
                float(
                    np.linalg.norm(
                        np.asarray(info["dot_position"], dtype=np.float64).reshape(2)
                        - np.asarray(info[target_key], dtype=np.float64).reshape(2)
                    )
                )
            )
        return vals

    # Legacy diagnostic: the target dot was recovered from the rendered goal image
    # in PLDM's 64px coordinates. Kept for continuity with older result files.
    image_centroid_distances = distances_to("target_dot_position")
    image_centroid_min = (
        min(image_centroid_distances) if image_centroid_distances else None
    )

    # Calibrated interim comparison: use the true PLDM env target when the server
    # exposes it, with the 64px-scale threshold corresponding to LeWM's 16px
    # threshold on a 224px render (16 / 3.5 ~= 4.57; DINO-WM uses 4.5).
    true_target_distances = distances_to("target_position_xy")
    calibrated_source = "target_position_xy" if true_target_distances else None
    if not true_target_distances:
        true_target_distances = image_centroid_distances
        calibrated_source = (
            "target_dot_position_fallback" if true_target_distances else None
        )
    calibrated_min = min(true_target_distances) if true_target_distances else None

    return {
        "two_room_final_distance_to_target": image_centroid_distances[-1]
        if image_centroid_distances
        else None,
        "two_room_min_distance_to_target": image_centroid_min,
        "two_room_lewm_scaled16_success": bool(
            image_centroid_min is not None and image_centroid_min < 16.0
        ),
        "two_room_lewm_scaled16_ever_success": bool(
            image_centroid_min is not None and image_centroid_min < 16.0
        ),
        "two_room_lewm_scaled16_final_success": bool(
            image_centroid_distances and image_centroid_distances[-1] < 16.0
        ),
        "two_room_dinowm_wall45_success": bool(
            image_centroid_min is not None and image_centroid_min < 4.5
        ),
        "two_room_dinowm_wall45_ever_success": bool(
            image_centroid_min is not None and image_centroid_min < 4.5
        ),
        "two_room_dinowm_wall45_final_success": bool(
            image_centroid_distances and image_centroid_distances[-1] < 4.5
        ),
        "two_room_pldm_calibrated_final_distance_to_true_target": (
            true_target_distances[-1] if true_target_distances else None
        ),
        "two_room_pldm_calibrated_min_distance_to_true_target": calibrated_min,
        "two_room_pldm_calibrated_success": bool(
            calibrated_min is not None and calibrated_min < 4.5
        ),
        "two_room_pldm_calibrated_ever_success": bool(
            calibrated_min is not None and calibrated_min < 4.5
        ),
        "two_room_pldm_calibrated_final_success": bool(
            true_target_distances and true_target_distances[-1] < 4.5
        ),
        "two_room_pldm_calibrated_target_source": calibrated_source,
    }


def _swm_two_room_metrics(result: dict[str, Any]) -> dict[str, Any]:
    infos = [
        info
        for info in result.get("gt_infos", [])
        if isinstance(info, dict)
        and isinstance(info.get("distance_to_target"), (int, float))
    ]
    if not infos:
        return {
            "swm_two_room_final_distance_to_target": None,
            "swm_two_room_min_distance_to_target": None,
            "swm_two_room_success_radius": None,
            "swm_two_room_success": None,
        }
    distances = [float(info["distance_to_target"]) for info in infos]
    radius = next(
        (
            float(info["success_radius"])
            for info in infos
            if isinstance(info.get("success_radius"), (int, float))
        ),
        16.0,
    )
    return {
        "swm_two_room_final_distance_to_target": distances[-1],
        "swm_two_room_min_distance_to_target": min(distances),
        "swm_two_room_success_radius": radius,
        "swm_two_room_success": bool(min(distances) < radius),
    }


def _reacher_distance_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract Reacher target-distance progress from ground-truth diagnostics."""
    distances = [
        float(info["distance_to_target"])
        for info in _metric_infos(result)
        if isinstance(info, dict)
        and isinstance(info.get("distance_to_target"), (int, float))
    ]
    if not distances:
        return {
            "reacher_final_distance_to_target": None,
            "reacher_min_distance_to_target": None,
            "reacher_target_radius": None,
            "reacher_target_final_success": None,
            "reacher_target_ever_success": None,
        }
    radius = None
    for info in _metric_infos(result):
        if isinstance(info, dict) and isinstance(
            info.get("target_radius"), (int, float)
        ):
            radius = float(info["target_radius"])
            break
    final_distance = distances[-1]
    min_distance = min(distances)
    return {
        "reacher_final_distance_to_target": final_distance,
        "reacher_min_distance_to_target": min_distance,
        "reacher_target_radius": radius,
        "reacher_target_final_success": (
            bool(final_distance <= radius) if radius is not None else None
        ),
        "reacher_target_ever_success": (
            bool(min_distance <= radius) if radius is not None else None
        ),
    }


def _reacher_qpos_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract LeWM Reacher qpos-match diagnostics when qpos is recorded."""
    infos = [
        info
        for info in _metric_infos(result)
        if isinstance(info, dict)
        and isinstance(info.get("qpos"), (list, tuple, np.ndarray))
        and isinstance(info.get("goal_qpos"), (list, tuple, np.ndarray))
    ]
    if not infos:
        return {
            "reacher_paper_qpos_final_success": None,
            "reacher_paper_qpos_ever_success": None,
            "reacher_paper_qpos_final_max_abs_error": None,
            "reacher_paper_qpos_min_max_abs_error": None,
        }

    def max_abs_error(info: dict[str, Any]) -> float:
        qpos = np.asarray(info["qpos"], dtype=np.float64)
        goal_qpos = np.asarray(info["goal_qpos"], dtype=np.float64)
        return float(np.max(np.abs(qpos - goal_qpos)))

    errors = [max_abs_error(info) for info in infos]
    return {
        "reacher_paper_qpos_final_success": bool(errors[-1] < 0.05),
        "reacher_paper_qpos_ever_success": any(error < 0.05 for error in errors),
        "reacher_paper_qpos_final_max_abs_error": errors[-1],
        "reacher_paper_qpos_min_max_abs_error": min(errors),
    }


def _cube_position_metrics(result: dict[str, Any]) -> dict[str, Any]:
    """Extract OGBench/LeWM cube position-success diagnostics."""
    infos = [
        info
        for info in _metric_infos(result)
        if isinstance(info, dict)
        and isinstance(
            info.get("privileged/block_0_pos", info.get("cube_pos")),
            (list, tuple, np.ndarray),
        )
        and isinstance(info.get("target_pos"), (list, tuple, np.ndarray))
    ]
    if not infos:
        return {
            "cube_paper_final_success": None,
            "cube_paper_ever_success": None,
            "cube_paper_final_distance": None,
            "cube_paper_min_distance": None,
            "cube_paper_success_threshold": 0.04,
        }

    distances = [
        float(
            np.linalg.norm(
                np.asarray(
                    info.get("privileged/block_0_pos", info.get("cube_pos")),
                    dtype=np.float64,
                )
                - np.asarray(info["target_pos"], dtype=np.float64)
            )
        )
        for info in infos
    ]
    successes = [
        bool(
            info.get(
                "primary_success",
                info.get("cube_position_success", info.get("success", False)),
            )
        )
        for info in infos
    ]
    threshold = 0.04
    return {
        "cube_paper_final_success": bool(successes[-1] or distances[-1] <= threshold),
        "cube_paper_ever_success": bool(any(successes) or min(distances) <= threshold),
        "cube_paper_final_distance": distances[-1],
        "cube_paper_min_distance": min(distances),
        "cube_paper_success_threshold": threshold,
    }


def _cem_kwargs_for_seed(args, seed: int) -> dict[str, Any]:
    """Resolve planner settings for one environment seed."""
    cem_kwargs: dict[str, Any] = {}
    if args.cem_population is not None:
        cem_kwargs["population"] = int(args.cem_population)
    if args.cem_iters is not None:
        cem_kwargs["iters"] = int(args.cem_iters)
    if args.elite_frac is not None:
        cem_kwargs["elite_frac"] = float(args.elite_frac)
    if args.init_std_scale is not None:
        cem_kwargs["init_std_scale"] = float(args.init_std_scale)
    cem_kwargs["use_stage_cost"] = bool(args.cem_use_stage_cost)
    cem_kwargs["action_smoothness_weight"] = float(
        args.cem_action_smoothness_weight
    )
    cem_kwargs["num_action_knots"] = args.cem_action_knots
    cem_kwargs["seed"] = seed if args.planner_seed is None else int(args.planner_seed)
    return cem_kwargs


def _p2_diagnostic_kwargs(args) -> dict[str, bool]:
    """Resolve diagnostic toggles passed into eval_mpc."""
    return {
        "propagate_prev_state": bool(args.propagate_prev_state),
        "stop_on_primary_success": bool(args.stop_on_primary_success),
    }


def _native_success(result: dict[str, Any]) -> bool:
    """Retain post-step primary success even when diagnostic stopping is early."""
    return bool(
        result.get("done", False) or result.get("primary_success_observed", False)
    )


def _run_one_seed(
    *,
    sim_class,
    benchmark: str,
    seed: int,
    out_dir: Path,
    args,
) -> dict[str, Any]:
    args._current_manifest_case = (
        case_for_seed(args.manifest, seed) if args.manifest is not None else None
    )
    frame_size = tuple(args._sim_frame_size)
    session, action_low, action_high = _session_and_bounds(
        benchmark, frame_size, args.max_steps, args
    )
    seed_dir = out_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    api = None
    if not args.no_world_api:
        cache_dir = args.world_api_cache_dir or (out_dir / "world_api_cache")
        api = GeometryOnlyWorldAPI(
            cache_dir=str(cache_dir),
            api_calls_dir=str(seed_dir / "api_calls"),
        )
    cem_kwargs = _cem_kwargs_for_seed(args, seed)

    result = eval_mpc(
        sim_class,
        session=session,
        action_low=action_low,
        action_high=action_high,
        horizon=args.horizon,
        apply_steps=args.apply_steps,
        max_episode_steps=args.max_steps,
        cem_kwargs=cem_kwargs,
        sim_frame_size=frame_size,
        sim_fps=args.sim_fps,
        seed=seed,
        plan_mode=args.plan_mode,
        plan_horizon=args.plan_horizon,
        eval_protocol=args.eval_protocol,
        start_goal_source=args.start_goal_source,
        num_eval=len(args.seed_values),
        api=api,
        **_p2_diagnostic_kwargs(args),
    )

    observations = result.get("observations_raw") or []
    _write_comparison_video(result, seed_dir / "mpc.mp4")
    _plot_cem_costs(result, seed_dir / "cem_costs.png")
    _save_frame(seed_dir / "start_true.png", observations[0] if observations else None)
    _save_frame(seed_dir / "final_true.png", observations[-1] if observations else None)
    _save_frame(seed_dir / "image_B_target.png", result.get("goal_image_raw"))
    _save_frame(seed_dir / "goal_vs_final_true.png", result.get("goal_image_raw"))

    metrics = {
        k: v
        for k, v in result.items()
        if k
        not in {
            "sim_predicted_frames",
            "sim_predicted_rollout",
            "observations_sim",
            "observations_raw",
            "execution_observations_raw",
            "execution_sim_predicted_frames",
            "goal_image_sim",
            "goal_image_raw",
        }
    }
    metrics.update(
        {
            "schema_version": P2_CASE_SCHEMA_VERSION,
            "case_status": "completed",
            "benchmark": benchmark,
            "seed": seed,
            "planner_seed": int(cem_kwargs["seed"]),
            "pipeline_git_commit": args._pipeline_git_commit,
            "pipeline_git_dirty": args._pipeline_git_dirty,
            "pipeline_git_diff_sha256": args._pipeline_git_diff_sha256,
            "simulator_sha256": args._simulator_sha256,
            "active_local_2d_toolbox_version": LOCAL_2D_TOOLBOX_VERSION,
            "p1_local_2d_toolbox_version": (
                None
                if args._runtime_contract is None
                else args._runtime_contract.get("local_2d_toolbox_version")
            ),
            "start_image_sha256": _array_sha256(
                observations[0] if observations else None
            ),
            "goal_image_sha256": _array_sha256(result.get("goal_image_raw")),
            "manifest_path": str(args.manifest_path) if args.manifest_path else None,
            "manifest_sha256": getattr(args, "_manifest_sha256", None),
            "manifest_case_id": (
                None
                if args._current_manifest_case is None
                else args._current_manifest_case.get("case_id")
            ),
            "eval_backend": args.eval_backend,
            "sim_fps": int(args.sim_fps),
            "world_api_fit_enabled": not bool(args.no_world_api),
            "world_api_rollout_enabled": False,
            "runtime_contract_path": (
                str(args._runtime_contract_path)
                if args._runtime_contract_path is not None
                else None
            ),
            "runtime_contract": args._runtime_contract,
            "resolved_eval_profile": (
                LEWM_ALIGNED_N50_V1.as_dict()
                if args.eval_protocol == EVAL_PROTOCOL_LEWM_ALIGNED_N50
                else None
            ),
            "pusht_goal_image_mode": args.pusht_goal_image_mode
            if benchmark == "pusht"
            else None,
            "pusht_strip_green_target": bool(args.pusht_strip_green_target)
            if benchmark == "pusht"
            else None,
            "reacher_goal_image_mode": args.reacher_goal_image_mode
            if benchmark == "reacher"
            else None,
            "task_success_protocol": args.task_success_protocol,
            "objective_intervention": args._objective_intervention,
            "objective_intervention_sha256": args._objective_intervention_sha256,
            "is_success": _native_success(result),
            "actions_applied": result.get("actions_applied"),
            "cem_histories": [
                [_jsonable(entry) for entry in history]
                for history in result.get("cem_histories", [])
            ],
        }
    )
    if benchmark == "pusht":
        metrics.update(_pusht_coverage_metrics(result))
        metrics.update(_pusht_pose_metrics(result))
        metrics["native_is_success"] = bool(metrics["is_success"])
        if metrics.get("pusht_max_coverage") is not None:
            metrics["is_success"] = bool(metrics["pusht_success_at_0_95"])
            metrics["success_metric"] = "pusht_success_at_0_95"
    if benchmark == "two_room":
        metrics.update(_two_room_paper_metrics(result))
    if benchmark == "two_room_swm":
        metrics.update(_swm_two_room_metrics(result))
    if benchmark == "reacher":
        metrics.update(_reacher_distance_metrics(result))
        metrics.update(_reacher_qpos_metrics(result))
        metrics["native_is_success"] = bool(metrics["is_success"])
        if metrics.get("reacher_target_ever_success") is not None:
            metrics["is_success"] = bool(
                metrics["reacher_target_ever_success"]
            )
            metrics["success_metric"] = "reacher_target_ever_success"
    if benchmark == "cube":
        metrics.update(_cube_position_metrics(result))
    if benchmark == "pusht":
        metrics["paper_primary_success"] = metrics.get(
            "pusht_paper_full_state_ever_success"
        )
        metrics["task_primary_success"] = metrics.get("pusht_success_at_0_95")
    elif benchmark == "two_room_swm":
        metrics["paper_primary_success"] = metrics.get("swm_two_room_success")
        metrics["task_primary_success"] = None
    elif benchmark == "reacher":
        metrics["paper_primary_success"] = metrics.get(
            "reacher_paper_qpos_ever_success"
        )
        metrics["task_primary_success"] = metrics.get(
            "reacher_target_ever_success"
        )
    elif benchmark == "cube":
        metrics["paper_primary_success"] = metrics.get("cube_paper_ever_success")
        metrics["task_primary_success"] = None
    metrics_tmp = seed_dir / "metrics.json.tmp"
    metrics_tmp.write_text(
        json.dumps(_jsonable(metrics), indent=2),
        encoding="utf-8",
    )
    metrics_tmp.replace(seed_dir / "metrics.json")
    print(
        f"[p2] seed={seed} done={result.get('done')} "
        f"steps={result.get('n_steps')} terminal_distance={result.get('terminal_distance'):.3f}"
    )
    return metrics


def _summary(
    metrics: list[dict[str, Any]], *, args, sim_path: Path, out_dir: Path
) -> dict[str, Any]:
    distances = [
        float(m["terminal_distance"])
        for m in metrics
        if m.get("terminal_distance") is not None
    ]
    n_success = sum(1 for m in metrics if m.get("done") or m.get("is_success"))
    native_n_success = sum(
        1
        for m in metrics
        if m.get("done")
        or m.get("native_is_success", m.get("is_success", False))
    )
    completed = [m for m in metrics if m.get("case_status") == "completed"]
    failed = [m for m in metrics if m.get("case_status") != "completed"]
    base = completed[0] if completed else metrics[0] if metrics else {}
    n_eval = len(metrics)
    summary = {
        "schema_version": P2_SUMMARY_SCHEMA_VERSION,
        "benchmark": args.benchmark,
        "sim_path": str(sim_path),
        "out_dir": str(out_dir),
        "seeds": args.seed_values,
        "n_requested": len(args.seed_values),
        "n_completed": len(completed),
        "n_failed": len(failed),
        "failed_cases": [
            {
                "seed": item.get("seed"),
                "manifest_case_id": item.get("manifest_case_id"),
                "error": item.get("error"),
            }
            for item in failed
        ],
        "n_success": n_success,
        "success_rate": n_success / n_eval if metrics else 0.0,
        "gate_conditional_success_rate": (
            n_success / len(completed) if completed else None
        ),
        "end_to_end_case_success_rate": (
            n_success / len(args.seed_values) if args.seed_values else None
        ),
        "native_n_success": native_n_success,
        "native_success_rate": native_n_success / n_eval if metrics else 0.0,
        "terminal_distance_mean": float(np.mean(distances)) if distances else None,
        "terminal_distance_worst": float(np.max(distances)) if distances else None,
        "terminal_distance_best": float(np.min(distances)) if distances else None,
        "pipeline_git_commit": getattr(args, "_pipeline_git_commit", None),
        "pipeline_git_dirty": getattr(args, "_pipeline_git_dirty", None),
        "pipeline_git_diff_sha256": getattr(
            args, "_pipeline_git_diff_sha256", None
        ),
        "simulator_sha256": getattr(args, "_simulator_sha256", None),
        "active_local_2d_toolbox_version": LOCAL_2D_TOOLBOX_VERSION,
        "p1_local_2d_toolbox_version": (
            None
            if getattr(args, "_runtime_contract", None) is None
            else args._runtime_contract.get("local_2d_toolbox_version")
        ),
        "planner_seed": getattr(args, "planner_seed", None),
        "planner_seed_source": (
            "explicit"
            if getattr(args, "planner_seed", None) is not None
            else "environment_seed"
        ),
        "planner_seeds": [
            int(m["planner_seed"])
            for m in metrics
            if m.get("planner_seed") is not None
        ],
        "timing": {
            "case_total_seconds": float(
                sum(
                    float((m.get("timing") or {}).get("total_seconds", 0.0))
                    for m in completed
                )
            ),
            "cem_total_seconds": float(
                sum(
                    float((m.get("timing") or {}).get("cem_total_seconds", 0.0))
                    for m in completed
                )
            ),
            "fit_total_seconds": float(
                sum(
                    float((m.get("timing") or {}).get("fit_total_seconds", 0.0))
                    for m in completed
                )
            ),
            "environment_total_seconds": float(
                sum(
                    float(
                        (m.get("timing") or {}).get(
                            "environment_total_seconds", 0.0
                        )
                    )
                    for m in completed
                )
            ),
        },
        "estimated_simulator_transitions": int(
            sum(int(m.get("estimated_simulator_transitions", 0)) for m in completed)
        ),
    }
    for key in [
        "eval_protocol",
        "start_goal_source",
        "num_eval",
        "plan_mode",
        "plan_horizon",
        "horizon",
        "apply_steps",
        "max_episode_steps",
        "effective_plan_steps",
        "effective_apply_steps",
        "cem_population",
        "cem_iters",
        "cem_elite_frac",
        "cem_topk",
        "cem_init_std_scale",
        "cem_use_stage_cost",
        "cem_action_smoothness_weight",
        "cem_num_action_knots",
        "propagate_prev_state",
        "stop_on_primary_success",
        "pusht_goal_image_mode",
        "pusht_strip_green_target",
        "reacher_goal_image_mode",
        "task_success_protocol",
        "objective_intervention",
        "objective_intervention_sha256",
        "manifest_path",
        "sim_fps",
        "world_api_fit_enabled",
        "world_api_rollout_enabled",
        "runtime_contract_path",
        "runtime_contract",
    ]:
        summary[key] = base.get(key)
    if args.benchmark == "pusht":
        max_coverages = [
            float(m["pusht_max_coverage"])
            for m in metrics
            if m.get("pusht_max_coverage") is not None
        ]
        final_coverages = [
            float(m["pusht_final_coverage"])
            for m in metrics
            if m.get("pusht_final_coverage") is not None
        ]
        pose_max_coverages = [
            float(m["pusht_pose_max_coverage"])
            for m in metrics
            if m.get("pusht_pose_max_coverage") is not None
        ]
        pose_final_coverages = [
            float(m["pusht_pose_final_coverage"])
            for m in metrics
            if m.get("pusht_pose_final_coverage") is not None
        ]
        summary.update(
            {
                "pusht_max_coverage_mean": float(np.mean(max_coverages))
                if max_coverages
                else None,
                "pusht_max_coverage_best": float(np.max(max_coverages))
                if max_coverages
                else None,
                "pusht_final_coverage_mean": float(np.mean(final_coverages))
                if final_coverages
                else None,
                "pusht_n_ge_0_50": sum(
                    1 for m in metrics if m.get("pusht_success_at_0_50")
                ),
                "pusht_n_ge_0_75": sum(
                    1 for m in metrics if m.get("pusht_success_at_0_75")
                ),
                "pusht_n_ge_0_90": sum(
                    1 for m in metrics if m.get("pusht_success_at_0_90")
                ),
                "pusht_n_ge_0_95": sum(
                    1 for m in metrics if m.get("pusht_success_at_0_95")
                ),
                "pusht_pose_max_coverage_mean": (
                    float(np.mean(pose_max_coverages))
                    if pose_max_coverages
                    else None
                ),
                "pusht_pose_max_coverage_best": (
                    float(np.max(pose_max_coverages))
                    if pose_max_coverages
                    else None
                ),
                "pusht_pose_final_coverage_mean": (
                    float(np.mean(pose_final_coverages))
                    if pose_final_coverages
                    else None
                ),
                "pusht_pose_n_ge_0_95": (
                    sum(1 for m in metrics if m.get("pusht_pose_ever_ge_0_95"))
                    if pose_max_coverages
                    else None
                ),
                "pusht_paper_block_final_n_success": sum(
                    1 for m in metrics if m.get("pusht_paper_block_final_success")
                ),
                "pusht_paper_block_ever_n_success": sum(
                    1 for m in metrics if m.get("pusht_paper_block_ever_success")
                ),
                "pusht_paper_full_state_final_n_success": (
                    sum(
                        1
                        for m in metrics
                        if m.get("pusht_paper_full_state_final_success")
                    )
                    if any(
                        m.get("pusht_paper_full_state_final_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "pusht_paper_full_state_ever_n_success": (
                    sum(
                        1
                        for m in metrics
                        if m.get("pusht_paper_full_state_ever_success")
                    )
                    if any(
                        m.get("pusht_paper_full_state_ever_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "paper_primary_metric": "pusht_paper_full_state_ever_success",
                "paper_primary_n_success": (
                    sum(
                        1
                        for m in metrics
                        if m.get("pusht_paper_full_state_ever_success")
                    )
                    if any(
                        m.get("pusht_paper_full_state_ever_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "task_primary_metric": "pusht_success_at_0_95",
                "task_primary_n_success": (
                    sum(1 for m in metrics if m.get("pusht_success_at_0_95"))
                    if max_coverages
                    else None
                ),
            }
        )
    if args.benchmark == "two_room":
        summary.update(
            {
                "two_room_lewm_scaled16_n_success": sum(
                    1 for m in metrics if m.get("two_room_lewm_scaled16_success")
                ),
                "two_room_dinowm_wall45_n_success": sum(
                    1 for m in metrics if m.get("two_room_dinowm_wall45_success")
                ),
                "two_room_pldm_calibrated_n_success": sum(
                    1 for m in metrics if m.get("two_room_pldm_calibrated_success")
                ),
                "two_room_min_distance_best": min(
                    (
                        float(m["two_room_min_distance_to_target"])
                        for m in metrics
                        if m.get("two_room_min_distance_to_target") is not None
                    ),
                    default=None,
                ),
                "two_room_pldm_calibrated_min_distance_best": min(
                    (
                        float(m["two_room_pldm_calibrated_min_distance_to_true_target"])
                        for m in metrics
                        if m.get("two_room_pldm_calibrated_min_distance_to_true_target")
                        is not None
                    ),
                    default=None,
                ),
                "two_room_pldm_calibrated_target_sources": sorted(
                    {
                        str(m.get("two_room_pldm_calibrated_target_source"))
                        for m in metrics
                        if m.get("two_room_pldm_calibrated_target_source") is not None
                    }
                ),
                "paper_primary_metric": "two_room_pldm_calibrated_ever_success",
                "paper_primary_n_success": sum(
                    1 for m in metrics if m.get("two_room_pldm_calibrated_ever_success")
                ),
            }
        )
    if args.benchmark == "two_room_swm":
        min_distances = [
            float(m["swm_two_room_min_distance_to_target"])
            for m in metrics
            if m.get("swm_two_room_min_distance_to_target") is not None
        ]
        summary.update(
            {
                "swm_two_room_n_success": sum(
                    1 for m in metrics if m.get("swm_two_room_success")
                ),
                "swm_two_room_min_distance_best": min(min_distances)
                if min_distances
                else None,
                "swm_two_room_min_distance_mean": float(np.mean(min_distances))
                if min_distances
                else None,
                "swm_two_room_success_radius": next(
                    (
                        float(m["swm_two_room_success_radius"])
                        for m in metrics
                        if m.get("swm_two_room_success_radius") is not None
                    ),
                    None,
                ),
                "paper_primary_metric": "swm_two_room_success",
                "paper_primary_n_success": sum(
                    1 for m in metrics if m.get("swm_two_room_success")
                ),
            }
        )
    if args.benchmark == "reacher":
        min_distances = [
            float(m["reacher_min_distance_to_target"])
            for m in metrics
            if m.get("reacher_min_distance_to_target") is not None
        ]
        final_distances = [
            float(m["reacher_final_distance_to_target"])
            for m in metrics
            if m.get("reacher_final_distance_to_target") is not None
        ]
        summary.update(
            {
                "reacher_min_distance_mean": float(np.mean(min_distances))
                if min_distances
                else None,
                "reacher_min_distance_best": float(np.min(min_distances))
                if min_distances
                else None,
                "reacher_final_distance_mean": float(np.mean(final_distances))
                if final_distances
                else None,
                "reacher_target_radius": base.get("reacher_target_radius"),
                "reacher_n_within_2x_radius": sum(
                    1
                    for m in metrics
                    if m.get("reacher_min_distance_to_target") is not None
                    and m.get("reacher_target_radius") is not None
                    and float(m["reacher_min_distance_to_target"])
                    <= 2.0 * float(m["reacher_target_radius"])
                ),
                "reacher_target_final_n_success": (
                    sum(1 for m in metrics if m.get("reacher_target_final_success"))
                    if any(
                        m.get("reacher_target_final_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "reacher_target_ever_n_success": (
                    sum(1 for m in metrics if m.get("reacher_target_ever_success"))
                    if any(
                        m.get("reacher_target_ever_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "reacher_paper_qpos_final_n_success": (
                    sum(1 for m in metrics if m.get("reacher_paper_qpos_final_success"))
                    if any(
                        m.get("reacher_paper_qpos_final_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "reacher_paper_qpos_ever_n_success": (
                    sum(1 for m in metrics if m.get("reacher_paper_qpos_ever_success"))
                    if any(
                        m.get("reacher_paper_qpos_ever_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "paper_primary_metric": "reacher_paper_qpos_ever_success",
                "paper_primary_n_success": (
                    sum(1 for m in metrics if m.get("reacher_paper_qpos_ever_success"))
                    if any(
                        m.get("reacher_paper_qpos_ever_success") is not None
                        for m in metrics
                    )
                    else None
                ),
                "reacher_paper_qpos_min_max_abs_error_best": min(
                    (
                        float(m["reacher_paper_qpos_min_max_abs_error"])
                        for m in metrics
                        if m.get("reacher_paper_qpos_min_max_abs_error") is not None
                    ),
                    default=None,
                ),
                "task_primary_metric": "reacher_target_ever_success",
                "task_primary_n_success": (
                    sum(1 for m in metrics if m.get("reacher_target_ever_success"))
                    if any(
                        m.get("reacher_target_ever_success") is not None
                        for m in metrics
                    )
                    else None
                ),
            }
        )
    if args.benchmark == "cube":
        min_distances = [
            float(m["cube_paper_min_distance"])
            for m in metrics
            if m.get("cube_paper_min_distance") is not None
        ]
        final_distances = [
            float(m["cube_paper_final_distance"])
            for m in metrics
            if m.get("cube_paper_final_distance") is not None
        ]
        summary.update(
            {
                "cube_paper_final_n_success": (
                    sum(1 for m in metrics if m.get("cube_paper_final_success"))
                    if any(
                        m.get("cube_paper_final_success") is not None for m in metrics
                    )
                    else None
                ),
                "cube_paper_ever_n_success": (
                    sum(1 for m in metrics if m.get("cube_paper_ever_success"))
                    if any(
                        m.get("cube_paper_ever_success") is not None for m in metrics
                    )
                    else None
                ),
                "cube_paper_min_distance_best": float(np.min(min_distances))
                if min_distances
                else None,
                "cube_paper_min_distance_mean": float(np.mean(min_distances))
                if min_distances
                else None,
                "cube_paper_final_distance_mean": float(np.mean(final_distances))
                if final_distances
                else None,
                "cube_paper_success_threshold": next(
                    (
                        float(m["cube_paper_success_threshold"])
                        for m in metrics
                        if m.get("cube_paper_success_threshold") is not None
                    ),
                    0.04,
                ),
                "paper_primary_metric": "cube_paper_ever_success",
                "paper_primary_n_success": (
                    sum(1 for m in metrics if m.get("cube_paper_ever_success"))
                    if any(
                        m.get("cube_paper_ever_success") is not None for m in metrics
                    )
                    else None
                ),
            }
        )
    if summary.get("paper_primary_n_success") is not None:
        summary["paper_primary_success_rate"] = (
            float(summary["paper_primary_n_success"]) / float(n_eval)
            if n_eval
            else None
        )
    else:
        summary["paper_primary_success_rate"] = None
    if summary.get("task_primary_n_success") is not None:
        summary["task_primary_success_rate"] = (
            float(summary["task_primary_n_success"]) / float(n_eval)
            if n_eval
            else None
        )
    else:
        summary["task_primary_success_rate"] = None
    if summary.get("task_primary_n_success") is not None:
        summary["success_metric"] = summary.get("task_primary_metric")
    elif summary.get("paper_primary_n_success") is not None:
        summary["success_metric"] = summary.get("paper_primary_metric")
    else:
        summary["success_metric"] = "native_done_or_success"
    return summary


def _parse_seeds(seed: int | None, seeds: str | None) -> list[int]:
    if seeds:
        return [int(x.strip()) for x in seeds.split(",") if x.strip()]
    if seed is not None:
        return [int(seed)]
    return [42, 43, 44, 45, 46]


def _write_per_case_exports(metrics: list[dict[str, Any]], out_dir: Path) -> None:
    jsonl_tmp = out_dir / "per_case.jsonl.tmp"
    with jsonl_tmp.open("w", encoding="utf-8") as stream:
        for item in metrics:
            stream.write(json.dumps(_jsonable(item), sort_keys=True) + "\n")
    jsonl_tmp.replace(out_dir / "per_case.jsonl")

    columns = [
        "schema_version",
        "case_status",
        "benchmark",
        "seed",
        "manifest_case_id",
        "planner_seed",
        "success_metric",
        "is_success",
        "native_is_success",
        "paper_primary_success",
        "task_primary_success",
        "done",
        "truncated",
        "n_steps",
        "terminal_distance",
        "estimated_simulator_transitions",
        "error",
    ]
    timing_columns = [
        "total_seconds",
        "fit_total_seconds",
        "cem_total_seconds",
        "environment_total_seconds",
    ]
    csv_tmp = out_dir / "per_case.csv.tmp"
    with csv_tmp.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=columns + [f"timing_{key}" for key in timing_columns],
        )
        writer.writeheader()
        for item in metrics:
            row = {key: item.get(key) for key in columns}
            timing = item.get("timing") or {}
            for key in timing_columns:
                row[f"timing_{key}"] = timing.get(key)
            writer.writerow(row)
    csv_tmp.replace(out_dir / "per_case.csv")


def _state_path(value: Any, path: str) -> np.ndarray:
    current = value
    for token in path.split("."):
        if isinstance(current, dict):
            current = current[token]
        else:
            current = getattr(current, token)
    return np.asarray(current, dtype=np.float64)


def _objective_intervention_class(sim_class, config: dict[str, Any]):
    components = config.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("objective intervention requires a non-empty components list")

    class ObjectiveInterventionSimulator(sim_class):
        def terminal_cost(self):
            total = 0.0
            for component in components:
                current = _state_path(self.state, str(component["state_path"]))
                target = _state_path(
                    self.target_state, str(component["target_path"])
                )
                delta = current - target
                if bool(component.get("periodic", False)):
                    delta = np.arctan2(np.sin(delta), np.cos(delta))
                weight = float(component.get("weight", 1.0))
                total += weight * float(np.sum(np.square(delta)))
            return total

    ObjectiveInterventionSimulator.__name__ = sim_class.__name__
    return ObjectiveInterventionSimulator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["pusht", "two_room", "two_room_swm", "reacher", "cube"],
        required=True,
    )
    parser.add_argument("--sim-path", type=Path, default=None)
    parser.add_argument(
        "--runtime-contract",
        type=Path,
        default=None,
        help=(
            "P1 runtime_contract.json. When omitted, use the file beside "
            "--sim-path if present; P2 rejects any FPS/frame/API/toolbox mismatch."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-label", default="")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed per-case metrics and run only missing/failed cases.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Explicitly replace an existing non-release output directory.",
    )
    parser.add_argument(
        "--case-shard-count",
        type=int,
        default=1,
        help="Split ordered cases into this many deterministic modulo shards.",
    )
    parser.add_argument(
        "--case-shard-index",
        type=int,
        default=0,
        help="Zero-based deterministic modulo shard to run.",
    )
    parser.add_argument("--class-name", default="GeneratedSimulator")
    parser.add_argument(
        "--objective-intervention",
        type=Path,
        default=None,
        help=(
            "Diagnostic JSON objective specification. Only terminal_cost is "
            "replaced; fit/state/update remain frozen."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--apply-steps", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument(
        "--sim-fps",
        type=int,
        default=10,
        help="Simulator FPS passed to GeneratedSimulator (must match P1; default 10).",
    )
    parser.add_argument("--plan-mode", choices=["mpc", "open_loop"], default="mpc")
    parser.add_argument("--plan-horizon", type=int, default=50)
    parser.add_argument(
        "--eval-protocol",
        choices=sorted(SUPPORTED_EVAL_PROTOCOLS),
        default=EVAL_PROTOCOL_VDA_DEFAULT,
    )
    parser.add_argument(
        "--eval-backend",
        choices=["native", "swm_reference"],
        default="native",
        help="Ground-truth environment backend. Frozen LeWM N=50 uses swm_reference.",
    )
    parser.add_argument(
        "--task-success-protocol",
        choices=["original", "task_success_v1"],
        default="original",
        help=(
            "SWM reference only: use original future-state goals/termination, "
            "or task-aligned visible goals with a fixed action budget."
        ),
    )
    parser.add_argument("--start-goal-source", default="session")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Optional matched-evaluation manifest JSON. Selects one case per seed.",
    )
    parser.add_argument(
        "--swm-hdf5-path",
        type=Path,
        default=None,
        help="Override the canonical HDF5 path recorded in an SWM manifest.",
    )
    parser.add_argument(
        "--swm-python",
        type=Path,
        default=None,
        help="Python interpreter for the isolated stable_worldmodel reference server.",
    )
    parser.add_argument(
        "--pusht-goal-image-mode",
        choices=["covered_goal", "start"],
        default="covered_goal",
        help=(
            "Push-T only: use the fixed covered-goal image_B, or a diagnostic "
            "mode where image_B is the start frame with the visible green target."
        ),
    )
    parser.add_argument(
        "--pusht-strip-green-target",
        dest="pusht_strip_green_target",
        action="store_true",
        help=(
            "Push-T only: remove green target pixels from image/image_B passed "
            "to simulator.fit while preserving them in raw diagnostic frames "
            "(canonical default)."
        ),
    )
    parser.add_argument(
        "--no-pusht-strip-green-target",
        dest="pusht_strip_green_target",
        action="store_false",
        help=(
            "Push-T diagnostic only: expose the green target to simulator.fit. "
            "Raw diagnostic frames always retain the target."
        ),
    )
    parser.set_defaults(pusht_strip_green_target=True)
    parser.add_argument(
        "--reacher-goal-image-mode",
        choices=["goal_state", "start"],
        default="goal_state",
        help=(
            "Reacher only: use the IK final-state image_B, or a diagnostic "
            "mode where image_B is the start frame with the visible red target."
        ),
    )
    parser.add_argument("--cem-population", type=int, default=None)
    parser.add_argument("--cem-iters", type=int, default=None)
    parser.add_argument("--elite-frac", type=float, default=None)
    parser.add_argument("--init-std-scale", type=float, default=None)
    parser.add_argument(
        "--cem-use-stage-cost",
        action="store_true",
        help="Accumulate the simulator's optional per-step trajectory cost.",
    )
    parser.add_argument(
        "--cem-action-smoothness-weight",
        type=float,
        default=0.0,
        help="Weight for scale-normalized consecutive-action changes (default 0).",
    )
    parser.add_argument(
        "--cem-action-knots",
        type=int,
        default=None,
        help="Sample this many action knots and interpolate across the horizon.",
    )
    parser.add_argument(
        "--planner-seed",
        type=int,
        default=None,
        help="CEM RNG seed. Defaults independently to each environment seed.",
    )
    parser.add_argument(
        "--no-propagate-prev-state",
        dest="propagate_prev_state",
        action="store_false",
        help="Refit each MPC cycle from images without propagated simulator state.",
    )
    parser.add_argument(
        "--stop-on-primary-success",
        action="store_true",
        help="Diagnostic: stop after a step reports info.primary_success=true.",
    )
    parser.set_defaults(propagate_prev_state=True)
    parser.add_argument(
        "--no-world-api",
        action="store_true",
        help=(
            "Disable WorldAPI during P2 evaluation. By default WorldAPI is "
            "available to simulator fit(image_A, image_B); generated rollouts "
            "should still keep update/terminal_cost/render_frame API-free."
        ),
    )
    parser.add_argument(
        "--world-api-cache-dir",
        type=Path,
        default=None,
        help=(
            "Optional WorldAPI disk cache directory for P2. Defaults to "
            "<out_dir>/world_api_cache."
        ),
    )
    args = parser.parse_args()

    args.manifest = load_manifest(args.manifest_path)
    args._manifest_sha256 = (
        manifest_sha256(args.manifest_path) if args.manifest_path is not None else None
    )
    if args.manifest is not None:
        if args.manifest.get("benchmark") != args.benchmark:
            raise SystemExit(
                f"manifest benchmark={args.manifest.get('benchmark')!r} does not match "
                f"--benchmark={args.benchmark!r}"
            )
        args.start_goal_source = "manifest"
    if args.eval_protocol == EVAL_PROTOCOL_LEWM_ALIGNED_N50:
        if args.manifest is None or args.manifest_path is None:
            raise SystemExit(
                f"{EVAL_PROTOCOL_LEWM_ALIGNED_N50} requires --manifest-path"
            )
        if args.seed is not None or args.seeds is not None:
            raise SystemExit(
                f"{EVAL_PROTOCOL_LEWM_ALIGNED_N50} uses the manifest's frozen "
                "50-case order; do not pass --seed/--seeds"
            )
        validate_manifest(
            args.manifest,
            profile=EVAL_PROTOCOL_LEWM_ALIGNED_N50,
        )
        expected_manifest_sha = LEWM_ALIGNED_MANIFEST_SHA256.get(args.benchmark)
        if args._manifest_sha256 != expected_manifest_sha:
            raise SystemExit(
                f"{EVAL_PROTOCOL_LEWM_ALIGNED_N50} manifest hash mismatch for "
                f"{args.benchmark}: expected {expected_manifest_sha}, "
                f"got {args._manifest_sha256}"
            )
        if args.plan_mode != "mpc":
            raise SystemExit(
                f"{EVAL_PROTOCOL_LEWM_ALIGNED_N50} requires --plan-mode mpc"
            )
        if args.task_success_protocol != "original":
            raise SystemExit(
                f"{EVAL_PROTOCOL_LEWM_ALIGNED_N50} requires "
                "--task-success-protocol original; task_success_v1 changes "
                "future-state goals and is a separate diagnostic protocol"
            )
        args.eval_backend = LEWM_ALIGNED_N50_V1.backend
        args.seed_values = [int(case["seed"]) for case in args.manifest["cases"]]
    else:
        args.seed_values = _parse_seeds(args.seed, args.seeds)
    if args.case_shard_count <= 0:
        raise SystemExit("--case-shard-count must be positive")
    if not 0 <= args.case_shard_index < args.case_shard_count:
        raise SystemExit(
            "--case-shard-index must satisfy 0 <= index < case-shard-count"
        )
    args._all_seed_values = list(args.seed_values)
    args.seed_values = [
        seed
        for index, seed in enumerate(args.seed_values)
        if index % args.case_shard_count == args.case_shard_index
    ]
    if not args.seed_values:
        raise SystemExit("selected case shard is empty")

    if args.eval_backend == "swm_reference":
        if args.manifest is None:
            raise SystemExit("swm_reference backend requires --manifest-path")
        recorded_hdf5 = args.manifest.get("source_hdf5")
        args.swm_hdf5_path = args.swm_hdf5_path or (
            Path(recorded_hdf5) if recorded_hdf5 else None
        )
        if args.swm_hdf5_path is None:
            raise SystemExit(
                "swm_reference backend requires --swm-hdf5-path or manifest.source_hdf5"
            )
        args.swm_hdf5_path = args.swm_hdf5_path.expanduser().resolve()
        if not args.swm_hdf5_path.is_file():
            raise SystemExit(f"SWM HDF5 file does not exist: {args.swm_hdf5_path}")

    sim_path = args.sim_path or _default_sim_path(args.benchmark)
    if sim_path is None:
        raise SystemExit(f"could not infer --sim-path for benchmark={args.benchmark}")
    sim_path = sim_path.resolve()
    if not sim_path.exists():
        raise SystemExit(f"simulator path does not exist: {sim_path}")
    args._simulator_sha256 = _file_sha256(sim_path)
    git_provenance = _pipeline_git_provenance()
    args._pipeline_git_commit = git_provenance["commit"]
    args._pipeline_git_dirty = git_provenance["dirty"]
    args._pipeline_git_diff_sha256 = git_provenance["diff_sha256"]

    inferred_contract = sim_path.parent / "runtime_contract.json"
    runtime_contract_path = (
        args.runtime_contract.resolve()
        if args.runtime_contract is not None
        else inferred_contract
        if inferred_contract.is_file()
        else None
    )
    args._runtime_contract_path = runtime_contract_path
    args._runtime_contract = None
    args._sim_frame_size = _frame_size_for(args.benchmark)
    if runtime_contract_path is not None:
        if not runtime_contract_path.is_file():
            raise SystemExit(
                f"runtime contract does not exist: {runtime_contract_path}"
            )
        contract = json.loads(runtime_contract_path.read_text(encoding="utf-8"))
        contract_frame_size = contract.get("frame_size")
        if (
            not isinstance(contract_frame_size, list)
            or len(contract_frame_size) != 2
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value <= 0
                for value in contract_frame_size
            )
        ):
            raise SystemExit(
                "P1 runtime contract has invalid frame_size: "
                f"{contract_frame_size!r}"
            )
        args._sim_frame_size = tuple(contract_frame_size)
        expected = {
            "simulator_class": args.class_name,
            "frame_size": list(args._sim_frame_size),
            "fps": int(args.sim_fps),
            "fit_world_api_enabled": not bool(args.no_world_api),
            "rollout_world_api_enabled": False,
            "local_2d_toolbox_version": LOCAL_2D_TOOLBOX_VERSION,
        }
        mismatches = {}
        for key, value in expected.items():
            p1_value = contract.get(key)
            compatible = p1_value == value
            if key == "local_2d_toolbox_version":
                compatible = str(p1_value) in LOCAL_2D_TOOLBOX_COMPATIBLE_VERSIONS
            if not compatible:
                mismatches[key] = {"p1": p1_value, "p2": value}
        if mismatches:
            raise SystemExit(
                "P1/P2 runtime contract mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        args._runtime_contract = contract

    default_root = REPO_ROOT / "viz_output" / f"smoke_p2_mpc_{args.benchmark}"
    out_root = args.out_dir or default_root
    if args.run_label:
        run_name = args.run_label
    else:
        run_name = f"{sim_path.parents[1].name}_p2_{args.eval_protocol}"
    out_dir = out_root / run_name
    if out_dir.exists() and not args.resume:
        if not args.overwrite_output:
            raise SystemExit(
                f"output directory already exists: {out_dir}; pass --resume "
                "or --overwrite-output explicitly"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_started_at = _utc_now()
    run_t0 = time.perf_counter()
    (out_dir / "RUNNING.json").write_text(
        json.dumps(
            {
                "schema_version": "run_status_v1",
                "status": "running",
                "benchmark": args.benchmark,
                "eval_protocol": args.eval_protocol,
                "case_shard_count": args.case_shard_count,
                "case_shard_index": args.case_shard_index,
                "started_at_utc": run_started_at,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[p2] benchmark={args.benchmark}")
    print(f"[p2] sim_path={sim_path}")
    print(f"[p2] out_dir={out_dir}")
    print(f"[p2] eval_protocol={args.eval_protocol} seeds={args.seed_values}")
    print(f"[p2] eval_backend={args.eval_backend}")
    print(f"[p2] sim_frame_size={args._sim_frame_size}")
    if args.benchmark == "pusht":
        print(
            "[p2] pusht_strip_green_target="
            f"{bool(args.pusht_strip_green_target)}"
        )
    print(f"[p2] world_api={'disabled' if args.no_world_api else 'enabled'}")
    print(
        "[p2] runtime_contract="
        f"{runtime_contract_path if runtime_contract_path is not None else 'legacy/absent'}"
    )
    if args.manifest_path:
        print(f"[p2] manifest_path={args.manifest_path}")

    sim_class = load_simulator_class(sim_path, args.class_name)
    args._objective_intervention = None
    args._objective_intervention_sha256 = None
    if args.objective_intervention is not None:
        intervention_path = args.objective_intervention.resolve()
        intervention = json.loads(intervention_path.read_text(encoding="utf-8"))
        if not isinstance(intervention, dict):
            raise SystemExit("objective intervention must be a JSON object")
        sim_class = _objective_intervention_class(sim_class, intervention)
        args._objective_intervention = intervention
        args._objective_intervention_sha256 = _file_sha256(intervention_path)
        print(
            "[p2] objective_intervention="
            f"{intervention_path} sha256={args._objective_intervention_sha256}"
        )
    server_context = (
        SWMReferenceServer(python_path=args.swm_python)
        if args.eval_backend == "swm_reference"
        else nullcontext(None)
    )
    with server_context as swm_server:
        args._swm_server = swm_server
        metrics = []
        for seed in args.seed_values:
            existing_path = out_dir / f"seed_{seed}" / "metrics.json"
            if args.resume and existing_path.is_file():
                existing = json.loads(existing_path.read_text(encoding="utf-8"))
                if existing.get("case_status") == "completed":
                    print(f"[p2] resume seed={seed}: using completed metrics")
                    metrics.append(existing)
                    continue
            try:
                metrics.append(
                    _run_one_seed(
                        sim_class=sim_class,
                        benchmark=args.benchmark,
                        seed=seed,
                        out_dir=out_dir,
                        args=args,
                    )
                )
            except Exception as exc:
                failure_dir = out_dir / f"seed_{seed}"
                failure_dir.mkdir(parents=True, exist_ok=True)
                manifest_case = (
                    case_for_seed(args.manifest, seed)
                    if args.manifest is not None
                    else None
                )
                failure = {
                    "schema_version": P2_CASE_SCHEMA_VERSION,
                    "case_status": "infrastructure_failure",
                    "benchmark": args.benchmark,
                    "seed": seed,
                    "manifest_case_id": (
                        None if manifest_case is None else manifest_case.get("case_id")
                    ),
                    "planner_seed": (
                        seed if args.planner_seed is None else int(args.planner_seed)
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                (failure_dir / "metrics.json").write_text(
                    json.dumps(failure, indent=2),
                    encoding="utf-8",
                )
                metrics.append(failure)
                print(f"[p2] seed={seed} FAILED: {failure['error']}", file=sys.stderr)
    _write_per_case_exports(metrics, out_dir)
    summary = _summary(metrics, args=args, sim_path=sim_path, out_dir=out_dir)
    summary.update(
        {
            "eval_backend": args.eval_backend,
            "manifest_sha256": (args._manifest_sha256),
            "resolved_eval_profile": (
                LEWM_ALIGNED_N50_V1.as_dict()
                if args.eval_protocol == EVAL_PROTOCOL_LEWM_ALIGNED_N50
                else None
            ),
            "swm_hdf5_path": (
                str(args.swm_hdf5_path)
                if args.eval_backend == "swm_reference"
                else None
            ),
            "case_shard_count": args.case_shard_count,
            "case_shard_index": args.case_shard_index,
            "all_manifest_seeds": args._all_seed_values,
            "started_at_utc": run_started_at,
            "finished_at_utc": _utc_now(),
            "wall_seconds": time.perf_counter() - run_t0,
        }
    )
    summary_tmp = out_dir / "summary.json.tmp"
    summary_tmp.write_text(
        json.dumps(_jsonable(summary), indent=2),
        encoding="utf-8",
    )
    summary_tmp.replace(out_dir / "summary.json")
    running_marker = out_dir / "RUNNING.json"
    if running_marker.exists():
        running_marker.unlink()
    completion_name = "COMPLETED.json" if summary["n_failed"] == 0 else "PARTIAL.json"
    (out_dir / completion_name).write_text(
        json.dumps(
            {
                "schema_version": "run_status_v1",
                "status": "completed"
                if summary["n_failed"] == 0
                else "partial_with_failures",
                "benchmark": args.benchmark,
                "n_requested": summary["n_requested"],
                "n_completed": summary["n_completed"],
                "n_failed": summary["n_failed"],
                "summary_sha256": _file_sha256(out_dir / "summary.json"),
                "finished_at_utc": summary["finished_at_utc"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[p2] success_metric={summary['success_metric']} "
        f"success_rate={summary['success_rate']:.3f} "
        f"n_success={summary['n_success']}/{len(metrics)}"
    )
    print(f"[p2] wrote {out_dir / 'summary.json'}")
    return 0 if summary["n_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
