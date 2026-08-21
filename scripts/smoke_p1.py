#!/usr/bin/env python
"""Shared P1 smoke: live VLM codegen for a registered benchmark.

The agentic generator, validation tools, prompt, final gate, held-out replay,
and training-fidelity sweep are the same for every benchmark. Benchmark
variation is supplied only by ``BenchmarkSpec`` from ``vdaworld.eval.benchmarks``.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import inspect
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from vdaworld.api.llm.methods.prompt import get_prompt
from vdaworld.api.vlm import VLMClient
from vdaworld.core.api import WorldAPI
from vdaworld.core.local_2d import LOCAL_2D_TOOLBOX_VERSION
from vdaworld.core.planning_agentic_generator import PlanningAgenticGenerator
from vdaworld.core.simulator import ActionConditionedSimulatorBase, SimulatorBase
from vdaworld.eval.benchmarks import (
    SANDBOX_FIT_TRAJECTORY,
    SHARED_GATE_MAX_WORST_RATIO,
    available_benchmarks,
    get_benchmark,
)
from vdaworld.eval.closed_loop import eval_held_out, render_side_by_side

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR_BASE = REPO_ROOT / "viz_output"
PROMPTS_DIR = REPO_ROOT / "prompts" / "action_conditioned"
LEGACY_PROMPTS_DIR = REPO_ROOT / "prompts" / "action_conditioned_pre_gemini_seg_2026_07_08"
GEMINI_SEG_PROMPT_SOURCE = REPO_ROOT / "prompts" / "action_conditioned_with_gemini_segmentation"

CAPTION = ""
SIMULATOR_CLASS_NAME = "GeneratedSimulator"
FPS = 10
MAX_TURNS = 36
N_AUTO_VALIDATE = 3
GATE_MAX_RETRIES = 3
GATE_RETRY_TURNS = 8
GATE_REALIZABILITY_MARGIN = 1.25
CEM_PLANNER = dict(
    population=200,
    elite_frac=0.1,
    iters=10,
    init_std_scale=0.5,
    seed=0,
)
P1_METRICS_SCHEMA_VERSION = "p1_metrics_v2"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(path)))
    return digest.hexdigest()


def _git_provenance() -> dict[str, object]:
    def run(*args: str) -> bytes | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                timeout=10,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain=v1", "--untracked-files=all")
    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": None if commit is None else commit.decode().strip(),
        "dirty": None if status is None else bool(status.strip()),
        "diff_sha256": (
            None if diff is None or not diff else hashlib.sha256(diff).hexdigest()
        ),
    }


def _structured_gate_summary(summary: str) -> dict[str, object]:
    first_line = (summary or "").splitlines()[0] if summary else ""
    checks: dict[str, str] = {}
    for key, value in re.findall(r"([a-zA-Z0-9_]+)=([a-zA-Z0-9_.-]+)", first_line):
        checks[key] = value
    return {"checks": checks, "raw": summary}


def _derive_dataset_interface(dataset_dir: Path) -> dict:
    """Mechanically derive observation/action shape facts from trajectories."""
    from vdaworld.core.dataset_tools import read_trajectory, view_image

    traj_dirs = sorted(dataset_dir.glob("trajectory_*"))
    if not traj_dirs:
        raise FileNotFoundError(f"no trajectories under {dataset_dir}")

    all_actions = []
    for i in range(len(traj_dirs)):
        tr = read_trajectory(dataset_dir, i)
        if tr["n_steps"] > 0:
            all_actions.append(tr["actions"])
    if not all_actions:
        raise ValueError(f"no actions found under {dataset_dir}")
    actions = np.concatenate(all_actions, axis=0)

    img = view_image(dataset_dir, 0, 0)
    frame_size = (int(img.shape[1]), int(img.shape[0]))
    return {
        "n_traj": len(traj_dirs),
        "frame_size": frame_size,
        "action_dim": int(actions.shape[1]),
        "action_low": actions.min(axis=0).astype(np.float64),
        "action_high": actions.max(axis=0).astype(np.float64),
        "n_actions_total": int(actions.shape[0]),
    }


def _collect_api_tool_names() -> str:
    api = WorldAPI(output_dir=None)
    return "\n".join(
        f"- `{name}`"
        for name, _ in inspect.getmembers(api, predicate=inspect.ismethod)
        if not name.startswith("_") and name != "segment"
    )


def _plot_cem_history(history, out_path: Path, title: str) -> None:
    iters = [h.iteration for h in history]
    mean_c = [h.mean_cost for h in history]
    elite_c = [h.elite_mean_cost for h in history]
    best_c = [h.best_cost for h in history]
    plt.figure(figsize=(6, 4))
    plt.plot(iters, mean_c, label="population mean", marker="o")
    plt.plot(iters, elite_c, label="elite mean", marker="s")
    plt.plot(iters, best_c, label="best", marker="^")
    plt.xlabel("CEM iteration")
    plt.ylabel("cost (terminal_cost)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def _load_generated_class(code_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location("simulator_gen", code_path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot create module spec from {code_path}")
    module = importlib.util.module_from_spec(spec)
    module.SimulatorBase = SimulatorBase
    module.ActionConditionedSimulatorBase = ActionConditionedSimulatorBase
    module.WorldAPI = WorldAPI
    sys.modules["simulator_gen"] = module
    spec.loader.exec_module(module)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"{code_path} does not define class {class_name!r}")
    if not issubclass(cls, ActionConditionedSimulatorBase):
        raise TypeError(
            f"{class_name} must subclass ActionConditionedSimulatorBase; "
            f"got MRO {cls.__mro__}"
        )
    return cls


def _auto_validate_trajectories(train_idxs: list[int]) -> list[int]:
    if len(train_idxs) <= N_AUTO_VALIDATE:
        return list(train_idxs)
    sel = np.linspace(0, len(train_idxs) - 1, N_AUTO_VALIDATE)
    return sorted({train_idxs[int(round(i))] for i in sel})


def _fmt_ratio(value) -> str:
    return "FAIL" if value is None else f"{float(value):.3f}"


def _fmt_frac(value) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.1f}%"


def _format_candidate_criteria(criteria: dict) -> str:
    if not criteria:
        return "n/a"
    return " ".join(
        f"{name}={'PASS' if item.get('passed') else 'FAIL'}"
        for name, item in sorted(criteria.items())
    )


def _print_spec(benchmark: str, spec, data: dict, fit_a: Path, fit_b: Path) -> None:
    deployed_low, deployed_high = spec.deployed_bounds_copy()
    print("=== BenchmarkSpec ===")
    print(f"  benchmark:       {benchmark}")
    print(f"  dataset_dir:     {spec.dataset_dir}")
    print(f"  frame_size:      {spec.frame_size}")
    print(f"  n_frames:        {spec.n_frames}")
    print(
        "  sandbox fit:     "
        f"traj_{SANDBOX_FIT_TRAJECTORY:04d} "
        f"frame_{spec.sandbox_fit_frame_a:04d} -> "
        f"frame_{spec.sandbox_fit_frame_b:04d} "
        f"({fit_a.name}, {fit_b.name})"
    )
    print(
        "  deployed bounds: "
        f"low={np.round(deployed_low, 3).tolist()} "
        f"high={np.round(deployed_high, 3).tolist()}"
    )
    print(f"  held_out_traj:   {spec.held_out_traj}")
    print(f"  dataset actions: low={np.round(data['action_low'], 3).tolist()} "
          f"high={np.round(data['action_high'], 3).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=available_benchmarks(),
        help="Registered benchmark name.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--load-existing",
        type=Path,
        default=None,
        help="Skip codegen and evaluate an existing simulator_gen.py.",
    )
    parser.add_argument("--model", type=str, default="gemini-3.6-flash")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--repeat-id",
        default=None,
        help="Stable generation-attempt identifier recorded in metrics.",
    )
    parser.add_argument(
        "--composition",
        default=None,
        help="Dataset-composition cell identifier recorded in metrics.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Explicitly allow replacing an existing output directory.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help="Override the benchmark's default P1 training dataset directory.",
    )
    parser.add_argument(
        "--sandbox-dataset-dir",
        type=Path,
        default=None,
        help=(
            "Fixed dataset supplying only the sandbox fit-frame pair. "
            "Defaults to the treatment dataset."
        ),
    )
    parser.add_argument(
        "--audit-dataset-dir",
        type=Path,
        default=None,
        help=(
            "Hidden post-generation audit dataset, unavailable to codegen tools. "
            "Defaults to the treatment dataset for legacy runs."
        ),
    )
    parser.add_argument(
        "--audit-trajectory",
        type=int,
        default=None,
        help="Trajectory index within --audit-dataset-dir (registry default otherwise).",
    )
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=None,
        help=(
            "Override prompt directory. Defaults to prompts/action_conditioned "
            f"(legacy rollback: {LEGACY_PROMPTS_DIR.name})."
        ),
    )
    parser.add_argument(
        "--disable-local-toolbox-tools",
        action="store_true",
        help=(
            "Hide derive_visual_model, get_runtime_toolbox_documentation, and "
            "estimate_dynamics_models from codegen. The inherited runtime "
            "helpers remain installed; use this for historical prompt controls."
        ),
    )
    parser.add_argument(
        "--prompt-addendum-file",
        type=Path,
        default=None,
        help=(
            "Append a small profile-specific Markdown addendum to the assembled "
            "prompt without modifying the selected prompt directory."
        ),
    )
    parser.add_argument(
        "--gate-cem-suite",
        action="store_true",
        help="Require the final gate to pass a harness-selected CEM stress suite.",
    )
    parser.add_argument(
        "--gate-p2-safety",
        action="store_true",
        help=(
            "Require the final gate to pass a P2 runtime-dependency safety "
            "smoke under the configured P1 runtime API contract."
        ),
    )
    parser.add_argument(
        "--no-world-api",
        action="store_true",
        help=(
            "Generate and validate with api=None, matching P2 launchers that "
            "disable WorldAPI. Inherited local helpers remain available."
        ),
    )
    parser.add_argument(
        "--deployment-goal-mode",
        choices=["final", "start"],
        default="final",
        help=(
            "How P2 will construct image_B at deployment. 'final' = a "
            "completed-task frame (training convention, no extra check); "
            "'start' = the current observation (gate probes fit(frame0, "
            "frame0) and fails degenerate target parses). Mirror the P2 "
            "protocol flag for this benchmark."
        ),
    )
    parser.add_argument(
        "--no-keep-best",
        action="store_true",
        help=(
            "Disable keep-best checkpointing (by default, when the final "
            "gate fails, the best-scoring earlier code version ships "
            "instead of the last edit)."
        ),
    )
    parser.add_argument(
        "--gate-ratio-advisory",
        action="store_true",
        help=(
            "Report the worst-ratio fidelity criterion but exclude it from "
            "gate pass/fail, letting planner-relevant checks (CEM suite, "
            "deployment-goal, invariances) drive the verdict. Use when the "
            "fixed ratio bar is unreachable for a benchmark's dynamics."
        ),
    )
    parser.add_argument(
        "--gate-state-consistency-required",
        action="store_true",
        help=(
            "Require the final gate's worst-trajectory state-consistency probe "
            "to have no drifting components."
        ),
    )
    parser.add_argument(
        "--gate-fit-quality-required",
        action="store_true",
        help=(
            "Require render_frame() immediately after fit() to match true frames "
            "under a normalized RMSE threshold."
        ),
    )
    parser.add_argument(
        "--no-gate-fit-quality",
        dest="gate_fit_quality_required",
        action="store_false",
        help="Disable the required post-final fit/render structural quality gate.",
    )
    parser.set_defaults(gate_fit_quality_required=True)
    parser.add_argument(
        "--gate-fit-quality-max-rmse",
        type=float,
        default=0.30,
        help="Normalized RMSE threshold for --gate-fit-quality-required.",
    )
    parser.add_argument(
        "--gate-fit-quality-max-scene-frac",
        type=float,
        default=0.35,
        help=(
            "Skip colour/foreground fit-quality labels whose true mask covers "
            "more than this fraction of the frame (near-full-frame scene "
            "appearance, not a trackable object). Set to 1.0 to restore the "
            "old no-skip behaviour."
        ),
    )
    parser.add_argument(
        "--gate-fit-quality-mode",
        choices=["all_labels", "primary_labels", "rmse_plus_foreground"],
        default="rmse_plus_foreground",
        help=(
            "How colour-label perception checks contribute to fit-quality: "
            "all_labels = every non-scene colour label hard-fails (legacy); "
            "primary_labels = only the largest two non-scene labels hard-fail; "
            "rmse_plus_foreground = colour labels advisory, hard on RMSE + "
            "non-scene foreground only (default)."
        ),
    )
    parser.add_argument(
        "--gate-state-consistency-include-latent",
        action="store_true",
        help=(
            "Hard-fail state-consistency on velocity/latent keys that still-"
            "frame fit() cannot re-observe (default: skip those keys)."
        ),
    )
    parser.add_argument(
        "--no-keep-best-include-cem",
        action="store_true",
        help=(
            "When --gate-cem-suite is set, keep-best reranks its top few "
            "checkpoints with a bounded cheap CEM probe before restoration. "
            "Pass this flag to use ratio/perception-only ranking."
        ),
    )
    parser.add_argument(
        "--gate-workflow-evidence",
        choices=["mandatory", "advisory", "off"],
        default="mandatory",
        help=(
            "Finish-guard evidence workflow: mandatory blocks finish until "
            "analyze_action_effects / view_fit_comparison / "
            "validate_state_consistency are present when those gates are "
            "required; advisory logs gaps only; off disables the check."
        ),
    )
    args = parser.parse_args()

    process_started_at = _utc_now()
    process_t0 = time.perf_counter()
    bench = get_benchmark(args.benchmark)
    if args.dataset_dir is not None:
        bench = replace(bench, dataset_dir=args.dataset_dir.resolve())
    dataset_dir = bench.dataset_dir.resolve()
    data = _derive_dataset_interface(dataset_dir)
    if tuple(data["frame_size"]) != tuple(bench.frame_size):
        raise ValueError(
            f"registry frame_size={bench.frame_size} but dataset frame_size={data['frame_size']}"
        )
    sandbox_dataset_dir = (
        args.sandbox_dataset_dir.resolve()
        if args.sandbox_dataset_dir is not None
        else dataset_dir
    )
    audit_dataset_dir = (
        args.audit_dataset_dir.resolve()
        if args.audit_dataset_dir is not None
        else dataset_dir
    )
    sandbox_data = _derive_dataset_interface(sandbox_dataset_dir)
    audit_data = _derive_dataset_interface(audit_dataset_dir)
    for label, interface in {
        "sandbox": sandbox_data,
        "audit": audit_data,
    }.items():
        if tuple(interface["frame_size"]) != tuple(bench.frame_size):
            raise ValueError(
                f"{label} dataset frame_size={interface['frame_size']} "
                f"does not match registry frame_size={bench.frame_size}"
            )
        if int(interface["action_dim"]) != int(data["action_dim"]):
            raise ValueError(
                f"{label} dataset action_dim={interface['action_dim']} "
                f"does not match treatment action_dim={data['action_dim']}"
            )
    audit_trajectory = (
        int(args.audit_trajectory)
        if args.audit_trajectory is not None
        else int(bench.held_out_traj)
    )
    if not (0 <= audit_trajectory < audit_data["n_traj"]):
        raise ValueError(
            f"audit_trajectory={audit_trajectory} out of range for "
            f"{audit_data['n_traj']} audit trajectories"
        )
    sandbox_traj = sandbox_dataset_dir / f"trajectory_{SANDBOX_FIT_TRAJECTORY:04d}"
    fit_a = sandbox_traj / f"frame_{bench.sandbox_fit_frame_a:04d}.png"
    fit_b = sandbox_traj / f"frame_{bench.sandbox_fit_frame_b:04d}.png"
    if not fit_a.exists() or not fit_b.exists():
        raise FileNotFoundError(f"sandbox fit frames missing: {fit_a}, {fit_b}")

    prompts_dir = (
        args.prompts_dir.resolve() if args.prompts_dir is not None else PROMPTS_DIR
    )
    prompt_suffix = ""
    if args.prompt_addendum_file is not None:
        addendum_path = args.prompt_addendum_file.resolve()
        prompt_suffix = addendum_path.read_text(encoding="utf-8").strip()
        if not prompt_suffix:
            raise ValueError(f"prompt addendum is empty: {addendum_path}")

    available_tools = _collect_api_tool_names()
    assembled = get_prompt(
        str(prompts_dir),
        "agentic_generation",
        caption=CAPTION,
        simulator_class_name=SIMULATOR_CLASS_NAME,
        max_turns=MAX_TURNS,
        available_tools=available_tools,
        frame_size=tuple(data["frame_size"]),
        fps=FPS,
        finish_guard_max_extra_turns=GATE_RETRY_TURNS * GATE_MAX_RETRIES,
    )
    if prompt_suffix:
        assembled = f"{assembled.rstrip()}\n\n{prompt_suffix}\n"

    if args.dry_run:
        _print_spec(args.benchmark, bench, data, fit_a, fit_b)
        print()
        print(assembled)
        print(f"\n--- prompt is {len(assembled)} chars, {assembled.count(chr(10)) + 1} lines ---")
        return 0

    if args.out_dir is not None:
        out_dir = args.out_dir.resolve()
    else:
        model_slug = args.model.replace("/", "_").replace(":", "_")
        out_dir = OUT_DIR_BASE / f"smoke_p1_{args.benchmark}_{model_slug}"

    if out_dir.exists():
        if not args.overwrite_output:
            raise FileExistsError(
                f"output directory already exists: {out_dir}; "
                "release attempts are immutable (pass --overwrite-output only "
                "for an explicitly non-release rerun)"
            )
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_state = {"complete": False}
    status_base = {
        "schema_version": "run_status_v1",
        "benchmark": args.benchmark,
        "repeat_id": args.repeat_id,
        "composition": args.composition,
        "started_at_utc": process_started_at,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": socket.gethostname(),
    }
    (out_dir / "RUNNING.json").write_text(
        json.dumps({**status_base, "status": "running"}, indent=2),
        encoding="utf-8",
    )

    def record_incomplete_exit() -> None:
        if run_state["complete"] or not out_dir.exists():
            return
        (out_dir / "FAILED.json").write_text(
            json.dumps(
                {
                    **status_base,
                    "status": "incomplete_or_failed",
                    "finished_at_utc": _utc_now(),
                    "wall_seconds": time.perf_counter() - process_t0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    atexit.register(record_incomplete_exit)
    provenance = {
        "git": _git_provenance(),
        "model": {
            "identifier": args.model,
            "backend": "gemini_api",
            "temperature": 0.0,
            "access_date_utc": process_started_at,
        },
        "prompt": {
            "assembled_sha256": _sha256_bytes(assembled.encode("utf-8")),
            "prompts_dir": str(prompts_dir),
            "addendum_path": (
                str(args.prompt_addendum_file.resolve())
                if args.prompt_addendum_file is not None
                else None
            ),
            "addendum_sha256": (
                _file_sha256(args.prompt_addendum_file.resolve())
                if args.prompt_addendum_file is not None
                else None
            ),
        },
        "datasets": {
            "treatment": {
                "path": str(dataset_dir),
                "tree_sha256": _tree_sha256(dataset_dir),
            },
            "sandbox": {
                "path": str(sandbox_dataset_dir),
                "tree_sha256": _tree_sha256(sandbox_dataset_dir),
            },
            "audit": {
                "path": str(audit_dataset_dir),
                "tree_sha256": _tree_sha256(audit_dataset_dir),
                "trajectory": audit_trajectory,
            },
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        },
        "host": socket.gethostname(),
    }
    timings: dict[str, float] = {}
    agentic_stage_dir = out_dir / "agentic"
    agentic_stage_dir.mkdir(parents=True, exist_ok=True)
    generated_code_path = out_dir / "simulator_gen.py"

    if args.load_existing is not None:
        print(f"=== Loading existing simulator_gen from {args.load_existing} ===")
        generated_code_path = args.load_existing.resolve()
        gen_meta = {
            "skipped_codegen": True,
            "loaded_from": str(generated_code_path),
        }
    else:
        if "GEMINI_API_KEY" not in os.environ:
            print(
                "ERROR: GEMINI_API_KEY is not set. Source the .env first:\n"
                "  set -a; source ../.env; set +a\n"
                "  export GEMINI_API_KEY=\"$GOOGLE_API_KEY\"",
                file=sys.stderr,
            )
            return 2

        print("=== P1 smoke: codegen via VLM ===")
        _print_spec(args.benchmark, bench, data, fit_a, fit_b)
        print(f"  model:           {args.model} (backend=gemini_api)")
        print(f"  prompts_dir:     {prompts_dir}")
        print("  no caption, no test-time image attached")

        vlm = VLMClient(
            model_name=args.model,
            temperature=0.0,
            backend="gemini_api",
            llm_interactions_dir=str(out_dir / "vlm_logs"),
        )

        audit_is_treatment = audit_dataset_dir == dataset_dir
        train_idxs = [
            i
            for i in range(data["n_traj"])
            if not (audit_is_treatment and i == audit_trajectory)
        ]
        auto_validate_trajs = _auto_validate_trajectories(train_idxs)
        deployed_bounds = bench.deployed_bounds_copy()
        print(
            f"  auto-validate trajectories: {auto_validate_trajs} "
            + (
                f"(audit t{audit_trajectory} excluded from treatment)"
                if audit_is_treatment
                else "(external audit unavailable to codegen tools)"
            )
        )

        generator = PlanningAgenticGenerator(
            vlm=vlm,
            prompts_path=str(prompts_dir),
            max_turns=MAX_TURNS,
            n_frames=bench.n_frames,
            start_image_path=str(fit_a.resolve()),
            goal_image_path=str(fit_b.resolve()),
            dataset_dir=dataset_dir,
            fps=FPS,
            frame_size=bench.frame_size,
            simulator_class_name=SIMULATOR_CLASS_NAME,
            output_dir=str(out_dir),
            cache_dir=None,
            available_tools=available_tools,
            caption=CAPTION,
            provide_image=False,
            restricted_tools=False,
            no_api=args.no_world_api,
            auto_validate_trajectories=auto_validate_trajs,
            gate_trajectories=train_idxs,
            gate_max_retries=GATE_MAX_RETRIES,
            gate_retry_turns=GATE_RETRY_TURNS,
            deployed_action_bounds=deployed_bounds,
            gate_max_worst_ratio=SHARED_GATE_MAX_WORST_RATIO,
            gate_realizability_margin=GATE_REALIZABILITY_MARGIN,
            gate_require_action_realizability=True,
            gate_require_goal_invariance=True,
            gate_require_cem_suite=args.gate_cem_suite,
            gate_require_p2_safety=args.gate_p2_safety,
            keep_best=not args.no_keep_best,
            deployment_goal_mode=args.deployment_goal_mode,
            gate_ratio_advisory=args.gate_ratio_advisory,
            gate_require_state_consistency=args.gate_state_consistency_required,
            gate_require_fit_quality=args.gate_fit_quality_required,
            gate_fit_quality_max_rmse=args.gate_fit_quality_max_rmse,
            gate_fit_quality_max_scene_frac=args.gate_fit_quality_max_scene_frac,
            gate_fit_quality_mode=args.gate_fit_quality_mode,
            gate_state_consistency_skip_latent=(
                not args.gate_state_consistency_include_latent
            ),
            keep_best_include_cem=(
                bool(args.gate_cem_suite) and not args.no_keep_best_include_cem
            ),
            gate_workflow_evidence=args.gate_workflow_evidence,
            enable_local_toolbox_tools=not args.disable_local_toolbox_tools,
            prompt_suffix=prompt_suffix,
        )
        ratio_desc = (
            f"worst_ratio<={SHARED_GATE_MAX_WORST_RATIO:.3f}"
            + (" [advisory]" if args.gate_ratio_advisory else "")
        )
        keep_best_desc = "off" if args.no_keep_best else "on"
        if not args.no_keep_best and args.gate_cem_suite and not args.no_keep_best_include_cem:
            keep_best_desc += "+cem"
        print(
            f"  final-gate trajectories: {train_idxs} "
            f"(no-crash + {ratio_desc} "
            "+ realizability + goal-invariance + target-independence"
            f"{' + cem-suite' if args.gate_cem_suite else ''}"
            f"{' + runtime-safety' if args.gate_p2_safety else ''}"
            f"{' + state-consistency' if args.gate_state_consistency_required else ''}"
            f"{' + fit-quality<=' + str(args.gate_fit_quality_max_rmse) + '(' + args.gate_fit_quality_mode + ')' if args.gate_fit_quality_required else ''}"
            f"{' + deployment-goal(' + args.deployment_goal_mode + ')' if args.deployment_goal_mode != 'final' else ''}, "
            f"max_retries={GATE_MAX_RETRIES}, retry_turns={GATE_RETRY_TURNS}, "
            f"keep_best={keep_best_desc}, workflow={args.gate_workflow_evidence})"
        )

        codegen_t0 = time.perf_counter()
        result = generator.generate(
            target_image_path=str(fit_a.resolve()),
            stage_dir=str(agentic_stage_dir),
        )
        timings["codegen_seconds"] = time.perf_counter() - codegen_t0

        with open(generated_code_path, "w", encoding="utf-8") as f:
            f.write(result.code)
        print(f"\n  saved generated simulator -> {generated_code_path}")
        print(f"  tool calls: {result.tool_call_count}")
        print(f"  input tokens:  {result.input_tokens}")
        print(f"  output tokens: {result.output_tokens}")
        print(f"  cached tokens: {result.cached_tokens}")
        gate_str = {True: "PASS", False: "FAIL", None: "n/a"}[result.gate_passed]
        print(f"  final fidelity gate: {gate_str}")
        if result.gate_summary:
            print("  " + result.gate_summary.replace("\n", "\n  "))
        if result.gate_passed is False:
            print(
                "  >>> WARNING: shipped simulator FAILED the final gate. "
                "Treat this run as a FAILED P1 generation, not a result."
            )

        gen_meta = {
            "skipped_codegen": False,
            "tool_call_count": result.tool_call_count,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "cached_tokens": result.cached_tokens,
            "gate_passed": result.gate_passed,
            "gate_summary": result.gate_summary,
            "gate": _structured_gate_summary(result.gate_summary),
            "api_retry_count": result.api_retry_count,
            "api_error_count": result.api_error_count,
        }

    print("\n=== Loading generated class and running eval ===")
    sim_class = _load_generated_class(generated_code_path, SIMULATOR_CLASS_NAME)
    print(f"  loaded class: {sim_class.__name__}")

    from vdaworld.core.dataset_tools import read_trajectory

    held_n_steps = read_trajectory(audit_dataset_dir, audit_trajectory)["n_steps"]
    cem_kwargs = dict(
        action_dim=data["action_dim"],
        action_low=data["action_low"],
        action_high=data["action_high"],
        horizon=held_n_steps,
        **CEM_PLANNER,
    )

    print(
        f"\n--- hidden audit eval (trajectory_{audit_trajectory:04d}, "
        f"horizon={held_n_steps}) ---"
    )
    eval_api = WorldAPI(output_dir=None)
    heldout_t0 = time.perf_counter()
    try:
        held = eval_held_out(
            sim_class,
            dataset_dir=audit_dataset_dir,
            traj_idx=audit_trajectory,
            cem_kwargs=cem_kwargs,
            frame_size=bench.frame_size,
            api=eval_api,
        )
        print(f"  image_distance: {held['image_distance']:.2f}")
        print(f"  action_distance: {held['action_distance']:.2f}")
        print(f"  CEM final elite_mean: {held['cem_history'][-1].elite_mean_cost:.3f}")
        print(f"  CEM final best:        {held['cem_history'][-1].best_cost:.3f}")
        render_side_by_side(
            held["predicted_frames"],
            held["true_frames"],
            out_dir / "held_out.mp4",
            fps=5,
        )
        _plot_cem_history(
            held["cem_history"],
            out_dir / "cem_held_out.png",
            f"CEM (hidden audit traj_{audit_trajectory:04d})",
        )
        Image.fromarray(held["predicted_frames"][-1]).save(
            out_dir / "held_out_predicted_final.png"
        )
        Image.fromarray(held["true_frames"][-1]).save(out_dir / "held_out_true_final.png")
        held_metrics = {
            "image_distance": held["image_distance"],
            "action_distance": held["action_distance"],
            "cem_history": [asdict(h) for h in held["cem_history"]],
        }
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        held_metrics = {"error": f"{type(exc).__name__}: {exc}"}
    timings["audit_eval_seconds"] = time.perf_counter() - heldout_t0

    print("\n--- training-fidelity sweep (all trajectories) ---")
    fidelity_t0 = time.perf_counter()
    try:
        from score_simulator_fidelity import score_one, summarise

        sweep = score_one(
            generated_code_path,
            dataset_dir,
            SIMULATOR_CLASS_NAME,
            bench.frame_size,
            FPS,
            audit_trajectory if audit_dataset_dir == dataset_dir else -1,
            no_api=args.no_world_api,
        )
        training_fidelity = summarise(sweep)
        per = training_fidelity["per_traj"]

        print(
            "  per-traj(train): "
            + " ".join(
                f"t{t}={_fmt_ratio(per[t])}"
                for t in sorted(per)
                if not (
                    audit_dataset_dir == dataset_dir and t == audit_trajectory
                )
            )
        )
        print(
            "  per-traj(all):   "
            + " ".join(f"t{t}={_fmt_ratio(per[t])}" for t in sorted(per))
            + (
                f"  (audit=t{audit_trajectory})"
                if audit_dataset_dir == dataset_dir
                else "  (audit is external)"
            )
        )
        stats = training_fidelity.get("train_stats", {})
        print(
            f"  TRAIN mean={_fmt_ratio(training_fidelity['train_mean'])}  "
            f"median={_fmt_ratio(stats.get('median'))}  "
            f"p75={_fmt_ratio(stats.get('p75'))}  "
            f"WORST={_fmt_ratio(training_fidelity['train_worst'])}  "
            f"best={_fmt_ratio(training_fidelity['train_best'])}  "
            f"n={stats.get('n_trajectories', 'n/a')}  "
            f"frac<1.0={_fmt_frac(stats.get('frac_below_1.0'))}  "
            f"frac<0.5={_fmt_frac(stats.get('frac_below_0.5'))}  "
            + (
                f"audit(t{audit_trajectory})="
                f"{_fmt_ratio(training_fidelity['held_out_ratio'])}"
                if audit_is_treatment
                else "audit=external (reported in held_out section)"
            )
        )
        print(
            "  candidate criteria: "
            + _format_candidate_criteria(training_fidelity.get("candidate_criteria", {}))
        )
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        training_fidelity = {"error": f"{type(exc).__name__}: {exc}"}
    timings["training_fidelity_seconds"] = time.perf_counter() - fidelity_t0

    runtime_contract = {
        "contract_version": 2,
        "simulator_class": SIMULATOR_CLASS_NAME,
        "frame_size": list(bench.frame_size),
        "fps": FPS,
        "fit_world_api_enabled": not bool(args.no_world_api),
        "rollout_world_api_enabled": False,
        "local_2d_toolbox_version": LOCAL_2D_TOOLBOX_VERSION,
    }
    with open(out_dir / "runtime_contract.json", "w", encoding="utf-8") as f:
        json.dump(runtime_contract, f, indent=2)

    timings["total_wall_seconds"] = time.perf_counter() - process_t0
    finished_at = _utc_now()
    scientific_status = (
        "valid_gate_pass"
        if gen_meta.get("gate_passed") is True
        else "scientific_gate_failure"
        if gen_meta.get("gate_passed") is False
        else "not_gated"
    )
    metrics = {
        "schema_version": P1_METRICS_SCHEMA_VERSION,
        "benchmark": args.benchmark,
        "repeat_id": args.repeat_id,
        "composition": args.composition,
        "scientific_status": scientific_status,
        "started_at_utc": process_started_at,
        "finished_at_utc": finished_at,
        "timings": timings,
        "provenance": {
            **provenance,
            "simulator_sha256": _file_sha256(generated_code_path),
            "runtime_contract_sha256": _file_sha256(
                out_dir / "runtime_contract.json"
            ),
        },
        "benchmark_spec": {
            "dataset_dir": str(dataset_dir),
            "sandbox_dataset_dir": str(sandbox_dataset_dir),
            "audit_dataset_dir": str(audit_dataset_dir),
            "frame_size": list(bench.frame_size),
            "n_frames": bench.n_frames,
            "sandbox_fit_trajectory": SANDBOX_FIT_TRAJECTORY,
            "sandbox_fit_frame_a": bench.sandbox_fit_frame_a,
            "sandbox_fit_frame_b": bench.sandbox_fit_frame_b,
            "audit_trajectory": audit_trajectory,
        },
        "codegen": gen_meta,
        "held_out": held_metrics,
        "training_fidelity": training_fidelity,
        "runtime_contract": runtime_contract,
    }
    metrics_tmp = out_dir / "metrics.json.tmp"
    with open(metrics_tmp, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics_tmp.replace(out_dir / "metrics.json")
    running_marker = out_dir / "RUNNING.json"
    if running_marker.exists():
        running_marker.unlink()
    (out_dir / "COMPLETED.json").write_text(
        json.dumps(
            {
                **status_base,
                "status": "completed",
                "scientific_status": scientific_status,
                "gate_passed": gen_meta.get("gate_passed"),
                "finished_at_utc": finished_at,
                "wall_seconds": timings["total_wall_seconds"],
                "metrics_sha256": _file_sha256(out_dir / "metrics.json"),
                "simulator_sha256": _file_sha256(generated_code_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    run_state["complete"] = True

    print(f"\nArtifacts at {out_dir}/")
    for name in sorted(p.name for p in out_dir.iterdir()):
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
