#!/usr/bin/env python3
"""Demo: Phase 2 — plan in a frozen Two-Room simulator and act in the env.

No API key. Uses the campaign's expert_only/repeat_1 programme.
Default is a cheap 1-case laptop run. Pass --faithful for thesis CEM settings
on 5 cases (still not the full 50).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--faithful",
        action="store_true",
        help="Use thesis CEM (300 x 30) on seeds 42-46. Slow.",
    )
    parser.add_argument(
        "--sim-path",
        type=Path,
        default=ROOT / "examples" / "two_room_expert_only_r1" / "simulator_gen.py",
        help="Generated simulator to evaluate (default: frozen Two-Room example).",
    )
    args = parser.parse_args()

    sim = args.sim_path.resolve()
    contract = sim.parent / "runtime_contract.json"
    if not sim.is_file():
        print(f"missing simulator: {sim}")
        return 2

    if args.faithful:
        seeds = "42,43,44,45,46"
        cem_pop, cem_iters = "300", "30"
        label = "demo_p2_faithful"
    else:
        seeds = "42"
        cem_pop, cem_iters = "80", "8"
        label = "demo_p2"

    out = ROOT / "viz_output" / label
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "smoke_p2_mpc.py"),
        "--benchmark",
        "two_room_swm",
        "--sim-path",
        str(sim),
        "--runtime-contract",
        str(contract),
        "--manifest-path",
        str(ROOT / "manifests" / "two_room_lewm_seed42_n50_offset25.json"),
        "--out-dir",
        str(out),
        "--run-label",
        label,
        "--eval-protocol",
        "vda_mpc_default",
        "--eval-backend",
        "native",
        "--task-success-protocol",
        "original",
        "--plan-mode",
        "mpc",
        "--plan-horizon",
        "50",
        "--horizon",
        "25",
        "--apply-steps",
        "25",
        "--max-steps",
        "50",
        "--cem-population",
        cem_pop,
        "--cem-iters",
        cem_iters,
        "--elite-frac",
        "0.1",
        "--sim-fps",
        "10",
        "--seeds",
        seeds,
        "--no-world-api",
        "--overwrite-output",
    ]
    print("[demo_p2] planning with a frozen Two-Room simulator (no Gemini key)")
    print("[demo_p2] seeds:", seeds, "CEM:", cem_pop, "x", cem_iters)
    print("[demo_p2] output:", out / label)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
