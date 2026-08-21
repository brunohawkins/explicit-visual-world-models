#!/usr/bin/env python3
"""Demo: Phase 1 — a VLM writes a Two-Room simulator from eight expert clips.

Needs GEMINI_API_KEY. This is one generation attempt, not the 80-run campaign.
It will not bit-match the thesis numbers (hosted model, no dated snapshot).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print(
            "P1 needs a Gemini API key.\n"
            "  export GEMINI_API_KEY=...\n"
            "If you only want to check that control works, run demo_p2.py instead "
            "(no key required)."
        )
        return 2

    out = ROOT / "viz_output" / "demo_p1"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "smoke_p1.py"),
        "--benchmark",
        "two_room_lewm",
        "--dataset-dir",
        str(ROOT / "dataset" / "two_room" / "expert_only"),
        "--sandbox-dataset-dir",
        str(ROOT / "dataset" / "two_room" / "_sandbox"),
        "--audit-dataset-dir",
        str(ROOT / "dataset" / "two_room" / "_audit"),
        "--audit-trajectory",
        "0",
        "--composition",
        "expert_only",
        "--repeat-id",
        "demo",
        "--model",
        "gemini-3.6-flash",
        "--prompts-dir",
        str(ROOT / "prompts" / "action_conditioned"),
        "--prompt-addendum-file",
        str(ROOT / "prompts" / "action_conditioned" / "legacy_tool_lite_addendum.md"),
        "--out-dir",
        str(out),
        "--no-world-api",
        "--gate-p2-safety",
        "--gate-cem-suite",
        "--gate-fit-quality-required",
        "--gate-fit-quality-max-rmse",
        "0.30",
        "--gate-fit-quality-mode",
        "rmse_plus_foreground",
        "--gate-fit-quality-max-scene-frac",
        "0.35",
        "--gate-workflow-evidence",
        "off",
        "--overwrite-output",
    ]
    print("[demo_p1] writing a simulator from eight Two-Room clips")
    print("[demo_p1] output:", out)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["GEMINI_API_KEY"] = key
    return subprocess.call(cmd, cwd=str(ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
