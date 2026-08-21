from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.test_calibrate_parameters import _GOOD_SIM, _build_dataset, _make_sandbox


def test_analyze_action_effects_reports_measured_correlations(tmp_path):
    dataset = tmp_path / "dataset"
    _build_dataset(dataset)
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)

    summary = sandbox.analyze_action_effects([0, 1], max_trajectories=2)

    assert "[analyze_action_effects]" in summary
    assert "action_dim=2" in summary
    assert "strongest measured action effects" in summary
    assert "red." in summary or "foreground." in summary
    assert "full-training action ranges" in summary
    assert "Advisory only" in summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_selector_penalizes_drift_over_nominal_gate_pass(tmp_path):
    stamp = "selector_test"
    run_label = "h5_a2_cem300x30_seeds42_46"
    root = tmp_path
    p1 = root / "viz_output" / "experiments" / stamp / "p1" / "pusht" / "expert_only"
    p2 = root / "viz_output" / "experiments" / stamp / "p2" / "pusht" / "expert_only"

    _write_json(
        p1 / "repeat_1" / "metrics.json",
        {
            "codegen": {
                "gate_passed": True,
                "gate_summary": (
                    "final fidelity gate: PASS | cem_suite=ok deploy_goal=ok "
                    "state_consistency=ok fit_quality=ok | worst_ratio=0.4 mean_ratio=0.3\n"
                    "state-consistency (advisory): t0: block=DRIFT(900%@t4)"
                ),
            },
            "training_fidelity": {"train_worst": 0.4, "train_mean": 0.3},
        },
    )
    _write_json(
        p1 / "repeat_1" / "agentic" / "tool_calls" / "fit_quality_gate.json",
        {"worst_normalized_rmse": 0.1, "mean_normalized_rmse": 0.1},
    )
    _write_json(
        p2 / "repeat_1" / run_label / "summary.json",
        {"success_rate": 0.0, "pusht_max_coverage_best": 0.1},
    )

    _write_json(
        p1 / "repeat_2" / "metrics.json",
        {
            "codegen": {
                "gate_passed": False,
                "gate_summary": (
                    "final fidelity gate: FAIL | cem_suite=ok deploy_goal=ok "
                    "state_consistency=ok fit_quality=ok | worst_ratio=0.6 mean_ratio=0.4"
                ),
            },
            "training_fidelity": {"train_worst": 0.6, "train_mean": 0.4},
        },
    )
    _write_json(
        p1 / "repeat_2" / "agentic" / "tool_calls" / "fit_quality_gate.json",
        {"worst_normalized_rmse": 0.1, "mean_normalized_rmse": 0.1},
    )
    _write_json(
        p2 / "repeat_2" / run_label / "summary.json",
        {"success_rate": 0.0, "pusht_max_coverage_best": 0.3},
    )

    script = Path(__file__).resolve().parents[1] / "scripts" / "select_gate_best_repeats.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--stamp",
            stamp,
            "--run-label",
            run_label,
            "--benchmarks",
            "pusht",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "selected=repeat_2" in result.stdout
    selected = json.loads(
        (root / "results" / stamp / "pusht" / "selected_simulator.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected["repeat"] == "repeat_2"
