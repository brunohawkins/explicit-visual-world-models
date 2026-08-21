"""Tests for the final fidelity gate (crash + worst-ratio + realizability + goal invariance).

Covers two layers:

1. ``PlanningCriticSandbox.final_fidelity_gate`` — verdict over a set of
   trajectories with three checks:
   (a) no crashes under replay, (b) worst reduction_ratio under a threshold,
   (c) offline action-realizability probe passes, and
   (d) target_state is invariant when only the observation frame changes.

2. ``PlanningAgenticGenerator._make_finish_guard`` / ``_final_gate_record`` — the
   guard is disabled (returns ``None``) when no gate trajectories are configured
   (so two-rooms is unaffected), and when enabled it returns a complaint up to
   ``gate_max_retries`` times, then yields (lets the model finish; the failure is
   recorded loudly by ``_final_gate_record``).

The synthetic dataset + GOOD simulator are reused from the calibrate tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_calibrate_parameters import (
    FRAME,
    TRUE_GAIN,
    _GOOD_SIM,
    _build_dataset,
    _make_sandbox,
)
from vdaworld.core.planning_agentic_generator import PlanningAgenticGenerator
from vdaworld.core.planning_critic_toolbox import _generic_fit_quality_color_masks

# A simulator that runs fit/loss/render but RAISES inside update() — i.e. it
# crashes under ground-truth-action replay, exactly like the 2026-06-03 shipped
# sim that called a hallucinated pymunk API.
_CRASH_SIM = _GOOD_SIM.replace(
    '        self.state["p"] = self.state["p"] + self.params["gain"]'
    " * np.asarray(a, dtype=float)",
    '        raise RuntimeError("hallucinated API call boom")',
)
assert "RuntimeError" in _CRASH_SIM, "crash-sim patch did not apply"

_TRUE_GAIN_SIM = _GOOD_SIM.replace(
    'self.params = {"gain": 0.2}',
    f'self.params = {{"gain": {TRUE_GAIN}}}',
)

_SHIFTED_RENDER_SIM = _TRUE_GAIN_SIM.replace(
    'x, y = int(round(self.state["p"][0])), int(round(self.state["p"][1]))',
    'x, y = int(round(self.state["p"][0] + 10)), int(round(self.state["p"][1]))',
)

_GOAL_DRIFT_SIM = _TRUE_GAIN_SIM.replace(
    '        self.target_state = {"p": self._centroid(image_B)}',
    '        self.target_state = {"p": self._centroid(image_A)}',
)

_EXPLOIT_SIM = _TRUE_GAIN_SIM.replace(
    '        self.state["p"] = self.state["p"] + self.params["gain"] * np.asarray(a, dtype=float)',
    '        a = np.asarray(a, dtype=float)\n'
    '        step = self.params["gain"] * a\n'
    "        # Out-of-distribution exploit: huge one-step jump on extreme actions.\n"
    "        if float(np.linalg.norm(a)) > 8.0:\n"
    "            step = 20.0 * a\n"
    '        self.state["p"] = self.state["p"] + step',
)

_API_DEPENDENT_SIM = _TRUE_GAIN_SIM.replace(
    '        self.state["p"] = self.state["p"] + self.params["gain"] * np.asarray(a, dtype=float)',
    '        self.state["p"] = self.api.must_not_exist(self.state["p"], a)',
)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    _build_dataset(root)
    return root


# --- layer 1: final_fidelity_gate --------------------------------------------

def test_good_sim_passes_full_gate(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    verdict = sandbox.final_fidelity_gate([0, 1, 2])
    assert verdict["passed"] is True, verdict
    assert verdict["crash_ok"] is True
    assert verdict["ratio_ok"] is True
    assert verdict["realizability_ok"] is True
    assert verdict["goal_invariance_ok"] is True
    assert verdict["complaint"] == ""
    assert all(r is not None for r in verdict["per_traj"].values()), verdict
    assert verdict["ratio_stats"]["n_trajectories"] == 3
    assert verdict["ratio_stats"]["frac_below_1.0"] == pytest.approx(1.0)
    assert set(verdict["candidate_criteria"]) == {"C1", "C2", "C3", "C4"}
    assert all(item["passed"] for item in verdict["candidate_criteria"].values())


def test_poor_ratio_sim_fails_ratio_gate(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOOD_SIM, dataset)
    verdict = sandbox.final_fidelity_gate([0, 1, 2], max_worst_ratio=0.5)
    assert verdict["passed"] is False, verdict
    assert verdict["crash_ok"] is True
    assert verdict["ratio_ok"] is False
    assert "worst reduction_ratio" in verdict["complaint"]
    assert verdict["candidate_criteria"]["C1"]["passed"] is False
    assert verdict["ratio_stats"]["median"] is not None


def test_exploit_sim_fails_realizability_gate(tmp_path, dataset):
    sandbox = _make_sandbox(
        tmp_path,
        _EXPLOIT_SIM,
        dataset,
        deployed_action_bounds=(
            np.array([-20.0, -20.0]),
            np.array([20.0, 20.0]),
        ),
    )
    verdict = sandbox.final_fidelity_gate(
        [0, 1, 2],
        max_worst_ratio=None,
        require_action_realizability=True,
    )
    assert verdict["passed"] is False, verdict
    assert verdict["crash_ok"] is True
    assert verdict["realizability_ok"] is False
    assert "realizability" in verdict["complaint"]


def test_goal_drift_sim_fails_goal_invariance_gate(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _GOAL_DRIFT_SIM, dataset)
    verdict = sandbox.final_fidelity_gate(
        [0, 1, 2],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=True,
    )
    assert verdict["passed"] is False, verdict
    assert verdict["goal_invariance_ok"] is False
    assert "goal-invariance" in verdict["complaint"]


def test_crashing_sim_fails_with_complaint(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _CRASH_SIM, dataset)
    verdict = sandbox.final_fidelity_gate([0, 1, 2])
    assert verdict["passed"] is False, verdict
    assert verdict["n_failed"] == 3  # crashes on every trajectory
    assert all(r is None for r in verdict["per_traj"].values()), verdict
    assert "FINAL VALIDATION GATE" in verdict["complaint"]
    assert "t0" in verdict["complaint"]
    assert "RuntimeError" in verdict["complaint"] or "exited" in verdict["complaint"]


def test_p2_safety_passes_for_api_free_sim(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    verdict = sandbox.final_fidelity_gate(
        [0, 1],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=False,
        require_p2_safety=True,
    )
    assert verdict["p2_safety_ok"] is True, verdict
    assert verdict["p2_safety"]["passed"] is True


def test_p2_safety_fails_for_api_dependent_update(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _API_DEPENDENT_SIM, dataset)
    verdict = sandbox.final_fidelity_gate(
        [0, 1],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=False,
        require_p2_safety=True,
    )
    assert verdict["passed"] is False, verdict
    assert verdict["p2_safety_ok"] is False
    assert "runtime-safety" in verdict["complaint"]


def test_cem_suite_gate_records_result_when_enabled(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    sandbox._run_cem_suite_check = lambda trajectories: {
        "passed": False,
        "results": [
            {
                "passed": False,
                "pair": {
                    "start_trajectory_index": 0,
                    "start_frame_index": 0,
                    "goal_trajectory_index": 1,
                    "goal_frame_index": 8,
                },
                "summary": "FAIL: synthetic CEM suite failure",
            }
        ],
    }
    verdict = sandbox.final_fidelity_gate(
        [0, 1],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=False,
        require_cem_suite=True,
    )
    assert "cem_suite_ok" in verdict
    assert verdict["passed"] is False
    assert verdict["cem_suite_ok"] is False
    assert verdict["cem_suite"] is not None
    assert verdict["cem_suite"]["results"]
    assert "CEM suite" in verdict["complaint"]


def test_state_consistency_required_blocks_drifting_components(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    sandbox._run_state_consistency_check = lambda trajectory_index, max_checkpoints=8: {
        "ok": True,
        "trajectory": int(trajectory_index),
        "drifting_components": ["p"],
        "parse_errors": {},
        "oneline": "t0: p=DRIFT(200%@t4)",
        "compact_lines": ["p: DRIFT — gap grows to 10 (200% of its motion)"],
    }
    verdict = sandbox.final_fidelity_gate(
        [0, 1],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=False,
        require_state_consistency=True,
    )
    assert verdict["passed"] is False, verdict
    assert verdict["state_consistency_ok"] is False
    assert "state-consistency" in verdict["complaint"]


def test_fit_quality_required_blocks_bad_fit_render_match(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    sandbox._run_fit_quality_check = lambda trajectories, max_rmse, max_probes=3: {
        "passed": False,
        "max_rmse": float(max_rmse),
        "worst_normalized_rmse": 0.9,
        "mean_normalized_rmse": 0.8,
        "n_probes": 2,
        "n_failed": 0,
        "probes": [],
    }
    verdict = sandbox.final_fidelity_gate(
        [0, 1],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=False,
        require_fit_quality=True,
        fit_quality_max_rmse=0.3,
    )
    assert verdict["passed"] is False, verdict
    assert verdict["fit_quality_ok"] is False
    assert "fit-quality" in verdict["complaint"]


def test_fit_quality_all_failed_returns_repair_complaint(tmp_path, dataset):
    """A debug/broken simulator must receive repair turns, not crash the guard."""
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    sandbox._run_fit_quality_check = lambda trajectories, max_rmse, max_probes=3: {
        "passed": False,
        "max_rmse": float(max_rmse),
        "worst_normalized_rmse": None,
        "mean_normalized_rmse": None,
        "n_probes": 3,
        "n_failed": 3,
        "n_visibility_failures": 0,
        "probes": [],
    }
    verdict = sandbox.final_fidelity_gate(
        [0, 1],
        max_worst_ratio=None,
        require_action_realizability=False,
        require_goal_invariance=False,
        require_fit_quality=True,
        fit_quality_max_rmse=0.3,
    )
    assert verdict["passed"] is False
    assert verdict["fit_quality_ok"] is False
    assert "worst_normalized_rmse=n/a" in verdict["complaint"]


def test_fit_quality_probes_start_middle_and_late_frames(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)

    verdict = sandbox._run_fit_quality_check([0], max_rmse=0.1, max_probes=3)

    assert verdict["passed"] is True, verdict
    assert [probe["phase"] for probe in verdict["probes"]] == [
        "start",
        "middle",
        "late",
    ]
    assert verdict["n_perception_failures"] == 0


def test_fit_quality_rejects_shift_hidden_by_background_rmse(tmp_path, dataset):
    sandbox = _make_sandbox(tmp_path, _SHIFTED_RENDER_SIM, dataset)

    verdict = sandbox._run_fit_quality_check([0], max_rmse=0.1, max_probes=3)

    assert verdict["worst_normalized_rmse"] < 0.1
    assert verdict["passed"] is False
    assert verdict["n_perception_failures"] > 0
    assert any(
        check["label"] == "red" and not check["passed"]
        for probe in verdict["probes"]
        for check in probe.get("component_checks", [])
    )


def test_fit_quality_blue_proxy_requires_saturation_and_dominance():
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    image[3, 3] = [143, 163, 184]
    image[4, 4] = [78, 126, 255]

    masks, _ = _generic_fit_quality_color_masks(image)

    assert not bool(masks["blue"][3, 3])
    assert bool(masks["blue"][4, 4])


# --- layer 2: finish-guard wiring -------------------------------------------

def _make_generator(dataset, gate_trajectories, gate_max_retries=2, **kwargs):
    """A PlanningAgenticGenerator with no real VLM — only the gate hooks are
    exercised, and they never touch the vlm/prompt machinery."""
    start = dataset / "trajectory_0000" / "frame_0000.png"
    goal = dataset / "trajectory_0000" / "frame_0008.png"
    return PlanningAgenticGenerator(
        vlm=None,
        prompts_path="",
        max_turns=5,
        n_frames=8,
        start_image_path=str(start),
        goal_image_path=str(goal),
        dataset_dir=str(dataset),
        fps=10,
        frame_size=(FRAME, FRAME),
        simulator_class_name="Simulator",
        output_dir="",
        cache_dir=None,
        available_tools="",
        caption="",
        gate_trajectories=gate_trajectories,
        gate_max_retries=gate_max_retries,
        **kwargs,
    )


def test_guard_without_gate_trajectories_skips_fidelity_gate(tmp_path, dataset):
    """No gate trajectories => finish guard and fidelity record stay opted out."""
    gen = _make_generator(dataset, gate_trajectories=None)
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    guard = gen._make_finish_guard(sandbox)
    assert guard is None
    assert gen._final_gate_record(sandbox) == (None, "")


def test_guard_complains_then_yields_after_budget(tmp_path, dataset):
    gen = _make_generator(dataset, gate_trajectories=[0, 1], gate_max_retries=2)
    sandbox = _make_sandbox(tmp_path, _CRASH_SIM, dataset)
    guard = gen._make_finish_guard(sandbox)
    assert guard is not None
    # First two voluntary finishes are blocked with a complaint. The model
    # "attempts repairs" (tool calls) between finishes, so the engagement
    # nudge stays out of the way and each finish consumes a retry.
    assert guard(1)
    sandbox._tool_call_idx += 1
    assert guard(2)
    sandbox._tool_call_idx += 1
    # ...then the retry budget is spent and the model is allowed to finish.
    assert guard(3) is None


def test_guard_grants_bounded_repairs_when_gate_errors(tmp_path, dataset):
    gen = _make_generator(dataset, gate_trajectories=[0, 1], gate_max_retries=2)
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)

    def broken_gate(_trajectories):
        raise TypeError("synthetic gate formatting failure")

    sandbox.final_fidelity_gate = broken_gate
    guard = gen._make_finish_guard(sandbox)
    assert guard is not None

    first = guard(1)
    assert first and "could not be evaluated" in first
    assert "TypeError" in first
    sandbox._tool_call_idx += 1
    second = guard(2)
    assert second and "could not be evaluated" in second
    sandbox._tool_call_idx += 1
    assert guard(3) is None


def test_guard_nudges_summary_only_replies_without_burning_retries(
    tmp_path, dataset
):
    """A gate complaint answered with prose only (no tool calls) must get a
    'call a tool' nudge that does NOT consume a repair retry (2026-07-07:
    Push-T r2 burned all three retries on identical status summaries)."""
    gen = _make_generator(dataset, gate_trajectories=[0, 1], gate_max_retries=2)
    sandbox = _make_sandbox(tmp_path, _CRASH_SIM, dataset)
    guard = gen._make_finish_guard(sandbox)

    first = guard(1)
    assert first and "MANDATORY FINAL VALIDATION GATE" in first
    # No tool calls since the complaint -> nudge, not a second complaint.
    nudge = guard(2)
    assert nudge and "MUST be a tool call" in nudge
    assert "MANDATORY FINAL VALIDATION GATE" not in nudge
    # Model engages -> normal complaint path resumes (retry 2 of 2)...
    sandbox._tool_call_idx += 1
    second = guard(3)
    assert second and "MANDATORY FINAL VALIDATION GATE" in second
    # ...and after the budget is spent the model may finish.
    sandbox._tool_call_idx += 1
    assert guard(4) is None


def test_guard_requires_perception_and_dynamics_evidence(tmp_path, dataset):
    gen = _make_generator(
        dataset,
        gate_trajectories=[0, 1],
        gate_require_fit_quality=True,
        gate_require_state_consistency=True,
    )
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    guard = gen._make_finish_guard(sandbox)
    assert guard is not None

    complaint = guard(1)

    assert complaint and "MANDATORY EVIDENCE CHECK" in complaint
    assert "analyze_action_effects" in complaint
    assert "two distinct" in complaint
    assert "validate_state_consistency" in complaint


def test_ratio_advisory_gate_lets_planner_checks_decide(tmp_path, dataset):
    """With gate_ratio_advisory, a poor worst-ratio is reported but does not
    block the gate; the planner-relevant criteria drive pass/fail."""
    sandbox = _make_sandbox(
        tmp_path, _GOOD_SIM, dataset, gate_ratio_advisory=True
    )
    verdict = sandbox.final_fidelity_gate([0, 1, 2], max_worst_ratio=0.5)
    assert verdict["ratio_ok"] is False
    assert verdict["ratio_advisory"] is True
    assert verdict["passed"] is True, verdict
    assert verdict["complaint"] == ""
    table = PlanningAgenticGenerator._format_gate_table(verdict)
    assert "ratio=adv-fail" in table


def test_ratio_advisory_complaint_marks_fidelity_advisory(tmp_path, dataset):
    """When another criterion fails, the complaint labels the ratio line
    ADVISORY so the model prioritises the real failures."""
    sandbox = _make_sandbox(
        tmp_path, _GOAL_DRIFT_SIM, dataset, gate_ratio_advisory=True
    )
    verdict = sandbox.final_fidelity_gate([0, 1, 2], max_worst_ratio=0.001)
    assert verdict["passed"] is False  # goal invariance fails
    assert "ADVISORY (fidelity)" in verdict["complaint"]
    assert "FAIL (fidelity)" not in verdict["complaint"]


def test_api_internal_error_tolerated_twice_then_aborts(tmp_path, dataset):
    from vdaworld.utils.error_handling import AbortLoopError

    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    msg1 = sandbox._register_api_internal_error()
    assert "do NOT keep relying" in msg1
    msg2 = sandbox._register_api_internal_error()
    assert "do NOT keep relying" in msg2
    with pytest.raises(AbortLoopError, match="persistent"):
        sandbox._register_api_internal_error()


def test_final_gate_record_flags_crash(tmp_path, dataset):
    gen = _make_generator(dataset, gate_trajectories=[0, 1])
    sandbox = _make_sandbox(tmp_path, _CRASH_SIM, dataset)
    passed, summary = gen._final_gate_record(sandbox)
    assert passed is False
    assert "FAIL" in summary
    assert "crash=fail" in summary


def test_guard_passes_for_running_sim(tmp_path, dataset):
    gen = _make_generator(dataset, gate_trajectories=[0, 1])
    sandbox = _make_sandbox(tmp_path, _TRUE_GAIN_SIM, dataset)
    guard = gen._make_finish_guard(sandbox)
    assert guard(1) is None
    passed, summary = gen._final_gate_record(sandbox)
    assert passed is True
    assert "PASS" in summary
