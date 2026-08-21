"""PlanningAgenticGenerator: AgenticGenerator extension for action-conditioned codegen.

Subclasses :class:`vdaworld.core.agentic_generator.AgenticGenerator` by
overriding two hooks:

* :meth:`_build_sandbox` — instantiates :class:`PlanningCriticSandbox`
  with both image paths + the training-trajectory dataset.
* :meth:`_build_tool_callables` — extends the base tool list with planning
  dataset/validation tools (including ``validate_action_realizability``).

The agentic prompt set for this generator lives at
``prompts/action_conditioned/`` and re-teaches the VLM the new API contract
(``update``, ``fit(A, B)``, ``terminal_cost``) and dataset-tool usage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from vdaworld.core.agentic_generator import AgenticGenerator, _STUB_CODE
from vdaworld.core.critic_toolbox import CriticSandbox
from vdaworld.core.planning_critic_toolbox import PlanningCriticSandbox


class PlanningAgenticGenerator(AgenticGenerator):
    """Drives an agentic codegen loop that produces an ActionConditionedSimulator.

    Args:
        start_image_path: Path to the test-time start PNG (image_A in fit).
        goal_image_path:  Path to the test-time goal PNG (image_B in fit).
        dataset_dir:      Root of the training-trajectory dataset.
        All other args forwarded to :class:`AgenticGenerator`.
    """

    def __init__(
        self,
        vlm,
        prompts_path: str,
        max_turns: int,
        n_frames: int,
        start_image_path: str,
        goal_image_path: str,
        dataset_dir: Union[str, Path],
        fps: int,
        frame_size: tuple[int, int],
        simulator_class_name: str,
        output_dir: str,
        cache_dir: str | None,
        available_tools: str,
        caption: str,
        provide_image: bool = True,
        restricted_tools: bool = False,
        no_api: bool = False,
        no_mhi: bool = False,
        auto_validate_trajectories: list[int] | None = None,
        gate_trajectories: list[int] | None = None,
        gate_max_retries: int = 2,
        gate_retry_turns: int = 6,
        deployed_action_bounds: tuple | None = None,
        gate_max_worst_ratio: float | None = 1.25,
        gate_realizability_margin: float = 1.25,
        gate_require_action_realizability: bool = True,
        gate_require_goal_invariance: bool = True,
        gate_require_cem_suite: bool = False,
        gate_require_p2_safety: bool = False,
        gate_goal_invariance_abs_tol: float = 3.0,
        gate_goal_invariance_rel_tol: float = 0.02,
        gate_require_target_independence: bool = True,
        keep_best: bool = True,
        deployment_goal_mode: str | None = None,
        gate_ratio_advisory: bool = False,
        gate_require_state_consistency: bool = False,
        gate_require_fit_quality: bool = False,
        gate_fit_quality_max_rmse: float = 0.30,
        gate_fit_quality_max_scene_frac: float = 0.35,
        gate_fit_quality_mode: str = "rmse_plus_foreground",
        gate_state_consistency_skip_latent: bool = True,
        keep_best_include_cem: bool = False,
        gate_workflow_evidence: str = "mandatory",
        enable_local_toolbox_tools: bool = True,
        prompt_suffix: str = "",
    ) -> None:
        super().__init__(
            vlm=vlm,
            prompts_path=prompts_path,
            max_turns=max_turns,
            n_frames=n_frames,
            input_image_path=start_image_path,  # used by parent as the first image attached to prompt
            fps=fps,
            frame_size=frame_size,
            simulator_class_name=simulator_class_name,
            output_dir=output_dir,
            cache_dir=cache_dir,
            available_tools=available_tools,
            caption=caption,
            provide_image=provide_image,
            restricted_tools=restricted_tools,
            no_api=no_api,
            no_mhi=no_mhi,
            prompt_suffix=prompt_suffix,
        )
        self._start_image_path = start_image_path
        self._goal_image_path = goal_image_path
        self._dataset_dir = Path(dataset_dir)
        self._auto_validate_trajectories = auto_validate_trajectories
        self._enable_local_toolbox_tools = bool(enable_local_toolbox_tools)
        # Final fidelity gate. Opt-in: None => no gate. When set, the guard
        # refuses a failing finish and can also inject a gate complaint at the
        # soft turn cap with bounded repair headroom.
        self._gate_trajectories = (
            list(gate_trajectories) if gate_trajectories else None
        )
        self._gate_max_retries = gate_max_retries
        self._gate_retry_turns = max(0, int(gate_retry_turns))
        self._deployed_action_bounds = deployed_action_bounds
        self._gate_max_worst_ratio = gate_max_worst_ratio
        self._gate_realizability_margin = gate_realizability_margin
        self._gate_require_action_realizability = gate_require_action_realizability
        self._gate_require_goal_invariance = gate_require_goal_invariance
        self._gate_require_cem_suite = gate_require_cem_suite
        self._gate_require_p2_safety = gate_require_p2_safety
        self._gate_goal_invariance_abs_tol = gate_goal_invariance_abs_tol
        self._gate_goal_invariance_rel_tol = gate_goal_invariance_rel_tol
        self._gate_require_target_independence = gate_require_target_independence
        self._keep_best = keep_best
        self._deployment_goal_mode = deployment_goal_mode
        self._gate_ratio_advisory = gate_ratio_advisory
        self._gate_require_state_consistency = bool(gate_require_state_consistency)
        self._gate_require_fit_quality = bool(gate_require_fit_quality)
        self._gate_fit_quality_max_rmse = float(gate_fit_quality_max_rmse)
        self._gate_fit_quality_max_scene_frac = float(gate_fit_quality_max_scene_frac)
        self._gate_fit_quality_mode = str(gate_fit_quality_mode)
        self._gate_state_consistency_skip_latent = bool(
            gate_state_consistency_skip_latent
        )
        self._keep_best_include_cem = bool(keep_best_include_cem)
        workflow = str(gate_workflow_evidence or "mandatory").strip().lower()
        if workflow not in {"mandatory", "advisory", "off"}:
            raise ValueError(
                "gate_workflow_evidence must be one of mandatory, advisory, off; "
                f"got {gate_workflow_evidence!r}"
            )
        self._gate_workflow_evidence = workflow

    def _build_sandbox(
        self,
        sandbox_dir: str,
        tool_calls_log_dir: str,
        world_api_log_dir: str,
    ) -> CriticSandbox:
        return PlanningCriticSandbox(
            code=_STUB_CODE,
            fps=self._fps,
            n_frames=self._n_frames,
            frame_size=self._frame_size,
            start_image_path=self._start_image_path,
            goal_image_path=self._goal_image_path,
            dataset_dir=self._dataset_dir,
            simulator_class_name=self._simulator_class_name,
            sandbox_dir=sandbox_dir,
            tool_calls_log_dir=tool_calls_log_dir,
            cache_dir=self._cache_dir,
            world_api_log_dir=world_api_log_dir,
            no_api=self._no_api,
            auto_validate_trajectories=self._auto_validate_trajectories,
            gate_trajectories=self._gate_trajectories,
            deployed_action_bounds=self._deployed_action_bounds,
            gate_max_worst_ratio=self._gate_max_worst_ratio,
            gate_realizability_margin=self._gate_realizability_margin,
            gate_require_action_realizability=self._gate_require_action_realizability,
            gate_require_goal_invariance=self._gate_require_goal_invariance,
            gate_require_cem_suite=self._gate_require_cem_suite,
            gate_require_p2_safety=self._gate_require_p2_safety,
            gate_goal_invariance_abs_tol=self._gate_goal_invariance_abs_tol,
            gate_goal_invariance_rel_tol=self._gate_goal_invariance_rel_tol,
            gate_require_target_independence=self._gate_require_target_independence,
            keep_best=self._keep_best,
            deployment_goal_mode=self._deployment_goal_mode,
            gate_ratio_advisory=self._gate_ratio_advisory,
            gate_require_state_consistency=self._gate_require_state_consistency,
            gate_require_fit_quality=self._gate_require_fit_quality,
            gate_fit_quality_max_rmse=self._gate_fit_quality_max_rmse,
            gate_fit_quality_max_scene_frac=self._gate_fit_quality_max_scene_frac,
            gate_fit_quality_mode=self._gate_fit_quality_mode,
            gate_state_consistency_skip_latent=self._gate_state_consistency_skip_latent,
            keep_best_include_cem=self._keep_best_include_cem,
        )

    @staticmethod
    def _format_gate_table(verdict: dict) -> str:
        """One-line-per-trajectory readable summary of a gate verdict."""
        def _f(x):
            return "CRASH" if x is None else f"{x:.3f}"

        def _sf(x):
            return "n/a" if x is None else f"{float(x):.3f}"

        def _pf(x):
            return "n/a" if x is None else f"{100.0 * float(x):.1f}%"

        per = verdict["per_traj"]
        cells = "  ".join(f"t{t}={_f(per[t])}" for t in sorted(per))
        status = "PASS" if verdict["passed"] else "FAIL"
        crash = "ok" if verdict.get("crash_ok", False) else "fail"
        if verdict.get("ratio_ok", False):
            ratio = "ok"
        elif verdict.get("ratio_advisory", False):
            ratio = "adv-fail"
        else:
            ratio = "fail"

        def _check_status(required_key: str, ok_key: str) -> str:
            if not verdict.get(required_key, False):
                return "not_run"
            return "ok" if verdict.get(ok_key, False) else "fail"

        realizability = _check_status(
            "require_action_realizability", "realizability_ok"
        )
        goal_inv = _check_status("require_goal_invariance", "goal_invariance_ok")
        target_indep = _check_status(
            "require_target_independence", "target_independence_ok"
        )
        cem_suite = _check_status("require_cem_suite", "cem_suite_ok")
        runtime_safety = _check_status("require_p2_safety", "p2_safety_ok")
        if verdict.get("require_state_consistency"):
            state_consistency = (
                "ok" if verdict.get("state_consistency_ok", False) else "fail"
            )
        elif verdict.get("state_consistency") is not None:
            state_consistency = (
                "adv-ok"
                if not (verdict.get("state_consistency") or {}).get(
                    "drifting_components"
                )
                else "adv-fail"
            )
        else:
            state_consistency = "not_run"
        fit_quality = _check_status("require_fit_quality", "fit_quality_ok")
        if verdict.get("deployment_goal") is None:
            deploy_goal = "n/a"
        else:
            deploy_goal = "ok" if verdict.get("deployment_goal_ok", True) else "fail"
        stats = verdict.get("ratio_stats", {})
        criteria = verdict.get("candidate_criteria", {})
        criteria_line = " ".join(
            f"{name}={'PASS' if item.get('passed') else 'FAIL'}"
            for name, item in sorted(criteria.items())
        )
        sc = verdict.get("state_consistency") or {}
        sc_label = "required" if verdict.get("require_state_consistency") else "advisory"
        sc_line = (
            f"\n  state-consistency ({sc_label}): {sc['oneline']}"
            if sc.get("ok") and sc.get("oneline")
            else ""
        )
        fq = verdict.get("fit_quality") or {}
        fq_line = (
            "\n  fit-quality "
            f"({'required' if verdict.get('require_fit_quality') else 'advisory'}): "
            f"worst_rmse={_sf(fq.get('worst_normalized_rmse'))} "
            f"mean_rmse={_sf(fq.get('mean_normalized_rmse'))} "
            f"threshold={_sf(fq.get('max_rmse'))}"
            if fq
            else ""
        )
        return (
            f"final fidelity gate: {status} | crash={crash} ratio={ratio} "
            f"realizability={realizability} goal_invariance={goal_inv} "
            f"target_indep={target_indep} "
            f"cem_suite={cem_suite} runtime_safety={runtime_safety} "
            f"state_consistency={state_consistency} fit_quality={fit_quality} "
            f"deploy_goal={deploy_goal} "
            f"| n_failed={verdict['n_failed']} | "
            f"worst_ratio={_sf(verdict['worst'])} mean_ratio={_sf(verdict['mean'])}\n"
            f"  {cells}\n"
            "  stats: "
            f"mean={_sf(stats.get('mean'))} median={_sf(stats.get('median'))} "
            f"p75={_sf(stats.get('p75'))} worst={_sf(stats.get('worst'))} "
            f"best={_sf(stats.get('best'))} "
            f"n={stats.get('n_trajectories', 'n/a')} "
            f"frac<1.0={_pf(stats.get('frac_below_1.0'))} "
            f"frac<0.5={_pf(stats.get('frac_below_0.5'))}\n"
            f"  candidate criteria: {criteria_line}"
            f"{sc_line}"
            f"{fq_line}"
        )

    def _make_finish_guard(self, sandbox: CriticSandbox):
        """Finish-guard for the full final gate.

        While retries remain, inject the gate complaint when the shipped code
        fails any criterion. The VLM loop invokes this both on voluntary finish
        and at the soft turn cap, so max-turn runs still get repair feedback.
        Disabled when no gate trajectories were configured.
        """
        if not self._gate_trajectories:
            return None
        assert isinstance(sandbox, PlanningCriticSandbox)
        trajs = self._gate_trajectories
        # last_tool_idx: sandbox tool-call count when we last complained.
        # If unchanged at the next guard fire, the model answered a gate
        # complaint with prose only (observed 2026-07-07: three retries burned
        # on identical status summaries) — demand a tool call WITHOUT
        # consuming a repair retry, up to a bounded number of nudges.
        state = {
            "retries": 0,
            "last_tool_idx": None,
            "nudges": 0,
            "workflow_prompts": 0,
        }
        max_nudges = 3
        max_workflow_prompts = 2

        def workflow_gaps() -> list[str]:
            # Count successful calls by tool name. Result summaries intentionally
            # start with "[tool_name] ...", so filtering on a leading "[" wrongly
            # treated almost every real tool result as a failure and burned
            # repair turns on already-satisfied evidence checks.
            successful = []
            for call in sandbox._tool_call_log:
                summary = str(call.get("result_summary", "")).lstrip()
                tool = str(call.get("tool", "")).strip()
                if not tool:
                    continue
                if summary.startswith("[error]") or "Traceback" in summary[:200]:
                    continue
                if call.get("error") or call.get("raised"):
                    continue
                successful.append(call)
            names = [str(call.get("tool", "")) for call in successful]
            gaps: list[str] = []
            if (
                self._gate_require_fit_quality
                or self._gate_require_state_consistency
            ) and "analyze_action_effects" not in names:
                gaps.append(
                    "call `analyze_action_effects` and verify the deployed action "
                    "sign/lag before finalising dynamics"
                )
            if self._gate_require_fit_quality:
                fit_pairs = {
                    (
                        str(call.get("args", {}).get("trajectory_index", "")),
                        str(call.get("args", {}).get("frame_index", "")),
                    )
                    for call in successful
                    if call.get("tool") == "view_fit_comparison"
                }
                if len(fit_pairs) < 2:
                    gaps.append(
                        "call `view_fit_comparison` on at least two distinct "
                        "training frames and repair structural mismatches"
                    )
            if (
                self._gate_require_state_consistency
                and "validate_state_consistency" not in names
            ):
                gaps.append(
                    "call `validate_state_consistency` on a difficult trajectory "
                    "and repair every drifting task-state component"
                )
            return gaps

        def guard(turns_used: int) -> str | None:
            if not sandbox.has_valid_simulator_class():
                return (
                    "You have not yet written a valid "
                    f"`{self._simulator_class_name}` class. Continue the task by "
                    "calling `write_code_from_scratch` or `edit_code`; do not finish "
                    "with status text only."
                )
            cur_tool_idx = sandbox._tool_call_idx
            if (
                state["last_tool_idx"] is not None
                and cur_tool_idx == state["last_tool_idx"]
                and state["nudges"] < max_nudges
            ):
                state["nudges"] += 1
                print(
                    f"[gate] finish-guard nudge {state['nudges']}/{max_nudges} "
                    f"at turn {turns_used}: no tool call since last complaint."
                )
                return (
                    "[gate] Your previous message contained no tool call. A "
                    "status summary does not fix a failing gate, and known "
                    "limitations must be REPAIRED, not documented. Your NEXT "
                    "message MUST be a tool call — `edit_code` / "
                    "`write_code_from_scratch` to fix the failures listed "
                    "above, or a `validate_*` tool to localise them. Do not "
                    "finish and do not repeat the summary."
                )
            if self._gate_workflow_evidence != "off":
                gaps = workflow_gaps()
                if gaps and state["workflow_prompts"] < max_workflow_prompts:
                    state["workflow_prompts"] += 1
                    state["last_tool_idx"] = sandbox._tool_call_idx
                    if self._gate_workflow_evidence == "advisory":
                        print(
                            "[gate] advisory evidence gaps (not blocking finish): "
                            + "; ".join(gaps)
                        )
                    else:
                        return (
                            "[MANDATORY EVIDENCE CHECK — do not finish from endpoint "
                            "metrics alone.] The current tool trace is missing:\n- "
                            + "\n- ".join(gaps)
                            + "\nUse these tools now. They expose perception and "
                            "dynamics failures that a low scalar replay loss can hide."
                        )
            if state["retries"] >= self._gate_max_retries:
                return None  # budget spent — let it finish; flagged in the record
            try:
                verdict = sandbox.final_fidelity_gate(trajs)
            except Exception as exc:
                # Do not silently end at the soft turn cap.  Candidate crashes
                # can surface here as gate errors; grant bounded repair turns
                # with the concrete exception instead of shipping debug code.
                print(
                    f"[gate] WARNING: finish-guard gate errored (non-fatal): "
                    f"{type(exc).__name__}: {exc}"
                )
                state["retries"] += 1
                state["last_tool_idx"] = sandbox._tool_call_idx
                return (
                    "[MANDATORY FINAL VALIDATION GATE — the current simulator "
                    "could not be evaluated.] "
                    f"The gate raised {type(exc).__name__}: {exc}. "
                    "Treat this as a simulator failure: remove temporary debug "
                    "exceptions/filesystem probes, restore a complete fit/update/"
                    "terminal_cost/render_frame implementation, then call a "
                    "validation tool. Do not finish with the broken candidate."
                )
            if verdict["passed"]:
                return None
            state["retries"] += 1
            # Record AFTER the gate ran: the gate's own validations log tool
            # calls, and only calls made by the MODEL after this complaint
            # should count as engagement.
            state["last_tool_idx"] = sandbox._tool_call_idx
            print(
                f"[gate] finish-guard {state['retries']}/{self._gate_max_retries} "
                f"at turn {turns_used}: {self._format_gate_table(verdict)}"
            )
            return verdict["complaint"]

        return guard

    def _finish_guard_turn_budget(self) -> tuple[int, int]:
        if not self._gate_trajectories:
            return (0, 0)
        return (
            self._gate_retry_turns,
            self._gate_retry_turns * max(0, int(self._gate_max_retries)),
        )

    def _final_gate_record(self, sandbox: CriticSandbox):
        """Record the gate verdict on the shipped code.

        Keep-best policy: a gate-PASSING final version always ships as-is;
        when the final version FAILS, restore the best-scoring checkpoint
        (lowest full-set worst ratio) and re-gate it, so a last-turn
        regression never ships over a better earlier version.
        """
        assert isinstance(sandbox, PlanningCriticSandbox)
        if not self._gate_trajectories:
            # No gate configured: still ship the best-scoring version.
            if sandbox.restore_best_checkpoint():
                record = sandbox.get_keep_best_record()
                print(
                    "[keep-best] restored tool-call "
                    f"{record['shipped_checkpoint_tool_call']} version "
                    f"(worst {record['shipped_worst_ratio']}) over the final "
                    f"sandbox (worst {record['final_sandbox_worst_ratio']})."
                )
            return (None, "")
        if not sandbox.has_valid_simulator_class():
            summary = (
                f"final fidelity gate: FAIL | no valid `{self._simulator_class_name}` "
                "class was written"
            )
            print(f"[gate] WARNING: shipped simulator FAILED the final gate. {summary}")
            return (False, summary)
        try:
            verdict = sandbox.final_fidelity_gate(self._gate_trajectories)
        except Exception as exc:
            # Record the error but never lose the run to gate infrastructure.
            summary = (
                "final fidelity gate: ERROR (non-fatal) | "
                f"{type(exc).__name__}: {exc}"
            )
            print(f"[gate] WARNING: final gate errored; shipping code anyway. {summary}")
            return (False, summary)
        keep_best_note = ""
        if not verdict["passed"]:
            restored = False
            try:
                restored = sandbox.restore_best_checkpoint()
            except Exception as exc:
                print(
                    "[keep-best] WARNING: restore errored (non-fatal): "
                    f"{type(exc).__name__}: {exc}"
                )
            if restored:
                record = sandbox.get_keep_best_record()
                keep_best_note = (
                    "\n  keep-best: final version failed the gate; restored "
                    f"tool-call {record['shipped_checkpoint_tool_call']} "
                    f"version (worst {record['shipped_worst_ratio']}) over "
                    "the final sandbox (worst "
                    f"{record['final_sandbox_worst_ratio']}) and re-gated."
                )
                print(f"[keep-best]{keep_best_note}")
                try:
                    verdict = sandbox.final_fidelity_gate(
                        self._gate_trajectories
                    )
                except Exception as exc:
                    summary = (
                        "final fidelity gate: ERROR after keep-best restore "
                        f"(non-fatal) | {type(exc).__name__}: {exc}"
                    )
                    print(f"[gate] WARNING: {summary}")
                    return (False, summary + keep_best_note)
        summary = self._format_gate_table(verdict) + keep_best_note
        if not verdict["passed"]:
            print(
                "[gate] WARNING: shipped simulator FAILED the final gate. "
                f"{summary}"
            )
        return (verdict["passed"], summary)

    def _build_tool_callables(self, sandbox: CriticSandbox) -> list:
        base = super()._build_tool_callables(sandbox)
        hidden_tool_names = {
            "view_motion_history_image",
            "view_debug",
            "read_terminal",
            "read_tool_call_metadata",
            "read_world_api_call_metadata",
        }
        base = [tool for tool in base if tool.__name__ not in hidden_tool_names]

        # Append action-conditioned dataset, perception, and validation tools.
        assert isinstance(sandbox, PlanningCriticSandbox)
        planning_tools = [
            sandbox.read_trajectory,
            sandbox.analyze_action_effects,
            sandbox.view_image,
            sandbox.view_fit_comparison,
            sandbox.view_action,
            sandbox.segment_image_with_gemini,
            sandbox.segment_trajectory_with_gemini,
            sandbox.validate_against_training,
            sandbox.validate_all_trajectories,
            sandbox.validate_state_consistency,
            sandbox.validate_action_realizability,
            sandbox.validate_goal_invariance,
            sandbox.validate_target_independence,
            sandbox.calibrate_transition_parameters,
            sandbox.calibrate_parameters,
            sandbox.validate_with_cem,
        ]
        if self._enable_local_toolbox_tools:
            planning_tools[7:7] = [
                sandbox.derive_visual_model,
                sandbox.get_runtime_toolbox_documentation,
                sandbox.calibrate_rigid_geometry,
                sandbox.calibrate_articulated_geometry,
                sandbox.validate_visual_reconstruction,
            ]
            planning_tools.insert(13, sandbox.estimate_dynamics_models)
        base.extend(planning_tools)
        return base
