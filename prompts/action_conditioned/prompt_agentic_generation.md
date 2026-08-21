[PART_DEFINITIONS]

[PART_DELIVERABLE]

[PART_ENVIRONMENT]

[PART_CODEGEN_INPUTS]

[PART_TEST_TIME]

[PART_CODE_RULES]

[PART_TOOLS]

[TURN_BUDGET_SECTION]

## Workflow

1. **Inspect + choose mechanisms** — read multiple trajectories, obey any declared action contract, inspect consecutive frames/actions, and call `analyze_action_effects`. Call `get_runtime_toolbox_documentation()` before implementing custom CV or physics. Decide which observed components are rigid, articulated, point-like, static, or contact-coupled; keep non-evolving quantities out of `self.state`.
2. **Lock perception first** — implement only `fit()` plus a diagnostic `render_frame()` initially. When identity/masks are non-trivial, call `derive_visual_model` rather than inventing fixed RGB thresholds or hand-entering a contour. Call `view_fit_comparison` on diverse poses and repair masks/geometry until current and target states are visually credible. Do not tune dynamics against a moving perception target.
3. **Identify then implement dynamics** — call `estimate_dynamics_models` once `fit()` is credible; combine its consecutive-frame regression with the declared action contract. Prefer the simplest model supported by both: inherited first-order/damped helpers for smooth systems, established kinematics for linked systems, and inherited Pymunk polygon contact when object motion is sparse/contact-coupled. Force/torque-like actions require calibrated latent velocity/damping; absolute setpoints require error-to-setpoint dynamics. Use `write_code_from_scratch` / `edit_code` → `check_compilation` → `run_simulation` after changes.
4. **Fidelity and localisation** — run `validate_against_training` on several trajectories, then `validate_all_trajectories`. Use `validate_state_consistency` to localise dynamics drift; if observable pose already disagrees at checkpoint 0, return to perception instead of fitting dynamics around it.
5. **Planner checks** — only after perception and expert replay are credible, run `validate_goal_invariance`, `validate_action_realizability`, `validate_target_independence`, and `validate_with_cem` on diverse pairs. A useful CEM pass must show state-space progress, not just a falling scalar cost.
6. **Finish** — repair concrete final-check failures with tool calls. A status summary or "known limitations" note does not repair the simulator.

**Budget checkpoint:** when 8 turns remain, stop exploratory/debug-only edits and make the sandbox a complete runnable candidate. Remove temporary exceptions and filesystem probes, implement every required method, then spend the remaining turns on validation and targeted repairs. The sandbox has no supported trajectory-dataset filesystem path; use `read_trajectory`, `view_action`, `analyze_action_effects`, and `calibrate_parameters` for dataset measurements.

The harness may also run automatic final checks, including fixed CEM stress pairs and a runtime-dependency safety smoke that permits WorldAPI (non-segmentation geometry helpers) during `fit(image_A, image_B)` but disables it during rollout. These checks test whether the simulator is planner-drivable and deployable under the P2 contract.

## Final response

Text only — no tools, no code:

1. Worst and mean `reduction_ratio`; goal invariance, realizability, CEM results (PASS/FAIL/STUCK), and whether rollout methods are runtime-safe without WorldAPI/perception calls.
2. Last terminal planning objective value from `run_simulation` (`terminal_cost()` in code); describe render sanity in words (no image data).
3. One sentence: what `self.state` holds and what one `update` does.
4. One sentence: what terminal state-space planning objective `terminal_cost()` computes and why CEM can drive it.
5. Known limitations.
