## Tools

### Inspect the dataset before coding

- **`read_trajectory(i, include_images=False, max_frames=8)`** — Primary trajectory-inspection tool. Returns frame/action metadata by default, including a declared action contract when the dataset provides one; that declared semantics/repeat/bounds record is authoritative. Set `include_images=True` when you need a contact sheet of sampled frames with action labels.
- **`analyze_action_effects(trajectory_indices=None, max_trajectories=3)`** — Coarse measurement before dynamics code: reports image-space component displacement correlations and co-motion hints. Its colour tracks may merge objects or include background, and pixel displacement gain is not a joint/pose dynamics constant. Use reliable signs as hypotheses, reject implausible masks/effects, and confirm all magnitudes with `validate_state_consistency` before adopting them.
- **`view_image(i, k)`** / **`view_action(i, k)`** — Close-up tools for specific frames/actions after `read_trajectory` reveals where to inspect.
- **`view_fit_comparison(i, k=0, ...)`** — LOOK at your own perception: runs your `fit()` on a real frame and returns [TRUE frame | your render | 50/50 blend] side by side. The middle panel must structurally match the left — same object count, poses, proportions (link angles, block shape/orientation, marker positions). Call on 2-3 diverse frames right after implementing `fit()`, and when `validate_state_consistency` reports DRIFT (to tell bad parsing from bad dynamics). If the render does not match reality, fix `fit()`/`render_frame()` BEFORE tuning dynamics — no dynamics work can compensate for a wrong parse.
- **`segment_image_with_gemini(i, k, query)`** — Optional perception-inspection tool for one frame. Use a natural-language referring expression when object identity, pose, or contact is visually ambiguous.
- **`segment_trajectory_with_gemini(i, query, max_frames=6)`** — Optional perception-inspection tool over sampled frames from one trajectory. Use it to get rough object tracks/centroids from the same query across time.
- **`derive_visual_model(i, k, query, component_index=0, simplify_px=1.0)`** — Optional fallback when deterministic local colour/appearance inspection cannot resolve an object's identity. It uses a Gemini mask on one TRAINING frame to emit a Lab appearance model and centroid-centred polygon template. A timeout, box-only response, or one-frame fit is not calibrated geometry; validate any emitted model across other frames and fall back promptly when no pixel mask is returned.
- **`calibrate_rigid_geometry(rgb_min, rgb_max, samples=None, max_samples=9, ...)`** — Deterministic multi-frame component measurement from inclusive RGB bounds. Reports robust dimensions, area-equivalent radius, and a canonical rigid template; use the median radius for circular pushers and validate templates on held-out poses.
- **`calibrate_articulated_geometry(rgb_min, rgb_max, base_xy, link_lengths, samples=None, max_samples=9)`** — Deterministic multi-frame fixed-chain calibration. After inspecting representative link colours, provide inclusive RGB bounds and approximate geometry; the tool samples diverse training frames, estimates one shared base/link geometry, and reports alternate pose branches and ambiguities. Bake the returned constants into the simulator rather than estimating dimensions independently in each `fit()`.
- **`get_runtime_toolbox_documentation(category="all")`** — Exact P1/P2-safe inherited perception/dynamics signatures and the installed local library list. Call this before writing custom CV, contact, kinematics, or integration code.

Gemini segmentation is codegen-only: use it to identify objects and derive deterministic local appearance/geometry constants, not as the P2 parser. WorldAPI may be available as `self.api` inside `fit(image_A, image_B)` at P1/P2 only for non-segmentation geometry helpers. The inherited local toolbox and installed libraries are available with `api=None`; rollout methods (`update`, `terminal_cost`, `planning_objective`, `render_frame`) must not call WorldAPI, Gemini, or any network service.

### Write and inspect code

- **`write_code_from_scratch`** / **`edit_code`** — Write or patch simulator source.
- **`check_compilation`** — After every edit.
- **`run_simulation`** — Smoke: `fit` + zero-action rollout.
- **`read_code`** — Inspect the current simulator source.
- **`view_rendered_frame(k)`** — Sim frame from last run.

### Validation (read-only; never edits code)

- **`validate_against_training(i)`** — Expert-action replay on one trajectory. Reports `reduction_ratio = final_loss / initial_loss`; lower means expert replay moved the simulator closer to the trajectory's target under your `terminal_cost`. It also reports checkpoint state-tracking errors against `fit(true_frame, goal_frame)` to reveal path drift.
- **`validate_all_trajectories()`** — All train trajs: mean + **worst** ratio.
- **`estimate_dynamics_models(trajectory_indices=None, max_trajectories=3)`** — After `fit()` is visually credible and `self.state` contains only compact numeric coordinates (not masks/images), reparses sampled consecutive true frames and ranks task-agnostic first-order, inertial, and setpoint models per safe state key. It reports skipped components and bounded held-out errors/coefficients.
- **`calibrate_transition_parameters(bounds, ...)`** — Preferred parameter calibration: teacher-forces sampled consecutive TRUE frames with one recorded action, fits selected `self.params` to normalized observable next-state error, and reports train/held-out totals plus per-key errors (including wrapped angles). Use narrow physical bounds. It never edits source; adopt values manually only when held-out errors improve.
- **`self.damped_dynamics_step(...)`** — Supports optional `position_low`, `position_high`, and per-dimension `periodic`. Use periodic wrapping only for declared continuous coordinates. For bounded joints, provide finite bounds and keep `periodic=False`; the helper clamps the position and cancels outward velocity at a limit.
- **`validate_state_consistency(i)`** — Localise dynamics error: replays ground-truth actions and compares the rollout state against your own `fit()` parse of the TRUE frame at each checkpoint, per component. Use on the worst trajectory to learn WHICH component diverges and WHEN (steady growth = wrong scale/sign; sudden jump = mis-modelled contact/event; all OK = fix `terminal_cost`/goal parsing instead of `update`). Velocity/latent keys that still-frame `fit()` cannot re-observe are reported as skipped/unobserved, not as dynamics failures. If a component is constant in the true parses but drifts in rollout, move it out of `self.state` into `target_state` / attrs.
- **`validate_goal_invariance(...)`** — `target_state` stable when only `image_A` changes.
- **`validate_target_independence(...)`** — identical actions under two different goals must give identical state trajectories. When the action contract has a fixed neutral action, neutral rollouts must not reduce `terminal_cost`; absolute setpoint spaces do not treat `[0, 0]` as a no-op. Gate-enforced; run before finishing.
- **`validate_action_realizability(...)`** — No per-step jumps above training max.
- **`validate_with_cem(...)`** — CEM on one start/target frame pair. PASS needs **goal-reach in state space**, not just a falling cost: the components in `target_state` must approach their targets (reported in units of max one-step motion). **STUCK/goal-reach FAIL** → fix dynamics or the terminal planning objective in `terminal_cost()`. Before finishing, run this on at least **5 diverse hard pairs**; do not trust a single pair, because it can be lucky or overfit.
- **`calibrate_parameters(bounds, ...)`** — Legacy terminal-ratio search. Prefer `calibrate_transition_parameters`; endpoint ratios can be fooled by periodic poses or an incomplete terminal cost. This tool also never edits source.

The harness may repeat some validation automatically at finish time, including fixed CEM stress pairs and a runtime-dependency safety smoke that allows WorldAPI in `fit()` but disables it during CEM rollout. Passing a hand-picked tool call is not enough; make the simulator generally planner-drivable and safe without perception/API calls during rollout.

### Gemini segmentation and WorldAPI

- Segmentation is provided only by the codegen tools `segment_image_with_gemini` / `segment_trajectory_with_gemini`; WorldAPI segmentation is not part of your API surface.
- **`get_api_documentation(names)`** — Call this before using non-segmentation `self.api.*` methods so you know the exact method signatures and return formats.

### `self.api` methods at runtime

Non-segmentation geometry helpers below may help `fit` instantiate state from images. Implement step limits, contact, and kinematics in your simulator state/update logic, not via API calls during rollout. Do not call `self.api.segment`; segmentation is provided only by the Gemini codegen tools.

[AVAILABLE_TOOLS]
