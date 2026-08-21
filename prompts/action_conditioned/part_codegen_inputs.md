## Inputs during code generation

During P1, treat the offline demonstrations as system-identification data. Infer what state variables exist, how actions change them, what visual cues define targets, and what model CEM can roll forward.

You receive **only**:

- **Training trajectories** — PNG frames + `actions.json` per `trajectory_XXXX/`.
- **Dataset tools** — `read_trajectory`, `view_image`, `view_action`.
- **Gemini segmentation tools** — `segment_image_with_gemini`, `segment_trajectory_with_gemini`; use these when object identity, masks, pose, or contact is visually ambiguous.
- **WorldAPI geometry helpers** may be available as `self.api`; use `get_api_documentation` before calling non-segmentation methods.

You do **not** receive:

- A natural-language task description or caption.
- Test-time start/target images.
- The real deployment environment (no env session APIs, no `gym.make`, in generated code).

### Inspect before coding

Treat each trajectory as a **time series**, not just a start/end pair. Infer both the dynamics and the state/target semantics needed for planning.

If motion lags an action or the same action produces different displacement depending on recent motion, include velocity or other latent physical state instead of modeling the action as an immediate pose increment. Latent rates (e.g. `dtheta`, `vel_*`) are integrated by `update(a)` and are not re-observed by still-frame `fit()` — do not expect `validate_state_consistency` to re-parse them from images.

Obey any declared action contract in trajectory metadata: joint/delta-like actions usually fit a calibrated first-order map from action to state change; force/torque-like actions need latent velocity plus damping and must be calibrated against measured Δstate / action before inventing second-order constants.

1. Start with `read_trajectory(i)` on multiple trajectories. It returns frame/action metadata and action summaries. If visual temporal structure is unclear, call `read_trajectory(i, include_images=True)` to request a contact sheet of sampled frames with action labels.
2. **Consecutive frames + actions** — after `read_trajectory`, use `view_image(t, k)` and `view_action(t, k)` for close-up detail at specific steps `k, k+1, …`. Infer control semantics (velocity, displacement, setpoint, joint torque, …) and the largest per-step state change the data exhibits.
3. **Perception checks** — if objects, contacts, poses, or visual cues are ambiguous, use Gemini segmentation tools to identify masks/segments, then use local geometry or non-segmentation WorldAPI helpers if needed to compute centroids, keypoints, poses, or primitives.
4. **Geometry consistency** — for articulated, linked, or rigid multi-part objects, measure inferred part lengths, relative offsets, or other fixed geometry across multiple frames and trajectories before hard-coding parameters. If a parser gives inconsistent geometry, fix the keypoint/pose extraction before tuning dynamics.
5. **State and objective representation** — decide which **movable state variables** are needed for dynamics and planning: object pose, robot/pusher pose, joints, contact state, obstacles, etc. Across demonstrations, infer the primary outcome from the relationship that consistently holds at successful endpoints, including a moved object or end effector reaching a persistent spatial cue. Variables needed to model how the system moves do not all need to be terminal targets. Constants that do not evolve under `update(a)` (fixed targets, obstacles, persistent markers) belong in `target_state` or ordinary attributes — never in `self.state`.

Treat `image_B` as evidence for the desired observable outcome, not a requirement to reproduce every changed pixel. Give the primary objective weight to the task-relevant component and its demonstrated relation to a target cue or goal pose; keep robot, pusher, or other configuration targets secondary unless their final pose itself defines success. Represent that primary task-space quantity under matching keys in `state` and `target_state` so planning progress is auditable. For extended or asymmetric objects, centroid distance alone is insufficient; also score orientation and spatial extent or overlap.

Use perception tools to instantiate or debug state, not to replace the action-conditioned dynamics. Segmentation is Gemini-only in this prompt; do not use WorldAPI segmentation. CEM rollout methods must not rely on Gemini segmentation or WorldAPI calls.

Check that action coordinates match the image frame (see Coordinates). Do not assume units, contact model, or state dimensionality you have not verified from the data.

### Before implementing `fit`

Use the inspection steps to define `self.state` and `self.target_state` in the same dynamics state space. After coding, run `view_fit_comparison` and `validate_goal_invariance` to check that current-state parsing and target parsing are separated.
