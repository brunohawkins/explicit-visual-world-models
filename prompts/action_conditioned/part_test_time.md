## Test-time protocol

After P1 codegen, your simulator source is frozen. The P2 evaluation harness uses it for model-predictive control: observe the real environment, fit simulator state, plan in the simulator, execute a short action chunk in the real environment, then observe and replan. The real environment may include novel configurations. Look at different trajectories to understand what these configurations may resemble.

### How CEM uses your simulator

The planner is **derivative-free**: it samples action sequences, clips them to harness bounds, and rolls each forward with `update(a)` for a fixed **horizon**. Before every candidate rollout it **deep-copies `self.state`** and restores it afterward. Each candidate is scored by **`terminal_cost()` at the end of the horizon only** — not a sum over intermediate steps, and not via gradients through your code. Design it over explicit simulator state, so low-cost terminal states are **physically reachable** under your `update`; use `validate_with_cem` to catch undrivable objectives. A lower cost must come from physical state progress, not from elapsed time, rollout length, or a step counter.

The terminal objective is allowed to reason about feasible paths and search guidance. It can include waypoints, bottlenecks, contact constraints, geodesic-like structure, orientation error, joint consistency, approach poses, or other task-relevant shaping. It must remain a scalar objective evaluated on the current simulated state and target state; it should not mutate state, choose actions, call Gemini segmentation or WorldAPI helpers, or run its own rollout.

### Phase A — Planning (simulator only)

1. `sim = [SIMULATOR_CLASS_NAME](frame_size=...)` matching training frame size.
2. `sim.fit(image_A, image_B)` — `image_A` = current observation; `image_B` = planner target image. Parse both into the same dynamics state space; visual markers/cues matter only if they define the desired state of the modeled objects.
3. **CEM** searches for an action sequence with low terminal `terminal_cost()` / planning-objective value entirely inside your simulator.

At P2, the harness may provide `self.api` (non-segmentation WorldAPI geometry helpers) so `fit(image_A, image_B)` can parse the current and target images; segmentation knowledge must already be baked into `fit`'s local parsing during codegen. Local deterministic runtime dependencies are also allowed if available (for example NumPy, PIL, scipy, PyMunk, or MuJoCo). Do not make CEM rollout depend on Gemini segmentation or WorldAPI calls: after `fit`, `update`, `terminal_cost`, `planning_objective`, and `render_frame` must use ordinary simulator state/target values only.

### Phase B — Execution (receding-horizon MPC)

4. Apply the first **`apply_steps` actions specified by the evaluation profile** in the real environment, then re-observe and replan. The profile may use a long action chunk, so one `update(a)` must represent one deployed environment action and repeated updates must remain stable.
5. Read the new real observation, call `sim.fit(current_obs, goal)` again, and re-plan with the profile's configured horizon. Do not assume fixed horizon or `apply_steps` values during code generation.
6. Real success is judged by the **environment's own metrics** (success flag, reward, task scores) — not sim loss or pixel match.

**Open-loop ablation** (optional): fit once, plan once over a long horizon, execute without re-fit — not the default P2 protocol.

### Start/target at deploy time

Test start and target may lie **outside** the states visited in training demos, but still within the scene's physical limits. Your `fit`, `update`, `terminal_cost`, and `render_frame` must generalise to such pairs — not only to trajectory start/end frames you inspected during codegen.

`image_B` may contain markers/cues in addition to movable objects. Decide which visual elements define the desired target in the same state representation used by `update` and `terminal_cost`; do not blindly copy the current movable-object pose from `image_B` if a separate target cue is present.

The harness may run a runtime-dependency safety smoke with WorldAPI available during `fit()` and then disabled during a short CEM plan. Treat crashes there as contract failures, not deployment quirks.
