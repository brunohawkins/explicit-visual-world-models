## Your deliverable

Write one Python class `[SIMULATOR_CLASS_NAME]` subclassing `ActionConditionedSimulatorBase`. It must provide at least these four methods: `fit`, `update`, `terminal_cost`, and `render_frame`. Private helper methods/functions are allowed.

At P2 test time, the evaluation harness calls your `fit` method on real observations, runs CEM against your simulator, executes the first `apply_steps` actions from the CEM plan in the real environment, then observes and replans.

### `fit(self, image_A, image_B)`

Parse `image_A` into `self.state` (the current explicit simulator state) and `image_B` into `self.target_state` (the desired values in that same state space). Segmentation understanding comes from the Gemini segmentation tools during codegen; bake what you learn into local deterministic parsing here. WorldAPI may be available as `self.api` inside `fit` (at P1 and P2) for non-segmentation geometry helpers such as primitive fitting or pose fitting; do not use WorldAPI segmentation.

`fit` instantiates the state for planning; it should not choose or simulate an action sequence.

- Choose state variables that explain observed motion and contacts: object poses, robot/pusher poses, joint states, modes, obstacles, or other task-relevant quantities.
- `target_state` should be expressed in the same variables that `update(a)` moves and `terminal_cost()` scores.
- Do not roll out actions inside `fit`; `update(a)` is responsible for dynamics.
- Self-check: changing `image_A` while holding `image_B` fixed should normally leave `target_state` unchanged. If `target_state` follows the current movable object pose, target parsing is probably wrong.
- Offline validation may use a trajectory's final frame as `image_B`; that can hide test-time target parsing bugs. Use `validate_goal_invariance` and fit/render inspection to check target parsing directly.
- Optional receding-horizon signature: you may declare `fit(self, image_A, image_B, prev_state=None)`. The P2 harness can pass the previous propagated state as `prev_state`, letting you preserve latent/latched variables that a single image cannot re-establish while re-grounding observable poses from `image_A`. Offline tools may still call the two-argument form, so `prev_state` must default to `None`.
- Never average disconnected segmentation regions. Select one coherent
  component using minimum area and, when available, prior observed centroid and
  expected area; check `valid`/`confidence` before accepting its geometry. A
  legacy image-centre fallback from an empty pose fit is not a valid
  observation.
- If refitting velocity, compare the current observed pose with the previous
  observed pose over the real elapsed time. Wrap only dimensions explicitly
  declared periodic; keep raw deltas for bounded/nonperiodic dimensions.

Pick the minimum sufficient state from the data. Use 2D, joint, mode, contact, or 3D variables as needed by the observed dynamics.

### `update(self, a)`

One control step. Mutate `self.state` as a non-trivial function of action `a`, using action semantics inferred from the trajectories. Respect plausible per-step motion/contact limits from the data; do not solve the task by teleporting state directly to the target.

`update(a)` must not read `self.target_state` or any goal-derived quantity to move state. The same start and same actions should produce the same state trajectory under different targets. The validation tool `validate_target_independence` checks for goal-driven dynamics and, when the action contract defines one, cost drift under a fixed neutral action. Do not treat `[0, 0]` as a no-op in an absolute-position action space.

### `terminal_cost(self) -> float`

Return the scalar terminal planning objective that CEM minimises after rolling out an action sequence. Lower should mean the final simulated state is better relative to `self.target_state`.

The objective may be structured and heuristic. It can include waypoints, approach terms, contact geometry, path feasibility, orientation terms, joint consistency, or other shaping that makes CEM's action search tractable. It should still be a function of the current simulated state, the target state, and fixed scene/model parameters; it should not call Gemini segmentation or WorldAPI helpers, mutate state, execute a policy, or perform its own rollout.

Score only task-relevant terminal variables. A robot, cursor, gripper, or pusher shown in `image_B` may be placed there only to make the goal object visible; do not make its exact image-B position a terminal target unless the demonstrations show that agent pose is itself part of task success.

Low-cost states should be reachable under `update(a)`. Do not add terms that make the cost fall merely because a rollout step counter or elapsed time changed; a lower cost should reflect physical progress, useful positioning, or a better feasible state relative to the target.

Prefer an objective that reflects the structure CEM needs to discover feasible actions, such as routes around obstacles, contact approach positions, bottlenecks, orientation alignment, or staged progress terms. If scene geometry or a required contact blocks the straight line to the target, a pure distance-to-target objective can trap CEM pressed against the obstruction — encode feasible-path or staged structure in those cases. Plain distance is fine when the route is unobstructed.

If the scene appears articulated and actions appear to control joints, model the controllable joint state and its dynamics directly. Use end-effector or object-position distance as a terminal-cost term only if it remains consistent with the modeled joint dynamics; do not let a visually convenient endpoint objective hide joint-state drift under expert-action replay.

### `render_frame(self) -> np.ndarray`

Return a uint8 RGB image of shape `(frame_size[1], frame_size[0], 3)` that depicts `self.state`. The render is for inspection and debugging: object count, layout, poses, and motion should be legible, but exact colours and cosmetic details need not match the dataset.

### State contract

- Per-rollout mutable state lives in `self.state` (a dictionary is recommended). The planner deep-copies and restores it between rollouts.
- `self.target_state` is set in `fit`, read by `terminal_cost`, and must not be mutated or used as a dynamics input by `update`.
- Latched state such as contact/attachment/mode flags should either be inferable from `image_A` or preserved via optional `prev_state`.
- Preserve all relevant physical values returned by dynamics helpers,
  including linear/angular velocities and modes; dropping them silently changes
  the next-step dynamics.
- Keep `terminal_cost()` as the task-goal objective. When intermediate path
  quality is physically meaningful, optionally implement side-effect-free
  `stage_cost(action, step_index)` using generic state constraints supported by
  the data (for example bounds, action realizability, collision, or stability);
  do not hard-code benchmark routes or test-case waypoints.
- Avoid using counters such as `step`, `t`, or elapsed time to make `terminal_cost()` shrink independently of physical progress. The base class already has `self.t`; only use time if it is genuinely part of the modeled physics.
- Tunable constants such as gains, friction, and geometry should live in `self.params` or ordinary instance attributes, not in per-rollout state unless they change dynamically.
