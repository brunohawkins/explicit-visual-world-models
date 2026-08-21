## Code rules

Hard constraints on generated code (beyond the method contracts above):

- **No filesystem or dataset I/O** in the shipped simulator (`open`, `pathlib`, `os`, `glob`, `subprocess`, `Image.open`, `np.load`, hard-coded dataset paths, trajectory/frame filenames, …). Inspect data with tools during codegen, then bake learned constants/models into ordinary Python data structures before finishing.
- **No real-environment imports or calls** — your class is a planner-side model. It must not import gym, construct benchmark envs, or query environment state APIs.
- **P2-safe local rollout** — WorldAPI (non-segmentation geometry helpers) may be available in P2 inside `fit(image_A, image_B)` to parse the current and target images more accurately. CEM rollouts must remain local and deterministic: do not call WorldAPI or any perception service from `update`, `terminal_cost`, `planning_objective`, or `render_frame`. If you use geometry helpers in `fit`, store only ordinary state/target values for rollout.
- **No degenerate stubs** — `update(a)` must change `self.state` nontrivially with `a`; `terminal_cost()` must vary with `self.state` and `self.target_state`; `render_frame()` must depict that state (not a flat colour).
- **Objective has no side effects** — `terminal_cost()` may include structured guidance or policy-shaped heuristics, but it must only return a scalar. It must not mutate state, choose/apply actions, run CEM, roll out trajectories, or call any perception/geometry API.
- **No time/step shortcuts in the objective** — `terminal_cost()` must not decrease just because time passes, a step counter increases, or a rollout gets longer. It may depend on time only if time is part of the physical state being modeled; otherwise it must improve only when the explicit simulated state improves relative to `target_state`.
- **No goal-driven dynamics** — `update(a)` must never read `self.target_state` or any goal-derived value. The gate replays identical actions under two different goals and rejects the code if the state trajectories differ. It checks neutral-action cost drift only when the deployed action contract has a fixed neutral action (`validate_target_independence`).
- **No planner-exploitable motion** — per-step state jumps must stay within limits implied by training data (`validate_action_realizability`).

Define only `[SIMULATOR_CLASS_NAME]` plus private helpers it needs.
