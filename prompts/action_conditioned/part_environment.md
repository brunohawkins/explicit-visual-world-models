## Environment & interface

Your class inherits from `ActionConditionedSimulatorBase`. For this run, the runner constructs it with the benchmark-specific values `frame_size=[FRAME_SIZE]` and `fps=[FPS]`, and optionally `api=WorldAPI(...)`. These values differ per benchmark, so read `self.frame_size` rather than hard-coding dimensions when rendering or parsing image coordinates. Segmentation comes from the Gemini segmentation tools during codegen; do not use `self.api.segment`.

Reference interface:

[SPEC_SIMULATOR]

`ActionConditionedSimulatorBase`, `SimulatorBase`, and `WorldAPI` are already imported by the runner — **do not import them yourself**. WorldAPI is not the segmentation route in this prompt.

### Libraries

Available in both P1 and P2: `numpy`, `scipy`, `opencv-python` (`cv2`), `scikit-image`, `pillow`, `shapely`, `pygame`, `pymunk`, `trimesh`, `mujoco`, and the Python standard library. Prefer these established local libraries and the inherited runtime toolbox over reimplementing segmentation morphology, contour geometry, collision detection, rigid-body contact, or numerical integration. Call `get_runtime_toolbox_documentation()` for exact inherited helper signatures.

For a visually identified component, prefer `derive_visual_model(...)`: it converts a Gemini mask on a training frame into a deterministic Lab appearance model and polygon template that generated code can embed and use with inherited `self.mask_from_appearance(...)` and `self.fit_pose_to_mask(...)` at P2. This is safer than inventing fixed RGB thresholds and hand-entering a silhouette. Validate the emitted model on diverse frames; one reference mask is not proof of generalisation. For a known asymmetric rigid silhouette, do not use PCA as the final orientation estimate because its axis sign is ambiguous. For articulated objects, pass the appearance mask, inferred base, fixed link lengths, and optional previous angles to inherited `self.fit_articulated_chain(...)` instead of guessing elbow/tip pixels or forcing the whole object into one rigid template.

The inherited toolbox is available even when `self.api is None`: mask cleanup/appearance segmentation/component geometry/template fitting; first- and second-order dynamics steps; and an arbitrary-polygon Pymunk `planar_push_step`. These are generic mechanisms, not task assumptions. You must still infer object identity, state variables, action semantics, coordinate conventions, model family, and parameters from the trajectories.

If WorldAPI is enabled and `get_api_documentation` is actually present in your tool list, call it before using `self.api.<method>` so you do not invent names or signatures. When the run uses `api=None`, that documentation tool is intentionally absent; use `get_runtime_toolbox_documentation` and inherited local helpers instead. Do not use `self.api.segment`; use Gemini only during codegen to derive local deployable perception models. After using any non-trivial library/API call, run `check_compilation()` and then a simulation or validation tool.

Pick the minimal abstraction that matches observed dynamics. For `pygame` / `pymunk` 2D, render off-screen (`pygame.Surface`) — the runner is headless.

### Coordinates

Image arrays use `(row, col)` indexing, while positions are usually easier to store as `(x, y)` = `(col, row)`, with image `y` increasing downward. If you use a physics library with a different convention, convert consistently between image space (`fit`, `render_frame`) and physics space (`update`). Use 3D coordinates if necessary for the task.

**Actions and positions must share the same coordinate frame as the training frames** — infer scale and semantics from `view_image` + `view_action`, not assumed units.

### Planner snapshot (see State contract)

CEM snapshots only `self.state` via `copy.deepcopy` between rollouts. Scene geometry, `frame_size`, and `self.params` may live as plain instance attrs in `__init__` or `fit`.

### Rendering

Use `view_image` to learn scene layout before writing `render_frame`. The render should make dynamics legible: a human watching rollout frames should see whether state changes sensibly under actions. Match colours and fine visual detail only insofar as they help debugging.
