## Optional local 2-D toolbox addendum

Keep the workflow above unchanged. The following helpers are optional
implementation aids, not a required staged workflow and not evidence that a
particular model family is correct.

Available in both P1 and P2: `numpy`, `scipy`, `opencv-python` (`cv2`),
`scikit-image`, `pillow`, `shapely`, `pygame`, `pymunk`, `trimesh`, and
`mujoco`. Call `get_runtime_toolbox_documentation()` only when an exact helper
signature is needed. Do not spend turns calling it repeatedly.

Generated simulators inherit deterministic local helpers even with `api=None`:
mask cleanup and appearance segmentation, connected-component geometry,
`fit_pose_to_mask`, `fit_articulated_chain`, first- and second-order dynamics,
and polygonal planar contact. Prefer these tested primitives over rewriting
contour search, collision handling, or integration.

Never average pixels across disconnected mask components. Use
`self.select_connected_component(...)` with a measured minimum area and, when
available, the previous observed centroid and calibrated expected area. Check
its `valid` and `confidence` metadata before updating state; an invalid result
is missing evidence, not an observation at the image centre. Apply the same
rule to pose fitting: `fit_pose_to_mask` preserves a legacy centre fallback for
empty masks, so do not use `cx`/`cy` unless `valid` is true and confidence is
credible.

For an asymmetric rigid component, PCA is only a coarse orientation proposal:
its axis sign is ambiguous. Clean the component mask, use a body-local polygon
template with a documented origin, and call
`self.fit_pose_to_mask(mask, template, n_angles=180)`. Validate centroid,
orientation direction, scale, and overlap on at least three substantially
different training frames. A low error on one frame is not sufficient.

For an articulated component, do not guess the base, link lengths, angle
convention, or whether the second angle is absolute or relative. Measure fixed
geometry across at least three diverse frames. After inspecting representative
link colours, call `calibrate_articulated_geometry(rgb_min, rgb_max, base_xy,
link_lengths)` and bake its fixed base/link constants into the simulator before
calling `self.fit_articulated_chain`. Its image coordinates have positive y
downward; with `relative_angles=True`, later angles are relative joint angles.
Inspect its alternate `candidates` and `ambiguous` flag, pass previous angles
for temporal continuity, and confirm that forward kinematics reproduces the
observed elbow and endpoint before fitting action gains. Parse current and goal
images into the same wrapped joint convention; endpoint agreement alone does
not identify the articulated goal branch.

Obey any declared state/task contract. Angle wrapping is a coordinate
convention, not evidence that every joint is periodic: wrap continuous joints,
but clamp bounded joints and use raw bounded-joint errors. If task success is
branch-invariant (for example, an endpoint reaching a marker), equivalent IK
branches must have equivalent objective value; do not force motion through a
joint limit merely to match the branch depicted in the goal image.

At each real observation/refit, estimate velocity from the current observed
pose and the previous observed pose over the actual elapsed seconds. Do not
subtract a propagated prediction from a new observation. Use wrapped deltas
only for dimensions explicitly declared periodic; use raw deltas for bounded
and other nonperiodic dimensions. When a dynamics helper returns position,
linear velocity, angular velocity, mode, or other physical state, preserve
every relevant returned value in `self.state` rather than discarding hidden
momentum between planning steps.

For an arbitrary or concave rigid body, do not pass one concave outline to
`pymunk.Poly`, which requires convex polygons. Use convex parts or inherited
`self.planar_push_step(...)`. Keep polygons, positions, velocities, targets,
and bounds in one caller coordinate system (the helper does no image-axis
conversion), and preserve the same template origin in perception and rendering.
Its exact return keys are `body_position`, `body_angle`, `body_velocity`,
`body_angular_velocity`, `pusher_position`, and `pusher_velocity`.

Optional codegen tools:

- `calibrate_rigid_geometry(rgb_min, rgb_max, ...)` measures a colour-isolated
  component over diverse frames and emits robust radius, dimensions, and a
  canonical template. Use the median radius for circular pushers.
- `validate_visual_reconstruction(...)` checks `fit()` plus `render_frame()`
  over diverse start/middle/late frames and reports foreground/component
  overlap, centroid error, component counts, and RMSE. Treat any structural
  mismatch as a perception problem even when replay ratios are good.
- `derive_visual_model(...)` can turn a successful training-frame Gemini mask
  into deployable Lab appearance and polygon literals. If segmentation returns
  no pixel mask or times out, fall back promptly to measured local CV rather
  than repeatedly calling it.
- `estimate_dynamics_models(...)` ranks simple action-conditioned models in the
  compact numeric state coordinates produced by your current `fit()`. Call it
  only after visual reconstruction is credible and masks/images are not stored
  as state; good regression in distorted coordinates is not faithful dynamics.
- `calibrate_transition_parameters(...)` teacher-forces sampled consecutive
  true frames and fits selected `self.params` to normalized observable one-step
  errors, with separate held-out transitions. Prefer it over terminal-ratio
  tuning; it reports values but never edits source.

Before final dynamics tuning, use `view_fit_comparison` on at least three
diverse frames. The true and rendered object centroids, extents, articulated
endpoints, and asymmetric orientation must agree. Repair perception first if
they do not.

Do not begin perception with Gemini segmentation. First use deterministic local
colour/appearance masks and multi-frame geometry calibration. Gemini is only an
optional fallback for unresolved object identity; a timeout or missing pixel
mask is not evidence about scene geometry.
